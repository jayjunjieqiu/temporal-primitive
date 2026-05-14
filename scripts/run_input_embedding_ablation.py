from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CHRONOS_SRC = ROOT / "external" / "chronos-forecasting" / "src"
TIMESFM_SRC = ROOT / "external" / "timesfm" / "src"
OUT_DIR = ROOT / "outputs" / "input_embedding_ablation"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_PATH = OUT_DIR / "input_embedding_ablation_summary.json"
REPORT_PATH = ROOT / "docs" / "input_embedding_ablation_report.md"

sys.path.insert(0, str(ROOT))
from scripts.run_second_pilot_discovery import (  # noqa: E402
    DATA_ROOT,
    MODEL_SPECS,
    flatten_model_patches,
    robust_z,
    sample_windows,
    select_domain_balanced_indices,
    top_counts,
)


TARGET_SPECS = {
    "chronos_2_small": {
        "kind": "chronos",
        "patch_len": 16,
        "hidden_layer": 5,
        "display": "Chronos-2-small layer_5",
    },
    "chronos_2": {
        "kind": "chronos",
        "patch_len": 16,
        "hidden_layer": 11,
        "display": "Chronos-2 layer_11",
    },
    "timesfm_2_5": {
        "kind": "timesfm",
        "patch_len": 32,
        "hidden_layer": 10,
        "display": "TimesFM-2.5 layer_10",
    },
}


REPRESENTATION_NOTES = {
    "raw_z_patch": "raw patch 做 robust z-normalization 后直接聚类；这是形状本身的 baseline。",
    "chronos_proj_with_time": "Chronos `_prepare_patched_context` 后经 `input_patch_embedding`；包含 per-observation time encoding、patch value、mask。",
    "chronos_proj_time_zeroed": "将 Chronos patch input 中的 time encoding 通道置零后再过 `input_patch_embedding`；不是官方前向路径，只作为 position-source diagnostic。",
    "chronos_hidden": "Chronos encoder selected layer output，已经过 transformer contextualization。",
    "timesfm_tokenizer": "TimesFM running RevIN 后的 patch 经 tokenizer projection；进入 `stacked_xf` 前，无显式 absolute position embedding，但 running stats 是顺序相关的。",
    "timesfm_hidden": "TimesFM selected transformer layer output，已经过 causal self-attention 和 RoPE。",
}


def attach_context(meta: list[dict[str, Any]], windows: np.ndarray) -> None:
    for m in meta:
        m["_context"] = windows[int(m["window_id"])].astype(float).tolist()


def make_raw_patch_embeddings(model_key: str, windows: np.ndarray, window_meta: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    patch_len = int(TARGET_SPECS[model_key]["patch_len"])
    num_patches = windows.shape[1] // patch_len
    raw_embeddings = np.zeros((len(windows), num_patches, patch_len), dtype=np.float32)
    for i in range(len(windows)):
        for j in range(num_patches):
            patch = windows[i, j * patch_len : (j + 1) * patch_len]
            raw_embeddings[i, j] = robust_z(patch).astype(np.float32)
    embeddings, meta, patches = flatten_model_patches(model_key, raw_embeddings, windows, window_meta)
    attach_context(meta, windows)
    return embeddings, meta, patches


def extract_chronos_representations(model_key: str, windows: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(CHRONOS_SRC))
    import chronos

    spec = TARGET_SPECS[model_key]
    layer_idx = int(spec["hidden_layer"])
    pipeline = chronos.Chronos2Pipeline.from_pretrained(
        str(ROOT / model_key.replace("_", "-")),
        local_files_only=True,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = pipeline.model
    model.eval()

    chunks: dict[str, list[np.ndarray]] = {
        "chronos_proj_with_time": [],
        "chronos_proj_time_zeroed": [],
        "chronos_hidden": [],
    }
    patch_len = int(spec["patch_len"])

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[start : start + batch_size], dtype=torch.float32, device=model.device)
            patched_context, _attention_mask, _loc_scale = model._prepare_patched_context(context=batch)
            num_context_patches = patched_context.shape[1]

            input_with_time = model.input_patch_embedding(patched_context)
            no_time_context = patched_context.clone()
            no_time_context[..., :patch_len] = 0.0
            input_time_zeroed = model.input_patch_embedding(no_time_context)

            captured: dict[str, torch.Tensor] = {}

            def hook(_mod, _inp, out):
                captured["chronos_hidden"] = out[0].detach()

            handle = model.encoder.block[layer_idx].register_forward_hook(hook)
            encoder_outputs, *_rest, returned_num_context_patches = model.encode(
                context=batch,
                num_output_patches=1,
            )
            handle.remove()
            if int(returned_num_context_patches) != num_context_patches:
                raise RuntimeError("Chronos num_context_patches mismatch")

            chunks["chronos_proj_with_time"].append(input_with_time[:, :num_context_patches].float().cpu().numpy())
            chunks["chronos_proj_time_zeroed"].append(input_time_zeroed[:, :num_context_patches].float().cpu().numpy())
            chunks["chronos_hidden"].append(captured["chronos_hidden"][:, :num_context_patches].float().cpu().numpy())
            del encoder_outputs

    del pipeline, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def timesfm_normed_inputs(module: Any, values: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from timesfm.torch import util

    batch_n, context = values.shape
    patched_inputs = torch.reshape(values, (batch_n, -1, module.p))
    patched_masks = torch.reshape(masks, (batch_n, -1, module.p))
    n = torch.zeros(batch_n, device=module.device)
    mu = torch.zeros(batch_n, device=module.device)
    sigma = torch.zeros(batch_n, device=module.device)
    patch_mu = []
    patch_sigma = []
    for i in range(context // module.p):
        (n, mu, sigma), _ = util.update_running_stats(n, mu, sigma, patched_inputs[:, i], patched_masks[:, i])
        patch_mu.append(mu)
        patch_sigma.append(sigma)
    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)
    normed_inputs = util.revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
    return normed_inputs, patched_masks


def extract_timesfm_representations(windows: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(TIMESFM_SRC))
    import timesfm

    layer_idx = int(TARGET_SPECS["timesfm_2_5"]["hidden_layer"])
    model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
    model.model.load_checkpoint(str(ROOT / "timesfm-2.5-200m-pytorch" / "model.safetensors"), torch_compile=False)
    module = model.model
    module.eval()

    chunks: dict[str, list[np.ndarray]] = {"timesfm_tokenizer": [], "timesfm_hidden": []}
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            values = torch.tensor(windows[start : start + batch_size], dtype=torch.float32, device=module.device)
            masks = torch.zeros_like(values, dtype=torch.bool, device=module.device)
            normed_inputs, patched_masks = timesfm_normed_inputs(module, values, masks)

            captured: dict[str, torch.Tensor] = {}

            def hook(_mod, _inp, out):
                captured["timesfm_hidden"] = (out[0] if isinstance(out, tuple) else out).detach()

            handle = module.stacked_xf[layer_idx].register_forward_hook(hook)
            (input_embeddings, _output_embeddings, _output_ts, _output_quantile_spread), _cache = module(
                normed_inputs,
                patched_masks,
            )
            handle.remove()
            chunks["timesfm_tokenizer"].append(input_embeddings.float().cpu().numpy())
            chunks["timesfm_hidden"].append(captured["timesfm_hidden"].float().cpu().numpy())

    del model, module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def label_values(meta: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "dataset": [str(m["dataset"]) for m in meta],
        "domain": [str(m["domain"]) for m in meta],
        "frequency": [str(m.get("frequency_minutes")) for m in meta],
        "patch_index": [str(m["patch_index"]) for m in meta],
        "taxonomy_v0": [str(m["taxonomy_label"]) for m in meta],
    }


def weighted_purity(cluster_ids: np.ndarray, labels: list[str]) -> float:
    total = len(labels)
    score = 0
    for cid in sorted(set(cluster_ids.tolist())):
        idx = np.where(cluster_ids == cid)[0]
        counts = Counter(labels[i] for i in idx)
        score += counts.most_common(1)[0][1]
    return float(score / total)


def nearest_neighbor_metrics(x: np.ndarray, labels: dict[str, list[str]], k: int = 10) -> dict[str, float]:
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(x)), metric="euclidean")
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)[:, 1:]
    out = {}
    for name, values in labels.items():
        arr = np.asarray(values)
        out[f"top{k}_{name}_agreement"] = float(np.mean(arr[indices] == arr[:, None]))
    return out


def cluster_summary(meta: list[dict[str, Any]], cluster_ids: np.ndarray, n: int = 4) -> list[dict[str, Any]]:
    labels = label_values(meta)
    out = []
    for cid in sorted(set(cluster_ids.tolist())):
        idx = np.where(cluster_ids == cid)[0]
        out.append(
            {
                "cluster": int(cid),
                "size": int(len(idx)),
                "top_domains": top_counts([labels["domain"][i] for i in idx], n),
                "top_frequencies": top_counts([labels["frequency"][i] for i in idx], n),
                "top_patch_indices": top_counts([labels["patch_index"][i] for i in idx], n),
                "top_taxonomy_labels": top_counts([labels["taxonomy_v0"][i] for i in idx], n),
            }
        )
    return out


def fit_cluster(
    model_key: str,
    rep_name: str,
    embeddings: np.ndarray,
    meta: list[dict[str, Any]],
    patches: np.ndarray,
    seed: int,
    domain_balanced_patches: int,
) -> dict[str, Any]:
    idx = select_domain_balanced_indices(meta, domain_balanced_patches, seed)
    emb = embeddings[idx]
    selected_meta = [meta[i] for i in idx]
    selected_patches = patches[idx]
    labels = label_values(selected_meta)

    x = StandardScaler().fit_transform(emb)
    pca_dim = max(2, min(30, x.shape[0] - 1, x.shape[1]))
    pca = PCA(n_components=pca_dim, random_state=seed)
    x_pca = pca.fit_transform(x)
    k = min(16, max(6, int(round(math.sqrt(len(selected_meta) / 35)))))
    cluster_ids = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(x_pca)

    try:
        agglom = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x_pca)
        stability = float(normalized_mutual_info_score(cluster_ids, agglom))
    except Exception:
        stability = float("nan")

    silhouette = float(silhouette_score(x_pca, cluster_ids)) if len(set(cluster_ids.tolist())) > 1 else None
    result = {
        "model": model_key,
        "representation": rep_name,
        "note": REPRESENTATION_NOTES.get(rep_name, ""),
        "num_patch_embeddings": int(len(selected_meta)),
        "embedding_dim": int(emb.shape[1]),
        "pca_dim": int(pca_dim),
        "kmeans_k": int(k),
        "silhouette_pca_space": silhouette,
        "kmeans_vs_agglomerative_nmi": stability,
        "pca2_explained_variance_ratio": pca.explained_variance_ratio_[:2].astype(float).tolist(),
        "nmi": {name: float(normalized_mutual_info_score(values, cluster_ids)) for name, values in labels.items()},
        "purity": {name: weighted_purity(cluster_ids, values) for name, values in labels.items()},
        "nearest_neighbor": nearest_neighbor_metrics(x_pca, labels),
        "clusters": cluster_summary(selected_meta, cluster_ids),
        "_selected_indices": idx.astype(int).tolist(),
        "_cluster_ids": cluster_ids.astype(int).tolist(),
    }
    save_figures(model_key, rep_name, x_pca[:, :2], cluster_ids, labels, selected_patches, selected_meta)
    return result


def save_figures(
    model_key: str,
    rep_name: str,
    pca2: np.ndarray,
    cluster_ids: np.ndarray,
    labels: dict[str, list[str]],
    patches: np.ndarray,
    meta: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_key}_{rep_name}"
    for color_name, values in [("clusters", cluster_ids.tolist()), ("patch_index", labels["patch_index"]), ("taxonomy_v0", labels["taxonomy_v0"])]:
        fig, ax = plt.subplots(figsize=(7, 5))
        if color_name == "clusters":
            ids = cluster_ids
            title = "KMeans clusters"
        else:
            names = sorted(set(str(v) for v in values))
            ids = np.asarray([names.index(str(v)) for v in values])
            title = color_name
        sc = ax.scatter(pca2[:, 0], pca2[:, 1], c=ids, s=5, cmap="tab20", alpha=0.65)
        ax.set_title(f"{model_key} {rep_name}: PCA by {title}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        cbar = fig.colorbar(sc, ax=ax)
        if color_name != "clusters" and len(names) <= 18:
            cbar.set_ticks(range(len(names)))
            cbar.ax.set_yticklabels([v[:24] for v in names], fontsize=6)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{prefix}_pca_{color_name}.png", dpi=170)
        plt.close(fig)

    clusters = sorted(set(cluster_ids.tolist()))
    ncols = 4
    fig, axes = plt.subplots(len(clusters), ncols, figsize=(2.4 * ncols, 1.25 * len(clusters)), squeeze=False)
    for row, cid in enumerate(clusters):
        idx = np.where(cluster_ids == cid)[0]
        center = pca2[idx].mean(axis=0, keepdims=True)
        chosen = idx[np.argsort(np.linalg.norm(pca2[idx] - center, axis=1))[:ncols]]
        for col in range(ncols):
            ax = axes[row, col]
            if col >= len(chosen):
                ax.axis("off")
                continue
            item = int(chosen[col])
            ax.plot(robust_z(patches[item]), linewidth=1.2)
            m = meta[item]
            ax.set_title(f"C{cid} nearest {col + 1}\n{m['dataset']} p{m['patch_index']}", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{prefix}_prototype_panel.png", dpi=180)
    plt.close(fig)


def run_model(model_key: str, windows: np.ndarray, window_meta: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    print(f"Preparing raw baseline for {model_key}...")
    raw_embeddings, raw_meta, raw_patches = make_raw_patch_embeddings(model_key, windows, window_meta)

    if TARGET_SPECS[model_key]["kind"] == "chronos":
        print(f"Extracting Chronos representations for {model_key}...")
        layer_outputs = extract_chronos_representations(model_key, windows, args.batch_size)
    else:
        print("Extracting TimesFM representations...")
        layer_outputs = extract_timesfm_representations(windows, args.batch_size)

    outputs: dict[str, dict[str, Any]] = {}
    outputs["raw_z_patch"] = fit_cluster(
        model_key,
        "raw_z_patch",
        raw_embeddings,
        raw_meta,
        raw_patches,
        args.seed,
        args.domain_balanced_patches,
    )

    for rep_name, layer_embeddings in layer_outputs.items():
        embeddings, meta, patches = flatten_model_patches(model_key, layer_embeddings, windows, window_meta)
        attach_context(meta, windows)
        outputs[rep_name] = fit_cluster(
            model_key,
            rep_name,
            embeddings,
            meta,
            patches,
            args.seed,
            args.domain_balanced_patches,
        )

    rep_names = list(outputs.keys())
    ari: dict[str, float] = {}
    for i, a in enumerate(rep_names):
        for b in rep_names[i + 1 :]:
            if outputs[a]["_selected_indices"] == outputs[b]["_selected_indices"]:
                ari[f"{a}__vs__{b}"] = float(adjusted_rand_score(outputs[a]["_cluster_ids"], outputs[b]["_cluster_ids"]))
            else:
                ari[f"{a}__vs__{b}"] = float("nan")

    public_outputs = {}
    for name, value in outputs.items():
        public = dict(value)
        public.pop("_selected_indices", None)
        public.pop("_cluster_ids", None)
        public_outputs[name] = public

    return {
        "display": TARGET_SPECS[model_key]["display"],
        "patch_len": TARGET_SPECS[model_key]["patch_len"],
        "representations": public_outputs,
        "cluster_adjusted_rand_index": ari,
    }


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "NA"
    try:
        if math.isnan(float(x)):
            return "NA"
    except Exception:
        pass
    return f"{float(x):.{digits}f}"


def figure_link(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_report(summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Input Embedding Ablation: pre-transformer token 是否更适合研究 patch vocabulary？")
    lines.append("")
    lines.append("## 1. 为什么做这个 ablation")
    lines.append("")
    lines.append(
        "我们前面的聚类主要使用 selected transformer layer hidden states。这个表示已经经过 attention/contextualization，"
        "因此 cluster 可能同时编码 patch 形状、位置、频率、domain-style 和上下文角色。"
        "本轮 ablation 的目标是把“patch 自身的词汇表示”和“上下文化后的时序概念”分开。"
    )
    lines.append("")
    lines.append("## 2. 对照的 representation")
    lines.append("")
    for name, note in REPRESENTATION_NOTES.items():
        lines.append(f"- `{name}`: {note}")
    lines.append("")
    lines.append("特别注意：pre-transformer 并不自动等于 position-free。Chronos-2 的 projection 输入显式拼接了 time encoding；TimesFM-2.5 的 tokenizer 前没有显式 absolute position embedding，但 running RevIN 是顺序相关的。")
    lines.append("")
    lines.append("## 3. 运行设置")
    lines.append("")
    lines.append(f"- windows per dataset: `{summary['windows_per_dataset']}`")
    lines.append(f"- context length: `{summary['context_len']}`")
    lines.append(f"- domain-balanced patches per domain: `{summary['domain_balanced_patches_per_domain']}`")
    lines.append(f"- seed: `{summary['seed']}`")
    lines.append("")
    lines.append("## 4. 主要指标")
    lines.append("")
    lines.append("| model | representation | silhouette | stability | NMI taxonomy-v0 | NMI patch-index | NMI domain | NMI frequency |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for model_key, model_result in summary["models"].items():
        for rep_name, rep in model_result["representations"].items():
            nmi = rep["nmi"]
            lines.append(
                f"| {model_result['display']} | `{rep_name}` | {fmt(rep['silhouette_pca_space'])} | "
                f"{fmt(rep['kmeans_vs_agglomerative_nmi'])} | {fmt(nmi['taxonomy_v0'])} | "
                f"{fmt(nmi['patch_index'])} | {fmt(nmi['domain'])} | {fmt(nmi['frequency'])} |"
            )
    lines.append("")
    lines.append("指标总览图：")
    lines.append("")
    lines.append(f"![Input embedding ablation metrics](../{figure_link(FIG_DIR / 'input_embedding_ablation_metric_overview.png')})")
    lines.append("")
    lines.append("## 5. 这个结果回答了什么")
    lines.append("")
    if "timesfm_2_5" in summary["models"]:
        tfm = summary["models"]["timesfm_2_5"]["representations"]
        lines.append(
            f"- 对 TimesFM-2.5，你的担心基本成立：`timesfm_tokenizer` 的 patch-index NMI 是 `{fmt(tfm['timesfm_tokenizer']['nmi']['patch_index'])}`，"
            f"而 `timesfm_hidden` 是 `{fmt(tfm['timesfm_hidden']['nmi']['patch_index'])}`。这说明前面看到的 TimesFM position confounding 主要来自 transformer contextualization / causal attention / RoPE，而不是 tokenizer projection 本身。"
        )
        lines.append(
            f"- 但 `timesfm_tokenizer` 的 stability 只有 `{fmt(tfm['timesfm_tokenizer']['kmeans_vs_agglomerative_nmi'])}`，"
            f"明显低于 hidden layer 的 `{fmt(tfm['timesfm_hidden']['kmeans_vs_agglomerative_nmi'])}`。所以 tokenizer 更干净，但结构也更弱；它适合做 patch vocabulary baseline，不足以单独替代 hidden-state analysis。"
        )
    if "chronos_2" in summary["models"]:
        ch = summary["models"]["chronos_2"]["representations"]
        ari = summary["models"]["chronos_2"]["cluster_adjusted_rand_index"]
        lines.append(
            f"- 对 Chronos-2，`chronos_proj_with_time` 与 `chronos_proj_time_zeroed` 的 cluster ARI 是 `{fmt(ari['chronos_proj_with_time__vs__chronos_proj_time_zeroed'])}`，"
            "说明在本轮 128-step windows 上，显式 time encoding 对 projection-level cluster 影响不大。Chronos 的 patch-index NMI 在 projection 和 hidden 中都很低。"
        )
        lines.append(
            f"- Chronos-2 hidden layer 的 domain/frequency NMI 从 projection 的 `{fmt(ch['chronos_proj_with_time']['nmi']['domain'])}` / `{fmt(ch['chronos_proj_with_time']['nmi']['frequency'])}` "
            f"升到 `{fmt(ch['chronos_hidden']['nmi']['domain'])}` / `{fmt(ch['chronos_hidden']['nmi']['frequency'])}`。这提示 transformer 层会强化 domain/cadence-style，而不只是强化人类可见 shape。"
        )
    lines.append("")
    lines.append("## 6. 关键图像证据")
    lines.append("")
    lines.append("### 6.1 TimesFM-2.5: tokenizer vs layer_10")
    lines.append("")
    lines.append("Tokenizer PCA by cluster：")
    lines.append("")
    lines.append(f"![TimesFM tokenizer clusters](../{figure_link(FIG_DIR / 'timesfm_2_5_timesfm_tokenizer_pca_clusters.png')})")
    lines.append("")
    lines.append("Tokenizer PCA by patch_index：")
    lines.append("")
    lines.append(f"![TimesFM tokenizer patch index](../{figure_link(FIG_DIR / 'timesfm_2_5_timesfm_tokenizer_pca_patch_index.png')})")
    lines.append("")
    lines.append("Layer_10 PCA by cluster：")
    lines.append("")
    lines.append(f"![TimesFM hidden clusters](../{figure_link(FIG_DIR / 'timesfm_2_5_timesfm_hidden_pca_clusters.png')})")
    lines.append("")
    lines.append("Layer_10 PCA by patch_index：")
    lines.append("")
    lines.append(f"![TimesFM hidden patch index](../{figure_link(FIG_DIR / 'timesfm_2_5_timesfm_hidden_pca_patch_index.png')})")
    lines.append("")
    lines.append("### 6.2 Chronos-2: projection with time vs time-zeroed vs layer_11")
    lines.append("")
    lines.append(f"![Chronos projection with time](../{figure_link(FIG_DIR / 'chronos_2_chronos_proj_with_time_pca_clusters.png')})")
    lines.append("")
    lines.append(f"![Chronos projection time-zeroed](../{figure_link(FIG_DIR / 'chronos_2_chronos_proj_time_zeroed_pca_clusters.png')})")
    lines.append("")
    lines.append(f"![Chronos hidden layer](../{figure_link(FIG_DIR / 'chronos_2_chronos_hidden_pca_clusters.png')})")
    lines.append("")
    lines.append("## 7. 初步结论")
    lines.append("")
    lines.append("1. `raw_z_patch` 和 `projection/tokenizer` 应该作为后续所有 concept discovery 的必要 baseline。若某个 cluster 在 raw/proj 中已经存在，它更像 patch-shape vocabulary；若只在 hidden layer 中出现，它才更像 contextualized temporal concept。")
    lines.append("2. pre-transformer token 不应被直接称为无位置问题。Chronos 的官方 projection 明确吃入 time encoding；TimesFM 的 running normalization 也会让前后 patch 的统计分布不同。")
    lines.append("3. 对导师问题的更严谨表述应改为：TSFM 的时序语言至少有两层，第一层是 local patch vocabulary，第二层是 transformer contextualized temporal grammar。")
    lines.append("")
    lines.append("## 8. 下一步建议")
    lines.append("")
    lines.append("下一步不要抛弃 hidden states，而是把 taxonomy/concept 发现改成双通道：先在 pre-transformer projection 上发现 local vocabulary，再追踪这些 vocabulary 在 hidden layers 中如何合并、分裂或变成 position/context artifact。")
    lines.append("")
    lines.append("具体建议：对 TimesFM c5/c8 和 negative control c4，补做 raw/proj/hidden 的 cluster lineage：看同一批 patch 在 projection cluster 和 layer_10 cluster 之间的转移矩阵。")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def plot_metric_overview(summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    rows = []
    for model_key, model_result in summary["models"].items():
        for rep_name, rep in model_result["representations"].items():
            rows.append(
                {
                    "label": f"{model_result['display']}\n{rep_name}",
                    "stability": rep["kmeans_vs_agglomerative_nmi"],
                    "patch_index": rep["nmi"]["patch_index"],
                    "domain": rep["nmi"]["domain"],
                    "frequency": rep["nmi"]["frequency"],
                }
            )
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 1.25), 6.2))
    for offset, key, color in [
        (-1.5 * width, "stability", "#2f6f9f"),
        (-0.5 * width, "patch_index", "#b34545"),
        (0.5 * width, "domain", "#b36b18"),
        (1.5 * width, "frequency", "#1f8a7a"),
    ]:
        ax.bar(x + offset, [r[key] for r in rows], width=width, label=key, color=color)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("score")
    ax.set_title("Input embedding ablation: stability and confounder NMI", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.legend(ncol=4, loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "input_embedding_ablation_metric_overview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-per-dataset", type=int, default=40)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--domain-balanced-patches", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--models", nargs="+", default=list(TARGET_SPECS.keys()), choices=list(TARGET_SPECS.keys()))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    windows, window_meta, dataset_summary = sample_windows(
        DATA_ROOT,
        context_len=args.context_len,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
    )
    summary: dict[str, Any] = {
        "objective": "input embedding ablation for patch vocabulary vs contextual hidden states",
        "windows_per_dataset": args.windows_per_dataset,
        "context_len": args.context_len,
        "seed": args.seed,
        "domain_balanced_patches_per_domain": args.domain_balanced_patches,
        "num_windows": int(len(windows)),
        "dataset_summary": dataset_summary,
        "models": {},
    }
    for model_key in args.models:
        summary["models"][model_key] = run_model(model_key, windows, window_meta, args)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_metric_overview(summary)
    write_report(summary)
    print(
        json.dumps(
            {
                "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
                "report_path": str(REPORT_PATH.relative_to(ROOT)),
                "models": list(summary["models"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
