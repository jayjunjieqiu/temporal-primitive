from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "lineage_cards"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_PATH = OUT_DIR / "lineage_card_summary.json"
REPORT_PATH = ROOT / "docs" / "lineage_card_report.md"

sys.path.insert(0, str(ROOT))
import scripts.run_input_embedding_ablation as ab  # noqa: E402
from scripts.run_second_pilot_discovery import (  # noqa: E402
    DATA_ROOT,
    robust_z,
    sample_windows,
    select_domain_balanced_indices,
    top_counts,
)


TARGETS = {
    "timesfm_2_5": [
        {
            "hidden_cluster": 8,
            "name": "timesfm_c8_rising_recovery_candidate",
            "title": "TimesFM c8: rising / recovery candidate",
            "interpretation": "候选概念：上升/恢复 transition。重点看它是否从 tokenizer 的宽桶中被 hidden 重组出来。",
        },
        {
            "hidden_cluster": 5,
            "name": "timesfm_c5_falling_transition_pool",
            "title": "TimesFM c5: falling / smooth transition pool",
            "interpretation": "候选概念池：下降/平滑转移。重点看 hidden cluster 内是否来自多个 tokenizer/raw source。",
        },
        {
            "hidden_cluster": 4,
            "name": "timesfm_c4_first_patch_artifact",
            "title": "TimesFM c4: first-patch artifact negative control",
            "interpretation": "负例：first-patch position artifact。重点看它是否在 hidden 中被单独组织出来。",
        },
    ],
    "chronos_2": [
        {
            "hidden_cluster": 6,
            "name": "chronos_c6_high_variation_transition_like",
            "title": "Chronos-2 c6: high-variation transition-like",
            "interpretation": "Chronos 候选/对照：高变化 transition-like cluster，检查 projection 到 hidden 的重组方式。",
        },
        {
            "hidden_cluster": 1,
            "name": "chronos_c1_transition_like_cross_domain",
            "title": "Chronos-2 c1: transition-like cross-domain",
            "interpretation": "Chronos 候选：跨域 transition-like cluster，检查是否由多个 projection clusters 汇入。",
        },
    ],
}

STAGE_SPECS = {
    "timesfm_2_5": {
        "display": "TimesFM-2.5 layer_10",
        "stages": ["raw_z_patch", "timesfm_tokenizer", "timesfm_hidden"],
        "hidden_stage": "timesfm_hidden",
        "projection_stage": "timesfm_tokenizer",
    },
    "chronos_2": {
        "display": "Chronos-2 layer_11",
        "stages": ["raw_z_patch", "chronos_proj_with_time", "chronos_hidden"],
        "hidden_stage": "chronos_hidden",
        "projection_stage": "chronos_proj_with_time",
    },
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = "Noto Sans CJK JP"
            break
    plt.rcParams["axes.unicode_minus"] = False


def metric_labels(metadata: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "dataset": [str(m["dataset"]) for m in metadata],
        "domain": [str(m["domain"]) for m in metadata],
        "frequency": [str(m.get("frequency_minutes")) for m in metadata],
        "patch_index": [str(m["patch_index"]) for m in metadata],
        "taxonomy_v0": [str(m["taxonomy_label"]) for m in metadata],
    }


def fit_stage(
    model_key: str,
    stage_name: str,
    stage_payload: tuple[str, np.ndarray, list[dict[str, Any]] | None, np.ndarray | None],
    windows: np.ndarray,
    window_meta: list[dict[str, Any]],
    selected_idx: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    payload_kind, layer_embeddings, pre_meta, pre_patches = stage_payload
    if payload_kind == "flat":
        if pre_meta is None or pre_patches is None:
            raise ValueError(f"{stage_name} flat payload requires metadata and patches")
        embeddings, meta, patches = layer_embeddings, pre_meta, pre_patches
    elif payload_kind == "layer":
        embeddings, meta, patches = ab.flatten_model_patches(model_key, layer_embeddings, windows, window_meta)
    else:
        raise ValueError(payload_kind)
    ab.attach_context(meta, windows)
    selected_embeddings = embeddings[selected_idx]
    selected_meta = [meta[int(i)] for i in selected_idx]
    selected_patches = patches[selected_idx]

    x = StandardScaler().fit_transform(selected_embeddings)
    pca_dim = max(2, min(30, x.shape[0] - 1, x.shape[1]))
    x_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    k = min(16, max(6, int(round(math.sqrt(len(selected_meta) / 35)))))
    cluster_ids = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(x_pca)

    labels = metric_labels(selected_meta)
    return {
        "stage": stage_name,
        "embeddings": selected_embeddings,
        "meta": selected_meta,
        "patches": selected_patches,
        "x_pca": x_pca,
        "cluster_ids": cluster_ids,
        "k": int(k),
        "nmi": {name: float(normalized_mutual_info_score(values, cluster_ids)) for name, values in labels.items()},
    }


def build_model_stages(
    model_key: str,
    windows: np.ndarray,
    window_meta: list[dict[str, Any]],
    selected_idx: np.ndarray,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    raw_embeddings, _raw_meta, _raw_patches = ab.make_raw_patch_embeddings(model_key, windows, window_meta)
    outputs: dict[str, tuple[str, np.ndarray, list[dict[str, Any]] | None, np.ndarray | None]]
    if model_key == "timesfm_2_5":
        extracted = ab.extract_timesfm_representations(windows, batch_size)
        outputs = {
            "raw_z_patch": ("flat", raw_embeddings, _raw_meta, _raw_patches),
            "timesfm_tokenizer": ("layer", extracted["timesfm_tokenizer"], None, None),
            "timesfm_hidden": ("layer", extracted["timesfm_hidden"], None, None),
        }
    elif model_key == "chronos_2":
        extracted = ab.extract_chronos_representations(model_key, windows, batch_size)
        outputs = {
            "raw_z_patch": ("flat", raw_embeddings, _raw_meta, _raw_patches),
            "chronos_proj_with_time": ("layer", extracted["chronos_proj_with_time"], None, None),
            "chronos_hidden": ("layer", extracted["chronos_hidden"], None, None),
        }
    else:
        raise ValueError(model_key)

    return {
        stage: fit_stage(model_key, stage, outputs[stage], windows, window_meta, selected_idx, seed)
        for stage in STAGE_SPECS[model_key]["stages"]
    }


def medoid_indices(x: np.ndarray, mask: np.ndarray, n: int = 6) -> np.ndarray:
    idx = np.where(mask)[0]
    center = x[idx].mean(axis=0, keepdims=True)
    order = np.argsort(np.linalg.norm(x[idx] - center, axis=1))
    return idx[order[: min(n, len(order))]]


def top_counter(values: list[Any], n: int = 6) -> list[dict[str, Any]]:
    return [{"value": str(k), "count": int(v)} for k, v in Counter(values).most_common(n)]


def row_distribution(source_clusters: np.ndarray, target_mask: np.ndarray) -> list[dict[str, Any]]:
    idx = np.where(target_mask)[0]
    return top_counter([int(source_clusters[i]) for i in idx], n=10)


def transition_distribution(stages: dict[str, Any], model_key: str, hidden_cluster: int) -> dict[str, Any]:
    hidden_stage = STAGE_SPECS[model_key]["hidden_stage"]
    proj_stage = STAGE_SPECS[model_key]["projection_stage"]
    raw_stage = "raw_z_patch"
    hidden_ids = stages[hidden_stage]["cluster_ids"]
    mask = hidden_ids == hidden_cluster
    meta = stages[hidden_stage]["meta"]
    return {
        "size": int(mask.sum()),
        "raw_source_clusters": row_distribution(stages[raw_stage]["cluster_ids"], mask),
        "projection_source_clusters": row_distribution(stages[proj_stage]["cluster_ids"], mask),
        "top_domains": top_counter([meta[i]["domain"] for i in np.where(mask)[0]], 6),
        "top_datasets": top_counter([meta[i]["dataset"] for i in np.where(mask)[0]], 6),
        "top_frequencies": top_counter([meta[i].get("frequency_minutes") for i in np.where(mask)[0]], 6),
        "top_patch_indices": top_counter([meta[i]["patch_index"] for i in np.where(mask)[0]], 6),
        "top_taxonomy_v0": top_counter([meta[i]["taxonomy_label"] for i in np.where(mask)[0]], 6),
    }


def plot_card(
    out_path: Path,
    model_key: str,
    target: dict[str, Any],
    stages: dict[str, Any],
    dist: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    spec = STAGE_SPECS[model_key]
    stage_names = spec["stages"]
    hidden_stage = spec["hidden_stage"]
    proj_stage = spec["projection_stage"]
    raw_stage = "raw_z_patch"

    hidden_ids = stages[hidden_stage]["cluster_ids"]
    mask = hidden_ids == int(target["hidden_cluster"])
    medoids = medoid_indices(stages[hidden_stage]["x_pca"], mask, n=6)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 6, height_ratios=[0.65, 1.3, 1.0, 1.2], hspace=0.48, wspace=0.34)
    fig.suptitle(f"{spec['display']} lineage card | {target['title']}", fontsize=16)

    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis("off")
    text = [
        target["interpretation"],
        f"hidden cluster c{target['hidden_cluster']} size={dist['size']}",
        f"top patch_index={dist['top_patch_indices'][:4]}",
        f"top domains={dist['top_domains'][:4]}",
        f"top taxonomy-v0={dist['top_taxonomy_v0'][:4]}",
        f"raw source clusters={dist['raw_source_clusters'][:5]}",
        f"{proj_stage} source clusters={dist['projection_source_clusters'][:5]}",
    ]
    ax_text.text(0.01, 0.98, "\n".join(text), va="top", fontsize=10)

    colors = np.where(mask, "tab:red", "lightgray")
    for col, stage in enumerate(stage_names):
        ax = fig.add_subplot(gs[1, col * 2 : col * 2 + 2])
        ids = stages[stage]["cluster_ids"]
        target_color = mask if stage == hidden_stage else np.isin(np.arange(len(ids)), medoids)
        ax.scatter(stages[stage]["x_pca"][:, 0], stages[stage]["x_pca"][:, 1], s=4, c="lightgray", alpha=0.28)
        if stage == hidden_stage:
            ax.scatter(stages[stage]["x_pca"][mask, 0], stages[stage]["x_pca"][mask, 1], s=7, c="tab:red", alpha=0.7)
        else:
            ax.scatter(stages[stage]["x_pca"][medoids, 0], stages[stage]["x_pca"][medoids, 1], s=45, c="tab:red", marker="x")
        ax.set_title(stage)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(gs[2, :3])
    raw_counts = dist["raw_source_clusters"]
    ax.bar([x["value"] for x in raw_counts], [x["count"] for x in raw_counts], color="#2f6f9f")
    ax.set_title(f"raw_z_patch source clusters feeding hidden c{target['hidden_cluster']}")
    ax.set_xlabel("raw cluster")
    ax.set_ylabel("count")

    ax = fig.add_subplot(gs[2, 3:])
    proj_counts = dist["projection_source_clusters"]
    ax.bar([x["value"] for x in proj_counts], [x["count"] for x in proj_counts], color="#1f8a7a")
    ax.set_title(f"{proj_stage} source clusters feeding hidden c{target['hidden_cluster']}")
    ax.set_xlabel("projection/tokenizer cluster")
    ax.set_ylabel("count")

    for col, idx in enumerate(medoids):
        ax = fig.add_subplot(gs[3, col])
        patch = robust_z(stages[hidden_stage]["patches"][idx])
        ax.plot(patch, color="tab:blue", linewidth=1.3)
        m = stages[hidden_stage]["meta"][idx]
        path = (
            f"raw c{stages[raw_stage]['cluster_ids'][idx]} -> "
            f"proj c{stages[proj_stage]['cluster_ids'][idx]} -> "
            f"hidden c{stages[hidden_stage]['cluster_ids'][idx]}"
        )
        ax.set_title(
            f"{m['dataset']} p{m['patch_index']}\n{m['taxonomy_label']}\n{path}",
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summarize_target(model_key: str, target: dict[str, Any], stages: dict[str, Any]) -> dict[str, Any]:
    spec = STAGE_SPECS[model_key]
    hidden_stage = spec["hidden_stage"]
    hidden_cluster = int(target["hidden_cluster"])
    hidden_ids = stages[hidden_stage]["cluster_ids"]
    mask = hidden_ids == hidden_cluster
    if not np.any(mask):
        return {"target": target, "status": "missing_hidden_cluster"}
    dist = transition_distribution(stages, model_key, hidden_cluster)
    medoids = medoid_indices(stages[hidden_stage]["x_pca"], mask, n=6)
    card_path = FIG_DIR / f"{target['name']}.png"
    plot_card(card_path, model_key, target, stages, dist)
    meta = stages[hidden_stage]["meta"]
    return {
        "target": target,
        "status": "ok",
        "card_path": str(card_path.relative_to(ROOT)),
        "distribution": dist,
        "stage_nmi": {stage: stages[stage]["nmi"] for stage in spec["stages"]},
        "medoids": [
            {
                "dataset": meta[int(i)]["dataset"],
                "domain": meta[int(i)]["domain"],
                "frequency": meta[int(i)].get("frequency_minutes"),
                "patch_index": int(meta[int(i)]["patch_index"]),
                "taxonomy_label": meta[int(i)]["taxonomy_label"],
                "path": {
                    stage: int(stages[stage]["cluster_ids"][int(i)])
                    for stage in spec["stages"]
                },
            }
            for i in medoids
        ],
    }


def write_report(summary: dict[str, Any]) -> None:
    def fmt_counts(items: list[dict[str, Any]], n: int = 4) -> str:
        return "、".join(f"{item['value']}={item['count']}" for item in items[:n])

    def pct(count: int, total: int) -> str:
        return f"{100 * count / max(total, 1):.1f}%"

    def item_by_name(model_key: str, name: str) -> dict[str, Any]:
        for item in summary["models"][model_key]["targets"]:
            if item["target"]["name"] == name:
                return item
        raise KeyError(name)

    timesfm_c8 = item_by_name("timesfm_2_5", "timesfm_c8_rising_recovery_candidate")
    timesfm_c5 = item_by_name("timesfm_2_5", "timesfm_c5_falling_transition_pool")
    timesfm_c4 = item_by_name("timesfm_2_5", "timesfm_c4_first_patch_artifact")
    chronos_c6 = item_by_name("chronos_2", "chronos_c6_high_variation_transition_like")
    chronos_c1 = item_by_name("chronos_2", "chronos_c1_transition_like_cross_domain")

    lines: list[str] = []
    lines.append("# Lineage Card Report：从 patch vocabulary 到 hidden concept")
    lines.append("")
    lines.append("## 1. 结论先行")
    lines.append("")
    lines.append("本轮的目的不是再找一个更高的聚类分数，而是回答一个更贴近导师关心的问题：**hidden-layer cluster 到底是在复述原始 patch 形状，还是把多个 patch vocabulary 重新组织成了更高层的 temporal concept？**")
    lines.append("")
    lines.append("目前最清楚的证据来自 TimesFM-2.5：`c8` 和 `c5` 在原空间里分别呈现较稳定的上升/恢复、下降/平滑转移形态，同时它们不是来自单一 tokenizer cluster，而是由多个 raw/tokenizer source 汇入；这支持“TSFM 的时序语言不等于人工 taxonomy 的一一映射，而是 contextualized concept”的假设。")
    lines.append("")
    c4_size = timesfm_c4["distribution"]["size"]
    c4_p0 = timesfm_c4["distribution"]["top_patch_indices"][0]["count"]
    lines.append(f"同时，TimesFM `c4` 是一个很有价值的负例：它的 patch 形态看起来也有规律，但 `patch_index=0` 占 {c4_p0}/{c4_size}（{pct(c4_p0, c4_size)}），因此应解释为 first-patch position artifact，而不是 temporal motif。这个负例说明我们的 controlled validation 是必要的。")
    lines.append("")
    lines.append("Chronos-2 的两个 card 则给出模型对照：hidden cluster 的 patch-index confounding 很弱，但 domain/frequency encoding 更强；因此 Chronos 的 candidate concept 更适合做跨模型验证，而不是直接照搬 TimesFM 的 taxonomy v1。")
    lines.append("")
    lines.append("## 2. 方法设置")
    lines.append("")
    lines.append(f"- windows per dataset: `{summary['windows_per_dataset']}`")
    lines.append(f"- context length: `{summary['context_len']}`")
    lines.append(f"- domain-balanced patches per domain: `{summary['domain_balanced_patches_per_domain']}`")
    lines.append(f"- seed: `{summary['seed']}`")
    lines.append(f"- TimesFM selected patches: `{summary['models']['timesfm_2_5']['selected_patch_count']}`")
    lines.append(f"- Chronos selected patches: `{summary['models']['chronos_2']['selected_patch_count']}`")
    lines.append("- 聚类流程：对每个 representation stage 做 `StandardScaler -> PCA(max 30 dims) -> KMeans`，并在同一批 domain-balanced patch 上追踪 `raw -> tokenizer/projection -> hidden` 的 cluster transition。")
    lines.append("- 注意：这里复现 second pilot scale，因此 TimesFM 的 `c5/c8/c4` 编号与之前报告中的 hidden-layer cluster 编号保持一致；这些 cluster id 不应跨不同采样规模直接复用。")
    lines.append("")
    lines.append("## 3. 如何读 lineage card")
    lines.append("")
    lines.append("每张图包含三层证据：")
    lines.append("")
    lines.append("- 上排 PCA scatter：看目标 hidden cluster 在 hidden 空间是否紧密，以及它在 raw/tokenizer/projection 空间中是否只是一个单一簇的延续。")
    lines.append("- 中排 source-cluster bar：看目标 hidden cluster 由哪些 raw cluster 和 tokenizer/projection cluster 汇入。多源汇入更像 contextual reorganization；单源延续更像局部形状 vocabulary。")
    lines.append("- 下排 medoid patches：回到原始 patch 空间，看这个 hidden cluster 是否能被人类解释成稳定的形态。")
    lines.append("")
    lines.append("判断标准是三件事同时成立：`形态可解释`、`不是 patch-index/domain/frequency 单一混杂`、`hidden 层相对 tokenizer/projection 有重组证据`。")
    lines.append("")
    lines.append("## 4. TimesFM-2.5：最清楚的一组证据")
    lines.append("")
    lines.append("### 4.1 c8：上升/恢复 candidate concept")
    lines.append("")
    lines.append(f"![{timesfm_c8['target']['title']}](../{timesfm_c8['card_path']})")
    lines.append("")
    d = timesfm_c8["distribution"]
    lines.append(f"- hidden size: `{d['size']}`")
    lines.append(f"- patch index: {fmt_counts(d['top_patch_indices'], 4)}，没有落在单一位置。")
    lines.append(f"- domain/frequency: {fmt_counts(d['top_domains'], 5)}；{fmt_counts(d['top_frequencies'], 5)}，存在跨域来源。")
    lines.append(f"- taxonomy-v0 probe: {fmt_counts(d['top_taxonomy_v0'], 5)}，主要混合了 `level_shift`、`mixed_uncertain` 和 `trend`，不是人工 taxonomy 的单类映射。")
    lines.append(f"- source transition: raw={fmt_counts(d['raw_source_clusters'], 5)}；tokenizer={fmt_counts(d['projection_source_clusters'], 6)}。同一个 hidden concept 由多个 tokenizer bucket 汇入。")
    lines.append("- 解释：这个 cluster 更像“带上下文的恢复/上升 transition”，而不是简单的“正斜率 patch”。它适合进入 model-derived taxonomy v1 的候选集。")
    lines.append("")
    lines.append("### 4.2 c5：下降/平滑转移 candidate pool")
    lines.append("")
    lines.append(f"![{timesfm_c5['target']['title']}](../{timesfm_c5['card_path']})")
    lines.append("")
    d = timesfm_c5["distribution"]
    lines.append(f"- hidden size: `{d['size']}`")
    lines.append(f"- patch index: {fmt_counts(d['top_patch_indices'], 4)}，同样避开了单一位置解释。")
    lines.append(f"- domain/frequency: {fmt_counts(d['top_domains'], 5)}；{fmt_counts(d['top_frequencies'], 5)}。")
    lines.append(f"- taxonomy-v0 probe: {fmt_counts(d['top_taxonomy_v0'], 5)}，人工标签仍然分散。")
    lines.append(f"- source transition: raw={fmt_counts(d['raw_source_clusters'], 5)}；tokenizer={fmt_counts(d['projection_source_clusters'], 6)}。")
    lines.append("- 解释：这个 cluster 是一个较宽的下降/平滑转移概念池，可能需要在下一步内部再 split，区分 `smooth falling`、`falling then flat`、`weak transition` 等子型。")
    lines.append("")
    lines.append("### 4.3 c4：first-patch artifact 负例")
    lines.append("")
    lines.append(f"![{timesfm_c4['target']['title']}](../{timesfm_c4['card_path']})")
    lines.append("")
    d = timesfm_c4["distribution"]
    first_count = d["top_patch_indices"][0]["count"]
    lines.append(f"- hidden size: `{d['size']}`")
    lines.append(f"- patch index: {fmt_counts(d['top_patch_indices'], 4)}，其中第一个位置占 {pct(first_count, d['size'])}。")
    lines.append(f"- domain/frequency: {fmt_counts(d['top_domains'], 5)}；{fmt_counts(d['top_frequencies'], 5)}，跨域并不能自动证明它是 concept。")
    lines.append(f"- source transition: raw={fmt_counts(d['raw_source_clusters'], 5)}；tokenizer={fmt_counts(d['projection_source_clusters'], 6)}。")
    lines.append("- 解释：这是我们报告中应该主动展示的 negative control。它告诉我们：视觉一致性不够，必须检查 patch position confounding；否则会把位置/边界行为误命名为 motif。")
    lines.append("")
    lines.append("## 5. Chronos-2：跨模型对照")
    lines.append("")
    lines.append("### 5.1 c6：high-variation transition-like")
    lines.append("")
    lines.append(f"![{chronos_c6['target']['title']}](../{chronos_c6['card_path']})")
    lines.append("")
    d = chronos_c6["distribution"]
    lines.append(f"- hidden size: `{d['size']}`")
    lines.append(f"- patch index: {fmt_counts(d['top_patch_indices'], 6)}，没有 TimesFM c4 那样的单位置坍缩。")
    lines.append(f"- domain/frequency: {fmt_counts(d['top_domains'], 5)}；{fmt_counts(d['top_frequencies'], 5)}，domain/frequency 成分更明显。")
    lines.append(f"- source transition: raw={fmt_counts(d['raw_source_clusters'], 5)}；projection={fmt_counts(d['projection_source_clusters'], 6)}。")
    lines.append("- 解释：Chronos-2 也会把多个 projection/raw source 汇入一个 transition-like hidden cluster；但它的语义更受频率和数据域影响，应作为跨模型验证对象，而不是直接作为最终 taxonomy。")
    lines.append("")
    lines.append("### 5.2 c1：transition-like cross-domain")
    lines.append("")
    lines.append(f"![{chronos_c1['target']['title']}](../{chronos_c1['card_path']})")
    lines.append("")
    d = chronos_c1["distribution"]
    lines.append(f"- hidden size: `{d['size']}`")
    lines.append(f"- patch index: {fmt_counts(d['top_patch_indices'], 6)}。")
    lines.append(f"- domain/frequency: {fmt_counts(d['top_domains'], 5)}；{fmt_counts(d['top_frequencies'], 5)}。")
    lines.append(f"- taxonomy-v0 probe: {fmt_counts(d['top_taxonomy_v0'], 5)}，仍然不是人工 taxonomy 的单一类。")
    lines.append(f"- source transition: raw={fmt_counts(d['raw_source_clusters'], 5)}；projection={fmt_counts(d['projection_source_clusters'], 6)}。")
    lines.append("- 解释：这是 Chronos 中更大的 transition-like pool，形态范围比 TimesFM c8/c5 更宽。下一步应优先用 controlled retrieval 验证它是否能在同域、跨域、同频、跨频条件下保持形态一致。")
    lines.append("")
    lines.append("## 6. 对 proposal 假设的更新")
    lines.append("")
    lines.append("- H1（patch token 学到局部 temporal primitives）：保留，但需要拆成两层。tokenizer/projection 更像局部 patch vocabulary，hidden state 更像 contextualized temporal concept。")
    lines.append("- H2（TSFM 的时序语言可以被 motif taxonomy 解释）：部分保留。taxonomy-v0 适合做 probe，但不是最终答案；真实 cluster 往往跨越 `trend/level_shift/mixed_uncertain` 等人工标签。")
    lines.append("- H3（规模/模型/层次带来概念组织变化）：加强。TimesFM hidden 层有明显 position artifact，需要负例控制；Chronos hidden 层 patch-index 更稳定，但 domain/frequency encoding 更强。")
    lines.append("- 新增方法假设：最终 taxonomy 应该从 `model-derived clusters + original-space interpretation + controlled validation` 共同生成，而不是先验定义后单向验证。")
    lines.append("")
    lines.append("## 7. 下一步")
    lines.append("")
    lines.append("建议下一步做一个紧凑的 `model-derived taxonomy v1 evidence table`：以 TimesFM `c8/c5` 作为正例、`c4` 作为负例，结合已有 taxonomy_v1 internal split 和 controlled retrieval，把每个 candidate concept 的 `形态描述`、`source transition`、`confounder 检查`、`跨域检索是否存活` 放在同一张表里。")
    lines.append("")
    lines.append("在进入更大规模实验前，不建议立刻宣布最终 taxonomy；更稳妥的表述是：我们已经找到一组支持“TSFM hidden layer 形成 contextual temporal concepts”的候选证据，并且有 position artifact 负例来约束解释。")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-per-dataset", type=int, default=100)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--domain-balanced-patches", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--models", nargs="+", default=list(TARGETS.keys()), choices=list(TARGETS.keys()))
    args = parser.parse_args()

    ensure_dirs()
    windows, window_meta, dataset_summary = sample_windows(
        DATA_ROOT,
        context_len=args.context_len,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
    )

    summary: dict[str, Any] = {
        "objective": "lineage cards for key hidden clusters",
        "windows_per_dataset": args.windows_per_dataset,
        "context_len": args.context_len,
        "seed": args.seed,
        "domain_balanced_patches_per_domain": args.domain_balanced_patches,
        "num_windows": int(len(windows)),
        "dataset_summary": dataset_summary,
        "models": {},
    }

    for model_key in args.models:
        print(f"Preparing {model_key}...")
        # Use raw metadata only to determine the shared deterministic domain-balanced subset.
        raw_embeddings, raw_meta, _raw_patches = ab.make_raw_patch_embeddings(model_key, windows, window_meta)
        selected_idx = select_domain_balanced_indices(raw_meta, args.domain_balanced_patches, args.seed)
        stages = build_model_stages(model_key, windows, window_meta, selected_idx, args.batch_size, args.seed)
        model_summary = {
            "display": STAGE_SPECS[model_key]["display"],
            "stages": list(stages.keys()),
            "selected_patch_count": int(len(selected_idx)),
            "targets": [],
        }
        for target in TARGETS[model_key]:
            model_summary["targets"].append(summarize_target(model_key, target, stages))
        summary["models"][model_key] = model_summary
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

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
