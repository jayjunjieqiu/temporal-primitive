from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "representation_lineage"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_PATH = OUT_DIR / "representation_lineage_summary.json"
REPORT_PATH = ROOT / "docs" / "representation_lineage_report.md"

import sys

sys.path.insert(0, str(ROOT))
import scripts.run_input_embedding_ablation as ab  # noqa: E402


MODEL_STAGE_SPECS = {
    "chronos_2_small": {
        "display": "Chronos-2-small layer_5",
        "stages": ["raw_z_patch", "chronos_proj_with_time", "chronos_hidden"],
        "hidden_stage": "chronos_hidden",
    },
    "chronos_2": {
        "display": "Chronos-2 layer_11",
        "stages": ["raw_z_patch", "chronos_proj_with_time", "chronos_proj_time_zeroed", "chronos_hidden"],
        "hidden_stage": "chronos_hidden",
    },
    "timesfm_2_5": {
        "display": "TimesFM-2.5 layer_10",
        "stages": ["raw_z_patch", "timesfm_tokenizer", "timesfm_hidden"],
        "hidden_stage": "timesfm_hidden",
    },
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def stage_key_to_human(stage: str) -> str:
    return stage


def pack_meta_key(meta: dict[str, Any]) -> tuple[int, int, int]:
    return int(meta["window_id"]), int(meta["patch_index"]), int(meta["global_start"])


def cluster_to_map(meta: list[dict[str, Any]], cluster_ids: np.ndarray, selected_indices: np.ndarray) -> dict[tuple[int, int, int], int]:
    out = {}
    for local_pos, global_idx in enumerate(selected_indices):
        out[pack_meta_key(meta[int(global_idx)])] = int(cluster_ids[local_pos])
    return out


def common_keys(keys_by_stage: dict[str, dict[tuple[int, int, int], int]], base_stage: str) -> list[tuple[int, int, int]]:
    base_keys = set(keys_by_stage[base_stage].keys())
    for mapping in keys_by_stage.values():
        base_keys &= set(mapping.keys())
    return sorted(base_keys)


def transition_matrix(source: list[int], target: list[int]) -> dict[str, Any]:
    s = np.asarray(source, dtype=int)
    t = np.asarray(target, dtype=int)
    s_labels = sorted(set(s.tolist()))
    t_labels = sorted(set(t.tolist()))
    s_index = {v: i for i, v in enumerate(s_labels)}
    t_index = {v: i for i, v in enumerate(t_labels)}
    mat = np.zeros((len(s_labels), len(t_labels)), dtype=int)
    for a, b in zip(s, t, strict=True):
        mat[s_index[int(a)], t_index[int(b)]] += 1
    row_sum = mat.sum(axis=1, keepdims=True)
    row_norm = np.divide(mat, np.where(row_sum == 0, 1, row_sum), dtype=float)
    row_max = row_norm.max(axis=1) if len(row_norm) else np.array([])
    row_entropy = np.zeros(len(row_norm), dtype=float)
    for i, row in enumerate(row_norm):
        nz = row[row > 0]
        row_entropy[i] = float(-(nz * np.log2(nz)).sum()) if len(nz) else 0.0
    return {
        "source_clusters": [int(v) for v in s_labels],
        "target_clusters": [int(v) for v in t_labels],
        "counts": mat.astype(int).tolist(),
        "row_normalized": row_norm.astype(float).tolist(),
        "row_max_mean": float(row_max.mean()) if len(row_max) else None,
        "row_entropy_mean": float(row_entropy.mean()) if len(row_entropy) else None,
        "row_argmax": [int(t_labels[int(i)]) if len(t_labels) else None for i in np.argmax(row_norm, axis=1)] if len(row_norm) else [],
    }


def plot_transition_heatmap(ax, matrix: np.ndarray, source_labels: list[int], target_labels: list[int], title: str, cmap: str = "Blues") -> None:
    import matplotlib.pyplot as plt

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("target cluster")
    ax.set_ylabel("source cluster")
    ax.set_xticks(range(len(target_labels)))
    ax.set_yticks(range(len(source_labels)))
    ax.set_xticklabels([str(x) for x in target_labels], fontsize=8)
    ax.set_yticklabels([str(x) for x in source_labels], fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= 0.18:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
    return im


def save_model_figure(model_key: str, model_summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    stages = model_summary["stages"]
    fig, axes = plt.subplots(1, len(model_summary["transitions"]), figsize=(5.2 * len(model_summary["transitions"]), 4.8))
    if len(model_summary["transitions"]) == 1:
        axes = [axes]
    last_im = None
    for ax, (transition_name, transition) in zip(axes, model_summary["transitions"].items(), strict=True):
        mat = np.asarray(transition["row_normalized"], dtype=float)
        last_im = plot_transition_heatmap(
            ax,
            mat,
            transition["source_clusters"],
            transition["target_clusters"],
            transition_name.replace("__to__", " → "),
        )
        ax.text(
            0.02,
            -0.18,
            f"ARI={transition['ari']:.3f} | NMI={transition['nmi']:.3f} | row-max={transition['row_max_mean']:.3f}",
            transform=ax.transAxes,
            fontsize=9,
        )
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02, label="row-normalized mass")
    fig.suptitle(f"{model_summary['display']} cluster lineage", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / f"{model_key}_lineage_heatmaps.png", dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Representation Lineage Report")
    lines.append("")
    lines.append("## 1. 为什么做 lineage")
    lines.append("")
    lines.append(
        "我们已经知道 raw/projection/tokenizer/hidden 的聚类稳定性不一样。lineage 的目标不是再找一次 cluster，"
        "而是看同一批 patch 在 representation 变换后是保留、拆分、合并，还是被 position/context 机制重新组织。"
    )
    lines.append("")
    lines.append("## 2. 方法")
    lines.append("")
    lines.append("- 同一批 windows，固定 seed。")
    lines.append("- 每个模型先取 `raw_z_patch` baseline，再取 pre-transformer 表示，再取 hidden layer。")
    lines.append("- 在同一 domain-balanced 子集上分别做 KMeans，然后比较 patch-level cluster transfer matrix。")
    lines.append("- 指标：`ARI`, `NMI`, row-normalized transition matrix, `row_max_mean`。")
    lines.append("")
    lines.append("## 3. 总览图")
    lines.append("")
    lines.append(f"![lineage overview](../{(FIG_DIR / 'lineage_overview.png').relative_to(ROOT)})")
    lines.append("")
    lines.append("## 4. 结果表")
    lines.append("")
    lines.append("| model | transition | ARI | NMI | row-max mean | row-entropy mean |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for model_key, model in summary["models"].items():
        for transition_name, transition in model["transitions"].items():
            lines.append(
                f"| {model['display']} | `{transition_name}` | {transition['ari']:.3f} | {transition['nmi']:.3f} | "
                f"{transition['row_max_mean']:.3f} | {transition['row_entropy_mean']:.3f} |"
            )
    lines.append("")
    lines.append("## 5. 模型级读法")
    lines.append("")
    for model_key, model in summary["models"].items():
        lines.append(f"### {model['display']}")
        lines.append("")
        lines.append(f"![{model['display']} lineage](../{(FIG_DIR / f'{model_key}_lineage_heatmaps.png').relative_to(ROOT)})")
        lines.append("")
        hidden_pair = "timesfm_tokenizer__to__timesfm_hidden" if model_key == "timesfm_2_5" else "chronos_proj_with_time__to__chronos_hidden"
        lines.append(
            f"- 总体上，`{model['hidden_stage']}` 相对前一层的 row-max mean = "
            f"{model['transitions'][hidden_pair]['row_max_mean']:.3f}, "
            f"说明 cluster identity 不是完全保留，而是在 hidden 中被重组。"
        )
        if model_key == "timesfm_2_5":
            lines.append(
                f"- TimesFM 的 `raw -> tokenizer` NMI 只有 {model['transitions']['raw_z_patch__to__timesfm_tokenizer']['nmi']:.3f}，"
                f"但 row-max mean 仍有 {model['transitions']['raw_z_patch__to__timesfm_tokenizer']['row_max_mean']:.3f}，"
                "说明 tokenizer 更像把 raw patch 压进少数宽桶，而不是形成稳定的一一对应概念。"
            )
            lines.append(
                f"- `tokenizer -> hidden` 的 NMI 升到 {model['transitions']['timesfm_tokenizer__to__timesfm_hidden']['nmi']:.3f}，"
                f"而 row-max mean 下降到 {model['transitions']['timesfm_tokenizer__to__timesfm_hidden']['row_max_mean']:.3f}，"
                "说明真正的概念重组主要发生在 transformer 层。"
            )
            lines.append(
                "- 这和我们前面的 patch-index 观察一致：tokenizer 本身并不强地编码 position，但 hidden 会把 patch 顺序结构明显放大。"
            )
        else:
            pre_key = "chronos_proj_with_time__to__chronos_hidden"
            lines.append(
                f"- Chronos 的 `projection -> hidden` NMI 为 {model['transitions'][pre_key]['nmi']:.3f}，"
                f"高于 `raw -> projection` 的 {model['transitions']['raw_z_patch__to__chronos_proj_with_time']['nmi']:.3f}；"
                "但 row-max mean 下降、entropy 上升，说明 hidden 不是简单保留 projection cluster，而是在上下文中重新分配 patch。"
            )
            if "chronos_proj_time_zeroed__to__chronos_hidden" in model["transitions"]:
                lines.append(
                    f"- `time_zeroed -> hidden` 的 NMI 是 {model['transitions']['chronos_proj_time_zeroed__to__chronos_hidden']['nmi']:.3f}，"
                    "表明 Chronos 的显式 time encoding 并不是这一轮 cluster 变化的主因。"
                )
        lines.append("")
    lines.append("## 6. 结论")
    lines.append("")
    lines.append("1. raw/projection/tokenizer 不是最终 concept；它们更像 patch vocabulary。")
    lines.append("2. hidden layer 不是简单的更强聚类，而是把 vocabulary 重新编排成 contextualized temporal concepts。")
    lines.append("3. 以后做 taxonomy，应当用 lineage 先筛掉只在前端表示里存在、但在 hidden 中消失的假概念。")
    lines.append("")
    lines.append("## 7. 下一步")
    lines.append("")
    lines.append("最值得做的是对 TimesFM 的 `c5/c8/c4`、Chronos 的 transition-like cluster 做 lineage card：把代表性 patch 在 raw / projection / hidden 三个空间里的路径放到一张图里。")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_overview(summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    rows = []
    for model_key, model in summary["models"].items():
        for transition_name, transition in model["transitions"].items():
            rows.append(
                {
                    "label": f"{model['display']}\n{transition_name}",
                    "ari": transition["ari"],
                    "nmi": transition["nmi"],
                    "row_max": transition["row_max_mean"],
                }
            )
    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 1.4), 5.2))
    x = np.arange(len(rows))
    ax.bar(x - 0.22, [r["ari"] for r in rows], width=0.22, label="ARI", color="#2f6f9f")
    ax.bar(x, [r["nmi"] for r in rows], width=0.22, label="NMI", color="#b36b18")
    ax.bar(x + 0.22, [r["row_max"] for r in rows], width=0.22, label="row-max mean", color="#1f8a7a")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("score")
    ax.set_title("Representation lineage overview")
    ax.legend(ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lineage_overview.png", dpi=180)
    plt.close(fig)


def run_model(model_key: str, windows: np.ndarray, window_meta: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    ab.FIG_DIR = FIG_DIR
    ab.OUT_DIR = OUT_DIR
    ab.CARD_DIR = FIG_DIR / "cards"
    ab.RETRIEVAL_DIR = FIG_DIR / "retrieval"
    ab.CARD_DIR.mkdir(parents=True, exist_ok=True)
    ab.RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)

    if model_key == "timesfm_2_5":
        raw_embeddings, raw_meta, raw_patches = ab.make_raw_patch_embeddings(model_key, windows, window_meta)
        tokenizer_outputs = ab.extract_timesfm_representations(windows, args.batch_size)
        stage_outputs = {
            "raw_z_patch": (raw_embeddings, raw_meta, raw_patches),
            "timesfm_tokenizer": ab.flatten_model_patches(model_key, tokenizer_outputs["timesfm_tokenizer"], windows, window_meta),
            "timesfm_hidden": ab.flatten_model_patches(model_key, tokenizer_outputs["timesfm_hidden"], windows, window_meta),
        }
    else:
        raw_embeddings, raw_meta, raw_patches = ab.make_raw_patch_embeddings(model_key, windows, window_meta)
        chronos_outputs = ab.extract_chronos_representations(model_key, windows, args.batch_size)
        stage_outputs = {
            "raw_z_patch": (raw_embeddings, raw_meta, raw_patches),
            "chronos_proj_with_time": ab.flatten_model_patches(model_key, chronos_outputs["chronos_proj_with_time"], windows, window_meta),
            "chronos_hidden": ab.flatten_model_patches(model_key, chronos_outputs["chronos_hidden"], windows, window_meta),
        }
        if "chronos_proj_time_zeroed" in chronos_outputs:
            stage_outputs["chronos_proj_time_zeroed"] = ab.flatten_model_patches(
                model_key, chronos_outputs["chronos_proj_time_zeroed"], windows, window_meta
            )

    stage_results: dict[str, Any] = {}
    for stage_name, (embeddings, meta, patches) in stage_outputs.items():
        ab.attach_context(meta, windows)
        res = ab.fit_cluster(
            model_key,
            f"{model_key}_{stage_name}",
            embeddings,
            meta,
            patches,
            args.seed,
            args.domain_balanced_patches,
        )
        stage_results[stage_name] = res

    transitions = {}
    stage_order = MODEL_STAGE_SPECS[model_key]["stages"]
    raw_selected = np.asarray(stage_results[stage_order[0]]["_selected_indices"], dtype=int)
    selected_consistent = all(
        np.array_equal(raw_selected, np.asarray(stage_results[stage]["_selected_indices"], dtype=int)) for stage in stage_order[1:]
    )
    if not selected_consistent:
        raise RuntimeError(f"Selected indices are not aligned for {model_key}; lineage expects deterministic shared selection.")

    meta_ref = stage_outputs["raw_z_patch"][1]
    common = [pack_meta_key(meta_ref[int(idx)]) for idx in raw_selected]

    # build per-stage aligned cluster assignments on the common key set
    aligned: dict[str, list[int]] = {}
    for stage in stage_order:
        aligned[stage] = [int(v) for v in np.asarray(stage_results[stage]["_cluster_ids"], dtype=int).tolist()]

    pair_order = [f"{stage_order[i]}__to__{stage_order[i+1]}" for i in range(len(stage_order) - 1)]
    if model_key == "timesfm_2_5":
        pair_order = ["raw_z_patch__to__timesfm_tokenizer", "timesfm_tokenizer__to__timesfm_hidden", "raw_z_patch__to__timesfm_hidden"]
    elif model_key.startswith("chronos"):
        pair_order = ["raw_z_patch__to__chronos_proj_with_time", "chronos_proj_with_time__to__chronos_hidden", "raw_z_patch__to__chronos_hidden"]
        if "chronos_proj_time_zeroed" in stage_order:
            pair_order.insert(2, "chronos_proj_time_zeroed__to__chronos_hidden")

    for pair in pair_order:
        src, tgt = pair.split("__to__")
        if src not in aligned or tgt not in aligned:
            continue
        tr = transition_matrix(aligned[src], aligned[tgt])
        tr["ari"] = float(adjusted_rand_score(aligned[src], aligned[tgt]))
        tr["nmi"] = float(normalized_mutual_info_score(aligned[src], aligned[tgt]))
        transitions[pair] = tr

    public_stage_results = {}
    for stage_name, res in stage_results.items():
        public = dict(res)
        public.pop("_selected_indices", None)
        public.pop("_cluster_ids", None)
        public_stage_results[stage_name] = public

    return {
        "display": MODEL_STAGE_SPECS[model_key]["display"],
        "hidden_stage": MODEL_STAGE_SPECS[model_key]["hidden_stage"],
        "pair_order": list(transitions.keys()),
        "stages": public_stage_results,
        "transitions": transitions,
        "common_patch_count": int(len(common)),
        "selected_indices": raw_selected.astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-per-dataset", type=int, default=40)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--domain-balanced-patches", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--models", nargs="+", default=list(MODEL_STAGE_SPECS.keys()), choices=list(MODEL_STAGE_SPECS.keys()))
    args = parser.parse_args()

    ensure_dirs()
    windows, window_meta, dataset_summary = ab.sample_windows(
        ab.DATA_ROOT,
        context_len=args.context_len,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
    )

    summary: dict[str, Any] = {
        "objective": "representation lineage from raw/projection/tokenizer to hidden",
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
        save_model_figure(model_key, summary["models"][model_key])
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    build_overview(summary)
    write_report(summary)
    print(
        json.dumps(
            {
                "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
                "report_path": str(REPORT_PATH.relative_to(ROOT)),
                "models": list(summary["models"].keys()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
