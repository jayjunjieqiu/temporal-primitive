from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_chronos_multilayer_cluster_validation import (  # noqa: E402
    DATA_ROOT,
    DISPLAY_NAMES,
    MACRO_DOMAIN_ORDER,
    build_rep_data,
    plot_center_nearest,
    sample_windows,
    select_balanced_indices,
)


REPRESENTATIONS = ["projection", "layer_6", "layer_11"]
BEST_K = {"projection": 6, "layer_6": 10, "layer_11": 6}
SOURCE_ORDER = [
    "traffic flow",
    "traffic speed",
    "road occupancy rates",
    "electricity consumption",
    "electricity transformer temperature",
    "weather",
    "Beijing air quality",
    "exchange rate",
    "illness data",
    "simulated Gaussian data",
    "simulated pulse data",
]


def color_ids(values: list[str], preferred_order: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    if preferred_order:
        order = [v for v in preferred_order if v in set(values)]
        order += sorted(v for v in set(values) if v not in set(order))
    else:
        order = sorted(set(values))
    mapping = {v: i for i, v in enumerate(order)}
    return np.asarray([mapping[v] for v in values], dtype=int), order


def fit_tsne_space(embeddings: np.ndarray, seed: int, perplexity: float) -> tuple[np.ndarray, dict[str, Any]]:
    x = StandardScaler().fit_transform(embeddings)
    # PCA precompression denoises high-dimensional hidden states before t-SNE;
    # clustering and visualization are both done on the final 2D t-SNE coordinates.
    pca_dim = max(2, min(50, x.shape[0] - 1, x.shape[1]))
    x_pre = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    effective_perplexity = float(min(perplexity, max(5, (len(x_pre) - 1) // 3)))
    kwargs = dict(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="euclidean",
    )
    try:
        x_tsne = TSNE(max_iter=1000, **kwargs).fit_transform(x_pre)
    except TypeError:
        x_tsne = TSNE(n_iter=1000, **kwargs).fit_transform(x_pre)
    return x_tsne.astype(np.float32), {"pca_pre_dim": int(pca_dim), "perplexity": effective_perplexity}


def plot_cluster_map(rep: str, x_tsne: np.ndarray, labels: np.ndarray, centers: np.ndarray, k: int, fig_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    sc = ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels, s=6, cmap="tab20", alpha=0.58, linewidths=0)
    ax.scatter(centers[:, 0], centers[:, 1], c="black", s=34, marker="x", linewidths=1.4)
    for cid in range(k):
        ax.text(centers[cid, 0], centers[cid, 1], f"C{cid}", fontsize=8.5, weight="bold")
    ax.set_title(f"{DISPLAY_NAMES.get(rep, rep)}: t-SNE then KMeans, K={k}", fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(color="#edf2f7", linewidth=0.7)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Cluster")
    fig.tight_layout()
    out = fig_dir / f"{rep}_tsne_kmeans_k{k}.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_domain_map(
    rep: str,
    x_tsne: np.ndarray,
    metadata: list[dict[str, Any]],
    label_key: str,
    preferred_order: list[str] | None,
    fig_dir: Path,
) -> str:
    values = [str(m[label_key]) for m in metadata]
    ids, order = color_ids(values, preferred_order)
    cmap = plt.get_cmap("tab20", max(1, len(order)))
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.scatter(
        x_tsne[:, 0],
        x_tsne[:, 1],
        c=ids,
        s=6,
        cmap=cmap,
        alpha=0.60,
        linewidths=0,
        vmin=-0.5,
        vmax=len(order) - 0.5,
    )
    ax.set_title(f"{DISPLAY_NAMES.get(rep, rep)}: t-SNE map colored by {label_key.replace('_', ' ')}", fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(color="#edf2f7", linewidth=0.7)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(i), markeredgecolor="none", markersize=5, label=label)
        for i, label in enumerate(order)
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.2, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    suffix = "source_domain" if label_key == "source_domain" else "macro_domain"
    out = fig_dir / f"{rep}_tsne_{suffix}_labels.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def write_cluster_report(path: Path, outputs: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# t-SNE 降维后 KMeans 聚类结果与原型图",
        "",
        "> 这份报告只看 `projection`、`layer_6` 和 `layer_11`。流程是先对 Chronos-2 representation 做 `StandardScaler + PCA precompression`，再用 t-SNE 降到 2D，并在这个 2D t-SNE space 中做 KMeans。也就是说，这份报告回答的是“t-SNE 可视化空间里能否形成直观 clusters”。",
        "",
        "## 1. 方法说明",
        "",
        "实际流程：",
        "",
        "```text",
        "Chronos-2 representation",
        "-> StandardScaler",
        "-> PCA precompression up to 50 dimensions",
        "-> t-SNE to 2D",
        "-> KMeans in 2D t-SNE space",
        "-> t-SNE scatter visualization",
        "-> t-SNE-space KMeans-center nearest raw patches as examples",
        "```",
        "",
        "K setting 使用每层当前最佳值：`projection K=6`，`layer_6 K=10`，`layer_11 K=6`。`layer_6` 用 `K=10` 是因为 per-layer K search 指向更细的 middle-layer split。",
        "",
        "注意：t-SNE 强调局部邻域可视化，不保留严格全局距离。因此这份报告主要用于 human inspection 和 presentation，不替代高维 representation-space stability / NMI / DTW validation。",
        "",
    ]
    for rep in REPRESENTATIONS:
        info = outputs[rep]
        display = DISPLAY_NAMES.get(rep, rep)
        lines += [
            f"## {display}",
            "",
            f"- K: `{info['k']}`",
            f"- clustered patches: `{info['num_points']}`",
            f"- t-SNE perplexity: `{info['tsne']['perplexity']}`",
            "",
            f"![{display} t-SNE KMeans](../{Path(info['cluster_map']).relative_to(ROOT)})",
            "",
            f"![{display} center-nearest prototypes](../{Path(info['prototype_panel']).relative_to(ROOT)})",
            "",
            "读法：上图是 t-SNE 2D space，颜色是 KMeans labels；下图是每个 cluster 中离 t-SNE-space KMeans center 最近的 raw patches。cluster ID 只表示可视化空间中的 neighborhood，不是 motif 名字。",
            "",
        ]
    lines += [
        "## 结论边界",
        "",
        "- 这些图可以说明 Chronos-2 representation space 中存在稳定 neighborhood。",
        "- `projection` 更接近 local patch token vocabulary。",
        "- `layer_6 K=10` 用来观察 middle layer 的更细 contextual substructure。",
        "- `layer_11` 更稳定但不一定更适合直接命名 raw-shape motif。",
        "- 是否能称为 candidate motif/prototype family，还需要 original-space DTW validation 和 domain/frequency/position confounder audit。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_domain_report(path: Path, outputs: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# t-SNE 降维后直接打上 domain 标签的诊断图",
        "",
        "> 这份报告不做聚类命名，只问一个 confounder 问题：Chronos-2 representation 在 t-SNE 2D 可视化空间里，是不是明显被 domain/source-domain 分开？",
        "",
        "这里主图使用 `source_domain` 标签，因为它最接近数据集描述里的 domain；同时补充 `macro_domain` 标签，便于快速看 Traffic / Energy / Environment 等大类是否主导结构。",
        "",
        "重要提醒：t-SNE 适合看局部邻域和可视化团块，但不能把二维距离当成严格全局几何。domain 混杂不能只靠肉眼判断，最终仍应结合 NMI、controlled retrieval 和 macro-domain evidence。",
        "",
    ]
    for rep in REPRESENTATIONS:
        info = outputs[rep]
        display = DISPLAY_NAMES.get(rep, rep)
        lines += [
            f"## {display}",
            "",
            f"### Source-domain labels",
            "",
            f"![{display} source-domain labels](../{Path(info['source_domain_map']).relative_to(ROOT)})",
            "",
            f"### Macro-domain labels",
            "",
            f"![{display} macro-domain labels](../{Path(info['macro_domain_map']).relative_to(ROOT)})",
            "",
            "读法：如果同一颜色在 t-SNE map 上形成大块分离，说明该层 representation 可能编码了较强 domain/source-domain information；如果颜色混合较多，说明该二维视图下 domain confounding 较弱。但这不是充分证据，仍需看 NMI 指标。",
            "",
        ]
    lines += [
        "## 对 TSFM 设计的启发",
        "",
        "- 如果 early layer 的 t-SNE map 更按 local shape 而不是 domain 分开，它更适合作为 `local patch vocabulary` 的观察窗口。",
        "- 如果 middle/top layer 的 t-SNE map 更受 domain 或 macro-domain 影响，说明 contextual representation 可能混入 domain/cadence information。",
        "- 新 TSFM 设计里需要显式区分 `shared temporal commonality` 和 `domain-specific heterogeneity`，否则 prototype family 可能退化成 domain labels。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-per-dataset", type=int, default=500)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-per-macro-domain", type=int, default=1500)
    parser.add_argument("--max-per-dataset-within-macro-domain", type=int, default=350)
    parser.add_argument("--max-per-source-domain", type=int, default=900)
    parser.add_argument("--tsne-perplexity", type=float, default=50.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "tsne_cluster_domain_reports")
    args = parser.parse_args()

    out_dir = args.output_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "docs").mkdir(exist_ok=True)

    windows, window_meta, dataset_summary = sample_windows(
        DATA_ROOT,
        args.context_len,
        args.windows_per_dataset,
        args.seed,
    )
    rep_data = build_rep_data(windows, window_meta, args.batch_size)

    outputs: dict[str, dict[str, Any]] = {}
    for rep in REPRESENTATIONS:
        data = rep_data[rep]
        idx = select_balanced_indices(
            data.metadata,
            "macro_domain",
            args.max_per_macro_domain,
            args.max_per_dataset_within_macro_domain,
            args.max_per_source_domain,
            args.seed,
        )
        metadata = [data.metadata[i] for i in idx]
        x_tsne, tsne_info = fit_tsne_space(data.embeddings[idx], args.seed, args.tsne_perplexity)
        patches = data.patches[idx]
        k = BEST_K[rep]
        kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=20).fit(x_tsne)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        cluster_map = plot_cluster_map(rep, x_tsne, labels, centers, k, fig_dir)
        source_domain_map = plot_domain_map(rep, x_tsne, metadata, "source_domain", SOURCE_ORDER, fig_dir)
        macro_domain_map = plot_domain_map(rep, x_tsne, metadata, "macro_domain", MACRO_DOMAIN_ORDER, fig_dir)
        plot_center_nearest(
            rep,
            list(range(k)),
            x_tsne,
            labels,
            centers,
            metadata,
            patches,
            fig_dir,
            output_name=f"{rep}_tsne_kmeans_k{k}_center_nearest",
            title_suffix=f"K={k}",
        )
        prototype_panel = str(fig_dir / f"{rep}_tsne_kmeans_k{k}_center_nearest.png")

        outputs[rep] = {
            "k": k,
            "num_points": int(len(idx)),
            "tsne": tsne_info,
            "cluster_map": cluster_map,
            "prototype_panel": prototype_panel,
            "source_domain_map": source_domain_map,
            "macro_domain_map": macro_domain_map,
        }

    summary = {
        "objective": "t-SNE-space KMeans and domain-label overlay reports for selected Chronos-2 layers",
        "representations": REPRESENTATIONS,
        "best_k": BEST_K,
        "windows_per_dataset": args.windows_per_dataset,
        "context_len": args.context_len,
        "seed": args.seed,
        "dataset_summary": dataset_summary,
        "outputs": outputs,
    }
    (out_dir / "tsne_cluster_domain_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_cluster_report(ROOT / "docs" / "97_tsne_then_kmeans_cluster_report.md", outputs)
    write_domain_report(ROOT / "docs" / "98_tsne_domain_label_report.md", outputs)
    print(json.dumps({"output_dir": str(out_dir), "reports": ["docs/97_tsne_then_kmeans_cluster_report.md", "docs/98_tsne_domain_label_report.md"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
