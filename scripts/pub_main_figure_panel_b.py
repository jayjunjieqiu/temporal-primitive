"""Publication panel b — representation atlas / cluster maps（SVG，对应整图第 2、3 行）。

复用 build_bolt_main_figure 的提取缓存与同一套处理流程（flat-filter → domain-balanced →
PCA(30)+KMeans → t-SNE），保证和 main_B 的聚类/着色一致；只把版式做成出版版：

  - 2 行（Layer 1 / Layer 12）× 3 列（model clusters | human motif v0 | macro domain）；
  - 列标题只在顶行写一次，行标签 "Layer N" 放最左；不画顶部 suptitle（caption 用户自写）；
  - SVG 矢量、文字可编辑（svg.fonttype='none'）、字体放大；
  - 每列 legend 放该列正下方。

纯 CPU（读缓存，不加载模型）。从仓库根目录运行：
    .venv/bin/python scripts/pub_main_figure_panel_b.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/temporal_primitive_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_bolt_main_figure import (  # noqa: E402
    DOMAIN_COLORS, MOTIF_LABELS, cluster_pca_fit, flatten_patches, label_patch,
    select_domain_balanced_indices,
)
from sklearn.manifold import TSNE  # noqa: E402

# 与 build_bolt_main_figure 默认一致（canonical 版）
LAYERS = [0, 11]
K = 8
SEED = 47
PERPLEXITY = 40.0
MIN_PATCH_STD = 0.15
MAX_PER_DOMAIN = 400
PCA_DIM: int | None = 30   # 聚类空间：30=PCA(现状)；None=完整 768 维 representation（--cluster-space full）
CACHE = ROOT / "outputs" / "figures" / "bolt_main_figure" / ".cache" / \
    "extract_train_wpd200_val150_ctx128_seed47_layers0-11.pkl"
OUT_DIR = ROOT / "figure_projects" / "pub_main_figure"
TSNE_CACHE = OUT_DIR / ".panel_b_tsne_cache.npz"  # 缓存 t-SNE 坐标，layout 微调时秒级重画

# 出版字号
FS_COLTITLE = 18
FS_ROWLABEL = 18
FS_AXIS = 12
FS_LEG = 14

# 列标题（结合论文叙事）——放进各 legend 框作标题，首字母大写（sentence case）
COL_TITLES = ["Training data domains", "Predefined motif labels", "Learned temporal primitives"]
DIM_MOTIFS = {"mixed_uncertain": "#cfcfcf"}
# motif 标签缩写 + 首字母大写（与 cluster C1.. / domain Traffic.. 统一）
MOTIF_SHORT = {
    "flat_low_information": "Flat", "trend": "Trend", "oscillation": "Oscillation",
    "impulse_spike": "Impulse", "burst_event": "Burst", "level_shift": "Level shift",
    "volatility_shift": "Vol. shift", "intermittent": "Intermittent", "mixed_uncertain": "Mixed",
}


def build_atlas_data():
    """复刻 build_bolt_main_figure.main() 的处理流程，返回 (layer_clusters, v0_labels, domain_labels)。"""
    if not CACHE.exists():
        raise SystemExit(f"extraction cache not found: {CACHE}\n先跑 build_bolt_main_figure.py 生成缓存。")
    with open(CACHE, "rb") as fh:
        b = pickle.load(fh)
    disc_z, disc_meta, disc_reps, patch_len = b["disc_z"], b["disc_meta"], b["disc_reps"], b["patch_len"]

    disc_flat = {L: flatten_patches(disc_reps[f"layer_{L}"], disc_z, disc_meta, patch_len) for L in LAYERS}

    # 近平直 patch 筛除（robust-z 窗口 std < 阈值），用 layer0 raw 判定、对各层一致施加
    raw0 = disc_flat[LAYERS[0]][1]
    keep = np.array([float(np.std(raw0[i])) >= MIN_PATCH_STD for i in range(len(raw0))])
    disc_flat = {L: (e[keep], r[keep], [m[i] for i in range(len(m)) if keep[i]])
                 for L, (e, r, m) in disc_flat.items()}

    meta0 = disc_flat[LAYERS[0]][2]
    sel = select_domain_balanced_indices(meta0, max_per_domain=MAX_PER_DOMAIN, seed=SEED)

    layer_clusters = {}
    for L in LAYERS:
        emb_b = disc_flat[L][0][sel]
        labels, _centers, pca_coords, _sc, _pca, _km = cluster_pca_fit(emb_b, K, SEED, PCA_DIM)
        layer_clusters[L] = (labels, pca_coords)

    raw_b0 = disc_flat[LAYERS[0]][1][sel]
    v0_labels = np.array([label_patch(raw_b0[i], patch_len).label for i in range(len(sel))])
    domain_labels = np.array([disc_flat[LAYERS[0]][2][i]["macro_domain"] for i in sel])
    return layer_clusters, v0_labels, domain_labels


def main() -> None:
    global K, PCA_DIM, OUT_DIR, TSNE_CACHE
    ap = argparse.ArgumentParser(description="panel b cluster maps（默认 PCA(30) 聚类）")
    ap.add_argument("--cluster-space", choices=["pca", "full"], default="pca",
                    help="pca=StandardScaler→PCA(30)→KMeans（现状）；full=直接在 768 维上 KMeans")
    ap.add_argument("--k", type=int, default=K, help="KMeans 簇数（full 版建议用 k-sweep 选出的 6）")
    args = ap.parse_args()
    K = args.k
    if args.cluster_space == "full":
        PCA_DIM = None
        OUT_DIR = ROOT / "figure_projects" / "pub_main_figure_fullrep"
        TSNE_CACHE = OUT_DIR / f".panel_b_tsne_cache_full_k{K}.npz"
        fig_h = round(7.0 * 3.3 / 4, 2) + 0.7   # full 版散点压到 3.3/4，再加顶部 legend 带
    else:
        PCA_DIM = 30
        OUT_DIR = ROOT / "figure_projects" / "pub_main_figure"
        TSNE_CACHE = OUT_DIR / (".panel_b_tsne_cache.npz" if K == 8
                                else f".panel_b_tsne_cache_pca_k{K}.npz")
        fig_h = 7.0 + 0.7                        # 加顶部 legend 带
    print(f"[panel b] cluster-space={args.cluster_space} (pca_dim={PCA_DIM}) k={K} h={fig_h} -> {OUT_DIR.name}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
    })
    # t-SNE 是唯一耗时步骤（~分钟级）；缓存坐标，layout 微调时秒级重画
    if TSNE_CACHE.exists():
        print(f"[panel b] loading t-SNE cache -> {TSNE_CACHE.name}")
        z = np.load(TSNE_CACHE, allow_pickle=True)
        xy_by_layer = {L: z[f"xy_{L}"] for L in LAYERS}
        labels_by_layer = {L: z[f"lab_{L}"] for L in LAYERS}
        v0_labels, domain_labels = z["v0"], z["dom"]
    else:
        layer_clusters, v0_labels, domain_labels = build_atlas_data()
        xy_by_layer, labels_by_layer = {}, {}
        for L in LAYERS:
            labels, pca_coords = layer_clusters[L]
            print(f"[panel b] t-SNE layer {L + 1} ({len(pca_coords)} pts) ...")
            xy_by_layer[L] = TSNE(2, perplexity=PERPLEXITY, init="pca", random_state=SEED,
                                  max_iter=1000).fit_transform(pca_coords)
            labels_by_layer[L] = labels
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(TSNE_CACHE, v0=v0_labels, dom=domain_labels,
                 **{f"xy_{L}": xy_by_layer[L] for L in LAYERS},
                 **{f"lab_{L}": labels_by_layer[L] for L in LAYERS})

    clu_cmap = plt.get_cmap("tab10")
    motif_cmap = plt.get_cmap("tab10")
    motif_color = {lab: motif_cmap(i % 10) for i, lab in enumerate(MOTIF_LABELS)}
    motif_color.update(DIM_MOTIFS)
    motif_draw_order = ([m for m in MOTIF_LABELS if m in DIM_MOTIFS]
                        + [m for m in MOTIF_LABELS if m not in DIM_MOTIFS])
    seen_domains = [d for d in DOMAIN_COLORS if d in set(domain_labels.tolist())]

    fig, axes = plt.subplots(len(LAYERS), 3, figsize=(15.5, fig_h), squeeze=False)
    for row, L in enumerate(LAYERS):
        labels, xy = labels_by_layer[L], xy_by_layer[L]

        # 列序：col0 = Domain，col1 = Predefined motifs，col2 = Learned primitives
        ax = axes[row, 0]
        for dom in seen_domains:
            m = domain_labels == dom
            ax.scatter(xy[m, 0], xy[m, 1], s=6, color=DOMAIN_COLORS[dom], alpha=0.65)

        ax = axes[row, 1]
        for lab in motif_draw_order:
            m = v0_labels == lab
            if not m.any():
                continue
            dim = lab in DIM_MOTIFS
            ax.scatter(xy[m, 0], xy[m, 1], s=5 if dim else 7,
                       color=motif_color[lab], alpha=0.12 if dim else 0.78,
                       zorder=1 if dim else 2)

        ax = axes[row, 2]
        for cid in range(K):
            m = labels == cid
            ax.scatter(xy[m, 0], xy[m, 1], s=6, color=clu_cmap(cid % 10), alpha=0.65)

        for col, ax in enumerate(axes[row]):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#1f2933"); sp.set_linewidth(0.8)
            # 列标题不再写在散点上方——改放进每列 legend 框的标题里（见下）
        # 行标签：Layer N（最左，竖排粗体）
        axes[row, 0].set_ylabel(f"Layer {L + 1}", fontsize=FS_ROWLABEL, fontweight="bold", labelpad=12)

    def _pretty(lab):
        return str(lab).replace("_", " ")

    clu_handles = [Line2D([0], [0], marker="o", ls="", ms=8, color=clu_cmap(c % 10), label=f"C{c + 1}")
                   for c in range(K)]
    # 图例顺序：交换 Burst <-> Intermittent，视觉更平衡（不影响散点着色）
    motif_order = list(MOTIF_LABELS)
    ib, ii = motif_order.index("burst_event"), motif_order.index("intermittent")
    motif_order[ib], motif_order[ii] = motif_order[ii], motif_order[ib]
    motif_handles = [Line2D([0], [0], marker="o", ls="", ms=8, color=motif_color[l], label=MOTIF_SHORT[l])
                     for l in motif_order]
    dom_handles = [Line2D([0], [0], marker="o", ls="", ms=8, color=DOMAIN_COLORS[d], label=_pretty(d))
                   for d in seen_domains]
    # 图例放到全图最上方（列标题作为各 legend 框标题，类似 panel c）；先用占位 top 取列中心，
    # 建好 legend 后量出其真实底边，再把散点网格顶 snap 到其正下方（留极小间隙）——既不空旷也不重叠
    fig.subplots_adjust(left=0.05, right=0.985, top=0.80, bottom=0.02, wspace=0.04, hspace=0.06)
    fig.canvas.draw()
    # 三组统一 ncol=3（3 排）+ mode="expand" 固定等宽 → 三框等大、不裁末列（Nature 细灰方框）；
    # 顶端对齐、向下展开；列标题进框作为加粗 legend title。
    LEG_W = 0.30
    LEG_Y = 0.995
    legs = []
    for handles, col in [(dom_handles, 0), (motif_handles, 1), (clu_handles, 2)]:
        pos = axes[0, col].get_position()
        xc = 0.5 * (pos.x0 + pos.x1)
        leg = fig.legend(
            handles=handles, loc="upper left",
            bbox_to_anchor=(xc - LEG_W / 2, LEG_Y, LEG_W, 0.0),
            mode="expand", ncol=3, fontsize=FS_LEG - 1, frameon=True,
            title=COL_TITLES[col], title_fontsize=FS_COLTITLE - 2,
            columnspacing=0.5, handletextpad=0.3, labelspacing=0.7,
            borderpad=0.8, fancybox=False, edgecolor="0.7", facecolor="white",
            framealpha=1.0,
        )
        leg.get_frame().set_linewidth(0.6)   # Nature 风格：方角、细灰边、等大三框
        leg.get_title().set_fontweight("bold")
        leg.set_alignment("center")          # 标题框在 legend 内居中（PNG）
        leg.get_title().set_horizontalalignment("center")  # SVG：标题用 text-anchor:middle → 换字体也居中（否则只有 PNG 居中）
        legs.append(leg)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    lb = min(leg.get_window_extent().transformed(inv).y0 for leg in legs)
    fig.subplots_adjust(top=lb - 0.018)   # 散点顶贴 legend 底，仅留极小间隙
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / "panel_b_cluster_maps.svg"
    png = OUT_DIR / "panel_b_cluster_maps.png"
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(png, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"[panel b] saved -> {svg}")
    print(f"[panel b] preview -> {png}")


if __name__ == "__main__":
    main()
