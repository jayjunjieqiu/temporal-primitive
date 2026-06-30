"""Publication panel c — Layer 12 cluster cards（raw patch-stack，SVG，对应整图第 4 行）。

复用 build_bolt_main_figure 的提取缓存与同一处理流程（flat-filter → domain-balanced →
PCA(30)+KMeans），取 layer 11（显示为 Layer 12）的 k=8 个 cluster，每簇画 center-nearest
TOP_N 条 z-normalized raw patch 的堆叠热图，标题下加整簇 macro-domain 构成条。

出版版差异：SVG 可编辑文字、字体放大、文案收紧、不画 suptitle（caption 用户自写）、
TOP_N=16（原 24，减少纵向占用）。纯 CPU（读缓存，不加载模型）。

从仓库根目录运行：
    .venv/bin/python scripts/pub_main_figure_panel_c.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/temporal_primitive_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_bolt_main_figure import (  # noqa: E402
    DOMAIN_COLORS, cluster_pca_fit, flatten_patches, select_domain_balanced_indices, z_normalize,
)

LAYER = 11           # encoder block 11 -> 显示 "Layer 12"
K = 8
SEED = 47
MIN_PATCH_STD = 0.15
MAX_PER_DOMAIN = 400
TOP_N = 16           # 每簇展示的 center-nearest patch 数（原 24）
PCA_DIM: int | None = 30   # 聚类空间：30=PCA(现状)；None=完整 768 维（--cluster-space full）
CACHE = ROOT / "outputs" / "figures" / "bolt_main_figure" / ".cache" / \
    "extract_train_wpd200_val150_ctx128_seed47_layers0-11.pkl"
OUT_DIR = ROOT / "figure_projects" / "pub_main_figure"

# 出版字号
FS_CARD = 14
FS_AXIS = 13
FS_CBAR = 12
FS_LEG = 13
FS_LEGTITLE = 14

# 与 a/b/d 同家族的克制 diverging heatmap（muted blue–warm white–brick red），
# 红端 = panel d 的 ACCENT_RED 同族，比默认 RdBu_r 更"高级"、和 seaborn-deep 调性一致。
MUTED_DIV = LinearSegmentedColormap.from_list(
    "muted_div", ["#355C8A", "#88A6C4", "#F3F1EC", "#CD8F7E", "#B23B3A"])


def build_cards_data():
    if not CACHE.exists():
        raise SystemExit(f"extraction cache not found: {CACHE}\n先跑 build_bolt_main_figure.py 生成缓存。")
    with open(CACHE, "rb") as fh:
        b = pickle.load(fh)
    disc_z, disc_meta, disc_reps, patch_len = b["disc_z"], b["disc_meta"], b["disc_reps"], b["patch_len"]
    emb, raw, meta = flatten_patches(disc_reps[f"layer_{LAYER}"], disc_z, disc_meta, patch_len)
    # 近平直 patch 筛除（raw 与层无关，结果同 canonical）
    keep = np.array([float(np.std(raw[i])) >= MIN_PATCH_STD for i in range(len(raw))])
    emb, raw, meta = emb[keep], raw[keep], [meta[i] for i in range(len(meta)) if keep[i]]
    sel = select_domain_balanced_indices(meta, max_per_domain=MAX_PER_DOMAIN, seed=SEED)
    emb_b, raw_b = emb[sel], raw[sel]
    meta_b = [meta[i] for i in sel]
    labels, centers, pca_coords, _sc, _pca, _km = cluster_pca_fit(emb_b, K, SEED, PCA_DIM)
    return labels, centers, pca_coords, raw_b, meta_b, patch_len


def main() -> None:
    global K, PCA_DIM, OUT_DIR
    ap = argparse.ArgumentParser(description="panel c cluster cards（默认 PCA(30) 聚类）")
    ap.add_argument("--cluster-space", choices=["pca", "full"], default="pca",
                    help="pca=StandardScaler→PCA(30)→KMeans（现状）；full=直接在 768 维上 KMeans")
    ap.add_argument("--k", type=int, default=K, help="KMeans 簇数（full 版建议用 k-sweep 选出的 6）")
    args = ap.parse_args()
    K = args.k
    PCA_DIM = None if args.cluster_space == "full" else 30
    OUT_DIR = (ROOT / "figure_projects" / "pub_main_figure_fullrep" if args.cluster_space == "full"
               else ROOT / "figure_projects" / "pub_main_figure")
    print(f"[panel c] cluster-space={args.cluster_space} (pca_dim={PCA_DIM}) k={K} -> {OUT_DIR.name}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
    })
    labels, centers, pca_coords, raw_patches, meta, patch_len = build_cards_data()

    # 2 行网格（row-major：C1..C3 上、C4..C6 下），版式更方正，便于和较宽的 panel d 同排放。
    NROWS = 2
    NCOLS = (K + NROWS - 1) // NROWS
    fig = plt.figure(figsize=(2.4 * NCOLS + 0.9, 2.75 * NROWS + 0.9))
    inv = fig.transFigure.inverted()

    # —— legend 放全图最上方；先量出它（含加粗标题）的真实高度，再把卡片网格 top 精确放到其正下方：
    #    既不被 legend 盖住（重叠），也不留大片空白 ——
    present = [d for d in DOMAIN_COLORS if any(m["macro_domain"] == d for m in meta)]
    handles = [Line2D([0], [0], marker="s", ls="", ms=9, color=DOMAIN_COLORS[d], label=d) for d in present]
    LEG_TOP = 0.985
    leg_kw = dict(handles=handles, loc="upper center", ncol=(len(handles) + 1) // 2, fontsize=FS_LEG,
                  title="Domain composition (per cluster)", title_fontsize=FS_LEGTITLE,
                  columnspacing=1.5, handletextpad=0.5, frameon=True, fancybox=False,
                  edgecolor="0.7", facecolor="white", framealpha=1.0, borderpad=0.7)
    _tmp = fig.legend(bbox_to_anchor=(0.5, LEG_TOP), **leg_kw)
    _tmp.get_title().set_fontweight("bold")
    fig.canvas.draw()
    leg_h = _tmp.get_window_extent().transformed(inv).height
    _tmp.remove()
    top = LEG_TOP - leg_h - 0.055          # 0.055 = 小间隙 + C1 卡片标题余量
    outer = fig.add_gridspec(NROWS, 1, hspace=0.34, left=0.075, right=0.895, top=top, bottom=0.10)

    # 共享 vmax，使所有卡同色标
    per_cluster, all_abs = [], []
    for cid in range(K):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            per_cluster.append(None)
            continue
        dist = np.linalg.norm(pca_coords[idx] - centers[cid], axis=1)
        order = np.argsort(dist)[: min(TOP_N, len(idx))]
        z = np.stack([z_normalize(raw_patches[i]) for i in idx[order]])
        per_cluster.append((idx, z))
        all_abs.append(np.abs(z).ravel())
    vmax = float(np.percentile(np.concatenate(all_abs), 97)) if all_abs else 2.5

    xt = [0, patch_len // 2 - 1, patch_len - 1]
    im = None
    bottom_axes, heat_axes = [], []     # 底部行 heatmap（放 "Time"）+ 全部 heatmap（对齐 colorbar）
    for r in range(NROWS):
        # 每个卡片行内：thin domain bar + heatmap，行内小间距；行间由 outer.hspace 控制
        inner = outer[r, 0].subgridspec(2, NCOLS, height_ratios=[0.10, 1.0], hspace=0.06, wspace=0.16)
        for col in range(NCOLS):
            cid = r * NCOLS + col
            bar_ax = fig.add_subplot(inner[0, col])
            ax = fig.add_subplot(inner[1, col])
            if cid >= K or per_cluster[cid] is None:
                bar_ax.axis("off"); ax.axis("off")
                continue
            idx, z = per_cluster[cid]
            im = ax.imshow(z, aspect="auto", interpolation="nearest", cmap=MUTED_DIV, vmin=-vmax, vmax=vmax)
            ax.set_yticks([]); heat_axes.append(ax)
            # x 轴含义相同（patch 内时间步）：只在最底行标刻度 + 整行底部写一次 "Time"
            if r == NROWS - 1:
                ax.set_xticks(xt)
                ax.tick_params(labelsize=FS_AXIS - 3, length=2)
                bottom_axes.append(ax)
            else:
                ax.set_xticks([])
            if col == 0:
                ax.set_ylabel("Patches (near → far)", fontsize=FS_AXIS)
            for sp in ax.spines.values():
                sp.set_color("#9aa1ab"); sp.set_linewidth(0.6)   # 轻描边，去掉原来偏重的近黑边

            comp = Counter(meta[i]["macro_domain"] for i in idx)
            total = sum(comp.values())
            left = 0.0
            for dom, cnt in sorted(comp.items(), key=lambda kv: -kv[1]):
                bar_ax.barh(0, cnt / total, left=left, height=1.0,
                            color=DOMAIN_COLORS.get(dom, DOMAIN_COLORS["Other"]), edgecolor="white", lw=0.3)
                left += cnt / total
            bar_ax.set_xlim(0, 1); bar_ax.set_ylim(-0.5, 0.5); bar_ax.axis("off")
            bar_ax.set_title(f"C{cid + 1}", fontsize=FS_CARD, pad=3)

    if im is not None:
        cb_y0 = min(a.get_position().y0 for a in heat_axes)   # colorbar 对齐 heatmap 块
        cb_y1 = max(a.get_position().y1 for a in heat_axes)
        cax = fig.add_axes((0.915, cb_y0, 0.011, cb_y1 - cb_y0))
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Z-norm. value (σ)", fontsize=FS_CBAR)
        cbar.ax.tick_params(labelsize=FS_CBAR - 2)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    # 单个 "Time"：居中于底部卡片行下方、紧贴刻度数字下沿，替代每张卡重复的 x 标签
    card_xc = 0.5 * (min(a.get_position().x0 for a in bottom_axes)
                     + max(a.get_position().x1 for a in bottom_axes)) if bottom_axes else 0.5
    card_bottom = (min(a.get_tightbbox(rend).transformed(inv).y0 for a in bottom_axes)
                   if bottom_axes else 0.2)
    fig.text(card_xc, card_bottom - 0.012, "Time", ha="center", va="top", fontsize=FS_AXIS)

    # 真正的 legend：居中于"全图"内容真实中心（含左 ylabel 与右 colorbar），顶端贴 LEG_TOP
    xs0, xs1 = [], []
    for a in fig.axes:
        tb = a.get_tightbbox(rend).transformed(inv)
        xs0.append(tb.x0); xs1.append(tb.x1)
    xc = 0.5 * (min(xs0) + max(xs1))
    leg = fig.legend(bbox_to_anchor=(xc, LEG_TOP), **leg_kw)
    leg.get_frame().set_linewidth(0.6)
    leg.get_title().set_fontweight("bold")   # 标题加粗，放最上方便于阅读
    leg.set_alignment("center")
    leg.get_title().set_horizontalalignment("center")   # SVG 也居中（text-anchor:middle，换字体不漂移）

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / "panel_b_cards_layer12.svg"  # figure panel b
    png = OUT_DIR / "panel_b_cards_layer12.png"
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[panel c] saved -> {svg}")
    print(f"[panel c] preview -> {png}")


if __name__ == "__main__":
    main()
