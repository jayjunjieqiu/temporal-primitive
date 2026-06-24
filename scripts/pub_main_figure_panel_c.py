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
CACHE = ROOT / "outputs" / "figures" / "bolt_main_figure" / ".cache" / \
    "extract_train_wpd200_val150_ctx128_seed47_layers0-11.pkl"
OUT_DIR = ROOT / "figure_projects" / "pub_main_figure"

# 出版字号
FS_CARD = 14
FS_AXIS = 13
FS_CBAR = 12
FS_LEG = 13
FS_LEGTITLE = 14


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
    labels, centers, pca_coords, _sc, _pca, _km = cluster_pca_fit(emb_b, K, SEED)
    return labels, centers, pca_coords, raw_b, meta_b, patch_len


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
    })
    labels, centers, pca_coords, raw_patches, meta, patch_len = build_cards_data()

    fig = plt.figure(figsize=(2.15 * K, 3.7))
    gs = fig.add_gridspec(2, K, height_ratios=[0.085, 1.0], hspace=0.05, wspace=0.16,
                          top=0.92, bottom=0.26)

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
    first_ax = last_ax = None
    seen_domains: set[str] = set()
    for cid in range(K):
        bar_ax = fig.add_subplot(gs[0, cid])
        ax = fig.add_subplot(gs[1, cid])
        if cid == 0:
            first_ax = ax
        if cid == K - 1:
            last_ax = ax
        if per_cluster[cid] is None:
            bar_ax.axis("off"); ax.axis("off")
            continue
        idx, z = per_cluster[cid]
        im = ax.imshow(z, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(xt)
        ax.set_yticks([])
        ax.tick_params(labelsize=FS_AXIS - 3, length=2)
        # x 轴含义相同（patch 内时间步），整行底部只写一次 "Time"，不在每张卡重复
        if cid == 0:
            ax.set_ylabel("Patches (near → far)", fontsize=FS_AXIS)
        for sp in ax.spines.values():
            sp.set_color("#1f2933"); sp.set_linewidth(0.8)

        comp = Counter(meta[i]["macro_domain"] for i in idx)
        total = sum(comp.values())
        left = 0.0
        for dom, cnt in sorted(comp.items(), key=lambda kv: -kv[1]):
            bar_ax.barh(0, cnt / total, left=left, height=1.0,
                        color=DOMAIN_COLORS.get(dom, DOMAIN_COLORS["Other"]), edgecolor="white", lw=0.3)
            left += cnt / total
            seen_domains.add(dom)
        bar_ax.set_xlim(0, 1); bar_ax.set_ylim(-0.5, 0.5); bar_ax.axis("off")
        bar_ax.set_title(f"C{cid + 1}", fontsize=FS_CARD, pad=3)

    cax = None
    if im is not None:
        cax = fig.add_axes((0.92, 0.30, 0.008, 0.45))
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Z-norm. value (σ)", fontsize=FS_CBAR)
        cbar.ax.tick_params(labelsize=FS_CBAR - 2)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    # 单个 "Time"：居中于卡片块（first→last card）下方、紧贴刻度数字下沿，替代每张卡重复的 x 标签
    card_xc = 0.5 * ((first_ax.get_position().x0 if first_ax is not None else 0.05)
                     + (last_ax.get_position().x1 if last_ax is not None else 0.9))
    card_bottom = first_ax.get_tightbbox(rend).transformed(inv).y0  # 含刻度数字的底沿
    fig.text(card_xc, card_bottom - 0.015, "Time", ha="center", va="top", fontsize=FS_AXIS)

    # Domain legend 居中于"全图"——含左侧 ylabel 与右侧 colorbar 的内容真实中心
    # （图用 bbox_inches=tight 裁剪保存，故内容中心 = 成图的视觉中心）
    xs0, xs1 = [], []
    for a in fig.axes:
        tb = a.get_tightbbox(rend).transformed(inv)
        xs0.append(tb.x0); xs1.append(tb.x1)
    xc = 0.5 * (min(xs0) + max(xs1))
    handles = [Line2D([0], [0], marker="s", ls="", ms=9, color=DOMAIN_COLORS[d], label=d)
               for d in DOMAIN_COLORS if d in seen_domains]
    leg = fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=FS_LEG,
                     bbox_to_anchor=(xc, 0.13), title="Domain composition (per cluster)",
                     title_fontsize=FS_LEGTITLE, columnspacing=1.5, handletextpad=0.5,
                     frameon=True, fancybox=False, edgecolor="0.7", facecolor="white",
                     framealpha=1.0, borderpad=0.7)
    leg.get_frame().set_linewidth(0.6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / "panel_c_cards_layer12.svg"
    png = OUT_DIR / "panel_c_cards_layer12.png"
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[panel c] saved -> {svg}")
    print(f"[panel c] preview -> {png}")


if __name__ == "__main__":
    main()
