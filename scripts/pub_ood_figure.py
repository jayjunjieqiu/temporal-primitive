"""Publication OOD-transfer figure（fullrep 版，SVG，可编辑文字）。

把 run_bolt_ood_transfer.py 的研究图重画成出版级、模块化 SVG，配色/字号与 main figure 统一。
故事（主线 3 的深化 = transferable generalization）：离 pretraining 分布最远（OOD）的 held-out
test patch，依然能被预训练发现的 model-derived pattern group 词表解释。

四个独立 panel（用户在 PPT 里左 a/b/c、右 d 拼）：
  a  panel_a_overlap      Layer 1 / Layer 12：训练(灰) vs held-out(按 OOD 着色) 的 t-SNE 重合
  b  panel_b_attribution  最 OOD 的 held-out patch 的「最近训练 pattern group(cluster)」+「最近训练 domain」环图
  c  panel_c_casestudy    最 OOD 的 held-out patch → 各自最近的训练 pattern group（filled sparkline）
  d  panel_d_ranking      各 held-out 数据集按 OOD 程度排名（Layer 1 vs Layer 12，全幅底部条带）

聚类/OOD 空间：--cluster-space full = 标准化后的完整 768 维（k 默认随 main fullrep 用 6）；
pca = StandardScaler→PCA(30)（现状）。OOD 分 = kNN(test→train) / 典型 train-train 近邻距离，越大越 OOD。
t-SNE 仅用于可视化（先 PCA(50) 提速），坐标缓存到 .ood_tsne_*.npz（gitignored）。纯 CPU（读 main 缓存）。

从仓库根目录运行：
    .venv/bin/python scripts/pub_ood_figure.py --cluster-space full --k 6
"""
from __future__ import annotations

import argparse
import glob
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_bolt_main_figure import (  # noqa: E402
    DOMAIN_COLORS, VALIDATION_EXCLUDE, _zcorr, cluster_pca_fit, flatten_patches,
    select_domain_balanced_indices, z_normalize,
)

CACHE_GLOB = str(ROOT / "outputs/figures/bolt_main_figure/.cache/extract_train_*.pkl")
CONTROL = {"Gaussian", "Pulse"}     # 合成 negative control：天然 OOD

# 统一配色（与 a/b/c/d main figure 同家族）
ACCENT_RED = "#B5403F"              # OOD test patch（克制砖红）
INK = "#2b3038"
TRAIN_GREY = "#c4c8ce"
L1_C, L12_C = "#E8C4A0", "#B5403F"  # 深度配色：暖色浅→深梯度（浅暖褐=Layer 1 → 砖红=Layer 12），
#                                     落在 OOD colormap 同一条暖轴上，与 panel a/b/c 暖调统一；深=更 OOD 语义自洽
# seaborn-deep（muted）调色板——cluster 大色块用它，和主图 domain 同调性，避免 tab10 满饱和显鲜艳
SNS_DEEP = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
            "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
# OOD 顺序色：浅琥珀 → 橙 → 砖红（红端与 main figure 同族），比 autumn_r 更克制
# OOD colormap：muted 暖色梯度（浅暖灰 → seaborn 橙 → 砖红），高端 = panel c test patch 同款 ACCENT_RED，
# 去掉原来跳脱的亮黄，让 panel a 回到和 b/c/d 一致的 seaborn-deep 克制调性
OOD_CMAP = LinearSegmentedColormap.from_list("ood", ["#EFE2D2", "#DD8452", "#B5403F"])

# 出版字号
FS_TITLE = 17
FS_LABEL = 15
FS_TICK = 12
FS_LEG = 12
FS_SMALL = 11


def _spark(ax, y, line_color, fill_color, lw=2.0, fill_alpha=0.20) -> None:
    """填充式 sparkline：曲线撑满单元（tight ylim）+ 浅色面积，避免细线浮在大白框里。"""
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    lo, hi = float(np.min(y)), float(np.max(y))
    pad = 0.12 * (hi - lo + 1e-9)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, len(y) - 1)
    ax.fill_between(x, y, lo - pad, color=fill_color, alpha=fill_alpha, linewidth=0)
    ax.plot(x, y, color=line_color, lw=lw, solid_capstyle="round")


def fit_layer(bundle, L, pca_dim, k, seed, min_std=0.15, max_per_domain=400, kk=10):
    """某层：flat-filter + domain-balanced 聚类 discovery，投影 validation，算 OOD 分。

    pca_dim=None → 在标准化后的完整 768 维上聚类/算距离（fullrep）；=30 → PCA(30)（现状）。
    """
    pl = bundle["patch_len"]
    d_emb, d_raw, d_meta = flatten_patches(bundle["disc_reps"][f"layer_{L}"], bundle["disc_z"], bundle["disc_meta"], pl)
    v_emb, v_raw, v_meta = flatten_patches(bundle["val_reps"][f"layer_{L}"], bundle["val_z"], bundle["val_meta_w"], pl)

    def nonflat(raw):
        return np.array([float(np.std(raw[i])) >= min_std for i in range(len(raw))])
    dk, vk = nonflat(d_raw), nonflat(v_raw)
    d_emb, d_raw, d_meta = d_emb[dk], d_raw[dk], [d_meta[i] for i in range(len(d_meta)) if dk[i]]
    v_emb, v_raw, v_meta = v_emb[vk], v_raw[vk], [v_meta[i] for i in range(len(v_meta)) if vk[i]]

    sel = select_domain_balanced_indices(d_meta, max_per_domain=max_per_domain, seed=seed)
    emb_b, raw_b = d_emb[sel], d_raw[sel]
    meta_b = [d_meta[i] for i in sel]
    labels, centers, train_coords, scaler, pca, _ = cluster_pca_fit(emb_b, k, seed, pca_dim)
    val_coords = scaler.transform(v_emb) if pca is None else pca.transform(scaler.transform(v_emb))

    nn = NearestNeighbors(n_neighbors=kk + 1).fit(train_coords)
    dtt, _ = nn.kneighbors(train_coords)
    base = float(np.median(dtt[:, 1:kk + 1].mean(axis=1)))
    dvt, _ = nn.kneighbors(val_coords)
    ood = dvt[:, :kk].mean(axis=1) / base

    proto = np.zeros((k, raw_b.shape[1]), dtype=np.float64)
    for c in range(k):
        m = labels == c
        if m.any():
            proto[c] = np.mean(np.stack([z_normalize(raw_b[i]) for i in np.where(m)[0]]), axis=0)
    return {"labels": labels, "centers": centers, "train_coords": train_coords, "raw_b": raw_b,
            "meta_b": meta_b, "proto": proto, "val_coords": val_coords, "val_raw": v_raw,
            "val_meta": v_meta, "ood": ood}


def per_dataset_ood(fit):
    ds = np.array([m["dataset"] for m in fit["val_meta"]])
    return {d: float(fit["ood"][ds == d].mean()) for d in sorted(set(ds.tolist()))}


def _tsne_xy(fit, seed, perplexity, cache_path, max_each=2500):
    """train+test 子样本的 2D t-SNE（先 PCA(50) 提速；坐标缓存）。返回 (xy, n_train, vi)。"""
    rng = np.random.default_rng(seed)
    tc, vc = fit["train_coords"], fit["val_coords"]
    ti = rng.choice(len(tc), min(max_each, len(tc)), replace=False)
    vi = rng.choice(len(vc), min(max_each, len(vc)), replace=False)
    if cache_path.exists():
        z = np.load(cache_path)
        return z["xy"], int(z["n_train"]), z["vi"]
    X = np.vstack([tc[ti], vc[vi]])
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=seed).fit_transform(X)
    xy = TSNE(2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(X)
    np.savez(cache_path, xy=xy, n_train=len(ti), vi=vi)
    return xy, len(ti), vi


# ---------------------------------------------------------------- panel a：overlap
def render_overlap(fits, layer_Ls, seed, perplexity, out_dir, cspace):
    plt.rcParams.update(_RC)
    vmin, vmax = 0.8, float(np.percentile(np.concatenate([fits[L]["ood"] for L in layer_Ls]), 96))
    fig = plt.figure(figsize=(8.0, 4.2))
    gs = fig.add_gridspec(1, len(layer_Ls), left=0.04, right=0.88, top=0.90, bottom=0.05, wspace=0.08)
    sc = None
    for col, L in enumerate(layer_Ls):
        xy, n_t, vi = _tsne_xy(fits[L], seed, perplexity, out_dir / f".ood_tsne_{cspace}_L{L}.npz")
        ax = fig.add_subplot(gs[0, col])
        ax.scatter(xy[:n_t, 0], xy[:n_t, 1], s=7, color=TRAIN_GREY, alpha=0.55, linewidths=0)
        sc = ax.scatter(xy[n_t:, 0], xy[n_t:, 1], s=9, c=np.clip(fits[L]["ood"][vi], vmin, vmax),
                        cmap=OOD_CMAP, alpha=0.85, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Layer {L + 1}", fontsize=FS_TITLE, fontweight="bold", pad=8)
        for spn in ax.spines.values():
            spn.set_color("#cfd4dc")
    cax = fig.add_axes((0.915, 0.20, 0.013, 0.55))
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("OOD score", fontsize=FS_LABEL)
    cb.ax.tick_params(labelsize=FS_TICK - 1)
    # 不放 legend（grey=training / 着色=test held-out 由 caption 说明）
    _save(fig, out_dir, "panel_a_overlap")


# ---------------------------------------------------------------- panel b：attribution donuts
def render_attribution(fit, k, ood_quantile, coh_thresh, out_dir):
    plt.rcParams.update(_RC)
    vm = fit["val_meta"]
    real = np.array([i for i, m in enumerate(vm) if m["dataset"] not in CONTROL])
    thr = np.quantile(fit["ood"][real], ood_quantile)
    ood_idx = real[fit["ood"][real] >= thr]
    nn1 = NearestNeighbors(n_neighbors=1).fit(fit["train_coords"])
    _, nbr = nn1.kneighbors(fit["val_coords"][ood_idx])
    nbr = nbr[:, 0]
    clu = [int(fit["labels"][t]) for t in nbr]
    dom = [fit["meta_b"][t]["macro_domain"] for t in nbr]
    coh = float(np.mean([_zcorr(z_normalize(fit["val_raw"][ood_idx[i]]), fit["proto"][clu[i]]) >= coh_thresh
                         for i in range(len(ood_idx))]))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4))
    # 子刊审美：donut（muted seaborn-deep 配色），类别名放外圈、百分比放环内白字（避免双行外标签拥挤重叠）

    def _donut(ax, counts, order, colors, names, title):
        pairs = sorted(zip(order, names, colors), key=lambda t: -counts[t[0]])  # 降序：大扇区聚拢、相邻小扇区更分散
        sizes = [counts[k] for k, _, _ in pairs]
        lbls = [nm for _, nm, _ in pairs]
        cols = [c for _, _, c in pairs]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=lbls, colors=cols, startangle=90, counterclock=False,
            wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.8),
            autopct=lambda p: f"{round(p)}%" if p >= 4 else "",  # 极小扇区不放内部 %（仅外圈留名称）
            pctdistance=0.78, labeldistance=1.08,
            textprops=dict(fontsize=FS_SMALL, color=INK))
        for t in autotexts:
            t.set_color("white"); t.set_fontweight("bold"); t.set_fontsize(FS_SMALL)
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=14)

    cc = Counter(clu); corder = sorted(cc)
    _donut(axes[0], cc, corder, [SNS_DEEP[c % 10] for c in corder],
           [f"C{c + 1}" for c in corder], "Nearest training group")
    dc = Counter(dom); dorder = [d for d in DOMAIN_COLORS if d in dc]
    _donut(axes[1], dc, dorder, [DOMAIN_COLORS[d] for d in dorder], dorder, "Nearest training domain")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.04, wspace=0.30)
    _save(fig, out_dir, "panel_b_attribution")
    return {"n_ood_patches": int(len(ood_idx)), "coverage": round(coh, 3)}


# ---------------------------------------------------------------- panel c：ranking
def render_ranking(ood_by_layer, out_dir):
    plt.rcParams.update(_RC)
    lns = list(ood_by_layer)
    datasets = sorted(set().union(*[set(d) for d in ood_by_layer.values()]),
                      key=lambda d: -ood_by_layer[lns[0]].get(d, 0))
    # 竖直分组柱（数据集在 x 轴）→ 全幅底部条带；按 Layer 1 OOD 降序
    x = np.arange(len(datasets))
    w = 0.40
    fig, ax = plt.subplots(figsize=(0.82 * len(datasets) + 3.0, 4.0))
    fig.subplots_adjust(left=0.045, right=0.997, bottom=0.30, top=0.94)
    for li, ln in enumerate(lns):
        ax.bar(x + (li - 0.5) * w, [ood_by_layer[ln].get(d, 0) for d in datasets], width=w,
               color=(L1_C, L12_C)[li % 2], label=ln.replace("layer", "Layer"),
               edgecolor="white", linewidth=0.6, zorder=3)
    # 合成 control 只用 * 标注（含义由 caption/docs 说明）
    xlabels = [f"{d} *" if d in CONTROL else d for d in datasets]
    ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=FS_TICK)
    ax.set_xlim(-0.7, len(datasets) - 0.3)
    ax.set_ylabel("OOD score", fontsize=FS_LABEL)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.margins(y=0.02)
    # legend 放图下方、紧凑无标题、横排两项；锚在旋转刻度标签下沿之下，避免重合
    ax.legend(fontsize=FS_LEG, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.26),
              ncol=2, handlelength=1.3, handletextpad=0.5, columnspacing=1.8, borderaxespad=0.0)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    for spn in ("top", "right"):
        ax.spines[spn].set_visible(False)
    _save(fig, out_dir, "panel_d_ranking")


# ---------------------------------------------------------------- panel d：case study
def render_casestudy(fit, n_samples, n_proto, out_dir):
    plt.rcParams.update(_RC)
    vm = fit["val_meta"]
    real = [i for i, m in enumerate(vm) if m["dataset"] not in CONTROL]
    real_sorted = sorted(real, key=lambda i: -fit["ood"][i])
    # 取最 OOD 的 patch，但每个数据集最多 cap 个 → case study 跨多个数据集，不至于全是同一个
    cap, chosen, per = 2, [], Counter()
    for i in real_sorted:
        d = vm[i]["dataset"]
        if per[d] >= cap:
            continue
        per[d] += 1
        chosen.append(i)
        if len(chosen) >= n_samples:
            break
    order = np.array(chosen)
    nn3 = NearestNeighbors(n_neighbors=n_proto).fit(fit["train_coords"])
    _, proto_idx = nn3.kneighbors(fit["val_coords"][order])

    ncol = 2 + n_proto                      # OOD patch | arrow | n_proto pattern groups
    # 版式 3：c 移到底部全幅，左列只剩 a+b；d 高度 ≈ a+b 堆叠高度（同一 pt 字号在拼版下视觉一致）
    fig_w = 1.25 * (1.3 + n_proto) + 1.4
    fig_h = 1.0 * n_samples + 0.8
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_samples, ncol, width_ratios=[1.2, 0.3] + [1.0] * n_proto,
                          hspace=0.42, wspace=0.13, left=0.135, right=0.99, top=0.92, bottom=0.03)
    head_x0 = head_x1 = 0.5
    for r, vi in enumerate(order):
        ax = fig.add_subplot(gs[r, 0])
        _spark(ax, z_normalize(fit["val_raw"][vi]), ACCENT_RED, ACCENT_RED, lw=2.4, fill_alpha=0.14)
        ax.set_facecolor("#f6f5f2")
        ax.set_xticks([]); ax.set_yticks([])
        for spn in ax.spines.values():
            spn.set_color("#d9c2bf"); spn.set_linewidth(0.9)
        ax.text(-0.07, 0.64, vm[vi]["dataset"], transform=ax.transAxes, ha="right", va="center",
                fontsize=FS_LABEL - 2, fontweight="bold", color="#1f2933", clip_on=False)
        ax.text(-0.07, 0.34, f"OOD = {fit['ood'][vi]:.2f}", transform=ax.transAxes, ha="right",
                va="center", fontsize=FS_SMALL, color="#8a9099", clip_on=False)
        if r == 0:
            ax.set_title("OOD test patch", fontsize=FS_LABEL, color=ACCENT_RED, fontweight="bold", pad=9)

        axa = fig.add_subplot(gs[r, 1]); axa.axis("off")
        axa.annotate("", xy=(0.95, 0.5), xytext=(0.02, 0.5), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#5b6470"))

        for j in range(n_proto):
            ax = fig.add_subplot(gs[r, 2 + j])
            if r == 0 and j == 0:
                head_x0 = ax.get_position().x0
            if r == 0 and j == n_proto - 1:
                head_x1 = ax.get_position().x1
            ti = int(proto_idx[r, j])
            dom = fit["meta_b"][ti]["macro_domain"]
            _spark(ax, z_normalize(fit["raw_b"][ti]), INK, DOMAIN_COLORS.get(dom, INK),
                   lw=2.0, fill_alpha=0.22)
            ax.set_xticks([]); ax.set_yticks([])
            for spn in ax.spines.values():
                spn.set_color("#cfd4dc"); spn.set_linewidth(0.8)
            ax.set_title(f"C{int(fit['labels'][ti]) + 1} · {dom}", fontsize=FS_SMALL,
                         color=DOMAIN_COLORS.get(dom, INK), fontweight="bold", pad=5)

    # 右侧块顶部统一标头（用循环里记下的行 0 proto 单元位置，避免再 add_subplot 盖出多余坐标轴）
    fig.text(0.5 * (head_x0 + head_x1), 0.965, "→  Nearest training groups", ha="center",
             va="bottom", fontsize=FS_LABEL, fontweight="bold", color="#1f2933")
    _save(fig, out_dir, "panel_c_casestudy")


_RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
       "svg.fonttype": "none", "axes.linewidth": 0.9}


def _save(fig, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[ood] saved -> {out_dir.name}/{stem}.svg")


def main() -> None:
    ap = argparse.ArgumentParser(description="publication OOD-transfer figure（默认 PCA(30)；--cluster-space full = fullrep）")
    ap.add_argument("--cluster-space", choices=["pca", "full"], default="pca")
    ap.add_argument("--k", type=int, default=8, help="KMeans 簇数（fullrep 建议 6）")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 11], help="overlap/ranking 用的层（0-based）")
    ap.add_argument("--primary-layer", type=int, default=11, help="attribution + case study 用哪层（默认 Layer 12）")
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--tsne-perplexity", type=float, default=40.0)
    ap.add_argument("--case-samples", type=int, default=8)
    ap.add_argument("--case-protos", type=int, default=3)
    ap.add_argument("--ood-quantile", type=float, default=0.75)
    ap.add_argument("--coh-thresh", type=float, default=0.6)
    args = ap.parse_args()

    pca_dim = None if args.cluster_space == "full" else 30
    out_dir = (ROOT / "figure_projects" / ("pub_ood_figure_fullrep" if args.cluster_space == "full"
                                           else "pub_ood_figure"))
    out_dir.mkdir(parents=True, exist_ok=True)   # 提前建好：t-SNE 坐标缓存要先写到这里
    print(f"[ood] cluster-space={args.cluster_space} (pca_dim={pca_dim}) k={args.k} -> {out_dir.name}")

    hits = sorted(glob.glob(CACHE_GLOB))
    if not hits:
        raise SystemExit("no extraction cache — run build_bolt_main_figure.py first")
    bundle = pickle.load(open(hits[-1], "rb"))
    print(f"[ood] using cache {Path(hits[-1]).name}")
    # validation 再过一遍 exclude（与 main panels 一致，缓存可能用旧 exclude 采的）
    vkeep = [i for i, m in enumerate(bundle["val_meta_w"]) if m["dataset"] not in VALIDATION_EXCLUDE]
    if len(vkeep) < len(bundle["val_meta_w"]):
        bundle["val_z"] = bundle["val_z"][vkeep]
        bundle["val_meta_w"] = [bundle["val_meta_w"][i] for i in vkeep]
        bundle["val_reps"] = {kk: vv[vkeep] for kk, vv in bundle["val_reps"].items()}

    fits = {L: fit_layer(bundle, L, pca_dim, args.k, args.seed) for L in args.layers}
    ood_by_layer = {f"layer {L + 1}": per_dataset_ood(fits[L]) for L in args.layers}
    pl = args.primary_layer
    if pl not in fits:
        fits[pl] = fit_layer(bundle, pl, pca_dim, args.k, args.seed)

    render_overlap(fits, args.layers, args.seed, args.tsne_perplexity, out_dir, args.cluster_space)
    render_ranking(ood_by_layer, out_dir)
    info = render_attribution(fits[pl], args.k, args.ood_quantile, args.coh_thresh, out_dir)
    render_casestudy(fits[pl], args.case_samples, args.case_protos, out_dir)
    print(f"[ood] primary layer {pl + 1}: OOD patches={info['n_ood_patches']}  coverage={info['coverage']}")


if __name__ == "__main__":
    main()
