"""Publication panel d — cross-domain retrieval（held-out test patch 当 query，SVG，OOD 风格）。

故事（与 panel c 互补，放在同一排、占地更宽）：
  挑几个**代表性的 held-out（unseen）test patch** 当 query，在 representation space 里检索**训练
  数据**中最近的 patch；同一个 query 会把来自**多个不同 domain** 的训练 patch 拉到一起 → 说明
  Chronos-Bolt 学到的是 domain-agnostic、可复用的 temporal primitive，而且能泛化到没见过的数据。

版式参考 OOD 示意图，每行一个 query：
    [held-out query（红）]  →  [最近训练 patch · domain A]  [· domain B]  [· domain C]  [· domain D]
行首是该 primitive 的形状描述名（SVG 里可直接改字）；每个检索结果上方按其 macro domain 配色标注。

配色与 panel a/b/c 统一：domain 用 seaborn-deep（DOMAIN_COLORS），query 红用克制的砖红，
检索曲线用 ink，避免 flat-UI 的高饱和/粉底（子刊审美）。纯 CPU（读缓存，不加载模型）。

从仓库根目录运行：
    .venv/bin/python scripts/pub_main_figure_panel_d.py                       # PCA(30) 聚类（现状）
    .venv/bin/python scripts/pub_main_figure_panel_d.py --cluster-space full --k 6
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_bolt_main_figure import (  # noqa: E402
    DOMAIN_COLORS, VALIDATION_EXCLUDE, _zcorr, cluster_pca_fit, flatten_patches,
    select_domain_balanced_indices, z_normalize,
)

LAYER = 11           # encoder block 11 -> 显示 "Layer 12"
K = 8
SEED = 47
MIN_PATCH_STD = 0.15
MAX_PER_DOMAIN = 400
PCA_DIM: int | None = 30   # 聚类空间：30=PCA(现状)；None=完整 768 维（--cluster-space full）
N_QUERIES = 5              # 画几个代表性 held-out query（不必每簇一个）
M_DOMAINS = 4             # 每个 query 检索回的、来自不同 domain 的最近训练 patch 数
CACHE = ROOT / "outputs" / "figures" / "bolt_main_figure" / ".cache" / \
    "extract_train_wpd200_val150_ctx128_seed47_layers0-11.pkl"
OUT_DIR = ROOT / "figure_projects" / "pub_main_figure"

# 统一配色（与 a/b/c 同家族）：克制的砖红 + ink + 中性灰箭头，杜绝 flat-UI 粉底
ACCENT_RED = "#B5403F"    # held-out query（克制砖红，子刊审美；非任何 domain 色）
INK = "#2b3038"           # 检索结果曲线
ARROW_C = "#5b6470"       # 箭头
QBG = "#f6f5f2"           # query 单元浅中性底（非粉色）

# primitive 的形状描述名（仅描述 prototype 形状，不是 validated motif）。键 = C 序号（1-based）。
SHAPE_NAMES: dict[int, str] = {
    1: "Gradual rise", 2: "Central peak", 3: "Steep rise",
    4: "Irregular fluctuation", 5: "Gradual fall", 6: "Rise, then dip",
}

# 出版字号
FS_NAME = 15
FS_SUB = 11
FS_HEAD = 14


def _spark(ax, y, line_color, fill_color, lw=2.2, fill_alpha=0.20) -> None:
    """填充式 sparkline：曲线撑满单元（tight ylim）+ 曲线下浅色面积，避免"细线浮在大白框里"的空旷感。"""
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    lo, hi = float(np.min(y)), float(np.max(y))
    pad = 0.12 * (hi - lo + 1e-9)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, len(y) - 1)
    ax.fill_between(x, y, lo - pad, color=fill_color, alpha=fill_alpha, linewidth=0)
    ax.plot(x, y, color=line_color, lw=lw, solid_capstyle="round")


def _auto_name(proto: np.ndarray) -> str:
    """k != 6 时给 prototype 一个保守的形状描述名（z-normalized 上）。"""
    d = np.diff(proto)
    zc = int(np.sum(np.diff(np.sign(d)) != 0))
    slope = float(proto[-1] - proto[0])
    if zc >= 6:
        return "Irregular fluctuation"
    amax, amin, n = int(np.argmax(proto)), int(np.argmin(proto)), len(proto)
    if zc >= 3 and 2 < amax < n - 3 and proto[amax] >= proto.max() - 1e-6:
        return "Central peak"
    if zc >= 3 and 2 < amin < n - 3 and proto[amin] <= proto.min() + 1e-6:
        return "Central dip"
    if slope > 0.4:
        return "Rising trend"
    if slope < -0.4:
        return "Falling trend"
    return "Level / flat"


def build_panel_d_data():
    """返回 (proto, labels, coords_tr, raw_tr, meta_tr, Xv, zval, vm, rep_assign, rep_dist)。

    labels = 训练 balanced 子集的簇标签；coords_tr/Xv 都在同一 discovery 聚类空间里
    （full 版=标准化后的 768 维；pca 版=PCA(30)）。
    """
    if not CACHE.exists():
        raise SystemExit(f"extraction cache not found: {CACHE}\n先跑 build_bolt_main_figure.py 生成缓存。")
    with open(CACHE, "rb") as fh:
        b = pickle.load(fh)
    disc_z, disc_meta, disc_reps = b["disc_z"], b["disc_meta"], b["disc_reps"]
    val_z, val_meta_w, val_reps, patch_len = b["val_z"], b["val_meta_w"], b["val_reps"], b["patch_len"]

    # validation 再过一遍 exclude（剔除 Chronos 预训练内 / 与训练子集重叠，缓存可能用旧 exclude 采的）
    vkeep = [i for i, m in enumerate(val_meta_w) if m["dataset"] not in VALIDATION_EXCLUDE]
    val_z = val_z[vkeep]
    val_meta_w = [val_meta_w[i] for i in vkeep]
    val_reps = {kk: vv[vkeep] for kk, vv in val_reps.items()}

    de, dr, dm = flatten_patches(disc_reps[f"layer_{LAYER}"], disc_z, disc_meta, patch_len)
    ve, vr, vm = flatten_patches(val_reps[f"layer_{LAYER}"], val_z, val_meta_w, patch_len)
    kd = np.array([float(np.std(dr[i])) >= MIN_PATCH_STD for i in range(len(dr))])
    de, dr, dm = de[kd], dr[kd], [dm[i] for i in range(len(dm)) if kd[i]]
    kv = np.array([float(np.std(vr[i])) >= MIN_PATCH_STD for i in range(len(vr))])
    ve, vr, vm = ve[kv], vr[kv], [vm[i] for i in range(len(vm)) if kv[i]]

    sel = select_domain_balanced_indices(dm, max_per_domain=MAX_PER_DOMAIN, seed=SEED)
    emb_b, raw_tr = de[sel], dr[sel]
    meta_tr = [dm[i] for i in sel]
    labels, centers, coords_tr, scaler, pca, _km = cluster_pca_fit(emb_b, K, SEED, PCA_DIM)

    proto = np.zeros((K, raw_tr.shape[1]), dtype=np.float64)
    for c in range(K):
        idx = np.where(labels == c)[0]
        if len(idx):
            proto[c] = np.mean(np.stack([z_normalize(raw_tr[i]) for i in idx]), axis=0)

    Xv = scaler.transform(ve) if pca is None else pca.transform(scaler.transform(ve))
    d_rep = np.linalg.norm(Xv[:, None, :] - centers[None, :, :], axis=2)
    rep_assign = d_rep.argmin(axis=1)
    rep_dist = d_rep.min(axis=1)
    zval = np.stack([z_normalize(vr[i]) for i in range(len(vr))])
    return proto, labels, coords_tr, raw_tr, meta_tr, Xv, zval, vm, rep_assign, rep_dist


def main() -> None:
    global K, PCA_DIM, OUT_DIR
    ap = argparse.ArgumentParser(description="panel d cross-domain retrieval（held-out query → 训练 patch）")
    ap.add_argument("--cluster-space", choices=["pca", "full"], default="pca",
                    help="pca=StandardScaler→PCA(30)→KMeans（现状）；full=直接在 768 维上 KMeans")
    ap.add_argument("--k", type=int, default=K, help="KMeans 簇数（full 版建议用 k-sweep 选出的 6）")
    ap.add_argument("--n-queries", type=int, default=N_QUERIES, help="画几个代表性 held-out query 行")
    args = ap.parse_args()
    K = args.k
    PCA_DIM = None if args.cluster_space == "full" else 30
    OUT_DIR = (ROOT / "figure_projects" / "pub_main_figure_fullrep" if args.cluster_space == "full"
               else ROOT / "figure_projects" / "pub_main_figure")
    nq = min(args.n_queries, K)
    print(f"[panel d] cluster-space={args.cluster_space} (pca_dim={PCA_DIM}) k={K} n_queries={nq} -> {OUT_DIR.name}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
    })
    proto, labels, coords_tr, raw_tr, meta_tr, Xv, zval, vm, rep_assign, rep_dist = build_panel_d_data()

    # 每簇的代表性 held-out query = 落在该簇、形状最像 prototype 的 held-out patch（让 query 一眼可读为该
    # primitive）；再按这个 coherence 取最干净的 nq 个簇（自然丢掉最"杂"的那个）。
    cohorder: dict[int, np.ndarray] = {}   # 每簇 held-out patch 按"像不像该 prototype"降序
    cands: dict[int, tuple[int, float]] = {}
    for c in range(K):
        a = np.where(rep_assign == c)[0]
        if len(a) == 0:
            continue
        cohs = np.array([_zcorr(zval[i], proto[c]) for i in a])
        srt = a[np.argsort(-cohs)]
        cohorder[c] = srt
        cands[c] = (int(srt[0]), float(cohs.max()))
    chosen = sorted(sorted(cands, key=lambda c: -cands[c][1])[:nq])  # coherence 最高的 nq 个，按 C 序排

    # query 数据集去重：尽量让每行 query 来自**不同** held-out 数据集（更能体现"多个 unseen 样本"），
    # 在各簇 coherence 降序里取第一个未用过的数据集；都用过则退回最像的那个。
    used_ds: set[str] = set()
    query_idx: dict[int, int] = {}
    for c in chosen:
        pick = next((int(i) for i in cohorder[c] if vm[int(i)]["dataset"] not in used_ds), None)
        if pick is None:
            pick = int(cohorder[c][0])
        used_ds.add(vm[pick]["dataset"])
        query_idx[c] = pick
    names = {c: SHAPE_NAMES.get(c + 1, _auto_name(proto[c])) for c in chosen}

    # 检索：限定在 query 所属簇（同一 primitive）的训练 patch 内，按 rep-space 距离取**不同 domain**的最近者。
    # 限定同簇 → 保证检索结果是同一 primitive 的跨域实例（而非全局最近的杂样本）。合成域不计入"真实跨域复用"。
    SKIP_DOMAINS = {"Synthetic", "Synthetic control", "Other"}
    retr: dict[int, list[tuple[int, str]]] = {}
    for c in chosen:
        qi = query_idx[c]
        members = np.where(labels == c)[0]
        order = members[np.argsort(np.linalg.norm(coords_tr[members] - Xv[qi], axis=1))]
        picked: list[tuple[int, str]] = []
        seen: set[str] = set()
        for ti in order:
            dom = meta_tr[int(ti)]["macro_domain"]
            if dom in seen or dom in SKIP_DOMAINS:
                continue
            seen.add(dom)
            picked.append((int(ti), dom))
            if len(picked) >= M_DOMAINS:
                break
        retr[c] = picked

    nrow = len(chosen)
    ncol = 2 + M_DOMAINS                              # query | arrow | M domain 检索
    fig_w = 1.55 * (1.65 + M_DOMAINS) + 0.9
    fig_h = 1.32 * nrow + 0.7
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrow, ncol, width_ratios=[1.25, 0.32] + [1.0] * M_DOMAINS,
                          hspace=0.42, wspace=0.13,
                          left=0.135, right=0.99, top=0.90, bottom=0.085)

    for r, c in enumerate(chosen):
        qi = query_idx[c]
        # --- col0: held-out query（克制砖红，浅中性底）---
        ax = fig.add_subplot(gs[r, 0])
        _spark(ax, zval[qi], ACCENT_RED, ACCENT_RED, lw=2.6, fill_alpha=0.14)
        ax.set_facecolor(QBG)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#d9c2bf"); sp.set_linewidth(0.9)
        ax.text(-0.07, 0.66, names[c], transform=ax.transAxes, ha="right", va="center",
                fontsize=FS_NAME, fontweight="bold", color="#1f2933", clip_on=False)
        # 两行右对齐：小号浅灰 eyebrow "Unseen dataset" + 稍大深灰数据集名（右边缘对齐上方 shape 名）
        ax.text(-0.07, 0.40, "Unseen dataset", transform=ax.transAxes, ha="right", va="center",
                fontsize=FS_SUB - 2, color="#b0b5bd", clip_on=False)
        ax.text(-0.07, 0.20, vm[qi]['dataset'], transform=ax.transAxes, ha="right", va="center",
                fontsize=FS_SUB + 1, color="#6b7280", fontweight="medium", clip_on=False)
        if r == 0:
            ax.set_title("Unseen test patch", fontsize=FS_HEAD, color=ACCENT_RED, fontweight="bold", pad=9)

        # --- col1: 箭头 ---
        axa = fig.add_subplot(gs[r, 1]); axa.axis("off")
        axa.annotate("", xy=(0.95, 0.5), xytext=(0.02, 0.5), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", lw=2.0, color=ARROW_C))

        # --- domain 检索列：ink 曲线 + 各自 domain 配色标题 ---
        picked = retr[c]
        for j in range(M_DOMAINS):
            ax = fig.add_subplot(gs[r, 2 + j])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#cfd4dc"); sp.set_linewidth(0.8)
            if j >= len(picked):
                ax.set_facecolor("#fbfbfc")
                continue
            ti, dom = picked[j]
            _spark(ax, z_normalize(raw_tr[ti]), INK, DOMAIN_COLORS.get(dom, INK),
                   lw=2.2, fill_alpha=0.22)
            ax.set_title(dom, fontsize=FS_HEAD - 1, color=DOMAIN_COLORS.get(dom, INK),
                         fontweight="bold", pad=6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / "panel_c_retrieval.svg"  # figure panel c
    png = OUT_DIR / "panel_c_retrieval.png"
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[panel d] queries(C)={[c + 1 for c in chosen]}  domains/row="
          f"{[[d for _, d in retr[c]] for c in chosen]}")
    print(f"[panel d] saved -> {svg}")
    print(f"[panel d] preview -> {png}")


if __name__ == "__main__":
    main()
