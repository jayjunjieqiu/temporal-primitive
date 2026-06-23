"""OOD transfer case study：离 pretraining 分布最远（OOD）的 test patch，也能被预训练发现的
primitive 词表解释 —— "transferable generalization"（主线 3 的深化）。

复用 build_bolt_main_figure.py 的提取缓存（无需 GPU）。在每层 PCA(30) 表征空间里：
  - discovery（训练子集，≈pretraining 分布）做 flat-filter + domain-balanced + KMeans 聚类。
  - validation（basicts held-out，剔除泄漏/预训练内数据集）投到同一空间。
  - OOD 分 = kNN距离(test→train) / 基线(train→train 典型近邻距离)，>1 越大越 OOD（可跨层比）。

产出（outputs/figures/bolt_ood_transfer/）：
  G  ood_overlap_layer{1,12}.png    训练(灰) vs 测试(按 OOD 着色) t-SNE 重合
  H  ood_ranking.png                每个 test 数据集 OOD 分排名（两层）
  I  ood_case_study.png             最 OOD 的真实样本 + 各自 3 个最近 training prototype（定性）
  J  ood_attribution_pie.png        全部 OOD patch 的 1-最近 prototype 的 cluster 饼 + domain 饼（定量）

从仓库根目录运行：
    .venv/bin/python scripts/run_bolt_ood_transfer.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path
from collections import Counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/temporal_primitive_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_bolt_main_figure import (  # noqa: E402
    DOMAIN_COLORS, VALIDATION_EXCLUDE, _zcorr, cluster_pca_fit, flatten_patches, z_normalize,
)
from scripts.run_second_pilot_discovery import select_domain_balanced_indices  # noqa: E402

CACHE_GLOB = str(ROOT / "outputs/figures/bolt_main_figure/.cache/extract_train_*.pkl")
FIG_DIR = ROOT / "outputs" / "figures" / "bolt_ood_transfer"
JSON_DIR = ROOT / "outputs" / "bolt_ood_transfer"
CONTROL = {"Gaussian", "Pulse"}  # 合成 negative control：天然 OOD，不进 transfer 饼图（单列）


def fit_layer(bundle, L, min_std, max_per_domain, seed, k):
    """在某层：flat-filter + balanced 聚类 discovery，投影 validation，算 OOD 分。"""
    pl = bundle["patch_len"]
    d_emb, d_raw, d_meta = flatten_patches(bundle["disc_reps"][f"layer_{L}"], bundle["disc_z"], bundle["disc_meta"], pl)
    v_emb, v_raw, v_meta = flatten_patches(bundle["val_reps"][f"layer_{L}"], bundle["val_z"], bundle["val_meta_w"], pl)

    def keep_nonflat(raw):
        return np.array([float(np.std(raw[i])) >= min_std for i in range(len(raw))])
    dk, vk = keep_nonflat(d_raw), keep_nonflat(v_raw)
    d_emb, d_raw, d_meta = d_emb[dk], d_raw[dk], [d_meta[i] for i in range(len(d_meta)) if dk[i]]
    v_emb, v_raw, v_meta = v_emb[vk], v_raw[vk], [v_meta[i] for i in range(len(v_meta)) if vk[i]]

    sel = select_domain_balanced_indices(d_meta, max_per_domain=max_per_domain, seed=seed)
    emb_b, raw_b = d_emb[sel], d_raw[sel]
    meta_b = [d_meta[i] for i in sel]
    labels, centers, train_pca, scaler, pca, _ = cluster_pca_fit(emb_b, k, seed)
    val_pca = pca.transform(scaler.transform(v_emb))

    # OOD 分：kNN 距离比（test→train 的平均近邻距离 / train 自身典型近邻距离）
    kk = 10
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(train_pca)
    dtt, _ = nn.kneighbors(train_pca)
    base = float(np.median(dtt[:, 1:kk + 1].mean(axis=1)))
    dvt, _ = nn.kneighbors(val_pca)
    ood = dvt[:, :kk].mean(axis=1) / base

    return {
        "labels": labels, "centers": centers, "train_pca": train_pca, "raw_b": raw_b, "meta_b": meta_b,
        "val_pca": val_pca, "val_raw": v_raw, "val_meta": v_meta, "ood": ood, "nn": nn,
    }


def per_dataset_ood(layer):
    ds = np.array([m["dataset"] for m in layer["val_meta"]])
    out = {}
    for d in sorted(set(ds.tolist())):
        out[d] = float(layer["ood"][ds == d].mean())
    return out


def render_overlap(layer_name, layer, seed, perplexity, out_path, max_each=2500):
    rng = np.random.default_rng(seed)
    tp, vp, ood = layer["train_pca"], layer["val_pca"], layer["ood"]
    ti = rng.choice(len(tp), min(max_each, len(tp)), replace=False)
    vi = rng.choice(len(vp), min(max_each, len(vp)), replace=False)
    X = np.vstack([tp[ti], vp[vi]])
    xy = TSNE(2, perplexity=perplexity, init="pca", random_state=seed, max_iter=1000).fit_transform(X)
    n_t = len(ti)
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.scatter(xy[:n_t, 0], xy[:n_t, 1], s=6, color="#b0b0b0", alpha=0.45,
               label=f"training (≈pretraining), n={n_t}")
    sc = ax.scatter(xy[n_t:, 0], xy[n_t:, 1], s=7, c=np.clip(ood[vi], 0.6, 2.5),
                    cmap="autumn_r", alpha=0.7)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("test OOD score (kNN dist ratio; >1 = OOD)", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2")
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"Chronos-Bolt {layer_name} — training vs held-out patch overlap\n"
                 "(shared PCA(30); red test points outside grey training cloud = OOD)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_ranking(ood_by_layer, out_path):
    layer_names = list(ood_by_layer.keys())
    datasets = sorted(set().union(*[set(d) for d in ood_by_layer.values()]),
                      key=lambda d: -ood_by_layer[layer_names[0]].get(d, 0))
    y = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(datasets) + 1.2))
    colors = ["#4C72B0", "#DD8452"]
    h = 0.38
    for li, ln in enumerate(layer_names):
        vals = [ood_by_layer[ln].get(d, 0) for d in datasets]
        ax.barh(y + (li - 0.5) * h, vals, height=h, color=colors[li % 2], label=ln)
    # 不画 "=1 in-distribution" 参照线：OOD 比值有 in/out-of-sample 偏差，IID null 不在 1，1 不是判别边界
    ax.set_yticks(y); ax.set_yticklabels(datasets, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("OOD score = mean kNN dist(test→train) / typical train neighbour dist")
    ax.legend(fontsize=8, title="layer")
    ax.set_title("Held-out test datasets ranked by OOD-ness vs training (≈pretraining) distribution", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_ood_distribution(layers, layer_Ls, out_path):
    """per-patch OOD 分布 KDE，不同 depth 叠同一张图（real held-out，剔合成 control）。
    不画 "=1" 参照线：OOD 比值有 in/out-of-sample 偏差，IID null 不在 1，1 不是判别边界。"""
    from scipy.stats import gaussian_kde
    col = {0: "#4C72B0", 11: "#DD8452"}
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    xs = np.linspace(0.3, 2.1, 400)
    for L in layer_Ls:
        lay = layers[L]
        real = np.array([m["dataset"] not in CONTROL for m in lay["val_meta"]])
        v = lay["ood"][real]
        dens = gaussian_kde(v)(xs)
        c = col.get(L, "#555555")
        ax.plot(xs, dens, color=c, lw=2,
                label=f"layer {L + 1}  (mean={v.mean():.2f}, std={v.std():.2f}, n={len(v)})")
        ax.fill_between(xs, dens, color=c, alpha=0.18)
    ax.set_xlabel("per-patch OOD score = mean kNN dist(test→train) / typical train neighbour dist")
    ax.set_ylabel("density (KDE)")
    ax.set_title("Held-out per-patch OOD distribution across depth\n"
                 "(real held-out patches, synthetic Gaussian/Pulse excluded; "
                 "deeper = narrower + right-shifted = homogenized, not less OOD)", fontsize=10)
    ax.legend(fontsize=9, title="encoder depth")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_case_study(layer, n_samples, n_proto, out_path):
    # 最 OOD 的真实样本（剔除合成 control），每个配 n_proto 个最近 training prototype
    vm = layer["val_meta"]
    real = np.array([i for i, m in enumerate(vm) if m["dataset"] not in CONTROL])
    order = real[np.argsort(layer["ood"][real])[::-1][:n_samples]]
    nn3 = NearestNeighbors(n_neighbors=n_proto).fit(layer["train_pca"])
    _, proto_idx = nn3.kneighbors(layer["val_pca"][order])

    ncol = 1 + n_proto
    fig, axes = plt.subplots(n_samples, ncol, figsize=(2.0 * ncol, 1.4 * n_samples), squeeze=False)
    for r, vi in enumerate(order):
        ax = axes[r, 0]
        ax.plot(z_normalize(layer["val_raw"][vi]), lw=1.4, color="#c0392b")
        ax.set_facecolor("#fbecea"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"{vm[vi]['dataset'][:10]}\nOOD={layer['ood'][vi]:.2f}", fontsize=6.5, rotation=0,
                      labelpad=22, va="center")
        if r == 0:
            ax.set_title("OOD test patch", fontsize=7, color="#c0392b", fontweight="bold")
        for j in range(n_proto):
            ax = axes[r, 1 + j]
            ti = proto_idx[r, j]
            dom = layer["meta_b"][ti]["macro_domain"]
            ax.plot(z_normalize(layer["raw_b"][ti]), lw=1.2, color=DOMAIN_COLORS.get(dom, "#1f2933"))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"C{int(layer['labels'][ti]) + 1}·{dom[:8]}", fontsize=6)
            if r == 0 and j == 0:
                ax.text(0.0, 1.28, "→ nearest training primitives", transform=ax.transAxes, fontsize=7,
                        color="#1f3a93", fontweight="bold")
    fig.suptitle("OOD case study — most-OOD held-out patches and their nearest training primitives "
                 "(transferable: OOD data still resembles known prototypes)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_attribution_pie(layer, k, ood_quantile, coh_thresh, out_path):
    """全部 OOD（真实）patch 的 1-最近 training patch 的 cluster / domain 分布饼图 + 覆盖率数字。"""
    vm = layer["val_meta"]
    real = np.array([i for i, m in enumerate(vm) if m["dataset"] not in CONTROL])
    thr = np.quantile(layer["ood"][real], ood_quantile)
    ood_idx = real[layer["ood"][real] >= thr]

    nn1 = NearestNeighbors(n_neighbors=1).fit(layer["train_pca"])
    _, nbr = nn1.kneighbors(layer["val_pca"][ood_idx])
    nbr = nbr[:, 0]
    clu = [int(layer["labels"][t]) for t in nbr]
    dom = [layer["meta_b"][t]["macro_domain"] for t in nbr]
    # 覆盖率：OOD patch 与其最近 cluster prototype shape 的相关 ≥ thresh 的比例
    proto = np.zeros((k, layer["raw_b"].shape[1]))
    for c in range(k):
        m = layer["labels"] == c
        if m.any():
            proto[c] = np.mean(np.stack([z_normalize(layer["raw_b"][i]) for i in np.where(m)[0]]), axis=0)
    coh = np.mean([_zcorr(z_normalize(layer["val_raw"][ood_idx[i]]), proto[clu[i]]) >= coh_thresh
                   for i in range(len(ood_idx))])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    clu_cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")
    cc = Counter(clu)
    clabels = [f"C{c + 1}" for c in sorted(cc)]
    axes[0].pie([cc[c] for c in sorted(cc)], labels=clabels, autopct="%1.0f%%", startangle=90,
                colors=[clu_cmap(c % 10) for c in sorted(cc)], textprops={"fontsize": 8})
    axes[0].set_title("nearest training PRIMITIVE (cluster)\nof OOD held-out patches", fontsize=10)
    dc = Counter(dom)
    dorder = [d for d in DOMAIN_COLORS if d in dc]
    axes[1].pie([dc[d] for d in dorder], labels=dorder, autopct="%1.0f%%", startangle=90,
                colors=[DOMAIN_COLORS[d] for d in dorder], textprops={"fontsize": 8})
    axes[1].set_title("nearest training DOMAIN\nof OOD held-out patches", fontsize=10)
    fig.suptitle(
        f"Transferable generalization — what training primitives absorb OOD test data\n"
        f"OOD = top {int((1 - ood_quantile) * 100)}% most-OOD real held-out patches (n={len(ood_idx)}); "
        f"{coh:.0%} have a training prototype with shape-corr ≥ {coh_thresh}",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"n_ood_patches": int(len(ood_idx)), "ood_threshold": round(float(thr), 3),
            "primitive_coverage_at_corr": {str(coh_thresh): round(float(coh), 3)},
            "cluster_distribution": {f"C{c + 1}": int(cc[c]) for c in sorted(cc)},
            "domain_distribution": dict(dc)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 11])
    ap.add_argument("--primary-layer", type=int, default=0, help="case study + 饼图用哪层（0-based block）")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--min-patch-std", type=float, default=0.15)
    ap.add_argument("--max-per-domain", type=int, default=400)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--tsne-perplexity", type=float, default=40.0)
    ap.add_argument("--case-samples", type=int, default=8)
    ap.add_argument("--case-protos", type=int, default=3)
    ap.add_argument("--ood-quantile", type=float, default=0.75, help="OOD = 该分位以上（默认 top 25%）")
    ap.add_argument("--coh-thresh", type=float, default=0.6)
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    hits = sorted(glob.glob(CACHE_GLOB))
    if not hits:
        raise SystemExit("no extraction cache — run build_bolt_main_figure.py first")
    bundle = pickle.load(open(hits[-1], "rb"))
    print(f"[ood] using cache {Path(hits[-1]).name}")
    # 应用 validation 排除（泄漏/预训练内）
    vk = [i for i, m in enumerate(bundle["val_meta_w"]) if m["dataset"] not in VALIDATION_EXCLUDE]
    if len(vk) < len(bundle["val_meta_w"]):
        bundle["val_z"] = bundle["val_z"][vk]
        bundle["val_meta_w"] = [bundle["val_meta_w"][i] for i in vk]
        bundle["val_reps"] = {kk: vv[vk] for kk, vv in bundle["val_reps"].items()}

    layers = {L: fit_layer(bundle, L, args.min_patch_std, args.max_per_domain, args.seed, args.k) for L in args.layers}
    ood_by_layer = {f"layer {L + 1}": per_dataset_ood(layers[L]) for L in args.layers}

    for L in args.layers:
        out = FIG_DIR / f"ood_overlap_layer{L + 1}.png"
        print(f"[ood] overlap t-SNE -> {out.name}")
        render_overlap(f"layer {L + 1}", layers[L], args.seed, args.tsne_perplexity, out)
    render_ranking(ood_by_layer, FIG_DIR / "ood_ranking.png")
    print("[ood] ranking -> ood_ranking.png")
    render_ood_distribution(layers, args.layers, FIG_DIR / "ood_distribution.png")
    print("[ood] distribution KDE -> ood_distribution.png")

    pl = args.primary_layer
    render_case_study(layers[pl], args.case_samples, args.case_protos, FIG_DIR / "ood_case_study.png")
    print("[ood] case study -> ood_case_study.png")
    pie_info = render_attribution_pie(layers[pl], args.k, args.ood_quantile, args.coh_thresh,
                                      FIG_DIR / "ood_attribution_pie.png")
    print(f"[ood] attribution pie -> ood_attribution_pie.png ; coverage={pie_info['primitive_coverage_at_corr']}")

    summary = {"model": "chronos-bolt-base", "primary_layer_block": pl,
               "metric": "OOD = mean kNN dist(test->train)/median train-train kNN dist (per layer PCA30)",
               "ood_by_dataset": ood_by_layer, "attribution": pie_info}
    (JSON_DIR / "bolt_ood_transfer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[ood] done.")


if __name__ == "__main__":
    main()
