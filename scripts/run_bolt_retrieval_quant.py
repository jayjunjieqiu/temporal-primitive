"""Quantitative retrieval statistics behind Fig. 4d / Supplementary Fig. 2 (Chronos-Bolt).

把 Fig. 4d 的"跨域 retrieval"从 qualitative examples 升级为聚合定量,回应 cherry-picking
担忧。完全复用 run_bolt_ood_transfer 的同一份提取缓存与 fit_layer(同一 discovery/held-out
split、同一 PCA(30) 空间),在每个 held-out(真实,剔除合成 control)patch 上:

1. **Cross-domain retrieval rate** —— 最近训练邻居来自不同 macro-domain 的比例(top-1;
   以及 top-k 里平均跨域比例 / 至少一个跨域)。
2. **Coherence vs random** —— 召回邻居与 query 的 shape 相关(z-normed value 相关)、
   以及 transition 相关(一阶差分相关),对比随机训练 patch 基线。**关键**:限定到
   *cross-domain* 最近邻时,coherence 仍显著高于 random ⇒ 跨域召回不是 cherry-pick。

层:默认同 S2 报 layer 1(block 0)与 layer 12(block 11)。

从仓库根目录运行:
    .venv/bin/python scripts/run_bolt_retrieval_quant.py
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_bolt_main_figure import VALIDATION_EXCLUDE, _zcorr, z_normalize  # noqa: E402
from scripts.run_bolt_ood_transfer import CACHE_GLOB, CONTROL, fit_layer  # noqa: E402

JSON_DIR = ROOT / "outputs" / "bolt_ood_transfer"


def _dom(meta: dict) -> str:
    return meta.get("macro_domain", meta.get("domain", meta.get("dataset", "?")))


def _diffcorr(a: np.ndarray, b: np.ndarray) -> float:
    """transition coherence：一阶差分序列的 Pearson 相关。"""
    da, db = np.diff(a), np.diff(b)
    if np.std(da) < 1e-8 or np.std(db) < 1e-8:
        return 0.0
    return float(np.corrcoef(da, db)[0, 1])


def _summ(vals: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> dict:
    """mean / std / bootstrap 95% CI over the per-query coherence values."""
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    nn = len(a)
    mean = float(a.mean()) if nn else 0.0
    std = float(a.std(ddof=1)) if nn > 1 else 0.0
    if nn > 1:
        boot = np.array([a[rng.integers(0, nn, nn)].mean() for _ in range(n_boot)])
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    else:
        lo = hi = mean
    return {"mean": round(mean, 3), "std": round(std, 3),
            "ci95_low": round(lo, 3), "ci95_high": round(hi, 3), "n": int(nn)}


def retrieval_quant(layer: dict, seed: int, knn: int, topn: int, n_rand_pairs: int) -> dict:
    rng = np.random.default_rng(seed)
    train_pca = layer["train_pca"]
    train_dom = np.array([_dom(m) for m in layer["meta_b"]])
    train_z = np.stack([z_normalize(r) for r in layer["raw_b"]])

    # 只在真实 held-out（剔除合成 control）上统计
    vm = layer["val_meta"]
    real = np.array([i for i, m in enumerate(vm) if m["dataset"] not in CONTROL])
    q_pca = layer["val_pca"][real]
    q_dom = np.array([_dom(vm[i]) for i in real])
    q_z = np.stack([z_normalize(layer["val_raw"][i]) for i in real])

    nn = NearestNeighbors(n_neighbors=min(topn, len(train_pca))).fit(train_pca)
    _, idx = nn.kneighbors(q_pca)  # (Nq, topn)

    # within-cluster retrieval（Fig.4d 实际过程：query 分到最近簇心，再簇内跨域取最近）
    centers = layer["centers"]
    labels_tr = np.asarray(layer["labels"])
    q_cluster = np.argmin(((q_pca[:, None, :] - centers[None, :, :]) ** 2).sum(-1), axis=1)

    n = len(real)
    top1_cross, anyk_cross, frac_k_cross = [], [], []
    coh_top1, coh_top1_diff = [], []
    coh_xd, coh_xd_diff = [], []                  # 全局最近的 *跨域* 邻居
    coh_wc_xd, coh_wc_xd_diff = [], []            # 簇内跨域最近邻（贴合 Fig.4d）
    has_xd = 0
    has_wc_xd = 0
    for i in range(n):
        nbr = idx[i]
        ndom = train_dom[nbr]
        top1_cross.append(ndom[0] != q_dom[i])
        ck = ndom[:knn] != q_dom[i]
        anyk_cross.append(bool(ck.any()))
        frac_k_cross.append(float(ck.mean()))
        # top-1 coherence
        coh_top1.append(_zcorr(q_z[i], train_z[nbr[0]]))
        coh_top1_diff.append(_diffcorr(q_z[i], train_z[nbr[0]]))
        # nearest cross-domain neighbour coherence（全局）
        xd = np.where(ndom != q_dom[i])[0]
        if len(xd):
            has_xd += 1
            j = nbr[xd[0]]
            coh_xd.append(_zcorr(q_z[i], train_z[j]))
            coh_xd_diff.append(_diffcorr(q_z[i], train_z[j]))
        # within-cluster cross-domain nearest neighbour（Fig.4d 过程）
        cand = np.where((labels_tr == q_cluster[i]) & (train_dom != q_dom[i]))[0]
        if len(cand):
            has_wc_xd += 1
            dd = ((train_pca[cand] - q_pca[i]) ** 2).sum(1)
            j = cand[int(np.argmin(dd))]
            coh_wc_xd.append(_zcorr(q_z[i], train_z[j]))
            coh_wc_xd_diff.append(_diffcorr(q_z[i], train_z[j]))

    # random baseline：随机 (query, train) 对
    qi = rng.integers(0, n, size=n_rand_pairs)
    ti = rng.integers(0, len(train_z), size=n_rand_pairs)
    rand_coh = float(np.mean([_zcorr(q_z[a], train_z[b]) for a, b in zip(qi, ti)]))
    rand_coh_diff = float(np.mean([_diffcorr(q_z[a], train_z[b]) for a, b in zip(qi, ti)]))

    # ---- raw-patch-space retrieval ceiling: nearest neighbour directly in z-normalised
    # patch space. Neighbours selected by raw shape maximise shape-correlation by
    # construction, so this is the informative control (the "shape ceiling") that
    # representation-space retrieval should be read against — random (≈0) is a trivial bar.
    nn_raw = NearestNeighbors(n_neighbors=1).fit(train_z)
    _, idx_raw = nn_raw.kneighbors(q_z)
    coh_raw = np.array([_zcorr(q_z[i], train_z[idx_raw[i, 0]]) for i in range(n)])

    # paired random baseline (one random training patch per query) so significance is paired
    rand_ti = rng.integers(0, len(train_z), size=n)
    coh_rand_paired = np.array([_zcorr(q_z[i], train_z[rand_ti[i]]) for i in range(n)])

    coh_rep = np.asarray(coh_top1, dtype=float)
    # Wilcoxon signed-rank: representation retrieval vs (a) random and (b) the raw-shape ceiling
    p_rep_vs_rand = float(wilcoxon(coh_rep, coh_rand_paired).pvalue)
    p_rep_vs_raw = float(wilcoxon(coh_rep, coh_raw).pvalue)

    # per-query-domain breakdown of representation-retrieval coherence
    per_dom = {}
    for d in sorted(set(q_dom.tolist())):
        m = q_dom == d
        per_dom[d] = {"n": int(m.sum()),
                      "mean_coherence_retrieved_top1": round(float(coh_rep[m].mean()), 3)}

    return {
        "n_query_patches": int(n),
        "cross_domain_retrieval_rate_top1": round(float(np.mean(top1_cross)), 3),
        "cross_domain_any_in_topk": round(float(np.mean(anyk_cross)), 3),
        "mean_cross_domain_fraction_topk": round(float(np.mean(frac_k_cross)), 3),
        "knn": knn,
        "coherence_retrieved_top1": round(float(np.mean(coh_top1)), 3),
        "coherence_retrieved_top1_transition": round(float(np.mean(coh_top1_diff)), 3),
        "coherence_cross_domain_neighbour": round(float(np.mean(coh_xd)), 3),
        "coherence_cross_domain_neighbour_transition": round(float(np.mean(coh_xd_diff)), 3),
        "coherence_within_cluster_cross_domain": round(float(np.mean(coh_wc_xd)), 3),
        "coherence_within_cluster_cross_domain_transition": round(float(np.mean(coh_wc_xd_diff)), 3),
        "queries_with_within_cluster_cross_domain": round(has_wc_xd / n, 3),
        "coherence_random_baseline": round(rand_coh, 3),
        "coherence_random_baseline_transition": round(rand_coh_diff, 3),
        "queries_with_cross_domain_neighbour": round(has_xd / n, 3),
        "coherence_raw_patch_ceiling_top1": round(float(coh_raw.mean()), 3),
        "dispersion": {
            "representation_retrieval_top1": _summ(coh_rep, rng),
            "raw_patch_retrieval_top1": _summ(coh_raw, rng),
            "random_paired": _summ(coh_rand_paired, rng),
        },
        "significance_wilcoxon_pvalue": {
            "representation_vs_random": p_rep_vs_rand,
            "representation_vs_raw_ceiling": p_rep_vs_raw,
        },
        "per_query_domain_coherence": per_dom,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 11])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--min-patch-std", type=float, default=0.15)
    ap.add_argument("--max-per-domain", type=int, default=400)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--rand-pairs", type=int, default=40000)
    args = ap.parse_args()

    hits = sorted(glob.glob(CACHE_GLOB))
    if not hits:
        raise SystemExit("no extraction cache — run build_bolt_main_figure.py first")
    bundle = pickle.load(open(hits[-1], "rb"))
    print(f"[retr-quant] using cache {Path(hits[-1]).name}")
    vk = [i for i, m in enumerate(bundle["val_meta_w"]) if m["dataset"] not in VALIDATION_EXCLUDE]
    if len(vk) < len(bundle["val_meta_w"]):
        bundle["val_z"] = bundle["val_z"][vk]
        bundle["val_meta_w"] = [bundle["val_meta_w"][i] for i in vk]
        bundle["val_reps"] = {kk: vv[vk] for kk, vv in bundle["val_reps"].items()}

    out = {}
    for L in args.layers:
        layer = fit_layer(bundle, L, args.min_patch_std, args.max_per_domain, args.seed, args.k)
        stats = retrieval_quant(layer, args.seed, args.knn, args.topn, args.rand_pairs)
        out[f"layer {L + 1}"] = stats
        print(f"[retr-quant] layer {L + 1}: xd-rate(top1)={stats['cross_domain_retrieval_rate_top1']:.0%} "
              f"any-xd-in-top{args.knn}={stats['cross_domain_any_in_topk']:.0%} | "
              f"coh rep={stats['coherence_retrieved_top1']:.3f} "
              f"raw-ceiling={stats['coherence_raw_patch_ceiling_top1']:.3f} "
              f"random={stats['coherence_random_baseline']:.3f} | "
              f"p(rep>rand)={stats['significance_wilcoxon_pvalue']['representation_vs_random']:.1e} "
              f"p(rep<ceil)={stats['significance_wilcoxon_pvalue']['representation_vs_raw_ceiling']:.1e}")

    summary = {
        "model": "chronos-bolt-base",
        "note": ("Aggregate retrieval statistics over real held-out patches (synthetic controls excluded), "
                 "same discovery/held-out split and PCA(30) space as Supplementary Fig. 2. "
                 "Coherence = Pearson corr of z-normalised patch shape; transition = corr of first differences. "
                 "Random baseline = random (held-out query, training patch) pairs. "
                 "coherence_raw_patch_ceiling_top1 = nearest neighbour retrieved directly in z-normalised raw-patch "
                 "space (the shape ceiling); representation retrieval is read against this, not only random. "
                 "dispersion = mean/std/bootstrap-95%CI over per-query coherence; "
                 "significance = Wilcoxon signed-rank p-values (representation vs random; representation vs raw ceiling)."),
        "config": vars(args),
        "by_layer": out,
    }
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    (JSON_DIR / "bolt_retrieval_quant_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[retr-quant] saved -> {JSON_DIR / 'bolt_retrieval_quant_summary.json'}")


if __name__ == "__main__":
    main()
