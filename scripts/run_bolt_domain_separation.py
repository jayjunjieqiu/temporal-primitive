"""Layer-wise macro-domain separation of Chronos-Bolt patch representations.

量化 docs/14 的 contextualization：随 encoder depth 增加，patch 表征是否越来越按 macro_domain
聚合。用**有监督分离度**（不是 inertia——inertia 无监督、依赖每层尺度、跨层不可比）：

- **10-NN domain accuracy**（headline，最直观）："一个 patch 的 10 个最近邻里有多少跟它同 domain"。
- **Calinski-Harabasz**（类间/类内方差比 = inertia 思路的正确归一化版）。
- **silhouette(domain)**（对照；macro domain 是重叠多模流形，这个会一直≈0，说明它不是合适指标）。

全部在每层各自的 PCA(30) 空间、domain-balanced 子集上算（与 main figure 同一管线）。discovery 数据 =
Chronos in-distribution 训练子集（见 scripts/chronos_training_data.py）。

层号显示 1-based（Nature 习惯）= encoder block 索引 + 1；tokenizer 为 pre-encoder 锚点。

从仓库根目录运行：
    .venv/bin/python scripts/run_bolt_domain_separation.py
"""
from __future__ import annotations

import argparse
import json
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
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_bolt_main_figure import flatten_patches  # noqa: E402
from scripts.chronos_bolt_backbone import extract_bolt_representations, load_bolt_pipeline  # noqa: E402
from scripts.chronos_training_data import sample_training_windows  # noqa: E402
from scripts.run_second_pilot_discovery import robust_z, select_domain_balanced_indices  # noqa: E402

OUT_JSON_DIR = ROOT / "outputs" / "bolt_contextualization"
FIG_DIR = ROOT / "outputs" / "figures" / "bolt_contextualization"


def knn_accuracy(X: np.ndarray, codes: np.ndarray, kk: int = 10) -> float:
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
    _, idx = nn.kneighbors(X)
    neigh = codes[idx[:, 1:]]  # 去掉自身
    pred = np.array([np.bincount(row).argmax() for row in neigh])
    return float((pred == codes).mean())


def separation_metrics(emb, windows_z, window_meta, patch_len, max_per_domain, seed, knn):
    _emb, _raw, meta = flatten_patches(emb, windows_z, window_meta, patch_len)
    sel = select_domain_balanced_indices(meta, max_per_domain=max_per_domain, seed=seed)
    Xp = PCA(n_components=min(30, _emb.shape[1]), random_state=seed).fit_transform(
        StandardScaler().fit_transform(_emb[sel])
    )
    doms = np.array([meta[i]["macro_domain"] for i in sel])
    _uniq, codes = np.unique(doms, return_inverse=True)
    counts = np.bincount(codes)
    return {
        "n": int(len(sel)),
        "knn_domain_acc": round(knn_accuracy(Xp, codes, knn), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(Xp, codes)), 2),
        "silhouette_domain": round(float(silhouette_score(Xp, codes)), 4),
        "majority_baseline": round(float(counts.max() / counts.sum()), 4),
        "n_domains": int(len(_uniq)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows-per-dataset", type=int, default=200)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--max-per-domain", type=int, default=400)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cache = OUT_JSON_DIR / ".cache" / (
        f"domsep_extract_wpd{args.windows_per_dataset}_ctx{args.context_len}"
        f"_seed{args.seed}_L{'-'.join(map(str, args.layers))}.pkl"
    )
    if not args.no_cache and cache.exists():
        print(f"[domsep] loading extraction cache -> {cache.name}")
        bundle = pickle.load(open(cache, "rb"))
        windows_z, window_meta, reps, patch_len = (
            bundle["windows_z"], bundle["window_meta"], bundle["reps"], bundle["patch_len"]
        )
    else:
        print(f"[domsep] sampling training discovery windows (per_dataset={args.windows_per_dataset})")
        windows, window_meta, _ = sample_training_windows(
            context_len=args.context_len, windows_per_dataset=args.windows_per_dataset, seed=args.seed
        )
        windows_z = np.stack([robust_z(w) for w in windows]).astype(np.float32)
        print(f"[domsep] {len(windows_z)} windows; extracting tokenizer + encoder layers {args.layers} ...")
        pipe = load_bolt_pipeline()
        patch_len = int(pipe.model.chronos_config.input_patch_size)
        reps = extract_bolt_representations(
            windows_z, batch_size=args.batch_size, layers=args.layers,
            include_tokenizer=True, pipeline=pipe, keep_pipeline=False,
        )
        if not args.no_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump({"windows_z": windows_z, "window_meta": window_meta,
                         "reps": reps, "patch_len": patch_len}, open(cache, "wb"))
            print(f"[domsep] cached extraction -> {cache.name}")

    # x 轴顺序：tokenizer（pre-encoder）-> encoder layers（1-based 显示）
    rep_order = ["tokenizer"] + [f"layer_{L}" for L in args.layers]
    rep_label = {"tokenizer": "tokenizer\n(input embed)", **{f"layer_{L}": f"enc L{L + 1}" for L in args.layers}}
    results = {}
    for rep in rep_order:
        if rep not in reps:
            continue
        m = separation_metrics(reps[rep], windows_z, window_meta, patch_len,
                               args.max_per_domain, args.seed, args.knn)
        results[rep] = m
        print(f"[domsep] {rep:<10} 10NN-acc={m['knn_domain_acc']:.3f} "
              f"CH={m['calinski_harabasz']:.1f} silhouette={m['silhouette_domain']:+.3f}")

    summary = {"model": "chronos-bolt-base", "metric_note": (
        "supervised macro-domain separation in per-layer PCA(30); knn_domain_acc = fraction of a "
        "patch's k nearest neighbours sharing its macro_domain; CH = between/within variance ratio; "
        "silhouette(domain) stays ~0 because domains are overlapping manifolds (kept as a foil)."),
        "config": vars(args), "results": results}
    (OUT_JSON_DIR / "bolt_domain_separation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 图：2 面板（10-NN domain acc / Calinski-Harabasz）随深度 ----
    order = [r for r in rep_order if r in results]
    xs = list(range(len(order)))
    xlabels = [rep_label[r] for r in order]
    tok_x = order.index("tokenizer") if "tokenizer" in order else -1
    acc = [results[r]["knn_domain_acc"] for r in order]
    ch = [results[r]["calinski_harabasz"] for r in order]
    base = results[order[0]]["majority_baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    axes[0].plot(xs, acc, "-o", color="#2ca02c", lw=2, ms=6)
    axes[0].axhline(base, ls="--", color="gray", lw=1)
    axes[0].text(0.02, base, f"majority baseline ({base:.2f})", color="gray", fontsize=8,
                 va="bottom", transform=axes[0].get_yaxis_transform())
    axes[0].set_ylabel("10-NN macro-domain accuracy  (↑ = more domain-aggregated)")
    axes[0].set_title("Domain aggregation vs depth\n(fraction of a patch's neighbours sharing its domain)", fontsize=10)

    axes[1].plot(xs, ch, "-o", color="#1f77b4", lw=2, ms=6)
    axes[1].set_ylabel("Calinski-Harabasz (between/within var ratio)  (↑ better)")
    axes[1].set_title("Domain variance-ratio vs depth\n(normalized 'inertia' done supervised)", fontsize=10)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=8)
        if tok_x >= 0:
            ax.axvline(tok_x + 0.5, color="#2ca02c", ls=":", lw=1)
            ax.axvspan(tok_x + 0.5, len(order) - 0.5, color="#2ca02c", alpha=0.06)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Chronos-Bolt — macro-domain aggregation increases with encoder depth (contextualization)\n"
        "supervised separation in per-layer PCA(30), domain-balanced training patches; "
        "inertia is unsuitable (scale-dependent), so we use kNN-accuracy + variance-ratio",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = FIG_DIR / "bolt_domain_separation_depth.png"
    fig.savefig(out, dpi=150)
    print(f"[domsep] saved -> {out}")


if __name__ == "__main__":
    main()
