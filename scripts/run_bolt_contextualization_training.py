"""Task 2 (训练数据版) — Chronos-Bolt contextualization，左面板用直接 probe accuracy 取代 NMI。

为什么不用 NMI（见 docs/14）：旧左面板是 NMI(KMeans 簇标签, confounder)，经 KMeans + 固定 k，
深层连续体切分会让它失真（domain 出现中层达峰后回落的 artifact）。这里改成**不经 KMeans 的直接
decodability**——对每个 confounder（domain / frequency / position）做 k-NN probe accuracy：从每层
表征预测该 confounder，看随深度怎么变。右面板（within-context 相似度）沿用，但统一到训练数据。

数据：Chronos in-distribution 训练子集（与 main figure / docs/15 一致）。默认复用
run_bolt_domain_separation.py 的全层提取缓存（无需再上 GPU）。层号显示 1-based（block + 1）。

从仓库根目录运行：
    .venv/bin/python scripts/run_bolt_contextualization_training.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/temporal_primitive_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_bolt_main_figure import flatten_patches  # noqa: E402
from scripts.chronos_bolt_backbone import extract_bolt_representations, load_bolt_pipeline  # noqa: E402
from scripts.chronos_training_data import DATASET_FREQ_MINUTES, sample_training_windows  # noqa: E402
from scripts.run_second_pilot_discovery import robust_z, select_domain_balanced_indices  # noqa: E402

JSON_DIR = ROOT / "outputs" / "bolt_contextualization"
FIG_DIR = ROOT / "outputs" / "figures" / "bolt_contextualization"
CACHE_DIR = JSON_DIR / ".cache"

# confounder 配色（与旧 NMI 面板一致：domain 蓝 / frequency 橙 / position 绿）
CONF_COLOR = {"domain": "#1f77b4", "frequency": "#ff7f0e", "position": "#2ca02c"}


def knn_accuracy(X: np.ndarray, codes: np.ndarray, kk: int = 10) -> float:
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
    _, idx = nn.kneighbors(X)
    neigh = codes[idx[:, 1:]]
    pred = np.array([np.bincount(row).argmax() for row in neigh])
    return float((pred == codes).mean())


def patch_labels(window_meta: list[dict[str, Any]], num_patches: int) -> dict[str, np.ndarray]:
    """与 flatten_patches 相同的 (window, patch) 顺序，给出 domain / frequency / position 标签。"""
    dom, freq, pos = [], [], []
    for meta in window_meta:
        fm = meta.get("frequency_minutes", DATASET_FREQ_MINUTES.get(meta.get("dataset_path")))
        for p in range(num_patches):
            dom.append(meta["macro_domain"])
            freq.append("none" if fm is None else str(fm))
            pos.append(p)
    return {"domain": np.asarray(dom), "frequency": np.asarray(freq), "position": np.asarray(pos)}


def _local_global_pairs(emb: np.ndarray, n_global_pairs: int, seed: int):
    n, p, d = emb.shape
    normed = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
    sim = np.einsum("npd,nqd->npq", normed, normed)
    iu = np.triu_indices(p, k=1)
    local_vals = sim[:, iu[0], iu[1]].reshape(-1)
    rng = np.random.default_rng(seed)
    flat = normed.reshape(n * p, d)
    win_of = np.repeat(np.arange(n), p)
    a = rng.integers(0, n * p, size=n_global_pairs)
    b = rng.integers(0, n * p, size=n_global_pairs)
    keep = win_of[a] != win_of[b]
    global_vals = np.einsum("id,id->i", flat[a[keep]], flat[b[keep]])
    return local_vals, global_vals


def within_context_similarity(emb: np.ndarray, n_global_pairs: int, seed: int) -> dict:
    """centered cosine：同 context（同窗口不同位置）vs 不同 context（跨窗口随机）。"""
    centered = emb - emb.reshape(-1, emb.shape[-1]).mean(axis=0)
    cl, cg = _local_global_pairs(centered, n_global_pairs, seed)
    return {"same_context": float(cl.mean()), "different_context": float(cg.mean()),
            "gap": float(cl.mean() - cg.mean())}


def load_extraction(args):
    """优先复用 domain-separation 全层缓存；否则重新提取（GPU）。"""
    hits = sorted(glob.glob(str(CACHE_DIR / "domsep_extract_*L0-1-2-3-4-5-6-7-8-9-10-11.pkl")))
    if hits and not args.no_cache:
        print(f"[ctx-train] reusing extraction cache -> {Path(hits[0]).name}")
        b = pickle.load(open(hits[0], "rb"))
        return b["windows_z"], b["window_meta"], b["reps"], b["patch_len"]
    print(f"[ctx-train] sampling training windows (per_dataset={args.windows_per_dataset}) + extracting ...")
    windows, window_meta, _ = sample_training_windows(
        context_len=args.context_len, windows_per_dataset=args.windows_per_dataset, seed=args.seed)
    windows_z = np.stack([robust_z(w) for w in windows]).astype(np.float32)
    pipe = load_bolt_pipeline()
    patch_len = int(pipe.model.chronos_config.input_patch_size)
    reps = extract_bolt_representations(windows_z, batch_size=args.batch_size, layers=list(range(12)),
                                        include_tokenizer=True, pipeline=pipe, keep_pipeline=False)
    return windows_z, window_meta, reps, patch_len


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows-per-dataset", type=int, default=200)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 3, 6, 9, 11],
                    help="要画的 encoder 层（0-based block 索引；显示为 +1）；默认稀疏取 5 层")
    ap.add_argument("--max-per-domain", type=int, default=400)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--global-pairs", type=int, default=400000)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    windows_z, window_meta, reps, patch_len = load_extraction(args)
    num_patches = reps["tokenizer"].shape[1]
    labels = patch_labels(window_meta, num_patches)
    # domain-balanced patch 子集（probe 用，避免高频域主导）
    _e0, _r0, meta0 = flatten_patches(reps["layer_0"], windows_z, window_meta, patch_len)
    sel = select_domain_balanced_indices(meta0, max_per_domain=args.max_per_domain, seed=args.seed)
    freq_ok = labels["frequency"][sel] != "none"  # frequency probe 剔除 synthetic

    rep_order = ["tokenizer"] + [f"layer_{L}" for L in args.layers]
    rep_label = {"tokenizer": "tokenizer\n(input embed)", **{f"layer_{L}": f"enc L{L + 1}" for L in args.layers}}
    results: dict[str, Any] = {}
    for rep in rep_order:
        emb = reps[rep]  # (N, P, d)
        flat, _, _ = flatten_patches(emb, windows_z, window_meta, patch_len)
        Xp = PCA(n_components=min(30, flat.shape[1]), random_state=args.seed).fit_transform(
            StandardScaler().fit_transform(flat[sel]))
        def code(arr):
            return np.unique(arr, return_inverse=True)[1]
        acc = {
            "domain": round(knn_accuracy(Xp, code(labels["domain"][sel]), args.knn), 4),
            "frequency": round(knn_accuracy(Xp[freq_ok], code(labels["frequency"][sel][freq_ok]), args.knn), 4),
            "position": round(knn_accuracy(Xp, code(labels["position"][sel]), args.knn), 4),
        }
        # 全局视角（appendix 对照）：Calinski-Harabasz 类间/类内方差比（局部 kNN 之外的补充）
        ch = {
            "domain": round(float(calinski_harabasz_score(Xp, code(labels["domain"][sel]))), 1),
            "frequency": round(float(calinski_harabasz_score(Xp[freq_ok], code(labels["frequency"][sel][freq_ok]))), 1),
            "position": round(float(calinski_harabasz_score(Xp, code(labels["position"][sel]))), 1),
        }
        sim = within_context_similarity(emb, args.global_pairs, args.seed)
        results[rep] = {"knn_probe_acc": acc, "calinski_harabasz": ch, "within_context_sim": sim}
        print(f"[ctx-train] {rep:<10} acc dom={acc['domain']:.3f} freq={acc['frequency']:.3f} "
              f"pos={acc['position']:.3f} | CH dom={ch['domain']:.0f} freq={ch['frequency']:.0f} "
              f"pos={ch['position']:.0f} | same-ctx sim={sim['same_context']:.3f}")

    # chance（majority baseline）
    chance = {
        "domain": float(np.bincount(np.unique(labels["domain"][sel], return_inverse=True)[1]).max() / sel.size),
        "frequency": float(np.bincount(np.unique(labels["frequency"][sel][freq_ok], return_inverse=True)[1]).max() / max(1, freq_ok.sum())),
        "position": 1.0 / num_patches,
    }
    summary = {"model": "chronos-bolt-base", "data": "chronos in-distribution training subset",
               "metric_note": ("left = k-NN probe accuracy of each confounder from the representation "
                               "(direct decodability, NO KMeans; replaces NMI); right = within-context "
                               "centered-cosine similarity. frequency probe excludes synthetic (no cadence)."),
               "chance": chance, "config": vars(args), "rep_order": rep_order, "results": results}
    (JSON_DIR / "bolt_contextualization_training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 图：左 = probe accuracy(domain/freq/position)，右 = within-context similarity ----
    xs = list(range(len(rep_order)))
    xlabels = [rep_label[r] for r in rep_order]
    tok_x = 0
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    for conf in ["domain", "frequency", "position"]:
        ys = [results[r]["knn_probe_acc"][conf] for r in rep_order]
        ax.plot(xs, ys, "-o", color=CONF_COLOR[conf], lw=2, ms=5, label=conf)
        ax.axhline(chance[conf], ls=":", color=CONF_COLOR[conf], lw=1, alpha=0.6)
    ax.set_ylabel("10-NN probe accuracy  (↑ = more decodable)")
    ax.set_title("Confounder decodability vs depth\n(direct k-NN probe; dotted = chance; replaces NMI)", fontsize=10)
    ax.legend(fontsize=9, title="confounder")

    ax = axes[1]
    same = [results[r]["within_context_sim"]["same_context"] for r in rep_order]
    diff = [results[r]["within_context_sim"]["different_context"] for r in rep_order]
    ax.plot(xs, same, "-o", color="#d62728", lw=2, ms=5, label="same context\n(diff-position patches)")
    ax.plot(xs, diff, "-o", color="#7f7f7f", lw=2, ms=5, label="different context\n(random patches)")
    ax.fill_between(xs, diff, same, color="#d62728", alpha=0.12)
    ax.set_ylabel("centered cosine similarity")
    ax.set_title("Within-context patch similarity vs depth", fontsize=10)
    ax.legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.axvline(tok_x + 0.5, color="#2ca02c", ls=":", lw=1)
        ax.axvspan(tok_x + 0.5, len(rep_order) - 0.5, color="#2ca02c", alpha=0.06)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Chronos-Bolt — representations become contextualized with depth (training data)\n"
        "value-only tokenizer → encoder layers make domain/cadence/position more decodable; "
        "within-context patches grow more similar (k-NN probe replaces KMeans-NMI)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = FIG_DIR / "bolt_contextualization_probe_depth.png"
    fig.savefig(out, dpi=150)
    print(f"[ctx-train] saved -> {out}")


if __name__ == "__main__":
    main()
