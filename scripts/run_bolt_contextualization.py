"""Task 2 — Chronos-Bolt contextualization analysis（随层 contextualized 的 clean 证据）。

两个 layer-wise 量随 representation 深度（tokenizer → encoder layers）变化：

1. **NMI（confounder absorption）**：每层在 PCA space 做 KMeans，算 cluster label 与
   `macro_domain` / `frequency` / `patch_index` 的 normalized mutual information。深层吸收
   更多 domain / cadence / position 信息 → contextualization 的一个侧面。

2. **local vs global patch similarity**：把同一窗口内的 patch 当 *local*（共享 context）、
   跨窗口随机 patch 当 *global*，比较 representation space 余弦相似度。深层 local−global
   gap 变大 = patch 表示越来越依赖它所在的 context = contextualized。

全部基于 clean Chronos-Bolt（input token value-only，无 time encoding）。

从仓库根目录运行：
    .venv/bin/python scripts/run_bolt_contextualization.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.chronos_bolt_backbone import extract_bolt_representations, load_bolt_pipeline  # noqa: E402
from scripts.run_prior_guided_probe_sanity_check import macro_domain  # noqa: E402
from scripts.run_second_pilot_discovery import DATA_ROOT, robust_z, sample_windows  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "bolt_contextualization"


def patch_level_labels(
    window_meta: list[dict[str, Any]], num_patches: int
) -> dict[str, np.ndarray]:
    macro, freq, pidx, win = [], [], [], []
    for i, meta in enumerate(window_meta):
        for p in range(num_patches):
            macro.append(macro_domain(meta.get("domain")))
            freq.append(str(meta.get("frequency_minutes")))
            pidx.append(p)
            win.append(i)
    return {
        "macro_domain": np.asarray(macro),
        "frequency": np.asarray(freq),
        "patch_index": np.asarray(pidx),
        "window": np.asarray(win),
    }


def nmi_for_layer(emb_flat: np.ndarray, labels: dict[str, np.ndarray], k: int, seed: int) -> dict:
    Xs = StandardScaler().fit_transform(emb_flat)
    pca = PCA(n_components=min(30, Xs.shape[1]), random_state=seed)
    Xp = pca.fit_transform(Xs)
    cluster_ids = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xp)
    return {
        name: float(normalized_mutual_info_score(values, cluster_ids))
        for name, values in labels.items()
        if name != "window"
    }


def _local_global_pairs(emb: np.ndarray, n_global_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (local_vals, global_vals) 的 cosine 相似度。

    local  = 同窗口、不同位置的 patch 对（共享 context）。
    global = 跨窗口随机 patch 对（不同 context）。
    """
    n, p, d = emb.shape
    normed = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)

    sim = np.einsum("npd,nqd->npq", normed, normed)  # (N, P, P)
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


def local_global_similarity(emb: np.ndarray, n_global_pairs: int, seed: int) -> dict:
    """同 context 不同位置 patch 的相似度是否随 depth 上升。

    研究问题（advisor）：同一 context 下不同位置 patch representation 之间的相似度会不会
    随 depth 增加而增加？直接量 = ``local_mean``。但绝对 cosine 受"深层表示空间整体散开"
    confound（global 也会一起降），所以同时报告：
    - ``raw``      : 原始 cosine（local / global / gap）。
    - ``centered`` : 每层先减全局均值方向（去掉 shared component）再算 cosine —— 去掉散开
                     confounder 后，同 context 耦合本身随 depth 的变化。
    - ``gap`` 与 ``ratio`` 是控制了 global baseline 的 contextualization 指标。
    """
    raw_local, raw_global = _local_global_pairs(emb, n_global_pairs, seed)

    centered = emb - emb.reshape(-1, emb.shape[-1]).mean(axis=0)
    cen_local, cen_global = _local_global_pairs(centered, n_global_pairs, seed)

    def pack(local_vals: np.ndarray, global_vals: np.ndarray) -> dict:
        lm, gm = float(local_vals.mean()), float(global_vals.mean())
        return {
            "local_mean": lm,
            "global_mean": gm,
            "gap": lm - gm,
            "ratio": float(lm / gm) if gm != 0 else float("nan"),
        }

    return {
        "raw": pack(raw_local, raw_global),
        "centered": pack(cen_local, cen_global),
        "n_local_pairs": int(raw_local.size),
        "n_global_pairs": int(raw_global.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-per-dataset", type=int, default=100)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3, 6, 9, 11])
    parser.add_argument("--k", type=int, default=10, help="KMeans clusters for NMI")
    parser.add_argument("--global-pairs", type=int, default=400000)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[ctx] sampling windows (per_dataset={args.windows_per_dataset}, ctx={args.context_len})")
    windows, window_meta, dataset_summary = sample_windows(
        DATA_ROOT,
        context_len=args.context_len,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
    )
    windows_z = np.stack([robust_z(w) for w in windows]).astype(np.float32)
    print(f"[ctx] {len(windows)} windows across {len(dataset_summary)} datasets")

    print(f"[ctx] extracting Bolt representations (layers={args.layers}) ...")
    pipe = load_bolt_pipeline()
    patch_len = int(pipe.model.chronos_config.input_patch_size)
    reps = extract_bolt_representations(
        windows_z,
        batch_size=args.batch_size,
        layers=args.layers,
        include_tokenizer=True,
        pipeline=pipe,
        keep_pipeline=False,
    )
    num_patches = reps["tokenizer"].shape[1]
    labels = patch_level_labels(window_meta, num_patches)

    rep_order = ["tokenizer"] + [f"layer_{L}" for L in args.layers]
    results: dict[str, Any] = {}
    for rep in rep_order:
        emb = reps[rep]  # (N, P, d)
        emb_flat = emb.reshape(-1, emb.shape[-1])
        print(f"[ctx] {rep}: NMI (k={args.k}) + local/global similarity ...")
        results[rep] = {
            "nmi": nmi_for_layer(emb_flat, labels, args.k, args.seed),
            "similarity": local_global_similarity(emb, args.global_pairs, args.seed),
        }

    summary = {
        "model": "chronos-bolt-base",
        "note": "clean successor to archived Chronos-2 pilot; input token value-only (no time encoding)",
        "config": {
            "windows_per_dataset": args.windows_per_dataset,
            "context_len": args.context_len,
            "patch_len": patch_len,
            "num_patches": int(num_patches),
            "layers": args.layers,
            "k": args.k,
            "seed": args.seed,
            "n_windows": int(len(windows)),
        },
        "rep_order": rep_order,
        "results": results,
    }
    out_json = args.out / "bolt_contextualization_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== layer-wise contextualization ===")
    hdr = f"{'rep':<12}{'NMI_dom':>8}{'NMI_freq':>9}{'NMI_pos':>8}"
    hdr += f"{'raw_loc':>9}{'raw_glob':>9}{'raw_gap':>8}{'cen_loc':>9}{'cen_glob':>9}{'cen_gap':>8}"
    print(hdr)
    for rep in rep_order:
        nm = results[rep]["nmi"]
        raw = results[rep]["similarity"]["raw"]
        cen = results[rep]["similarity"]["centered"]
        print(
            f"{rep:<12}{nm['macro_domain']:>8.3f}{nm['frequency']:>9.3f}{nm['patch_index']:>8.3f}"
            f"{raw['local_mean']:>9.3f}{raw['global_mean']:>9.3f}{raw['gap']:>8.3f}"
            f"{cen['local_mean']:>9.3f}{cen['global_mean']:>9.3f}{cen['gap']:>8.3f}"
        )
    print(f"\n[ctx] saved -> {out_json}")


if __name__ == "__main__":
    main()
