"""Task 1 — Chronos-Bolt main figure（迁到 clean Bolt）。

老师反馈：main 图不要 first-difference / power-spectrum 行，只留 **raw patch stack**；加上
**最深层**；下面用**分 domain 的 prototype example** 那张。全部迁到 clean Chronos-Bolt。

产出 modular PNG（用户偏好手动拼接，不要自动合成整图）：
  - `bolt_patch_stack_cards_layer0.png`   : layer_0 (shallow) 的 raw-only patch-stack cards
  - `bolt_patch_stack_cards_layer11.png`  : layer_11 (deepest, contextualized) 的 raw-only cards
  - `bolt_domain_balanced_prototype_panel.png` : domain-balanced 聚类的 prototype example panel

方法（two-space principle，见 docs/00_narrative_rules.md §5.1）：在 representation space 用
StandardScaler → PCA(30) → KMeans(k) 生成候选 cluster；回到 original time-series space 用
z-normalized raw patch 展示形状。t-SNE 不参与（这里只要 cards/prototype，不要 atlas）。

注意：这些是 clean Chronos-Bolt 结构证据，但仍是 *candidate* cluster，不是命名好的 motif；
图注遵守 narrative rules（不要把 raw KMeans cluster 直接叫 motif）。

从仓库根目录运行：
    .venv/bin/python scripts/build_bolt_main_figure.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/temporal_primitive_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.chronos_bolt_backbone import extract_bolt_representations, load_bolt_pipeline  # noqa: E402
from scripts.run_prior_guided_probe_sanity_check import macro_domain  # noqa: E402
from scripts.run_second_pilot_discovery import (  # noqa: E402
    DATA_ROOT,
    robust_z,
    sample_windows,
    select_domain_balanced_indices,
)

OUTPUT_DIR = ROOT / "outputs" / "figures" / "bolt_main_figure"


def z_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x - float(np.mean(x))) / max(float(np.std(x)), eps)).astype(np.float32)


def flatten_patches(
    layer_emb: np.ndarray, windows_z: np.ndarray, window_meta: list[dict[str, Any]], patch_len: int
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n, num_patches, _ = layer_emb.shape
    emb_list, raw_list, meta_list = [], [], []
    for i in range(n):
        for p in range(num_patches):
            raw = windows_z[i, p * patch_len : (p + 1) * patch_len]
            if raw.shape[0] < patch_len:
                continue
            emb_list.append(layer_emb[i, p])
            raw_list.append(raw)
            meta_list.append(
                {
                    "dataset": window_meta[i]["dataset"],
                    "domain": window_meta[i].get("domain"),
                    "macro_domain": macro_domain(window_meta[i].get("domain")),
                    "patch_index": p,
                }
            )
    return np.stack(emb_list).astype(np.float32), np.stack(raw_list).astype(np.float32), meta_list


def cluster_pca(emb: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """StandardScaler -> PCA(30) -> KMeans。返回 (labels, centers_in_pca, pca_coords)。"""
    Xs = StandardScaler().fit_transform(emb)
    Xp = PCA(n_components=min(30, Xs.shape[1]), random_state=seed).fit_transform(Xs)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xp)
    return km.labels_, km.cluster_centers_, Xp


def render_raw_cards(
    layer_name: str,
    labels: np.ndarray,
    centers: np.ndarray,
    pca_coords: np.ndarray,
    raw_patches: np.ndarray,
    meta: list[dict[str, Any]],
    k: int,
    top_n: int,
    out_path: Path,
) -> dict[str, Any]:
    """每个 cluster 一张 raw patch-stack 卡（center-nearest top_n，z-normalized imshow）。"""
    fig, axes = plt.subplots(1, k, figsize=(2.05 * k, 3.2), squeeze=False)
    card_info = []
    for cid in range(k):
        ax = axes[0, cid]
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            ax.axis("off")
            continue
        dist = np.linalg.norm(pca_coords[idx] - centers[cid], axis=1)
        order = np.argsort(dist)[: min(top_n, len(idx))]
        chosen = idx[order]
        z = np.stack([z_normalize(raw_patches[i]) for i in chosen])
        vmax = float(np.percentile(np.abs(z), 97))
        ax.imshow(z, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks([0, 7, 15])
        ax.set_yticks([])
        ax.tick_params(labelsize=6, length=2)
        dom = Counter(meta[i]["macro_domain"] for i in chosen).most_common(1)[0][0]
        ax.set_title(f"C{cid + 1}  (n={len(idx)})\n{dom}", fontsize=8, pad=3)
        ax.set_xlabel("time", fontsize=7)
        if cid == 0:
            ax.set_ylabel("rank (center→far)", fontsize=7.5)
        for sp in ax.spines.values():
            sp.set_color("#1f2933")
            sp.set_linewidth(0.8)
        card_info.append({"cluster": f"C{cid + 1}", "size": int(len(idx)), "dominant_macro_domain": dom})
    fig.suptitle(
        f"Chronos-Bolt {layer_name} — raw patch-stack candidate clusters (k={k}, center-nearest {top_n})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"output": str(out_path), "clusters": card_info}


def render_cluster_maps(
    layer_clusters: dict[str, tuple[np.ndarray, np.ndarray]],
    k: int,
    seed: int,
    perplexity: float,
    out_path: Path,
) -> dict[str, Any]:
    """两个 depth 的聚类散点图（中间 plate）。

    KMeans 在 PCA(30) space 完成（cluster labels 来自那里）；t-SNE 只把同一 PCA 坐标投到 2D
    做 visualization（不参与聚类，见图注）。每个 panel 一个 depth，点按 cluster 上色。
    """
    names = list(layer_clusters.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 5.0), squeeze=False)
    cmap = plt.get_cmap("tab10")
    info: dict[str, Any] = {}
    for col, name in enumerate(names):
        ax = axes[0, col]
        labels, pca_coords = layer_clusters[name]
        tsne = TSNE(
            n_components=2, perplexity=perplexity, init="pca",
            random_state=seed, max_iter=1000,
        )
        xy = tsne.fit_transform(pca_coords)
        for cid in range(k):
            m = labels == cid
            ax.scatter(xy[m, 0], xy[m, 1], s=6, color=cmap(cid % 10), alpha=0.6, label=f"C{cid + 1}")
        ax.set_title(f"Chronos-Bolt {name}\nKMeans clusters (PCA space) · t-SNE view", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t-SNE-1", fontsize=8)
        ax.set_ylabel("t-SNE-2", fontsize=8)
        for sp in ax.spines.values():
            sp.set_color("#1f2933")
            sp.set_linewidth(0.8)
        if col == len(names) - 1:
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, title="cluster")
        info[name] = {"n_points": int(len(labels))}
    fig.suptitle(
        "Representation atlas across depth — cluster structure reorganizes (KMeans in PCA space; "
        "t-SNE for visualization only)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"output": str(out_path), "panels": info, "perplexity": perplexity}


def render_prototype_panel(
    emb: np.ndarray,
    raw_patches: np.ndarray,
    meta: list[dict[str, Any]],
    k: int,
    seed: int,
    proto_per_cluster: int,
    max_per_domain: int,
    out_path: Path,
    layer_name: str,
) -> dict[str, Any]:
    """domain-balanced 聚类的 prototype example panel：行=cluster，列=最近原型（line plot）。"""
    sel = select_domain_balanced_indices(meta, max_per_domain=max_per_domain, seed=seed)
    labels, centers, pca_coords = cluster_pca(emb[sel], k, seed)

    fig, axes = plt.subplots(k, proto_per_cluster, figsize=(2.0 * proto_per_cluster, 1.3 * k), squeeze=False)
    panel_info = []
    for row, cid in enumerate(range(k)):
        idx_local = np.where(labels == cid)[0]
        if len(idx_local) == 0:
            for col in range(proto_per_cluster):
                axes[row, col].axis("off")
            continue
        dist = np.linalg.norm(pca_coords[idx_local] - centers[cid], axis=1)
        order = np.argsort(dist)[:proto_per_cluster]
        chosen = sel[idx_local[order]]
        dom = Counter(meta[i]["macro_domain"] for i in sel[idx_local]).most_common(1)[0][0]
        panel_info.append({"cluster": f"C{cid + 1}", "size": int(len(idx_local)), "dominant_macro_domain": dom})
        for col in range(proto_per_cluster):
            ax = axes[row, col]
            if col >= len(chosen):
                ax.axis("off")
                continue
            item = int(chosen[col])
            ax.plot(robust_z(raw_patches[item]), lw=1.3, color="#1f2933")
            m = meta[item]
            ax.set_title(f"{m['macro_domain'][:12]} p{m['patch_index']}", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"C{cid + 1}", fontsize=9, rotation=0, labelpad=12, va="center")
    fig.suptitle(
        f"Chronos-Bolt {layer_name} — domain-balanced prototype examples "
        f"(k={k}, ≤{max_per_domain}/domain, center-nearest)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"output": str(out_path), "clusters": panel_info}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-per-dataset", type=int, default=120)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--card-layers", type=int, nargs="+", default=[0, 11])
    parser.add_argument("--prototype-layers", type=int, nargs="+", default=[0, 11])
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--proto-per-cluster", type=int, default=6)
    parser.add_argument("--max-per-domain", type=int, default=400)
    parser.add_argument("--tsne-perplexity", type=float, default=40.0, help="cluster-map t-SNE perplexity (viz only)")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[main-fig] sampling windows (per_dataset={args.windows_per_dataset})")
    windows, window_meta, _ = sample_windows(
        DATA_ROOT, context_len=args.context_len, windows_per_dataset=args.windows_per_dataset, seed=args.seed
    )
    windows_z = np.stack([robust_z(w) for w in windows]).astype(np.float32)
    print(f"[main-fig] {len(windows)} windows")

    layers = sorted(set(args.card_layers) | set(args.prototype_layers))
    print(f"[main-fig] extracting Bolt layers {layers} ...")
    pipe = load_bolt_pipeline()
    patch_len = int(pipe.model.chronos_config.input_patch_size)
    reps = extract_bolt_representations(
        windows_z, batch_size=args.batch_size, layers=layers, include_tokenizer=False,
        pipeline=pipe, keep_pipeline=False,
    )

    summary: dict[str, Any] = {"model": "chronos-bolt-base", "patch_len": patch_len,
                               "config": vars(args) | {"out": str(args.out)}, "cards": {}, "prototype_panel": {}}

    # 各层共享 flatten（raw patch / meta 不随层变）
    flat_cache: dict[int, tuple] = {}
    for L in layers:
        flat_cache[L] = flatten_patches(reps[f"layer_{L}"], windows_z, window_meta, patch_len)

    # domain-balanced 子集：避免 Traffic 等高频 domain 主导 cluster（否则每簇 dominant domain
    # 都被 Traffic 占满、shape 结构被淹没）。cards 与 prototype 共用同一平衡子集。
    meta0 = flat_cache[layers[0]][2]
    sel = select_domain_balanced_indices(meta0, max_per_domain=args.max_per_domain, seed=args.seed)
    summary["n_balanced_patches"] = int(len(sel))
    print(f"[main-fig] domain-balanced subset: {len(sel)} patches")

    layer_clusters: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for L in args.card_layers:
        emb, raw_patches, meta = flat_cache[L]
        emb_b = emb[sel]
        raw_b = raw_patches[sel]
        meta_b = [meta[i] for i in sel]
        labels, centers, pca_coords = cluster_pca(emb_b, args.k, args.seed)
        layer_clusters[f"layer_{L}"] = (labels, pca_coords)
        out = args.out / f"bolt_patch_stack_cards_layer{L}.png"
        print(f"[main-fig] layer_{L}: rendering raw-only cards -> {out.name}")
        summary["cards"][f"layer_{L}"] = render_raw_cards(
            f"layer_{L}", labels, centers, pca_coords, raw_b, meta_b, args.k, args.top_n, out
        )

    # 中间 plate：两个 depth 的聚类散点图（t-SNE view of PCA-space KMeans）
    cmap_out = args.out / "bolt_cluster_maps.png"
    print(f"[main-fig] rendering cluster maps ({list(layer_clusters)}) -> {cmap_out.name}")
    summary["cluster_maps"] = render_cluster_maps(
        layer_clusters, args.k, args.seed, args.tsne_perplexity, cmap_out
    )

    summary["prototype_panel"] = {}
    for Lp in args.prototype_layers:
        emb, raw_patches, meta = flat_cache[Lp]
        out = args.out / f"bolt_domain_balanced_prototype_panel_layer{Lp}.png"
        print(f"[main-fig] layer_{Lp}: rendering domain-balanced prototype panel -> {out.name}")
        summary["prototype_panel"][f"layer_{Lp}"] = render_prototype_panel(
            emb, raw_patches, meta, args.k, args.seed, args.proto_per_cluster,
            args.max_per_domain, out, f"layer_{Lp}"
        )

    (args.out / "bolt_main_figure_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"[main-fig] saved modular PNGs + summary -> {args.out}")


if __name__ == "__main__":
    main()
