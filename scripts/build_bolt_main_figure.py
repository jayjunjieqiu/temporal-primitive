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
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.chronos_bolt_backbone import extract_bolt_representations, load_bolt_pipeline  # noqa: E402
from scripts.explore_motif_taxonomy import LABELS as MOTIF_LABELS  # noqa: E402
from scripts.explore_motif_taxonomy import label_patch  # noqa: E402
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
    v0_labels: np.ndarray,
    k: int,
    seed: int,
    perplexity: float,
    out_path: Path,
) -> dict[str, Any]:
    """representation atlas（中间 plate）：每行一个 depth，左=模型 KMeans cluster，
    右=human motif taxonomy v0（shapelet-inspired probe，不是 ground truth）。

    KMeans 在 PCA(30) space 完成；t-SNE 只把同一 PCA 坐标投到 2D 做 visualization（不参与
    聚类）。左右两列共享同一套 t-SNE 点，只是着色不同，方便对比"模型 cluster vs 人工 motif"。
    """
    names = list(layer_clusters.keys())
    clu_cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")
    motif_cmap = plt.get_cmap("tab10")
    motif_color = {lab: motif_cmap(i % 10) for i, lab in enumerate(MOTIF_LABELS)}
    # mixed_uncertain 是 probe 的"兜底"类、占比大，淡化成浅灰画在底层，让真正 fired 的 motif 突出
    DIM_MOTIFS = {"mixed_uncertain": "#cfcfcf"}
    motif_color.update(DIM_MOTIFS)
    # 绘制顺序：先画淡化类（底层），再画其余（顶层）
    motif_draw_order = [m for m in MOTIF_LABELS if m in DIM_MOTIFS] + [m for m in MOTIF_LABELS if m not in DIM_MOTIFS]

    fig, axes = plt.subplots(len(names), 2, figsize=(11.0, 4.6 * len(names)), squeeze=False)
    info: dict[str, Any] = {}
    for row, name in enumerate(names):
        labels, pca_coords = layer_clusters[name]
        xy = TSNE(n_components=2, perplexity=perplexity, init="pca",
                  random_state=seed, max_iter=1000).fit_transform(pca_coords)

        ax = axes[row, 0]  # 模型 KMeans cluster
        for cid in range(k):
            m = labels == cid
            ax.scatter(xy[m, 0], xy[m, 1], s=5, color=clu_cmap(cid % (10 if k <= 10 else 20)), alpha=0.6)
        ax.set_title(f"Chronos-Bolt {name}\nmodel-derived KMeans clusters (k={k})", fontsize=10)

        ax = axes[row, 1]  # human motif taxonomy v0
        for lab in motif_draw_order:
            m = v0_labels == lab
            if not m.any():
                continue
            dim = lab in DIM_MOTIFS
            ax.scatter(xy[m, 0], xy[m, 1], s=4 if dim else 6,
                       color=motif_color[lab], alpha=0.12 if dim else 0.75,
                       zorder=1 if dim else 2)
        ax.set_title(f"Chronos-Bolt {name}\nhuman motif taxonomy v0 (probe)", fontsize=10)

        for col, ax in enumerate(axes[row]):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("t-SNE-1", fontsize=8)
            if col == 0:
                ax.set_ylabel("t-SNE-2", fontsize=8)
            for sp in ax.spines.values():
                sp.set_color("#1f2933")
                sp.set_linewidth(0.8)
        info[name] = {"n_points": int(len(labels))}

    clu_handles = [
        Line2D([0], [0], marker="o", ls="", ms=6, color=clu_cmap(c % (10 if k <= 10 else 20)), label=f"C{c + 1}")
        for c in range(k)
    ]
    motif_handles = [Line2D([0], [0], marker="o", ls="", ms=6, color=motif_color[l], label=l) for l in MOTIF_LABELS]
    leg_clu = fig.legend(handles=clu_handles, loc="upper left", bbox_to_anchor=(0.87, 0.88), fontsize=8,
                         title="model cluster (left col)", title_fontsize=9)
    fig.add_artist(leg_clu)
    fig.legend(handles=motif_handles, loc="upper left", bbox_to_anchor=(0.87, 0.55), fontsize=8,
               title="human motif v0 (right col)", title_fontsize=9)
    fig.suptitle(
        "Representation atlas across depth — model-derived clusters vs human motif taxonomy v0\n"
        "(KMeans in PCA space; t-SNE for visualization only; v0 = shapelet-inspired probe, not ground truth)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 0.86, 0.94))
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
    """cross-domain prototype example panel：行=cluster，列=**不同 macro domain** 里离 cluster
    中心最近的最佳代表（line plot）。强调同一 shape family 跨域复用，而不是集中在某几个域。
    """
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
        # 每个 macro domain 取该域里离中心最近的代表，再按距离排序取前 proto_per_cluster 个不同域
        best_by_dom: dict[str, tuple[float, int]] = {}
        for j, li in enumerate(idx_local):
            g = int(sel[li])
            dom = meta[g]["macro_domain"]
            if dom not in best_by_dom or dist[j] < best_by_dom[dom][0]:
                best_by_dom[dom] = (float(dist[j]), g)
        ranked = sorted(best_by_dom.items(), key=lambda kv: kv[1][0])[:proto_per_cluster]
        panel_info.append(
            {"cluster": f"C{cid + 1}", "size": int(len(idx_local)),
             "n_domains_present": len(best_by_dom), "domains_shown": [d for d, _ in ranked]}
        )
        for col in range(proto_per_cluster):
            ax = axes[row, col]
            if col >= len(ranked):
                ax.axis("off")
                continue
            dom, (_dist, item) = ranked[col]
            m = meta[item]
            ax.plot(robust_z(raw_patches[item]), lw=1.3, color="#1f2933")
            ax.set_title(f"{dom[:14]} p{m['patch_index']}", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"C{cid + 1}", fontsize=9, rotation=0, labelpad=12, va="center")
    fig.suptitle(
        f"Chronos-Bolt {layer_name} — cross-domain prototype examples "
        f"(k={k}, best per distinct macro-domain, center-nearest)",
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
    parser.add_argument("--k", type=int, default=8)
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

    # human motif taxonomy v0 标签（shapelet-inspired probe；只依赖 raw patch，与层无关）
    raw_b0 = flat_cache[layers[0]][1][sel]
    v0_labels = np.array([label_patch(raw_b0[i], patch_len).label for i in range(len(sel))])

    # 中间 plate：每行一个 depth，左=模型 KMeans cluster，右=human motif v0
    cmap_out = args.out / "bolt_cluster_maps.png"
    print(f"[main-fig] rendering cluster maps + human motif v0 ({list(layer_clusters)}) -> {cmap_out.name}")
    summary["cluster_maps"] = render_cluster_maps(
        layer_clusters, v0_labels, args.k, args.seed, args.tsne_perplexity, cmap_out
    )

    summary["prototype_panel"] = {}
    for Lp in args.prototype_layers:
        emb, raw_patches, meta = flat_cache[Lp]
        out = args.out / f"bolt_cross_domain_prototype_panel_layer{Lp}.png"
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
