"""额外探索：Chronos-Bolt main 图的聚类数 k sweep。

目的：看 k 比 6 多一点 / 少一点时，prototype panel 是否更有意义、cluster map 是否更清晰。
为效率：Bolt representation **只提取一次**，每层的 PCA(30) 与 t-SNE(2D) **各算一次**，然后
对每个 k 只重新跑 KMeans（在 PCA space）→ 重新上色 / 重排 prototype。

产出（outputs/figures/bolt_main_figure/k_sweep/）：每个 k 一组
  - cluster_maps_k{K}.png            : layer_0 / layer_11 两 depth 散点（t-SNE view）
  - prototype_layer0_k{K}.png        : layer_0 domain-balanced prototype panel
  - prototype_layer11_k{K}.png       : layer_11 版

从仓库根目录运行：
    .venv/bin/python scripts/sweep_bolt_cluster_k.py --k-list 4 5 6 8 10 12
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

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
from scripts.build_bolt_main_figure import flatten_patches  # noqa: E402
from scripts.chronos_bolt_backbone import extract_bolt_representations, load_bolt_pipeline  # noqa: E402
from scripts.run_second_pilot_discovery import (  # noqa: E402
    DATA_ROOT,
    robust_z,
    sample_windows,
    select_domain_balanced_indices,
)

OUTPUT_DIR = ROOT / "outputs" / "figures" / "bolt_main_figure" / "k_sweep"


def render_cluster_maps(layer_xy_labels: dict, k: int, out_path: Path) -> None:
    names = list(layer_xy_labels.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 5.0), squeeze=False)
    cmap = plt.get_cmap("tab20" if k > 10 else "tab10")
    for col, name in enumerate(names):
        ax = axes[0, col]
        xy, labels = layer_xy_labels[name]
        for cid in range(k):
            m = labels == cid
            ax.scatter(xy[m, 0], xy[m, 1], s=6, color=cmap(cid % (20 if k > 10 else 10)), alpha=0.6, label=f"C{cid + 1}")
        ax.set_title(f"Chronos-Bolt {name}  (k={k})\nKMeans (PCA space) · t-SNE view", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#1f2933")
            sp.set_linewidth(0.8)
        if col == len(names) - 1:
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, ncol=1 if k <= 8 else 2, title="cluster")
    fig.suptitle(f"Representation atlas — k={k}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_prototype_panel(labels, pca_coords, raw_b, meta_b, k, proto_per_cluster, out_path, layer_name) -> None:
    fig, axes = plt.subplots(k, proto_per_cluster, figsize=(2.0 * proto_per_cluster, 1.2 * k), squeeze=False)
    for row, cid in enumerate(range(k)):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            for col in range(proto_per_cluster):
                axes[row, col].axis("off")
            continue
        center = pca_coords[idx].mean(axis=0, keepdims=True)
        order = np.argsort(np.linalg.norm(pca_coords[idx] - center, axis=1))[:proto_per_cluster]
        chosen = idx[order]
        dom = Counter(meta_b[i]["macro_domain"] for i in idx).most_common(1)[0][0]
        for col in range(proto_per_cluster):
            ax = axes[row, col]
            if col >= len(chosen):
                ax.axis("off")
                continue
            item = int(chosen[col])
            ax.plot(robust_z(raw_b[item]), lw=1.3, color="#1f2933")
            m = meta_b[item]
            ax.set_title(f"{m['macro_domain'][:12]} p{m['patch_index']}", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"C{cid + 1}\n{dom[:10]}", fontsize=7, rotation=0, labelpad=20, va="center")
    fig.suptitle(f"Chronos-Bolt {layer_name} — domain-balanced prototypes (k={k})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-list", type=int, nargs="+", default=[4, 5, 6, 8, 10, 12])
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 11])
    parser.add_argument("--windows-per-dataset", type=int, default=120)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--max-per-domain", type=int, default=400)
    parser.add_argument("--proto-per-cluster", type=int, default=6)
    parser.add_argument("--tsne-perplexity", type=float, default=40.0)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] sampling windows (per_dataset={args.windows_per_dataset})")
    windows, window_meta, _ = sample_windows(
        DATA_ROOT, context_len=args.context_len, windows_per_dataset=args.windows_per_dataset, seed=args.seed
    )
    windows_z = np.stack([robust_z(w) for w in windows]).astype(np.float32)

    print(f"[sweep] extracting Bolt layers {args.layers} ...")
    pipe = load_bolt_pipeline()
    patch_len = int(pipe.model.chronos_config.input_patch_size)
    reps = extract_bolt_representations(
        windows_z, batch_size=128, layers=args.layers, include_tokenizer=False,
        pipeline=pipe, keep_pipeline=False,
    )

    # balanced subset（用 layer 0 的 meta，patch 对齐，跨层一致）
    _, _, meta0 = flatten_patches(reps[f"layer_{args.layers[0]}"], windows_z, window_meta, patch_len)
    sel = select_domain_balanced_indices(meta0, max_per_domain=args.max_per_domain, seed=args.seed)
    print(f"[sweep] balanced subset: {len(sel)} patches")

    # 每层：PCA(30) 与 t-SNE 各算一次
    per_layer = {}
    for L in args.layers:
        emb, raw, meta = flatten_patches(reps[f"layer_{L}"], windows_z, window_meta, patch_len)
        emb_b, raw_b, meta_b = emb[sel], raw[sel], [meta[i] for i in sel]
        Xp = PCA(n_components=30, random_state=args.seed).fit_transform(StandardScaler().fit_transform(emb_b))
        print(f"[sweep] layer_{L}: t-SNE (perplexity={args.tsne_perplexity}) ...")
        xy = TSNE(n_components=2, perplexity=args.tsne_perplexity, init="pca",
                  random_state=args.seed, max_iter=1000).fit_transform(Xp)
        per_layer[f"layer_{L}"] = {"Xp": Xp, "xy": xy, "raw_b": raw_b, "meta_b": meta_b}

    # 每个 k：重跑 KMeans，渲染 cluster maps + prototype panels
    for k in args.k_list:
        xy_labels = {}
        for name, d in per_layer.items():
            labels = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(d["Xp"])
            d[f"labels_k{k}"] = labels
            xy_labels[name] = (d["xy"], labels)
            render_prototype_panel(
                labels, d["Xp"], d["raw_b"], d["meta_b"], k, args.proto_per_cluster,
                args.out / f"prototype_{name}_k{k}.png", name,
            )
        render_cluster_maps(xy_labels, k, args.out / f"cluster_maps_k{k}.png")
        print(f"[sweep] k={k}: rendered cluster_maps + prototype panels")

    (args.out / "k_sweep_config.json").write_text(
        json.dumps({"k_list": args.k_list, "layers": args.layers, "n_balanced": int(len(sel)),
                    "seed": args.seed, "tsne_perplexity": args.tsne_perplexity}, indent=2),
        encoding="utf-8",
    )
    print(f"[sweep] done -> {args.out}")


if __name__ == "__main__":
    main()
