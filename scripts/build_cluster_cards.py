from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "cluster_cards"
CARD_DIR = OUT_DIR / "cards"
RETRIEVAL_DIR = OUT_DIR / "retrieval"
SUMMARY_PATH = OUT_DIR / "cluster_card_summary.json"

sys.path.insert(0, str(ROOT))
from scripts.run_second_pilot_discovery import (  # noqa: E402
    DATA_ROOT,
    MODEL_SPECS,
    extract_chronos_layers,
    extract_timesfm_layers,
    flatten_model_patches,
    robust_z,
    sample_windows,
    select_domain_balanced_indices,
    top_counts,
)


TARGETS = [
    {
        "model": "chronos_2",
        "layer": "layer_11",
        "cluster": 1,
        "temporary_name": "transition_like_cross_domain",
        "reason": "large cross-domain cluster; mixed_uncertain/level_shift/trend dominant; low patch-index confounding",
    },
    {
        "model": "chronos_2",
        "layer": "layer_11",
        "cluster": 6,
        "temporary_name": "high_variation_transition_like",
        "reason": "large enough; multiple domains/frequencies; strong transition-like raw slope",
    },
    {
        "model": "chronos_2",
        "layer": "layer_6",
        "cluster": 2,
        "temporary_name": "midlayer_transition_like",
        "reason": "low top-domain and top-frequency dominance; useful middle-layer comparison",
    },
    {
        "model": "chronos_2",
        "layer": "layer_6",
        "cluster": 7,
        "temporary_name": "gaussian_noise_artifact",
        "reason": "dominated by simulated Gaussian data; negative control artifact",
    },
    {
        "model": "timesfm_2_5",
        "layer": "layer_10",
        "cluster": 8,
        "temporary_name": "timesfm_transition_like",
        "reason": "transition-like labels; moderate domain/frequency spread; TimesFM concept candidate",
    },
    {
        "model": "timesfm_2_5",
        "layer": "layer_10",
        "cluster": 5,
        "temporary_name": "timesfm_smooth_transition_like",
        "reason": "mixed transition labels; visually smooth in pilot panels; not single-domain dominated",
    },
    {
        "model": "timesfm_2_5",
        "layer": "layer_10",
        "cluster": 4,
        "temporary_name": "timesfm_patch_position_artifact",
        "reason": "patch-index purity equals 1.0; negative control for position artifact",
    },
]


def metric_labels(metadata: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "dataset": [str(m["dataset"]) for m in metadata],
        "domain": [str(m["domain"]) for m in metadata],
        "frequency": [str(m.get("frequency_minutes")) for m in metadata],
        "patch_index": [str(m["patch_index"]) for m in metadata],
        "taxonomy_v0": [str(m["taxonomy_label"]) for m in metadata],
    }


def fit_domain_balanced_clusters(
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    raw_patches: np.ndarray,
    max_per_domain: int,
    seed: int,
) -> dict[str, Any]:
    idx = select_domain_balanced_indices(metadata, max_per_domain, seed)
    emb = embeddings[idx]
    meta = [metadata[i] for i in idx]
    patches = raw_patches[idx]
    x = StandardScaler().fit_transform(emb)
    pca_dim = max(2, min(30, x.shape[0] - 1, x.shape[1]))
    pca = PCA(n_components=pca_dim, random_state=seed)
    x_pca = pca.fit_transform(x)
    k = min(16, max(6, int(round(math.sqrt(len(meta) / 35)))))
    cluster_ids = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(x_pca)
    labels = metric_labels(meta)
    return {
        "indices": idx,
        "embeddings": emb,
        "metadata": meta,
        "raw_patches": patches,
        "x_pca": x_pca,
        "cluster_ids": cluster_ids,
        "k": k,
        "pca_dim": pca_dim,
        "global_nmi": {name: float(normalized_mutual_info_score(values, cluster_ids)) for name, values in labels.items()},
    }


def distribution(values: list[Any], n: int = 8) -> list[dict[str, Any]]:
    return top_counts([str(v) for v in values], n=n)


def cluster_stats(meta: list[dict[str, Any]], cluster_ids: np.ndarray, cluster: int) -> dict[str, Any]:
    idx = np.where(cluster_ids == cluster)[0]
    if len(idx) == 0:
        return {
            "size": 0,
            "top_datasets": [],
            "top_domains": [],
            "top_frequencies": [],
            "top_patch_indices": [],
            "top_taxonomy_labels": [],
            "raw_stats": {
                "std_mean": None,
                "std_median": None,
                "range_mean": None,
                "abs_slope_mean": None,
                "zero_ratio_mean": None,
            },
            "top_domain_share": 0.0,
            "top_frequency_share": 0.0,
            "top_patch_index_share": 0.0,
            "top_taxonomy_v0_share": 0.0,
        }
    labels = metric_labels(meta)
    raw_std = [meta[i]["raw_std"] for i in idx]
    raw_range = [meta[i]["raw_range"] for i in idx]
    slope = [abs(meta[i]["robust_slope"]) for i in idx]
    zero = [meta[i]["zero_ratio"] for i in idx]
    out = {
        "size": int(len(idx)),
        "top_datasets": distribution([labels["dataset"][i] for i in idx]),
        "top_domains": distribution([labels["domain"][i] for i in idx]),
        "top_frequencies": distribution([labels["frequency"][i] for i in idx]),
        "top_patch_indices": distribution([labels["patch_index"][i] for i in idx]),
        "top_taxonomy_labels": distribution([labels["taxonomy_v0"][i] for i in idx]),
        "raw_stats": {
            "std_mean": float(np.mean(raw_std)),
            "std_median": float(np.median(raw_std)),
            "range_mean": float(np.mean(raw_range)),
            "abs_slope_mean": float(np.mean(slope)),
            "zero_ratio_mean": float(np.mean(zero)),
        },
    }
    for key, top_key in [
        ("domain", "top_domains"),
        ("frequency", "top_frequencies"),
        ("patch_index", "top_patch_indices"),
        ("taxonomy_v0", "top_taxonomy_labels"),
    ]:
        out[f"top_{key}_share"] = out[top_key][0]["count"] / len(idx) if len(idx) else 0.0
    return out


def confounder_warnings(stats: dict[str, Any]) -> list[str]:
    warnings = []
    if stats["top_domain_share"] >= 0.60:
        warnings.append("single-domain dominated")
    if stats["top_frequency_share"] >= 0.70:
        warnings.append("single-frequency dominated")
    if stats["top_patch_index_share"] >= 0.55:
        warnings.append("patch-position dominated")
    if stats["raw_stats"]["zero_ratio_mean"] >= 0.30:
        warnings.append("zero-heavy / low-activity artifact risk")
    if stats["raw_stats"]["std_median"] <= 1e-6:
        warnings.append("near-flat raw patches")
    return warnings or ["no severe single-factor warning"]


def medoid_indices(x: np.ndarray, cluster_ids: np.ndarray, cluster: int, n: int) -> np.ndarray:
    idx = np.where(cluster_ids == cluster)[0]
    center = x[idx].mean(axis=0, keepdims=True)
    order = np.argsort(np.linalg.norm(x[idx] - center, axis=1))
    return idx[order[: min(n, len(order))]]


def cluster_member_shape_summary(patches: np.ndarray, member_idx: np.ndarray) -> dict[str, Any]:
    z = np.asarray([robust_z(p) for p in patches[member_idx]])
    return {
        "mean_curve": np.mean(z, axis=0).astype(float).tolist(),
        "std_curve": np.std(z, axis=0).astype(float).tolist(),
        "q10_curve": np.quantile(z, 0.10, axis=0).astype(float).tolist(),
        "q90_curve": np.quantile(z, 0.90, axis=0).astype(float).tolist(),
    }


def retrieval_for_query(
    query_idx: int,
    x: np.ndarray,
    meta: list[dict[str, Any]],
    patches: np.ndarray,
    conditions: dict[str, np.ndarray],
    top_k: int,
) -> dict[str, Any]:
    result = {}
    q = x[query_idx : query_idx + 1]
    q_patch = robust_z(patches[query_idx])
    q_meta = meta[query_idx]
    for name, mask in conditions.items():
        allowed = np.where(mask)[0]
        allowed = allowed[allowed != query_idx]
        if len(allowed) == 0:
            result[name] = {"status": "empty"}
            continue
        dists = np.linalg.norm(x[allowed] - q, axis=1)
        order = np.argsort(dists)[: min(top_k, len(dists))]
        nn = allowed[order]
        z_nn = np.asarray([robust_z(patches[i]) for i in nn])
        shape_corr = []
        for patch in z_nn:
            if np.std(patch) < 1e-8 or np.std(q_patch) < 1e-8:
                shape_corr.append(0.0)
            else:
                shape_corr.append(float(np.corrcoef(q_patch, patch)[0, 1]))
        result[name] = {
            "status": "ok",
            "neighbor_indices": nn.astype(int).tolist(),
            "mean_embedding_distance": float(np.mean(dists[order])),
            "mean_shape_correlation": float(np.mean(shape_corr)),
            "taxonomy_v0_agreement": float(np.mean([meta[i]["taxonomy_label"] == q_meta["taxonomy_label"] for i in nn])),
            "domain_diversity": int(len(set(meta[i]["domain"] for i in nn))),
            "frequency_diversity": int(len(set(str(meta[i].get("frequency_minutes")) for i in nn))),
            "patch_index_diversity": int(len(set(str(meta[i]["patch_index"]) for i in nn))),
            "neighbors": [
                {
                    "dataset": meta[i]["dataset"],
                    "domain": meta[i]["domain"],
                    "frequency": meta[i].get("frequency_minutes"),
                    "patch_index": int(meta[i]["patch_index"]),
                    "taxonomy_label": meta[i]["taxonomy_label"],
                    "shape_correlation": float(shape_corr[j]),
                }
                for j, i in enumerate(nn)
            ],
        }
    return result


def build_conditions(query_idx: int, meta: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    q = meta[query_idx]
    domain = q["domain"]
    freq = str(q.get("frequency_minutes"))
    patch_index = str(q["patch_index"])
    return {
        "unrestricted": np.ones(len(meta), dtype=bool),
        "same_patch_index": np.asarray([str(m["patch_index"]) == patch_index for m in meta], dtype=bool),
        "same_frequency": np.asarray([str(m.get("frequency_minutes")) == freq for m in meta], dtype=bool),
        "cross_domain": np.asarray([m["domain"] != domain for m in meta], dtype=bool),
        "same_domain": np.asarray([m["domain"] == domain for m in meta], dtype=bool),
    }


def plot_cluster_card(
    out_path: Path,
    target: dict[str, Any],
    stats: dict[str, Any],
    warnings: list[str],
    x_pca: np.ndarray,
    cluster_ids: np.ndarray,
    meta: list[dict[str, Any]],
    patches: np.ndarray,
    medoids: np.ndarray,
    shape_summary: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 4, height_ratios=[0.7, 1.1, 1.1, 1.2])
    title = (
        f"{target['model']} {target['layer']} c{target['cluster']} | "
        f"{target['temporary_name']} | n={stats['size']}"
    )
    fig.suptitle(title, fontsize=14)

    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis("off")
    text = [
        f"reason: {target['reason']}",
        f"warnings: {', '.join(warnings)}",
        f"top domains: {stats['top_domains'][:3]}",
        f"top frequencies: {stats['top_frequencies'][:3]}",
        f"top patch indices: {stats['top_patch_indices'][:3]}",
        f"top taxonomy-v0: {stats['top_taxonomy_labels'][:3]}",
        f"raw stats: {stats['raw_stats']}",
    ]
    ax_text.text(0.01, 0.95, "\n".join(text), va="top", fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    mask = cluster_ids == target["cluster"]
    ax.scatter(x_pca[:, 0], x_pca[:, 1], s=2, alpha=0.15, color="gray")
    ax.scatter(x_pca[mask, 0], x_pca[mask, 1], s=4, alpha=0.6, color="tab:blue")
    ax.scatter(x_pca[medoids, 0], x_pca[medoids, 1], s=35, color="tab:red", marker="x")
    ax.set_title("PCA cluster location")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1:])
    mean = np.asarray(shape_summary["mean_curve"])
    std = np.asarray(shape_summary["std_curve"])
    q10 = np.asarray(shape_summary["q10_curve"])
    q90 = np.asarray(shape_summary["q90_curve"])
    t = np.arange(len(mean))
    ax.plot(t, mean, color="tab:blue", label="mean z-patch")
    ax.fill_between(t, q10, q90, color="tab:blue", alpha=0.18, label="q10-q90")
    ax.fill_between(t, mean - std, mean + std, color="tab:orange", alpha=0.12, label="+/- std")
    ax.set_title("Cluster z-normalized shape summary")
    ax.legend(fontsize=8)

    for j, idx in enumerate(medoids[:4]):
        ax = fig.add_subplot(gs[2, j])
        patch = patches[idx]
        ax.plot(patch, color="tab:gray", linewidth=1.0, label="raw")
        ax2 = ax.twinx()
        ax2.plot(robust_z(patch), color="tab:blue", linewidth=1.2, label="z")
        m = meta[idx]
        ax.set_title(f"medoid {j}: {m['dataset']} p{m['patch_index']} {m['taxonomy_label']}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax2.set_yticks([])

    for j, idx in enumerate(medoids[:4]):
        ax = fig.add_subplot(gs[3, j])
        m = meta[idx]
        context = np.asarray(m["_context"], dtype=float)
        z_context = robust_z(context)
        ax.plot(z_context, color="tab:gray", linewidth=1.0)
        ax.axvspan(m["patch_start_in_window"], m["patch_end_in_window"] - 1, color="tab:red", alpha=0.18)
        ax.set_title(f"context: {m['dataset']} node={m['node']} start={m['start']}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_retrieval(
    out_path: Path,
    target: dict[str, Any],
    query_idx: int,
    retrieval: dict[str, Any],
    meta: list[dict[str, Any]],
    patches: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    conditions = list(retrieval.keys())
    top_k_plot = 5
    fig, axes = plt.subplots(len(conditions), top_k_plot + 1, figsize=(15, 2.2 * len(conditions)), squeeze=False)
    q_patch = robust_z(patches[query_idx])
    q_meta = meta[query_idx]
    for row, condition in enumerate(conditions):
        axes[row, 0].plot(q_patch, color="tab:red", linewidth=1.3)
        axes[row, 0].set_title(f"query\n{q_meta['dataset']} p{q_meta['patch_index']} {q_meta['taxonomy_label']}", fontsize=7)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        info = retrieval[condition]
        if info["status"] != "ok":
            axes[row, 1].text(0.1, 0.5, condition + "\nempty")
            for col in range(1, top_k_plot + 1):
                axes[row, col].axis("off")
            continue
        nn = info["neighbor_indices"][:top_k_plot]
        for col, idx in enumerate(nn, start=1):
            axes[row, col].plot(robust_z(patches[idx]), color="tab:blue", linewidth=1.1)
            m = meta[idx]
            axes[row, col].set_title(
                f"{condition}\n{m['dataset']} p{m['patch_index']} {m['taxonomy_label']}\nr={info['neighbors'][col-1]['shape_correlation']:.2f}",
                fontsize=6,
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(f"Controlled retrieval: {target['model']} {target['layer']} c{target['cluster']}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def attach_context_to_metadata(meta: list[dict[str, Any]], windows: np.ndarray) -> None:
    for m in meta:
        m["_context"] = windows[int(m["window_id"])].astype(float).tolist()


def load_model_layer_bundle(
    model_key: str,
    windows: np.ndarray,
    window_meta: list[dict[str, Any]],
    layer_name: str,
    batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    layer_idx = int(layer_name.split("_")[1])
    original_layers = MODEL_SPECS[model_key]["layers"]
    MODEL_SPECS[model_key]["layers"] = [layer_idx]
    try:
        if MODEL_SPECS[model_key]["kind"] == "chronos":
            outputs = extract_chronos_layers(model_key, windows, batch_size)
        else:
            outputs = extract_timesfm_layers(windows, batch_size)
    finally:
        MODEL_SPECS[model_key]["layers"] = original_layers
    embeddings, meta, patches = flatten_model_patches(model_key, outputs[layer_name], windows, window_meta)
    attach_context_to_metadata(meta, windows)
    return embeddings, meta, patches


def evaluate_target(
    target: dict[str, Any],
    bundle: dict[str, Any],
    second_pilot_layer: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    meta = bundle["metadata"]
    patches = bundle["raw_patches"]
    x_pca = bundle["x_pca"]
    cluster_ids = bundle["cluster_ids"]
    stats = cluster_stats(meta, cluster_ids, int(target["cluster"]))
    if stats["size"] == 0:
        return {
            "target": target,
            "status": "skipped_empty_cluster",
            "note": "The requested cluster id is absent in this run. This can happen in smoke tests with a different sample size.",
        }
    warnings = confounder_warnings(stats)
    member_idx = np.where(cluster_ids == target["cluster"])[0]
    medoids = medoid_indices(x_pca, cluster_ids, int(target["cluster"]), n=4)
    shape_summary = cluster_member_shape_summary(patches, member_idx)
    query_idx = int(medoids[0])
    retrieval = retrieval_for_query(
        query_idx=query_idx,
        x=x_pca,
        meta=meta,
        patches=patches,
        conditions=build_conditions(query_idx, meta),
        top_k=top_k,
    )

    stem = f"{target['model']}_{target['layer']}_c{target['cluster']}_{target['temporary_name']}"
    card_path = CARD_DIR / f"{stem}.png"
    retrieval_path = RETRIEVAL_DIR / f"{stem}_retrieval.png"
    plot_cluster_card(card_path, target, stats, warnings, x_pca, cluster_ids, meta, patches, medoids, shape_summary)
    plot_retrieval(retrieval_path, target, query_idx, retrieval, meta, patches)

    # The script reclusters deterministically; record the original second-pilot metrics too.
    return {
        "target": target,
        "card_path": str(card_path.relative_to(ROOT)),
        "retrieval_path": str(retrieval_path.relative_to(ROOT)),
        "cluster_stats": stats,
        "confounder_warnings": warnings,
        "medoids": [
            {
                "dataset": meta[i]["dataset"],
                "domain": meta[i]["domain"],
                "frequency": meta[i].get("frequency_minutes"),
                "patch_index": int(meta[i]["patch_index"]),
                "taxonomy_label": meta[i]["taxonomy_label"],
                "global_start": int(meta[i]["global_start"]),
                "node": int(meta[i]["node"]),
            }
            for i in medoids
        ],
        "query": {
            "dataset": meta[query_idx]["dataset"],
            "domain": meta[query_idx]["domain"],
            "frequency": meta[query_idx].get("frequency_minutes"),
            "patch_index": int(meta[query_idx]["patch_index"]),
            "taxonomy_label": meta[query_idx]["taxonomy_label"],
        },
        "retrieval": retrieval,
        "second_pilot_layer_metrics": {
            "silhouette": second_pilot_layer["silhouette_pca_space"],
            "stability": second_pilot_layer["kmeans_vs_agglomerative_nmi"],
            "nmi": second_pilot_layer["nmi"],
            "purity": second_pilot_layer["purity"],
            "kmeans_k": second_pilot_layer["kmeans_k"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-per-dataset", type=int, default=100)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--domain-balanced-patches", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)

    second_pilot = json.loads((ROOT / "outputs" / "second_pilot_discovery_summary.json").read_text(encoding="utf-8"))
    windows, window_meta, dataset_summary = sample_windows(
        DATA_ROOT,
        context_len=args.context_len,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
    )

    by_model_layer: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for target in TARGETS:
        by_model_layer.setdefault((target["model"], target["layer"]), []).append(target)

    summary: dict[str, Any] = {
        "objective": "cluster-card and controlled-retrieval analysis",
        "source_second_pilot": "outputs/second_pilot_discovery_summary.json",
        "num_windows": int(len(windows)),
        "windows_per_dataset": args.windows_per_dataset,
        "context_len": args.context_len,
        "seed": args.seed,
        "domain_balanced_patches_per_domain": args.domain_balanced_patches,
        "dataset_summary": dataset_summary,
        "targets": [],
    }

    for (model_key, layer_name), targets in by_model_layer.items():
        print(f"Preparing {model_key} {layer_name}...")
        embeddings, meta, patches = load_model_layer_bundle(model_key, windows, window_meta, layer_name, args.batch_size)
        bundle = fit_domain_balanced_clusters(
            embeddings,
            meta,
            patches,
            max_per_domain=args.domain_balanced_patches,
            seed=args.seed,
        )
        for target in targets:
            layer_metrics = second_pilot["models"][model_key]["layers"][layer_name]["domain_balanced"]
            summary["targets"].append(evaluate_target(target, bundle, layer_metrics, top_k=args.top_k))

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "num_windows": summary["num_windows"],
                "num_targets": len(summary["targets"]),
                "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
