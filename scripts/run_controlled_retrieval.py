from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
CONTROL_DIR = OUTPUT_DIR / "controlled_retrieval"
FIGURE_DIR = OUTPUT_DIR / "figures" / "controlled_retrieval"

sys.path.insert(0, str(ROOT))
from scripts.run_second_pilot_discovery import (  # noqa: E402
    DATA_ROOT,
    MODEL_SPECS,
    extract_chronos_layers,
    extract_timesfm_layers,
    flatten_model_patches,
    robust_z,
    sample_windows,
    top_counts,
)


def parse_targets(text: str) -> dict[str, set[str]]:
    targets: dict[str, set[str]] = defaultdict(set)
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        model, layer = item.split(":", 1)
        targets[model].add(layer)
    return targets


def metadata_labels(metadata: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "dataset": [str(m["dataset"]) for m in metadata],
        "domain": [str(m["domain"]) for m in metadata],
        "frequency": [str(m.get("frequency_minutes")) for m in metadata],
        "patch_index": [str(m["patch_index"]) for m in metadata],
        "taxonomy_v0": [str(m["taxonomy_label"]) for m in metadata],
    }


def weighted_purity(cluster_ids: np.ndarray, labels: list[str]) -> float:
    total = len(labels)
    score = 0
    for cid in sorted(set(cluster_ids.tolist())):
        idx = np.where(cluster_ids == cid)[0]
        score += Counter(labels[i] for i in idx).most_common(1)[0][1]
    return float(score / total)


def fit_cluster_space(embeddings: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x = StandardScaler().fit_transform(embeddings)
    pca_dim = max(2, min(30, x.shape[0] - 1, x.shape[1]))
    pca = PCA(n_components=pca_dim, random_state=seed)
    x_pca = pca.fit_transform(x)
    k = min(16, max(6, int(round(np.sqrt(len(x_pca) / 35)))))
    cluster_ids = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(x_pca)
    info = {
        "num_patch_embeddings": int(len(x_pca)),
        "embedding_dim": int(embeddings.shape[1]),
        "pca_dim": int(pca_dim),
        "kmeans_k": int(k),
        "silhouette_pca_space": float(silhouette_score(x_pca, cluster_ids)),
        "pca2_explained_variance_ratio": pca.explained_variance_ratio_[:2].astype(float).tolist(),
    }
    return x_pca, cluster_ids, info


def cluster_profile(cluster_ids: np.ndarray, metadata: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = metadata_labels(metadata)
    global_profile = {
        "purity": {name: weighted_purity(cluster_ids, values) for name, values in labels.items()},
        "nmi": {name: float(normalized_mutual_info_score(values, cluster_ids)) for name, values in labels.items()},
    }
    clusters = []
    for cid in sorted(set(cluster_ids.tolist())):
        idx = np.where(cluster_ids == cid)[0]
        size = len(idx)
        clusters.append(
            {
                "cluster": int(cid),
                "size": int(size),
                "top_datasets": top_counts([labels["dataset"][i] for i in idx], 5),
                "top_domains": top_counts([labels["domain"][i] for i in idx], 5),
                "top_frequencies": top_counts([labels["frequency"][i] for i in idx], 5),
                "top_patch_indices": top_counts([labels["patch_index"][i] for i in idx], 5),
                "top_taxonomy_labels": top_counts([labels["taxonomy_v0"][i] for i in idx], 5),
                "top_domain_fraction": float(Counter(labels["domain"][i] for i in idx).most_common(1)[0][1] / size),
                "top_frequency_fraction": float(Counter(labels["frequency"][i] for i in idx).most_common(1)[0][1] / size),
                "top_patch_index_fraction": float(Counter(labels["patch_index"][i] for i in idx).most_common(1)[0][1] / size),
                "mean_raw_std": float(np.mean([metadata[i]["raw_std"] for i in idx])),
                "mean_raw_range": float(np.mean([metadata[i]["raw_range"] for i in idx])),
                "mean_abs_robust_slope": float(np.mean([abs(metadata[i]["robust_slope"]) for i in idx])),
                "mean_zero_ratio": float(np.mean([metadata[i]["zero_ratio"] for i in idx])),
            }
        )
    return global_profile, clusters


def select_candidate_clusters(
    clusters: list[dict[str, Any]],
    min_size: int,
    max_domain_fraction: float,
    max_frequency_fraction: float,
    max_patch_fraction: float,
) -> list[int]:
    selected = []
    for c in clusters:
        if c["size"] < min_size:
            continue
        if c["top_domain_fraction"] > max_domain_fraction:
            continue
        if c["top_frequency_fraction"] > max_frequency_fraction:
            continue
        if c["top_patch_index_fraction"] > max_patch_fraction:
            continue
        selected.append(int(c["cluster"]))
    return selected


def medoids_for_cluster(x_pca: np.ndarray, cluster_ids: np.ndarray, cid: int, n: int) -> list[int]:
    idx = np.where(cluster_ids == cid)[0]
    center = x_pca[idx].mean(axis=0, keepdims=True)
    order = np.argsort(np.linalg.norm(x_pca[idx] - center, axis=1))
    return [int(v) for v in idx[order[:n]]]


def make_constraint(name: str, query: dict[str, Any], cluster_ids: np.ndarray, query_idx: int) -> Callable[[int], bool]:
    if name == "unconstrained":
        return lambda i: i != query_idx
    if name == "same_patch_index":
        return lambda i: i != query_idx and str(query["patch_index"]) == str(query_idx_meta[i]["patch_index"])
    raise RuntimeError("constraint requires closure binding")


def patch_correlation(a: np.ndarray, b: np.ndarray) -> float:
    za = robust_z(a)
    zb = robust_z(b)
    denom = float(np.linalg.norm(za) * np.linalg.norm(zb))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(za, zb) / denom)


query_idx_meta: list[dict[str, Any]] = []


def constrained_neighbors(
    nn_order: np.ndarray,
    query_idx: int,
    metadata: list[dict[str, Any]],
    cluster_ids: np.ndarray,
    raw_patches: np.ndarray,
    constraint_name: str,
    k: int,
) -> dict[str, Any]:
    query = metadata[query_idx]

    def ok(i: int) -> bool:
        if i == query_idx:
            return False
        if constraint_name == "unconstrained":
            return True
        if constraint_name == "same_patch_index":
            return str(metadata[i]["patch_index"]) == str(query["patch_index"])
        if constraint_name == "same_frequency":
            return str(metadata[i].get("frequency_minutes")) == str(query.get("frequency_minutes"))
        if constraint_name == "cross_domain":
            return str(metadata[i]["domain"]) != str(query["domain"])
        if constraint_name == "same_patch_index_cross_domain":
            return str(metadata[i]["patch_index"]) == str(query["patch_index"]) and str(metadata[i]["domain"]) != str(query["domain"])
        if constraint_name == "same_frequency_cross_domain":
            return str(metadata[i].get("frequency_minutes")) == str(query.get("frequency_minutes")) and str(metadata[i]["domain"]) != str(query["domain"])
        raise ValueError(f"unknown constraint: {constraint_name}")

    selected = [int(i) for i in nn_order[query_idx] if ok(int(i))][:k]
    if not selected:
        return {"constraint": constraint_name, "num_neighbors": 0}
    q_patch = raw_patches[query_idx]
    return {
        "constraint": constraint_name,
        "num_neighbors": len(selected),
        "same_cluster_rate": float(np.mean(cluster_ids[selected] == cluster_ids[query_idx])),
        "mean_patch_correlation": float(np.mean([patch_correlation(q_patch, raw_patches[i]) for i in selected])),
        "top_domains": top_counts([metadata[i]["domain"] for i in selected], 4),
        "top_datasets": top_counts([metadata[i]["dataset"] for i in selected], 4),
        "top_taxonomy_labels": top_counts([metadata[i]["taxonomy_label"] for i in selected], 4),
        "top_patch_indices": top_counts([str(metadata[i]["patch_index"]) for i in selected], 4),
        "neighbors": [
            {
                "index": int(i),
                "cluster": int(cluster_ids[i]),
                "dataset": metadata[i]["dataset"],
                "domain": metadata[i]["domain"],
                "frequency": metadata[i].get("frequency_minutes"),
                "patch_index": int(metadata[i]["patch_index"]),
                "taxonomy_label": metadata[i]["taxonomy_label"],
                "global_start": int(metadata[i]["global_start"]),
                "node": int(metadata[i]["node"]),
            }
            for i in selected[:5]
        ],
    }


def inspect_candidates(
    target_name: str,
    x_pca: np.ndarray,
    cluster_ids: np.ndarray,
    metadata: list[dict[str, Any]],
    raw_patches: np.ndarray,
    windows: np.ndarray,
    candidate_clusters: list[int],
    medoids_per_cluster: int,
    k: int,
) -> dict[str, Any]:
    nn = NearestNeighbors(n_neighbors=min(len(x_pca), max(300, k * 80)), metric="euclidean")
    nn.fit(x_pca)
    nn_order = nn.kneighbors(x_pca, return_distance=False)
    constraints = [
        "unconstrained",
        "same_patch_index",
        "same_frequency",
        "cross_domain",
        "same_patch_index_cross_domain",
        "same_frequency_cross_domain",
    ]
    out_clusters = []
    for cid in candidate_clusters:
        medoids = medoids_for_cluster(x_pca, cluster_ids, cid, medoids_per_cluster)
        medoid_records = []
        for midx in medoids:
            medoid_records.append(
                {
                    "index": int(midx),
                    "cluster": int(cid),
                    "metadata": {
                        "dataset": metadata[midx]["dataset"],
                        "domain": metadata[midx]["domain"],
                        "frequency": metadata[midx].get("frequency_minutes"),
                        "patch_index": int(metadata[midx]["patch_index"]),
                        "taxonomy_label": metadata[midx]["taxonomy_label"],
                        "taxonomy_confidence": metadata[midx]["taxonomy_confidence"],
                        "global_start": int(metadata[midx]["global_start"]),
                        "node": int(metadata[midx]["node"]),
                    },
                    "retrieval": [
                        constrained_neighbors(nn_order, midx, metadata, cluster_ids, raw_patches, cname, k)
                        for cname in constraints
                    ],
                }
            )
        out_clusters.append({"cluster": int(cid), "medoids": medoid_records})
    save_context_panel(target_name, candidate_clusters, x_pca, cluster_ids, metadata, windows)
    save_retrieval_panel(target_name, out_clusters, raw_patches, metadata)
    return {"candidate_clusters": out_clusters}


def save_context_panel(
    target_name: str,
    candidate_clusters: list[int],
    x_pca: np.ndarray,
    cluster_ids: np.ndarray,
    metadata: list[dict[str, Any]],
    windows: np.ndarray,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        clusters = candidate_clusters[:8]
        if not clusters:
            return
        fig, axes = plt.subplots(len(clusters), 3, figsize=(9, 1.7 * len(clusters)), squeeze=False)
        for row, cid in enumerate(clusters):
            medoids = medoids_for_cluster(x_pca, cluster_ids, cid, 3)
            for col, midx in enumerate(medoids):
                ax = axes[row, col]
                meta = metadata[midx]
                context = robust_z(windows[int(meta["window_id"])])
                ax.plot(context, linewidth=1.0)
                ax.axvspan(meta["patch_start_in_window"], meta["patch_end_in_window"] - 1, alpha=0.18)
                ax.set_title(
                    f"c{cid} {meta['dataset']} p{meta['patch_index']} {meta['taxonomy_label']}",
                    fontsize=7,
                )
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{target_name}_context_medoids.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"context panel skipped for {target_name}: {type(exc).__name__}: {exc}")


def save_retrieval_panel(
    target_name: str,
    inspected: list[dict[str, Any]],
    raw_patches: np.ndarray,
    metadata: list[dict[str, Any]],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        rows = []
        for cluster in inspected[:6]:
            if not cluster["medoids"]:
                continue
            medoid = cluster["medoids"][0]
            cross = next((r for r in medoid["retrieval"] if r["constraint"] == "same_patch_index_cross_domain"), None)
            neighbor_ids = [n["index"] for n in (cross or {}).get("neighbors", [])[:5]]
            rows.append((cluster["cluster"], medoid["index"], neighbor_ids))
        if not rows:
            return
        fig, axes = plt.subplots(len(rows), 6, figsize=(12, 1.65 * len(rows)), squeeze=False)
        for row, (cid, qidx, nbrs) in enumerate(rows):
            ids = [qidx] + nbrs
            for col in range(6):
                ax = axes[row, col]
                if col >= len(ids):
                    ax.axis("off")
                    continue
                idx = ids[col]
                ax.plot(robust_z(raw_patches[idx]), linewidth=1.0)
                meta = metadata[idx]
                role = "Q" if col == 0 else f"N{col}"
                ax.set_title(f"{role} c{cid} {meta['dataset']} {meta['taxonomy_label']}", fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{target_name}_same_index_cross_domain_retrieval.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"retrieval panel skipped for {target_name}: {type(exc).__name__}: {exc}")


def analyze_target(
    model_key: str,
    layer_name: str,
    layer_embeddings: np.ndarray,
    windows: np.ndarray,
    window_metadata: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    embeddings, metadata, raw_patches = flatten_model_patches(model_key, layer_embeddings, windows, window_metadata)
    x_pca, cluster_ids, cluster_space = fit_cluster_space(embeddings, args.seed)
    global_profile, clusters = cluster_profile(cluster_ids, metadata)
    candidate_clusters = select_candidate_clusters(
        clusters,
        min_size=args.min_cluster_size,
        max_domain_fraction=args.max_domain_fraction,
        max_frequency_fraction=args.max_frequency_fraction,
        max_patch_fraction=args.max_patch_fraction,
    )
    target_name = f"controlled_{model_key}_{layer_name}"
    inspection = inspect_candidates(
        target_name,
        x_pca,
        cluster_ids,
        metadata,
        raw_patches,
        windows,
        candidate_clusters,
        args.medoids_per_cluster,
        args.retrieval_k,
    )
    return {
        "window_embedding_shape": list(layer_embeddings.shape),
        **cluster_space,
        **global_profile,
        "num_candidate_clusters": len(candidate_clusters),
        "candidate_cluster_ids": candidate_clusters,
        "clusters": clusters,
        **inspection,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-per-dataset", type=int, default=100)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument(
        "--targets",
        type=str,
        default="chronos_2_small:layer_5,chronos_2:layer_11,timesfm_2_5:layer_10,timesfm_2_5:layer_19",
    )
    parser.add_argument("--min-cluster-size", type=int, default=300)
    parser.add_argument("--max-domain-fraction", type=float, default=0.45)
    parser.add_argument("--max-frequency-fraction", type=float, default=0.60)
    parser.add_argument("--max-patch-fraction", type=float, default=0.40)
    parser.add_argument("--medoids-per-cluster", type=int, default=3)
    parser.add_argument("--retrieval-k", type=int, default=10)
    args = parser.parse_args()

    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    targets = parse_targets(args.targets)
    windows, window_metadata, dataset_summary = sample_windows(
        DATA_ROOT,
        context_len=args.context_len,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
    )

    results: dict[str, Any] = {
        "objective": "prototype inspection and controlled retrieval for model-derived temporal concepts",
        "data_root": str(DATA_ROOT),
        "excluded_datasets": ["BLAST"],
        "context_len": args.context_len,
        "windows_per_dataset": args.windows_per_dataset,
        "num_windows": int(len(windows)),
        "targets": args.targets,
        "candidate_selection": {
            "min_cluster_size": args.min_cluster_size,
            "max_domain_fraction": args.max_domain_fraction,
            "max_frequency_fraction": args.max_frequency_fraction,
            "max_patch_fraction": args.max_patch_fraction,
        },
        "dataset_summary": dataset_summary,
        "models": {},
    }

    for model_key, layer_names in targets.items():
        print(f"Extracting {model_key} for layers {sorted(layer_names)}...", flush=True)
        if MODEL_SPECS[model_key]["kind"] == "chronos":
            layer_outputs = extract_chronos_layers(model_key, windows, args.batch_size)
        else:
            layer_outputs = extract_timesfm_layers(windows, args.batch_size)
        model_result = {"patch_len": MODEL_SPECS[model_key]["patch_len"], "layers": {}}
        for layer_name in sorted(layer_names):
            print(f"Analyzing {model_key} {layer_name}...", flush=True)
            model_result["layers"][layer_name] = analyze_target(
                model_key,
                layer_name,
                layer_outputs[layer_name],
                windows,
                window_metadata,
                args,
            )
        results["models"][model_key] = model_result
        partial = CONTROL_DIR / "controlled_retrieval_summary.partial.json"
        partial.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    out = OUTPUT_DIR / "controlled_retrieval_summary.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (CONTROL_DIR / "controlled_retrieval_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "num_windows": results["num_windows"],
                "models": {
                    model: {
                        "patch_len": info["patch_len"],
                        "layers": {
                            layer: {
                                "num_candidate_clusters": layer_info["num_candidate_clusters"],
                                "candidate_cluster_ids": layer_info["candidate_cluster_ids"],
                            }
                            for layer, layer_info in info["layers"].items()
                        },
                    }
                    for model, info in results["models"].items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
