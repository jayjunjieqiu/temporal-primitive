from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "taxonomy_v1_pilot"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_PATH = OUT_DIR / "taxonomy_v1_pilot_summary.json"

sys.path.insert(0, str(ROOT))
from scripts.build_cluster_cards import (  # noqa: E402
    TARGETS,
    fit_domain_balanced_clusters,
    load_model_layer_bundle,
)
from scripts.run_cluster_level_validation import (  # noqa: E402
    compact_retrieval_for_json,
    query_metadata,
    retrieve_for_query,
    summarize_query_set,
)
from scripts.run_second_pilot_discovery import DATA_ROOT, robust_z, sample_windows  # noqa: E402


PARENT_TARGETS = [
    {
        "model": "timesfm_2_5",
        "layer": "layer_10",
        "cluster": 8,
        "temporary_name": "timesfm_transition_like",
        "pilot_role": "strong_candidate",
        "n_subclusters": 3,
    },
    {
        "model": "timesfm_2_5",
        "layer": "layer_10",
        "cluster": 5,
        "temporary_name": "timesfm_smooth_transition_like",
        "pilot_role": "heterogeneous_candidate",
        "n_subclusters": 4,
    },
    {
        "model": "chronos_2",
        "layer": "layer_11",
        "cluster": 6,
        "temporary_name": "high_variation_transition_like",
        "pilot_role": "chronos_split_control",
        "n_subclusters": 4,
    },
]


def _freq(value: Any) -> str:
    return str(value)


def local_kmeans(embeddings: np.ndarray, member_idx: np.ndarray, k: int, seed: int) -> dict[str, Any]:
    x = StandardScaler().fit_transform(embeddings[member_idx])
    n_components = max(2, min(12, x.shape[0] - 1, x.shape[1]))
    local_x = PCA(n_components=n_components, random_state=seed).fit_transform(x)
    labels = KMeans(n_clusters=min(k, len(member_idx)), random_state=seed, n_init=30).fit_predict(local_x)
    return {"local_x": local_x, "labels": labels.astype(int), "n_components": int(n_components)}


def select_diverse_queries_from_indices(
    x: np.ndarray,
    member_idx: np.ndarray,
    meta: list[dict[str, Any]],
    n_queries: int,
) -> list[int]:
    if len(member_idx) == 0:
        return []
    center = x[member_idx].mean(axis=0, keepdims=True)
    distances = np.linalg.norm(x[member_idx] - center, axis=1)
    ordered = member_idx[np.argsort(distances)]
    selected: list[int] = []
    domain_counts: Counter[str] = Counter()
    patch_counts: Counter[str] = Counter()
    freq_counts: Counter[str] = Counter()
    max_domain = max(2, int(np.ceil(n_queries / 4)))
    max_patch = max(2, int(np.ceil(n_queries / 4)))
    max_freq = max(2, int(np.ceil(n_queries / 4)))
    for raw_idx in ordered:
        idx = int(raw_idx)
        domain = str(meta[idx]["domain"])
        patch = str(meta[idx]["patch_index"])
        freq = _freq(meta[idx].get("frequency_minutes"))
        if domain_counts[domain] >= max_domain or patch_counts[patch] >= max_patch or freq_counts[freq] >= max_freq:
            continue
        selected.append(idx)
        domain_counts[domain] += 1
        patch_counts[patch] += 1
        freq_counts[freq] += 1
        if len(selected) >= min(n_queries, len(member_idx)):
            return selected
    for raw_idx in ordered:
        idx = int(raw_idx)
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= min(n_queries, len(member_idx)):
            break
    return selected


def patch_shape_descriptors(patches: np.ndarray, member_idx: np.ndarray) -> dict[str, Any]:
    z = np.asarray([robust_z(patches[i]) for i in member_idx])
    mean = np.mean(z, axis=0)
    t = np.linspace(-1.0, 1.0, len(mean))
    slope = float(np.polyfit(t, mean, 1)[0])
    endpoint_delta = float(mean[-1] - mean[0])
    roughness = float(np.mean(np.abs(np.diff(mean, n=2)))) if len(mean) > 2 else 0.0
    return {
        "mean_curve": mean.astype(float).tolist(),
        "std_curve": np.std(z, axis=0).astype(float).tolist(),
        "slope": slope,
        "endpoint_delta": endpoint_delta,
        "roughness": roughness,
        "mean_abs_patch_slope": float(np.mean([abs(float(np.polyfit(np.arange(len(robust_z(patches[i]))), robust_z(patches[i]), 1)[0])) for i in member_idx])),
    }


def distribution(values: list[Any], n: int = 6) -> list[dict[str, Any]]:
    return [{"value": str(k), "count": int(v)} for k, v in Counter([str(v) for v in values]).most_common(n)]


def member_stats(meta: list[dict[str, Any]], patches: np.ndarray, member_idx: np.ndarray) -> dict[str, Any]:
    desc = patch_shape_descriptors(patches, member_idx)
    return {
        "size": int(len(member_idx)),
        "top_domains": distribution([meta[i]["domain"] for i in member_idx]),
        "top_frequencies": distribution([meta[i].get("frequency_minutes") for i in member_idx]),
        "top_patch_indices": distribution([meta[i]["patch_index"] for i in member_idx]),
        "top_taxonomy_v0": distribution([meta[i]["taxonomy_label"] for i in member_idx]),
        "shape": {
            "slope": desc["slope"],
            "endpoint_delta": desc["endpoint_delta"],
            "roughness": desc["roughness"],
            "mean_abs_patch_slope": desc["mean_abs_patch_slope"],
        },
        "top_domain_share": float(Counter([meta[i]["domain"] for i in member_idx]).most_common(1)[0][1] / len(member_idx)),
        "top_frequency_share": float(Counter([_freq(meta[i].get("frequency_minutes")) for i in member_idx]).most_common(1)[0][1] / len(member_idx)),
        "top_patch_index_share": float(Counter([str(meta[i]["patch_index"]) for i in member_idx]).most_common(1)[0][1] / len(member_idx)),
    }


def validation_tags(condition_summary: dict[str, Any], stats: dict[str, Any]) -> list[str]:
    tags = []
    cross = condition_summary.get("cross_domain", {})
    cross_patch = condition_summary.get("cross_domain_same_patch_index", {})
    cross_freq = condition_summary.get("cross_domain_same_frequency", {})
    if cross.get("mean_shape_correlation_mean", 0.0) >= 0.25:
        tags.append("cross_domain_shape_survives")
    if cross_patch.get("mean_shape_correlation_mean", 0.0) >= 0.25:
        tags.append("cross_domain_same_patch_survives")
    if cross_freq.get("mean_shape_correlation_mean", 0.0) >= 0.25:
        tags.append("cross_domain_same_frequency_survives")
    if stats["top_patch_index_share"] >= 0.75:
        tags.append("patch_position_risk")
    if stats["top_domain_share"] >= 0.65:
        tags.append("domain_risk")
    if stats["top_frequency_share"] >= 0.75:
        tags.append("frequency_risk")
    return tags or ["no_strong_survival_signal"]


def validate_member_set(
    member_idx: np.ndarray,
    x: np.ndarray,
    meta: list[dict[str, Any]],
    patches: np.ndarray,
    n_queries: int,
    top_k: int,
) -> dict[str, Any]:
    query_indices = select_diverse_queries_from_indices(x, member_idx, meta, n_queries)
    per_query_full = []
    per_query_json = []
    for q_idx in query_indices:
        retrieval = retrieve_for_query(q_idx, x, meta, patches, top_k)
        row_full = {"query": query_metadata([q_idx], meta)[0], "retrieval": retrieval}
        per_query_full.append(row_full)
        per_query_json.append({"query": row_full["query"], "retrieval": compact_retrieval_for_json(retrieval)})
    condition_summary = summarize_query_set(query_indices, per_query_full, meta) if query_indices else {}
    return {
        "num_queries": int(len(query_indices)),
        "queries": query_metadata(query_indices, meta),
        "query_distribution": {
            "domains": dict(Counter(meta[i]["domain"] for i in query_indices)),
            "frequencies": dict(Counter(_freq(meta[i].get("frequency_minutes")) for i in query_indices)),
            "patch_indices": dict(Counter(str(meta[i]["patch_index"]) for i in query_indices)),
            "taxonomy_v0": dict(Counter(meta[i]["taxonomy_label"] for i in query_indices)),
        },
        "condition_summary": condition_summary,
        "per_query": per_query_json,
    }


def plot_parent_split(
    out_path: Path,
    title: str,
    local_x: np.ndarray,
    labels: np.ndarray,
    sub_results: list[dict[str, Any]],
    patches: np.ndarray,
    parent_member_idx: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    k = len(sub_results)
    fig = plt.figure(figsize=(15, 3.8 + 2.2 * k))
    gs = fig.add_gridspec(k + 1, 3, height_ratios=[1.25] + [1.0] * k)
    ax = fig.add_subplot(gs[0, 0])
    for sub_id in range(k):
        mask = labels == sub_id
        ax.scatter(local_x[mask, 0], local_x[mask, 1], s=8, alpha=0.7, label=f"s{sub_id}")
    ax.set_title("local PCA split")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1:])
    for sub in sub_results:
        mean = np.asarray(sub["shape_summary"]["mean_curve"])
        ax.plot(mean, linewidth=1.5, label=f"s{sub['subcluster']} n={sub['stats']['size']}")
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_title("subcluster mean z-patch")
    ax.legend(fontsize=8)

    for row, sub in enumerate(sub_results, start=1):
        member_idx = np.asarray(sub["member_indices"], dtype=int)
        desc = sub["shape_summary"]
        mean = np.asarray(desc["mean_curve"])
        std = np.asarray(desc["std_curve"])
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(mean, color="tab:blue")
        ax.fill_between(np.arange(len(mean)), mean - std, mean + std, color="tab:blue", alpha=0.15)
        ax.set_title(
            f"s{sub['subcluster']} mean | tags: {', '.join(sub['validation_tags'][:3])}",
            fontsize=8,
        )
        ax.set_xticks([])

        ax = fig.add_subplot(gs[row, 1])
        center = local_x[labels == sub["subcluster"]].mean(axis=0, keepdims=True)
        local_members = np.where(labels == sub["subcluster"])[0]
        order = np.argsort(np.linalg.norm(local_x[local_members] - center, axis=1))[:4]
        for j, local_pos in enumerate(local_members[order]):
            global_idx = int(parent_member_idx[local_pos])
            ax.plot(robust_z(patches[global_idx]) + 4 * j, linewidth=1.0)
        ax.set_title("example medoid-like patches", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[row, 2])
        conditions = ["unrestricted", "same_patch_index", "cross_domain", "cross_domain_same_patch_index", "cross_domain_same_frequency"]
        vals = [
            sub["validation"]["condition_summary"].get(c, {}).get("mean_shape_correlation_mean", np.nan)
            for c in conditions
        ]
        ax.bar(np.arange(len(conditions)), vals, color=["gray", "tab:blue", "tab:red", "tab:red", "tab:red"], alpha=0.75)
        ax.axhline(0.25, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.1, 0.8)
        ax.set_xticks(np.arange(len(conditions)))
        ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=6)
        ax.set_title("controlled retrieval", fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def analyze_parent_split(
    target: dict[str, Any],
    bundle: dict[str, Any],
    seed: int,
    n_queries: int,
    top_k: int,
) -> dict[str, Any]:
    embeddings = bundle["embeddings"]
    meta = bundle["metadata"]
    patches = bundle["raw_patches"]
    x = bundle["x_pca"]
    parent_ids = bundle["cluster_ids"]
    parent_cluster = int(target["cluster"])
    member_idx = np.where(parent_ids == parent_cluster)[0]
    local = local_kmeans(embeddings, member_idx, int(target["n_subclusters"]), seed)
    sub_results = []
    for sub_id in sorted(set(local["labels"].tolist())):
        sub_member_idx = member_idx[local["labels"] == sub_id]
        stats = member_stats(meta, patches, sub_member_idx)
        validation = validate_member_set(sub_member_idx, x, meta, patches, n_queries, top_k)
        shape_summary = patch_shape_descriptors(patches, sub_member_idx)
        tags = validation_tags(validation["condition_summary"], stats)
        sub_results.append(
            {
                "subcluster": int(sub_id),
                "member_indices": sub_member_idx.astype(int).tolist(),
                "stats": stats,
                "shape_summary": shape_summary,
                "validation": validation,
                "validation_tags": tags,
            }
        )

    stem = f"{target['model']}_{target['layer']}_c{target['cluster']}_{target['temporary_name']}"
    fig_path = FIG_DIR / f"{stem}_internal_split.png"
    plot_parent_split(
        fig_path,
        f"Internal split: {target['model']} {target['layer']} c{target['cluster']} | {target['temporary_name']}",
        local["local_x"],
        local["labels"],
        sub_results,
        patches,
        member_idx,
    )
    return {
        "target": target,
        "parent_size": int(len(member_idx)),
        "n_subclusters": int(len(sub_results)),
        "local_pca_components": local["n_components"],
        "figure_path": str(fig_path.relative_to(ROOT)),
        "subclusters": [
            {k: v for k, v in sub.items() if k != "member_indices"}
            for sub in sub_results
        ],
    }


def plot_position_stratified(out_path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 2.4 * len(rows)), squeeze=False)
    for row_id, row in enumerate(rows):
        pos = row["patch_index"]
        local_x = np.asarray(row["local_x"])
        labels = np.asarray(row["labels"])
        for cid in sorted(set(labels.tolist())):
            axes[row_id, 0].scatter(local_x[labels == cid, 0], local_x[labels == cid, 1], s=5, alpha=0.55, label=f"k{cid}")
        axes[row_id, 0].set_title(f"p{pos} local PCA")
        axes[row_id, 0].set_xticks([])
        axes[row_id, 0].set_yticks([])
        axes[row_id, 0].legend(fontsize=6)

        for col, parent in enumerate(["c8", "c5", "other"], start=1):
            ax = axes[row_id, col]
            for cluster in row["clusters"]:
                if cluster["dominant_parent"] != parent:
                    continue
                mean = np.asarray(cluster["shape_summary"]["mean_curve"])
                ax.plot(mean, label=f"k{cluster['cluster']} n={cluster['size']}")
            ax.set_title(f"p{pos} {parent}")
            ax.set_xticks([])
            ax.legend(fontsize=6)
    fig.suptitle("TimesFM layer_10 position-stratified clusters", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def analyze_timesfm_position_strata(bundle: dict[str, Any], seed: int) -> dict[str, Any]:
    embeddings = bundle["embeddings"]
    meta = bundle["metadata"]
    patches = bundle["raw_patches"]
    parent_ids = bundle["cluster_ids"]
    rows = []
    for patch_index in [0, 1, 2, 3]:
        idx = np.asarray([i for i, m in enumerate(meta) if int(m["patch_index"]) == patch_index], dtype=int)
        local = local_kmeans(embeddings, idx, k=6, seed=seed + patch_index)
        clusters = []
        for cid in sorted(set(local["labels"].tolist())):
            members = idx[local["labels"] == cid]
            parent_counts = Counter([f"c{int(parent_ids[i])}" for i in members])
            dominant_parent, dominant_count = parent_counts.most_common(1)[0]
            shape = patch_shape_descriptors(patches, members)
            clusters.append(
                {
                    "cluster": int(cid),
                    "size": int(len(members)),
                    "dominant_parent": dominant_parent if dominant_parent in {"c8", "c5"} else "other",
                    "dominant_parent_raw": dominant_parent,
                    "dominant_parent_share": float(dominant_count / len(members)),
                    "top_parent_clusters": [{"value": k, "count": int(v)} for k, v in parent_counts.most_common(4)],
                    "stats": member_stats(meta, patches, members),
                    "shape_summary": shape,
                }
            )
        rows.append(
            {
                "patch_index": int(patch_index),
                "size": int(len(idx)),
                "local_x": local["local_x"][:, :2].astype(float).tolist(),
                "labels": local["labels"].astype(int).tolist(),
                "clusters": clusters,
            }
        )
    fig_path = FIG_DIR / "timesfm_2_5_layer_10_position_stratified_clusters.png"
    plot_position_stratified(fig_path, rows)
    compact_rows = []
    for row in rows:
        compact = {k: v for k, v in row.items() if k not in {"local_x", "labels"}}
        compact_rows.append(compact)
    return {"figure_path": str(fig_path.relative_to(ROOT)), "positions": compact_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-per-dataset", type=int, default=100)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--domain-balanced-patches", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--queries-per-subcluster", type=int, default=12)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    windows, window_meta, dataset_summary = sample_windows(
        DATA_ROOT,
        context_len=args.context_len,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
    )

    by_model_layer: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for target in PARENT_TARGETS:
        by_model_layer[(target["model"], target["layer"])].append(target)

    summary: dict[str, Any] = {
        "objective": "model-derived taxonomy v1 pilot: internal splitting and position-stratified audit",
        "num_windows": int(len(windows)),
        "windows_per_dataset": args.windows_per_dataset,
        "context_len": args.context_len,
        "seed": args.seed,
        "domain_balanced_patches_per_domain": args.domain_balanced_patches,
        "queries_per_subcluster": args.queries_per_subcluster,
        "top_k": args.top_k,
        "dataset_summary": dataset_summary,
        "parent_splits": [],
        "position_stratified": None,
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
            print(f"Splitting {target['model']} {target['layer']} c{target['cluster']}...")
            summary["parent_splits"].append(
                analyze_parent_split(
                    target=target,
                    bundle=bundle,
                    seed=args.seed,
                    n_queries=args.queries_per_subcluster,
                    top_k=args.top_k,
                )
            )
        if model_key == "timesfm_2_5" and layer_name == "layer_10":
            print("Running TimesFM position-stratified audit...")
            summary["position_stratified"] = analyze_timesfm_position_strata(bundle, seed=args.seed)

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
                "num_parent_splits": len(summary["parent_splits"]),
                "figure_dir": str(FIG_DIR.relative_to(ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
