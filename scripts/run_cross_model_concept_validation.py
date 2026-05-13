from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "cross_model_validation"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_PATH = OUT_DIR / "cross_model_validation_summary.json"
PROTOTYPE_BANK_PATH = OUT_DIR / "prototype_bank.json"

sys.path.insert(0, str(ROOT))
from scripts.build_cluster_cards import fit_domain_balanced_clusters, load_model_layer_bundle  # noqa: E402
from scripts.run_second_pilot_discovery import DATA_ROOT, robust_z, sample_windows  # noqa: E402
from scripts.run_taxonomy_v1_pilot import local_kmeans, patch_shape_descriptors  # noqa: E402


CONCEPT_SPECS = [
    {
        "name": "strong_rising_recovery",
        "kind": "candidate",
        "parent_cluster": 8,
        "subcluster": 0,
        "parent_k": 3,
        "temporary_definition": "sustained upward patch trajectory, often recovery or rising transition",
    },
    {
        "name": "strong_falling_transition",
        "kind": "candidate",
        "parent_cluster": 5,
        "subcluster": 1,
        "parent_k": 4,
        "temporary_definition": "sustained strong downward patch trajectory",
    },
    {
        "name": "smooth_falling_transition",
        "kind": "candidate",
        "parent_cluster": 5,
        "subcluster": 0,
        "parent_k": 4,
        "temporary_definition": "smoother and weaker downward transition",
    },
    {
        "name": "artifact_first_patch_behavior",
        "kind": "negative_control",
        "parent_cluster": 4,
        "subcluster": None,
        "parent_k": None,
        "temporary_definition": "TimesFM first-patch position artifact",
    },
]


MODEL_EVAL_SPECS = [
    {"model": "timesfm_2_5", "layer": "layer_10", "display": "TimesFM-2.5 layer_10"},
    {"model": "chronos_2", "layer": "layer_11", "display": "Chronos-2 layer_11"},
    {"model": "chronos_2_small", "layer": "layer_5", "display": "Chronos-2-small layer_5"},
]


def dist(values: list[Any], n: int = 6) -> list[dict[str, Any]]:
    return [{"value": str(k), "count": int(v)} for k, v in Counter([str(v) for v in values]).most_common(n)]


def select_bank_members(
    member_idx: np.ndarray,
    x: np.ndarray,
    medoid_count: int,
    neighbor_count: int,
) -> tuple[list[int], list[int]]:
    center = x[member_idx].mean(axis=0, keepdims=True)
    order = np.argsort(np.linalg.norm(x[member_idx] - center, axis=1))
    medoids = [int(i) for i in member_idx[order[: min(medoid_count, len(order))]]]
    chosen = set(medoids)
    neighbors: list[int] = []
    for medoid in medoids:
        dists = np.linalg.norm(x[member_idx] - x[medoid : medoid + 1], axis=1)
        for idx in member_idx[np.argsort(dists)]:
            item = int(idx)
            if item in chosen:
                continue
            neighbors.append(item)
            chosen.add(item)
            if len(neighbors) >= neighbor_count:
                return medoids, neighbors
    return medoids, neighbors


def build_timesfm_concept_members(
    timesfm_bundle: dict[str, Any],
    seed: int,
) -> dict[str, np.ndarray]:
    embeddings = timesfm_bundle["embeddings"]
    cluster_ids = timesfm_bundle["cluster_ids"]
    out: dict[str, np.ndarray] = {}
    split_cache: dict[tuple[int, int], dict[str, Any]] = {}
    for concept in CONCEPT_SPECS:
        parent = int(concept["parent_cluster"])
        parent_idx = np.where(cluster_ids == parent)[0]
        if concept["subcluster"] is None:
            out[concept["name"]] = parent_idx
            continue
        key = (parent, int(concept["parent_k"]))
        if key not in split_cache:
            split_cache[key] = local_kmeans(embeddings, parent_idx, int(concept["parent_k"]), seed)
        local = split_cache[key]
        out[concept["name"]] = parent_idx[local["labels"] == int(concept["subcluster"])]
    return out


def build_prototype_bank(
    timesfm_bundle: dict[str, Any],
    concept_members: dict[str, np.ndarray],
    medoid_count: int,
    neighbor_count: int,
) -> list[dict[str, Any]]:
    meta = timesfm_bundle["metadata"]
    patches = timesfm_bundle["raw_patches"]
    x = timesfm_bundle["x_pca"]
    bank = []
    for concept in CONCEPT_SPECS:
        name = concept["name"]
        members = concept_members[name]
        medoids, neighbors = select_bank_members(members, x, medoid_count, neighbor_count)
        for source, indices in [("medoid", medoids), ("high_confidence_neighbor", neighbors)]:
            for idx in indices:
                m = meta[idx]
                bank.append(
                    {
                        "concept": name,
                        "kind": concept["kind"],
                        "source": source,
                        "timesfm_index": int(idx),
                        "window_id": int(m["window_id"]),
                        "dataset": m["dataset"],
                        "domain": m["domain"],
                        "frequency": m.get("frequency_minutes"),
                        "timesfm_patch_index": int(m["patch_index"]),
                        "timesfm_patch_start": int(m["patch_start_in_window"]),
                        "timesfm_patch_end": int(m["patch_end_in_window"]),
                        "taxonomy_v0": m["taxonomy_label"],
                        "shape_summary": {
                            "slope": float(m["robust_slope"]),
                            "raw_std": float(m["raw_std"]),
                            "zero_ratio": float(m["zero_ratio"]),
                        },
                        "z_patch": robust_z(patches[idx]).astype(float).tolist(),
                    }
                )
    return bank


def map_bank_to_model_indices(
    bank: list[dict[str, Any]],
    model_key: str,
    meta: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    lookup = {(int(m["window_id"]), int(m["patch_index"])): i for i, m in enumerate(meta)}
    mapped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in bank:
        window_id = int(item["window_id"])
        timesfm_patch = int(item["timesfm_patch_index"])
        if model_key == "timesfm_2_5":
            patch_indices = [timesfm_patch]
        else:
            patch_indices = [timesfm_patch * 2, timesfm_patch * 2 + 1]
        for patch_index in patch_indices:
            idx = lookup.get((window_id, patch_index))
            if idx is None:
                continue
            mapped[idx].append(item)
    return mapped


def choose_label(items: list[dict[str, Any]]) -> str:
    counts = Counter(item["concept"] for item in items)
    return counts.most_common(1)[0][0]


def prototype_space_metrics(
    embeddings: np.ndarray,
    patches: np.ndarray,
    meta: list[dict[str, Any]],
    mapped: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    indices = np.asarray(sorted(mapped), dtype=int)
    labels = [choose_label(mapped[int(i)]) for i in indices]
    kinds = [
        "negative_control" if any(item["kind"] == "negative_control" for item in mapped[int(i)]) else "candidate"
        for i in indices
    ]
    if len(indices) < 3:
        return {"status": "too_few_mapped", "num_mapped": int(len(indices))}
    x = StandardScaler().fit_transform(embeddings[indices])
    pca_dim = max(2, min(20, x.shape[0] - 1, x.shape[1]))
    x_pca = PCA(n_components=pca_dim, random_state=47).fit_transform(x)

    same_concept = []
    same_concept_top5 = []
    same_kind = []
    shape_corr_1nn = []
    for row, idx in enumerate(indices):
        dists = np.linalg.norm(x_pca - x_pca[row : row + 1], axis=1)
        order = np.argsort(dists)
        nn = [j for j in order if j != row]
        top1 = nn[0]
        top5 = nn[: min(5, len(nn))]
        same_concept.append(labels[top1] == labels[row])
        same_concept_top5.append(float(np.mean([labels[j] == labels[row] for j in top5])))
        same_kind.append(kinds[top1] == kinds[row])
        shape_corr_1nn.append(shape_corr(patches[int(idx)], patches[int(indices[top1])]))

    concept_rows = []
    for concept in sorted(set(labels)):
        local = [i for i, label in enumerate(labels) if label == concept]
        concept_rows.append(
            {
                "concept": concept,
                "num_mapped": int(len(local)),
                "top_domains": dist([meta[int(indices[i])]["domain"] for i in local]),
                "top_patch_indices": dist([meta[int(indices[i])]["patch_index"] for i in local]),
                "mean_1nn_same_concept": float(np.mean([same_concept[i] for i in local])),
                "mean_top5_same_concept": float(np.mean([same_concept_top5[i] for i in local])),
                "mean_1nn_shape_correlation": float(np.mean([shape_corr_1nn[i] for i in local])),
            }
        )

    try:
        sil = float(silhouette_score(x_pca, labels)) if len(set(labels)) > 1 else None
    except Exception:
        sil = None

    return {
        "status": "ok",
        "num_mapped": int(len(indices)),
        "num_concepts": int(len(set(labels))),
        "pca_dim": int(pca_dim),
        "silhouette_by_concept": sil,
        "mean_1nn_same_concept": float(np.mean(same_concept)),
        "mean_top5_same_concept": float(np.mean(same_concept_top5)),
        "mean_1nn_same_kind": float(np.mean(same_kind)),
        "mean_1nn_shape_correlation": float(np.mean(shape_corr_1nn)),
        "concepts": concept_rows,
        "indices": indices.astype(int).tolist(),
        "labels": labels,
        "x_pca2": x_pca[:, :2].astype(float).tolist(),
    }


def shape_corr(a: np.ndarray, b: np.ndarray) -> float:
    za = robust_z(a)
    zb = robust_z(b)
    if np.std(za) < 1e-8 or np.std(zb) < 1e-8:
        return 0.0
    return float(np.corrcoef(za, zb)[0, 1])


def global_retrieval_metrics(
    x_all: np.ndarray,
    patches: np.ndarray,
    meta: list[dict[str, Any]],
    mapped: dict[int, list[dict[str, Any]]],
    top_k: int,
) -> dict[str, Any]:
    label_by_index = {idx: choose_label(items) for idx, items in mapped.items()}
    kind_by_index = {
        idx: "negative_control" if any(item["kind"] == "negative_control" for item in items) else "candidate"
        for idx, items in mapped.items()
    }
    query_indices = np.asarray(sorted(mapped), dtype=int)
    rows_by_concept: dict[str, list[dict[str, float]]] = defaultdict(list)
    for idx in query_indices:
        q = x_all[idx : idx + 1]
        dists = np.linalg.norm(x_all - q, axis=1)
        order = [int(i) for i in np.argsort(dists) if int(i) != int(idx)]
        nn = order[:top_k]
        q_label = label_by_index[int(idx)]
        q_kind = kind_by_index[int(idx)]
        row = {
            "mean_shape_correlation": float(np.mean([shape_corr(patches[int(idx)], patches[j]) for j in nn])),
            "prototype_hit_rate": float(np.mean([j in label_by_index for j in nn])),
            "same_concept_hit_rate": float(np.mean([label_by_index.get(j) == q_label for j in nn])),
            "same_kind_hit_rate": float(np.mean([kind_by_index.get(j) == q_kind for j in nn])),
            "domain_diversity": float(len(set(meta[j]["domain"] for j in nn))),
            "frequency_diversity": float(len(set(str(meta[j].get("frequency_minutes")) for j in nn))),
            "patch_index_diversity": float(len(set(str(meta[j]["patch_index"]) for j in nn))),
        }
        rows_by_concept[q_label].append(row)

    out = {}
    for concept, rows in rows_by_concept.items():
        out[concept] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        out[concept]["num_queries"] = int(len(rows))
    return out


def random_matched_baseline(
    x_all: np.ndarray,
    patches: np.ndarray,
    meta: list[dict[str, Any]],
    mapped: dict[int, list[dict[str, Any]]],
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    query_indices = sorted(mapped)
    by_patch: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        if i not in mapped:
            by_patch[str(m["patch_index"])].append(i)
    random_indices = []
    for idx in query_indices:
        patch = str(meta[idx]["patch_index"])
        pool = by_patch.get(patch, [])
        if pool:
            random_indices.append(int(rng.choice(pool)))
    if not random_indices:
        return {}
    rows = []
    for idx in random_indices:
        dists = np.linalg.norm(x_all - x_all[idx : idx + 1], axis=1)
        nn = [int(i) for i in np.argsort(dists) if int(i) != int(idx)][:top_k]
        rows.append(
            {
                "mean_shape_correlation": float(np.mean([shape_corr(patches[int(idx)], patches[j]) for j in nn])),
                "prototype_hit_rate": float(np.mean([j in mapped for j in nn])),
            }
        )
    return {
        "num_queries": int(len(rows)),
        "mean_shape_correlation": float(np.mean([r["mean_shape_correlation"] for r in rows])),
        "prototype_hit_rate": float(np.mean([r["prototype_hit_rate"] for r in rows])),
    }


def evaluate_model(
    spec: dict[str, str],
    windows: np.ndarray,
    window_meta: list[dict[str, Any]],
    bank: list[dict[str, Any]],
    domain_balanced_patches: int,
    batch_size: int,
    top_k: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    embeddings, meta, patches = load_model_layer_bundle(spec["model"], windows, window_meta, spec["layer"], batch_size)
    x_scaled = StandardScaler().fit_transform(embeddings)
    pca_dim = max(2, min(30, x_scaled.shape[0] - 1, x_scaled.shape[1]))
    x_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(x_scaled)
    mapped = map_bank_to_model_indices(bank, spec["model"], meta)
    proto = prototype_space_metrics(embeddings, patches, meta, mapped)
    global_metrics = global_retrieval_metrics(x_pca, patches, meta, mapped, top_k)
    baseline = random_matched_baseline(x_pca, patches, meta, mapped, top_k, seed)
    result = {
        "model": spec["model"],
        "layer": spec["layer"],
        "display": spec["display"],
        "evaluation_split": "full_equal_per_dataset",
        "num_embeddings": int(len(meta)),
        "num_mapped_prototype_patches": int(len(mapped)),
        "mapped_concept_counts": dict(Counter(choose_label(items) for items in mapped.values())),
        "prototype_space": {k: v for k, v in proto.items() if k not in {"indices", "labels", "x_pca2"}},
        "global_retrieval": global_metrics,
        "matched_random_baseline": baseline,
    }
    plot_payload = {
        "metadata": meta,
        "patches": patches,
        "prototype_space": proto,
        "mapped": mapped,
    }
    return result, plot_payload


def plot_concept_curves(out_path: Path, model_payloads: dict[str, dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    concepts = [c["name"] for c in CONCEPT_SPECS]
    fig, axes = plt.subplots(len(model_payloads), len(concepts), figsize=(3.2 * len(concepts), 2.4 * len(model_payloads)), squeeze=False)
    for row, (display, payload) in enumerate(model_payloads.items()):
        meta = payload["metadata"]
        patches = payload["patches"]
        mapped = payload["mapped"]
        label_by_idx = {idx: choose_label(items) for idx, items in mapped.items()}
        for col, concept in enumerate(concepts):
            ax = axes[row, col]
            idx = [i for i, label in label_by_idx.items() if label == concept]
            if not idx:
                ax.axis("off")
                continue
            z = np.asarray([robust_z(patches[i]) for i in idx])
            mean = np.mean(z, axis=0)
            std = np.std(z, axis=0)
            ax.plot(mean, color="tab:blue")
            ax.fill_between(np.arange(len(mean)), mean - std, mean + std, color="tab:blue", alpha=0.15)
            ax.set_title(f"{display}\n{concept}\nn={len(idx)}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Mapped prototype-bank mean curves by model", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_agreement_bars(out_path: Path, model_results: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = [r["display"] for r in model_results]
    top1 = [r["prototype_space"].get("mean_1nn_same_concept", np.nan) for r in model_results]
    top5 = [r["prototype_space"].get("mean_top5_same_concept", np.nan) for r in model_results]
    shape = [r["prototype_space"].get("mean_1nn_shape_correlation", np.nan) for r in model_results]
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, top1, width, label="1NN same concept")
    ax.bar(x, top5, width, label="top5 same concept")
    ax.bar(x + width, shape, width, label="1NN shape corr")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Prototype-space concept transfer metrics")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_global_retrieval(out_path: Path, model_results: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    concepts = [c["name"] for c in CONCEPT_SPECS]
    fig, axes = plt.subplots(1, len(model_results), figsize=(4.2 * len(model_results), 4.2), squeeze=False)
    for col, result in enumerate(model_results):
        ax = axes[0, col]
        vals = [result["global_retrieval"].get(c, {}).get("mean_shape_correlation", np.nan) for c in concepts]
        ax.bar(np.arange(len(concepts)), vals, color=["tab:green", "tab:orange", "tab:orange", "tab:red"], alpha=0.75)
        base = result.get("matched_random_baseline", {}).get("mean_shape_correlation")
        if base is not None:
            ax.axhline(base, color="black", linestyle="--", linewidth=1, label="matched random")
        ax.set_title(result["display"])
        ax.set_ylim(-0.1, 0.9)
        ax.set_xticks(np.arange(len(concepts)))
        ax.set_xticklabels(concepts, rotation=45, ha="right", fontsize=7)
        if col == 0:
            ax.set_ylabel("global retrieval mean shape corr")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-per-dataset", type=int, default=100)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--domain-balanced-patches", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--medoids-per-concept", type=int, default=20)
    parser.add_argument("--neighbors-per-concept", type=int, default=20)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    windows, window_meta, dataset_summary = sample_windows(
        DATA_ROOT,
        context_len=args.context_len,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
    )

    print("Preparing TimesFM prototype source...")
    timesfm_embeddings, timesfm_meta, timesfm_patches = load_model_layer_bundle(
        "timesfm_2_5", windows, window_meta, "layer_10", args.batch_size
    )
    timesfm_bundle = fit_domain_balanced_clusters(
        timesfm_embeddings,
        timesfm_meta,
        timesfm_patches,
        max_per_domain=args.domain_balanced_patches,
        seed=args.seed,
    )
    concept_members = build_timesfm_concept_members(timesfm_bundle, args.seed)
    bank = build_prototype_bank(
        timesfm_bundle,
        concept_members,
        medoid_count=args.medoids_per_concept,
        neighbor_count=args.neighbors_per_concept,
    )
    PROTOTYPE_BANK_PATH.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")

    model_results = []
    plot_payloads = {}
    for spec in MODEL_EVAL_SPECS:
        print(f"Evaluating {spec['display']}...")
        result, payload = evaluate_model(
            spec,
            windows,
            window_meta,
            bank,
            domain_balanced_patches=args.domain_balanced_patches,
            batch_size=args.batch_size,
            top_k=args.top_k,
            seed=args.seed,
        )
        model_results.append(result)
        plot_payloads[spec["display"]] = payload

    curve_path = FIG_DIR / "cross_model_prototype_curves.png"
    agreement_path = FIG_DIR / "cross_model_prototype_space_agreement.png"
    retrieval_path = FIG_DIR / "cross_model_global_retrieval_shape.png"
    plot_concept_curves(curve_path, plot_payloads)
    plot_agreement_bars(agreement_path, model_results)
    plot_global_retrieval(retrieval_path, model_results)

    summary = {
        "objective": "cross-model validation of TimesFM-derived taxonomy v1 pilot concepts",
        "num_windows": int(len(windows)),
        "windows_per_dataset": args.windows_per_dataset,
        "context_len": args.context_len,
        "seed": args.seed,
        "domain_balanced_patches_per_domain": args.domain_balanced_patches,
        "top_k": args.top_k,
        "prototype_bank": {
            "path": str(PROTOTYPE_BANK_PATH.relative_to(ROOT)),
            "num_items": int(len(bank)),
            "counts_by_concept": dict(Counter(item["concept"] for item in bank)),
            "counts_by_source": dict(Counter(item["source"] for item in bank)),
            "source_model": "TimesFM-2.5 layer_10 domain-balanced internal splits",
        },
        "dataset_summary": dataset_summary,
        "concept_specs": CONCEPT_SPECS,
        "model_results": model_results,
        "figures": {
            "prototype_curves": str(curve_path.relative_to(ROOT)),
            "prototype_space_agreement": str(agreement_path.relative_to(ROOT)),
            "global_retrieval_shape": str(retrieval_path.relative_to(ROOT)),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
                "prototype_bank": str(PROTOTYPE_BANK_PATH.relative_to(ROOT)),
                "num_prototypes": len(bank),
                "figure_dir": str(FIG_DIR.relative_to(ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
