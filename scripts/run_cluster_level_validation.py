from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "cluster_validation"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_PATH = OUT_DIR / "cluster_level_validation_summary.json"

sys.path.insert(0, str(ROOT))
from scripts.build_cluster_cards import (  # noqa: E402
    TARGETS,
    cluster_stats,
    confounder_warnings,
    fit_domain_balanced_clusters,
    load_model_layer_bundle,
)
from scripts.run_second_pilot_discovery import DATA_ROOT, robust_z, sample_windows  # noqa: E402


PRIMARY_TARGET_KEYS = {
    ("chronos_2", "layer_11", 6),
    ("timesfm_2_5", "layer_10", 8),
}


def _label_frequency(value: Any) -> str:
    return str(value)


def build_position_frequency_conditions(query_idx: int, meta: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    q = meta[query_idx]
    domain = q["domain"]
    freq = _label_frequency(q.get("frequency_minutes"))
    patch_index = str(q["patch_index"])

    same_patch = np.asarray([str(m["patch_index"]) == patch_index for m in meta], dtype=bool)
    same_freq = np.asarray([_label_frequency(m.get("frequency_minutes")) == freq for m in meta], dtype=bool)
    same_domain = np.asarray([m["domain"] == domain for m in meta], dtype=bool)
    cross_domain = ~same_domain

    return {
        "unrestricted": np.ones(len(meta), dtype=bool),
        "same_patch_index": same_patch,
        "same_frequency": same_freq,
        "same_patch_index_and_frequency": same_patch & same_freq,
        "cross_domain": cross_domain,
        "cross_domain_same_patch_index": cross_domain & same_patch,
        "cross_domain_same_frequency": cross_domain & same_freq,
        "cross_domain_same_patch_index_and_frequency": cross_domain & same_patch & same_freq,
        "same_domain": same_domain,
    }


def shape_corr(a: np.ndarray, b: np.ndarray) -> float:
    za = robust_z(a)
    zb = robust_z(b)
    if np.std(za) < 1e-8 or np.std(zb) < 1e-8:
        return 0.0
    return float(np.corrcoef(za, zb)[0, 1])


def retrieve_for_query(
    query_idx: int,
    x: np.ndarray,
    meta: list[dict[str, Any]],
    patches: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    q = x[query_idx : query_idx + 1]
    q_meta = meta[query_idx]
    out: dict[str, Any] = {}
    for condition, mask in build_position_frequency_conditions(query_idx, meta).items():
        allowed = np.where(mask)[0]
        allowed = allowed[allowed != query_idx]
        if len(allowed) == 0:
            out[condition] = {"status": "empty", "available": 0}
            continue
        dists = np.linalg.norm(x[allowed] - q, axis=1)
        order = np.argsort(dists)[: min(top_k, len(dists))]
        nn = allowed[order]
        corr = [shape_corr(patches[query_idx], patches[i]) for i in nn]
        out[condition] = {
            "status": "ok",
            "available": int(len(allowed)),
            "actual_k": int(len(nn)),
            "mean_embedding_distance": float(np.mean(dists[order])),
            "mean_shape_correlation": float(np.mean(corr)),
            "median_shape_correlation": float(np.median(corr)),
            "positive_shape_fraction": float(np.mean([c > 0.25 for c in corr])),
            "taxonomy_v0_agreement": float(np.mean([meta[i]["taxonomy_label"] == q_meta["taxonomy_label"] for i in nn])),
            "domain_diversity": int(len(set(meta[i]["domain"] for i in nn))),
            "frequency_diversity": int(len(set(_label_frequency(meta[i].get("frequency_minutes")) for i in nn))),
            "patch_index_diversity": int(len(set(str(meta[i]["patch_index"]) for i in nn))),
            "neighbor_indices": nn.astype(int).tolist(),
            "shape_correlations": [float(c) for c in corr],
        }
    return out


def select_diverse_medoid_queries(
    x: np.ndarray,
    cluster_ids: np.ndarray,
    meta: list[dict[str, Any]],
    cluster: int,
    n_queries: int,
) -> list[int]:
    member_idx = np.where(cluster_ids == cluster)[0]
    if len(member_idx) == 0:
        return []
    center = x[member_idx].mean(axis=0, keepdims=True)
    distances = np.linalg.norm(x[member_idx] - center, axis=1)
    ordered = member_idx[np.argsort(distances)]

    selected: list[int] = []
    domain_counts: Counter[str] = Counter()
    patch_counts: Counter[str] = Counter()
    freq_counts: Counter[str] = Counter()
    max_domain = max(2, math.ceil(n_queries / 4))
    max_patch = max(2, math.ceil(n_queries / 4))
    max_freq = max(2, math.ceil(n_queries / 4))

    for idx in ordered:
        domain = str(meta[int(idx)]["domain"])
        patch = str(meta[int(idx)]["patch_index"])
        freq = _label_frequency(meta[int(idx)].get("frequency_minutes"))
        if domain_counts[domain] >= max_domain or patch_counts[patch] >= max_patch or freq_counts[freq] >= max_freq:
            continue
        selected.append(int(idx))
        domain_counts[domain] += 1
        patch_counts[patch] += 1
        freq_counts[freq] += 1
        if len(selected) >= n_queries:
            return selected

    for idx in ordered:
        if int(idx) not in selected:
            selected.append(int(idx))
        if len(selected) >= n_queries:
            break
    return selected


def summarize_query_set(
    query_indices: list[int],
    per_query: list[dict[str, Any]],
    meta: list[dict[str, Any]],
) -> dict[str, Any]:
    conditions = list(build_position_frequency_conditions(query_indices[0], meta).keys()) if query_indices else []
    condition_summary: dict[str, Any] = {}
    for condition in conditions:
        rows = [q["retrieval"][condition] for q in per_query if q["retrieval"][condition]["status"] == "ok"]
        empty_count = sum(1 for q in per_query if q["retrieval"][condition]["status"] != "ok")
        if not rows:
            condition_summary[condition] = {"status": "empty_all", "ok_queries": 0, "empty_queries": empty_count}
            continue
        metric_names = [
            "mean_embedding_distance",
            "mean_shape_correlation",
            "median_shape_correlation",
            "positive_shape_fraction",
            "taxonomy_v0_agreement",
            "domain_diversity",
            "frequency_diversity",
            "patch_index_diversity",
        ]
        summary = {
            "status": "ok",
            "ok_queries": int(len(rows)),
            "empty_queries": int(empty_count),
            "mean_available": float(np.mean([r["available"] for r in rows])),
            "mean_actual_k": float(np.mean([r["actual_k"] for r in rows])),
        }
        for metric in metric_names:
            values = np.asarray([r[metric] for r in rows], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))
            summary[f"{metric}_median"] = float(np.median(values))
        condition_summary[condition] = summary
    return condition_summary


def query_metadata(query_indices: list[int], meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": meta[i]["dataset"],
            "domain": meta[i]["domain"],
            "frequency": meta[i].get("frequency_minutes"),
            "patch_index": int(meta[i]["patch_index"]),
            "taxonomy_label": meta[i]["taxonomy_label"],
            "raw_std": float(meta[i]["raw_std"]),
            "robust_slope": float(meta[i]["robust_slope"]),
            "zero_ratio": float(meta[i]["zero_ratio"]),
        }
        for i in query_indices
    ]


def compact_retrieval_for_json(retrieval: dict[str, Any]) -> dict[str, Any]:
    compact = {}
    for condition, info in retrieval.items():
        keep = {k: v for k, v in info.items() if k not in {"neighbor_indices", "shape_correlations"}}
        if info.get("status") == "ok":
            keep["shape_correlations"] = info["shape_correlations"]
        compact[condition] = keep
    return compact


def target_key(target: dict[str, Any]) -> tuple[str, str, int]:
    return str(target["model"]), str(target["layer"]), int(target["cluster"])


def pass_fail_tags(condition_summary: dict[str, Any], target: dict[str, Any]) -> list[str]:
    tags = []
    cross = condition_summary.get("cross_domain", {})
    cross_patch = condition_summary.get("cross_domain_same_patch_index", {})
    cross_freq = condition_summary.get("cross_domain_same_frequency", {})
    same_patch = condition_summary.get("same_patch_index", {})
    if cross.get("status") == "ok" and cross.get("mean_shape_correlation_mean", 0.0) >= 0.25:
        tags.append("cross_domain_shape_survives")
    if cross_patch.get("status") == "ok" and cross_patch.get("mean_shape_correlation_mean", 0.0) >= 0.25:
        tags.append("cross_domain_same_patch_survives")
    if cross_freq.get("status") == "ok" and cross_freq.get("mean_shape_correlation_mean", 0.0) >= 0.25:
        tags.append("cross_domain_same_frequency_survives")
    if same_patch.get("status") == "ok" and same_patch.get("mean_shape_correlation_mean", 0.0) >= 0.25:
        tags.append("same_patch_shape_survives")
    if target_key(target) == ("timesfm_2_5", "layer_10", 4):
        tags.append("known_patch_position_artifact_control")
    return tags or ["no_strong_survival_signal"]


def plot_condition_bars(out_path: Path, result: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    conditions = list(result["condition_summary"].keys())
    y = [result["condition_summary"][c].get("mean_shape_correlation_mean", np.nan) for c in conditions]
    err = [result["condition_summary"][c].get("mean_shape_correlation_std", 0.0) for c in conditions]
    colors = ["tab:red" if "cross_domain" in c else "tab:blue" if "same_patch" in c else "tab:gray" for c in conditions]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(np.arange(len(conditions)), y, yerr=err, color=colors, alpha=0.75, capsize=3)
    ax.axhline(0.25, color="black", linewidth=1, linestyle="--", label="working survival threshold")
    ax.set_ylabel("mean top-k shape correlation across queries")
    ax.set_title(
        f"{result['target']['model']} {result['target']['layer']} c{result['target']['cluster']} | "
        f"{result['target']['temporary_name']}"
    )
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(conditions, rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_query_panel(
    out_path: Path,
    result: dict[str, Any],
    query_indices: list[int],
    per_query: list[dict[str, Any]],
    meta: list[dict[str, Any]],
    patches: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    rows = min(8, len(query_indices))
    conditions = ["unrestricted", "same_patch_index", "same_frequency", "cross_domain", "cross_domain_same_patch_index"]
    fig, axes = plt.subplots(rows, len(conditions) + 1, figsize=(14, 2.1 * rows), squeeze=False)
    for row in range(rows):
        q_idx = query_indices[row]
        q_meta = meta[q_idx]
        axes[row, 0].plot(robust_z(patches[q_idx]), color="tab:red", linewidth=1.1)
        axes[row, 0].set_title(f"query\n{q_meta['dataset']} p{q_meta['patch_index']}\n{q_meta['taxonomy_label']}", fontsize=6)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        retrieval = per_query[row]["retrieval"]
        for col, condition in enumerate(conditions, start=1):
            info = retrieval[condition]
            ax = axes[row, col]
            if info["status"] != "ok" or not info["neighbor_indices"]:
                ax.text(0.1, 0.5, f"{condition}\nempty", fontsize=7)
                ax.axis("off")
                continue
            nn_idx = int(info["neighbor_indices"][0])
            nn_meta = meta[nn_idx]
            ax.plot(robust_z(patches[nn_idx]), color="tab:blue", linewidth=1.0)
            ax.set_title(
                f"{condition}\n{nn_meta['dataset']} p{nn_meta['patch_index']}\nr={info['shape_correlations'][0]:.2f}",
                fontsize=5.5,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"Multi-query retrieval examples: {result['target']['model']} {result['target']['layer']} "
        f"c{result['target']['cluster']}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def validate_target(
    target: dict[str, Any],
    bundle: dict[str, Any],
    n_queries: int,
    top_k: int,
) -> dict[str, Any]:
    meta = bundle["metadata"]
    patches = bundle["raw_patches"]
    x_pca = bundle["x_pca"]
    cluster_ids = bundle["cluster_ids"]
    cluster = int(target["cluster"])
    stats = cluster_stats(meta, cluster_ids, cluster)
    if stats["size"] == 0:
        return {"target": target, "status": "skipped_empty_cluster"}

    query_indices = select_diverse_medoid_queries(x_pca, cluster_ids, meta, cluster, n_queries)
    per_query_full = []
    per_query_json = []
    for q_idx in query_indices:
        retrieval = retrieve_for_query(q_idx, x_pca, meta, patches, top_k)
        row_full = {
            "query": query_metadata([q_idx], meta)[0],
            "retrieval": retrieval,
        }
        per_query_full.append(row_full)
        per_query_json.append(
            {
                "query": row_full["query"],
                "retrieval": compact_retrieval_for_json(retrieval),
            }
        )

    condition_summary = summarize_query_set(query_indices, per_query_full, meta)
    result: dict[str, Any] = {
        "target": target,
        "status": "ok",
        "cluster_stats": stats,
        "confounder_warnings": confounder_warnings(stats),
        "num_queries": int(len(query_indices)),
        "queries": query_metadata(query_indices, meta),
        "query_distribution": {
            "domains": dict(Counter(q["domain"] for q in query_metadata(query_indices, meta))),
            "frequencies": dict(Counter(_label_frequency(q["frequency"]) for q in query_metadata(query_indices, meta))),
            "patch_indices": dict(Counter(str(q["patch_index"]) for q in query_metadata(query_indices, meta))),
            "taxonomy_v0": dict(Counter(q["taxonomy_label"] for q in query_metadata(query_indices, meta))),
        },
        "condition_summary": condition_summary,
        "per_query": per_query_json,
    }
    result["validation_tags"] = pass_fail_tags(condition_summary, target)

    stem = f"{target['model']}_{target['layer']}_c{target['cluster']}_{target['temporary_name']}"
    bar_path = FIG_DIR / f"{stem}_condition_bars.png"
    query_path = FIG_DIR / f"{stem}_query_examples.png"
    plot_condition_bars(bar_path, result)
    plot_query_panel(query_path, result, query_indices, per_query_full, meta, patches)
    result["figure_paths"] = {
        "condition_bars": str(bar_path.relative_to(ROOT)),
        "query_examples": str(query_path.relative_to(ROOT)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-per-dataset", type=int, default=100)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--domain-balanced-patches", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--queries-per-cluster", type=int, default=16)
    parser.add_argument(
        "--target-scope",
        choices=["primary", "all"],
        default="all",
        help="primary validates the two strongest concept candidates; all also validates weak candidates and controls.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.target_scope == "primary":
        targets = [t for t in TARGETS if target_key(t) in PRIMARY_TARGET_KEYS]
    else:
        targets = TARGETS

    windows, window_meta, dataset_summary = sample_windows(
        DATA_ROOT,
        context_len=args.context_len,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
    )

    by_model_layer: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        by_model_layer[(str(target["model"]), str(target["layer"]))].append(target)

    summary: dict[str, Any] = {
        "objective": "multi-query position/frequency-aware cluster-level controlled validation",
        "source_cluster_cards": "outputs/cluster_cards/cluster_card_summary.json",
        "num_windows": int(len(windows)),
        "windows_per_dataset": args.windows_per_dataset,
        "context_len": args.context_len,
        "seed": args.seed,
        "domain_balanced_patches_per_domain": args.domain_balanced_patches,
        "queries_per_cluster": args.queries_per_cluster,
        "top_k": args.top_k,
        "target_scope": args.target_scope,
        "dataset_summary": dataset_summary,
        "targets": [],
    }

    for (model_key, layer_name), layer_targets in by_model_layer.items():
        print(f"Preparing {model_key} {layer_name}...")
        embeddings, meta, patches = load_model_layer_bundle(model_key, windows, window_meta, layer_name, args.batch_size)
        bundle = fit_domain_balanced_clusters(
            embeddings,
            meta,
            patches,
            max_per_domain=args.domain_balanced_patches,
            seed=args.seed,
        )
        for target in layer_targets:
            print(f"Validating {target['model']} {target['layer']} c{target['cluster']}...")
            summary["targets"].append(
                validate_target(
                    target=target,
                    bundle=bundle,
                    n_queries=args.queries_per_cluster,
                    top_k=args.top_k,
                )
            )

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
                "num_targets": len(summary["targets"]),
                "figure_dir": str(FIG_DIR.relative_to(ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
