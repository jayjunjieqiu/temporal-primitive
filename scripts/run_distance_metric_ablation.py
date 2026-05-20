from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "outputs" / "distance_metric_ablation"
FIG_DPI = 260

sys.path.insert(0, str(ROOT))
from scripts.run_second_pilot_discovery import DATA_ROOT, robust_z, sample_windows  # noqa: E402
from scripts.run_chronos_multilayer_cluster_validation import (  # noqa: E402
    DISPLAY_NAMES,
    MACRO_DOMAIN_ORDER,
    PATCH_LEN,
    REPRESENTATIONS,
    add_macro_metadata,
    build_rep_data,
    fit_pca_space,
    select_balanced_indices,
)


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def zpatches(patches: np.ndarray) -> np.ndarray:
    return np.asarray([robust_z(p).astype(np.float32) for p in patches], dtype=np.float32)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 1.0 if np.allclose(a, b) else 0.0
    return abs(float(np.corrcoef(a, b)[0, 1]))


def corr_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - abs_corr(a, b))


def constrained_dtw(a: np.ndarray, b: np.ndarray, radius: int = 2) -> float:
    n = len(a)
    m = len(b)
    band = max(radius, abs(n - m))
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    steps = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            cost = float((a[i - 1] - b[j - 1]) ** 2)
            choices = (dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            move = int(np.argmin(choices))
            prev = choices[move]
            dp[i, j] = cost + prev
            if move == 0:
                steps[i, j] = steps[i - 1, j] + 1
            elif move == 1:
                steps[i, j] = steps[i, j - 1] + 1
            else:
                steps[i, j] = steps[i - 1, j - 1] + 1
    path_len = max(1, int(steps[n, m]))
    return float(math.sqrt(dp[n, m] / path_len))


def pairwise_metric(values: np.ndarray, metric: str, radius: int) -> np.ndarray:
    n = len(values)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if metric == "raw_euclidean":
                d = euclidean_distance(values[i], values[j])
            elif metric == "correlation":
                d = corr_distance(values[i], values[j])
            elif metric == "dtw":
                d = constrained_dtw(values[i], values[j], radius=radius)
            else:
                raise ValueError(metric)
            out[i, j] = out[j, i] = d
    return out


def mean_upper(mat: np.ndarray) -> float:
    if mat.shape[0] < 2:
        return 0.0
    return float(np.mean(mat[np.triu_indices(mat.shape[0], k=1)]))


def medoid_and_neighbors(local_indices: np.ndarray, z: np.ndarray, metric: str, radius: int, n_neighbors: int = 4) -> list[int]:
    if len(local_indices) == 0:
        return []
    local_z = z[local_indices]
    dist = pairwise_metric(local_z, metric, radius)
    medoid_pos = int(np.argmin(np.mean(dist, axis=1)))
    order = np.argsort(dist[medoid_pos])[: min(n_neighbors, len(local_indices))]
    return [int(local_indices[i]) for i in order]


def rep_center_neighbors(indices: np.ndarray, x_pca: np.ndarray, center: np.ndarray, n_neighbors: int = 4) -> list[int]:
    if len(indices) == 0:
        return []
    order = np.argsort(np.linalg.norm(x_pca[indices] - center, axis=1))[: min(n_neighbors, len(indices))]
    return [int(indices[i]) for i in order]


def count_share(metadata: list[dict[str, Any]], indices: np.ndarray, key: str) -> tuple[int, float, str]:
    counts = Counter(str(metadata[int(i)].get(key)) for i in indices)
    if not counts:
        return 0, 0.0, ""
    value, count = counts.most_common(1)[0]
    return len(counts), float(count / max(1, len(indices))), value


def cluster_subset(indices: np.ndarray, max_size: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= max_size:
        return np.asarray(indices, dtype=int)
    return np.asarray(rng.choice(indices, size=max_size, replace=False), dtype=int)


def metric_baseline_indices(total: int, cluster_indices: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.ones(total, dtype=bool)
    mask[cluster_indices] = False
    pool = np.where(mask)[0]
    if len(pool) < size:
        pool = np.arange(total)
    return np.asarray(rng.choice(pool, size=size, replace=False), dtype=int)


def is_low_information(meta: dict[str, Any]) -> bool:
    raw_std = float(meta.get("raw_std", 0.0) or 0.0)
    raw_range = float(meta.get("raw_range", 0.0) or 0.0)
    zero_ratio = float(meta.get("zero_ratio", 0.0) or 0.0)
    return zero_ratio >= 0.95 or raw_std < 1e-8 or raw_range < 1e-8


def low_information_share(metadata: list[dict[str, Any]], indices: np.ndarray) -> float:
    if len(indices) == 0:
        return 0.0
    return float(np.mean([is_low_information(metadata[int(i)]) for i in indices]))


def shape_eligible_indices(
    indices: np.ndarray,
    metadata: list[dict[str, Any]],
    min_count: int = 8,
    flat_dominated_threshold: float = 0.50,
) -> tuple[np.ndarray, str, float]:
    indices = np.asarray(indices, dtype=int)
    flat_share = low_information_share(metadata, indices)
    eligible = np.asarray([int(i) for i in indices if not is_low_information(metadata[int(i)])], dtype=int)
    if flat_share >= flat_dominated_threshold or len(eligible) < min_count:
        return indices, "flat-dominated" if flat_share >= flat_dominated_threshold else "all-patches", flat_share
    return eligible, "shape-filtered", flat_share


def shape_eligible_universe(metadata: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([i for i, m in enumerate(metadata) if not is_low_information(m)], dtype=int)


def baseline_indices_from_pool(
    total: int,
    cluster_indices: np.ndarray,
    size: int,
    rng: np.random.Generator,
    allowed_pool: np.ndarray | None = None,
) -> np.ndarray:
    if allowed_pool is None or len(allowed_pool) < size:
        return metric_baseline_indices(total, cluster_indices, size, rng)
    cluster_set = set(int(i) for i in cluster_indices)
    pool = np.asarray([int(i) for i in allowed_pool if int(i) not in cluster_set], dtype=int)
    if len(pool) < size:
        return metric_baseline_indices(total, cluster_indices, size, rng)
    return np.asarray(rng.choice(pool, size=size, replace=False), dtype=int)


def cluster_metric_row(
    setting_id: str,
    rep: str,
    k: int,
    cid: int,
    cluster_indices: np.ndarray,
    metadata: list[dict[str, Any]],
    z: np.ndarray,
    max_patches: int,
    radius: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    effective_indices, prototype_mode, flat_share = shape_eligible_indices(cluster_indices, metadata)
    sampled = cluster_subset(effective_indices, max_patches, rng)
    allowed_pool = shape_eligible_universe(metadata) if prototype_mode == "shape-filtered" else None
    random_idx = baseline_indices_from_pool(len(z), cluster_indices, len(sampled), rng, allowed_pool=allowed_pool)
    sampled_z = z[sampled]
    random_z = z[random_idx]
    raw_intra = mean_upper(pairwise_metric(sampled_z, "raw_euclidean", radius))
    raw_base = mean_upper(pairwise_metric(random_z, "raw_euclidean", radius))
    corr_intra = mean_upper(pairwise_metric(sampled_z, "correlation", radius))
    corr_base = mean_upper(pairwise_metric(random_z, "correlation", radius))
    dtw_intra = mean_upper(pairwise_metric(sampled_z, "dtw", radius))
    dtw_base = mean_upper(pairwise_metric(random_z, "dtw", radius))
    raw_ratio = raw_intra / max(raw_base, 1e-8)
    corr_ratio = corr_intra / max(corr_base, 1e-8)
    dtw_ratio = dtw_intra / max(dtw_base, 1e-8)
    dtw_gain = raw_ratio - dtw_ratio

    macro_div, macro_share, macro_top = count_share(metadata, cluster_indices, "macro_domain")
    dataset_div, dataset_share, dataset_top = count_share(metadata, cluster_indices, "dataset")
    source_div, source_share, source_top = count_share(metadata, cluster_indices, "source_domain")
    freq_div, freq_share, freq_top = count_share(metadata, cluster_indices, "frequency_minutes")
    patch_div, patch_share, patch_top = count_share(metadata, cluster_indices, "patch_index")
    confounder = max(dataset_share, source_share, freq_share, patch_share)
    if dtw_gain >= 0.08 and confounder < 0.80:
        label = "dtw_benefited"
    elif confounder >= 0.80:
        label = "confounded"
    else:
        label = "unchanged"

    return {
        "setting_id": setting_id,
        "representation": rep,
        "k": int(k),
        "cluster": int(cid),
        "cluster_size": int(len(cluster_indices)),
        "sample_size": int(len(sampled)),
        "shape_metric_mode": prototype_mode,
        "low_information_share": flat_share,
        "intra_raw_euclidean": raw_intra,
        "baseline_raw_euclidean": raw_base,
        "raw_euclidean_ratio": raw_ratio,
        "intra_correlation_distance": corr_intra,
        "baseline_correlation_distance": corr_base,
        "correlation_ratio": corr_ratio,
        "intra_dtw_radius_2": dtw_intra,
        "baseline_dtw_radius_2": dtw_base,
        "dtw_ratio": dtw_ratio,
        "dtw_gain_over_euclidean": dtw_gain,
        "macro_domain_diversity": macro_div,
        "top_macro_domain": macro_top,
        "top_macro_domain_share": macro_share,
        "dataset_diversity": dataset_div,
        "top_dataset": dataset_top,
        "top_dataset_share": dataset_share,
        "source_domain_diversity": source_div,
        "top_source_domain": source_top,
        "top_source_domain_share": source_share,
        "frequency_diversity": freq_div,
        "top_frequency": freq_top,
        "top_frequency_share": freq_share,
        "patch_index_diversity": patch_div,
        "top_patch_index": patch_top,
        "top_patch_index_share": patch_share,
        "confounder_max_share": confounder,
        "evidence_label": label,
    }


def make_settings(summary: dict[str, Any], reps: list[str], k_settings: list[str]) -> list[dict[str, Any]]:
    selected = summary["selected_k"]
    shared_k = int(selected["recommended_shared_k"])
    out: list[dict[str, Any]] = []
    if "shared" in k_settings:
        for rep in reps:
            out.append({"rep": rep, "k": shared_k, "setting_id": f"{rep}_k{shared_k}", "kind": "shared"})
    if "layer_specific" in k_settings or "k10" in k_settings:
        for rep in reps:
            k = int(selected["per_layer"][rep]["recommended_k"])
            if k != shared_k or "k10" in k_settings:
                if "k10" in k_settings and k != 10:
                    continue
                out.append({"rep": rep, "k": k, "setting_id": f"{rep}_k{k}", "kind": "layer_specific"})
    dedup = {}
    for item in out:
        dedup[(item["rep"], item["k"])] = item
    return list(dedup.values())


def fit_setting(rep_space: dict[str, Any], k: int, seed: int) -> tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=k, random_state=seed, n_init=30).fit(rep_space["x_pca"])
    return model, model.labels_


def plot_embedding_cluster_audit(
    setting: dict[str, Any],
    x2: np.ndarray,
    labels: np.ndarray,
    centers2: np.ndarray,
    metadata: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    fig_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    row_by_cluster = {int(r["cluster"]): r for r in cluster_rows}
    dtw_values = np.asarray([row_by_cluster[int(c)]["dtw_gain_over_euclidean"] for c in labels], dtype=float)
    conf_values = np.asarray([row_by_cluster[int(c)]["confounder_max_share"] for c in labels], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, sharey=True)
    sc0 = axes[0].scatter(x2[:, 0], x2[:, 1], c=labels, s=5, cmap="tab20", alpha=0.58, linewidths=0)
    axes[0].scatter(centers2[:, 0], centers2[:, 1], c="black", marker="x", s=28, linewidths=1.1)
    for cid in sorted(set(labels.tolist())):
        axes[0].text(centers2[cid, 0], centers2[cid, 1], f"C{cid}", fontsize=7, weight="bold")
    axes[0].set_title("KMeans clusters")
    fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.03).set_label("Cluster")

    sc1 = axes[1].scatter(x2[:, 0], x2[:, 1], c=dtw_values, s=5, cmap="RdYlGn", alpha=0.68, linewidths=0)
    axes[1].set_title("DTW gain over Euclidean")
    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.03).set_label("Gain")

    sc2 = axes[2].scatter(x2[:, 0], x2[:, 1], c=conf_values, s=5, cmap="YlOrRd", alpha=0.68, linewidths=0, vmin=0, vmax=1)
    axes[2].set_title("Confounder risk")
    fig.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.03).set_label("Max share")

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(color="#eef2f5", linewidth=0.5)
    title = f"{DISPLAY_NAMES.get(setting['rep'], setting['rep'])} K={setting['k']}: embedding and confounder audit"
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / f"embedding_cluster_audit_{setting['setting_id']}.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_patch_cell(ax: Any, patches: np.ndarray, indices: list[int], metadata: list[dict[str, Any]], title: str) -> None:
    colors = ["#1f6f9f", "#7aa6c2", "#9ec4d7", "#c4dbe6"]
    for pos, item in enumerate(indices):
        lw = 1.55 if pos == 0 else 0.95
        alpha = 1.0 if pos == 0 else 0.62
        ax.plot(robust_z(patches[item]), color=colors[min(pos, len(colors) - 1)], linewidth=lw, alpha=alpha)
    if indices:
        m = metadata[int(indices[0])]
        subtitle = f"{m['dataset']}, {m['macro_domain']}, p{m['patch_index']}"
    else:
        subtitle = "empty"
    ax.set_title(f"{title}\n{subtitle}", fontsize=7.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#222831")
        spine.set_linewidth(0.75)


def plot_prototype_metric_comparison(
    setting: dict[str, Any],
    x_pca: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    metadata: list[dict[str, Any]],
    patches: np.ndarray,
    z: np.ndarray,
    radius: int,
    max_patches: int,
    seed: int,
    fig_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed + setting["k"])
    clusters = sorted(set(labels.tolist()))
    fig, axes = plt.subplots(len(clusters), 4, figsize=(11.8, 1.45 * len(clusters)), squeeze=False)
    for row, cid in enumerate(clusters):
        idx = np.where(labels == cid)[0]
        effective_idx, mode, _flat_share = shape_eligible_indices(idx, metadata)
        sampled = cluster_subset(effective_idx, max_patches, rng)
        title_suffix = "flat" if mode == "flat-dominated" else ("shape" if mode == "shape-filtered" else "all")
        columns = [
            ("Rep center", rep_center_neighbors(effective_idx, x_pca, centers[cid], 4)),
            ("Raw L2 medoid", medoid_and_neighbors(sampled, z, "raw_euclidean", radius, 4)),
            ("Corr medoid", medoid_and_neighbors(sampled, z, "correlation", radius, 4)),
            ("DTW medoid", medoid_and_neighbors(sampled, z, "dtw", radius, 4)),
        ]
        for col, (name, selected) in enumerate(columns):
            plot_patch_cell(axes[row, col], patches, selected, metadata, f"C{cid} | {name} | {title_suffix}")
    fig.suptitle(f"{DISPLAY_NAMES.get(setting['rep'], setting['rep'])} K={setting['k']}: shape-filtered prototype selection", fontsize=14, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=0.85, w_pad=0.6)
    fig.savefig(fig_dir / f"prototype_metric_comparison_{setting['setting_id']}.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_distance_heatmap(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    def display_row(row: dict[str, Any]) -> str:
        rep = DISPLAY_NAMES.get(str(row["representation"]), str(row["representation"]).replace("_", " ").title())
        return f"{rep} K{row['k']} C{row['cluster']}"

    metrics = [
        ("raw_euclidean_ratio", "Raw L2 ratio ↓"),
        ("correlation_ratio", "Corr ratio ↓"),
        ("dtw_ratio", "DTW ratio ↓"),
        ("dtw_gain_over_euclidean", "DTW gain ↑"),
        ("macro_domain_diversity", "Macro diversity ↑"),
        ("low_information_share", "Low-info share ↓"),
        ("confounder_max_share", "Confounder ↓"),
    ]
    labels = [display_row(r) for r in rows]
    data = np.asarray([[float(r[key]) for key, _name in metrics] for r in rows], dtype=float)
    scaled = data.copy()
    for j in range(scaled.shape[1]):
        col = scaled[:, j]
        lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
        scaled[:, j] = 0.5 if hi - lo < 1e-12 else (col - lo) / (hi - lo)
    fig_h = max(5.0, 0.26 * len(rows))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    im = ax.imshow(scaled, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([name for _key, name in metrics], rotation=25, ha="right", fontsize=8.5)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=5.8, color="white" if scaled[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).set_label("Column-normalized value")
    ax.set_title("Cluster-level original-space distance diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "distance_metric_heatmap.png", dpi=FIG_DPI)
    plt.close(fig)


def topk_by_metric(
    query: int,
    candidates: np.ndarray,
    metric: str,
    x_pca: np.ndarray,
    z: np.ndarray,
    radius: int,
    k: int,
) -> list[int]:
    candidates = np.asarray([int(i) for i in candidates if int(i) != int(query)], dtype=int)
    if len(candidates) == 0:
        return []
    if metric == "representation":
        d = np.linalg.norm(x_pca[candidates] - x_pca[query], axis=1)
    elif metric == "raw_euclidean":
        d = np.linalg.norm(z[candidates] - z[query], axis=1)
    elif metric == "correlation":
        d = np.asarray([corr_distance(z[query], z[i]) for i in candidates])
    elif metric == "dtw":
        d = np.asarray([constrained_dtw(z[query], z[i], radius=radius) for i in candidates])
    else:
        raise ValueError(metric)
    order = np.argsort(d)[: min(k, len(candidates))]
    return [int(candidates[i]) for i in order]


def retrieval_summary(query: int, neighbors: list[int], metadata: list[dict[str, Any]], z: np.ndarray) -> dict[str, Any]:
    if not neighbors:
        return {"shape_correlation": 0.0, "macro_domain_diversity": 0}
    corr = [abs_corr(z[query], z[i]) for i in neighbors]
    return {
        "shape_correlation": float(np.mean(corr)),
        "macro_domain_diversity": int(len({str(metadata[i]["macro_domain"]) for i in neighbors})),
    }


def choose_retrieval_queries(rows: list[dict[str, Any]], labels_by_setting: dict[str, np.ndarray], limit: int = 4) -> list[tuple[str, int]]:
    scored = sorted(rows, key=lambda r: (r["evidence_label"] == "dtw_benefited", r["dtw_gain_over_euclidean"]), reverse=True)
    out: list[tuple[str, int]] = []
    for row in scored:
        setting_id = str(row["setting_id"])
        cid = int(row["cluster"])
        if (setting_id, cid) not in out:
            out.append((setting_id, cid))
        if len(out) >= limit:
            break
    return out


def plot_retrieval_examples(
    query_specs: list[dict[str, Any]],
    payload_by_setting: dict[str, dict[str, Any]],
    fig_dir: Path,
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    if not query_specs:
        return []
    metrics = [
        ("representation", "Rep Euclidean"),
        ("raw_euclidean", "Raw L2"),
        ("correlation", "Correlation"),
        ("dtw", "DTW"),
    ]
    fig, axes = plt.subplots(len(query_specs), len(metrics), figsize=(13.5, 1.75 * len(query_specs)), squeeze=False)
    rows = []
    for row, spec in enumerate(query_specs):
        payload = payload_by_setting[spec["setting_id"]]
        query = int(spec["query"])
        if is_low_information(payload["metadata"][query]):
            candidates = np.arange(len(payload["metadata"]))
        else:
            candidates = shape_eligible_universe(payload["metadata"])
        rep_top = set(topk_by_metric(query, candidates, "representation", payload["x_pca"], payload["z"], payload["radius"], 3))
        for col, (metric, title) in enumerate(metrics):
            nn = topk_by_metric(query, candidates, metric, payload["x_pca"], payload["z"], payload["radius"], 3)
            info = retrieval_summary(query, nn, payload["metadata"], payload["z"])
            overlap = len(rep_top.intersection(nn)) / max(1, len(rep_top))
            rows.append(
                {
                    "setting_id": spec["setting_id"],
                    "cluster": int(spec["cluster"]),
                    "query_index": query,
                    "metric": metric,
                    "neighbors": ";".join(str(i) for i in nn),
                    "top3_overlap_with_representation": float(overlap),
                    **info,
                }
            )
            ax = axes[row, col]
            ax.plot(payload["z"][query], color="#111111", linewidth=1.55, label="query")
            for item in nn:
                ax.plot(payload["z"][item], color="#2b6f9f", alpha=0.44, linewidth=1.0)
            q_meta = payload["metadata"][query]
            ax.set_title(
                f"{title}\n{spec['setting_id']} C{spec['cluster']} | r={info['shape_correlation']:.2f} | ov={overlap:.2f}\n"
                f"{q_meta['dataset']}, {q_meta['macro_domain']}, p{q_meta['patch_index']}",
                fontsize=6.4,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Retrieval comparison: same query, different distance metrics", fontsize=14, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=0.95, w_pad=0.6)
    fig.savefig(fig_dir / "retrieval_metric_comparison_examples.png", dpi=FIG_DPI)
    plt.close(fig)
    return rows


def find_failure_cases(payload: dict[str, Any], labels: np.ndarray, seed: int, radius: int, max_pairs: int = 320) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = len(payload["z"])
    sample = np.asarray(rng.choice(np.arange(n), size=min(max_pairs, n), replace=False), dtype=int)
    pairs = []
    for a_pos in range(len(sample)):
        for b_pos in range(a_pos + 1, len(sample)):
            i, j = int(sample[a_pos]), int(sample[b_pos])
            rep = euclidean_distance(payload["x_pca"][i], payload["x_pca"][j])
            dtw = constrained_dtw(payload["z"][i], payload["z"][j], radius=radius)
            corr = abs_corr(payload["z"][i], payload["z"][j])
            pairs.append((i, j, rep, dtw, corr))
    if not pairs:
        return []
    rep_vals = np.asarray([p[2] for p in pairs])
    dtw_vals = np.asarray([p[3] for p in pairs])
    rep_hi = float(np.quantile(rep_vals, 0.85))
    rep_lo = float(np.quantile(rep_vals, 0.15))
    dtw_hi = float(np.quantile(dtw_vals, 0.85))
    dtw_lo = float(np.quantile(dtw_vals, 0.15))
    over = sorted([p for p in pairs if p[3] <= dtw_lo and p[4] <= 0.20], key=lambda p: (p[3], p[4]))
    rep_near_dtw_far = sorted([p for p in pairs if p[2] <= rep_lo and p[3] >= dtw_hi], key=lambda p: (p[2], -p[3]))
    dtw_near_rep_far = sorted([p for p in pairs if p[3] <= dtw_lo and p[2] >= rep_hi], key=lambda p: (p[3], -p[2]))
    selected = []
    for name, candidates in [
        ("DTW over-warping", over),
        ("Rep-near raw-DTW-far", rep_near_dtw_far),
        ("Raw-DTW-near rep-far", dtw_near_rep_far),
    ]:
        if candidates:
            i, j, rep, dtw, corr = candidates[0]
            selected.append({"case": name, "i": i, "j": j, "rep_distance": rep, "dtw_distance": dtw, "shape_correlation": corr})
    return selected


def plot_failure_cases(cases: list[dict[str, Any]], payload: dict[str, Any], fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not cases:
        return
    fig, axes = plt.subplots(len(cases), 2, figsize=(8.5, 1.9 * len(cases)), squeeze=False)
    for row, case in enumerate(cases):
        for col, key in enumerate(["i", "j"]):
            idx = int(case[key])
            ax = axes[row, col]
            ax.plot(payload["z"][idx], color="#2b6f9f", linewidth=1.35)
            meta = payload["metadata"][idx]
            ax.set_title(
                f"{case['case']} | patch {col + 1}\n{meta['dataset']}, {meta['macro_domain']}, p{meta['patch_index']}\n"
                f"rep={case['rep_distance']:.2f}, dtw={case['dtw_distance']:.2f}, r={case['shape_correlation']:.2f}",
                fontsize=6.8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Failure cases: when distance metrics can mislead", fontsize=14, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=0.9, w_pad=0.7)
    fig.savefig(fig_dir / "dtw_failure_cases.png", dpi=FIG_DPI)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(out_dir: Path, settings: list[dict[str, Any]], cluster_rows: list[dict[str, Any]], motif_rows: list[dict[str, Any]], failure_cases: list[dict[str, Any]]) -> None:
    fig_rel = lambda name: "../outputs/distance_metric_ablation/figures/" + name
    lines = []
    lines.append("# Distance Metric Ablation Report: DTW vs Euclidean for Chronos-2 Patch Concepts")
    lines.append("")
    lines.append("## 1. Advisor Question")
    lines.append("")
    lines.append("这份报告回答一个很具体的问题：Chronos-2 的 model-derived motif clusters 在 original time-series space 中看起来不够 coherent，是否部分来自 distance function / prototype selection metric 不合适。")
    lines.append("")
    lines.append("最新 verdict：Euclidean/KMeans cluster label 不能作为最终 motif taxonomy。它只能作为 TSFM representation geometry 的 diagnostic baseline / candidate neighborhood sampler；真正进入 `model-derived motif taxonomy v1` 的 candidate motif/prototype family，必须通过 warping-aware original-space validation，尤其是 DTW-aware prototype selection、controlled retrieval 和 confounder audit。")
    lines.append("")
    lines.append("本报告采用 **two-space distance principle**：在 `Chronos-2 representation space` 中用 Euclidean geometry 发现模型内部的 patch-token neighborhoods；在 `original time-series space` 中用 DTW geometry 验证这些 neighborhoods 是否对应 coherent temporal shapes。换句话说，Euclidean 用来回答“模型认为哪些 token 相近”，DTW 用来回答“这些 token 回到原空间后是否同形”。")
    lines.append("")
    lines.append("## 2. Method")
    lines.append("")
    lines.append("- fixed model: `Chronos-2`")
    lines.append("- representation-space candidate generation: `StandardScaler -> PCA(max 30 dims) -> KMeans` with Euclidean geometry")
    lines.append("- K selection remains a representation-space operating-point choice, not a DTW clustering objective")
    lines.append("- original-space validation metrics: constrained DTW as the primary shape-coherence metric; z-normalized raw Euclidean and `1 - |correlation|` as diagnostic controls")
    lines.append("- DTW setting: Sakoe-Chiba radius `2`; radius sensitivity `1/2/3` is recorded in the summary JSON")
    lines.append("- prototype figures use shape-eligible patches when a cluster is not flat-dominated; low-information / near-flat patches are retained only for flat-dominated diagnostic clusters")
    lines.append("- zero-reference guide lines are removed from all patch waveform panels, so horizontal curves represent actual low-information patches rather than plotting guides")
    lines.append("- external weak motif labels are excluded from the main evidence because the current deterministic probe is not reliable enough for paper-level claims")
    lines.append("")
    lines.append("## 3. Visual Evidence I: Embedding Cluster Maps")
    lines.append("")
    lines.append("每张图从左到右分别是 KMeans cluster、cluster-level DTW gain、confounder risk。DTW gain 为正表示该 cluster 在 DTW 下相对 raw Euclidean 更紧；confounder risk 越高，越不应把该 cluster 命名为 motif/prototype family。")
    for setting in settings:
        lines.append("")
        lines.append(f"![{setting['setting_id']} embedding audit]({fig_rel(f'embedding_cluster_audit_{setting['setting_id']}.png')})")
    lines.append("")
    lines.append("## 4. Visual Evidence II: Prototype Selection Comparison")
    lines.append("")
    lines.append("每一行展示一个 representation-space cluster；四列分别使用 representation center、raw Euclidean medoid、correlation medoid 和 DTW medoid 选择 prototype patches。前一列回答模型空间中心是什么，后三列回答原空间用不同距离看会得到什么 prototype。所有 clusters 都展示，不做 cherry-picking。为避免横线型 low-information patches 污染非 flat clusters，本版对非 flat-dominated cluster 使用 shape-eligible subset；flat-dominated cluster 仍保留为 diagnostic。")
    for setting in settings:
        lines.append("")
        lines.append(f"![{setting['setting_id']} prototype comparison]({fig_rel(f'prototype_metric_comparison_{setting['setting_id']}.png')})")
    lines.append("")
    lines.append("## 5. Quantitative Evidence")
    lines.append("")
    lines.append("下图按 cluster 汇总 original-space distance diagnostics。ratio 是 intra-cluster distance / matched random baseline distance；越低说明该 metric 下 cluster 越紧。DTW ratio 是 motif/prototype 命名时更重要的 gate；raw Euclidean / correlation 用于说明距离选择是否改变解释。")
    lines.append("")
    lines.append(f"![distance metric heatmap]({fig_rel('distance_metric_heatmap.png')})")
    lines.append("")
    top_gain = sorted(cluster_rows, key=lambda r: r["dtw_gain_over_euclidean"], reverse=True)[:8]
    lines.append("| setting | cluster | DTW gain ↑ | raw ratio ↓ | DTW ratio ↓ | low-info ↓ | macro diversity ↑ | confounder ↓ | label |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in top_gain:
        lines.append(
            f"| `{row['setting_id']}` | C{row['cluster']} | {row['dtw_gain_over_euclidean']:.3f} | "
            f"{row['raw_euclidean_ratio']:.3f} | {row['dtw_ratio']:.3f} | {row['low_information_share']:.3f} | {row['macro_domain_diversity']} | "
            f"{row['confounder_max_share']:.3f} | `{row['evidence_label']}` |"
        )
    lines.append("")
    lines.append("## 6. Retrieval Comparison")
    lines.append("")
    lines.append("同一个 query patch 分别用 representation Euclidean、raw Euclidean、correlation 和 DTW 做 retrieval。若 DTW 找到的 neighbors 更同形但与 representation retrieval overlap 很低，说明 cluster 可能包含多个 raw-shape subfamilies。")
    lines.append("")
    lines.append(f"![retrieval comparison]({fig_rel('retrieval_metric_comparison_examples.png')})")
    lines.append("")
    lines.append("## 7. Failure Cases")
    lines.append("")
    lines.append("DTW 不是无条件更好。下面展示 DTW over-warping、representation-near but raw-DTW-far、raw-DTW-near but representation-far 三类风险。")
    lines.append("")
    lines.append(f"![DTW failure cases]({fig_rel('dtw_failure_cases.png')})")
    if failure_cases:
        lines.append("")
        lines.append("| case | rep distance | DTW distance | shape corr |")
        lines.append("|---|---:|---:|---:|")
        for case in failure_cases:
            lines.append(f"| {case['case']} | {case['rep_distance']:.3f} | {case['dtw_distance']:.3f} | {case['shape_correlation']:.3f} |")
    lines.append("")
    lines.append("## 8. Final Answer")
    lines.append("")
    benefited = [r for r in cluster_rows if r["evidence_label"] == "dtw_benefited"]
    confounded = [r for r in cluster_rows if r["evidence_label"] == "confounded"]
    lines.append(f"当前 ablation 支持一个更明确的结论：Euclidean representation clustering 不能作为最终 motif taxonomy 机制。它可以揭示 Chronos-2 patch-token representation geometry，但会把 time-shifted / phase-shifted / locally warped shapelet-like patterns 处理得不够稳健。本轮共有 `{len(benefited)}` 个 cluster 被标记为 `dtw_benefited`，但也有 `{len(confounded)}` 个 cluster 存在较高 confounder risk。")
    lines.append("")
    lines.append("因此，后续路线不是抛弃 Euclidean，而是明确分工：Euclidean/KMeans 保留为 representation-space neighborhood discovery；DTW-aware original-space validation 升级为命名 candidate motif/prototype family 的必要条件。对于 spike / burst / oscillation 等局部错位敏感 motif，优先使用 DTW medoid 和 DTW controlled retrieval 做证据；同时保留 failure cases，防止 DTW over-warping 被误读成真实 temporal concept。")
    (ROOT / "docs" / "12_distance_metric_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-summary", type=Path, default=ROOT / "outputs" / "chronos_multilayer_validation" / "cluster_validation_summary.json")
    parser.add_argument("--representations", type=str, default="projection,layer_0,layer_6,layer_11")
    parser.add_argument("--k-settings", type=str, default="shared,layer_specific")
    parser.add_argument("--dtw-radius", type=int, default=2)
    parser.add_argument("--dtw-radius-sweep", type=str, default="1,2,3")
    parser.add_argument("--max-patches-per-cluster", type=int, default=120)
    parser.add_argument("--retrieval-top-k", type=int, default=8)
    parser.add_argument("--failure-pair-sample", type=int, default=220)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(args.input_summary.read_text(encoding="utf-8"))
    reps = parse_csv(args.representations)
    settings = make_settings(summary, reps, parse_csv(args.k_settings))
    seed = int(summary.get("seed", 47))
    context_len = int(summary.get("context_len", 128))
    windows_per_dataset = int(summary.get("windows_per_dataset", 500))

    windows, window_meta, dataset_summary = sample_windows(DATA_ROOT, context_len=context_len, windows_per_dataset=windows_per_dataset, seed=seed)
    rep_data = build_rep_data(windows, window_meta, args.batch_size)

    rep_spaces: dict[str, dict[str, Any]] = {}
    for rep in reps:
        data = rep_data[rep]
        idx = select_balanced_indices(
            data.metadata,
            summary.get("balance_mode", "macro_domain"),
            int(summary.get("max_per_macro_domain", 1500)),
            int(summary.get("max_per_dataset_within_macro_domain", 350)),
            700,
            seed,
        )
        metadata = [data.metadata[int(i)] for i in idx]
        for m in metadata:
            if "macro_domain" not in m:
                add_macro_metadata([m])
        x_pca, _scaler, pca = fit_pca_space(data.embeddings[idx], seed)
        patches = data.patches[idx]
        rep_spaces[rep] = {"x_pca": x_pca, "metadata": metadata, "patches": patches, "z": zpatches(patches), "pca": pca}

    rng = np.random.default_rng(seed)
    cluster_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    payload_by_setting: dict[str, dict[str, Any]] = {}
    labels_by_setting: dict[str, np.ndarray] = {}

    for setting in settings:
        rep = setting["rep"]
        k = int(setting["k"])
        space = rep_spaces[rep]
        model, labels = fit_setting(space, k, seed)
        labels_by_setting[setting["setting_id"]] = labels
        rows_for_setting = []
        for cid in sorted(set(labels.tolist())):
            idx = np.where(labels == cid)[0]
            row = cluster_metric_row(
                setting["setting_id"],
                rep,
                k,
                cid,
                idx,
                space["metadata"],
                space["z"],
                args.max_patches_per_cluster,
                args.dtw_radius,
                rng,
            )
            cluster_rows.append(row)
            rows_for_setting.append(row)
        payload_by_setting[setting["setting_id"]] = {
            **space,
            "labels": labels,
            "centers": model.cluster_centers_,
            "radius": args.dtw_radius,
        }
        plot_embedding_cluster_audit(
            setting,
            space["x_pca"][:, :2],
            labels,
            model.cluster_centers_[:, :2],
            space["metadata"],
            rows_for_setting,
            fig_dir,
        )
        plot_prototype_metric_comparison(
            setting,
            space["x_pca"],
            labels,
            model.cluster_centers_,
            space["metadata"],
            space["patches"],
            space["z"],
            args.dtw_radius,
            args.max_patches_per_cluster,
            seed,
            fig_dir,
        )

    plot_distance_heatmap(cluster_rows, fig_dir)

    query_specs = []
    for setting_id, cid in choose_retrieval_queries(cluster_rows, labels_by_setting):
        payload = payload_by_setting[setting_id]
        cluster_idx = np.where(payload["labels"] == cid)[0]
        if len(cluster_idx) == 0:
            continue
        effective_idx, _mode, _flat_share = shape_eligible_indices(cluster_idx, payload["metadata"])
        medoid = medoid_and_neighbors(effective_idx, payload["z"], "dtw", args.dtw_radius, 1)
        if medoid:
            query_specs.append({"setting_id": setting_id, "cluster": cid, "query": medoid[0]})
    retrieval_rows = plot_retrieval_examples(query_specs, payload_by_setting, fig_dir)

    motif_rows: list[dict[str, Any]] = []

    failure_payload = payload_by_setting.get("layer_6_k10") or payload_by_setting[settings[-1]["setting_id"]]
    failure_cases = find_failure_cases(failure_payload, failure_payload["labels"], seed, args.dtw_radius, max_pairs=args.failure_pair_sample)
    plot_failure_cases(failure_cases, failure_payload, fig_dir)

    radius_sensitivity: list[dict[str, Any]] = []
    for radius in [int(x) for x in parse_csv(args.dtw_radius_sweep)]:
        for setting_id, payload in payload_by_setting.items():
            labels = payload["labels"]
            for cid in sorted(set(labels.tolist())):
                idx = np.where(labels == cid)[0]
                sampled = cluster_subset(idx, min(args.max_patches_per_cluster, 60), rng)
                random_idx = metric_baseline_indices(len(payload["z"]), idx, len(sampled), rng)
                dtw_intra = mean_upper(pairwise_metric(payload["z"][sampled], "dtw", radius))
                dtw_base = mean_upper(pairwise_metric(payload["z"][random_idx], "dtw", radius))
                radius_sensitivity.append(
                    {
                        "setting_id": setting_id,
                        "cluster": int(cid),
                        "dtw_radius": int(radius),
                        "dtw_ratio": float(dtw_intra / max(dtw_base, 1e-8)),
                    }
                )

    write_csv(out_dir / "cluster_metric_table.csv", cluster_rows)
    write_csv(out_dir / "retrieval_comparison_table.csv", retrieval_rows)
    summary_out = {
        "objective": "DTW vs Euclidean original-space ablation for Chronos-2 patch clusters",
        "input_summary": str(args.input_summary),
        "settings": settings,
        "dtw_radius": args.dtw_radius,
        "dtw_radius_sensitivity": radius_sensitivity,
        "max_patches_per_cluster": args.max_patches_per_cluster,
        "num_windows": int(len(windows)),
        "dataset_summary": dataset_summary,
        "cluster_rows": cluster_rows,
        "motif_family_rows": motif_rows,
        "failure_cases": failure_cases,
        "output_files": {
            "cluster_metric_table": str(out_dir / "cluster_metric_table.csv"),
            "retrieval_comparison_table": str(out_dir / "retrieval_comparison_table.csv"),
            "figures": str(fig_dir),
            "report": str(ROOT / "docs" / "12_distance_metric_ablation_report.md"),
        },
    }
    save_json(out_dir / "distance_ablation_summary.json", summary_out)
    write_report(out_dir, settings, cluster_rows, motif_rows, failure_cases)
    print(json.dumps({"summary": str(out_dir / "distance_ablation_summary.json"), "settings": [s["setting_id"] for s in settings]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
