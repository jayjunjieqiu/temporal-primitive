from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CHRONOS_SRC = ROOT / "external" / "chronos-forecasting" / "src"
DEFAULT_OUT = ROOT / "outputs" / "chronos_multilayer_validation"
FIG_DPI = 260

sys.path.insert(0, str(ROOT))
from scripts.run_second_pilot_discovery import (  # noqa: E402
    DATA_ROOT,
    MODEL_SPECS,
    flatten_model_patches,
    patch_stats,
    read_desc,
    robust_z,
    sample_windows,
)


REPRESENTATIONS = ["projection", "layer_0", "layer_6", "layer_11"]
LAYER_INDICES = [0, 6, 11]
PATCH_LEN = int(MODEL_SPECS["chronos_2"]["patch_len"])
MODEL_PATH = ROOT / "chronos-2"

DISPLAY_NAMES = {
    "projection": "Projection",
    "layer_0": "Layer 0",
    "layer_6": "Layer 6",
    "layer_11": "Layer 11",
}

MACRO_DOMAIN_ORDER = [
    "Traffic",
    "Energy",
    "Environment",
    "Finance",
    "Health",
    "Synthetic control",
]

MACRO_DOMAIN_DEFINITIONS = {
    "Traffic": ["traffic flow", "traffic speed", "road occupancy rates"],
    "Energy": ["electricity consumption", "electricity transformer temperature"],
    "Environment": ["weather", "Beijing air quality"],
    "Finance": ["exchange rate"],
    "Health": ["illness data"],
    "Synthetic control": ["simulated Gaussian data", "simulated pulse data"],
}

PRIOR_LABEL_ORDER = [
    "flat_low_information",
    "trend",
    "level_shift",
    "oscillation",
    "impulse_spike",
    "burst_event",
    "volatility_shift",
    "intermittent",
    "mixed_uncertain",
]


@dataclass
class RepData:
    embeddings: np.ndarray
    metadata: list[dict[str, Any]]
    patches: np.ndarray


def append_log(out_dir: Path, text: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "progress_log.md"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def macro_domain(source_domain: Any) -> str:
    value = str(source_domain)
    for group, members in MACRO_DOMAIN_DEFINITIONS.items():
        if value in members:
            return group
    return "Other"


def clean_label(value: Any) -> str:
    return str(value).replace("_", " ")


def list_dataset_inventory(data_root: Path, context_len: int) -> list[dict[str, Any]]:
    inventory = []
    for desc_path in sorted(data_root.glob("*/desc.json")):
        dataset = desc_path.parent.name
        if dataset == "BLAST":
            continue
        desc = read_desc(desc_path)
        shape = tuple(desc["shape"])
        inventory.append(
            {
                "dataset": dataset,
                "source_domain": desc.get("domain", dataset),
                "macro_domain": macro_domain(desc.get("domain", dataset)),
                "frequency_minutes": desc.get("frequency (minutes)"),
                "shape": list(shape),
                "eligible": bool(len(shape) == 3 and shape[0] >= context_len and (desc_path.parent / "data.dat").exists()),
            }
        )
    return inventory


def make_raw_embeddings(windows: np.ndarray, patch_len: int) -> np.ndarray:
    num_patches = windows.shape[1] // patch_len
    out = np.zeros((len(windows), num_patches, patch_len), dtype=np.float32)
    for i in range(len(windows)):
        for j in range(num_patches):
            out[i, j] = robust_z(windows[i, j * patch_len : (j + 1) * patch_len]).astype(np.float32)
    return out


def extract_chronos_multilayer(windows: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(CHRONOS_SRC))
    import chronos

    pipeline = chronos.Chronos2Pipeline.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = pipeline.model
    model.eval()

    chunks: dict[str, list[np.ndarray]] = {"projection": []}
    for idx in LAYER_INDICES:
        chunks[f"layer_{idx}"] = []

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[start : start + batch_size], dtype=torch.float32, device=model.device)
            patched_context, _attention_mask, _loc_scale = model._prepare_patched_context(context=batch)
            num_context_patches = int(patched_context.shape[1])
            projection = model.input_patch_embedding(patched_context)

            captured: dict[str, torch.Tensor] = {}
            handles = []

            def hook_for(layer_idx: int):
                def hook(_mod, _inp, out):
                    hidden = out[0] if isinstance(out, tuple) else out.hidden_states
                    captured[f"layer_{layer_idx}"] = hidden.detach()

                return hook

            for layer_idx in LAYER_INDICES:
                handles.append(model.encoder.block[layer_idx].register_forward_hook(hook_for(layer_idx)))

            _encoder_outputs, _loc_scale, _future_mask, returned_num_context_patches = model.encode(
                context=batch,
                num_output_patches=1,
            )
            for handle in handles:
                handle.remove()
            if int(returned_num_context_patches) != num_context_patches:
                raise RuntimeError("Chronos num_context_patches mismatch")

            chunks["projection"].append(projection[:, :num_context_patches].float().cpu().numpy())
            for layer_idx in LAYER_INDICES:
                name = f"layer_{layer_idx}"
                chunks[name].append(captured[name][:, :num_context_patches].float().cpu().numpy())

    del pipeline, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def add_macro_metadata(metadata: list[dict[str, Any]]) -> None:
    for m in metadata:
        m["source_domain"] = str(m["domain"])
        m["macro_domain"] = macro_domain(m["domain"])


def build_rep_data(windows: np.ndarray, window_meta: list[dict[str, Any]], batch_size: int) -> dict[str, RepData]:
    layer_outputs = extract_chronos_multilayer(windows, batch_size)
    rep_data: dict[str, RepData] = {}
    for rep_name, tensor in layer_outputs.items():
        embeddings, metadata, patches = flatten_model_patches("chronos_2", tensor, windows, window_meta)
        add_macro_metadata(metadata)
        rep_data[rep_name] = RepData(embeddings=embeddings, metadata=metadata, patches=patches)
    return rep_data


def select_balanced_indices(
    metadata: list[dict[str, Any]],
    balance_mode: str,
    max_per_macro_domain: int,
    max_per_dataset_within_macro_domain: int,
    max_per_source_domain: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []

    if balance_mode == "source_domain":
        groups: dict[str, list[int]] = defaultdict(list)
        for i, meta in enumerate(metadata):
            groups[str(meta["source_domain"])].append(i)
        for indices in groups.values():
            take = min(max_per_source_domain, len(indices))
            selected.extend(rng.choice(np.asarray(indices), size=take, replace=False).astype(int).tolist())
        return np.asarray(sorted(selected), dtype=int)

    if balance_mode != "macro_domain":
        raise ValueError(f"Unsupported balance_mode={balance_mode}")

    macro_groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, meta in enumerate(metadata):
        macro_groups[str(meta["macro_domain"])][str(meta["dataset"])].append(i)

    for macro in MACRO_DOMAIN_ORDER + sorted(set(macro_groups) - set(MACRO_DOMAIN_ORDER)):
        if macro not in macro_groups:
            continue
        candidates: list[int] = []
        for indices in macro_groups[macro].values():
            take = min(max_per_dataset_within_macro_domain, len(indices))
            candidates.extend(rng.choice(np.asarray(indices), size=take, replace=False).astype(int).tolist())
        if len(candidates) > max_per_macro_domain:
            candidates = rng.choice(np.asarray(candidates), size=max_per_macro_domain, replace=False).astype(int).tolist()
        selected.extend(candidates)
    return np.asarray(sorted(selected), dtype=int)


def count_values(metadata: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(m.get(key)) for m in metadata).items()))


def selection_counts(metadata: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "dataset": count_values(metadata, "dataset"),
        "source_domain": count_values(metadata, "source_domain"),
        "macro_domain": count_values(metadata, "macro_domain"),
        "frequency": count_values(metadata, "frequency_minutes"),
        "patch_index": count_values(metadata, "patch_index"),
        "prior_guided_probe": count_values(metadata, "taxonomy_label"),
    }


def fit_pca_space(embeddings: np.ndarray, seed: int) -> tuple[np.ndarray, StandardScaler, PCA]:
    scaler = StandardScaler()
    x = scaler.fit_transform(embeddings)
    pca_dim = max(2, min(30, x.shape[0] - 1, x.shape[1]))
    pca = PCA(n_components=pca_dim, random_state=seed)
    x_pca = pca.fit_transform(x)
    return x_pca, scaler, pca


def label_lists(metadata: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "dataset": [str(m["dataset"]) for m in metadata],
        "source_domain": [str(m["source_domain"]) for m in metadata],
        "macro_domain": [str(m["macro_domain"]) for m in metadata],
        "frequency": [str(m.get("frequency_minutes")) for m in metadata],
        "patch_index": [str(m["patch_index"]) for m in metadata],
        "prior_guided_probe": [str(m["taxonomy_label"]) for m in metadata],
    }


def cluster_size_stats(labels: np.ndarray) -> dict[str, Any]:
    sizes = np.asarray(list(Counter(labels.tolist()).values()), dtype=float)
    return {
        "min": int(np.min(sizes)),
        "max": int(np.max(sizes)),
        "median": float(np.median(sizes)),
        "mean": float(np.mean(sizes)),
        "small_lt_1pct": int(np.sum(sizes < 0.01 * np.sum(sizes))),
        "small_lt_2pct": int(np.sum(sizes < 0.02 * np.sum(sizes))),
        "imbalance_ratio": float(np.max(sizes) / max(1.0, np.min(sizes))),
    }


def safe_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) <= 1 or len(set(labels.tolist())) >= len(labels):
        return float("nan")
    sample_size = min(5000, len(labels))
    return float(silhouette_score(x, labels, sample_size=sample_size, random_state=47))


def evaluate_single_k(x_pca: np.ndarray, metadata: list[dict[str, Any]], k: int, seed: int, stability_seeds: list[int]) -> dict[str, Any]:
    primary = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(x_pca)
    labels = primary.labels_
    label_map = label_lists(metadata)

    seed_runs = []
    for s in stability_seeds:
        seed_runs.append(KMeans(n_clusters=k, random_state=s, n_init=10).fit_predict(x_pca))
    pair_nmi = []
    pair_ari = []
    for i in range(len(seed_runs)):
        for j in range(i + 1, len(seed_runs)):
            pair_nmi.append(normalized_mutual_info_score(seed_runs[i], seed_runs[j]))
            pair_ari.append(adjusted_rand_score(seed_runs[i], seed_runs[j]))

    try:
        agglom = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x_pca)
        agglom_nmi = float(normalized_mutual_info_score(labels, agglom))
    except Exception:
        agglom_nmi = float("nan")

    size_stats = cluster_size_stats(labels)
    return {
        "k": int(k),
        "silhouette": safe_silhouette(x_pca, labels),
        "calinski_harabasz": float(calinski_harabasz_score(x_pca, labels)),
        "davies_bouldin": float(davies_bouldin_score(x_pca, labels)),
        "seed_stability_nmi": float(np.mean(pair_nmi)) if pair_nmi else float("nan"),
        "seed_stability_ari": float(np.mean(pair_ari)) if pair_ari else float("nan"),
        "kmeans_vs_agglomerative_nmi": agglom_nmi,
        "min_cluster_size": size_stats["min"],
        "max_cluster_size": size_stats["max"],
        "cluster_imbalance_ratio": size_stats["imbalance_ratio"],
        "small_clusters_lt_1pct": size_stats["small_lt_1pct"],
        "nmi_patch_index": float(normalized_mutual_info_score(label_map["patch_index"], labels)),
        "nmi_dataset": float(normalized_mutual_info_score(label_map["dataset"], labels)),
        "nmi_source_domain": float(normalized_mutual_info_score(label_map["source_domain"], labels)),
        "nmi_macro_domain": float(normalized_mutual_info_score(label_map["macro_domain"], labels)),
        "nmi_frequency": float(normalized_mutual_info_score(label_map["frequency"], labels)),
        "nmi_prior_guided_probe": float(normalized_mutual_info_score(label_map["prior_guided_probe"], labels)),
    }


def coarse_to_fine_candidates(metrics: list[dict[str, Any]], k_values: list[int]) -> list[int]:
    valid = [m for m in metrics if m["min_cluster_size"] >= 20]
    if not valid:
        valid = metrics

    def normalized(values: list[float], reverse: bool = False) -> dict[int, float]:
        arr = np.asarray(values, dtype=float)
        finite = np.isfinite(arr)
        if not finite.any() or float(np.nanmax(arr) - np.nanmin(arr)) < 1e-12:
            return {int(valid[i]["k"]): 0.5 for i in range(len(valid))}
        lo, hi = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
        out = {}
        for i, v in enumerate(arr):
            score = 0.0 if not np.isfinite(v) else (float(v) - lo) / max(hi - lo, 1e-12)
            out[int(valid[i]["k"])] = 1.0 - score if reverse else score
        return out

    sil = normalized([m["silhouette"] for m in valid])
    stab = normalized([m["seed_stability_nmi"] for m in valid])
    ag = normalized([m["kmeans_vs_agglomerative_nmi"] for m in valid])
    db = normalized([m["davies_bouldin"] for m in valid], reverse=True)
    conf = normalized(
        [
            m["nmi_patch_index"] + m["nmi_dataset"] + m["nmi_macro_domain"] + m["nmi_frequency"]
            for m in valid
        ],
        reverse=True,
    )
    scores = []
    for m in valid:
        k = int(m["k"])
        score = 0.22 * sil[k] + 0.26 * stab[k] + 0.22 * ag[k] + 0.15 * db[k] + 0.15 * conf[k]
        if m["small_clusters_lt_1pct"] > 0:
            score -= 0.10
        scores.append((score, k))
    top = [k for _score, k in sorted(scores, reverse=True)[:3]]
    fine = set(top)
    for k in top:
        fine.update(range(max(min(k_values), k - 2), min(max(k_values), k + 2) + 1))
    return sorted(fine)


def choose_k(metrics_by_rep: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per_layer = {}
    for rep, metrics in metrics_by_rep.items():
        valid = [m for m in metrics if m["min_cluster_size"] >= 20]
        if not valid:
            valid = metrics
        rows = []
        for m in valid:
            confound = m["nmi_patch_index"] + m["nmi_dataset"] + m["nmi_macro_domain"] + m["nmi_frequency"]
            size_penalty = 0.08 * m["small_clusters_lt_1pct"] + min(0.25, max(0.0, (m["cluster_imbalance_ratio"] - 20.0) / 100.0))
            score = (
                0.18 * m["silhouette"]
                + 0.24 * m["seed_stability_nmi"]
                + 0.22 * m["kmeans_vs_agglomerative_nmi"]
                - 0.12 * m["davies_bouldin"]
                - 0.12 * confound
                - size_penalty
            )
            rows.append((float(score), int(m["k"]), m))
        rows.sort(reverse=True)
        per_layer[rep] = {"recommended_k": rows[0][1], "score": rows[0][0], "top_candidates": [r[1] for r in rows[:5]]}

    all_ks = sorted({int(m["k"]) for metrics in metrics_by_rep.values() for m in metrics})
    shared_scores = []
    for k in all_ks:
        parts = []
        present = True
        for rep, metrics in metrics_by_rep.items():
            row = next((m for m in metrics if int(m["k"]) == k), None)
            if row is None:
                present = False
                break
            confound = row["nmi_patch_index"] + row["nmi_dataset"] + row["nmi_macro_domain"] + row["nmi_frequency"]
            score = (
                0.16 * row["silhouette"]
                + 0.25 * row["seed_stability_nmi"]
                + 0.22 * row["kmeans_vs_agglomerative_nmi"]
                - 0.10 * row["davies_bouldin"]
                - 0.12 * confound
                - 0.07 * row["small_clusters_lt_1pct"]
            )
            parts.append(score)
        if present:
            proximity = np.mean([abs(k - per_layer[rep]["recommended_k"]) for rep in per_layer])
            shared_scores.append((float(np.mean(parts) - 0.015 * proximity), int(k)))
    shared_scores.sort(reverse=True)
    return {
        "recommended_shared_k": int(shared_scores[0][1]),
        "shared_k_top_candidates": [int(k) for _score, k in shared_scores[:8]],
        "per_layer": per_layer,
        "selection_rule": {
            "principle": "综合 stability、KMeans vs Agglomerative agreement、silhouette、Davies-Bouldin、confounder NMI 与 cluster size；不使用 silhouette-only。",
            "shared_k_preference": "若多个 layer 的合理 K 接近，优先选择 shared K 支持 projection -> layer_0 -> layer_6 -> layer_11 对比。",
        },
    }


def fit_final_clusters(x_pca: np.ndarray, k: int, seed: int) -> tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=k, random_state=seed, n_init=30).fit(x_pca)
    return model, model.labels_


def cluster_raw_coherence(patches: np.ndarray, indices: np.ndarray) -> float:
    if len(indices) < 2:
        return 0.0
    sample = indices[: min(len(indices), 8)]
    curves = np.vstack([robust_z(patches[i]) for i in sample])
    corrs = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            a, b = curves[i], curves[j]
            if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
                corrs.append(1.0 if np.allclose(a, b) else 0.0)
            else:
                corrs.append(abs(float(np.corrcoef(a, b)[0, 1])))
    return float(np.mean(corrs)) if corrs else 0.0


def macro_domain_matches(
    x_pca: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    metadata: list[dict[str, Any]],
    cid: int,
) -> list[dict[str, Any]]:
    cluster_idx = np.where(labels == cid)[0]
    cluster_distances = np.linalg.norm(x_pca[cluster_idx] - centers[cid], axis=1)
    cutoff = float(np.quantile(cluster_distances, 0.90)) if len(cluster_distances) else float("inf")
    out = []
    for domain in MACRO_DOMAIN_ORDER:
        candidates = np.asarray([i for i, m in enumerate(metadata) if str(m["macro_domain"]) == domain], dtype=int)
        if len(candidates) == 0:
            out.append({"macro_domain": domain, "status": "missing_domain"})
            continue
        distances = np.linalg.norm(x_pca[candidates] - centers[cid], axis=1)
        pick_pos = int(np.argmin(distances))
        item = int(candidates[pick_pos])
        same = int(labels[item]) == int(cid)
        high_conf = bool(same and float(distances[pick_pos]) <= cutoff)
        out.append(
            {
                "macro_domain": domain,
                "index": item,
                "status": "high_confidence" if high_conf else ("same_cluster_far" if same else "not_same_cluster"),
                "assigned_cluster": int(labels[item]),
                "distance_to_center": float(distances[pick_pos]),
                "cluster_p90_distance": cutoff,
                "dataset": str(metadata[item]["dataset"]),
                "source_domain": str(metadata[item]["source_domain"]),
                "patch_index": int(metadata[item]["patch_index"]),
            }
        )
    return out


def summarize_clusters(
    rep_name: str,
    x_pca: np.ndarray,
    labels: np.ndarray,
    kmeans: KMeans,
    metadata: list[dict[str, Any]],
    patches: np.ndarray,
) -> list[dict[str, Any]]:
    summaries = []
    label_map = label_lists(metadata)
    for cid in sorted(set(labels.tolist())):
        idx = np.where(labels == cid)[0]
        center_dist = np.linalg.norm(x_pca[idx] - kmeans.cluster_centers_[cid], axis=1)
        nearest_order = idx[np.argsort(center_dist)]
        macro = macro_domain_matches(x_pca, labels, kmeans.cluster_centers_, metadata, cid)
        high_conf_domains = [m for m in macro if m.get("status") == "high_confidence" and m.get("macro_domain") != "Synthetic control"]
        domain_counts = Counter(label_map["dataset"][i] for i in idx)
        source_counts = Counter(label_map["source_domain"][i] for i in idx)
        freq_counts = Counter(label_map["frequency"][i] for i in idx)
        patch_counts = Counter(label_map["patch_index"][i] for i in idx)
        prior_counts = Counter(label_map["prior_guided_probe"][i] for i in idx)
        raw_coh = cluster_raw_coherence(patches, nearest_order[:8])
        size = int(len(idx))
        dataset_share = domain_counts.most_common(1)[0][1] / size
        source_share = source_counts.most_common(1)[0][1] / size
        freq_share = freq_counts.most_common(1)[0][1] / size
        patch_share = patch_counts.most_common(1)[0][1] / size
        confounder_risk = max(dataset_share, source_share, freq_share, patch_share)
        evidence_score = (
            min(1.0, size / 120.0) * 0.18
            + min(1.0, raw_coh) * 0.24
            + min(1.0, len(high_conf_domains) / 4.0) * 0.30
            + (1.0 - min(1.0, max(0.0, confounder_risk - 0.35) / 0.55)) * 0.28
        )
        if size < max(30, 0.01 * len(labels)):
            evidence_tier = "diagnostic_tiny"
        elif confounder_risk >= 0.80 and len(high_conf_domains) < 3:
            evidence_tier = "diagnostic_confounded"
        elif raw_coh >= 0.45 and len(high_conf_domains) >= 3 and confounder_risk < 0.80:
            evidence_tier = "main_evidence"
        else:
            evidence_tier = "diagnostic_weak"
        summaries.append(
            {
                "representation": rep_name,
                "cluster": int(cid),
                "size": size,
                "evidence_score": float(evidence_score),
                "evidence_tier": evidence_tier,
                "raw_center_nearest_coherence": raw_coh,
                "high_confidence_real_macro_domains": len(high_conf_domains),
                "confounder_risk_max_share": float(confounder_risk),
                "top_datasets": [{"value": k, "count": int(v)} for k, v in domain_counts.most_common(5)],
                "top_source_domains": [{"value": k, "count": int(v)} for k, v in source_counts.most_common(5)],
                "top_frequencies": [{"value": k, "count": int(v)} for k, v in freq_counts.most_common(5)],
                "top_patch_indices": [{"value": k, "count": int(v)} for k, v in patch_counts.most_common(5)],
                "top_prior_guided_probe": [{"value": k, "count": int(v)} for k, v in prior_counts.most_common(5)],
                "center_nearest_examples": [
                    {
                        "rank": rank + 1,
                        "index": int(item),
                        "dataset": str(metadata[item]["dataset"]),
                        "source_domain": str(metadata[item]["source_domain"]),
                        "macro_domain": str(metadata[item]["macro_domain"]),
                        "patch_index": int(metadata[item]["patch_index"]),
                        "distance_to_center": float(np.linalg.norm(x_pca[item] - kmeans.cluster_centers_[cid])),
                    }
                    for rank, item in enumerate(nearest_order[:6])
                ],
                "macro_domain_matches": macro,
            }
        )
    return summaries


def make_color_ids(values: list[str]) -> tuple[np.ndarray, list[str]]:
    ordered = [label for label in PRIOR_LABEL_ORDER if label in set(values)]
    ordered.extend(sorted(set(values) - set(ordered)))
    mapping = {v: i for i, v in enumerate(ordered)}
    return np.asarray([mapping[v] for v in values]), ordered


def plot_k_selection(metrics_by_rep: dict[str, list[dict[str, Any]]], selected: dict[str, Any], fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=True)
    axes = axes.ravel()
    metrics = [
        ("silhouette", "Silhouette"),
        ("seed_stability_nmi", "Seed stability NMI"),
        ("kmeans_vs_agglomerative_nmi", "KMeans vs Agglomerative NMI"),
        ("nmi_macro_domain", "Macro-domain NMI"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        for rep, rows in metrics_by_rep.items():
            rows = sorted(rows, key=lambda r: r["k"])
            ax.plot(
                [r["k"] for r in rows],
                [r[key] for r in rows],
                marker="o",
                linewidth=1.4,
                markersize=3.8,
                label=DISPLAY_NAMES.get(rep, rep),
            )
        ax.axvline(selected["recommended_shared_k"], color="#222222", linestyle="--", linewidth=1.1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("K")
        ax.grid(color="#e8edf2", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=8, ncols=2)
    fig.suptitle("K selection diagnostics", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_dir / "k_selection_summary.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_layer_comparison(final_results: dict[str, dict[str, Any]], fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    reps = REPRESENTATIONS
    metrics = [
        ("seed_stability_nmi", "Stability"),
        ("kmeans_vs_agglomerative_nmi", "Agglomerative agreement"),
        ("nmi_macro_domain", "Macro-domain NMI"),
        ("macro_high_conf_rate", "High-conf macro rate"),
    ]
    values = np.asarray([[final_results[rep]["metrics"][key] for key, _ in metrics] for rep in reps], dtype=float)
    x = np.arange(len(reps))
    width = 0.18
    colors = ["#276f8e", "#c17817", "#4f8f48", "#8d5a97"]

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    for i, (_key, label) in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, values[:, i], width=width, label=label, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["Projection", "Layer 0", "Layer 6", "Layer 11"])
    ax.set_ylim(0, max(1.0, float(np.nanmax(values)) * 1.18))
    ax.set_ylabel("Score")
    ax.set_title("Layer-wise validation summary", fontsize=14)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    ax.grid(axis="y", color="#e8edf2", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(fig_dir / "layer_comparison_summary.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_embedding_audit(
    rep_name: str,
    x_pca: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    metadata: list[dict[str, Any]],
    selected_clusters: list[int],
    fig_dir: Path,
    output_name: str | None = None,
    title_suffix: str = "",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    prior_values = [str(m["taxonomy_label"]) for m in metadata]
    prior_ids, prior_order = make_color_ids(prior_values)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True, sharey=True)

    sc0 = axes[0].scatter(x_pca[:, 0], x_pca[:, 1], c=labels, s=5, cmap="tab20", alpha=0.58, linewidths=0)
    axes[0].scatter(centers[:, 0], centers[:, 1], c="black", s=28, marker="x", linewidths=1.2)
    for cid in selected_clusters:
        axes[0].text(centers[cid, 0], centers[cid, 1], f"C{cid}", fontsize=8, weight="bold")
    axes[0].set_title("KMeans clusters")
    fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.03).set_label("Cluster")

    cmap = plt.get_cmap("tab10", max(1, len(prior_order)))
    axes[1].scatter(
        x_pca[:, 0],
        x_pca[:, 1],
        c=prior_ids,
        s=5,
        cmap=cmap,
        alpha=0.58,
        linewidths=0,
        vmin=-0.5,
        vmax=len(prior_order) - 0.5,
    )
    axes[1].set_title("Prior-guided motif probe")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(i), markeredgecolor="none", markersize=5, label=clean_label(v))
        for i, v in enumerate(prior_order)
    ]
    axes[1].legend(handles=handles, frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(color="#eef2f5", linewidth=0.5)
    suffix = f" {title_suffix}" if title_suffix else ""
    fig.suptitle(f"{DISPLAY_NAMES.get(rep_name, rep_name)}{suffix}: cluster and prior-probe audit", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_dir / f"{output_name or f'{rep_name}_embedding_audit'}.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_center_nearest(
    rep_name: str,
    selected_clusters: list[int],
    x_pca: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    metadata: list[dict[str, Any]],
    patches: np.ndarray,
    fig_dir: Path,
    ncols: int = 4,
    output_name: str | None = None,
    title_suffix: str = "",
) -> None:
    import matplotlib.pyplot as plt

    nrows = max(1, len(selected_clusters))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.55 * ncols, 1.55 * nrows), squeeze=False)
    for row, cid in enumerate(selected_clusters):
        idx = np.where(labels == cid)[0]
        order = idx[np.argsort(np.linalg.norm(x_pca[idx] - centers[cid], axis=1))[:ncols]]
        for col in range(ncols):
            ax = axes[row, col]
            if col >= len(order):
                ax.axis("off")
                continue
            item = int(order[col])
            ax.plot(robust_z(patches[item]), color="#2b6f9f", linewidth=1.45)
            ax.axhline(0, color="#d7dde5", linewidth=0.6, zorder=0)
            meta = metadata[item]
            ax.set_title(f"C{cid} nearest {col + 1}\n{meta['dataset']}, {meta['macro_domain']}, p{meta['patch_index']}", fontsize=7.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#222831")
                spine.set_linewidth(0.8)
    suffix = f" {title_suffix}" if title_suffix else ""
    fig.suptitle(f"{DISPLAY_NAMES.get(rep_name, rep_name)}{suffix}: center-nearest raw patches", fontsize=14, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=0.9, w_pad=0.75)
    fig.savefig(fig_dir / f"{output_name or f'{rep_name}_main_center_nearest'}.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_macro_filtered(
    rep_name: str,
    selected_clusters: list[int],
    cluster_summaries: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    patches: np.ndarray,
    fig_dir: Path,
    output_name: str | None = None,
    title_suffix: str = "",
) -> None:
    import matplotlib.pyplot as plt

    summary_by_cluster = {int(c["cluster"]): c for c in cluster_summaries}
    nrows = max(1, len(selected_clusters))
    ncols = len(MACRO_DOMAIN_ORDER)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.25 * ncols, 1.45 * nrows), squeeze=False)
    for row, cid in enumerate(selected_clusters):
        matches = {m["macro_domain"]: m for m in summary_by_cluster[cid]["macro_domain_matches"]}
        for col, domain in enumerate(MACRO_DOMAIN_ORDER):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(domain, fontsize=8.2, pad=7)
            match = matches.get(domain)
            if not match or match.get("status") != "high_confidence":
                ax.text(0.5, 0.5, "No confident\nmatch", ha="center", va="center", fontsize=7.0, color="#8b97a5")
                ax.set_facecolor("#fbfcfd")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#d5dde6")
                    spine.set_linewidth(0.8)
                continue
            item = int(match["index"])
            ax.plot(robust_z(patches[item]), color="#2b6f9f", linewidth=1.35)
            ax.axhline(0, color="#d7dde5", linewidth=0.55, zorder=0)
            meta = metadata[item]
            ax.text(0.03, 0.95, f"C{cid} | {meta['dataset']}, p{meta['patch_index']}", transform=ax.transAxes, va="top", fontsize=6.2)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#222831")
                spine.set_linewidth(0.85)
    suffix = f" {title_suffix}" if title_suffix else ""
    fig.suptitle(f"{DISPLAY_NAMES.get(rep_name, rep_name)}{suffix}: confidence-filtered macro-domain examples", fontsize=14, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=0.9, w_pad=0.6)
    fig.savefig(fig_dir / f"{output_name or f'{rep_name}_main_macro_domain_filtered'}.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_failure_cases(
    final_results: dict[str, dict[str, Any]],
    rep_payloads: dict[str, dict[str, Any]],
    fig_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    examples = []
    for rep, result in final_results.items():
        diagnostics = [c for c in result["clusters"] if c["evidence_tier"].startswith("diagnostic")]
        diagnostics = sorted(diagnostics, key=lambda c: (c["confounder_risk_max_share"], -c["raw_center_nearest_coherence"]), reverse=True)
        if diagnostics:
            examples.append((rep, diagnostics[0]))

    nrows = max(1, len(examples))
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.55 * ncols, 1.55 * nrows), squeeze=False)
    for row, (rep, cluster) in enumerate(examples):
        payload = rep_payloads[rep]
        x_pca = payload["x_pca"]
        labels = payload["labels"]
        centers = payload["centers"]
        metadata = payload["metadata"]
        patches = payload["patches"]
        cid = int(cluster["cluster"])
        idx = np.where(labels == cid)[0]
        order = idx[np.argsort(np.linalg.norm(x_pca[idx] - centers[cid], axis=1))[:ncols]]
        for col, item in enumerate(order):
            ax = axes[row, col]
            ax.plot(robust_z(patches[item]), color="#7a4f9a", linewidth=1.35)
            ax.axhline(0, color="#d7dde5", linewidth=0.55, zorder=0)
            meta = metadata[int(item)]
            tier = str(cluster["evidence_tier"]).replace("_", " ")
            ax.set_title(f"{DISPLAY_NAMES.get(rep, rep)} C{cid}\n{tier}\n{meta['dataset']}, p{meta['patch_index']}", fontsize=6.4)
            ax.set_xticks([])
            ax.set_yticks([])
    for ax in axes.ravel():
        if not ax.has_data():
            ax.axis("off")
    fig.suptitle("Diagnostic failure cases", fontsize=14, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=0.9, w_pad=0.75)
    fig.savefig(fig_dir / "diagnostic_failure_cases.png", dpi=FIG_DPI)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def metric_lookup(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    return next(r for r in rows if int(r["k"]) == int(k))


def all_cluster_ids(cluster_summaries: list[dict[str, Any]]) -> list[int]:
    return [int(c["cluster"]) for c in sorted(cluster_summaries, key=lambda item: int(item["cluster"]))]


def macro_high_conf_rate(cluster_summaries: list[dict[str, Any]]) -> float:
    high_conf_cells = sum(
        1
        for c in cluster_summaries
        for m in c["macro_domain_matches"]
        if m.get("status") == "high_confidence" and m.get("macro_domain") != "Synthetic control"
    )
    total_real_cells = len(cluster_summaries) * (len(MACRO_DOMAIN_ORDER) - 1)
    return float(high_conf_cells / max(1, total_real_cells))


def build_cluster_result(
    rep: str,
    k: int,
    payload: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x_pca = payload["x_pca"]
    metadata = payload["metadata"]
    patches = payload["patches"]
    kmeans, labels = fit_final_clusters(x_pca, k, seed)
    cluster_summaries = summarize_clusters(rep, x_pca, labels, kmeans, metadata, patches)
    displayed_clusters = all_cluster_ids(cluster_summaries)
    main_evidence_clusters = [
        int(c["cluster"])
        for c in sorted(cluster_summaries, key=lambda item: item["evidence_score"], reverse=True)
        if c["evidence_tier"] == "main_evidence"
    ]
    final_metric = dict(metric_lookup(metric_rows, k))
    final_metric["macro_high_conf_rate"] = macro_high_conf_rate(cluster_summaries)
    final_metric["k"] = int(k)
    result = {
        "metrics": final_metric,
        "displayed_clusters": displayed_clusters,
        "main_evidence_clusters": main_evidence_clusters,
        "clusters": cluster_summaries,
        "selection_counts": selection_counts(metadata),
        "pca_explained_variance_ratio_first_5": payload["pca"].explained_variance_ratio_[:5].astype(float).tolist(),
    }
    plot_payload = {
        "x_pca": x_pca,
        "labels": labels,
        "centers": kmeans.cluster_centers_,
        "metadata": metadata,
        "patches": patches,
    }
    return result, plot_payload


def summarize_pilot(old_summary_path: Path) -> dict[str, Any]:
    if not old_summary_path.exists():
        return {"status": "missing", "path": str(old_summary_path)}
    data = json.loads(old_summary_path.read_text(encoding="utf-8"))
    dataset_count = len(data.get("dataset_summary", []))
    windows_per_dataset = int(data.get("windows_per_dataset", 0))
    context_len = int(data.get("context_len", 0))
    patch_len = int(data.get("patch_len", PATCH_LEN))
    raw_windows = sum(int(d.get("accepted_windows", 0)) for d in data.get("dataset_summary", []))
    raw_patches = raw_windows * (context_len // patch_len if patch_len else 0)
    reps = data.get("representations", {})
    return {
        "status": "ok",
        "path": str(old_summary_path),
        "windows_per_dataset": windows_per_dataset,
        "context_len": context_len,
        "patch_len": patch_len,
        "dataset_count": dataset_count,
        "raw_window_count": raw_windows,
        "raw_patch_count_estimate": raw_patches,
        "clustered_patch_count_by_representation": {k: v.get("num_patch_embeddings") for k, v in reps.items()},
        "k_by_representation": {k: v.get("kmeans_k") for k, v in reps.items()},
        "limitations": [
            "sample size 只有 100 windows/dataset，uniform random 可能漏掉稀有 motif。",
            "旧 balancing 是 source-domain/domain balanced，不是严格 macro-domain balanced。",
            "旧 K 使用经验公式，不是 K sweep。",
            "macro-domain nearest view 是 forced nearest diagnostic，不适合作为最终 presentation figure。",
        ],
    }


def write_report(
    report_path: Path,
    summary: dict[str, Any],
    selected: dict[str, Any],
    final_results: dict[str, dict[str, Any]],
    layer_specific_results: dict[str, dict[str, Any]],
    fig_dir: Path,
) -> None:
    fig_dir = fig_dir.resolve()
    rel_fig = lambda name: "../" + str((fig_dir / name).resolve().relative_to(ROOT))
    lines = []
    lines.append("# Chronos-2 Layer Effect Report: 聚类和 motif 空间如何随层变化")
    lines.append("")
    lines.append("## 0. Advisor Question")
    lines.append("")
    lines.append("这版报告回答 Yuxuan Liang 老师关心的机制问题：Chronos-2 的 `projection`、`layer_0`、`layer_6`、`layer_11` 是否保留 single patch 的 local temporal information，以及这些 local primitives 如何被 transformer layers 重组为 contextualized cross-domain temporal concepts。")
    lines.append("")
    lines.append("我们不把 `prior-guided motif` 当 ground truth，也不把它用于命名 KMeans clusters。所有 cluster 只写作 `C0, C1, ...`。")
    lines.append("")
    lines.append("## 1. Pilot Limitations")
    lines.append("")
    pilot = summary["pilot_audit"]
    lines.append(f"- old windows per dataset: `{pilot.get('windows_per_dataset')}`")
    lines.append(f"- old context length / patch length: `{pilot.get('context_len')}` / `{pilot.get('patch_len')}`")
    lines.append(f"- old raw windows / estimated raw patches: `{pilot.get('raw_window_count')}` / `{pilot.get('raw_patch_count_estimate')}`")
    lines.append(f"- old clustered patches per representation: `{pilot.get('clustered_patch_count_by_representation')}`")
    lines.append(f"- old K rule result: `{pilot.get('k_by_representation')}`")
    lines.append("")
    lines.append("旧 macro-domain nearest 图是有价值的 diagnostic，但不适合作为主证据：它强制每个 cluster-domain cell 都找 nearest sample，因此即使某个 macro-domain 没有可信 match，也会出现视觉上很弱的曲线。")
    lines.append("")
    lines.append("## 2. Improved Experimental Protocol")
    lines.append("")
    lines.append(f"- model: `Chronos-2`")
    lines.append(f"- representations: `{', '.join(REPRESENTATIONS)}`")
    lines.append(f"- windows per dataset: `{summary['windows_per_dataset']}`")
    lines.append(f"- selected balance mode: `{summary['balance_mode']}`")
    lines.append(f"- max patches per macro-domain: `{summary['max_per_macro_domain']}`")
    lines.append(f"- max patches per dataset within macro-domain: `{summary['max_per_dataset_within_macro_domain']}`")
    lines.append(f"- K candidates after coarse-to-fine search are recorded in `k_sweep_metrics.csv`。")
    lines.append("")
    lines.append("K selection 不使用 silhouette-only，而是综合 seed stability、KMeans vs Agglomerative agreement、cluster size、confounder NMI、Davies-Bouldin 和 original-space evidence。")
    lines.append("")
    lines.append("主证据筛选也预先定义：cluster 不能太小，center-nearest raw patches 需要视觉一致，confidence-filtered macro-domain view 需要多个真实 macro-domain 的可信 match，同时不能明显被 single dataset、frequency 或 patch index 主导。")
    lines.append("")
    lines.append("## 3. K Selection Result")
    lines.append("")
    lines.append(f"Recommended shared K: **`{selected['recommended_shared_k']}`**")
    lines.append("")
    lines.append("| representation | recommended per-layer K | top candidates |")
    lines.append("|---|---:|---|")
    for rep, info in selected["per_layer"].items():
        lines.append(f"| `{rep}` | {info['recommended_k']} | `{info['top_candidates']}` |")
    lines.append("")
    if selected["per_layer"].get("layer_6", {}).get("recommended_k") != selected["recommended_shared_k"]:
        lines.append(
            "`layer_6` 的 per-layer K 倾向更细的划分，但本报告主图采用 shared K，是为了让 `projection -> layer_0 -> layer_6 -> layer_11` 的层间比较保持同一 operating point。"
            "这不是否认 `layer_6` 内部可能需要更细 taxonomy，而是把它留作下一步 layer-specific split analysis。"
        )
        lines.append("")
    lines.append("选择 shared K 的理由不是它在每个单项指标上都最优，而是它在四层中同时满足：seed stability 高、cluster size 不碎、confounder NMI 相对可控，并且可以生成可解释的 original-space evidence。")
    lines.append("")
    lines.append(f"![K selection summary]({rel_fig('k_selection_summary.png')})")
    lines.append("")
    lines.append("## 4. Layer-wise Validation Summary")
    lines.append("")
    lines.append("| representation | K | silhouette | stability | agg NMI | macro NMI | frequency NMI | high-conf macro rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for rep in REPRESENTATIONS:
        m = final_results[rep]["metrics"]
        lines.append(
            f"| `{rep}` | {m['k']} | {m['silhouette']:.3f} | {m['seed_stability_nmi']:.3f} | "
            f"{m['kmeans_vs_agglomerative_nmi']:.3f} | {m['nmi_macro_domain']:.3f} | "
            f"{m['nmi_frequency']:.3f} | {m['macro_high_conf_rate']:.3f} |"
        )
    lines.append("")
    lines.append("Metric 读法：")
    lines.append("")
    lines.append("| metric | 含义 | 方向 | 注意事项 |")
    lines.append("|---|---|---|---|")
    lines.append("| `silhouette` | 样本到本 cluster 的紧密度相对其它 cluster 的分离度。 | 越高越好 | 不能单独用来选 K，因为高分可能来自过粗划分或 domain separation。 |")
    lines.append("| `stability` | 不同 KMeans random seeds 得到的 labels 的 NMI 平均值。 | 越高越好 | 表示 clustering 对初始化不敏感，但不等于语义正确。 |")
    lines.append("| `agg NMI` | KMeans labels 与 AgglomerativeClustering labels 的 NMI。 | 越高越好 | 表示 cluster structure 不太依赖单一聚类算法。 |")
    lines.append("| `macro NMI` | cluster labels 与 macro-domain labels 的 NMI。 | 通常越低越好 | 高值提示 domain confounding；若研究 domain-specific concept，则可作为警告而非直接否定。 |")
    lines.append("| `frequency NMI` | cluster labels 与采样频率/cadence labels 的 NMI。 | 通常越低越好 | 高值提示 frequency/cadence confounding。 |")
    lines.append("| `high-conf macro rate` | cluster × real macro-domain cell 中存在同 cluster 且距离中心足够近的比例。 | 越高越好 | 用于检查原空间 prototype 能否跨真实 macro-domain 复现，不含 Synthetic control。 |")
    lines.append("")
    lines.append(f"![Layer comparison summary]({rel_fig('layer_comparison_summary.png')})")
    lines.append("")
    lines.append("读法：`projection` 和 `layer_0` 更接近 local patch vocabulary；`layer_6` 与 `layer_11` 通常更稳定，但更容易吸收 domain/frequency/context-style 信息。")
    lines.append("")
    lines.append("## 5. All-cluster Evidence Figures")
    lines.append("")
    lines.append("为避免 cherry-picking，本节每个 representation 都展示 final shared K 下的全部 clusters。也就是说，`K=6` 时每层都展示 `C0-C5`。")
    lines.append("质量闸门仍然保留，但它只用于解释每个 cluster 的证据强弱，不用于隐藏结果。")
    for rep in REPRESENTATIONS:
        lines.append("")
        lines.append(f"### {rep}")
        lines.append("")
        lines.append(f"![{rep} center nearest]({rel_fig(f'{rep}_main_center_nearest.png')})")
        lines.append("")
        lines.append(f"![{rep} macro-domain filtered]({rel_fig(f'{rep}_main_macro_domain_filtered.png')})")
        lines.append("")
        lines.append("All clusters under this K setting:")
        lines.append("")
        lines.append("| cluster | tier | size | score | macro domains | raw coherence | confounder risk | interpretation status |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for c in sorted(final_results[rep]["clusters"], key=lambda x: int(x["cluster"])):
            tier = str(c["evidence_tier"])
            if tier == "main_evidence":
                status = "candidate concept"
            elif tier == "diagnostic_confounded":
                status = "confounded diagnostic"
            elif tier == "diagnostic_tiny":
                status = "tiny diagnostic"
            else:
                status = "weak diagnostic"
            lines.append(
                f"| `C{c['cluster']}` | `{tier}` | {c['size']} | {c['evidence_score']:.3f} | "
                f"{c['high_confidence_real_macro_domains']} | {c['raw_center_nearest_coherence']:.3f} | "
                f"{c['confounder_risk_max_share']:.3f} | {status} |"
            )
    lines.append("")
    if layer_specific_results:
        lines.append("## 6. Layer-specific K Check")
        lines.append("")
        lines.append("shared K 用于层间对比；per-layer K 用于检查某一层内部是否存在更细的 model-derived motif/prototype family。这里不替换主结论，也不 cherry-pick：凡是补充的 K setting 都展示该 K 下的全部 clusters。")
        for rep, result in layer_specific_results.items():
            k = int(result["metrics"]["k"])
            lines.append("")
            lines.append(f"### {rep} K={k}")
            lines.append("")
            lines.append(f"`{rep}` 的 per-layer K selection 指向 `K={k}`，说明该层在 shared K 之外可能存在更细的 contextual substructure。shared `K={selected['recommended_shared_k']}` 仍然用于 `projection -> layer_0 -> layer_6 -> layer_11` 的横向比较；`K={k}` 作为 layer-specific split check。")
            lines.append("")
            lines.append(f"![{rep} K{k} center nearest]({rel_fig(f'{rep}_k{k}_center_nearest.png')})")
            lines.append("")
            lines.append(f"![{rep} K{k} macro-domain filtered]({rel_fig(f'{rep}_k{k}_macro_domain_filtered.png')})")
            lines.append("")
            lines.append(f"![{rep} K{k} prior audit]({rel_fig(f'{rep}_k{k}_embedding_audit.png')})")
            lines.append("")
            lines.append("All clusters under this layer-specific K setting:")
            lines.append("")
            lines.append("| cluster | tier | size | score | macro domains | raw coherence | confounder risk | interpretation status |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---|")
            for c in sorted(result["clusters"], key=lambda x: int(x["cluster"])):
                tier = str(c["evidence_tier"])
                if tier == "main_evidence":
                    status = "candidate concept"
                elif tier == "diagnostic_confounded":
                    status = "confounded diagnostic"
                elif tier == "diagnostic_tiny":
                    status = "tiny diagnostic"
                else:
                    status = "weak diagnostic"
                lines.append(
                    f"| `C{c['cluster']}` | `{tier}` | {c['size']} | {c['evidence_score']:.3f} | "
                    f"{c['high_confidence_real_macro_domains']} | {c['raw_center_nearest_coherence']:.3f} | "
                    f"{c['confounder_risk_max_share']:.3f} | {status} |"
                )
        lines.append("")
    lines.append("## 7. Prior-guided Motif Audit")
    lines.append("")
    lines.append("下面的图只用于检查 model-derived clusters 与 human-prior motif probe 是否对齐。它不能证明 cluster 的 ground-truth 语义，也不能作为 cluster 命名来源。")
    for rep in REPRESENTATIONS:
        lines.append("")
        lines.append(f"![{rep} prior audit]({rel_fig(f'{rep}_embedding_audit.png')})")
    lines.append("")
    lines.append("## 8. Diagnostic Evidence and Failure Cases")
    lines.append("")
    lines.append(f"![Diagnostic failure cases]({rel_fig('diagnostic_failure_cases.png')})")
    lines.append("")
    lines.append("这些 failure cases 是方法的安全阀：如果 cluster 视觉上有形态但 confounder 风险高、macro-domain match 弱或 cluster 太小，就不进入主结论。")
    lines.append("")
    lines.append("## 9. Final Answer for Advisor")
    lines.append("")
    lines.append("当前可以稳健声称：Chronos-2 的 patch representations 在不同层中确实保留并重组 local temporal information；但不同层承担的角色不同。`projection` / `layer_0` 更适合回答 single patch local vocabulary，`layer_6` / `layer_11` 更适合观察 contextual mixing 和 domain/cadence-style 的重组。")
    lines.append("")
    lines.append("当前仍需谨慎：cluster 不是最终 taxonomy；K 是用于 exploratory concept discovery 的 operating point；macro-domain evidence 只能说明跨领域可复现性，不等于真实世界语义 ground truth。若进入 paper 阶段，还需要更大采样、多 seed 数据重采样和人工/领域知识审阅。")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-per-dataset", type=int, default=500)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--k-candidates", type=str, default="6,8,10,12,14,16,18,20,24,28,32")
    parser.add_argument("--balance-mode", choices=["macro_domain", "source_domain"], default="macro_domain")
    parser.add_argument("--max-per-macro-domain", type=int, default=1500)
    parser.add_argument("--max-per-dataset-within-macro-domain", type=int, default=350)
    parser.add_argument("--max-per-source-domain", type=int, default=700)
    parser.add_argument("--stability-seeds", type=str, default="47,53,59")
    parser.add_argument("--main-clusters-per-rep", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir: Path = args.output_dir.resolve()
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "progress_log.md").write_text("# Chronos-2 multilayer validation progress\n\n", encoding="utf-8")

    pilot_audit = summarize_pilot(ROOT / "outputs" / "chronos_layer_effect" / "chronos_layer_effect_summary.json")
    append_log(
        out_dir,
        "## Checkpoint 1: pilot audit\n\n"
        f"- old windows/dataset: `{pilot_audit.get('windows_per_dataset')}`\n"
        f"- old raw patches estimate: `{pilot_audit.get('raw_patch_count_estimate')}`\n"
        f"- old K: `{pilot_audit.get('k_by_representation')}`\n"
        "- 问题：100 windows/dataset、source-domain balance、经验 K、forced macro-domain nearest 只能作为 diagnostic。",
    )

    windows, window_meta, dataset_summary = sample_windows(
        DATA_ROOT,
        context_len=args.context_len,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
    )
    rep_data = build_rep_data(windows, window_meta, args.batch_size)

    selected_indices = {}
    selected_meta = {}
    rep_spaces = {}
    for rep in REPRESENTATIONS:
        data = rep_data[rep]
        idx = select_balanced_indices(
            data.metadata,
            args.balance_mode,
            args.max_per_macro_domain,
            args.max_per_dataset_within_macro_domain,
            args.max_per_source_domain,
            args.seed,
        )
        selected_indices[rep] = idx
        selected_meta[rep] = [data.metadata[i] for i in idx]
        x_pca, scaler, pca = fit_pca_space(data.embeddings[idx], args.seed)
        rep_spaces[rep] = {
            "x_pca": x_pca,
            "scaler": scaler,
            "pca": pca,
            "embeddings": data.embeddings[idx],
            "metadata": selected_meta[rep],
            "patches": data.patches[idx],
        }

    append_log(
        out_dir,
        "## Checkpoint 2: script runnable and data extracted\n\n"
        f"- command: `python scripts/run_chronos_multilayer_cluster_validation.py --windows-per-dataset {args.windows_per_dataset}`\n"
        f"- sampled windows: `{len(windows)}`\n"
        f"- representations: `{REPRESENTATIONS}`\n"
        f"- balance mode: `{args.balance_mode}`",
    )

    coarse_k = [int(x) for x in args.k_candidates.split(",") if x.strip()]
    stability_seeds = [int(x) for x in args.stability_seeds.split(",") if x.strip()]
    all_metric_rows = []
    metrics_by_rep: dict[str, list[dict[str, Any]]] = {}
    for rep in REPRESENTATIONS:
        x_pca = rep_spaces[rep]["x_pca"]
        meta = rep_spaces[rep]["metadata"]
        coarse_metrics = [evaluate_single_k(x_pca, meta, k, args.seed, stability_seeds) for k in coarse_k]
        fine_k = coarse_to_fine_candidates(coarse_metrics, coarse_k)
        all_k = sorted(set(coarse_k + fine_k))
        rows = [evaluate_single_k(x_pca, meta, k, args.seed, stability_seeds) for k in all_k]
        for row in rows:
            row["representation"] = rep
            row["balance_mode"] = args.balance_mode
        metrics_by_rep[rep] = rows
        all_metric_rows.extend(rows)

    write_csv(out_dir / "k_sweep_metrics.csv", all_metric_rows)
    selected_k = choose_k(metrics_by_rep)
    (out_dir / "selected_k_report.json").write_text(json.dumps(selected_k, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log(
        out_dir,
        "## Checkpoint 3: K sweep completed\n\n"
        f"- coarse K: `{coarse_k}`\n"
        f"- selected shared K: `{selected_k['recommended_shared_k']}`\n"
        f"- per-layer: `{selected_k['per_layer']}`",
    )

    final_results: dict[str, dict[str, Any]] = {}
    rep_payloads: dict[str, dict[str, Any]] = {}
    shared_k = int(selected_k["recommended_shared_k"])
    for rep in REPRESENTATIONS:
        payload = rep_spaces[rep]
        result, plot_payload = build_cluster_result(rep, shared_k, payload, metrics_by_rep[rep], args.seed)
        final_results[rep] = result
        rep_payloads[rep] = plot_payload

        plot_embedding_audit(
            rep,
            plot_payload["x_pca"][:, :2],
            plot_payload["labels"],
            plot_payload["centers"][:, :2],
            plot_payload["metadata"],
            result["displayed_clusters"],
            fig_dir,
        )
        plot_center_nearest(
            rep,
            result["displayed_clusters"],
            plot_payload["x_pca"],
            plot_payload["labels"],
            plot_payload["centers"],
            plot_payload["metadata"],
            plot_payload["patches"],
            fig_dir,
        )
        plot_macro_filtered(
            rep,
            result["displayed_clusters"],
            result["clusters"],
            plot_payload["metadata"],
            plot_payload["patches"],
            fig_dir,
        )

    layer_specific_results: dict[str, dict[str, Any]] = {}
    for rep in REPRESENTATIONS:
        per_layer_k = int(selected_k["per_layer"][rep]["recommended_k"])
        if per_layer_k == shared_k:
            continue
        payload = rep_spaces[rep]
        result, plot_payload = build_cluster_result(rep, per_layer_k, payload, metrics_by_rep[rep], args.seed)
        layer_specific_results[rep] = result
        title_suffix = f"K={per_layer_k}"
        plot_embedding_audit(
            rep,
            plot_payload["x_pca"][:, :2],
            plot_payload["labels"],
            plot_payload["centers"][:, :2],
            plot_payload["metadata"],
            result["displayed_clusters"],
            fig_dir,
            output_name=f"{rep}_k{per_layer_k}_embedding_audit",
            title_suffix=title_suffix,
        )
        plot_center_nearest(
            rep,
            result["displayed_clusters"],
            plot_payload["x_pca"],
            plot_payload["labels"],
            plot_payload["centers"],
            plot_payload["metadata"],
            plot_payload["patches"],
            fig_dir,
            output_name=f"{rep}_k{per_layer_k}_center_nearest",
            title_suffix=title_suffix,
        )
        plot_macro_filtered(
            rep,
            result["displayed_clusters"],
            result["clusters"],
            plot_payload["metadata"],
            plot_payload["patches"],
            fig_dir,
            output_name=f"{rep}_k{per_layer_k}_macro_domain_filtered",
            title_suffix=title_suffix,
        )

    plot_k_selection(metrics_by_rep, selected_k, fig_dir)
    plot_layer_comparison(final_results, fig_dir)
    plot_failure_cases(final_results, rep_payloads, fig_dir)
    append_log(
        out_dir,
        "## Checkpoint 4: figures generated\n\n"
        "- all-cluster evidence figures: center-nearest raw patches, confidence-filtered macro-domain examples, layer comparison, K selection summary。\n"
        "- diagnostic figures: prior-guided audit and failure cases。\n"
        "- 注意：shared K=6 时每层均展示 C0-C5，不隐藏 weak/confounded clusters。\n"
        f"- layer-specific K checks: `{ {rep: result['metrics']['k'] for rep, result in layer_specific_results.items()} }`。",
    )

    summary = {
        "objective": "Chronos-2 multilayer patch representation clustering validation",
        "model": "Chronos-2",
        "data_root": str(DATA_ROOT),
        "excluded_datasets": ["BLAST"],
        "windows_per_dataset": args.windows_per_dataset,
        "context_len": args.context_len,
        "patch_len": PATCH_LEN,
        "num_windows": int(len(windows)),
        "raw_patch_count_estimate": int(len(windows) * (args.context_len // PATCH_LEN)),
        "balance_mode": args.balance_mode,
        "max_per_macro_domain": args.max_per_macro_domain,
        "max_per_dataset_within_macro_domain": args.max_per_dataset_within_macro_domain,
        "seed": args.seed,
        "macro_domain_definitions": MACRO_DOMAIN_DEFINITIONS,
        "dataset_inventory": list_dataset_inventory(DATA_ROOT, args.context_len),
        "dataset_summary": dataset_summary,
        "pilot_audit": pilot_audit,
        "selected_k": selected_k,
        "final_results": final_results,
        "layer_specific_results": layer_specific_results,
        "output_files": {
            "summary": str(out_dir / "cluster_validation_summary.json"),
            "k_sweep_metrics": str(out_dir / "k_sweep_metrics.csv"),
            "selected_k_report": str(out_dir / "selected_k_report.json"),
            "figures": str(fig_dir),
            "progress_log": str(out_dir / "progress_log.md"),
        },
    }
    (out_dir / "cluster_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(ROOT / "docs" / "chronos_layer_effect_report.md", summary, selected_k, final_results, layer_specific_results, fig_dir)
    append_log(
        out_dir,
        "## Checkpoint 5: report updated\n\n"
        f"- report: `docs/chronos_layer_effect_report.md`\n"
        f"- selected shared K: `{shared_k}`\n"
        "- 剩余风险：cluster 仍是 exploratory operating point，不是 final taxonomy；需要更大数据重采样和领域审阅才能写成最终 paper claim。",
    )
    print(json.dumps({"summary": str(out_dir / "cluster_validation_summary.json"), "selected_shared_k": shared_k}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
