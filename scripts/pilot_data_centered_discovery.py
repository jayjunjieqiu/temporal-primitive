from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data/junjieqiu/datasets/basicts_datasets")
CHRONOS_SRC = ROOT / "external" / "chronos-forecasting" / "src"
TIMESFM_SRC = ROOT / "external" / "timesfm" / "src"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

sys.path.insert(0, str(ROOT))
from scripts.explore_motif_taxonomy import label_patch  # noqa: E402


MODEL_SPECS = {
    "chronos_2_small": {"kind": "chronos", "path": ROOT / "chronos-2-small", "patch_len": 16},
    "chronos_2": {"kind": "chronos", "path": ROOT / "chronos-2", "patch_len": 16},
    "timesfm_2_5": {"kind": "timesfm", "path": ROOT / "timesfm-2.5-200m-pytorch", "patch_len": 32},
}


def read_desc(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _x: None)


def interpolate_nans(x: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.mean() < 0.95:
        return None
    if finite.all():
        return x
    idx = np.arange(len(x))
    filled = x.copy()
    filled[~finite] = np.interp(idx[~finite], idx[finite], x[finite]).astype(np.float32)
    return filled


def robust_z(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > eps:
        return (x - med) / (1.4826 * mad + eps)
    std = float(np.std(x))
    return (x - float(np.mean(x))) / max(std, eps)


def patch_stats(patch: np.ndarray, patch_len: int) -> dict[str, Any]:
    result = label_patch(patch, patch_len)
    x = np.asarray(patch, dtype=np.float64)
    z = robust_z(x)
    t = np.linspace(-1.0, 1.0, len(x))
    slope = float(np.polyfit(t, z, deg=1)[0]) if len(x) > 2 else 0.0
    return {
        "taxonomy_label": result.label,
        "taxonomy_confidence": float(result.confidence),
        "taxonomy_fired": result.fired,
        "raw_mean": float(np.mean(x)),
        "raw_std": float(np.std(x)),
        "raw_min": float(np.min(x)),
        "raw_max": float(np.max(x)),
        "robust_slope": slope,
        "features": result.features,
    }


def sample_windows(
    data_root: Path,
    context_len: int,
    windows_per_dataset: int,
    seed: int,
    max_attempts_per_dataset: int = 5000,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    dataset_summary: list[dict[str, Any]] = []

    for desc_path in sorted(data_root.glob("*/desc.json")):
        dataset = desc_path.parent.name
        if dataset == "BLAST":
            continue
        desc = read_desc(desc_path)
        shape = tuple(desc["shape"])
        if len(shape) != 3 or shape[0] < context_len:
            dataset_summary.append({"dataset": dataset, "status": "skipped", "reason": f"shape={shape}"})
            continue

        data_path = desc_path.parent / "data.dat"
        if not data_path.exists():
            dataset_summary.append({"dataset": dataset, "status": "skipped", "reason": "missing data.dat"})
            continue

        data = np.memmap(data_path, dtype="float32", mode="r", shape=shape)
        accepted = 0
        attempts = 0
        seen: set[tuple[int, int]] = set()
        while accepted < windows_per_dataset and attempts < max_attempts_per_dataset:
            attempts += 1
            node = int(rng.integers(0, shape[1]))
            start = int(rng.integers(0, shape[0] - context_len + 1))
            key = (node, start)
            if key in seen:
                continue
            seen.add(key)
            values = np.asarray(data[start : start + context_len, node, 0], dtype=np.float32)
            values = interpolate_nans(values)
            if values is None:
                continue
            if float(np.nanstd(values)) < 1e-6:
                continue
            windows.append(values)
            metadata.append(
                {
                    "window_id": len(metadata),
                    "dataset": dataset,
                    "domain": desc.get("domain", dataset),
                    "frequency_minutes": desc.get("frequency (minutes)"),
                    "node": node,
                    "feature": 0,
                    "start": start,
                    "end": start + context_len,
                    "context_len": context_len,
                }
            )
            accepted += 1
        dataset_summary.append(
            {
                "dataset": dataset,
                "domain": desc.get("domain", dataset),
                "shape": list(shape),
                "accepted_windows": accepted,
                "attempts": attempts,
                "status": "ok" if accepted else "empty",
            }
        )

    return np.stack(windows).astype(np.float32), metadata, dataset_summary


def extract_chronos_embeddings(model_key: str, windows: np.ndarray, batch_size: int) -> np.ndarray:
    sys.path.insert(0, str(CHRONOS_SRC))
    import chronos

    spec = MODEL_SPECS[model_key]
    pipeline = chronos.Chronos2Pipeline.from_pretrained(
        str(spec["path"]),
        local_files_only=True,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = pipeline.model
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[start : start + batch_size], dtype=torch.float32, device=model.device)
            encoder_outputs, _loc_scale, _future_mask, num_context_patches = model.encode(
                context=batch,
                num_output_patches=1,
            )
            hidden = encoder_outputs[0][:, :num_context_patches].detach().float().cpu().numpy()
            chunks.append(hidden)
    del pipeline, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0)


def extract_timesfm_embeddings(windows: np.ndarray, batch_size: int) -> np.ndarray:
    sys.path.insert(0, str(TIMESFM_SRC))
    import timesfm
    from timesfm.torch import util

    model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
    model.model.load_checkpoint(str(ROOT / "timesfm-2.5-200m-pytorch" / "model.safetensors"), torch_compile=False)
    module = model.model
    module.eval()
    chunks: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            values = torch.tensor(windows[start : start + batch_size], dtype=torch.float32, device=module.device)
            masks = torch.zeros_like(values, dtype=torch.bool, device=module.device)
            batch_n, context = values.shape
            patched_inputs = torch.reshape(values, (batch_n, -1, module.p))
            patched_masks = torch.reshape(masks, (batch_n, -1, module.p))
            n = torch.zeros(batch_n, device=module.device)
            mu = torch.zeros(batch_n, device=module.device)
            sigma = torch.zeros(batch_n, device=module.device)
            patch_mu = []
            patch_sigma = []
            for i in range(context // module.p):
                (n, mu, sigma), _ = util.update_running_stats(n, mu, sigma, patched_inputs[:, i], patched_masks[:, i])
                patch_mu.append(mu)
                patch_sigma.append(sigma)
            context_mu = torch.stack(patch_mu, dim=1)
            context_sigma = torch.stack(patch_sigma, dim=1)
            normed_inputs = util.revin(patched_inputs, context_mu, context_sigma, reverse=False)
            normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
            (_input_embeds, output_embeds, _output_ts, _output_quantiles), _ = module(normed_inputs, patched_masks)
            chunks.append(output_embeds.detach().float().cpu().numpy())

    del model, module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0)


def flatten_model_patches(
    model_key: str,
    window_embeddings: np.ndarray,
    windows: np.ndarray,
    window_metadata: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    patch_len = int(MODEL_SPECS[model_key]["patch_len"])
    flat_embeddings: list[np.ndarray] = []
    patch_metadata: list[dict[str, Any]] = []
    for i, meta in enumerate(window_metadata):
        num_patches = window_embeddings.shape[1]
        for j in range(num_patches):
            patch_start = j * patch_len
            patch_end = patch_start + patch_len
            if patch_end > windows.shape[1]:
                continue
            patch = windows[i, patch_start:patch_end]
            stats = patch_stats(patch, patch_len)
            flat_embeddings.append(window_embeddings[i, j])
            patch_metadata.append(
                {
                    **meta,
                    "model": model_key,
                    "patch_index": j,
                    "patch_len": patch_len,
                    "patch_start_in_window": patch_start,
                    "patch_end_in_window": patch_end,
                    "global_start": int(meta["start"] + patch_start),
                    "global_end": int(meta["start"] + patch_end),
                    **stats,
                }
            )
    return np.stack(flat_embeddings).astype(np.float32), patch_metadata


def weighted_purity(cluster_ids: np.ndarray, labels: list[str]) -> float:
    total = len(labels)
    score = 0
    for cid in sorted(set(cluster_ids.tolist())):
        idx = np.where(cluster_ids == cid)[0]
        counts = Counter(labels[i] for i in idx)
        score += counts.most_common(1)[0][1]
    return float(score / total)


def top_counts(values: list[Any], n: int = 5) -> list[dict[str, Any]]:
    return [{"value": str(k), "count": int(v)} for k, v in Counter(values).most_common(n)]


def cluster_embeddings(model_key: str, embeddings: np.ndarray, metadata: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    x = StandardScaler().fit_transform(embeddings)
    pca_dim = max(2, min(20, x.shape[0] - 1, x.shape[1]))
    x_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    k = min(12, max(4, int(round(math.sqrt(len(metadata) / 20)))))
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=20)
    cluster_ids = kmeans.fit_predict(x_pca)

    datasets = [m["dataset"] for m in metadata]
    domains = [m["domain"] for m in metadata]
    taxonomy = [m["taxonomy_label"] for m in metadata]
    freq = [str(m.get("frequency_minutes")) for m in metadata]
    patch_indices = [str(m["patch_index"]) for m in metadata]
    silhouette = float(silhouette_score(x_pca, cluster_ids)) if len(set(cluster_ids)) > 1 else None

    clusters = []
    for cid in sorted(set(cluster_ids.tolist())):
        idx = np.where(cluster_ids == cid)[0]
        raw_std = [metadata[i]["raw_std"] for i in idx]
        slope = [metadata[i]["robust_slope"] for i in idx]
        examples = []
        for i in idx[:5]:
            examples.append(
                {
                    "dataset": metadata[i]["dataset"],
                    "domain": metadata[i]["domain"],
                    "global_start": metadata[i]["global_start"],
                    "node": metadata[i]["node"],
                    "taxonomy_label": metadata[i]["taxonomy_label"],
                    "raw_std": metadata[i]["raw_std"],
                    "robust_slope": metadata[i]["robust_slope"],
                }
            )
        clusters.append(
            {
                "cluster": int(cid),
                "size": int(len(idx)),
                "top_datasets": top_counts([datasets[i] for i in idx]),
                "top_domains": top_counts([domains[i] for i in idx]),
                "top_taxonomy_labels": top_counts([taxonomy[i] for i in idx]),
                "top_frequencies": top_counts([freq[i] for i in idx]),
                "top_patch_indices": top_counts([patch_indices[i] for i in idx]),
                "mean_raw_std": float(np.mean(raw_std)),
                "mean_abs_robust_slope": float(np.mean(np.abs(slope))),
                "examples": examples,
            }
        )

    pca2 = x_pca[:, :2]
    save_scatter(model_key, pca2, cluster_ids, domains)

    return {
        "num_patch_embeddings": int(len(metadata)),
        "embedding_dim": int(embeddings.shape[1]),
        "pca_dim_for_clustering": int(pca_dim),
        "pca2_explained_variance_ratio": PCA(n_components=2, random_state=seed).fit(x).explained_variance_ratio_.astype(float).tolist(),
        "kmeans_k": int(k),
        "silhouette_pca_space": silhouette,
        "purity_by_dataset": weighted_purity(cluster_ids, datasets),
        "purity_by_domain": weighted_purity(cluster_ids, domains),
        "purity_by_taxonomy_v0": weighted_purity(cluster_ids, taxonomy),
        "purity_by_patch_index": weighted_purity(cluster_ids, patch_indices),
        "purity_by_frequency": weighted_purity(cluster_ids, freq),
        "nmi_dataset": float(normalized_mutual_info_score(datasets, cluster_ids)),
        "nmi_domain": float(normalized_mutual_info_score(domains, cluster_ids)),
        "nmi_taxonomy_v0": float(normalized_mutual_info_score(taxonomy, cluster_ids)),
        "nmi_patch_index": float(normalized_mutual_info_score(patch_indices, cluster_ids)),
        "nmi_frequency": float(normalized_mutual_info_score(freq, cluster_ids)),
        "global_top_datasets": top_counts(datasets, 8),
        "global_top_domains": top_counts(domains, 8),
        "global_top_taxonomy_labels": top_counts(taxonomy, 8),
        "global_top_patch_indices": top_counts(patch_indices, 8),
        "clusters": clusters,
    }


def save_scatter(model_key: str, pca2: np.ndarray, cluster_ids: np.ndarray, domains: list[str]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(pca2[:, 0], pca2[:, 1], c=cluster_ids, s=8, cmap="tab20", alpha=0.75)
        ax.set_title(f"{model_key}: PCA of patch embeddings by cluster")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(scatter, ax=ax, label="cluster")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"data_centered_{model_key}_pca_clusters.png", dpi=180)
        plt.close(fig)

        domain_names = sorted(set(domains))
        domain_ids = np.asarray([domain_names.index(d) for d in domains])
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(pca2[:, 0], pca2[:, 1], c=domain_ids, s=8, cmap="tab20", alpha=0.75)
        ax.set_title(f"{model_key}: PCA of patch embeddings by domain")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        cbar = fig.colorbar(scatter, ax=ax, ticks=range(len(domain_names)))
        cbar.ax.set_yticklabels([d[:24] for d in domain_names], fontsize=6)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"data_centered_{model_key}_pca_domains.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"figure generation skipped for {model_key}: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-per-dataset", type=int, default=12)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--models",
        type=str,
        default="chronos_2_small,chronos_2,timesfm_2_5",
        help="Comma-separated model keys.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    windows, window_metadata, dataset_summary = sample_windows(
        DATA_ROOT,
        context_len=args.context_len,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
    )
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    results: dict[str, Any] = {
        "objective": "data/model-centered patch concept discovery pilot",
        "data_root": str(DATA_ROOT),
        "excluded_datasets": ["BLAST"],
        "context_len": args.context_len,
        "windows_per_dataset": args.windows_per_dataset,
        "num_windows": int(len(windows)),
        "dataset_summary": dataset_summary,
        "models": {},
    }

    for model_key in selected_models:
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unknown model key: {model_key}")
        print(f"Extracting {model_key}...")
        if MODEL_SPECS[model_key]["kind"] == "chronos":
            window_embeddings = extract_chronos_embeddings(model_key, windows, args.batch_size)
        else:
            window_embeddings = extract_timesfm_embeddings(windows, args.batch_size)
        patch_embeddings, patch_metadata = flatten_model_patches(model_key, window_embeddings, windows, window_metadata)
        analysis = cluster_embeddings(model_key, patch_embeddings, patch_metadata, args.seed)
        results["models"][model_key] = {
            "patch_len": MODEL_SPECS[model_key]["patch_len"],
            "window_embedding_shape": list(window_embeddings.shape),
            **analysis,
        }

    out = OUTPUT_DIR / "data_centered_discovery_summary.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
