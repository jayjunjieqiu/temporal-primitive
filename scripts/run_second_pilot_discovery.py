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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data/junjieqiu/datasets/basicts_datasets")
CHRONOS_SRC = ROOT / "external" / "chronos-forecasting" / "src"
TIMESFM_SRC = ROOT / "external" / "timesfm" / "src"
OUTPUT_DIR = ROOT / "outputs"
PILOT_DIR = OUTPUT_DIR / "second_pilot"
FIGURE_DIR = OUTPUT_DIR / "figures" / "second_pilot"

sys.path.insert(0, str(ROOT))
from scripts.explore_motif_taxonomy import label_patch  # noqa: E402


MODEL_SPECS = {
    "chronos_2_small": {
        "kind": "chronos",
        "path": ROOT / "chronos-2-small",
        "patch_len": 16,
        "layers": [0, 3, 5],
    },
    "chronos_2": {
        "kind": "chronos",
        "path": ROOT / "chronos-2",
        "patch_len": 16,
        "layers": [0, 6, 11],
    },
    "timesfm_2_5": {
        "kind": "timesfm",
        "path": ROOT / "timesfm-2.5-200m-pytorch",
        "patch_len": 32,
        "layers": [0, 10, 19],
    },
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
    zero_ratio = float(np.mean(np.isclose(x, 0.0)))
    return {
        "taxonomy_label": result.label,
        "taxonomy_confidence": float(result.confidence),
        "taxonomy_fired": result.fired,
        "raw_mean": float(np.mean(x)),
        "raw_std": float(np.std(x)),
        "raw_range": float(np.max(x) - np.min(x)),
        "zero_ratio": zero_ratio,
        "robust_slope": slope,
    }


def sample_windows(
    data_root: Path,
    context_len: int,
    windows_per_dataset: int,
    seed: int,
    max_attempts_per_dataset: int = 50000,
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
            if values is None or float(np.nanstd(values)) < 1e-6:
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
                "frequency_minutes": desc.get("frequency (minutes)"),
                "shape": list(shape),
                "accepted_windows": accepted,
                "attempts": attempts,
                "status": "ok" if accepted else "empty",
            }
        )

    return np.stack(windows).astype(np.float32), metadata, dataset_summary


def extract_chronos_layers(model_key: str, windows: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
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
    layer_names = [f"layer_{idx}" for idx in spec["layers"]]
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in layer_names}

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[start : start + batch_size], dtype=torch.float32, device=model.device)
            captured: dict[str, torch.Tensor] = {}
            handles = []

            def hook_for(layer_idx: int):
                def hook(_mod, _inp, out):
                    hidden = out[0] if isinstance(out, tuple) else out.hidden_states
                    captured[f"layer_{layer_idx}"] = hidden.detach()

                return hook

            for layer_idx in spec["layers"]:
                handles.append(model.encoder.block[layer_idx].register_forward_hook(hook_for(layer_idx)))

            _encoder_outputs, _loc_scale, _future_mask, num_context_patches = model.encode(
                context=batch,
                num_output_patches=1,
            )
            for handle in handles:
                handle.remove()
            for name in layer_names:
                chunks[name].append(captured[name][:, :num_context_patches].float().cpu().numpy())

    del pipeline, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def extract_timesfm_layers(windows: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(TIMESFM_SRC))
    import timesfm
    from timesfm.torch import util

    spec = MODEL_SPECS["timesfm_2_5"]
    model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
    model.model.load_checkpoint(str(ROOT / "timesfm-2.5-200m-pytorch" / "model.safetensors"), torch_compile=False)
    module = model.model
    module.eval()
    layer_names = [f"layer_{idx}" for idx in spec["layers"]]
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in layer_names}

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

            captured: dict[str, torch.Tensor] = {}
            handles = []

            def hook_for(layer_idx: int):
                def hook(_mod, _inp, out):
                    hidden = out[0] if isinstance(out, tuple) else out
                    captured[f"layer_{layer_idx}"] = hidden.detach()

                return hook

            for layer_idx in spec["layers"]:
                handles.append(module.stacked_xf[layer_idx].register_forward_hook(hook_for(layer_idx)))
            module(normed_inputs, patched_masks)
            for handle in handles:
                handle.remove()
            for name in layer_names:
                chunks[name].append(captured[name].float().cpu().numpy())

    del model, module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def flatten_model_patches(
    model_key: str,
    layer_embeddings: np.ndarray,
    windows: np.ndarray,
    window_metadata: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    patch_len = int(MODEL_SPECS[model_key]["patch_len"])
    flat_embeddings: list[np.ndarray] = []
    patch_metadata: list[dict[str, Any]] = []
    raw_patches: list[np.ndarray] = []
    for i, meta in enumerate(window_metadata):
        for j in range(layer_embeddings.shape[1]):
            patch_start = j * patch_len
            patch_end = patch_start + patch_len
            if patch_end > windows.shape[1]:
                continue
            patch = windows[i, patch_start:patch_end]
            flat_embeddings.append(layer_embeddings[i, j])
            raw_patches.append(patch)
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
                    **patch_stats(patch, patch_len),
                }
            )
    return np.stack(flat_embeddings).astype(np.float32), patch_metadata, np.stack(raw_patches).astype(np.float32)


def top_counts(values: list[Any], n: int = 5) -> list[dict[str, Any]]:
    return [{"value": str(k), "count": int(v)} for k, v in Counter(values).most_common(n)]


def weighted_purity(cluster_ids: np.ndarray, labels: list[str]) -> float:
    total = len(labels)
    score = 0
    for cid in sorted(set(cluster_ids.tolist())):
        idx = np.where(cluster_ids == cid)[0]
        counts = Counter(labels[i] for i in idx)
        score += counts.most_common(1)[0][1]
    return float(score / total)


def select_domain_balanced_indices(metadata: list[dict[str, Any]], max_per_domain: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        groups[str(meta["domain"])].append(i)
    selected = []
    for indices in groups.values():
        arr = np.asarray(indices)
        take = min(max_per_domain, len(arr))
        selected.extend(rng.choice(arr, size=take, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def cluster_once(
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    seed: int,
    prefix: str,
    raw_patches: np.ndarray | None = None,
    prototype_limit: int = 4,
) -> dict[str, Any]:
    x = StandardScaler().fit_transform(embeddings)
    pca_dim = max(2, min(30, x.shape[0] - 1, x.shape[1]))
    pca = PCA(n_components=pca_dim, random_state=seed)
    x_pca = pca.fit_transform(x)
    k = min(16, max(6, int(round(math.sqrt(len(metadata) / 35)))))
    cluster_ids = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(x_pca)

    try:
        agglom_ids = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x_pca)
        agglom_nmi = float(normalized_mutual_info_score(cluster_ids, agglom_ids))
    except Exception:
        agglom_nmi = float("nan")

    labels = {
        "dataset": [m["dataset"] for m in metadata],
        "domain": [m["domain"] for m in metadata],
        "taxonomy_v0": [m["taxonomy_label"] for m in metadata],
        "patch_index": [str(m["patch_index"]) for m in metadata],
        "frequency": [str(m.get("frequency_minutes")) for m in metadata],
    }

    if raw_patches is not None:
        save_prototype_panels(prefix, x_pca, cluster_ids, metadata, raw_patches, prototype_limit)
    save_scatter(prefix, x_pca[:, :2], cluster_ids, labels)

    clusters = []
    for cid in sorted(set(cluster_ids.tolist())):
        idx = np.where(cluster_ids == cid)[0]
        raw_std = [metadata[i]["raw_std"] for i in idx]
        raw_range = [metadata[i]["raw_range"] for i in idx]
        slope = [metadata[i]["robust_slope"] for i in idx]
        zero_ratio = [metadata[i]["zero_ratio"] for i in idx]
        clusters.append(
            {
                "cluster": int(cid),
                "size": int(len(idx)),
                "top_datasets": top_counts([labels["dataset"][i] for i in idx], 5),
                "top_domains": top_counts([labels["domain"][i] for i in idx], 5),
                "top_taxonomy_labels": top_counts([labels["taxonomy_v0"][i] for i in idx], 5),
                "top_patch_indices": top_counts([labels["patch_index"][i] for i in idx], 5),
                "top_frequencies": top_counts([labels["frequency"][i] for i in idx], 5),
                "mean_raw_std": float(np.mean(raw_std)),
                "mean_raw_range": float(np.mean(raw_range)),
                "mean_abs_robust_slope": float(np.mean(np.abs(slope))),
                "mean_zero_ratio": float(np.mean(zero_ratio)),
            }
        )

    nn_metrics = nearest_neighbor_metrics(x_pca, labels)
    silhouette = float(silhouette_score(x_pca, cluster_ids)) if len(set(cluster_ids)) > 1 else None
    return {
        "num_patch_embeddings": int(len(metadata)),
        "embedding_dim": int(embeddings.shape[1]),
        "pca_dim": int(pca_dim),
        "pca2_explained_variance_ratio": pca.explained_variance_ratio_[:2].astype(float).tolist(),
        "kmeans_k": int(k),
        "silhouette_pca_space": silhouette,
        "kmeans_vs_agglomerative_nmi": agglom_nmi,
        "purity": {name: weighted_purity(cluster_ids, values) for name, values in labels.items()},
        "nmi": {name: float(normalized_mutual_info_score(values, cluster_ids)) for name, values in labels.items()},
        "nearest_neighbor": nn_metrics,
        "global_top": {name: top_counts(values, 8) for name, values in labels.items()},
        "clusters": clusters,
    }


def nearest_neighbor_metrics(x: np.ndarray, labels: dict[str, list[str]], k: int = 10) -> dict[str, float]:
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(x)), metric="euclidean")
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)[:, 1:]
    out = {}
    for name, values in labels.items():
        arr = np.asarray(values)
        out[f"top{k}_{name}_agreement"] = float(np.mean(arr[indices] == arr[:, None]))
    return out


def save_scatter(prefix: str, pca2: np.ndarray, cluster_ids: np.ndarray, labels: dict[str, list[str]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(pca2[:, 0], pca2[:, 1], c=cluster_ids, s=5, cmap="tab20", alpha=0.65)
        ax.set_title(f"{prefix}: PCA by cluster")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(scatter, ax=ax, label="cluster")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{prefix}_pca_clusters.png", dpi=170)
        plt.close(fig)

        for color_name in ["domain", "patch_index", "taxonomy_v0"]:
            names = sorted(set(labels[color_name]))
            ids = np.asarray([names.index(v) for v in labels[color_name]])
            fig, ax = plt.subplots(figsize=(7, 5))
            scatter = ax.scatter(pca2[:, 0], pca2[:, 1], c=ids, s=5, cmap="tab20", alpha=0.65)
            ax.set_title(f"{prefix}: PCA by {color_name}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            cbar = fig.colorbar(scatter, ax=ax, ticks=range(len(names)))
            if len(names) <= 18:
                cbar.ax.set_yticklabels([v[:24] for v in names], fontsize=6)
            fig.tight_layout()
            fig.savefig(FIGURE_DIR / f"{prefix}_pca_{color_name}.png", dpi=170)
            plt.close(fig)
    except Exception as exc:
        print(f"scatter skipped for {prefix}: {type(exc).__name__}: {exc}")


def save_prototype_panels(
    prefix: str,
    x_pca: np.ndarray,
    cluster_ids: np.ndarray,
    metadata: list[dict[str, Any]],
    raw_patches: np.ndarray,
    prototype_limit: int,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        clusters = sorted(set(cluster_ids.tolist()))
        ncols = prototype_limit
        nrows = len(clusters)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 1.35 * nrows), squeeze=False)
        for row, cid in enumerate(clusters):
            idx = np.where(cluster_ids == cid)[0]
            center = x_pca[idx].mean(axis=0, keepdims=True)
            order = np.argsort(np.linalg.norm(x_pca[idx] - center, axis=1))[:prototype_limit]
            chosen = idx[order]
            for col in range(ncols):
                ax = axes[row, col]
                if col >= len(chosen):
                    ax.axis("off")
                    continue
                item = int(chosen[col])
                patch = robust_z(raw_patches[item])
                ax.plot(patch, linewidth=1.2)
                meta = metadata[item]
                ax.set_title(
                    f"C{cid} nearest {col + 1}\n{meta['dataset']} p{meta['patch_index']}",
                    fontsize=6,
                )
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{prefix}_prototype_panel.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"prototype panel skipped for {prefix}: {type(exc).__name__}: {exc}")


def analyze_model_layers(
    model_key: str,
    layer_outputs: dict[str, np.ndarray],
    windows: np.ndarray,
    window_metadata: list[dict[str, Any]],
    seed: int,
    domain_balanced_per_domain: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer_name, layer_embeddings in layer_outputs.items():
        embeddings, metadata, raw_patches = flatten_model_patches(model_key, layer_embeddings, windows, window_metadata)
        full_prefix = f"second_pilot_{model_key}_{layer_name}_full"
        full = cluster_once(embeddings, metadata, seed, full_prefix, raw_patches=raw_patches)
        domain_idx = select_domain_balanced_indices(metadata, domain_balanced_per_domain, seed)
        balanced_prefix = f"second_pilot_{model_key}_{layer_name}_domain_balanced"
        balanced = cluster_once(
            embeddings[domain_idx],
            [metadata[i] for i in domain_idx],
            seed,
            balanced_prefix,
            raw_patches=raw_patches[domain_idx],
        )
        result[layer_name] = {
            "window_embedding_shape": list(layer_embeddings.shape),
            "full_equal_per_dataset": full,
            "domain_balanced": balanced,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-per-dataset", type=int, default=100)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--domain-balanced-patches", type=int, default=700)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument(
        "--models",
        type=str,
        default="chronos_2_small,chronos_2,timesfm_2_5",
        help="Comma-separated model keys.",
    )
    args = parser.parse_args()

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    windows, window_metadata, dataset_summary = sample_windows(
        DATA_ROOT,
        context_len=args.context_len,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
    )

    results: dict[str, Any] = {
        "objective": "second pilot for data/model-centered TSFM patch concept discovery",
        "data_root": str(DATA_ROOT),
        "excluded_datasets": ["BLAST"],
        "context_len": args.context_len,
        "windows_per_dataset": args.windows_per_dataset,
        "num_windows": int(len(windows)),
        "domain_balanced_patches_per_domain": args.domain_balanced_patches,
        "dataset_summary": dataset_summary,
        "models": {},
    }

    for model_key in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"Extracting selected layers for {model_key}...")
        if MODEL_SPECS[model_key]["kind"] == "chronos":
            layer_outputs = extract_chronos_layers(model_key, windows, args.batch_size)
        else:
            layer_outputs = extract_timesfm_layers(windows, args.batch_size)
        results["models"][model_key] = {
            "patch_len": MODEL_SPECS[model_key]["patch_len"],
            "selected_layers": list(layer_outputs.keys()),
            "layers": analyze_model_layers(
                model_key,
                layer_outputs,
                windows,
                window_metadata,
                args.seed,
                args.domain_balanced_patches,
            ),
        }
        out_partial = PILOT_DIR / "second_pilot_discovery_summary.partial.json"
        out_partial.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    out = OUTPUT_DIR / "second_pilot_discovery_summary.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (PILOT_DIR / "second_pilot_discovery_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.print_json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(
            json.dumps(
                {
                    "num_windows": results["num_windows"],
                    "models": {
                        model: {
                            "patch_len": info["patch_len"],
                            "selected_layers": info["selected_layers"],
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
