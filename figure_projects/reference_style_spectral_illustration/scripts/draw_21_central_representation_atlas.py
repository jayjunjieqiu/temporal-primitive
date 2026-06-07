from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/temporal_primitive_mplconfig")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 8))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 8))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import BoundaryNorm


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "figure_projects" / "reference_style_spectral_illustration"
ASSETS = PROJECT / "assets"
OUT = ASSETS / "central_representation_atlas_layer0_clean.png"
SUMMARY = ASSETS / "central_representation_atlas_layer0_clean_summary.json"
CACHE = PROJECT / "cache"
REDUCER_CACHE = PROJECT / "cache" / "reducers"

sys.path.insert(0, str(ROOT))
from scripts.run_second_pilot_discovery import MODEL_SPECS, sample_windows, select_domain_balanced_indices  # noqa: E402
from scripts.run_chronos_multilayer_cluster_validation import (  # noqa: E402
    DATA_ROOT,
    CHRONOS_SRC,
    MACRO_DOMAIN_ORDER,
    MODEL_PATH,
    macro_domain,
    select_balanced_indices,
)


def preprocess_embeddings(embeddings: np.ndarray, seed: int, pca_dim_arg: int) -> tuple[np.ndarray, dict[str, Any]]:
    x = StandardScaler().fit_transform(embeddings)
    pca_dim = max(2, min(int(pca_dim_arg), x.shape[0] - 1, x.shape[1]))
    x_pre = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    return x_pre.astype(np.float32), {"pca_pre_dim": int(pca_dim)}


def fit_sklearn_tsne_space(
    x_pre: np.ndarray,
    seed: int,
    perplexity: float,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {"pca_pre_dim": int(x_pre.shape[1])}
    effective_perplexity = float(min(perplexity, max(5, (len(x_pre) - 1) // 3)))
    kwargs = dict(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="euclidean",
    )
    try:
        x_tsne = TSNE(max_iter=max_iter, **kwargs).fit_transform(x_pre)
    except TypeError:
        x_tsne = TSNE(n_iter=max_iter, **kwargs).fit_transform(x_pre)
    info.update(
        {
            "backend": "sklearn_tsne",
            "perplexity": effective_perplexity,
            "max_iter": int(max_iter),
            "init": "pca",
        }
    )
    return x_tsne.astype(np.float32), info


def fit_cuml_tsne_space(
    x_pre: np.ndarray,
    seed: int,
    perplexity: float,
    max_iter: int,
    rapids_env: str,
    init: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    REDUCER_CACHE.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {"pca_pre_dim": int(x_pre.shape[1])}
    effective_perplexity = float(min(perplexity, max(5, (len(x_pre) - 1) // 3)))
    stem = f"cuml_layer0_n{len(x_pre)}_d{x_pre.shape[1]}_seed{seed}_p{effective_perplexity:g}_it{max_iter}_init{init}"
    input_path = REDUCER_CACHE / f"{stem}_input.npy"
    output_path = REDUCER_CACHE / f"{stem}_tsne.npy"
    helper_summary_path = REDUCER_CACHE / f"{stem}_summary.json"
    np.save(input_path, x_pre)

    helper = Path(__file__).resolve().parent / "cuml_tsne_helper.py"
    cmd = [
        "mamba",
        "run",
        "-n",
        rapids_env,
        "python",
        str(helper),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--summary",
        str(helper_summary_path),
        "--seed",
        str(seed),
        "--perplexity",
        str(effective_perplexity),
        "--max-iter",
        str(max_iter),
        "--init",
        init,
    ]
    env = os.environ.copy()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)

    x_tsne = np.load(output_path).astype(np.float32, copy=False)
    helper_info = json.loads(helper_summary_path.read_text(encoding="utf-8"))
    info.update(
        {
            **helper_info,
            "backend": "cuml_tsne",
            "perplexity": effective_perplexity,
            "max_iter": int(max_iter),
            "rapids_env": rapids_env,
            "input_cache": str(input_path),
            "output_cache": str(output_path),
        }
    )
    return x_tsne, info


def fit_tsne_space(
    x_pre: np.ndarray,
    seed: int,
    perplexity: float,
    max_iter: int,
    reducer: str,
    rapids_env: str,
    cuml_init: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if reducer == "sklearn_tsne":
        return fit_sklearn_tsne_space(x_pre, seed, perplexity, max_iter)
    if reducer == "cuml_tsne":
        return fit_cuml_tsne_space(x_pre, seed, perplexity, max_iter, rapids_env, cuml_init)
    raise ValueError(f"Unsupported reducer={reducer}")


def add_macro_metadata(metadata: list[dict[str, Any]]) -> None:
    for m in metadata:
        m["source_domain"] = str(m["domain"])
        m["macro_domain"] = macro_domain(m["domain"])


def build_patch_metadata(window_meta: list[dict[str, Any]], context_len: int) -> list[dict[str, Any]]:
    patch_len = int(MODEL_SPECS["chronos_2"]["patch_len"])
    metadata: list[dict[str, Any]] = []
    num_patches = context_len // patch_len
    for i, meta in enumerate(window_meta):
        for j in range(num_patches):
            patch_start = j * patch_len
            patch_end = patch_start + patch_len
            metadata.append(
                {
                    **meta,
                    "original_window_index": i,
                    "model": "chronos_2",
                    "patch_index": j,
                    "patch_len": patch_len,
                    "patch_start_in_window": patch_start,
                    "patch_end_in_window": patch_end,
                    "global_start": int(meta["start"] + patch_start),
                    "global_end": int(meta["start"] + patch_end),
                }
            )
    add_macro_metadata(metadata)
    return metadata


def gather_selected_embeddings(
    layer_embeddings: np.ndarray,
    selected_metadata: list[dict[str, Any]],
    original_to_reduced: dict[int, int],
) -> np.ndarray:
    selected: list[np.ndarray] = []
    for meta in selected_metadata:
        reduced_idx = original_to_reduced[int(meta["original_window_index"])]
        selected.append(layer_embeddings[reduced_idx, int(meta["patch_index"])])
    return np.stack(selected).astype(np.float32)


def extract_layer0_only(windows: np.ndarray, batch_size: int) -> np.ndarray:
    sys.path.insert(0, str(CHRONOS_SRC))
    import chronos

    pipeline = chronos.Chronos2Pipeline.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = pipeline.model
    model.eval()
    chunks: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[start : start + batch_size], dtype=torch.float32, device=model.device)
            captured: dict[str, torch.Tensor] = {}

            def hook(_mod, _inp, out):
                hidden = out[0] if isinstance(out, tuple) else out.hidden_states
                captured["layer_0"] = hidden.detach()

            handle = model.encoder.block[0].register_forward_hook(hook)
            _encoder_outputs, _loc_scale, _future_mask, num_context_patches = model.encode(
                context=batch,
                num_output_patches=1,
            )
            handle.remove()
            hidden = captured["layer_0"][:, :num_context_patches].float()
            if not torch.isfinite(hidden).all():
                bad = int((~torch.isfinite(hidden)).sum().item())
                raise RuntimeError(
                    f"Non-finite layer_0 hidden states in batch start={start}, "
                    f"batch_size={len(batch)}, nonfinite_values={bad}"
                )
            chunks.append(hidden.cpu().numpy())

    del pipeline, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0)


def select_window_balanced_patch_indices(
    metadata: list[dict[str, Any]],
    max_per_macro_domain: int,
    max_per_dataset_within_macro_domain: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    macro_groups: dict[str, dict[str, dict[int, list[int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, meta in enumerate(metadata):
        macro_groups[str(meta["macro_domain"])][str(meta["dataset"])][int(meta["original_window_index"])].append(i)

    selected: list[int] = []
    macro_order = MACRO_DOMAIN_ORDER + sorted(set(macro_groups) - set(MACRO_DOMAIN_ORDER))
    for macro in macro_order:
        if macro not in macro_groups:
            continue
        macro_candidates: list[int] = []
        for windows_by_dataset in macro_groups[macro].values():
            window_ids = sorted(windows_by_dataset)
            if not window_ids:
                continue
            patches_per_window = max(len(v) for v in windows_by_dataset.values())
            max_windows = max(1, math.ceil(max_per_dataset_within_macro_domain / max(1, patches_per_window)))
            take_windows = min(max_windows, len(window_ids))
            chosen_windows = rng.choice(np.asarray(window_ids), size=take_windows, replace=False).astype(int).tolist()
            dataset_candidates = [idx for wid in chosen_windows for idx in windows_by_dataset[int(wid)]]
            if len(dataset_candidates) > max_per_dataset_within_macro_domain:
                dataset_candidates = rng.choice(
                    np.asarray(dataset_candidates),
                    size=max_per_dataset_within_macro_domain,
                    replace=False,
                ).astype(int).tolist()
            macro_candidates.extend(dataset_candidates)
        if len(macro_candidates) > max_per_macro_domain:
            macro_candidates = rng.choice(np.asarray(macro_candidates), size=max_per_macro_domain, replace=False).astype(int).tolist()
        selected.extend(macro_candidates)
    return np.asarray(sorted(selected), dtype=int)


def cache_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    stem = (
        f"layer0_selected_w{args.windows_per_dataset}_ctx{args.context_len}_seed{args.seed}"
        f"_mm{args.max_per_macro_domain}_md{args.max_per_dataset_within_macro_domain}_ms{args.max_per_source_domain}"
        f"_{args.selection_mode}"
    )
    return (
        CACHE / f"{stem}_embeddings.npy",
        CACHE / f"{stem}_metadata.json",
        CACHE / f"{stem}_dataset_summary.json",
        CACHE / f"{stem}_raw_patches.npy",
    )


def gather_selected_raw_patches(windows: np.ndarray, selected_metadata: list[dict[str, Any]]) -> np.ndarray:
    patches: list[np.ndarray] = []
    for meta in selected_metadata:
        window_idx = int(meta["original_window_index"])
        start = int(meta["patch_start_in_window"])
        end = int(meta["patch_end_in_window"])
        patches.append(windows[window_idx, start:end])
    return np.stack(patches).astype(np.float32)


def load_or_build_raw_patches(args: argparse.Namespace, metadata: list[dict[str, Any]], timings: dict[str, float]) -> np.ndarray:
    _emb_path, _meta_path, _ds_path, raw_path = cache_paths(args)
    if args.use_cache and raw_path.exists():
        t0 = time.perf_counter()
        patches = np.load(raw_path)
        timings["load_raw_patch_cache_sec"] = time.perf_counter() - t0
        return patches.astype(np.float32, copy=False)

    t0 = time.perf_counter()
    windows, _window_meta, _dataset_summary = sample_windows(
        DATA_ROOT,
        args.context_len,
        args.windows_per_dataset,
        args.seed,
    )
    patches = gather_selected_raw_patches(windows, metadata)
    timings["rebuild_raw_patches_sec"] = time.perf_counter() - t0
    if args.use_cache:
        np.save(raw_path, patches)
    return patches


def stratified_cap_indices(metadata: list[dict[str, Any]], max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or len(metadata) <= max_points:
        return np.arange(len(metadata), dtype=int)
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        groups[str(meta["macro_domain"])].append(i)

    selected: list[int] = []
    remaining_budget = max_points
    remaining_groups = {k: v[:] for k, v in groups.items()}
    ordered = MACRO_DOMAIN_ORDER + sorted(set(remaining_groups) - set(MACRO_DOMAIN_ORDER))
    while remaining_budget > 0 and remaining_groups:
        active = [g for g in ordered if g in remaining_groups and remaining_groups[g]]
        if not active:
            break
        per_group = max(1, remaining_budget // len(active))
        for group in active:
            values = remaining_groups[group]
            take = min(per_group, len(values), remaining_budget)
            if take <= 0:
                continue
            picks = rng.choice(np.asarray(values), size=take, replace=False).astype(int).tolist()
            selected.extend(picks)
            picked = set(picks)
            remaining_groups[group] = [v for v in values if v not in picked]
            remaining_budget -= take
            if remaining_budget <= 0:
                break
        remaining_groups = {k: v for k, v in remaining_groups.items() if v}
    return np.asarray(sorted(selected), dtype=int)


def load_or_build_layer0(args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    emb_path, meta_path, ds_path, raw_path = cache_paths(args)
    timings: dict[str, float] = {}
    if args.use_cache and emb_path.exists() and meta_path.exists() and ds_path.exists():
        t0 = time.perf_counter()
        embeddings = np.load(emb_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        dataset_summary = json.loads(ds_path.read_text(encoding="utf-8"))
        timings["load_cache_sec"] = time.perf_counter() - t0
        return embeddings, metadata, dataset_summary, timings

    t0 = time.perf_counter()
    windows, window_meta, dataset_summary = sample_windows(
        DATA_ROOT,
        args.context_len,
        args.windows_per_dataset,
        args.seed,
    )
    timings["sample_windows_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    patch_metadata = build_patch_metadata(window_meta, args.context_len)
    if args.selection_mode == "window_balanced":
        selected_patch_idx = select_window_balanced_patch_indices(
            patch_metadata,
            args.max_per_macro_domain,
            args.max_per_dataset_within_macro_domain,
            args.seed,
        )
    elif args.selection_mode == "patch_balanced":
        selected_patch_idx = select_balanced_indices(
            patch_metadata,
            "macro_domain",
            args.max_per_macro_domain,
            args.max_per_dataset_within_macro_domain,
            args.max_per_source_domain,
            args.seed,
        )
    elif args.selection_mode == "source_domain_balanced":
        selected_patch_idx = select_domain_balanced_indices(
            patch_metadata,
            args.max_per_source_domain,
            args.seed,
        )
    else:
        raise ValueError(f"Unsupported selection_mode={args.selection_mode}")
    selected_metadata = [patch_metadata[int(i)] for i in selected_patch_idx]
    unique_window_indices = sorted({int(m["original_window_index"]) for m in selected_metadata})
    original_to_reduced = {orig: i for i, orig in enumerate(unique_window_indices)}
    reduced_windows = windows[unique_window_indices]
    timings["preselect_patches_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    layer0 = extract_layer0_only(reduced_windows, args.batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["extract_layer0_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    embeddings = gather_selected_embeddings(layer0, selected_metadata, original_to_reduced)
    raw_patches = gather_selected_raw_patches(windows, selected_metadata)
    metadata = selected_metadata
    timings["gather_selected_embeddings_sec"] = time.perf_counter() - t0
    timings["candidate_windows"] = float(len(windows))
    timings["unique_extracted_windows"] = float(len(unique_window_indices))
    timings["selected_patches"] = float(len(selected_metadata))

    if args.use_cache:
        np.save(emb_path, embeddings)
        np.save(raw_path, raw_patches)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        ds_path.write_text(json.dumps(dataset_summary, ensure_ascii=False), encoding="utf-8")
    return embeddings, metadata, dataset_summary, timings


def center_nearest_indices(cluster_space: np.ndarray, labels: np.ndarray, centers: np.ndarray, k: int) -> np.ndarray:
    nearest: list[int] = []
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            nearest.append(-1)
            continue
        distances = np.linalg.norm(cluster_space[idx] - centers[cid], axis=1)
        nearest.append(int(idx[int(np.argmin(distances))]))
    return np.asarray(nearest, dtype=int)


def cluster_display_positions(x_tsne: np.ndarray, labels: np.ndarray, center_nearest: np.ndarray, k: int) -> np.ndarray:
    positions = np.zeros((k, 2), dtype=np.float32)
    for cid in range(k):
        if center_nearest[cid] >= 0:
            positions[cid] = x_tsne[int(center_nearest[cid])]
        else:
            idx = np.where(labels == cid)[0]
            positions[cid] = np.mean(x_tsne[idx], axis=0) if len(idx) else 0.0
    return positions


def draw_atlas(x_tsne: np.ndarray, labels: np.ndarray, display_positions: np.ndarray, k: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    display_labels = labels + 1
    cmap = plt.get_cmap("tab10", k)
    norm = BoundaryNorm(np.arange(0.5, k + 1.5, 1.0), k)
    sc = ax.scatter(
        x_tsne[:, 0],
        x_tsne[:, 1],
        c=display_labels,
        s=7.0,
        cmap=cmap,
        norm=norm,
        alpha=0.62,
        linewidths=0,
    )
    ax.scatter(display_positions[:, 0], display_positions[:, 1], c="black", s=58, marker="x", linewidths=2.0)
    for cid in range(k):
        ax.text(
            display_positions[cid, 0] + 0.8,
            display_positions[cid, 1] + 0.8,
            f"C{cid + 1}",
            fontsize=10.5,
            weight="bold",
            color="black",
        )
    ax.set_xlabel("t-SNE 1", fontsize=16)
    ax.set_ylabel("t-SNE 2", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#1f2933")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.035, ticks=np.arange(1, k + 1))
    cbar.set_label("Cluster", fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def count_metadata(metadata: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for meta in metadata:
        counts[str(meta.get(key))] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def save_intermediates(
    args: argparse.Namespace,
    embeddings: np.ndarray,
    raw_patches: np.ndarray,
    metadata: list[dict[str, Any]],
    x_pca: np.ndarray,
    x_tsne: np.ndarray,
    cluster_input: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    center_nearest: np.ndarray,
    display_positions: np.ndarray,
) -> dict[str, Any]:
    if not args.save_intermediates:
        return {}
    out_dir = args.intermediate_dir / args.output.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "embeddings": out_dir / "embeddings.npy",
        "raw_patches": out_dir / "raw_patches.npy",
        "pca_coordinates": out_dir / "pca_coordinates.npy",
        "tsne_coordinates": out_dir / "tsne_coordinates.npy",
        "cluster_input_coordinates": out_dir / "cluster_input_coordinates.npy",
        "cluster_labels": out_dir / "cluster_labels.npy",
        "cluster_centers": out_dir / "cluster_centers.npy",
        "center_nearest_indices": out_dir / "center_nearest_indices.npy",
        "center_nearest_raw_patches": out_dir / "center_nearest_raw_patches.npy",
        "display_positions": out_dir / "display_positions.npy",
        "metadata": out_dir / "metadata.json",
        "cluster_summary": out_dir / "cluster_summary.json",
    }
    np.save(paths["embeddings"], embeddings)
    np.save(paths["raw_patches"], raw_patches)
    np.save(paths["pca_coordinates"], x_pca)
    np.save(paths["tsne_coordinates"], x_tsne)
    np.save(paths["cluster_input_coordinates"], cluster_input)
    np.save(paths["cluster_labels"], labels)
    np.save(paths["cluster_centers"], centers)
    np.save(paths["center_nearest_indices"], center_nearest)
    valid_nearest = np.asarray([i for i in center_nearest.tolist() if i >= 0], dtype=int)
    np.save(paths["center_nearest_raw_patches"], raw_patches[valid_nearest] if len(valid_nearest) else np.empty((0, raw_patches.shape[1]), dtype=np.float32))
    np.save(paths["display_positions"], display_positions)
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    cluster_rows: list[dict[str, Any]] = []
    for cid in range(args.k):
        idx = np.where(labels == cid)[0]
        nearest_idx = int(center_nearest[cid])
        row = {
            "cluster": int(cid + 1),
            "zero_based_cluster": int(cid),
            "size": int(len(idx)),
            "center_nearest_local_index": nearest_idx,
            "center_nearest_metadata": metadata[nearest_idx] if nearest_idx >= 0 else None,
            "top_datasets": count_metadata([metadata[int(i)] for i in idx], "dataset"),
            "top_source_domains": count_metadata([metadata[int(i)] for i in idx], "source_domain"),
            "top_macro_domains": count_metadata([metadata[int(i)] for i in idx], "macro_domain"),
            "top_frequencies": count_metadata([metadata[int(i)] for i in idx], "frequency_minutes"),
            "top_patch_indices": count_metadata([metadata[int(i)] for i in idx], "patch_index"),
        }
        cluster_rows.append(row)
    paths["cluster_summary"].write_text(
        json.dumps(
            {
                "cluster_space": args.cluster_space,
                "visualization_space": "tsne",
                "cluster_count": int(args.k),
                "clusters": cluster_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-per-dataset", type=int, default=500)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--tsne-perplexity", type=float, default=50.0)
    parser.add_argument("--tsne-max-iter", type=int, default=500)
    parser.add_argument("--max-tsne-points", type=int, default=3000)
    parser.add_argument("--reducer", choices=["sklearn_tsne", "cuml_tsne"], default="sklearn_tsne")
    parser.add_argument("--rapids-env", default="rapids-tsne")
    parser.add_argument("--cuml-init", choices=["random", "pca"], default="random")
    parser.add_argument("--cluster-space", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY)
    parser.add_argument("--intermediate-dir", type=Path, default=ASSETS / "intermediates")
    parser.add_argument("--save-intermediates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-per-macro-domain", type=int, default=1500)
    parser.add_argument("--max-per-dataset-within-macro-domain", type=int, default=350)
    parser.add_argument("--max-per-source-domain", type=int, default=900)
    parser.add_argument(
        "--selection-mode",
        choices=["window_balanced", "patch_balanced", "source_domain_balanced"],
        default="window_balanced",
    )
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    embeddings, metadata, dataset_summary, timings = load_or_build_layer0(args)
    raw_patches = load_or_build_raw_patches(args, metadata, timings)

    finite_mask = np.isfinite(embeddings).all(axis=1)
    dropped_nonfinite = int(np.sum(~finite_mask))
    if dropped_nonfinite:
        embeddings = embeddings[finite_mask]
        raw_patches = raw_patches[finite_mask]
        metadata = [m for m, keep in zip(metadata, finite_mask.tolist()) if keep]
    timings["dropped_nonfinite_embeddings"] = float(dropped_nonfinite)

    t0 = time.perf_counter()
    tsne_idx = stratified_cap_indices(metadata, args.max_tsne_points, args.seed)
    embeddings_for_tsne = embeddings[tsne_idx]
    raw_patches_for_tsne = raw_patches[tsne_idx]
    metadata_for_tsne = [metadata[int(i)] for i in tsne_idx]
    timings["tsne_stratified_cap_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    x_pca, pca_info = preprocess_embeddings(embeddings_for_tsne, args.seed, args.pca_dim)
    timings["preprocess_pca_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    x_tsne, tsne_info = fit_tsne_space(
        x_pca,
        args.seed,
        args.tsne_perplexity,
        args.tsne_max_iter,
        args.reducer,
        args.rapids_env,
        args.cuml_init,
    )
    timings["tsne_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cluster_input = x_pca if args.cluster_space == "pca" else x_tsne
    kmeans = KMeans(n_clusters=args.k, random_state=args.seed, n_init=20).fit(cluster_input)
    center_nearest = center_nearest_indices(cluster_input, kmeans.labels_, kmeans.cluster_centers_, args.k)
    display_positions = cluster_display_positions(x_tsne, kmeans.labels_, center_nearest, args.k)
    timings["kmeans_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    draw_atlas(x_tsne, kmeans.labels_, display_positions, args.k, args.output)
    timings["draw_sec"] = time.perf_counter() - t0
    intermediate_paths = save_intermediates(
        args,
        embeddings_for_tsne,
        raw_patches_for_tsne,
        metadata_for_tsne,
        x_pca,
        x_tsne,
        cluster_input,
        kmeans.labels_,
        kmeans.cluster_centers_,
        center_nearest,
        display_positions,
    )
    timings["total_sec"] = time.perf_counter() - total_start
    args.summary_output.write_text(
        json.dumps(
            {
                "source": "end-to-end redraw, no PNG cropping",
                "model": "Chronos-2 archived pilot",
                "representation": "layer_0",
                "k": args.k,
                "cluster_space": args.cluster_space,
                "windows_per_dataset": args.windows_per_dataset,
                "context_len": args.context_len,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "use_cache": args.use_cache,
                "selection_mode": args.selection_mode,
                "num_all_patches": int(len(metadata)),
                "num_selected_patches": int(len(metadata)),
                "num_tsne_patches": int(len(metadata_for_tsne)),
                "max_tsne_points": int(args.max_tsne_points),
                "pca": pca_info,
                "tsne": tsne_info,
                "reducer": args.reducer,
                "cluster_center_marker": "center-nearest point displayed in t-SNE space",
                "intermediate_paths": intermediate_paths,
                "timings_sec": timings,
                "dataset_summary": dataset_summary,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
