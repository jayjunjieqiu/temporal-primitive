from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/temporal_primitive_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "figure_projects" / "reference_style_spectral_illustration"
ASSETS = PROJECT / "assets"
DEFAULT_INPUT = ASSETS / "intermediates" / "central_representation_atlas_layer0_aligned_k6_pca_cluster_cuml_tsne"
DEFAULT_OUT_DIR = ASSETS / "patch_stack_exemplar_cards_layer0_k6"


def z_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x - float(np.mean(x))) / max(float(np.std(x)), eps)).astype(np.float32)


def first_difference(x: np.ndarray) -> np.ndarray:
    return np.diff(x, prepend=x[..., :1], axis=-1).astype(np.float32)


def power_spectrum(z_patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coeff = np.fft.rfft(z_patches, axis=1)
    power = np.abs(coeff) ** 2
    power[:, 0] = 0.0
    power = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-8)
    freqs = np.fft.rfftfreq(z_patches.shape[1], d=1.0).astype(np.float32)
    return power.astype(np.float32), freqs


def score_clusters(
    raw_patches: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    cluster_input: np.ndarray,
    k: int,
    top_n: int,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        distances = np.linalg.norm(cluster_input[idx] - centers[cid], axis=1)
        order = np.argsort(distances)[: min(top_n, len(idx))]
        top_idx = idx[order].astype(int)
        z = np.stack([z_normalize(raw_patches[i]) for i in top_idx])
        mean_shape = np.mean(z, axis=0)
        diffs = np.stack([first_difference(z_i) for z_i in z])
        power, freqs = power_spectrum(z)
        coherence = float(np.mean([np.corrcoef(z_i, mean_shape)[0, 1] if np.std(z_i) > 1e-8 and np.std(mean_shape) > 1e-8 else 0.0 for z_i in z]))
        shape_energy = float(np.mean(np.abs(np.diff(mean_shape))))
        spectral_peakiness = float(np.max(np.mean(power[:, 1:], axis=0)))
        visual_score = 0.55 * coherence + 0.25 * shape_energy + 0.20 * spectral_peakiness
        scored.append(
            {
                "cluster": cid,
                "score": visual_score,
                "coherence": coherence,
                "shape_energy": shape_energy,
                "spectral_peakiness": spectral_peakiness,
                "top_indices": top_idx.tolist(),
                "top_n_indices": top_idx.tolist(),
                "mean_shape": mean_shape.astype(float).tolist(),
                "mean_diff": np.mean(diffs, axis=0).astype(float).tolist(),
                "mean_power": np.mean(power, axis=0).astype(float).tolist(),
                "frequency_bins": freqs.astype(float).tolist(),
            }
        )
    scored.sort(key=lambda r: (-r["score"], r["cluster"]))
    return scored


def draw_patch_stack_card(
    ax: plt.Axes,
    patches: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    show_ylabel: bool = False,
    x_label: str = "time",
) -> None:
    im = ax.imshow(patches, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    n_cols = patches.shape[1]
    if n_cols >= 16:
        ax.set_xticks([0, 3, 7, 11, 15])
    else:
        ax.set_xticks([0, max(0, n_cols // 2), n_cols - 1])
    ax.set_yticks([])
    ax.tick_params(labelsize=7, length=2)
    ax.set_title(title, fontsize=10.0, pad=4)
    if show_ylabel:
        ax.set_ylabel("rank", fontsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#1f2933")
        spine.set_linewidth(0.8)
    ax.set_xlabel(x_label, fontsize=8.5)
    return im


def draw_cluster_card(fig: plt.Figure, outer, row: int, info: dict[str, Any], raw_patches: np.ndarray, top_count: int | None = None) -> None:
    cid = int(info["cluster"])
    idx = np.asarray(info["top_indices"][:top_count] if top_count else info["top_indices"], dtype=int)
    z = np.stack([z_normalize(raw_patches[i]) for i in idx])
    diff = np.stack([first_difference(z_i) for z_i in z])
    power, _freqs = power_spectrum(z)
    vmax_raw = float(np.percentile(np.abs(z), 97))
    vmax_diff = float(np.percentile(np.abs(diff), 97))
    vmax_pow = float(np.percentile(power[:, 1:], 97))
    axs = [fig.add_subplot(outer[row, j]) for j in range(3)]
    im0 = draw_patch_stack_card(axs[0], z, "Raw patch stack", "RdBu_r", -vmax_raw, vmax_raw, show_ylabel=True, x_label="")
    im1 = draw_patch_stack_card(axs[1], diff, "First difference stack", "coolwarm", -vmax_diff, vmax_diff, x_label="time")
    im2 = draw_patch_stack_card(axs[2], power[:, 1:], "Power spectrum stack", "magma", 0.0, vmax_pow, x_label="frequency bin")
    axs[0].text(
        -0.12,
        1.1,
        f"C{cid + 1}",
        transform=axs[0].transAxes,
        fontsize=13,
        fontweight="bold",
        color="#111827",
        ha="left",
        va="top",
    )
    axs[0].text(
        0.0,
        -0.22,
        f"score={info['score']:.2f}  coh={info['coherence']:.2f}",
        transform=axs[0].transAxes,
        fontsize=7.6,
        color="#4b5563",
        ha="left",
        va="top",
    )
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_color("#1f2933")
            spine.set_linewidth(0.8)
    return im0, im1, im2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--selected-count", type=int, default=5)
    parser.add_argument("--card-top-n", type=int, default=24)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--version-tag", default="v2")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_patches = np.load(args.input_dir / "raw_patches.npy")
    labels = np.load(args.input_dir / "cluster_labels.npy")
    centers = np.load(args.input_dir / "cluster_centers.npy")
    cluster_input = np.load(args.input_dir / "cluster_input_coordinates.npy")
    metadata = json.loads((args.input_dir / "metadata.json").read_text(encoding="utf-8"))

    scored_clusters = score_clusters(raw_patches, labels, centers, cluster_input, args.k, args.top_n)
    selected = scored_clusters[: args.selected_count]
    all_by_cluster = sorted(scored_clusters, key=lambda r: int(r["cluster"]))
    combined_path = args.output_dir / f"patch_stack_exemplar_cards_layer0_k6_selected_{args.version_tag}.png"
    all_path = args.output_dir / f"patch_stack_exemplar_cards_layer0_k6_all_clusters_{args.version_tag}.png"

    fig = plt.figure(figsize=(12.0, 12.8))
    outer = fig.add_gridspec(len(selected), 3, wspace=0.18, hspace=0.42)
    for row, info in enumerate(selected):
        draw_cluster_card(fig, outer, row, info, raw_patches, top_count=args.card_top_n)
        cid = int(info["cluster"])
        meta = metadata[int(info["top_indices"][0])]
        fig.axes[row * 3].text(
            0.0,
            1.18,
            f"{meta['dataset']} | {meta['domain']}",
            transform=fig.axes[row * 3].transAxes,
            fontsize=7.5,
            color="#6b7280",
            ha="left",
            va="bottom",
        )
    fig.savefig(combined_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig = plt.figure(figsize=(12.0, 15.2))
    outer = fig.add_gridspec(args.k, 3, wspace=0.18, hspace=0.42)
    for row, info in enumerate(all_by_cluster):
        draw_cluster_card(fig, outer, row, info, raw_patches, top_count=args.card_top_n)
    fig.savefig(all_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    individual_dir = args.output_dir / "cards"
    individual_dir.mkdir(parents=True, exist_ok=True)
    for info in all_by_cluster:
        cid = int(info["cluster"])
        idx = np.asarray(info["top_indices"][: args.card_top_n], dtype=int)
        z = np.stack([z_normalize(raw_patches[i]) for i in idx])
        diff = np.stack([first_difference(z_i) for z_i in z])
        power, _ = power_spectrum(z)
        fig = plt.figure(figsize=(7.8, 2.8))
        gs = fig.add_gridspec(1, 3, wspace=0.18)
        axs = [fig.add_subplot(gs[0, j]) for j in range(3)]
        vmax_raw = float(np.percentile(np.abs(z), 97))
        vmax_diff = float(np.percentile(np.abs(diff), 97))
        vmax_pow = float(np.percentile(power[:, 1:], 97))
        draw_patch_stack_card(axs[0], z, "Raw patch stack", "RdBu_r", -vmax_raw, vmax_raw, show_ylabel=True, x_label="")
        draw_patch_stack_card(axs[1], diff, "First difference", "coolwarm", -vmax_diff, vmax_diff, x_label="time")
        draw_patch_stack_card(axs[2], power[:, 1:], "Power spectrum", "magma", 0.0, vmax_pow, x_label="frequency bin")
        axs[0].text(
            -0.12,
            1.14,
            f"C{cid + 1}",
            transform=axs[0].transAxes,
            fontsize=15,
            fontweight="bold",
            color="#111827",
        )
        fig.savefig(individual_dir / f"C{cid + 1}_patch_stack_card_{args.version_tag}.png", dpi=320, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    selected_rows = []
    for info in selected:
        cid = int(info["cluster"])
        for rank, idx in enumerate(info["top_indices"], start=1):
            selected_rows.append(
                {
                    "cluster": f"C{cid + 1}",
                    "zero_based_cluster": cid,
                    "rank": rank,
                    "local_index": int(idx),
                    "metadata": metadata[int(idx)],
                }
            )

    np.savez_compressed(
        args.output_dir / f"patch_stack_exemplar_cards_layer0_k6_data_{args.version_tag}.npz",
        cluster_input_coordinates=cluster_input,
        cluster_labels=labels,
        cluster_centers=centers,
        raw_patches=raw_patches,
        selected_clusters=np.asarray([int(info["cluster"]) for info in selected], dtype=int),
    )
    (args.output_dir / f"patch_stack_exemplar_cards_layer0_k6_summary_{args.version_tag}.json").write_text(
        json.dumps(
            {
                "source_intermediate_dir": str(args.input_dir),
                "visual_role": "illustrative patch-stack evidence for region-level latent structure",
                "selected_count": len(selected),
                "card_top_n": int(args.card_top_n),
                "selected_clusters": [
                    {
                        "cluster": f"C{int(info['cluster']) + 1}",
                        "zero_based_cluster": int(info["cluster"]),
                        "score": float(info["score"]),
                        "coherence": float(info["coherence"]),
                        "shape_energy": float(info["shape_energy"]),
                        "spectral_peakiness": float(info["spectral_peakiness"]),
                        "top_indices": [int(x) for x in info["top_indices"]],
                    }
                    for info in selected
                ],
                "omitted_clusters": [f"C{i + 1}" for i in range(args.k) if i not in {int(info["cluster"]) for info in selected}],
                "combined_output": str(combined_path),
                "all_cluster_output": str(all_path),
                "individual_cards": [str(individual_dir / f"C{i + 1}_patch_stack_card_{args.version_tag}.png") for i in range(args.k)],
                "selected_examples": selected_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(combined_path)
    print(all_path)


if __name__ == "__main__":
    main()
