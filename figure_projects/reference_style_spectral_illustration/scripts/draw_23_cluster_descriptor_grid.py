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
DEFAULT_OUT = ASSETS / "cluster_descriptor_grid_layer0_k6.png"


def z_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x - float(np.mean(x))) / max(float(np.std(x)), eps)).astype(np.float32)


def first_difference(x: np.ndarray) -> np.ndarray:
    return np.diff(x, prepend=x[:1]).astype(np.float32)


def band_median_iqr(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.median(values, axis=0)
    q1 = np.percentile(values, 25, axis=0)
    q3 = np.percentile(values, 75, axis=0)
    return med.astype(np.float32), q1.astype(np.float32), q3.astype(np.float32)


def power_spectrum(z_patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coeff = np.fft.rfft(z_patches, axis=1)
    power = np.abs(coeff) ** 2
    power[:, 0] = 0.0
    power = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-8)
    freqs = np.fft.rfftfreq(z_patches.shape[1], d=1.0).astype(np.float32)
    return power.astype(np.float32), freqs


def top_counts(values: list[Any], n: int = 4) -> str:
    counts: dict[str, int] = {}
    for v in values:
        counts[str(v)] = counts.get(str(v), 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return ", ".join(f"{k}({v})" for k, v in items)


def draw_row(ax: plt.Axes, center_patch: np.ndarray, med: np.ndarray, q1: np.ndarray, q3: np.ndarray, title: str, color: str) -> None:
    x = np.arange(len(med))
    ax.fill_between(x, q1, q3, color=color, alpha=0.12, lw=0)
    ax.plot(x, med, color=color, lw=2.3, label="median")
    ax.plot(x, center_patch, color=color, lw=3.4, alpha=0.9, label="center nearest")
    ax.axhline(0.0, color="#9aa3af", lw=0.7, alpha=0.65)
    ax.set_xlim(0, len(med) - 1)
    ax.set_title(title, fontsize=10.5, pad=4)
    ax.tick_params(labelsize=8, width=0.7, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#1f2933")


def draw_power_row(ax: plt.Axes, center_power: np.ndarray, med: np.ndarray, q1: np.ndarray, q3: np.ndarray, freqs: np.ndarray, title: str, color: str) -> None:
    x = freqs[1:]
    med = med[1:]
    q1 = q1[1:]
    q3 = q3[1:]
    center_power = center_power[1:]
    ax.fill_between(x, q1, q3, color=color, alpha=0.12, lw=0)
    ax.plot(x, med, color=color, lw=2.3)
    ax.plot(x, center_power, color=color, lw=3.4, alpha=0.9)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(0, max(0.05, float(np.max(q3)) * 1.08))
    ax.set_title(title, fontsize=10.5, pad=4)
    ax.tick_params(labelsize=8, width=0.7, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#1f2933")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_patches = np.load(args.input_dir / "raw_patches.npy")
    labels = np.load(args.input_dir / "cluster_labels.npy")
    centers = np.load(args.input_dir / "cluster_centers.npy")
    cluster_input = np.load(args.input_dir / "cluster_input_coordinates.npy")
    metadata = json.loads((args.input_dir / "metadata.json").read_text(encoding="utf-8"))

    k = int(labels.max()) + 1
    cmap = plt.get_cmap("tab10", k)
    fig = plt.figure(figsize=(16.0, 8.2))
    gs = fig.add_gridspec(3, k, wspace=0.16, hspace=0.34)
    summary: dict[str, Any] = {}

    for cid in range(k):
        idx = np.where(labels == cid)[0]
        distances = np.linalg.norm(cluster_input[idx] - centers[cid], axis=1)
        center_idx = int(idx[int(np.argmin(distances))])
        z = np.stack([z_normalize(raw_patches[i]) for i in idx])
        diff = np.stack([first_difference(z_i) for z_i in z])
        power, freqs = power_spectrum(z)
        z_med, z_q1, z_q3 = band_median_iqr(z)
        d_med, d_q1, d_q3 = band_median_iqr(diff)
        p_med, p_q1, p_q3 = band_median_iqr(power)
        color = cmap(cid)
        ax0 = fig.add_subplot(gs[0, cid])
        ax1 = fig.add_subplot(gs[1, cid])
        ax2 = fig.add_subplot(gs[2, cid])
        draw_row(ax0, z_normalize(raw_patches[center_idx]), z_med, z_q1, z_q3, f"C{cid + 1} raw", color)
        draw_row(ax1, first_difference(z_normalize(raw_patches[center_idx])), d_med, d_q1, d_q3, f"C{cid + 1} diff", color)
        draw_power_row(ax2, power_spectrum(np.stack([z_normalize(raw_patches[center_idx])]))[0][0], p_med, p_q1, p_q3, freqs, f"C{cid + 1} power", color)
        ax0.text(
            0.02,
            1.15,
            f"{metadata[center_idx]['dataset']} | {metadata[center_idx]['domain']}",
            transform=ax0.transAxes,
            fontsize=7.3,
            color="#6b7280",
            ha="left",
            va="bottom",
        )
        summary[f"C{cid + 1}"] = {
            "size": int(len(idx)),
            "center_nearest_index": center_idx,
            "center_nearest_metadata": metadata[center_idx],
            "top_datasets": top_counts([metadata[i]["dataset"] for i in idx]),
            "top_domains": top_counts([metadata[i]["domain"] for i in idx]),
            "top_frequencies": top_counts([metadata[i]["frequency_minutes"] for i in idx]),
        }

    fig.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (output.parent / "cluster_descriptor_grid_layer0_k6_summary.json").write_text(
        json.dumps(
            {
                "source_intermediate_dir": str(args.input_dir),
                "visual_role": "cluster-level prototype/descriptor grid",
                "output": str(output),
                "clusters": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
