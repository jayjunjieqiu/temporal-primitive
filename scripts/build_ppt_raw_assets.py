#!/usr/bin/env python3
"""Build compact raw figure assets for manual PPT layout."""

from __future__ import annotations

import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_ppt_assets import (
    COLORS,
    CONCEPT_COLORS,
    CONCEPT_LABELS,
    FIG,
    compact_count_label,
    condition_metric,
    display_label,
    domain_balanced_rows,
    find_cluster_validation,
    find_parent_subcluster,
    load_prototype_bank,
    save_fig,
    setup_matplotlib,
    taxonomy_prototype,
    trim_image,
    write_json,
)


RAW_OUT = ROOT / "outputs" / "ppt_raw_assets"

PRIOR_ORDER = [
    "flat_low_information",
    "trend",
    "oscillation",
    "impulse_spike",
    "burst_event",
    "level_shift",
    "volatility_shift",
    "intermittent",
    "mixed_uncertain",
]

PRIOR_LABELS = {
    "flat_low_information": "flat / low information",
    "trend": "trend",
    "oscillation": "oscillation",
    "impulse_spike": "spike",
    "burst_event": "burst",
    "level_shift": "level shift",
    "volatility_shift": "volatility shift",
    "intermittent": "intermittent",
    "mixed_uncertain": "mixed / uncertain",
}

PRIOR_CRITERIA = {
    "flat_low_information": "low std and range",
    "trend": "large slope, high R2",
    "oscillation": "FFT power, crossings",
    "impulse_spike": "one extreme point",
    "burst_event": "dense active run",
    "level_shift": "mean jump split",
    "volatility_shift": "variance ratio split",
    "intermittent": "separated active events",
    "mixed_uncertain": "conflict or low confidence",
}

PRIOR_COLORS = [
    COLORS["gray"],
    COLORS["blue"],
    COLORS["violet"],
    COLORS["red"],
    COLORS["amber"],
    COLORS["teal"],
    COLORS["green"],
    COLORS["blue"],
    COLORS["gray"],
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_raw(fig: plt.Figure, name: str) -> Path:
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    path = RAW_OUT / name
    fig.savefig(path, dpi=360, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def save_pil(img: Image.Image, name: str) -> Path:
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    path = RAW_OUT / name
    img.save(path)
    return path


def wrap_lines(text: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def fig_prior_probe_shapes_and_limits() -> Path:
    """Show what the prior-guided motif probe looks like and why it is not ground truth."""
    summary = read_json(ROOT / "outputs/second_pilot_discovery_summary.json")
    metric_specs = [
        ("TimesFM-2.5\nlayer 10", "timesfm_2_5", "layer_10"),
        ("Chronos-2\nlayer 11", "chronos_2", "layer_11"),
        ("Chronos-2-small\nlayer 5", "chronos_2_small", "layer_5"),
    ]
    metrics = [
        ("taxonomy_v0", "prior probe", COLORS["green"]),
        ("domain", "domain", COLORS["amber"]),
        ("frequency", "frequency", COLORS["teal"]),
        ("patch_index", "patch index", COLORS["red"]),
    ]
    values = np.asarray(
        [
            [summary["models"][model]["layers"][layer]["domain_balanced"]["nmi"][key] for key, _label, _color in metrics]
            for _display, model, layer in metric_specs
        ],
        dtype=np.float64,
    )

    fig = plt.figure(figsize=(11.6, 5.7))
    outer = fig.add_gridspec(1, 2, left=0.055, right=0.98, bottom=0.11, top=0.93, wspace=0.22, width_ratios=[1.28, 1.0])

    left = outer[0, 0].subgridspec(3, 3, hspace=0.34, wspace=0.24)
    for i, name in enumerate(PRIOR_ORDER):
        ax = fig.add_subplot(left[i // 3, i % 3])
        y = taxonomy_prototype(name, n=64)
        x = np.linspace(0, 1, len(y))
        pad = max(0.25, 0.12 * float(np.ptp(y)))
        ax.plot(x, y, color=PRIOR_COLORS[i], lw=1.9, solid_capstyle="round")
        ax.axhline(0, color=COLORS["grid"], lw=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(float(y.min() - pad), float(y.max() + pad))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(PRIOR_LABELS[name], loc="left", fontsize=9.8)
        ax.text(0.0, -0.21, PRIOR_CRITERIA[name], transform=ax.transAxes, fontsize=6.9, color=COLORS["muted"], va="top")
        if i == 0:
            ax.text(-0.18, 1.30, "A", transform=ax.transAxes, fontsize=14, fontweight="bold", color=COLORS["ink"])
            ax.text(0.02, 1.30, "Prior-probe prototype shapes", transform=ax.transAxes, fontsize=11.6, fontweight="bold", color=COLORS["ink"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
            spine.set_linewidth(0.7)

    ax = fig.add_subplot(outer[0, 1])
    x = np.arange(len(metric_specs))
    width = 0.18
    for j, (_key, label, color) in enumerate(metrics):
        ax.bar(x + (j - 1.5) * width, values[:, j], width=width, color=color, label=label, alpha=0.92)
        for xi, yi in zip(x + (j - 1.5) * width, values[:, j]):
            ax.text(xi, yi + 0.012, f"{yi:.2f}", ha="center", va="bottom", fontsize=7.4, color=COLORS["ink"])
    ax.set_title("B  Cluster alignment is modest", loc="left", fontsize=11.6, fontweight="bold")
    ax.set_ylabel("NMI with KMeans labels")
    ax.set_xticks(x)
    ax.set_xticklabels([d for d, _m, _l in metric_specs], fontsize=8.8)
    ax.set_ylim(0, 0.50)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
    ax.legend(frameon=False, fontsize=8.0, loc="upper right")
    ax.text(
        0.00,
        -0.19,
        "The prior probe provides interpretable weak labels, but hidden-space clusters also align with domain, frequency, and position.",
        transform=ax.transAxes,
        fontsize=8.8,
        color=COLORS["muted"],
        va="top",
        wrap=True,
    )

    write_json(
        RAW_OUT / "prior_probe_shapes_and_limits_summary.json",
        {
            "purpose": "Slide 3 evidence that the prior-guided motif probe is an interpretable weak probe, not ground truth",
            "motif_classes": [PRIOR_LABELS[name] for name in PRIOR_ORDER],
            "nmi_metrics": {
                display.replace("\n", " "): {label: float(values[i, j]) for j, (_key, label, _color) in enumerate(metrics)}
                for i, (display, _model, _layer) in enumerate(metric_specs)
            },
            "interpretation": "prior-probe NMI is modest and not uniquely dominant across models; clusters also encode domain/frequency/position confounders",
        },
    )
    return save_raw(fig, "fig_prior_probe_shapes_and_limits.png")


def fig_prior_probe_shapes_only() -> Path:
    """Standalone prior-probe prototype shapes for manual PPT layout."""
    fig = plt.figure(figsize=(8.3, 6.15))
    grid = fig.add_gridspec(3, 3, left=0.06, right=0.98, bottom=0.09, top=0.94, hspace=0.52, wspace=0.27)
    for i, name in enumerate(PRIOR_ORDER):
        ax = fig.add_subplot(grid[i // 3, i % 3])
        y = taxonomy_prototype(name, n=64)
        x = np.linspace(0, 1, len(y))
        pad = max(0.25, 0.12 * float(np.ptp(y)))
        ax.plot(x, y, color=PRIOR_COLORS[i], lw=2.1, solid_capstyle="round")
        ax.axhline(0, color=COLORS["grid"], lw=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(float(y.min() - pad), float(y.max() + pad))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(PRIOR_LABELS[name], loc="left", fontsize=10.8)
        ax.text(0.0, -0.24, PRIOR_CRITERIA[name], transform=ax.transAxes, fontsize=7.8, color=COLORS["muted"], va="top")
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
            spine.set_linewidth(0.8)
    return save_raw(fig, "fig_prior_probe_prototype_shapes.png")


def fig_prior_probe_alignment_only() -> Path:
    """Standalone NMI evidence that prior-guided motif labels are weak probes."""
    summary = read_json(ROOT / "outputs/second_pilot_discovery_summary.json")
    metric_specs = [
        ("TimesFM-2.5\nlayer 10", "timesfm_2_5", "layer_10"),
        ("Chronos-2\nlayer 11", "chronos_2", "layer_11"),
        ("Chronos-2-small\nlayer 5", "chronos_2_small", "layer_5"),
    ]
    metrics = [
        ("taxonomy_v0", "prior probe", COLORS["green"]),
        ("domain", "domain", COLORS["amber"]),
        ("frequency", "frequency", COLORS["teal"]),
        ("patch_index", "patch index", COLORS["red"]),
    ]
    values = np.asarray(
        [
            [summary["models"][model]["layers"][layer]["domain_balanced"]["nmi"][key] for key, _label, _color in metrics]
            for _display, model, layer in metric_specs
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(metric_specs))
    width = 0.17
    for j, (_key, label, color) in enumerate(metrics):
        ax.bar(x + (j - 1.5) * width, values[:, j], width=width, color=color, label=label, alpha=0.92)
        for xi, yi in zip(x + (j - 1.5) * width, values[:, j]):
            ax.text(xi, yi + 0.011, f"{yi:.2f}", ha="center", va="bottom", fontsize=8.6, color=COLORS["ink"])
    ax.set_title("Cluster alignment with the prior probe is modest", loc="left", fontsize=13.0, fontweight="bold")
    ax.set_ylabel("NMI with KMeans labels")
    ax.set_xticks(x)
    ax.set_xticklabels([d for d, _m, _l in metric_specs], fontsize=9.6)
    ax.set_ylim(0, 0.50)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.7)
    ax.legend(frameon=False, fontsize=9.0, ncol=4, loc="upper center", bbox_to_anchor=(0.50, 1.02))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.20, top=0.83)
    return save_raw(fig, "fig_prior_probe_alignment_evidence.png")


def fig_prior_probe_distribution() -> Path:
    """Show prior-guided probe label distribution on the domain-balanced real patch bank."""
    summary = read_json(ROOT / "outputs/second_pilot_discovery_summary.json")
    specs = [
        ("TimesFM-2.5", "patch length 32", "timesfm_2_5", "layer_10"),
        ("Chronos-2", "patch length 16", "chronos_2", "layer_11"),
        ("Chronos-2-small", "patch length 16", "chronos_2_small", "layer_5"),
    ]
    counts_by_model = []
    totals = []
    for _display, _patch, model, layer in specs:
        rows = summary["models"][model]["layers"][layer]["domain_balanced"]["global_top"]["taxonomy_v0"]
        counts = {row["value"]: int(row["count"]) for row in rows}
        counts_by_model.append(counts)
        totals.append(sum(counts.values()))

    dist_colors = {
        "flat_low_information": "#7f8a90",
        "trend": "#2d6f9f",
        "oscillation": "#6f5aa8",
        "impulse_spike": "#b84a4d",
        "burst_event": "#bd7620",
        "level_shift": "#2d9383",
        "volatility_shift": "#3b7f3b",
        "intermittent": "#5c8fb8",
        "mixed_uncertain": "#99a1a6",
    }
    fig, ax = plt.subplots(figsize=(8.6, 3.35))
    y = np.arange(len(specs))
    left = np.zeros(len(specs), dtype=np.float64)
    for i, label in enumerate(PRIOR_ORDER):
        vals = np.asarray([counts.get(label, 0) / total for counts, total in zip(counts_by_model, totals)], dtype=np.float64)
        ax.barh(y, vals, left=left, color=dist_colors[label], edgecolor="white", linewidth=0.6, label=PRIOR_LABELS[label])
        for yi, li, vi in zip(y, left, vals):
            if vi >= 0.08:
                ax.text(li + vi / 2, yi, f"{vi*100:.0f}%", ha="center", va="center", fontsize=7.2, color="white", fontweight="bold")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([f"{name}\n{patch}, n={total}" for (name, patch, _m, _l), total in zip(specs, totals)], fontsize=9.0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of domain-balanced real patches")
    ax.set_title("Prior-probe label distribution on the real patch bank", loc="left", fontsize=13.0, fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
    ax.legend(frameon=False, fontsize=7.3, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.50), borderaxespad=0.0)
    ax.invert_yaxis()
    write_json(
        RAW_OUT / "prior_probe_distribution_summary.json",
        {
            "purpose": "distribution of prior-guided motif probe labels on the domain-balanced real patch bank",
            "models": [
                {
                    "model": display,
                    "patch_length": patch,
                    "total": int(total),
                    "counts": {PRIOR_LABELS[k]: int(counts.get(k, 0)) for k in PRIOR_ORDER},
                }
                for (display, patch, _m, _l), counts, total in zip(specs, counts_by_model, totals)
            ],
            "interpretation": "mixed/uncertain and level-shift-like patches dominate the weak probe distribution; this reinforces that prior-guided labels are probes rather than balanced supervision.",
        },
    )
    fig.subplots_adjust(left=0.25, right=0.78, bottom=0.18, top=0.82)
    return save_raw(fig, "fig_prior_probe_distribution_real_patch_bank.png")


def fig_discovery_protocol_flow() -> Path:
    """Publication-style process figure for the discover-first, name-second protocol."""
    fig, ax = plt.subplots(figsize=(10.8, 2.45))
    ax.axis("off")
    steps = [
        ("Cross-domain\npatch bank", "heterogeneous\nreal series"),
        ("Frozen TSFM\nrepresentations", "Chronos and\nTimesFM"),
        ("Candidate\nclustering", "PCA, KMeans,\nstability"),
        ("Original-space\ninspection", "prototype and\nshape coherence"),
        ("Confounder\naudit", "domain, frequency,\nposition"),
        ("Controlled\nretrieval", "candidate motif\nfamily"),
    ]
    xs = np.linspace(0.07, 0.93, len(steps))
    y = 0.58
    box_w = 0.135
    box_h = 0.42
    for i, ((title, subtitle), x) in enumerate(zip(steps, xs)):
        color = COLORS["blue"] if i in (1, 2) else COLORS["green"] if i == 5 else COLORS["gray"]
        rect = plt.Rectangle((x - box_w / 2, y - box_h / 2), box_w, box_h, transform=ax.transAxes, fill=True, fc="white", ec=color, lw=1.4)
        ax.add_patch(rect)
        ax.text(x, y + 0.065, title, transform=ax.transAxes, ha="center", va="center", fontsize=9.2, fontweight="bold", color=COLORS["ink"])
        ax.text(x, y - 0.105, subtitle, transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color=COLORS["muted"])
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - box_w / 2 - 0.01, y),
                xytext=(x + box_w / 2 + 0.01, y),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color=COLORS["muted"], lw=1.2),
            )
    ax.text(0.50, 0.13, "KMeans labels generate candidates; motif names require audit.", transform=ax.transAxes, ha="center", fontsize=10.5, fontweight="bold", color=COLORS["red"])
    return save_raw(fig, "fig_discover_first_protocol_flow.png")


def fig_hypothesis_evidence_map() -> Path:
    """Map each research hypothesis to the actual evidence slides/assets."""
    fig, ax = plt.subplots(figsize=(10.6, 3.65))
    ax.axis("off")
    rows = [
        ("H1", "Cross-domain temporal primitives", "Candidate families should recur across heterogeneous domains.", "Original-space prototypes; controlled retrieval; cross-model bars", "partially supported"),
        ("H2", "Contextual reorganization", "Hidden states should reorganize local patch vocabulary.", "Raw to projection to hidden lineage; NMI audit", "supported with caveats"),
        ("H3", "Scale, architecture, confounders", "Chronos-2-small, Chronos-2, and TimesFM may organize motifs differently.", "Model-native candidates; position and domain audit", "open but testable"),
    ]
    ax.text(0.035, 0.93, "Do the slides answer the hypotheses?", transform=ax.transAxes, fontsize=14, fontweight="bold", color=COLORS["ink"])
    y0 = 0.70
    for i, (hid, title, question, evidence, status) in enumerate(rows):
        y = y0 - i * 0.25
        ax.add_patch(plt.Rectangle((0.035, y - 0.085), 0.93, 0.18, transform=ax.transAxes, fill=False, ec=COLORS["grid"], lw=0.9))
        ax.text(0.065, y, hid, transform=ax.transAxes, fontsize=18, fontweight="bold", color=COLORS["blue"], va="center")
        ax.text(0.16, y + 0.045, title, transform=ax.transAxes, fontsize=10.2, fontweight="bold", color=COLORS["ink"], va="center")
        ax.text(0.16, y - 0.030, wrap_lines(question, 45), transform=ax.transAxes, fontsize=7.9, color=COLORS["muted"], va="center")
        ax.text(0.54, y, wrap_lines(evidence, 34), transform=ax.transAxes, fontsize=8.4, color=COLORS["ink"], va="center")
        status_color = COLORS["green"] if "supported" in status else COLORS["amber"]
        ax.text(0.88, y, status, transform=ax.transAxes, fontsize=8.4, color=status_color, fontweight="bold", ha="center", va="center")
    ax.text(0.035, 0.055, "Current claim: a discovery protocol and pilot candidate families, not a final motif taxonomy.", transform=ax.transAxes, fontsize=9.0, color=COLORS["muted"])
    return save_raw(fig, "fig_hypothesis_evidence_map.png")


def fig_prior_probe_literature_map() -> Path:
    """Compact evidence map from TSDM literature to our deterministic prior-guided motif probe."""
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axis("off")
    rows = [
        ("Motif discovery", "Matrix Profile", "recurring subsequences, discords", "Yeh et al., 2016"),
        ("Shapelet explanation", "Shapelets", "discriminative local shapes", "Ye and Keogh, 2009"),
        ("Symbolic patterns", "PAA and SAX", "trend, oscillation, coarse words", "Lin et al., 2007"),
        ("Segmentation", "Change-point detection", "level and variance shifts", "Truong et al., 2020"),
        ("Event anomalies", "Robust outlier and runs", "spike, burst, intermittent events", "Chandola et al., 2009"),
    ]
    ax.text(0.03, 0.93, "Literature roots of the prior-guided motif probe", transform=ax.transAxes, fontsize=14, fontweight="bold", color=COLORS["ink"])
    ax.text(0.03, 0.86, "The probe operationalizes familiar TSDM concepts; it is an interpretation anchor, not supervision.", transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"])
    headers = ["TSDM thread", "Method family", "What it contributes", "Anchor reference"]
    xs = [0.04, 0.25, 0.48, 0.80]
    for x, h in zip(xs, headers):
        ax.text(x, 0.76, h, transform=ax.transAxes, fontsize=9.6, fontweight="bold", color=COLORS["ink"])
    for i, row in enumerate(rows):
        y = 0.65 - i * 0.115
        ax.add_patch(plt.Rectangle((0.03, y - 0.043), 0.94, 0.082, transform=ax.transAxes, fill=True, fc="#f8faf9" if i % 2 == 0 else "white", ec=COLORS["grid"], lw=0.55))
        for x, text, width in zip(xs, row, [18, 22, 30, 22]):
            ax.text(x, y, wrap_lines(text, width), transform=ax.transAxes, fontsize=8.6, color=COLORS["ink"], va="center")
    ax.text(0.03, 0.08, "Operational detectors used in this project: linear fit, spectral concentration, robust z-score, active-run statistics, mean/variance split scores, and conflict-aware uncertainty.", transform=ax.transAxes, fontsize=8.8, color=COLORS["muted"])
    return save_raw(fig, "fig_prior_probe_literature_map.png")


def fig_evidence_gap_audit() -> Path:
    """Advisor-facing audit of claims that need evidence or cautious wording."""
    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    ax.axis("off")
    rows = [
        ("Prior probe is not ground truth", "now covered", "prototype shapes, NMI, real-patch distribution"),
        ("Cluster is a motif family", "covered for strongest case", "original-space prototypes and retrieval"),
        ("Other clusters are not hidden", "covered", "cluster outcome gallery and negative control"),
        ("Chronos-native evidence", "pilot only", "native candidates shown; full retrieval remains next step"),
        ("Final taxonomy claim", "not yet", "must stay as model-derived taxonomy pilot"),
    ]
    ax.text(0.03, 0.91, "Evidence gap audit for the advisor meeting", transform=ax.transAxes, fontsize=14, fontweight="bold", color=COLORS["ink"])
    ax.text(0.03, 0.83, "Use this as a checklist: strong claims need figures; unfinished claims should be phrased as next steps.", transform=ax.transAxes, fontsize=9.2, color=COLORS["muted"])
    headers = ["Claim or risk", "Current status", "Evidence to show"]
    xs = [0.05, 0.49, 0.69]
    for x, h in zip(xs, headers):
        ax.text(x, 0.70, h, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=COLORS["ink"])
    status_colors = {"now covered": COLORS["green"], "covered for strongest case": COLORS["green"], "covered": COLORS["green"], "pilot only": COLORS["amber"], "not yet": COLORS["red"]}
    for i, (claim, status, evidence) in enumerate(rows):
        y = 0.59 - i * 0.105
        ax.add_patch(plt.Rectangle((0.035, y - 0.040), 0.94, 0.078, transform=ax.transAxes, fill=False, ec=COLORS["grid"], lw=0.6))
        ax.text(xs[0], y, claim, transform=ax.transAxes, fontsize=8.7, color=COLORS["ink"], va="center")
        ax.text(xs[1], y, wrap_lines(status, 18), transform=ax.transAxes, fontsize=8.0, color=status_colors[status], va="center", fontweight="bold")
        ax.text(xs[2], y, wrap_lines(evidence, 42), transform=ax.transAxes, fontsize=8.0, color=COLORS["muted"], va="center")
    return save_raw(fig, "fig_evidence_gap_audit.png")


def fig_timesfm_clustering_triptych() -> Path:
    sources = [
        ("A", "KMeans candidates", FIG / "second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_pca_clusters.png"),
        ("B", "Prior motif probe", FIG / "second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_pca_taxonomy_v0.png"),
        ("C", "Patch index", FIG / "second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_pca_patch_index.png"),
    ]
    panel_w, panel_h, title_h, gap = 980, 820, 96, 70
    canvas = Image.new("RGB", (panel_w * 3 + gap * 2, panel_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        from scripts.build_ppt_assets import font
        label_font = font(38, True)
        title_font = font(32, True)
    except Exception:
        label_font = None
        title_font = None
    for i, (letter, title, path) in enumerate(sources):
        img = trim_image(path, crop_top=70, crop_bottom=55, crop_left=45, crop_right=245)
        img.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
        x = i * (panel_w + gap)
        draw.text((x + 4, 18), letter, font=label_font, fill=COLORS["ink"])
        draw.text((x + 64, 22), title, font=title_font, fill=COLORS["ink"])
        canvas.paste(img, (x + (panel_w - img.width) // 2, title_h + (panel_h - img.height) // 2))
    return save_pil(canvas, "fig_timesfm_clustering_triptych_labeled.png")


def fig_domain_balanced_falling_family() -> Path:
    grouped = load_prototype_bank()
    concept = "strong_falling_transition"
    rows = grouped[concept]
    balanced = domain_balanced_rows(rows, max_per_domain=5)
    full_mat = np.asarray([r["z_patch"] for r in rows], dtype=np.float64)
    mat = np.asarray([r["z_patch"] for r in balanced], dtype=np.float64)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    full_mean = full_mat.mean(axis=0)
    corr = np.asarray([np.corrcoef(row, mean)[0, 1] for row in mat])
    order = np.argsort(-corr)
    top = mat[order[:9]]
    top_corr = corr[order[:9]]

    domains: dict[str, int] = defaultdict(int)
    patches: dict[str, int] = defaultdict(int)
    full_domains: dict[str, int] = defaultdict(int)
    for r in rows:
        full_domains[r["domain"]] += 1
    for r in balanced:
        domains[r["domain"]] += 1
        patches[f"p{r['timesfm_patch_index']}"] += 1

    fig = plt.figure(figsize=(11.8, 5.8))
    gs = fig.add_gridspec(2, 4, left=0.07, right=0.98, bottom=0.12, top=0.94, hspace=0.34, wspace=0.34, width_ratios=[1.25, 1, 1, 1])
    x = np.linspace(0, 1, mat.shape[1])

    ax = fig.add_subplot(gs[:, 0])
    ax.fill_between(x, mean - std, mean + std, color=COLORS["blue"], alpha=0.15, lw=0)
    ax.plot(x, mean, color=COLORS["blue"], lw=2.4, label="domain-balanced mean")
    ax.plot(x, full_mean, color=COLORS["gray"], lw=1.3, ls="--", label="full-bank mean")
    ax.axhline(0, color=COLORS["grid"], lw=0.8)
    ax.set_title("A  Prototype curve", loc="left", fontsize=11.5)
    ax.set_xlabel("normalized time")
    ax.set_ylabel("z-normalized value")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)

    sub = gs[:, 1:3].subgridspec(3, 3, hspace=0.14, wspace=0.12)
    ymin, ymax = float(top.min() - 0.3), float(top.max() + 0.3)
    for i, patch in enumerate(top):
        pax = fig.add_subplot(sub[i // 3, i % 3])
        pax.plot(x, patch, color=COLORS["blue"], lw=1.5)
        pax.axhline(0, color=COLORS["grid"], lw=0.55)
        pax.set_xlim(0, 1)
        pax.set_ylim(ymin, ymax)
        pax.set_xticks([])
        pax.set_yticks([])
        if i == 0:
            pax.set_title("B  Representative patches", loc="left", fontsize=11.5)
        pax.text(0.04, 0.88, f"r={top_corr[i]:.2f}", transform=pax.transAxes, fontsize=7.8, color=COLORS["muted"])
        for spine in pax.spines.values():
            spine.set_color(COLORS["grid"])
            spine.set_linewidth(0.7)

    axd = fig.add_subplot(gs[0, 3])
    ditems = sorted(domains.items(), key=lambda kv: (-kv[1], kv[0]))
    axd.barh(range(len(ditems)), [v for _, v in ditems], color=COLORS["blue"], alpha=0.88)
    axd.set_yticks(range(len(ditems)))
    axd.set_yticklabels([display_label(k) for k, _ in ditems], fontsize=8)
    axd.invert_yaxis()
    axd.set_title("C  Display domains", loc="left", fontsize=11.5)
    axd.set_xlabel("count")
    axd.grid(axis="x", color=COLORS["grid"], lw=0.6)

    axp = fig.add_subplot(gs[1, 3])
    pitems = sorted(patches.items())
    axp.bar([k for k, _ in pitems], [v for _, v in pitems], color=COLORS["teal"], alpha=0.9)
    axp.set_title("D  Patch positions", loc="left", fontsize=11.5)
    axp.set_ylabel("count")
    axp.grid(axis="y", color=COLORS["grid"], lw=0.6)

    write_json(
        RAW_OUT / "domain_balanced_falling_family_summary.json",
        {
            "concept": CONCEPT_LABELS[concept],
            "full_count": len(rows),
            "display_count": len(balanced),
            "selection_rule": "per-domain cap of 5 for display; full-bank counts retained for risk statement",
            "full_domain_counts": dict(sorted(full_domains.items(), key=lambda kv: (-kv[1], kv[0]))),
            "display_domain_counts": dict(ditems),
            "display_patch_position_counts": dict(pitems),
        },
    )
    return save_raw(fig, "fig_domain_balanced_falling_family.png")


def find_cluster(model: str, layer: str, cluster: int) -> dict[str, Any]:
    summary = read_json(ROOT / "outputs/second_pilot_discovery_summary.json")
    clusters = summary["models"][model]["layers"][layer]["domain_balanced"]["clusters"]
    for item in clusters:
        if int(item["cluster"]) == int(cluster):
            return item
    raise KeyError((model, layer, cluster))


def crop_panel_cells(path: Path, row: int, use_columns: list[int]) -> list[np.ndarray]:
    img = Image.open(path).convert("RGB")
    row_h = img.height / round(img.height / 243)
    col_w = img.width / 4
    cells = []
    for col in use_columns:
        left = int(col * col_w + 25)
        right = int((col + 1) * col_w - 25)
        top = int(row * row_h + 38)
        bottom = int((row + 1) * row_h - 10)
        cell = img.crop((left, top, right, bottom))
        draw = ImageDraw.Draw(cell)
        # Remove top/right debug-panel spines inherited from the source
        # prototype panel while keeping the curve itself intact.
        draw.rectangle((0, 0, cell.width, 7), fill="white")
        draw.rectangle((cell.width - 7, 0, cell.width, cell.height), fill="white")
        cells.append(np.asarray(cell))
    return cells


def extract_curve_from_cell(cell: np.ndarray) -> np.ndarray:
    """Digitize the blue prototype curve from an existing prototype-panel cell."""
    arr = np.asarray(cell)
    # Matplotlib tab:blue is much bluer than axes/spines/text. Keep only the
    # curve pixels and reconstruct one y-value per x-bin.
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    mask = (b > 105) & (g > 70) & (r < 80) & ((b - r) > 55)
    yy, xx = np.where(mask)
    if len(xx) < 8:
        return np.zeros(16, dtype=np.float64)
    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]
    bins = np.linspace(xx.min(), xx.max(), 33)
    values = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (xx >= lo) & (xx < hi)
        if not np.any(idx):
            values.append(np.nan)
        else:
            values.append(float(np.median(yy[idx])))
    y = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(y)
    grid = np.arange(len(y))
    if valid.sum() >= 2:
        y[~valid] = np.interp(grid[~valid], grid[valid], y[valid])
    elif valid.sum() == 1:
        y[~valid] = y[valid][0]
    else:
        y[:] = 0.0
    y = -y
    y = (y - y.mean()) / (y.std() + 1e-8)
    return y


def plot_curve_cell(ax: plt.Axes, curve: np.ndarray, color: str = "#2C7FB8") -> None:
    x = np.linspace(0, 1, len(curve))
    pad = max(0.35, 0.18 * float(np.ptp(curve)))
    ax.plot(x, curve, color=color, lw=1.9, solid_capstyle="round")
    ax.set_xlim(0, 1)
    ax.set_ylim(float(curve.min() - pad), float(curve.max() + pad))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d6e0e5")
    ax.spines["bottom"].set_color("#222222")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def fig_model_native_candidate_patches() -> Path:
    specs = [
        ("TimesFM-2.5", "timesfm_2_5", "layer_10", 5, "smooth transition-like", FIG / "second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_prototype_panel.png", 5, [1, 2, 3]),
        ("Chronos-2", "chronos_2", "layer_11", 9, "falling transition-like", FIG / "second_pilot/second_pilot_chronos_2_layer_11_domain_balanced_prototype_panel.png", 9, [0, 1, 2]),
        ("Chronos-2-small", "chronos_2_small", "layer_5", 13, "rising transition-like", FIG / "second_pilot/second_pilot_chronos_2_small_layer_5_domain_balanced_prototype_panel.png", 13, [0, 1, 2]),
    ]
    fig = plt.figure(figsize=(9.6, 4.8))
    gs = fig.add_gridspec(3, 4, left=0.045, right=0.99, bottom=0.09, top=0.96, hspace=0.36, wspace=0.12, width_ratios=[0.92, 1, 1, 1])
    summary = []
    for r, (display, model, layer, cluster_id, name, panel, row, cols) in enumerate(specs):
        cluster = find_cluster(model, layer, cluster_id)
        ax_text = fig.add_subplot(gs[r, 0])
        ax_text.axis("off")
        ax_text.text(0.0, 0.88, display, fontsize=12.0, fontweight="bold", color=COLORS["ink"], va="top")
        ax_text.text(0.0, 0.60, f"{display_label(layer)} · cluster {cluster_id}", fontsize=8.4, color=COLORS["muted"], va="top")
        ax_text.text(0.0, 0.39, name, fontsize=8.8, color=COLORS["ink"], va="top")
        ax_text.text(0.0, 0.17, f"n={cluster['size']}", fontsize=8.4, color=COLORS["muted"], va="top")
        for c, img in enumerate(crop_panel_cells(panel, row, cols), start=1):
            ax = fig.add_subplot(gs[r, c])
            plot_curve_cell(ax, extract_curve_from_cell(img))
        summary.append(
            {
                "model": display,
                "layer": display_label(layer),
                "cluster": cluster_id,
                "temporary_name": name,
                "size": cluster["size"],
                "top_domains": cluster["top_domains"],
                "top_taxonomy_labels": cluster["top_taxonomy_labels"],
                "top_patch_indices": cluster["top_patch_indices"],
            }
        )
    write_json(
        RAW_OUT / "model_native_candidate_patches_summary.json",
        {
            "purpose": "raw PPT figure source; model-native second-pilot candidates only",
            "candidates": summary,
        },
    )
    return save_raw(fig, "fig_model_native_candidate_patches_redrawn.png")


def fig_lineage_metric_bars() -> Path:
    summary = read_json(ROOT / "outputs/input_embedding_ablation/input_embedding_ablation_summary.json")
    reps_by_model = {
        "TimesFM-2.5": ("timesfm_2_5", ["raw_z_patch", "timesfm_tokenizer", "timesfm_hidden"]),
        "Chronos-2": ("chronos_2", ["raw_z_patch", "chronos_proj_with_time", "chronos_hidden"]),
        "Chronos-2-small": ("chronos_2_small", ["raw_z_patch", "chronos_proj_with_time", "chronos_hidden"]),
    }
    metrics = ["domain", "frequency", "taxonomy_v0", "patch_index"]
    metric_labels = ["domain", "frequency", "prior probe", "patch index"]
    colors = [COLORS["amber"], COLORS["teal"], COLORS["green"], COLORS["red"]]
    xlabels = ["raw", "tokenizer\n/projection", "hidden"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.55), sharey=True)
    for ax, (display, (key, reps)) in zip(axes, reps_by_model.items()):
        x = np.arange(len(reps))
        ax.axvspan(1.65, 2.35, color="#eef4f0", alpha=0.8, zorder=0)
        for mi, metric in enumerate(metrics):
            vals = [summary["models"][key]["representations"][rep]["nmi"][metric] for rep in reps]
            ax.plot(x, vals, color=colors[mi], lw=1.9, marker="o", ms=5.2, label=metric_labels[mi], zorder=3)
            ax.text(x[-1] + 0.055, vals[-1], f"{vals[-1]:.2f}", fontsize=7.5, color=colors[mi], va="center")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8.5)
        ax.set_title(display, loc="left", fontsize=11.0, fontweight="bold")
        ax.set_xlim(-0.18, 2.42)
        ax.set_ylim(0, 0.46)
        ax.grid(axis="y", color=COLORS["grid"], lw=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(2.00, 0.435, "contextualized", ha="center", va="center", fontsize=7.2, color=COLORS["muted"])
    axes[0].set_ylabel("NMI with cluster labels")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.0, ncol=4, loc="upper right", bbox_to_anchor=(0.98, 0.99))
    fig.text(0.02, 0.955, "H2", fontsize=12.8, fontweight="bold", color=COLORS["ink"])
    fig.text(0.065, 0.955, "Representation lineage fingerprints", fontsize=11.8, fontweight="bold", color=COLORS["ink"])
    fig.text(0.065, 0.035, "Hidden states reorganize local patch vocabulary, while also exposing architecture-specific confounders.", fontsize=8.0, color=COLORS["muted"])
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.78, wspace=0.20)
    return save_raw(fig, "fig_lineage_metric_story.png")


def fig_controlled_retrieval_heatmap() -> Path:
    taxonomy = read_json(ROOT / "outputs/taxonomy_v1_pilot/taxonomy_v1_pilot_summary.json")
    concept_sources = {
        "strong_rising_recovery": find_parent_subcluster(taxonomy, 8, 0),
        "strong_falling_transition": find_parent_subcluster(taxonomy, 5, 1),
        "smooth_falling_transition": find_parent_subcluster(taxonomy, 5, 0),
        "artifact_first_patch_behavior": find_cluster_validation(4),
    }
    conditions = ["unrestricted", "same_patch_index", "same_frequency", "cross_domain", "cross_domain_same_patch_index"]
    cond_labels = ["unrestricted", "same patch", "same frequency", "cross-domain", "cross-domain\nsame patch"]
    concepts = list(concept_sources.keys())
    shape = np.zeros((len(concepts), len(conditions)))
    for i, concept in enumerate(concepts):
        if concept == "artifact_first_patch_behavior":
            summary = concept_sources[concept]["condition_summary"]
        else:
            summary = concept_sources[concept]["validation"]["condition_summary"]
        for j, cond in enumerate(conditions):
            shape[i, j] = condition_metric(summary.get(cond, {"status": "empty"}), "mean_shape_correlation")

    fig, ax = plt.subplots(figsize=(9.8, 4.25))
    im = ax.imshow(shape, cmap="YlGnBu", vmin=0.0, vmax=0.9, aspect="auto")
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(cond_labels, fontsize=8.2)
    ax.set_yticks(np.arange(len(concepts)))
    ax.set_yticklabels([CONCEPT_LABELS[c] for c in concepts], fontsize=8.8)
    for i in range(shape.shape[0]):
        for j in range(shape.shape[1]):
            value = shape[i, j]
            color = "white" if (not np.isnan(value) and value >= 0.62) else COLORS["ink"]
            ax.text(
                j,
                i,
                "NA" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10.2,
                fontweight="bold",
                color=color,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.set_ylabel("shape correlation", rotation=270, labelpad=12)
    return save_raw(fig, "fig_controlled_retrieval_heatmap_readable.png")


def fig_cluster_outcome_gallery() -> Path:
    """Show that hidden-space clustering produces multiple outcome types."""
    specs = [
        (
            "Clean candidate",
            "transition-like",
            "TimesFM-2.5",
            "layer 10 · cluster 5",
            FIG / "second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_prototype_panel.png",
            5,
            [1, 2, 3],
            "timesfm_2_5",
            "layer_10",
            5,
        ),
        (
            "Clean candidate",
            "falling transition-like",
            "Chronos-2",
            "layer 11 · cluster 9",
            FIG / "second_pilot/second_pilot_chronos_2_layer_11_domain_balanced_prototype_panel.png",
            9,
            [0, 1, 2],
            "chronos_2",
            "layer_11",
            9,
        ),
        (
            "Broad mixed control",
            "flat representatives",
            "TimesFM-2.5",
            "layer 10 · cluster 2",
            FIG / "second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_prototype_panel.png",
            2,
            [0, 1, 2],
            "timesfm_2_5",
            "layer_10",
            2,
        ),
        (
            "Event-like control",
            "spike / volatility",
            "TimesFM-2.5",
            "layer 10 · cluster 0",
            FIG / "second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_prototype_panel.png",
            0,
            [0, 1, 2],
            "timesfm_2_5",
            "layer_10",
            0,
        ),
        (
            "Position-confounded cluster",
            "noisy first-patch mix",
            "TimesFM-2.5",
            "layer 10 · cluster 3",
            FIG / "second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_prototype_panel.png",
            3,
            [0, 1, 2],
            "timesfm_2_5",
            "layer_10",
            3,
        ),
        (
            "Confounded cluster",
            "first-patch behavior",
            "TimesFM-2.5",
            "layer 10 · cluster 4",
            FIG / "second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_prototype_panel.png",
            4,
            [0, 1, 2],
            "timesfm_2_5",
            "layer_10",
            4,
        ),
    ]
    fig = plt.figure(figsize=(10.2, 8.0))
    gs = fig.add_gridspec(
        len(specs),
        4,
        left=0.045,
        right=0.99,
        bottom=0.055,
        top=0.97,
        hspace=0.26,
        wspace=0.11,
        width_ratios=[1.06, 1, 1, 1],
    )
    summary = []
    for r, (role, name, display, layer_text, panel, row, cols, model, layer, cluster_id) in enumerate(specs):
        cluster = find_cluster(model, layer, cluster_id)
        ax_text = fig.add_subplot(gs[r, 0])
        ax_text.axis("off")
        ax_text.text(0.0, 0.88, role, fontsize=10.8, fontweight="bold", color=COLORS["ink"], va="top")
        ax_text.text(0.0, 0.62, name, fontsize=9.2, color=COLORS["ink"], va="top")
        ax_text.text(0.0, 0.38, display, fontsize=8.4, color=COLORS["muted"], va="top")
        ax_text.text(0.0, 0.18, f"{layer_text}; n={cluster['size']}", fontsize=7.8, color=COLORS["muted"], va="top")
        for c, img in enumerate(crop_panel_cells(panel, row, cols), start=1):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(COLORS["grid"])
                spine.set_linewidth(0.65)
        summary.append(
            {
                "role": role,
                "temporary_name": name,
                "model": display,
                "layer": layer_text,
                "cluster": cluster_id,
                "size": cluster["size"],
                "top_domains": cluster["top_domains"],
                "top_taxonomy_labels": cluster["top_taxonomy_labels"],
                "top_patch_indices": cluster["top_patch_indices"],
            }
        )
    write_json(
        RAW_OUT / "cluster_outcome_gallery_summary.json",
        {
            "purpose": "show non-cherry-picked cluster outcomes for advisor meeting",
            "note": "clusters are examples from domain-balanced second-pilot hidden-space clustering; labels are temporary descriptions",
            "clusters": summary,
        },
    )
    return save_raw(fig, "fig_cluster_outcome_gallery_v2.png")


def fig_negative_control_artifact() -> Path:
    grouped = load_prototype_bank()
    artifact = np.asarray([r["z_patch"] for r in grouped["artifact_first_patch_behavior"]], dtype=np.float64)
    cross = read_json(ROOT / "outputs/cross_model_validation/cross_model_validation_summary.json")
    model_rows = {r["display"]: r for r in cross["model_results"]}
    displays = ["TimesFM-2.5 layer_10", "Chronos-2 layer_11", "Chronos-2-small layer_5"]
    model_short = ["TimesFM", "Chronos-2", "Chronos-small"]

    fig = plt.figure(figsize=(9.8, 3.5))
    gs = fig.add_gridspec(1, 3, left=0.07, right=0.98, bottom=0.18, top=0.88, wspace=0.35, width_ratios=[1.25, 1.0, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 1, artifact.shape[1])
    mean = artifact.mean(axis=0)
    std = artifact.std(axis=0)
    ax.fill_between(x, mean - std, mean + std, color=COLORS["red"], alpha=0.15, lw=0)
    ax.plot(x, mean, color=COLORS["red"], lw=2.2)
    ax.axhline(0, color=COLORS["grid"], lw=0.8)
    ax.set_title("A  Position-bound prototype", loc="left", fontsize=11.5)
    ax.set_xlabel("normalized time")
    ax.set_ylabel("z-normalized value")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)

    ax2 = fig.add_subplot(gs[0, 1])
    vals = [model_rows[d]["global_retrieval"]["artifact_first_patch_behavior"]["patch_index_diversity"] for d in displays]
    ax2.barh(range(len(vals)), vals, color=[COLORS["red"], COLORS["gray"], COLORS["gray"]])
    ax2.set_yticks(range(len(vals)))
    ax2.set_yticklabels(model_short, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("patch-index diversity")
    ax2.set_title("B  Position audit", loc="left", fontsize=11.5)
    ax2.grid(axis="x", color=COLORS["grid"], lw=0.6)

    ax3 = fig.add_subplot(gs[0, 2])
    hit = [model_rows[d]["global_retrieval"]["artifact_first_patch_behavior"]["same_concept_hit_rate"] for d in displays]
    base = [model_rows[d]["matched_random_baseline"]["prototype_hit_rate"] for d in displays]
    xs = np.arange(len(hit))
    ax3.bar(xs - 0.16, hit, width=0.32, color=COLORS["red"], label="artifact hit")
    ax3.bar(xs + 0.16, base, width=0.32, color=COLORS["gray"], label="matched random")
    ax3.set_xticks(xs)
    ax3.set_xticklabels(model_short, rotation=20, ha="right", fontsize=8.5)
    ax3.set_ylabel("hit rate")
    ax3.set_title("C  Retrieval sanity", loc="left", fontsize=11.5)
    ax3.legend(frameon=False, fontsize=7.8)
    ax3.grid(axis="y", color=COLORS["grid"], lw=0.6)
    return save_raw(fig, "fig_negative_control_artifact.png")


def fig_cross_model_validation_bars() -> Path:
    summary = read_json(ROOT / "outputs/cross_model_validation/cross_model_validation_summary.json")
    model_displays = [row["display"] for row in summary["model_results"]]
    model_short = ["TimesFM", "Chronos-2", "Chronos-small"]
    concepts = ["strong_falling_transition", "smooth_falling_transition", "strong_rising_recovery", "artifact_first_patch_behavior"]
    values = np.zeros((len(concepts), len(model_displays)))
    baselines = []
    for j, row in enumerate(summary["model_results"]):
        baselines.append(row["matched_random_baseline"]["mean_shape_correlation"])
        for i, concept in enumerate(concepts):
            values[i, j] = row["global_retrieval"][concept]["mean_shape_correlation"]

    fig = plt.figure(figsize=(10.6, 3.75))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.96, bottom=0.22, top=0.80, wspace=0.34, width_ratios=[1.35, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(model_displays))
    ax.fill_between([-0.35, len(x) - 0.65], [0.0, 0.0], [0.52, 0.52], color="#f2f5f4", zorder=0)
    ax.plot(x, baselines, color=COLORS["ink"], marker="o", ms=5.0, lw=1.3, zorder=3)
    for i, concept in enumerate(concepts):
        ax.plot(x, values[i], color=CONCEPT_COLORS[concept], marker="o", ms=5.6, lw=1.8, zorder=4)
        if concept == "strong_falling_transition":
            for xi, yi, base in zip(x, values[i], baselines):
                ax.vlines(xi, base, yi, color=CONCEPT_COLORS[concept], lw=2.0, alpha=0.45, zorder=2)
                ax.text(xi + 0.035, yi + 0.025, f"{yi-base:+.2f}", fontsize=7.5, color=CONCEPT_COLORS[concept], fontweight="bold")
    label_offsets = {
        "strong_falling_transition": 0.020,
        "strong_rising_recovery": 0.020,
        "smooth_falling_transition": -0.018,
        "artifact_first_patch_behavior": -0.010,
    }
    for i, concept in enumerate(concepts):
        ax.text(2.06, values[i, -1] + label_offsets[concept], CONCEPT_LABELS[concept], color=CONCEPT_COLORS[concept], fontsize=7.4, va="center")
    ax.text(2.06, baselines[-1] - 0.018, "matched random", color=COLORS["ink"], fontsize=7.4, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(model_short, fontsize=8.8)
    ax.set_xlim(-0.48, 2.46)
    ax.set_ylabel("mean shape correlation")
    ax.set_ylim(0, 0.96)
    ax.set_title("A  Prototype retrieval transfer", loc="left", fontsize=11.4, fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    sil = [row["prototype_space"]["silhouette_by_concept"] for row in summary["model_results"]]
    nn = [row["prototype_space"]["mean_1nn_same_concept"] for row in summary["model_results"]]
    yy = np.arange(len(model_short))
    ax2.hlines(yy, sil, nn, color=COLORS["grid"], lw=4.0, zorder=1)
    ax2.scatter(sil, yy, s=60, color=COLORS["blue"], label="silhouette", zorder=3)
    ax2.scatter(nn, yy, s=70, color=COLORS["green"], label="1-NN same concept", zorder=3)
    for yi, s, n in zip(yy, sil, nn):
        ax2.text(s + 0.025, yi - 0.10, f"{s:.2f}", fontsize=7.0, color=COLORS["blue"])
        n_x = n - 0.035 if n > 0.93 else n + 0.025
        n_ha = "right" if n > 0.93 else "left"
        n_y = yi - 0.15 if yi == yy[-1] else yi + 0.16
        ax2.text(n_x, n_y, f"{n:.2f}", fontsize=7.0, color=COLORS["green"], ha=n_ha)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(model_short, fontsize=8.8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.02)
    ax2.set_xlabel("agreement score", fontsize=8.5)
    ax2.set_title("B  Prototype-space agreement", loc="left", fontsize=11.4, fontweight="bold")
    ax2.grid(axis="x", color=COLORS["grid"], lw=0.65)
    ax2.text(0.02, -0.35, "blue: silhouette; green: 1-NN same concept", transform=ax2.transAxes, fontsize=7.4, color=COLORS["muted"], va="top")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig.text(0.02, 0.955, "H3", fontsize=12.8, fontweight="bold", color=COLORS["ink"])
    fig.text(0.070, 0.955, "Cross-model validation: candidate families transfer unevenly across architectures", fontsize=11.5, fontweight="bold", color=COLORS["ink"])
    fig.text(0.070, 0.055, "Strong falling transition is the most stable positive candidate; first-patch behavior remains a negative control rather than a motif family.", fontsize=8.0, color=COLORS["muted"])
    return save_raw(fig, "fig_cross_model_validation_story.png")


def main() -> None:
    setup_matplotlib()
    paths = [
        fig_hypothesis_evidence_map(),
        fig_evidence_gap_audit(),
        fig_discovery_protocol_flow(),
        fig_prior_probe_literature_map(),
        fig_prior_probe_shapes_and_limits(),
        fig_prior_probe_shapes_only(),
        fig_prior_probe_alignment_only(),
        fig_prior_probe_distribution(),
        fig_timesfm_clustering_triptych(),
        fig_domain_balanced_falling_family(),
        fig_model_native_candidate_patches(),
        fig_cluster_outcome_gallery(),
        fig_lineage_metric_bars(),
        fig_controlled_retrieval_heatmap(),
        fig_negative_control_artifact(),
        fig_cross_model_validation_bars(),
    ]
    manifest = {
        "purpose": "raw images for manual PPT layout; no full-slide text blocks",
        "paths": [str(p.relative_to(ROOT)) for p in paths],
    }
    write_json(RAW_OUT / "raw_asset_manifest.json", manifest)
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
