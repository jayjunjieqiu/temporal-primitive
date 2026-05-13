#!/usr/bin/env python3
"""Build publication-style assets for the advisor meeting PPT draft."""

from __future__ import annotations

import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ppt_assets"
FIG = ROOT / "outputs" / "figures"

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

COLORS = {
    "ink": "#1b2227",
    "muted": "#5a6770",
    "light": "#eff3f5",
    "grid": "#dce4e8",
    "blue": "#2f6f9f",
    "teal": "#1f8a7a",
    "green": "#3a7a38",
    "amber": "#b36b18",
    "red": "#b34545",
    "violet": "#6d5aa7",
    "gray": "#7a858c",
    "paper": "#ffffff",
}

CONCEPT_COLORS = {
    "strong_rising_recovery": COLORS["green"],
    "strong_falling_transition": COLORS["blue"],
    "smooth_falling_transition": COLORS["teal"],
    "artifact_first_patch_behavior": COLORS["red"],
}

CONCEPT_LABELS = {
    "strong_rising_recovery": "strong rising / recovery",
    "strong_falling_transition": "strong falling transition",
    "smooth_falling_transition": "smooth falling transition",
    "artifact_first_patch_behavior": "first-patch artifact",
}

DETECTOR_LABELS = {
    "deterministic_statistics": "deterministic statistics",
    "linear_fit": "linear fit",
    "spectral_concentration": "spectral concentration",
    "robust_outlier_count": "robust outlier count",
    "active_run_statistics": "active run statistics",
    "change_point_mean": "change-point mean",
    "change_point_variance": "change-point variance",
    "sparse_run_statistics": "sparse run statistics",
    "conflict_and_margin_policy": "conflict and margin policy",
}


def display_label(text: object) -> str:
    return str(text).replace("_", " ")


def setup_matplotlib() -> None:
    # Register TTC fonts explicitly; matplotlib's cache may not discover CJK
    # collections in a minimal runtime environment.
    font_manager.fontManager.addfont(FONT_REG)
    font_manager.fontManager.addfont(FONT_BOLD)
    cjk_name = font_manager.FontProperties(fname=FONT_REG).get_name()
    mpl.rcParams.update(
        {
            "font.family": cjk_name,
            "font.sans-serif": [cjk_name, "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.9,
            "axes.labelcolor": COLORS["ink"],
            "axes.titleweight": "bold",
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size=size)


def wrap(text: str, width: int = 20) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def save_fig(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    return path


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def z_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def taxonomy_prototype(name: str, n: int = 64) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    if name == "flat_low_information":
        return np.zeros(n)
    if name == "trend":
        return z_norm(t)
    if name == "oscillation":
        return z_norm(np.sin(2 * math.pi * 2.5 * t))
    if name == "impulse_spike":
        x = np.zeros(n)
        x[n // 2] = 4.0
        return z_norm(x)
    if name == "burst_event":
        x = np.zeros(n)
        start = n // 2 - 7
        burst = np.hanning(14) * 2.4
        x[start : start + len(burst)] = burst
        return z_norm(x)
    if name == "level_shift":
        x = np.r_[np.zeros(n // 2), np.ones(n - n // 2) * 1.8]
        return z_norm(x)
    if name == "volatility_shift":
        x = np.r_[0.12 * np.sin(np.linspace(0, 4 * math.pi, n // 2)), 0.9 * np.sin(np.linspace(0, 8 * math.pi, n - n // 2))]
        return z_norm(x)
    if name == "intermittent":
        x = np.zeros(n)
        x[[10, 23, 39, 52]] = [1.3, 1.8, 1.5, 1.9]
        return z_norm(x)
    if name == "mixed_uncertain":
        x = 0.8 * (t - 0.5) + 1.1 * np.sin(2 * math.pi * 1.3 * t)
        x[n // 2 + 4] += 2.0
        return z_norm(x)
    raise KeyError(name)


def draw_slide_header(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.035, 0.962, title, ha="left", va="top", fontsize=18, fontweight="bold", color=COLORS["ink"])
    if subtitle:
        fig.text(0.035, 0.925, subtitle, ha="left", va="top", fontsize=10.5, color=COLORS["muted"])


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )


def compact_count_label(items: list[dict[str, Any]], limit: int = 3) -> str:
    return ", ".join(f"{display_label(x['value'])}: {x['count']}" for x in items[:limit])


def domain_balanced_rows(rows: list[dict[str, Any]], max_per_domain: int = 5) -> list[dict[str, Any]]:
    """Select a display subset with an explicit per-domain cap."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["domain"])].append(row)
    selected: list[dict[str, Any]] = []
    for domain in sorted(grouped):
        domain_rows = grouped[domain]
        selected.extend(domain_rows[: min(max_per_domain, len(domain_rows))])
    return selected


def build_slide01_story_question() -> Path:
    fig = plt.figure(figsize=(13.3, 7.5))
    draw_slide_header(
        fig,
        "What Is the Temporal Language of TSFMs?",
        "From prior-guided motif probes to model-derived candidate motif families",
    )
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.70])
    ax.axis("off")

    xs = [0.08, 0.28, 0.50, 0.72, 0.90]
    labels = [
        ("heterogeneous\nraw series", "cross-domain data"),
        ("patch token", "tokenization / projection"),
        ("hidden state", "contextualized representation"),
        ("candidate clusters", "discover first"),
        ("candidate motif families", "name after audit"),
    ]
    colors = [COLORS["gray"], COLORS["teal"], COLORS["blue"], COLORS["amber"], COLORS["green"]]
    for i, (x, (main, sub)) in enumerate(zip(xs, labels)):
        ax.scatter([x], [0.58], s=1700, color=colors[i], alpha=0.12, edgecolors=colors[i], linewidths=1.4)
        ax.scatter([x], [0.58], s=240, color=colors[i], alpha=0.98)
        ax.text(x, 0.43, main, ha="center", va="top", fontsize=13, fontweight="bold", color=COLORS["ink"])
        ax.text(x, 0.31, sub, ha="center", va="top", fontsize=10, color=COLORS["muted"])
        if i < len(xs) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.055, 0.58),
                xytext=(x + 0.055, 0.58),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["muted"], lw=1.5),
            )
    ax.text(
        0.50,
        0.11,
        "The target is not forecasting score, but patch-token representations that may carry shared temporal knowledge.",
        ha="center",
        fontsize=13,
        color=COLORS["ink"],
    )
    return save_fig(fig, "ppt_slide01_story_question.png")


def build_slide02_gap_and_question() -> Path:
    fig = plt.figure(figsize=(13.3, 7.5))
    draw_slide_header(
        fig,
        "TSFM / STFM Gap: Performance Works, Token-Level Mechanisms Remain Unclear",
        "We need auditable patch-level motif/prototype families for shared representations.",
    )
    ax = fig.add_axes([0.06, 0.13, 0.88, 0.70])
    ax.axis("off")
    columns = [
        ("Motivation", "cross-domain generalization\nOOD transfer\nheterogeneous data"),
        ("Observed", "frozen TSFM hidden space\nhas non-random neighborhoods"),
        ("Unknown", "domain identity?\nfrequency / position?\nshared temporal motifs?"),
        ("Our angle", "model-derived motif taxonomy\ndiscovery protocol"),
    ]
    x0, gap, w, h = 0.02, 0.045, 0.215, 0.55
    for i, (title, body) in enumerate(columns):
        x = x0 + i * (w + gap)
        ax.add_patch(plt.Rectangle((x, 0.27), w, h, fill=False, ec=COLORS["grid"], lw=1.2))
        ax.text(x + 0.02, 0.76, title, ha="left", va="top", fontsize=13, fontweight="bold", color=COLORS["ink"])
        ax.text(x + 0.02, 0.66, body, ha="left", va="top", fontsize=12, color=COLORS["muted"], linespacing=1.7)
    ax.annotate("", xy=(0.91, 0.18), xytext=(0.09, 0.18), arrowprops=dict(arrowstyle="-|>", lw=1.6, color=COLORS["ink"]))
    ax.text(0.50, 0.08, "Question: what do patch tokens learn?", ha="center", fontsize=17, fontweight="bold", color=COLORS["ink"])
    return save_fig(fig, "ppt_slide02_gap_and_question.png")


def build_slide03_research_hypotheses() -> Path:
    fig = plt.figure(figsize=(13.3, 7.5))
    draw_slide_header(
        fig,
        "Three Research Hypotheses",
        "Each hypothesis maps to one evidence block in the meeting deck.",
    )
    ax = fig.add_axes([0.055, 0.14, 0.89, 0.70])
    ax.axis("off")

    cards = [
        (
            "H1",
            "Cross-domain temporal primitives",
            "If TSFMs learn shared temporal knowledge, patch-token space should contain motif/prototype families reused across heterogeneous domains.",
            "Evidence: clustering + original-space prototypes + controlled retrieval",
            COLORS["green"],
        ),
        (
            "H2",
            "Contextual reorganization",
            "Tokenizer/projection states should behave like local patch vocabulary, while hidden states may merge local patterns into contextualized motif families.",
            "Evidence: raw -> tokenizer/projection -> hidden lineage",
            COLORS["blue"],
        ),
        (
            "H3",
            "Scale, architecture, and confounders",
            "Motif organization should vary across Chronos-2-small, Chronos-2, and TimesFM-2.5, and may be entangled with domain, frequency, or position shortcuts.",
            "Evidence: cross-model validation + confounder audit",
            COLORS["amber"],
        ),
    ]

    for i, (hid, title, body, evidence, color) in enumerate(cards):
        y = 0.70 - i * 0.29
        ax.add_patch(plt.Rectangle((0.00, y), 1.00, 0.22, fill=False, ec=COLORS["grid"], lw=1.2))
        ax.add_patch(plt.Rectangle((0.00, y), 0.105, 0.22, color=color, alpha=0.12, ec=color, lw=1.2))
        ax.text(0.052, y + 0.11, hid, ha="center", va="center", fontsize=22, fontweight="bold", color=color)
        ax.text(0.14, y + 0.165, title, ha="left", va="top", fontsize=14, fontweight="bold", color=COLORS["ink"])
        ax.text(0.14, y + 0.105, body, ha="left", va="top", fontsize=10.8, color=COLORS["muted"], wrap=True)
        ax.text(0.14, y + 0.030, evidence, ha="left", va="bottom", fontsize=10.2, color=color, fontweight="bold")

    return save_fig(fig, "ppt_slide03_research_hypotheses.png")


def build_slide03_taxonomy_v0_math() -> Path:
    cfg = read_yaml(ROOT / "configs/motif_taxonomy_v0.yaml")
    label_to_cfg = {c["name"]: c for c in cfg["classes"]}
    order = [
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
    formulas = {
        "flat_low_information": "std(x)<0.08; range(x)<0.30",
        "trend": "|β1|≥0.75; R²≥0.70",
        "oscillation": "max Pk/ΣPk≥0.42; zero-crossing≥3",
        "impulse_spike": "max|z|≥4; count(|z|>3)≤2",
        "burst_event": "0.15≤ρa≤0.60; longest run/n≥0.12",
        "level_shift": "maxτ |μR-μL|/pooled σ ≥1.45",
        "volatility_shift": "maxτ σ ratio≥2.2; mean-change≤1.25",
        "intermittent": "# active runs≥2; 0.08≤ρa≤0.40",
        "mixed_uncertain": "detector conflict or low confidence",
    }
    roots = {
        "flat_low_information": "control label",
        "trend": "linear fit / shapelet-like segment",
        "oscillation": "FFT, SAX/PAA symbolic patterns",
        "impulse_spike": "robust outlier / anomaly events",
        "burst_event": "event-run statistics",
        "level_shift": "change-point segmentation",
        "volatility_shift": "variance change-point",
        "intermittent": "sparse event sequence",
        "mixed_uncertain": "confidence policy",
    }
    colors = [
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

    fig = plt.figure(figsize=(13.3, 7.5))
    draw_slide_header(
        fig,
        "Prior-Guided Motif Probe: Operational Definitions",
        "These are deterministic weak labels, not ground truth; they anchor interpretation of model-derived clusters.",
    )

    ax_note = fig.add_axes([0.04, 0.78, 0.92, 0.10])
    ax_note.axis("off")
    ax_note.text(0.00, 0.58, "Common notation", fontsize=11, fontweight="bold", color=COLORS["ink"])
    ax_note.text(
        0.18,
        0.58,
        "z=(x-median(x))/(1.4826 MAD(x)+ε), active={|z|>2}, τ scans middle splits.",
        fontsize=11,
        color=COLORS["muted"],
    )
    ax_note.text(0.00, 0.12, "Literature roots", fontsize=11, fontweight="bold", color=COLORS["ink"])
    ax_note.text(
        0.18,
        0.12,
        "Matrix Profile/motif discovery, shapelets, SAX/PAA, change-point detection, anomaly/event-run statistics.",
        fontsize=11,
        color=COLORS["muted"],
    )

    for i, name in enumerate(order):
        row = i // 3
        col = i % 3
        card_x = 0.045 + col * 0.315
        card_y = 0.105 + (2 - row) * 0.205
        card_w = 0.285
        card_h = 0.165
        fig.patches.append(
            plt.Rectangle(
                (card_x, card_y),
                card_w,
                card_h,
                transform=fig.transFigure,
                fill=False,
                ec=COLORS["grid"],
                lw=0.8,
            )
        )
        fig.text(card_x + 0.008, card_y + card_h - 0.030, display_label(name), fontsize=11.0, fontweight="bold", color=COLORS["ink"])
        fig.text(card_x + 0.008, card_y + card_h - 0.055, roots[name], fontsize=7.8, color=COLORS["muted"])
        fig.text(card_x + 0.008, card_y + 0.012, formulas[name], fontsize=8.0, color=COLORS["ink"])
        detector = label_to_cfg[name]["primary_detector"] if name in label_to_cfg else ""
        fig.text(card_x + card_w - 0.008, card_y + 0.012, DETECTOR_LABELS.get(detector, display_label(detector)), fontsize=8.0, color=COLORS["muted"], ha="right")

        ax = fig.add_axes([card_x + 0.018, card_y + 0.050, card_w - 0.036, 0.065])
        x = taxonomy_prototype(name)
        t = np.linspace(0, 1, len(x))
        ax.plot(t, x, lw=2.0, color=colors[i])
        ax.axhline(0, lw=0.7, color=COLORS["grid"])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    return save_fig(fig, "ppt_slide04_prior_motif_probe_math.png")


def build_backup_taxonomy_table() -> Path:
    cfg = read_yaml(ROOT / "configs/motif_taxonomy_v0.yaml")
    rows = []
    for cls in cfg["classes"]:
        thresholds = cls.get("default_thresholds", {})
        threshold_text = "; ".join(f"{display_label(k)}={v}" for k, v in thresholds.items())
        robust = "/".join(map(str, cls.get("robust_at_patch_lengths", [])))
        partial = "/".join(map(str, cls.get("partially_robust_at_patch_lengths", [])))
        patch_note = f"robust: {robust}" + (f"; partial: {partial}" if partial else "")
        rows.append((display_label(cls["name"]), DETECTOR_LABELS.get(cls["primary_detector"], display_label(cls["primary_detector"])), threshold_text, patch_note))

    fig = plt.figure(figsize=(13.3, 7.5))
    draw_slide_header(
        fig,
        "Backup: Full Operational Definition of the Prior-Guided Motif Probe",
        "Thresholds come from synthetic calibration; real-data use is limited to weak probing and prototype-bank seeding.",
    )
    ax = fig.add_axes([0.035, 0.08, 0.93, 0.80])
    ax.axis("off")
    col_x = [0.00, 0.24, 0.45, 0.83]
    headers = ["class", "detector", "default thresholds", "patch length"]
    for x, h in zip(col_x, headers):
        ax.text(x, 0.98, h, fontsize=10.5, fontweight="bold", color=COLORS["ink"], va="top", transform=ax.transAxes)
    ax.plot([0, 1], [0.94, 0.94], transform=ax.transAxes, color=COLORS["ink"], lw=1.0)
    y = 0.90
    for name, detector, threshold, patch_note in rows:
        ax.text(col_x[0], y, name, fontsize=9.4, color=COLORS["ink"], va="top", transform=ax.transAxes)
        ax.text(col_x[1], y, detector, fontsize=8.8, color=COLORS["muted"], va="top", transform=ax.transAxes)
        ax.text(col_x[2], y, wrap(threshold, 48), fontsize=8.3, color=COLORS["ink"], va="top", transform=ax.transAxes)
        ax.text(col_x[3], y, patch_note, fontsize=8.3, color=COLORS["muted"], va="top", transform=ax.transAxes)
        y -= 0.095
        ax.plot([0, 1], [y + 0.018, y + 0.018], transform=ax.transAxes, color=COLORS["grid"], lw=0.6)
    return save_fig(fig, "ppt_backup_prior_probe_full_math.png")


def build_slide04_discovery_protocol() -> Path:
    fig = plt.figure(figsize=(13.3, 7.5))
    draw_slide_header(
        fig,
        "Discover First, Name Second",
        "A model-derived cluster becomes a candidate motif family only after original-space inspection and confounder audit.",
    )
    ax = fig.add_axes([0.05, 0.12, 0.90, 0.72])
    ax.axis("off")
    steps = [
        ("1", "Patch bank", "BasicTS non-BLAST\ncross-domain windows"),
        ("2", "Representation", "raw / tokenizer / hidden\nChronos-2, Chronos-2-small, TimesFM-2.5"),
        ("3", "Candidate generation", "PCA + KMeans\nmulti-K / stability as diagnostic"),
        ("4", "Original-space inspection", "prototype curves\nmedoids / nearest neighbors"),
        ("5", "Confounder audit", "domain / frequency / position\nscale / zero-ratio"),
        ("6", "Controlled retrieval", "same-position / same-frequency\ncross-domain / cross-model"),
    ]
    xs = np.linspace(0.08, 0.92, 6)
    y = 0.56
    for i, (num, title, desc) in enumerate(steps):
        ax.scatter(xs[i], y, s=1150, color=COLORS["light"], edgecolor=COLORS["ink"], linewidth=1.1)
        ax.text(xs[i], y + 0.01, num, ha="center", va="center", fontsize=17, fontweight="bold", color=COLORS["ink"])
        ax.text(xs[i], y - 0.16, title, ha="center", va="top", fontsize=11.5, fontweight="bold", color=COLORS["ink"])
        ax.text(xs[i], y - 0.24, desc, ha="center", va="top", fontsize=8.8, color=COLORS["muted"], linespacing=1.4)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.05, y), xytext=(xs[i] + 0.05, y), arrowprops=dict(arrowstyle="-|>", color=COLORS["muted"], lw=1.4))
    ax.text(0.50, 0.13, "KMeans label ≠ motif label", ha="center", fontsize=18, fontweight="bold", color=COLORS["red"])
    ax.text(0.50, 0.06, "naming rule: shape coherence + retrieval survival + low confounding", ha="center", fontsize=12, color=COLORS["muted"])
    return save_fig(fig, "ppt_slide05_discovery_protocol.png")


def trim_image(path: Path, crop_top: int = 55, crop_bottom: int = 25, crop_left: int = 35, crop_right: int = 25) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.crop((crop_left, crop_top, img.width - crop_right, img.height - crop_bottom))


def build_slide06_clustering_evidence() -> Path:
    sources = [
        (
            "A",
            "candidate clusters",
            FIG / "second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_pca_clusters.png",
            "KMeans gives candidate neighborhoods",
        ),
        (
            "B",
            "colored by prior motif probe",
            FIG / "second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_pca_taxonomy_v0.png",
            "not a one-to-one mapping",
        ),
        (
            "C",
            "colored by patch index",
            FIG / "second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_pca_patch_index.png",
            "position confounding appears",
        ),
    ]
    width, height = 3600, 1660
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((120, 78), "TimesFM-2.5 layer 10: Hidden Space Has Structure, but Not a Direct Copy of the Prior Probe", font=font(60, True), fill=COLORS["ink"])
    draw.text((120, 155), "domain-balanced second pilot; PCA only for visualization; KMeans only for candidate generation", font=font(34), fill=COLORS["muted"])
    panel_w, panel_h = 1040, 930
    gap = 90
    for i, (letter, title, path, caption) in enumerate(sources):
        img = trim_image(path, crop_top=70, crop_bottom=55, crop_left=45, crop_right=245)
        img.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
        x = 120 + i * (panel_w + gap)
        y = 330
        draw.text((x, y - 90), letter, font=font(62, True), fill=COLORS["ink"])
        draw.text((x + 78, y - 78), title, font=font(39, True), fill=COLORS["ink"])
        draw.rectangle((x, y, x + panel_w, y + panel_h), outline=COLORS["grid"], width=4)
        canvas.paste(img, (x + (panel_w - img.width) // 2, y + (panel_h - img.height) // 2))
    out = OUT / "ppt_slide06_timesfm_clustering_evidence.png"
    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def load_prototype_bank() -> dict[str, list[dict[str, Any]]]:
    rows = read_json(ROOT / "outputs/cross_model_validation/prototype_bank.json")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["concept"]].append(row)
    return grouped


def concept_matrix(grouped: dict[str, list[dict[str, Any]]], concept: str) -> np.ndarray:
    return np.asarray([row["z_patch"] for row in grouped[concept]], dtype=np.float64)


def draw_patch_examples(ax: plt.Axes, patches: np.ndarray, color: str, max_n: int = 5) -> None:
    n = min(max_n, len(patches))
    for i in range(n):
        y = patches[i]
        ax.plot(np.linspace(0, 1, len(y)), y + i * 4.0, lw=1.1, color=color, alpha=0.90)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(-2.8, max(2.8, (n - 1) * 4 + 2.8))
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
        spine.set_linewidth(0.8)


def build_slide07_original_space_concepts() -> Path:
    grouped = load_prototype_bank()
    concept = "strong_falling_transition"
    rows = grouped[concept]
    balanced_rows = domain_balanced_rows(rows, max_per_domain=5)
    mat = np.asarray([row["z_patch"] for row in rows], dtype=np.float64)
    balanced_mat = np.asarray([row["z_patch"] for row in balanced_rows], dtype=np.float64)
    mean = mat.mean(axis=0)
    balanced_mean = balanced_mat.mean(axis=0)
    corr = np.asarray([np.corrcoef(row, balanced_mean)[0, 1] for row in balanced_mat])
    order = np.argsort(-corr)
    top = balanced_mat[order[:12]]
    top_corr = corr[order[:12]]
    domain_counts: dict[str, int] = defaultdict(int)
    balanced_domain_counts: dict[str, int] = defaultdict(int)
    patch_counts: dict[str, int] = defaultdict(int)
    taxonomy_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        domain_counts[row["domain"]] += 1
    for row in balanced_rows:
        balanced_domain_counts[row["domain"]] += 1
        patch_counts[f"p{row['timesfm_patch_index']}"] += 1
        taxonomy_counts[row["taxonomy_v0"]] += 1

    fig = plt.figure(figsize=(13.3, 7.5))
    draw_slide_header(
        fig,
        "Domain-Balanced Original-Space Evidence: The Cleanest Candidate Family",
        "Display subset uses a per-domain cap; the full source bank is still reported as a confounder risk.",
    )
    gs = fig.add_gridspec(2, 4, left=0.055, right=0.96, bottom=0.12, top=0.79, hspace=0.35, wspace=0.30, width_ratios=[1.25, 1, 1, 1])

    ax_mean = fig.add_subplot(gs[:, 0])
    x = np.linspace(0, 1, mat.shape[1])
    std = mat.std(axis=0)
    bstd = balanced_mat.std(axis=0)
    ax_mean.fill_between(x, balanced_mean - bstd, balanced_mean + bstd, color=COLORS["blue"], alpha=0.12, lw=0, label="balanced subset ±1 SD")
    ax_mean.plot(x, balanced_mean, color=COLORS["blue"], lw=2.8, label="balanced mean")
    ax_mean.plot(x, mean, color=COLORS["gray"], lw=1.4, ls="--", label="full-bank mean")
    ax_mean.axhline(0, color=COLORS["grid"], lw=0.8)
    ax_mean.set_title("A  strong falling transition prototype", loc="left")
    ax_mean.set_xlabel("normalized time within patch")
    ax_mean.set_ylabel("z-normalized value")
    ax_mean.grid(axis="y", color=COLORS["grid"], lw=0.6)
    ax_mean.legend(frameon=False, fontsize=8, loc="lower left")
    ax_mean.text(0.03, 0.96, f"display n={len(balanced_mat)}; full n={len(mat)}", transform=ax_mean.transAxes, fontsize=8.8, color=COLORS["muted"], va="top")

    sub_gs = gs[:, 1:3].subgridspec(3, 4, hspace=0.16, wspace=0.12)
    y_min = float(np.min(top)) - 0.3
    y_max = float(np.max(top)) + 0.3
    for idx in range(12):
        ax = fig.add_subplot(sub_gs[idx // 4, idx % 4])
        y = top[idx]
        ax.plot(x, y, color=COLORS["blue"], lw=1.5)
        ax.axhline(0, color=COLORS["grid"], lw=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.set_title("B  most representative patches", loc="left")
        ax.text(0.04, 0.90, f"r={top_corr[idx]:.2f}", transform=ax.transAxes, fontsize=8.8, color=COLORS["muted"], va="top")
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
            spine.set_linewidth(0.7)

    ax_meta = fig.add_subplot(gs[0, 3])
    domains = sorted(balanced_domain_counts.items(), key=lambda kv: -kv[1])[:5]
    ax_meta.barh(np.arange(len(domains)), [v for _, v in domains], color=COLORS["blue"], alpha=0.86)
    ax_meta.set_yticks(np.arange(len(domains)))
    ax_meta.set_yticklabels([k for k, _ in domains])
    ax_meta.invert_yaxis()
    ax_meta.set_title("C  balanced source domains", loc="left")
    ax_meta.set_xlabel("count")
    ax_meta.grid(axis="x", color=COLORS["grid"], lw=0.6)

    ax_pos = fig.add_subplot(gs[1, 3])
    patches = sorted(patch_counts.items())
    ax_pos.bar([k for k, _ in patches], [v for _, v in patches], color=COLORS["teal"], alpha=0.90, label="patch index")
    ax_pos.set_title("D  position spread", loc="left")
    ax_pos.set_ylabel("count")
    ax_pos.grid(axis="y", color=COLORS["grid"], lw=0.6)
    write_json(
        OUT / "domain_balanced_prototype_bank_summary.json",
        {
            "concept": CONCEPT_LABELS[concept],
            "selection_rule": "per-domain cap for PPT display subset; full-bank counts retained for confounder audit",
            "max_per_domain": 5,
            "full_count": len(rows),
            "display_count": len(balanced_rows),
            "full_domain_counts": dict(sorted(domain_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "display_domain_counts": dict(sorted(balanced_domain_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "display_patch_index_counts": dict(sorted(patch_counts.items())),
        },
    )

    return save_fig(fig, "ppt_slide07_representative_falling_evidence.png")


def build_backup_candidate_prototype_quality() -> Path:
    grouped = load_prototype_bank()
    concepts = ["strong_falling_transition", "smooth_falling_transition", "strong_rising_recovery", "artifact_first_patch_behavior"]
    fig = plt.figure(figsize=(13.3, 7.5))
    draw_slide_header(
        fig,
        "Backup: Prototype Quality Across Candidate and Control Families",
        "Patches are sorted by correlation to each family mean; this is for transparency, not the main claim.",
    )
    gs = fig.add_gridspec(4, 6, left=0.055, right=0.96, bottom=0.10, top=0.82, hspace=0.42, wspace=0.18)
    for row, concept in enumerate(concepts):
        mat = concept_matrix(grouped, concept)
        mean = mat.mean(axis=0)
        corr = np.asarray([np.corrcoef(p, mean)[0, 1] for p in mat])
        order = np.argsort(-corr)[:5]
        color = CONCEPT_COLORS[concept]
        ax_mean = fig.add_subplot(gs[row, 0])
        x = np.linspace(0, 1, mat.shape[1])
        ax_mean.plot(x, mean, color=color, lw=2.0)
        ax_mean.axhline(0, color=COLORS["grid"], lw=0.6)
        ax_mean.set_xticks([])
        ax_mean.set_yticks([])
        ax_mean.set_ylabel(CONCEPT_LABELS[concept], rotation=0, ha="right", va="center", labelpad=56, fontsize=8.5)
        if row == 0:
            ax_mean.set_title("mean", fontsize=9)
        for col, idx in enumerate(order, start=1):
            ax = fig.add_subplot(gs[row, col])
            ax.plot(x, mat[idx], color=color, lw=1.4)
            ax.axhline(0, color=COLORS["grid"], lw=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.04, 0.88, f"r={corr[idx]:.2f}", transform=ax.transAxes, fontsize=8.5, color=COLORS["muted"])
            if row == 0:
                ax.set_title(f"top {col}", fontsize=9)
        for ax in fig.axes:
            for spine in ax.spines.values():
                spine.set_color(COLORS["grid"])
                spine.set_linewidth(0.7)
    return save_fig(fig, "ppt_backup_candidate_prototype_quality.png")


def build_slide08_representation_lineage() -> Path:
    summary = read_json(ROOT / "outputs/input_embedding_ablation/input_embedding_ablation_summary.json")
    reps_by_model = {
        "TimesFM-2.5": ("timesfm_2_5", ["raw_z_patch", "timesfm_tokenizer", "timesfm_hidden"]),
        "Chronos-2": ("chronos_2", ["raw_z_patch", "chronos_proj_with_time", "chronos_hidden"]),
    }
    metrics = ["taxonomy_v0", "domain", "frequency", "patch_index"]
    metric_labels = ["prior probe", "domain", "frequency", "patch index"]
    colors = [COLORS["green"], COLORS["amber"], COLORS["teal"], COLORS["red"]]
    fig = plt.figure(figsize=(13.3, 7.5))
    draw_slide_header(
        fig,
        "Representation Lineage: Tokenizer/Projection and Hidden States Answer Different Questions",
        "NMI between KMeans labels and metadata/probe labels; lower patch-index NMI means weaker position confounding.",
    )
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.96, bottom=0.11, top=0.78, wspace=0.22)
    for col, (display, (model_key, reps)) in enumerate(reps_by_model.items()):
        ax = fig.add_subplot(gs[0, col])
        x = np.arange(len(reps))
        width = 0.18
        for mi, metric in enumerate(metrics):
            values = [summary["models"][model_key]["representations"][rep]["nmi"][metric] for rep in reps]
            ax.bar(x + (mi - 1.5) * width, values, width=width, color=colors[mi], label=metric_labels[mi])
        ax.set_title(display, loc="left")
        ax.set_ylabel("NMI")
        ax.set_ylim(0, 0.48)
        ax.set_xticks(x)
        labels = [r.replace("timesfm_", "").replace("chronos_", "").replace("_", "\n") for r in reps]
        ax.set_xticklabels(labels)
        ax.grid(axis="y", color=COLORS["grid"], lw=0.6, alpha=0.8)
        panel_label(ax, "AB"[col])
        if col == 1:
            ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return save_fig(fig, "ppt_slide08_representation_lineage.png")


def build_slide09_negative_control() -> Path:
    grouped = load_prototype_bank()
    artifact = concept_matrix(grouped, "artifact_first_patch_behavior")
    cross = read_json(ROOT / "outputs/cross_model_validation/cross_model_validation_summary.json")
    model_rows = {r["display"]: r for r in cross["model_results"]}
    fig = plt.figure(figsize=(13.3, 7.5))
    draw_slide_header(
        fig,
        "Negative Control: Visual Coherence Can Still Be a Position Artifact",
        "TimesFM first-patch behavior is shape-coherent, but its support is position-bound.",
    )
    gs = fig.add_gridspec(1, 3, left=0.07, right=0.96, bottom=0.11, top=0.78, wspace=0.28, width_ratios=[1.25, 1.0, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 1, artifact.shape[1])
    mean = artifact.mean(axis=0)
    std = artifact.std(axis=0)
    ax.fill_between(x, mean - std, mean + std, color=COLORS["red"], alpha=0.14, lw=0)
    ax.plot(x, mean, lw=2.5, color=COLORS["red"])
    ax.axhline(0, color=COLORS["grid"], lw=0.8)
    ax.set_title("A  first-patch artifact prototype", loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.03, 0.92, "patch index = 0 for all prototypes", transform=ax.transAxes, fontsize=9.5, color=COLORS["red"], va="top")

    ax2 = fig.add_subplot(gs[0, 1])
    displays = ["TimesFM-2.5 layer_10", "Chronos-2 layer_11", "Chronos-2-small layer_5"]
    vals = [model_rows[d]["global_retrieval"]["artifact_first_patch_behavior"]["patch_index_diversity"] for d in displays]
    ax2.barh(np.arange(len(vals)), vals, color=[COLORS["red"], COLORS["gray"], COLORS["gray"]])
    ax2.set_yticks(np.arange(len(vals)))
    ax2.set_yticklabels(["TimesFM", "Chronos-2", "Chronos-small"])
    ax2.set_xlabel("patch-index diversity")
    ax2.set_title("B  position audit", loc="left")
    ax2.grid(axis="x", color=COLORS["grid"], lw=0.6)

    ax3 = fig.add_subplot(gs[0, 2])
    hit = [model_rows[d]["global_retrieval"]["artifact_first_patch_behavior"]["same_concept_hit_rate"] for d in displays]
    base = [model_rows[d]["matched_random_baseline"]["prototype_hit_rate"] for d in displays]
    xloc = np.arange(len(hit))
    ax3.bar(xloc - 0.16, hit, width=0.32, color=COLORS["red"], label="artifact hit")
    ax3.bar(xloc + 0.16, base, width=0.32, color=COLORS["gray"], label="matched random")
    ax3.set_xticks(xloc)
    ax3.set_xticklabels(["TimesFM", "Chronos-2", "Chronos-small"], rotation=20, ha="right")
    ax3.set_ylabel("hit rate")
    ax3.set_title("C  cross-model sanity", loc="left")
    ax3.grid(axis="y", color=COLORS["grid"], lw=0.6)
    ax3.legend(frameon=False, fontsize=8)

    return save_fig(fig, "ppt_slide09_artifact_negative_control.png")


def build_slide10_taxonomy_v1_pilot() -> Path:
    taxonomy = read_json(ROOT / "outputs/taxonomy_v1_pilot/taxonomy_v1_pilot_summary.json")
    grouped = load_prototype_bank()
    concepts = ["strong_falling_transition", "smooth_falling_transition", "strong_rising_recovery", "artifact_first_patch_behavior"]
    status = {
        "strong_falling_transition": "primary candidate",
        "smooth_falling_transition": "merge candidate",
        "strong_rising_recovery": "candidate, weaker transfer",
        "artifact_first_patch_behavior": "negative control",
    }
    fig = plt.figure(figsize=(13.3, 7.5))
    draw_slide_header(
        fig,
        "Model-Derived Candidate Family Inventory",
        "Internal splitting turns broad clusters into cleaner prototype families; this is not a final taxonomy.",
    )
    gs = fig.add_gridspec(2, 4, left=0.055, right=0.96, bottom=0.12, top=0.78, hspace=0.34, wspace=0.22)
    for j, concept in enumerate(concepts):
        mat = concept_matrix(grouped, concept)
        color = CONCEPT_COLORS[concept]
        ax = fig.add_subplot(gs[0, j])
        x = np.linspace(0, 1, mat.shape[1])
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14, lw=0)
        ax.plot(x, mean, color=color, lw=2.4)
        ax.axhline(0, color=COLORS["grid"], lw=0.8)
        ax.set_title(CONCEPT_LABELS[concept], loc="left", fontsize=9.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.02, -0.23, status[concept], transform=ax.transAxes, fontsize=8.5, color=COLORS["muted"], va="top")
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")
    headers = ["family", "source", "role", "main risk"]
    rows = [
        ("strong falling transition", "TimesFM c5 internal split", "most defensible so far", "merge with smooth falling"),
        ("smooth falling transition", "TimesFM c5 internal split", "secondary / merge candidate", "traffic-flow dominance"),
        ("strong rising recovery", "TimesFM c8 internal split", "candidate", "weaker Chronos retrieval"),
        ("first-patch artifact", "TimesFM c4", "negative control", "position confounding"),
    ]
    xs = [0.02, 0.31, 0.56, 0.78]
    for x, h in zip(xs, headers):
        ax_table.text(x, 0.92, h, transform=ax_table.transAxes, fontsize=10.5, fontweight="bold", color=COLORS["ink"])
    ax_table.plot([0.02, 0.97], [0.84, 0.84], transform=ax_table.transAxes, color=COLORS["ink"], lw=1.0)
    y = 0.70
    for row in rows:
        for x, item in zip(xs, row):
            ax_table.text(x, y, item, transform=ax_table.transAxes, fontsize=9.4, color=COLORS["ink"] if x == xs[0] else COLORS["muted"])
        ax_table.plot([0.02, 0.97], [y - 0.12, y - 0.12], transform=ax_table.transAxes, color=COLORS["grid"], lw=0.6)
        y -= 0.20
    return save_fig(fig, "ppt_slide10_candidate_family_inventory.png")


def find_parent_subcluster(taxonomy: dict[str, Any], parent_cluster: int, subcluster: int) -> dict[str, Any]:
    for parent in taxonomy["parent_splits"]:
        target = parent["target"]
        if target["cluster"] == parent_cluster:
            for sub in parent["subclusters"]:
                if sub["subcluster"] == subcluster:
                    return sub
    raise KeyError((parent_cluster, subcluster))


def find_second_pilot_cluster(model: str, layer: str, cluster: int) -> dict[str, Any]:
    summary = read_json(ROOT / "outputs/second_pilot_discovery_summary.json")
    layer_data = summary["models"][model]["layers"][layer]["domain_balanced"]
    for item in layer_data["clusters"]:
        if int(item["cluster"]) == int(cluster):
            return item
    raise KeyError((model, layer, cluster))


def crop_prototype_row(path: Path, row: int, columns: int = 4, use_columns: list[int] | None = None) -> list[Image.Image]:
    img = Image.open(path).convert("RGB")
    row_h = img.height / round(img.height / 243)
    col_w = img.width / columns
    cells: list[Image.Image] = []
    for col in (use_columns if use_columns is not None else list(range(columns))):
        left = int(col * col_w + 25)
        right = int((col + 1) * col_w - 25)
        top = int(row * row_h + 54)
        bottom = int((row + 1) * row_h - 10)
        cells.append(img.crop((left, top, right, bottom)))
    return cells


def build_slide08_native_model_evidence() -> Path:
    """Show model-native candidate clusters for TimesFM, Chronos-2, and Chronos-2-small."""
    specs = [
        {
            "display": "TimesFM-2.5",
            "model": "timesfm_2_5",
            "layer": "layer_10",
            "cluster": 5,
            "name": "smooth transition-like family",
            "panel": FIG / "second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_prototype_panel.png",
            "row": 5,
            "use_columns": [1, 2, 3],
            "color": COLORS["blue"],
        },
        {
            "display": "Chronos-2",
            "model": "chronos_2",
            "layer": "layer_11",
            "cluster": 10,
            "name": "sharp transition-like family",
            "panel": FIG / "second_pilot/second_pilot_chronos_2_layer_11_domain_balanced_prototype_panel.png",
            "row": 10,
            "use_columns": [0, 1, 2],
            "color": COLORS["teal"],
        },
        {
            "display": "Chronos-2-small",
            "model": "chronos_2_small",
            "layer": "layer_5",
            "cluster": 8,
            "name": "falling transition-like family",
            "panel": FIG / "second_pilot/second_pilot_chronos_2_small_layer_5_domain_balanced_prototype_panel.png",
            "row": 8,
            "use_columns": [0, 1, 2],
            "color": COLORS["green"],
        },
    ]
    width, height = 3600, 1580
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((120, 72), "Model-Native Candidate Evidence Across TSFMs", font=font(62, True), fill=COLORS["ink"])
    draw.text(
        (120, 150),
        "Each row is discovered within the model's own domain-balanced hidden space; labels are temporary candidate families.",
        font=font(34),
        fill=COLORS["muted"],
    )
    summary_rows = []
    y0 = 270
    row_gap = 405
    patch_w, patch_h = 520, 240
    for i, spec in enumerate(specs):
        cluster = find_second_pilot_cluster(spec["model"], spec["layer"], spec["cluster"])
        y = y0 + i * row_gap
        draw.text((120, y), spec["display"], font=font(48, True), fill=spec["color"])
        layer_text = display_label(spec["layer"])
        draw.text((120, y + 58), f"{layer_text}, cluster {spec['cluster']} · {spec['name']}", font=font(33, True), fill=COLORS["ink"])
        draw.text(
            (120, y + 107),
            f"n={cluster['size']} · top domains: {compact_count_label(cluster['top_domains'], 2)}",
            font=font(28),
            fill=COLORS["muted"],
        )
        draw.text(
            (120, y + 150),
            f"prior probe: {compact_count_label(cluster['top_taxonomy_labels'], 2)} · patch index: {compact_count_label(cluster['top_patch_indices'], 2)}",
            font=font(28),
            fill=COLORS["muted"],
        )
        cells = crop_prototype_row(spec["panel"], spec["row"], use_columns=spec["use_columns"])
        for col, cell in enumerate(cells):
            cell = cell.resize((patch_w, patch_h), Image.Resampling.LANCZOS)
            x = 1650 + col * (patch_w + 80)
            draw.rectangle((x, y + 8, x + patch_w, y + 8 + patch_h), outline=COLORS["grid"], width=4)
            canvas.paste(cell, (x, y + 8))
        summary_rows.append(
            {
                "model": spec["display"],
                "layer": layer_text,
                "cluster": int(spec["cluster"]),
                "temporary_name": spec["name"],
                "size": int(cluster["size"]),
                "top_domains": cluster["top_domains"],
                "top_taxonomy_labels": cluster["top_taxonomy_labels"],
                "top_patch_indices": cluster["top_patch_indices"],
                "mean_abs_robust_slope": cluster["mean_abs_robust_slope"],
                "note": "native candidate from second-pilot domain-balanced hidden-space clustering",
            }
        )

    write_json(
        OUT / "native_model_candidate_summary.json",
        {
            "purpose": "advisor-meeting native candidate evidence; not final taxonomy",
            "selection_rule": "choose visually interpretable transition-like clusters from each model's own domain-balanced second-pilot clustering",
            "candidates": summary_rows,
        },
    )
    out = OUT / "ppt_slide08_native_model_evidence.png"
    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def find_cluster_validation(target_cluster: int) -> dict[str, Any]:
    val = read_json(ROOT / "outputs/cluster_validation/cluster_level_validation_summary.json")
    for item in val["targets"]:
        if item["target"]["model"] == "timesfm_2_5" and item["target"]["cluster"] == target_cluster:
            return item
    raise KeyError(target_cluster)


def condition_metric(condition: dict[str, Any], key: str) -> float:
    if condition.get("status") != "ok":
        return float("nan")
    if key in condition:
        return float(condition[key])
    mean_key = f"{key}_mean"
    if mean_key in condition:
        return float(condition[mean_key])
    return float("nan")


def build_slide11_controlled_retrieval() -> Path:
    taxonomy = read_json(ROOT / "outputs/taxonomy_v1_pilot/taxonomy_v1_pilot_summary.json")
    concept_sources = {
        "strong_rising_recovery": find_parent_subcluster(taxonomy, 8, 0),
        "strong_falling_transition": find_parent_subcluster(taxonomy, 5, 1),
        "smooth_falling_transition": find_parent_subcluster(taxonomy, 5, 0),
        "artifact_first_patch_behavior": find_cluster_validation(4),
    }
    conditions = ["unrestricted", "same_patch_index", "same_frequency", "cross_domain", "cross_domain_same_patch_index"]
    cond_labels = ["unrestricted", "same patch", "same freq.", "cross-domain", "cross-domain\nsame patch"]
    concepts = list(concept_sources.keys())
    shape = np.zeros((len(concepts), len(conditions)))
    pos_frac = np.zeros_like(shape)
    for i, concept in enumerate(concepts):
        if concept == "artifact_first_patch_behavior":
            summary = concept_sources[concept]["condition_summary"]
        else:
            summary = concept_sources[concept]["validation"]["condition_summary"]
        for j, cond in enumerate(conditions):
            c = summary.get(cond, {"status": "empty"})
            shape[i, j] = condition_metric(c, "mean_shape_correlation")
            pos_frac[i, j] = condition_metric(c, "positive_shape_fraction")

    fig = plt.figure(figsize=(13.3, 7.5))
    draw_slide_header(
        fig,
        "Controlled Retrieval: Candidate Motifs Must Survive Confounder Controls",
        "Metric: top-k mean shape correlation; prior motif labels are only explanatory probes.",
    )
    gs = fig.add_gridspec(1, 2, left=0.10, right=0.95, bottom=0.12, top=0.78, wspace=0.28, width_ratios=[1.30, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(shape, cmap="YlGnBu", vmin=0.0, vmax=0.9, aspect="auto")
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(cond_labels)
    ax.set_yticks(np.arange(len(concepts)))
    ax.set_yticklabels([CONCEPT_LABELS[c] for c in concepts])
    ax.set_title("A  shape coherence under retrieval controls", loc="left")
    for i in range(shape.shape[0]):
        for j in range(shape.shape[1]):
            value = shape[i, j]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.8, color=COLORS["ink"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.set_ylabel("shape corr.", rotation=270, labelpad=14)

    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(concepts))
    yvals = shape[:, conditions.index("cross_domain")]
    bars = ax2.bar(x, yvals, color=[CONCEPT_COLORS[c] for c in concepts], alpha=0.92)
    baseline = 0.50
    ax2.axhline(baseline, color=COLORS["muted"], lw=1.0, ls="--")
    ax2.text(2.9, baseline + 0.02, "rough survival line", ha="right", fontsize=8, color=COLORS["muted"])
    ax2.set_ylim(0, 0.9)
    ax2.set_ylabel("cross-domain shape corr.")
    ax2.set_title("B  cross-domain survival", loc="left")
    ax2.set_xticks(x)
    ax2.set_xticklabels([wrap(CONCEPT_LABELS[c], 13) for c in concepts], rotation=0)
    ax2.grid(axis="y", color=COLORS["grid"], lw=0.6)
    for bar, value in zip(bars, yvals):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", fontsize=8.5, color=COLORS["ink"])

    return save_fig(fig, "ppt_slide11_controlled_retrieval.png")


def build_slide12_cross_model_validation() -> Path:
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

    fig = plt.figure(figsize=(13.3, 7.5))
    draw_slide_header(
        fig,
        "Cross-Model Validation: Do TimesFM-Derived Prototypes Transfer to Chronos?",
        "Metric: global retrieval mean shape correlation; matched random controls for easy shape matches.",
    )
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.96, bottom=0.12, top=0.78, wspace=0.28, width_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(model_displays))
    width = 0.18
    for i, concept in enumerate(concepts):
        ax.bar(x + (i - 1.5) * width, values[i], width=width, color=CONCEPT_COLORS[concept], label=CONCEPT_LABELS[concept])
    ax.plot(x, baselines, color=COLORS["ink"], marker="o", lw=1.4, label="matched random")
    ax.set_xticks(x)
    ax.set_xticklabels(model_short)
    ax.set_ylabel("mean shape correlation")
    ax.set_ylim(0, 1.0)
    ax.set_title("A  global retrieval transfer", loc="left")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax2 = fig.add_subplot(gs[0, 1])
    sil = [row["prototype_space"]["silhouette_by_concept"] for row in summary["model_results"]]
    nn = [row["prototype_space"]["mean_1nn_same_concept"] for row in summary["model_results"]]
    ax2.bar(x - 0.15, sil, width=0.30, color=COLORS["blue"], label="silhouette by concept")
    ax2.bar(x + 0.15, nn, width=0.30, color=COLORS["green"], label="1-NN same concept")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_short)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("B  prototype-space agreement", loc="left")
    ax2.grid(axis="y", color=COLORS["grid"], lw=0.6)
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    return save_fig(fig, "ppt_slide12_cross_model_validation.png")


def build_backup_layer_choice() -> Path:
    summary = read_json(ROOT / "outputs/input_embedding_ablation/input_embedding_ablation_summary.json")
    models = [
        ("TimesFM-2.5", "timesfm_2_5", ["raw_z_patch", "timesfm_tokenizer", "timesfm_hidden"]),
        ("Chronos-2", "chronos_2", ["raw_z_patch", "chronos_proj_with_time", "chronos_hidden"]),
        ("Chronos-2-small", "chronos_2_small", ["raw_z_patch", "chronos_proj_with_time", "chronos_hidden"]),
    ]
    fig = plt.figure(figsize=(13.3, 7.5))
    draw_slide_header(
        fig,
        "Backup: Why Not Use Only Tokenizer/Projection or Only Hidden States?",
        "Tokenizer/projection states are closer to local vocabulary; hidden states are contextualized but introduce model-specific confounders.",
    )
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.96, bottom=0.12, top=0.78, wspace=0.25)
    for i, (title, key, reps) in enumerate(models):
        ax = fig.add_subplot(gs[0, i])
        x = np.arange(len(reps))
        taxonomy = [summary["models"][key]["representations"][r]["nmi"]["taxonomy_v0"] for r in reps]
        domain = [summary["models"][key]["representations"][r]["nmi"]["domain"] for r in reps]
        position = [summary["models"][key]["representations"][r]["nmi"]["patch_index"] for r in reps]
        ax.plot(x, taxonomy, marker="o", color=COLORS["green"], label="prior probe")
        ax.plot(x, domain, marker="o", color=COLORS["amber"], label="domain")
        ax.plot(x, position, marker="o", color=COLORS["red"], label="patch index")
        ax.set_xticks(x)
        ax.set_xticklabels(["raw", "tokenizer\n/proj", "hidden"])
        ax.set_ylim(0, 0.45)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
        if i == 0:
            ax.set_ylabel("NMI")
        if i == 2:
            ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return save_fig(fig, "ppt_backup_layer_choice.png")


def build_manifest(paths: list[Path]) -> None:
    manifest = {
        "purpose": "advisor meeting PPT assets",
        "style": "publication-style white background, consistent panel labels, Chinese-English labels",
        "paths": [str(path.relative_to(ROOT)) for path in paths],
    }
    (OUT / "ppt_asset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    setup_matplotlib()
    OUT.mkdir(parents=True, exist_ok=True)
    builders = [
        build_slide01_story_question,
        build_slide02_gap_and_question,
        build_slide03_research_hypotheses,
        build_slide03_taxonomy_v0_math,
        build_backup_taxonomy_table,
        build_slide04_discovery_protocol,
        build_slide06_clustering_evidence,
        build_slide07_original_space_concepts,
        build_backup_candidate_prototype_quality,
        build_slide08_native_model_evidence,
        build_slide08_representation_lineage,
        build_slide09_negative_control,
        build_slide10_taxonomy_v1_pilot,
        build_slide11_controlled_retrieval,
        build_slide12_cross_model_validation,
        build_backup_layer_choice,
    ]
    paths = [builder() for builder in builders]
    build_manifest(paths)
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
