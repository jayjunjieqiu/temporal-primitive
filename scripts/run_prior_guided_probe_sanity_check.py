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
DATA_ROOT = Path("/data/junjieqiu/datasets/basicts_datasets")
OUTPUT_DIR = ROOT / "outputs" / "prior_guided_probe_sanity"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "docs" / "prior_guided_motif_probe_sanity_check.md"

sys.path.insert(0, str(ROOT))
from scripts.explore_motif_taxonomy import LABELS, generate_dataset, label_patch, robust_z  # noqa: E402

FIG_DPI = 240
DISPLAY_LABELS = {
    "flat_low_information": "Flat / low information",
    "trend": "Trend",
    "oscillation": "Oscillation",
    "impulse_spike": "Impulse spike",
    "burst_event": "Burst event",
    "level_shift": "Level shift",
    "volatility_shift": "Volatility shift",
    "intermittent": "Intermittent",
    "mixed_uncertain": "Mixed / uncertain",
}
SHORT_LABELS = {
    "flat_low_information": "Flat",
    "trend": "Trend",
    "oscillation": "Osc.",
    "impulse_spike": "Spike",
    "burst_event": "Burst",
    "level_shift": "Level shift",
    "volatility_shift": "Vol. shift",
    "intermittent": "Intermittent",
    "mixed_uncertain": "Mixed",
}

MACRO_DOMAIN_DEFINITIONS = {
    "Traffic": ["traffic flow", "traffic speed", "road occupancy rates"],
    "Energy": ["electricity consumption", "electricity transformer temperature"],
    "Environment": ["weather", "Beijing air quality"],
    "Finance": ["exchange rate"],
    "Health": ["illness data"],
    "Synthetic control": ["simulated Gaussian data", "simulated pulse data"],
}


def read_desc(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _x: None)


def macro_domain(source_domain: Any) -> str:
    text = str(source_domain)
    for macro, domains in MACRO_DOMAIN_DEFINITIONS.items():
        if text in domains:
            return macro
    return "Other"


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


def count_items(values: list[Any], limit: int | None = None) -> list[dict[str, Any]]:
    items = [{"value": str(k), "count": int(v)} for k, v in Counter(values).most_common()]
    return items if limit is None else items[:limit]


def sample_real_patches(
    data_root: Path,
    patch_len: int,
    context_len: int,
    windows_per_dataset: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
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
        source_domain = str(desc.get("domain", dataset))
        accepted_windows = 0
        attempts = 0
        seen: set[tuple[int, int]] = set()
        while accepted_windows < windows_per_dataset and attempts < max(2000, windows_per_dataset * 80):
            attempts += 1
            node = int(rng.integers(0, shape[1]))
            start = int(rng.integers(0, shape[0] - context_len + 1))
            key = (node, start)
            if key in seen:
                continue
            seen.add(key)
            window = np.asarray(data[start : start + context_len, node, 0], dtype=np.float32)
            window = interpolate_nans(window)
            if window is None or float(np.nanstd(window)) < 1e-7:
                continue
            accepted_windows += 1
            for patch_index, patch_start in enumerate(range(0, context_len - patch_len + 1, patch_len)):
                patch = np.asarray(window[patch_start : patch_start + patch_len], dtype=np.float64)
                result = label_patch(patch, patch_len)
                records.append(
                    {
                        "patch": patch.astype(float).tolist(),
                        "dataset": dataset,
                        "source_domain": source_domain,
                        "macro_domain": macro_domain(source_domain),
                        "frequency_minutes": desc.get("frequency (minutes)"),
                        "node": node,
                        "window_start": start,
                        "patch_index": patch_index,
                        "patch_start": int(start + patch_start),
                        "patch_end": int(start + patch_start + patch_len),
                        "patch_len": patch_len,
                        "label": result.label,
                        "confidence": float(result.confidence),
                        "fired": result.fired,
                        "features": result.features,
                    }
                )
        dataset_summary.append(
            {
                "dataset": dataset,
                "source_domain": source_domain,
                "macro_domain": macro_domain(source_domain),
                "frequency_minutes": desc.get("frequency (minutes)"),
                "accepted_windows": accepted_windows,
                "accepted_patches": accepted_windows * (context_len // patch_len),
                "attempts": attempts,
                "status": "ok" if accepted_windows else "empty",
            }
        )
    return records, dataset_summary


def summarize_real_records(records: list[dict[str, Any]], high_confidence: float) -> dict[str, Any]:
    by_label = {label: [r for r in records if r["label"] == label] for label in LABELS}
    by_macro: dict[str, Counter[str]] = defaultdict(Counter)
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    by_patch_index: dict[str, Counter[str]] = defaultdict(Counter)
    high_by_label = {}
    feature_summary = {}
    for record in records:
        by_macro[str(record["macro_domain"])][str(record["label"])] += 1
        by_dataset[str(record["dataset"])][str(record["label"])] += 1
        by_patch_index[str(record["patch_index"])][str(record["label"])] += 1
    for label, subset in by_label.items():
        high = [r for r in subset if float(r["confidence"]) >= high_confidence and label != "mixed_uncertain"]
        high_by_label[label] = len(high)
        if subset:
            feature_summary[label] = {
                "mean_confidence": float(np.mean([r["confidence"] for r in subset])),
                "median_raw_std": float(np.median([r["features"]["raw_std"] for r in subset])),
                "median_raw_range": float(np.median([r["features"]["raw_range"] for r in subset])),
                "top_macro_domains": count_items([r["macro_domain"] for r in subset], 5),
                "top_datasets": count_items([r["dataset"] for r in subset], 5),
            }

    return {
        "num_real_patches": len(records),
        "label_distribution": count_items([r["label"] for r in records]),
        "high_confidence_threshold": high_confidence,
        "high_confidence_counts": high_by_label,
        "mixed_uncertain_rate": float(np.mean([r["label"] == "mixed_uncertain" for r in records])) if records else 0.0,
        "by_macro_domain": {k: dict(v) for k, v in sorted(by_macro.items())},
        "by_dataset": {k: dict(v) for k, v in sorted(by_dataset.items())},
        "by_patch_index": {k: dict(v) for k, v in sorted(by_patch_index.items(), key=lambda kv: int(kv[0]))},
        "feature_summary_by_label": feature_summary,
    }


def summarize_synthetic(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for patch_len in sorted({int(r["patch_len"]) for r in records}):
        subset = [r for r in records if int(r["patch_len"]) == patch_len]
        total = len(subset)
        correct = [r for r in subset if r["true_label"] == r["pred_label"]]
        hard = [r for r in subset if r["pred_label"] != "mixed_uncertain"]
        per_true = {}
        for label in LABELS:
            items = [r for r in subset if r["true_label"] == label]
            per_true[label] = {
                "n": len(items),
                "recall": float(np.mean([r["pred_label"] == label for r in items])) if items else 0.0,
                "uncertain_rate": float(np.mean([r["pred_label"] == "mixed_uncertain" for r in items])) if items else 0.0,
                "pred_distribution": dict(Counter(r["pred_label"] for r in items)),
            }
        out[str(patch_len)] = {
            "num_patches": total,
            "accuracy_including_mixed": float(len(correct) / total) if total else 0.0,
            "non_uncertain_coverage": float(len(hard) / total) if total else 0.0,
            "per_true_label": per_true,
        }
    return out


def select_diverse_examples(
    records: list[dict[str, Any]],
    label: str,
    n: int,
    high_confidence: float | None = None,
    require_label: bool = True,
) -> list[dict[str, Any]]:
    subset = [r for r in records if (r["label"] == label if require_label else True)]
    if high_confidence is not None:
        subset = [r for r in subset if float(r["confidence"]) >= high_confidence]
    subset = sorted(subset, key=lambda r: float(r["confidence"]), reverse=True)
    selected: list[dict[str, Any]] = []

    used_macro: set[str] = set()
    for record in subset:
        macro = str(record["macro_domain"])
        if macro in used_macro:
            continue
        selected.append(record)
        used_macro.add(macro)
        if len(selected) >= n:
            return selected

    used_dataset = {str(record["dataset"]) for record in selected}
    for record in subset:
        dataset = str(record["dataset"])
        if dataset in used_dataset or record in selected:
            continue
        selected.append(record)
        used_dataset.add(dataset)
        if len(selected) >= n:
            return selected

    for record in subset:
        if record not in selected:
            selected.append(record)
        if len(selected) >= n:
            break
    return selected


def setup_matplotlib() -> Any:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def display_z(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    std = float(np.std(x))
    if std <= eps:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / std


def taxonomy_prototype(label: str, n: int = 64) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    if label == "flat_low_information":
        return np.zeros(n)
    if label == "trend":
        return display_z(t)
    if label == "oscillation":
        return display_z(np.sin(2.0 * math.pi * 2.5 * t))
    if label == "impulse_spike":
        x = np.zeros(n)
        x[n // 2] = 4.0
        return display_z(x)
    if label == "burst_event":
        x = np.zeros(n)
        start = n // 2 - 7
        burst = np.hanning(14) * 2.4
        x[start : start + len(burst)] = burst
        return display_z(x)
    if label == "level_shift":
        x = np.r_[np.zeros(n // 2), np.ones(n - n // 2) * 1.8]
        return display_z(x)
    if label == "volatility_shift":
        left = 0.12 * np.sin(np.linspace(0, 4.0 * math.pi, n // 2))
        right = 0.9 * np.sin(np.linspace(0, 8.0 * math.pi, n - n // 2))
        return display_z(np.r_[left, right])
    if label == "intermittent":
        x = np.zeros(n)
        x[[10, 23, 39, 52]] = [1.3, 1.8, 1.5, 1.9]
        return display_z(x)
    if label == "mixed_uncertain":
        x = 0.8 * (t - 0.5) + 1.1 * np.sin(2.0 * math.pi * 1.3 * t)
        x[n // 2 + 4] += 2.0
        return display_z(x)
    raise KeyError(label)


def plot_expected_prototypes() -> Path:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 7.3), sharex=True, sharey=True)
    colors = ["#5B7DB1", "#D28B2C", "#4A9B6E", "#C75A5A", "#7C62B8", "#4C9AA6", "#B07B45", "#777777", "#A35C7A"]
    for ax, label, color in zip(axes.ravel(), LABELS, colors):
        y = taxonomy_prototype(label)
        max_abs = float(np.max(np.abs(y)))
        if max_abs > 0:
            y = y / max_abs * 3.1
        x = np.linspace(0.0, 1.0, len(y))
        ax.axhline(0.0, color="#D8D8D8", lw=0.8)
        ax.plot(x, y, color=color, lw=2.2)
        ax.fill_between(x, 0.0, y, color=color, alpha=0.12, lw=0)
        ax.set_title(DISPLAY_LABELS[label], fontsize=10.5, fontweight="bold", pad=4)
        ax.set_ylim(-3.8, 3.8)
        ax.set_xticks([0.0, 1.0])
        ax.set_yticks([-3, 0, 3])
    fig.suptitle("Expected shape prototypes used by the prior-guided motif probe", fontsize=14, y=0.992)
    fig.text(0.5, 0.020, "Reference shapes are visually normalized illustrative prototypes, not ground-truth labels for real patches.", ha="center", fontsize=9.0, color="#555555")
    fig.tight_layout(rect=[0.02, 0.045, 1.0, 0.952], h_pad=1.0, w_pad=0.55)
    out = FIG_DIR / "expected_prior_guided_prototype_shapes.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    return out


def plot_real_high_confidence(records: list[dict[str, Any]], high_confidence: float) -> Path:
    plt = setup_matplotlib()
    rows = LABELS
    cols = 6
    fig, axes = plt.subplots(len(rows), cols, figsize=(13.2, 11.2), sharex=True)
    colors = {
        "Traffic": "#3B6FB6",
        "Energy": "#D98C2B",
        "Environment": "#4A9B6E",
        "Finance": "#7C5FB5",
        "Health": "#C9505A",
        "Synthetic control": "#777777",
        "Other": "#8C8C8C",
    }
    for row, label in enumerate(rows):
        examples = select_diverse_examples(records, label, cols, None if label == "mixed_uncertain" else high_confidence)
        for col in range(cols):
            ax = axes[row, col]
            ax.axhline(0.0, color="#D8D8D8", lw=0.8)
            if col < len(examples):
                record = examples[col]
                z = display_z(np.asarray(record["patch"], dtype=np.float64))
                x = np.arange(len(z))
                macro = str(record["macro_domain"])
                ax.plot(x, z, color=colors.get(macro, "#555555"), lw=1.8)
                ax.fill_between(x, 0, z, color=colors.get(macro, "#555555"), alpha=0.12, lw=0)
                title = f"{record['dataset']} | {macro}\nconf {record['confidence']:.2f}"
                ax.set_title(title, fontsize=7.1, pad=2.0)
            else:
                ax.text(0.5, 0.5, "No example", ha="center", va="center", transform=ax.transAxes, fontsize=7)
            ax.set_ylim(-4.3, 4.3)
            ax.set_xticks([0, max(1, int(records[0]["patch_len"]) - 1)])
            if col == 0:
                ax.set_ylabel(DISPLAY_LABELS[label], fontsize=8.5)
            else:
                ax.set_yticklabels([])
    fig.suptitle("Prior-guided motif probe: high-confidence real patches in original time space", fontsize=14, y=0.996)
    fig.text(0.5, 0.012, "Non-overlapping 16-step patches; curves are z-normalized only for shape inspection.", ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.982], h_pad=0.7, w_pad=0.45)
    out = FIG_DIR / "real_patch_high_confidence_examples.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    return out


def plot_distribution(summary: dict[str, Any]) -> Path:
    plt = setup_matplotlib()
    labels = LABELS
    macros = [m for m in MACRO_DOMAIN_DEFINITIONS if m in summary["by_macro_domain"]]
    macros += [m for m in summary["by_macro_domain"] if m not in macros]

    fig = plt.figure(figsize=(12.8, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    counts = Counter({item["value"]: item["count"] for item in summary["label_distribution"]})
    y = np.arange(len(labels))
    values = [counts.get(label, 0) for label in labels]
    ax0.barh(y, values, color="#4F7CAC", alpha=0.92)
    ax0.set_yticks(y)
    ax0.set_yticklabels([DISPLAY_LABELS[label] for label in labels])
    ax0.invert_yaxis()
    ax0.set_xlabel("patch count")
    ax0.set_title("Overall label distribution", fontsize=11, fontweight="bold")
    for yi, value in zip(y, values):
        ax0.text(value + max(values) * 0.01, yi, f"{value}", va="center", fontsize=8)

    ax1 = fig.add_subplot(gs[0, 1])
    mat = np.zeros((len(labels), len(macros)), dtype=float)
    for j, macro in enumerate(macros):
        macro_counts = summary["by_macro_domain"].get(macro, {})
        total = max(1, sum(int(v) for v in macro_counts.values()))
        for i, label in enumerate(labels):
            mat[i, j] = int(macro_counts.get(label, 0)) / total
    im = ax1.imshow(mat, cmap="Blues", vmin=0.0, vmax=max(0.01, float(np.nanmax(mat))))
    ax1.set_xticks(np.arange(len(macros)))
    ax1.set_xticklabels(macros, rotation=30, ha="right")
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_yticklabels([DISPLAY_LABELS[label] for label in labels])
    ax1.set_title("Within-domain label proportion", fontsize=11, fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(macros)):
            ax1.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7.0, color="#1B1B1B")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.035, pad=0.02)
    cbar.ax.set_ylabel("proportion", rotation=270, labelpad=12)

    fig.suptitle("Prior-guided motif probe distribution on real patch bank", fontsize=14, y=0.996)
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.95])
    out = FIG_DIR / "real_patch_label_distribution.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    return out


def plot_synthetic_confusion(synthetic_records: list[dict[str, Any]], patch_len: int) -> Path:
    plt = setup_matplotlib()
    mat = np.zeros((len(LABELS), len(LABELS)), dtype=float)
    for i, true_label in enumerate(LABELS):
        subset = [r for r in synthetic_records if int(r["patch_len"]) == patch_len and r["true_label"] == true_label]
        total = max(1, len(subset))
        pred_counts = Counter(r["pred_label"] for r in subset)
        for j, pred_label in enumerate(LABELS):
            mat[i, j] = pred_counts.get(pred_label, 0) / total
    fig, ax = plt.subplots(figsize=(9.6, 8.0))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_xticklabels([DISPLAY_LABELS[label] for label in LABELS], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_yticklabels([DISPLAY_LABELS[label] for label in LABELS])
    ax.set_xlabel("predicted probe label")
    ax.set_ylabel("synthetic generator family")
    ax.set_title(f"Synthetic calibration confusion, patch length {patch_len}", fontsize=13, fontweight="bold")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if mat[i, j] >= 0.05:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7.2, color="white" if mat[i, j] > 0.45 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = FIG_DIR / f"synthetic_calibration_confusion_patch{patch_len}.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    return out


def plot_ambiguity_examples(records: list[dict[str, Any]]) -> Path:
    plt = setup_matplotlib()
    ambiguous = [
        r
        for r in records
        if r["label"] == "mixed_uncertain" or len(r["fired"]) >= 2 or (r["label"] != "mixed_uncertain" and r["confidence"] < 0.62)
    ]
    ambiguous = sorted(ambiguous, key=lambda r: (r["label"] != "mixed_uncertain", -len(r["fired"]), r["confidence"]))
    examples = []
    seen: set[tuple[str, str]] = set()
    for record in ambiguous:
        key = (str(record["dataset"]), str(record["patch_index"]))
        if key in seen:
            continue
        seen.add(key)
        examples.append(record)
        if len(examples) >= 18:
            break
    fig, axes = plt.subplots(3, 6, figsize=(13.0, 5.7), sharex=True, sharey=True)
    for ax, record in zip(axes.ravel(), examples):
        z = display_z(np.asarray(record["patch"], dtype=np.float64))
        x = np.arange(len(z))
        ax.axhline(0.0, color="#D8D8D8", lw=0.8)
        ax.plot(x, z, color="#B14E4E", lw=1.7)
        fired = "+".join(SHORT_LABELS.get(v, v) for v in record["fired"][:2]) if record["fired"] else "none"
        ax.set_title(f"{SHORT_LABELS[record['label']]}\n{record['dataset']}\n{fired}", fontsize=6.6, pad=2.0)
        ax.set_ylim(-4.8, 4.8)
    for ax in axes.ravel()[len(examples) :]:
        ax.axis("off")
    fig.suptitle("Ambiguous or rule-conflict examples in original time space", fontsize=14, y=0.995)
    fig.text(0.5, 0.015, "These examples define the boundary where the probe should remain weak rather than become ground truth.", ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0.02, 0.04, 1.0, 0.95], h_pad=0.8, w_pad=0.5)
    out = FIG_DIR / "real_patch_ambiguity_examples.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    return out


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_report(summary: dict[str, Any], figures: dict[str, str]) -> None:
    real = summary["real_patch_summary"]
    syn16 = summary["synthetic_summary"]["16"]
    syn32 = summary["synthetic_summary"]["32"]
    lines = [
        "# Prior-Guided Motif Probe 原空间 Sanity Check",
        "",
        "## 1. 目的",
        "",
        "这个检查补上一个关键证据：`motif taxonomy v0` / `prior-guided motif probe` 不是 ground truth，但在用它审计 TSFM representation clusters 之前，我们必须先确认这套 deterministic classifier 在 original time-series space 中确实能选出符合人类直觉的 `shapelet-like local patterns`。",
        "",
        "因此本报告只回答一个窄问题：这些规则能否按预期把 patch 分成 trend、oscillation、spike、burst、level shift、volatility shift、intermittent、flat 和 mixed/uncertain？它不用于给 KMeans cluster 命名，也不证明真实世界存在唯一正确的 motif taxonomy。",
        "",
        "## 2. 实验设置",
        "",
        f"- real dataset root: `{summary['data_root']}`",
        f"- excluded dataset: `BLAST`",
        f"- patch length: `{summary['patch_len']}`",
        f"- context length: `{summary['context_len']}`",
        f"- windows per dataset: `{summary['windows_per_dataset']}`",
        f"- sampled real patches: `{real['num_real_patches']}`",
        f"- high-confidence threshold for real examples: `{real['high_confidence_threshold']}`",
        "",
        "Classifier 来自 `scripts/explore_motif_taxonomy.py::label_patch`，核心是 robust z-normalization 后的 deterministic shape statistics：linear fit、FFT/spectral concentration、robust outlier count、active-run statistics、mean change score、variance ratio，以及 raw std/range。",
        "",
        "| probe label | 主要数学/统计判定 | 稳健性判断 |",
        "|---|---|---|",
        "| `flat_low_information` | raw std 与 raw range 很低 | control label，真实数据中受预处理和量纲影响 |",
        "| `trend` | robust z 后 linear fit 的 `abs_slope` 和 `R2` 高 | 16-step patch 下相对稳健 |",
        "| `oscillation` | FFT dominant component power ratio 高，且 zero-crossing 足够 | 16-step 下边界敏感，32-step 更稳 |",
        "| `impulse_spike` | 1-2 个 isolated extreme robust z-score points | soft probe，容易受 outlier 和边界影响 |",
        "| `burst_event` | contiguous active run，宽于单点 spike 但短于整段 patch | soft probe，容易和 spike/intermittent 混淆 |",
        "| `level_shift` | best two-segment mean difference / pooled variance 高 | 可用，但 smooth ramp 会与 trend 混淆 |",
        "| `volatility_shift` | split 前后 std ratio 高，mean change 不强 | 谨慎使用，需要更多 change-point audit |",
        "| `intermittent` | 多个 separated active runs，且没有单个 dominant impulse | 16-step 下常退化成 spike，32-step 更稳 |",
        "| `mixed_uncertain` | detector 弱、冲突或 top-score margin 太小 | 必须保留，避免 false hard label |",
        "",
        "## 3. Synthetic Calibration",
        "",
        "Synthetic calibration 用带已知生成机制的 patches 检查规则是否大体按设计工作。它不是现实语义 ground truth，但可以暴露规则之间的混淆边界。",
        "",
        f"- patch length 16: accuracy including mixed = `{syn16['accuracy_including_mixed']:.3f}`, non-uncertain coverage = `{syn16['non_uncertain_coverage']:.3f}`",
        f"- patch length 32: accuracy including mixed = `{syn32['accuracy_including_mixed']:.3f}`, non-uncertain coverage = `{syn32['non_uncertain_coverage']:.3f}`",
        "",
        f"![Synthetic patch16 confusion](../{figures['synthetic_confusion_16']})",
        "",
        f"![Synthetic patch32 confusion](../{figures['synthetic_confusion_32']})",
        "",
        "读图方式：行是 synthetic generator family，列是 probe 预测标签。对角线越强，说明规则越接近预设语义；非对角线暴露了规则混淆。例如短 patch 下 `burst_event`、`intermittent` 和 `impulse_spike` 容易相互混淆，这也是后续报告中一直把 event-like labels 作为 soft audit probe 的原因。",
        "",
        "### 3.1 关键负结果：当前 classifier 不能可靠得到我们想要的 motif 分类",
        "",
        "这个 sanity check 的最重要结论不是“规则大体可用”，而是相反：**当前 deterministic classifier 无法稳定、准确地恢复我们希望表达的 human-prior motif classes**。因此它不应被继续写成一个可靠的 `motif classifier`，最多只能作为一个失败边界明确的 weak audit probe / negative control。",
        "",
        "量化证据如下：",
        "",
        f"- patch length 16 的 overall accuracy including mixed 只有 `{syn16['accuracy_including_mixed']:.3f}`；也就是说，在 Chronos-2 的 16-step patch setting 下，规则分类和我们预设生成机制的一致性很弱。",
        f"- patch length 32 虽然更好，但 accuracy including mixed 也只有 `{syn32['accuracy_including_mixed']:.3f}`，仍不足以支持把它当作 reliable ground-truth-like labels。",
        f"- 在 16-step 下，`impulse_spike` recall 只有 `{syn16['per_true_label']['impulse_spike']['recall']:.3f}`，`burst_event` recall 只有 `{syn16['per_true_label']['burst_event']['recall']:.3f}`，`intermittent` recall 只有 `{syn16['per_true_label']['intermittent']['recall']:.3f}`。",
        f"- `intermittent` 在 16-step synthetic patches 中大量被判成 `impulse_spike`，而 `burst_event` 又经常被判成 `volatility_shift` 或 `mixed_uncertain`。这说明 active-run / outlier / variance-ratio 规则在短 patch 上没有形成我们直觉中清楚的语义边界。",
        f"- 真实数据上 `mixed_uncertain` rate 达到 `{real['mixed_uncertain_rate']:.3f}`，并且 `level_shift` 与 `mixed_uncertain` 占比很高，说明当前规则很容易把复杂局部形态压到少数统计触发项上。",
        "",
        "因此，后续所有 TSFM cluster 分析中，`prior-guided motif` 必须降级为 **human-prior diagnostic annotation**：它只能帮助我们发现“model-derived cluster 与人类先验是否一一对应”这个问题，不能作为 cluster 命名依据，不能作为 taxonomy v1 的监督信号，也不能被当作模型学到了某个 motif 的证明。",
        "",
        "## 4. Real Patch Original-Space Inspection",
        "",
        "先放一张 reference figure：这是我们之前 PPT 里展示过的 human-prior expected shapes，也就是 classifier 期望捕捉的形态原型。它们只用于帮助读者理解规则目标，不代表真实数据中的 ground truth motif。",
        "",
        f"![Expected prior-guided prototype shapes](../{figures['expected_prototypes']})",
        "",
        "下面这张图直接回到真实数据 patch 的原空间。每一行是一个 probe label，每个小图是 classifier 选出的 high-confidence 或代表性 patch。曲线只做 z-normalization 以便比较形状；它们仍是原始时间轴上的 patch，不是 embedding-space 投影。",
        "",
        f"![Real high confidence examples](../{figures['real_examples']})",
        "",
        "这张图的作用是回答老师最可能追问的问题：`prior-guided motif` 到底长什么样？但对照 expected shapes 后可以看到，真实数据中的高置信样本并不总是落在人类直觉中的干净 prototype 上。`trend`、部分 `oscillation` 和部分 `level_shift` 还算可解释；但 `impulse_spike`、`burst_event`、`volatility_shift`、`intermittent` 经常混入 step-like、ramp-like、scale-driven 或 boundary-cut 形态。因此这张图支持的结论是：当前 probe 可以暴露规则失败边界，但不能证明它已经正确 classify patches。",
        "",
        "## 5. Label Distribution on Real Patch Bank",
        "",
        f"真实数据上的 `mixed_uncertain` rate 为 `{real['mixed_uncertain_rate']:.3f}`。这不是失败，而是我们主动保留 classifier 边界的安全阀：当规则冲突或信号不够强时，不强行把 patch 贴成先验 motif。",
        "",
        f"![Real label distribution](../{figures['real_distribution']})",
        "",
        "这张图说明 prior-guided motif probe 在真实 patch bank 上并不是均匀分布的，也会受到 macro-domain 和 dataset composition 影响。更重要的是，label distribution 反映的是 detector 触发频率，不等于真实 motif 语义分布。因此在 TSFM cluster audit 中，它只能作为解释性 diagnostic，不能作为训练标签、cluster ground truth 或 taxonomy v1 的候选来源。",
        "",
        "## 6. Ambiguity and Failure Boundary",
        "",
        f"![Ambiguity examples](../{figures['ambiguity_examples']})",
        "",
        "这些例子是报告里应该主动展示的 failure boundary：一些 patch 同时触发多个 detector，或者形态处在 spike/burst/intermittent、trend/level-shift、flat/noise 的边界。后续如果 cluster 的 prior-guided motif 分布不纯，不能立刻说 cluster 失败；更合理的解释是 TSFM 的 model-derived motif clusters 可能不是 human-prior taxonomy 的一一映射。",
        "",
        "## 7. 可以稳健声称什么",
        "",
        "- 当前 `prior-guided motif probe` 没有通过 reliable classifier 的 sanity check；它不能稳定恢复我们预期的 human-prior motif classes。",
        "- `trend`、部分 `oscillation` 和部分 `level_shift` 可以作为弱解释线索；但即便这些类别也不能被视为 ground truth。",
        "- `impulse_spike`、`burst_event`、`volatility_shift`、`intermittent` 在短 patch 上混淆严重，应主要作为 failure-boundary evidence，而不是正向语义标签。",
        "- `mixed_uncertain` 是必要标签，用来避免把 composite / boundary-cut patches 误当作干净 motif。",
        "- 这个 sanity check 支持我们在 TSFM cluster 报告中使用 `prior-guided motif probe NMI` 和 label distribution 作为 audit evidence，但解释方向应是：model-derived clusters 不应被迫对齐一个本身不可靠的 human-prior classifier。",
        "",
        "## 8. 下一步接入 TSFM Cluster 报告",
        "",
        "建议在 `Chronos-2` layer-wise validation 报告中引用本报告作为前置负结果：先说明 human-prior probe 没有成为可靠 classifier，再展示 model-derived clusters 与该 probe 并非一一对应。这样 narrative 会更严谨：我们不是先验定义 taxonomy 后强行套模型，而是证明先验 classifier 本身不足以定义 patch-level temporal concepts，因此需要转向 model-derived motif taxonomy discovery protocol。",
        "",
        "## 9. 输出文件",
        "",
        f"- summary: `{summary['summary_path']}`",
        f"- expected prototypes: `{figures['expected_prototypes']}`",
        f"- real examples: `{figures['real_examples']}`",
        f"- real distribution: `{figures['real_distribution']}`",
        f"- ambiguity examples: `{figures['ambiguity_examples']}`",
        f"- synthetic patch16 confusion: `{figures['synthetic_confusion_16']}`",
        f"- synthetic patch32 confusion: `{figures['synthetic_confusion_32']}`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--windows-per-dataset", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--high-confidence", type=float, default=0.65)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    real_records, dataset_summary = sample_real_patches(
        data_root=args.data_root,
        patch_len=args.patch_len,
        context_len=args.context_len,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
    )
    synthetic_records = generate_dataset(samples_per_setting=5, seed=13)
    real_summary = summarize_real_records(real_records, args.high_confidence)
    synthetic_summary = summarize_synthetic(synthetic_records)

    figures = {
        "expected_prototypes": rel(plot_expected_prototypes()),
        "real_examples": rel(plot_real_high_confidence(real_records, args.high_confidence)),
        "real_distribution": rel(plot_distribution(real_summary)),
        "ambiguity_examples": rel(plot_ambiguity_examples(real_records)),
        "synthetic_confusion_16": rel(plot_synthetic_confusion(synthetic_records, 16)),
        "synthetic_confusion_32": rel(plot_synthetic_confusion(synthetic_records, 32)),
    }

    summary = {
        "script": "scripts/run_prior_guided_probe_sanity_check.py",
        "data_root": str(args.data_root),
        "patch_len": args.patch_len,
        "context_len": args.context_len,
        "windows_per_dataset": args.windows_per_dataset,
        "seed": args.seed,
        "dataset_summary": dataset_summary,
        "real_patch_summary": real_summary,
        "synthetic_summary": synthetic_summary,
        "figures": figures,
        "summary_path": rel(OUTPUT_DIR / "prior_guided_probe_sanity_summary.json"),
        "report_path": rel(REPORT_PATH),
    }
    (OUTPUT_DIR / "prior_guided_probe_sanity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(summary, figures)
    print(json.dumps({"summary": summary["summary_path"], "report": summary["report_path"], "figures": figures}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
