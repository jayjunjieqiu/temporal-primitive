from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

LABELS = [
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


@dataclass
class LabelResult:
    label: str
    confidence: float
    fired: list[str]
    features: dict[str, float]


def robust_z(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > eps:
        return (x - med) / (1.4826 * mad + eps)
    std = float(np.std(x))
    return (x - float(np.mean(x))) / max(std, eps)


def longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def count_true_runs(mask: np.ndarray) -> int:
    padded = np.r_[False, mask.astype(bool), False]
    return int(np.sum((~padded[:-1]) & padded[1:]))


def linear_features(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    t = np.linspace(-1.0, 1.0, n)
    slope, intercept = np.polyfit(t, x, deg=1)
    pred = slope * t + intercept
    ss_res = float(np.sum((x - pred) ** 2))
    ss_tot = float(np.sum((x - np.mean(x)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    return float(slope), float(max(0.0, min(1.0, r2)))


def best_mean_split_score(x: np.ndarray) -> tuple[float, int]:
    n = len(x)
    best_score = 0.0
    best_idx = n // 2
    for idx in range(max(3, n // 4), min(n - 3, 3 * n // 4) + 1):
        left = x[:idx]
        right = x[idx:]
        pooled = math.sqrt((float(np.var(left)) + float(np.var(right))) / 2.0 + 1e-8)
        score = abs(float(np.mean(right) - np.mean(left))) / pooled
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_score, best_idx


def best_std_ratio(x: np.ndarray) -> float:
    n = len(x)
    best = 1.0
    for idx in range(max(3, n // 4), min(n - 3, 3 * n // 4) + 1):
        left_std = float(np.std(x[:idx])) + 1e-6
        right_std = float(np.std(x[idx:])) + 1e-6
        best = max(best, left_std / right_std, right_std / left_std)
    return best


def spectral_features(x: np.ndarray) -> tuple[float, float, int]:
    centered = x - np.mean(x)
    power = np.abs(np.fft.rfft(centered)) ** 2
    if len(power) <= 2 or float(np.sum(power[1:])) <= 1e-8:
        return 0.0, 0.0, 0
    nonzero = power[1:]
    dom_idx = int(np.argmax(nonzero) + 1)
    ratio = float(power[dom_idx] / np.sum(nonzero))
    zero_crossings = int(np.sum(np.diff(np.signbit(centered)) != 0))
    return ratio, float(dom_idx), zero_crossings


def extract_features(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    z = robust_z(x)
    slope, r2 = linear_features(z)
    mean_score, mean_split = best_mean_split_score(z)
    std_ratio = best_std_ratio(z)
    spectral_ratio, dominant_cycles, zero_crossings = spectral_features(z)
    active = np.abs(z) > 2.0
    strong_active = np.abs(z) > 3.0
    return {
        "raw_std": float(np.std(x)),
        "raw_range": float(np.max(x) - np.min(x)),
        "abs_slope": abs(slope),
        "trend_r2": r2,
        "mean_change_score": mean_score,
        "mean_split_index": float(mean_split),
        "std_ratio": std_ratio,
        "spectral_ratio": spectral_ratio,
        "dominant_cycles": dominant_cycles,
        "zero_crossings": float(zero_crossings),
        "max_robust_z": float(np.max(np.abs(z))),
        "active_ratio": float(np.mean(active)),
        "strong_active_count": float(np.sum(strong_active)),
        "active_runs": float(count_true_runs(active)),
        "longest_active_run": float(longest_true_run(active)),
    }


def score_detectors(features: dict[str, float], patch_len: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    scores["flat_low_information"] = max(
        0.0,
        min(1.0, (0.08 - features["raw_std"]) / 0.08),
        min(1.0, (0.30 - features["raw_range"]) / 0.30),
    )
    spike_points = features["strong_active_count"]
    scores["impulse_spike"] = 0.0
    if features["max_robust_z"] >= 4.0 and spike_points <= 2 and features["longest_active_run"] <= 2:
        scores["impulse_spike"] = min(1.0, features["max_robust_z"] / 8.0)

    min_cycles = 1.0 if patch_len <= 16 else 1.5
    scores["oscillation"] = 0.0
    if (
        features["spectral_ratio"] >= 0.42
        and features["dominant_cycles"] >= min_cycles
        and features["zero_crossings"] >= 3
    ):
        scores["oscillation"] = min(1.0, features["spectral_ratio"])

    scores["trend"] = 0.0
    if features["abs_slope"] >= 0.75 and features["trend_r2"] >= 0.70 and scores["impulse_spike"] < 0.55:
        scores["trend"] = min(1.0, 0.5 * features["trend_r2"] + 0.5 * min(1.0, features["abs_slope"] / 1.8))

    scores["level_shift"] = 0.0
    if features["mean_change_score"] >= 1.45 and features["trend_r2"] <= 0.86:
        scores["level_shift"] = min(1.0, features["mean_change_score"] / 4.0)

    scores["volatility_shift"] = 0.0
    if features["std_ratio"] >= 2.2 and features["mean_change_score"] <= 1.25:
        scores["volatility_shift"] = min(1.0, features["std_ratio"] / 5.0)

    longest_ratio = features["longest_active_run"] / patch_len
    scores["burst_event"] = 0.0
    if 0.15 <= features["active_ratio"] <= 0.60 and longest_ratio >= 0.12 and scores["impulse_spike"] < 0.55:
        scores["burst_event"] = min(1.0, 0.5 * features["active_ratio"] / 0.60 + 0.5 * min(1.0, longest_ratio / 0.35))

    scores["intermittent"] = 0.0
    if (
        0.08 <= features["active_ratio"] <= 0.40
        and features["active_runs"] >= 2
        and longest_ratio <= 0.20
        and scores["impulse_spike"] < 0.55
    ):
        scores["intermittent"] = min(1.0, 0.5 * features["active_ratio"] / 0.40 + 0.5 * min(1.0, features["active_runs"] / 4.0))

    return scores


def label_patch(x: np.ndarray, patch_len: int) -> LabelResult:
    features = extract_features(x)
    scores = score_detectors(features, patch_len)
    fired = [name for name, score in scores.items() if score >= 0.55]
    non_flat = [name for name in fired if name != "flat_low_information"]

    if not fired:
        return LabelResult("mixed_uncertain", 0.35, [], features)
    if fired == ["flat_low_information"]:
        return LabelResult("flat_low_information", scores["flat_low_information"], fired, features)

    ranked = sorted(((score, name) for name, score in scores.items() if name != "flat_low_information"), reverse=True)
    top_score, top_name = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if len(non_flat) >= 2 and (top_score - second_score) <= 0.18:
        return LabelResult("mixed_uncertain", top_score - second_score, fired, features)
    if top_score < 0.55:
        return LabelResult("mixed_uncertain", top_score, fired, features)
    return LabelResult(top_name, top_score, fired, features)


def add_noise(x: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    return x + rng.normal(0.0, noise, size=len(x))


def event_center(length: int, alignment: float) -> int:
    return int(np.clip(round((0.20 + 0.60 * alignment) * (length - 1)), 1, length - 2))


def synthetic_patch(
    label: str,
    patch_len: int,
    amplitude: float,
    noise: float,
    alignment: float,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, patch_len)
    x = np.zeros(patch_len, dtype=np.float64)
    center = event_center(patch_len, alignment)

    if label == "flat_low_information":
        x = np.zeros(patch_len)
    elif label == "trend":
        sign = -1.0 if rng.random() < 0.5 else 1.0
        curvature = rng.normal(0.0, 0.05) * (t - 0.5) ** 2
        x = sign * amplitude * (t - 0.5) + curvature
    elif label == "oscillation":
        cycles = rng.choice([1.5, 2.0, 3.0]) if patch_len >= 32 else rng.choice([1.0, 1.5, 2.0])
        phase = 2.0 * math.pi * alignment
        x = amplitude * np.sin(2.0 * math.pi * cycles * t + phase)
    elif label == "impulse_spike":
        x[center] = amplitude * rng.choice([-1.0, 1.0])
        if rng.random() < 0.35 and center + 1 < patch_len:
            x[center + 1] = 0.55 * x[center]
    elif label == "burst_event":
        width = max(3, int(round(patch_len * rng.uniform(0.15, 0.30))))
        start = int(np.clip(center - width // 2, 0, patch_len - width))
        window = np.hanning(width)
        carrier = np.sin(np.linspace(0, rng.uniform(1.5, 3.5) * math.pi, width))
        x[start : start + width] += amplitude * (0.65 + 0.35 * carrier) * np.maximum(window, 0.2)
    elif label == "level_shift":
        x[center:] = amplitude * rng.choice([-1.0, 1.0])
    elif label == "volatility_shift":
        left_scale = noise * 0.4 + 0.02
        right_scale = amplitude * 0.35 + noise
        if rng.random() < 0.5:
            left_scale, right_scale = right_scale, left_scale
        x[:center] = rng.normal(0.0, left_scale, size=center)
        x[center:] = rng.normal(0.0, right_scale, size=patch_len - center)
        return x
    elif label == "intermittent":
        num_events = 2 if patch_len <= 16 else rng.integers(3, 5)
        positions = np.linspace(2, patch_len - 3, num_events).astype(int)
        jitter = rng.integers(-1, 2, size=num_events)
        for pos in np.clip(positions + jitter, 1, patch_len - 2):
            x[pos] += amplitude * rng.choice([0.8, 1.0, 1.2])
    elif label == "mixed_uncertain":
        x = 0.65 * amplitude * (t - 0.5)
        x[center] += amplitude * 0.9
        if patch_len >= 32:
            x += 0.35 * amplitude * np.sin(2.0 * math.pi * 2.0 * t)
    else:
        raise ValueError(f"unknown label: {label}")

    return add_noise(x, rng, noise)


def generate_dataset(samples_per_setting: int = 5, seed: int = 13) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for patch_len in [16, 32]:
        for label in LABELS:
            for noise in [0.02, 0.08, 0.16]:
                for amplitude in [0.7, 1.2, 2.0]:
                    for alignment in [0.0, 0.33, 0.66, 1.0]:
                        for _ in range(samples_per_setting):
                            patch = synthetic_patch(label, patch_len, amplitude, noise, alignment, rng)
                            result = label_patch(patch, patch_len)
                            records.append(
                                {
                                    "patch": patch.astype(float).tolist(),
                                    "patch_len": patch_len,
                                    "true_label": label,
                                    "pred_label": result.label,
                                    "confidence": result.confidence,
                                    "fired": result.fired,
                                    "noise": noise,
                                    "amplitude": amplitude,
                                    "alignment": alignment,
                                    "features": result.features,
                                }
                            )
    return records


def nearest_neighbor_agreement(records: list[dict[str, Any]], patch_len: int) -> dict[str, float]:
    subset = [r for r in records if r["patch_len"] == patch_len]
    patches = np.asarray([r["patch"] for r in subset], dtype=np.float64)
    labels = [r["true_label"] for r in subset]
    pred_labels = [r["pred_label"] for r in subset]
    z = np.asarray([robust_z(p) for p in patches])
    dists = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    nn = np.argmin(dists, axis=1)
    true_agree = np.mean([labels[i] == labels[j] for i, j in enumerate(nn)])
    pred_agree = np.mean([pred_labels[i] == pred_labels[j] for i, j in enumerate(nn)])
    return {
        "raw_patch_nn_true_label_agreement": float(true_agree),
        "raw_patch_nn_pred_label_agreement": float(pred_agree),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for patch_len in [16, 32]:
        subset = [r for r in records if r["patch_len"] == patch_len]
        y_true = [r["true_label"] for r in subset]
        y_pred = [r["pred_label"] for r in subset]
        total = len(subset)
        hard = [i for i, p in enumerate(y_pred) if p != "mixed_uncertain"]
        non_mixed_truth = [i for i, t in enumerate(y_true) if t != "mixed_uncertain"]
        coverage = len(hard) / total
        ambiguity = y_pred.count("mixed_uncertain") / total
        accuracy_all = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
        accuracy_non_mixed_truth = float(np.mean([y_true[i] == y_pred[i] for i in non_mixed_truth]))
        per_label = {}
        for label in LABELS:
            idx = [i for i, t in enumerate(y_true) if t == label]
            per_label[label] = {
                "n": len(idx),
                "pred_distribution": dict(Counter(y_pred[i] for i in idx)),
                "recall": float(np.mean([y_pred[i] == label for i in idx])) if idx else 0.0,
                "uncertain_rate": float(np.mean([y_pred[i] == "mixed_uncertain" for i in idx])) if idx else 0.0,
            }
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        summary[str(patch_len)] = {
            "num_patches": total,
            "coverage_non_uncertain": float(coverage),
            "ambiguity_rate": float(ambiguity),
            "accuracy_all_including_mixed": accuracy_all,
            "accuracy_non_mixed_truth": accuracy_non_mixed_truth,
            "pred_distribution": dict(Counter(y_pred)),
            "per_true_label": per_label,
            "confusion_matrix_labels": LABELS,
            "confusion_matrix": cm.astype(int).tolist(),
            **nearest_neighbor_agreement(records, patch_len),
        }
    return summary


def run_library_smoke(records: list[dict[str, Any]]) -> dict[str, Any]:
    status: dict[str, Any] = {}
    sample32 = np.asarray([r["patch"] for r in records if r["patch_len"] == 32], dtype=np.float64)
    labels32 = [r["true_label"] for r in records if r["patch_len"] == 32]
    try:
        import stumpy

        long_series = sample32[:80].reshape(-1)
        mp = stumpy.stump(long_series, m=32)
        motif_idx = int(np.nanargmin(mp[:, 0]))
        discord_idx = int(np.nanargmax(mp[:, 0]))
        status["stumpy_matrix_profile"] = {
            "status": "ok",
            "matrix_profile_shape": list(mp.shape),
            "example_motif_index": motif_idx,
            "example_discord_index": discord_idx,
        }
    except Exception as exc:
        status["stumpy_matrix_profile"] = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}

    try:
        import ruptures as rpt

        patch = next(np.asarray(r["patch"]) for r in records if r["patch_len"] == 32 and r["true_label"] == "level_shift")
        cps = rpt.Pelt(model="l2").fit(patch).predict(pen=2.0)
        status["ruptures_change_point"] = {"status": "ok", "predicted_change_points": [int(v) for v in cps]}
    except Exception as exc:
        status["ruptures_change_point"] = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}

    try:
        from pyts.approximation import SymbolicAggregateApproximation
        from pyts.bag_of_words import BagOfWords

        sax = SymbolicAggregateApproximation(n_bins=4, strategy="quantile")
        sax_words = sax.fit_transform(sample32[:12])
        bow = BagOfWords(window_size=8, word_size=4, n_bins=4, strategy="quantile")
        bow_words = bow.transform(sample32[:12])
        status["pyts_symbolic"] = {
            "status": "ok",
            "sax_shape": list(sax_words.shape),
            "bag_of_words_examples": bow_words[:2].tolist(),
        }
    except Exception as exc:
        status["pyts_symbolic"] = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}

    try:
        from aeon.transformations.collection.shapelet_based import RandomShapeletTransform

        status["aeon_shapelet"] = {
            "status": "import_ok",
            "class": RandomShapeletTransform.__name__,
            "note": "Not fitted in this script because supervised shapelet extraction needs stable class labels first.",
        }
    except Exception as exc:
        status["aeon_shapelet"] = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}

    try:
        from tslearn.clustering import TimeSeriesKMeans

        keep = [i for i, label in enumerate(labels32[:120]) if label != "mixed_uncertain"]
        x = sample32[:120][keep][..., None]
        n_clusters = min(6, len(set(np.asarray(labels32[:120])[keep])))
        pred = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", random_state=0, n_init=2).fit_predict(x)
        status["tslearn_subsequence_clustering"] = {
            "status": "ok",
            "n_clusters": int(n_clusters),
            "cluster_distribution": dict(Counter(int(v) for v in pred)),
            "warning": "Use only as an exploratory diagnostic; subsequence clustering is not a reliable label source by itself.",
        }
    except Exception as exc:
        status["tslearn_subsequence_clustering"] = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}

    return status


def save_figures(summary: dict[str, Any]) -> list[str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    try:
        import matplotlib.pyplot as plt

        for patch_len, info in summary["patch_length_summary"].items():
            cm = np.asarray(info["confusion_matrix"], dtype=float)
            row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
            cm_norm = cm / row_sum
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"Motif taxonomy calibration, patch_len={patch_len}")
            ax.set_xticks(range(len(LABELS)))
            ax.set_yticks(range(len(LABELS)))
            ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(LABELS, fontsize=7)
            ax.set_xlabel("predicted")
            ax.set_ylabel("synthetic family")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            out = FIGURE_DIR / f"motif_taxonomy_confusion_patch{patch_len}.png"
            fig.savefig(out, dpi=180)
            plt.close(fig)
            saved.append(str(out.relative_to(ROOT)))
    except Exception as exc:
        saved.append(f"figure_generation_blocked: {type(exc).__name__}: {exc}")
    return saved


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    records = generate_dataset()
    patch_summary = summarize_records(records)
    summary = {
        "taxonomy_version": "motif_taxonomy_v0",
        "num_records": len(records),
        "labels": LABELS,
        "patch_lengths": [16, 32],
        "generation_grid": {
            "noise": [0.02, 0.08, 0.16],
            "amplitude": [0.7, 1.2, 2.0],
            "alignment": [0.0, 0.33, 0.66, 1.0],
            "samples_per_setting": 5,
        },
        "patch_length_summary": patch_summary,
        "library_smoke": run_library_smoke(records),
    }
    summary["figures"] = save_figures(summary)

    compact_records = []
    for r in records[:36]:
        compact_records.append(
            {
                "patch_len": r["patch_len"],
                "true_label": r["true_label"],
                "pred_label": r["pred_label"],
                "confidence": r["confidence"],
                "noise": r["noise"],
                "amplitude": r["amplitude"],
                "alignment": r["alignment"],
                "fired": r["fired"],
            }
        )
    summary["example_records"] = compact_records

    out = OUTPUT_DIR / "motif_taxonomy_exploration_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
