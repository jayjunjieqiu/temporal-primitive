#!/usr/bin/env python3
"""Feasibility smoke test for temporal primitive patch analysis.

This script:
- builds a tiny synthetic calibration set with labeled motifs
- runs a brute-force motif discovery routine (matrix-profile style)
- loads Chronos-2-small, Chronos-2, and TimesFM-2.5 from local weights
- extracts patch embeddings and selected-layer hidden states
- writes a compact JSON summary under artifacts/feasibility/
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "feasibility"
DEFAULT_CHRONOS_SRC = Path(os.environ.get("CHRONOS_SRC", "/tmp/chronos-forecasting/src"))
DEFAULT_TIMESFM_SRC = Path(os.environ.get("TIMESFM_SRC", "/tmp/timesfm/src"))


def _ensure_source_paths() -> None:
    for path in (DEFAULT_CHRONOS_SRC, DEFAULT_TIMESFM_SRC):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _install_sklearn_stubs() -> None:
    """Chronos imports sklearn at module import time; create a tiny stub if absent."""

    if importlib.util.find_spec("sklearn") is not None:
        return

    sklearn = types.ModuleType("sklearn")
    sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None, is_package=True)
    sklearn.__path__ = []  # type: ignore[attr-defined]

    preprocessing = types.ModuleType("sklearn.preprocessing")
    preprocessing.__spec__ = importlib.machinery.ModuleSpec("sklearn.preprocessing", loader=None)

    metrics = types.ModuleType("sklearn.metrics")
    metrics.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)

    class _DummyEncoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    def _roc_curve(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise RuntimeError("sklearn.metrics.roc_curve stub should never be called in the smoke test")

    preprocessing.OrdinalEncoder = _DummyEncoder
    preprocessing.TargetEncoder = _DummyEncoder
    metrics.roc_curve = _roc_curve

    sklearn.preprocessing = preprocessing  # type: ignore[attr-defined]
    sklearn.metrics = metrics  # type: ignore[attr-defined]

    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.preprocessing"] = preprocessing
    sys.modules["sklearn.metrics"] = metrics


def _znorm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-8:
        return np.zeros_like(x)
    return (x - mu) / sigma


def _synthetic_series(length: int = 128, seed: int = 7) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32)

    series: list[np.ndarray] = []
    labels: list[str] = []
    events: list[list[tuple[int, int]]] = []

    def add(label: str, x: np.ndarray, event_spans: list[tuple[int, int]]) -> None:
        series.append(x.astype(np.float32))
        labels.append(label)
        events.append(event_spans)

    add(
        "trend",
        0.02 * t + 0.12 * np.sin(2 * np.pi * t / 64.0) + rng.normal(0, 0.03, size=length),
        [(0, length)],
    )
    add(
        "oscillation",
        1.2 * np.sin(2 * np.pi * t / 16.0) + 0.25 * np.sin(2 * np.pi * t / 8.0)
        + rng.normal(0, 0.04, size=length),
        [(0, length)],
    )
    spike = rng.normal(0, 0.05, size=length)
    spike += 3.5 * np.exp(-0.5 * ((t - 48.0) / 1.8) ** 2)
    add("spike", spike, [(40, 56)])
    burst = rng.normal(0, 0.04, size=length)
    burst[44:84] += 0.8 * np.sin(2 * np.pi * np.arange(40) / 3.0) + 0.35 * rng.normal(0, 1, size=40)
    add("burst", burst, [(44, 84)])
    regime = np.where(t < 64, -1.0 + 0.04 * np.sin(2 * np.pi * t / 24.0), 1.1 + 0.04 * np.sin(2 * np.pi * t / 24.0))
    regime = regime + rng.normal(0, 0.03, size=length)
    add("regime_shift", regime, [(56, 72)])
    intermittent = rng.normal(0, 0.05, size=length)
    for c in (16, 40, 64, 88, 112):
        intermittent += 1.8 * np.exp(-0.5 * ((t - c) / 1.4) ** 2)
    add("intermittent", intermittent, [(12, 20), (36, 44), (60, 68), (84, 92), (108, 116)])

    data = np.stack(series, axis=0)
    return {"series": data, "labels": labels, "events": events}


def _patch_labels_for_series(label: str, event_spans: list[tuple[int, int]], length: int, patch_len: int) -> list[str]:
    n_patches = math.ceil(length / patch_len)
    labels = []
    for idx in range(n_patches):
        start = idx * patch_len
        end = min(length, start + patch_len)
        if label in {"trend", "oscillation"}:
            labels.append(label)
            continue
        overlap = 0
        for a, b in event_spans:
            overlap += max(0, min(end, b) - max(start, a))
        labels.append(label if overlap > 0 else "background")
    return labels


def _build_patch_label_bank(data: dict[str, Any], patch_len: int) -> list[list[str]]:
    series = data["series"]
    labels = data["labels"]
    events = data["events"]
    return [
        _patch_labels_for_series(label, spans, len(series[i]), patch_len)
        for i, (label, spans) in enumerate(zip(labels, events))
    ]


def _brute_force_motifs(series: np.ndarray, window: int = 16, top_k: int = 3) -> list[dict[str, Any]]:
    series = np.asarray(series, dtype=np.float32)
    if len(series) < window * 2:
        window = max(8, len(series) // 4)
    subseqs = np.stack([_znorm(series[i : i + window]) for i in range(len(series) - window + 1)], axis=0)

    pairs: list[dict[str, Any]] = []
    for i in range(len(subseqs)):
        for j in range(i + 1, len(subseqs)):
            if abs(i - j) < window // 2:
                continue
            dist = float(np.linalg.norm(subseqs[i] - subseqs[j]))
            pairs.append({"distance": dist, "i": i, "j": j})
    pairs.sort(key=lambda x: x["distance"])
    return pairs[:top_k]


def _tensor_summary(t: torch.Tensor) -> dict[str, Any]:
    t = t.detach().cpu()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "mean": float(t.float().mean().item()),
        "std": float(t.float().std(unbiased=False).item()),
    }


def _chronos_extract(model_dir: Path, batch: np.ndarray) -> dict[str, Any]:
    _install_sklearn_stubs()
    from chronos import Chronos2Model

    model = Chronos2Model.from_pretrained(str(model_dir))
    model.eval()

    selected_layers = sorted({0, max(0, model.config.num_layers // 2), model.config.num_layers - 1})
    captures: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles = []

    def register(name: str, module: torch.nn.Module) -> None:
        def _hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output.hidden_states if hasattr(output, "hidden_states") else output
            if isinstance(tensor, (tuple, list)):
                tensor = tensor[0]
            captures[name].append(tensor.detach().cpu())

        handles.append(module.register_forward_hook(_hook))

    register("patch_embedding", model.input_patch_embedding)
    for idx in selected_layers:
        register(f"layer_{idx}", model.encoder.block[idx])

    x = torch.from_numpy(batch).to(torch.float32)
    with torch.no_grad():
        encoder_outputs, _loc_scale, _pf_mask, num_context_patches = model.encode(x, num_output_patches=1)

    for handle in handles:
        handle.remove()

    patch_context = captures["patch_embedding"][0]
    final_hidden = encoder_outputs.last_hidden_state[:, :num_context_patches, :]
    layer_outputs = {
        name: captures[name][0][:, :num_context_patches, :]
        for name in captures
        if name.startswith("layer_")
    }

    result = {
        "model_dir": str(model_dir),
        "num_context_patches": int(num_context_patches),
        "selected_layers": selected_layers,
        "patch_embedding": _tensor_summary(patch_context),
        "final_context_hidden": _tensor_summary(final_hidden),
        "layer_outputs": {k: _tensor_summary(v) for k, v in layer_outputs.items()},
        "raw_last_hidden_shape": list(encoder_outputs.last_hidden_state.shape),
    }
    del model
    return result


def _timesfm_extract(model_dir: Path, batch: np.ndarray) -> dict[str, Any]:
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(model_dir), torch_compile=False)
    model.model.eval()

    selected_layers = sorted({0, max(0, model.model.x // 2), model.model.x - 1})
    captures: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles = []

    def register(name: str, module: torch.nn.Module) -> None:
        def _hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            captures[name].append(tensor.detach().cpu())

        handles.append(module.register_forward_hook(_hook))

    for idx in selected_layers:
        register(f"layer_{idx}", model.model.stacked_xf[idx])

    patch_len = model.model.p
    batch = np.asarray(batch, dtype=np.float32)
    n_patches = batch.shape[1] // patch_len
    inputs = torch.from_numpy(batch.reshape(batch.shape[0], n_patches, patch_len)).to(model.model.device)
    masks = torch.zeros_like(inputs, dtype=torch.bool)

    with torch.no_grad():
        (input_embeddings, output_embeddings, output_ts, output_quantile_spread), _caches = model.model(
            inputs, masks
        )

    for handle in handles:
        handle.remove()

    result = {
        "model_dir": str(model_dir),
        "patch_len": patch_len,
        "num_input_patches": int(n_patches),
        "selected_layers": selected_layers,
        "input_embeddings": _tensor_summary(input_embeddings),
        "final_hidden": _tensor_summary(output_embeddings),
        "forecast_head": _tensor_summary(output_ts),
        "quantile_head": _tensor_summary(output_quantile_spread),
        "layer_outputs": {k: _tensor_summary(v[0]) for k, v in captures.items()},
    }
    del model
    return result


def main() -> int:
    _ensure_source_paths()

    data = _synthetic_series()
    patch_labels_16 = _build_patch_label_bank(data, patch_len=16)
    patch_labels_32 = _build_patch_label_bank(data, patch_len=32)

    artifacts = {
        "series": data["series"],
        "series_labels": data["labels"],
        "event_spans": data["events"],
        "patch_labels_16": patch_labels_16,
        "patch_labels_32": patch_labels_32,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(ARTIFACT_DIR / "synthetic_calibration.npz", **{k: np.array(v, dtype=object) for k, v in artifacts.items()})

    motif_results = {
        label: _brute_force_motifs(series, window=16, top_k=3)
        for label, series in zip(data["labels"], data["series"])
    }
    with (ARTIFACT_DIR / "motif_pairs.json").open("w", encoding="utf-8") as f:
        json.dump(motif_results, f, indent=2)

    chronos_small = _chronos_extract(ROOT / "chronos-2-small", data["series"][:2])
    chronos_full = _chronos_extract(ROOT / "chronos-2", data["series"][:2])
    timesfm = _timesfm_extract(ROOT / "timesfm-2.5-200m-pytorch", data["series"][:2])

    summary = {
        "synthetic_dataset": {
            "num_series": int(data["series"].shape[0]),
            "length": int(data["series"].shape[1]),
            "series_labels": data["labels"],
        },
        "motif_discovery": {
            "method": "brute-force z-normalized self-join",
            "window": 16,
            "top_pairs": motif_results,
        },
        "chronos_2_small": chronos_small,
        "chronos_2": chronos_full,
        "timesfm_2_5": timesfm,
    }

    with (ARTIFACT_DIR / "smoke_results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
