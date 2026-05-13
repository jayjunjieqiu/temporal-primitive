from __future__ import annotations

import json
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TIMESFM_SRC = ROOT / "external" / "timesfm" / "src"
CHRONOS_SRC = ROOT / "external" / "chronos-forecasting" / "src"


def make_synthetic_series(length: int = 128, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    noise = lambda scale=0.02: rng.normal(0.0, scale, size=length).astype(np.float32)

    spike = noise()
    spike[length // 2] += 2.5

    burst = noise()
    start = length // 2 - 8
    burst[start : start + 16] += np.sin(np.linspace(0, 5 * np.pi, 16)).astype(np.float32) * 1.2

    regime_shift = noise()
    regime_shift[length // 2 :] += 1.4

    intermittent = noise()
    intermittent[10::24] += 1.4
    intermittent[11::24] += 1.1

    return {
        "trend": (2.0 * t - 1.0 + noise()).astype(np.float32),
        "oscillation": (np.sin(2 * np.pi * 6 * t) + noise()).astype(np.float32),
        "spike": spike.astype(np.float32),
        "burst": burst.astype(np.float32),
        "regime_shift": regime_shift.astype(np.float32),
        "intermittent": intermittent.astype(np.float32),
    }


def z_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def patch_bank(series: dict[str, np.ndarray], patch_len: int = 32) -> tuple[np.ndarray, list[str]]:
    patches: list[np.ndarray] = []
    labels: list[str] = []
    for label, values in series.items():
        usable = len(values) // patch_len * patch_len
        for patch in values[:usable].reshape(-1, patch_len):
            patches.append(patch)
            labels.append(label)
    return np.stack(patches).astype(np.float32), labels


def nearest_patch_summary(patches: np.ndarray, labels: list[str]) -> dict[str, Any]:
    zpatches = z_normalize(patches)
    dists = np.linalg.norm(zpatches[:, None, :] - zpatches[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    nn = dists.argmin(axis=1)
    same_label = np.array([labels[i] == labels[j] for i, j in enumerate(nn)])
    top_pairs = []
    flat_order = np.argsort(dists, axis=None)
    seen: set[tuple[int, int]] = set()
    for flat_idx in flat_order:
        i, j = np.unravel_index(flat_idx, dists.shape)
        a, b = sorted((int(i), int(j)))
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        top_pairs.append(
            {
                "left": a,
                "right": b,
                "left_label": labels[a],
                "right_label": labels[b],
                "distance": float(dists[i, j]),
            }
        )
        if len(top_pairs) == 5:
            break
    return {
        "num_patches": len(labels),
        "nearest_neighbor_label_agreement": float(same_label.mean()),
        "top_pairs": top_pairs,
    }


def timesfm_extract(series: dict[str, np.ndarray]) -> dict[str, Any]:
    sys.path.insert(0, str(TIMESFM_SRC))
    import timesfm
    from timesfm.torch import util

    model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
    model.model.load_checkpoint(
        str(ROOT / "timesfm-2.5-200m-pytorch" / "model.safetensors"),
        torch_compile=False,
    )
    module = model.model
    device = module.device

    values = np.stack(list(series.values())).astype(np.float32)
    inputs = torch.tensor(values, dtype=torch.float32, device=device)
    masks = torch.zeros_like(inputs, dtype=torch.bool, device=device)

    batch_size, context = inputs.shape
    patched_inputs = torch.reshape(inputs, (batch_size, -1, module.p))
    patched_masks = torch.reshape(masks, (batch_size, -1, module.p))

    n = torch.zeros(batch_size, device=device)
    mu = torch.zeros(batch_size, device=device)
    sigma = torch.zeros(batch_size, device=device)
    patch_mu = []
    patch_sigma = []
    for i in range(context // module.p):
        (n, mu, sigma), _ = util.update_running_stats(
            n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
        )
        patch_mu.append(mu)
        patch_sigma.append(sigma)
    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)
    normed_inputs = util.revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

    capture_layers = sorted({0, len(module.stacked_xf) // 2, len(module.stacked_xf) - 1})
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def hook_for(layer_idx: int):
        def hook(_mod, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captured[f"layer_{layer_idx}"] = hidden.detach().cpu()

        return hook

    for idx in capture_layers:
        handles.append(module.stacked_xf[idx].register_forward_hook(hook_for(idx)))

    with torch.no_grad():
        (input_embeds, output_embeds, output_ts, output_quantiles), _ = module(
            normed_inputs, patched_masks
        )

    for handle in handles:
        handle.remove()

    return {
        "status": "ok",
        "device": str(device),
        "patch_len": module.p,
        "num_layers": len(module.stacked_xf),
        "input_embeddings_shape": list(input_embeds.shape),
        "final_embeddings_shape": list(output_embeds.shape),
        "point_head_shape": list(output_ts.shape),
        "quantile_head_shape": list(output_quantiles.shape),
        "captured_layer_shapes": {k: list(v.shape) for k, v in captured.items()},
    }


def chronos_probe(model_name: str) -> dict[str, Any]:
    sys.path.insert(0, str(CHRONOS_SRC))
    if importlib.util.find_spec("sklearn") is None:
        sklearn_stub = types.ModuleType("sklearn")
        preprocessing_stub = types.ModuleType("sklearn.preprocessing")
        metrics_stub = types.ModuleType("sklearn.metrics")
        sklearn_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
        preprocessing_stub.__spec__ = importlib.machinery.ModuleSpec(
            "sklearn.preprocessing", loader=None
        )
        metrics_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)

        class _UnavailableSklearnEncoder:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "scikit-learn is not installed; this stub only lets Chronos model "
                    "modules import for embedding smoke tests without categorical covariates."
                )

        preprocessing_stub.OrdinalEncoder = _UnavailableSklearnEncoder
        preprocessing_stub.TargetEncoder = _UnavailableSklearnEncoder
        metrics_stub.roc_curve = lambda *args, **kwargs: (_ for _ in ()).throw(
            ImportError("scikit-learn is not installed; roc_curve is unavailable.")
        )
        sklearn_stub.preprocessing = preprocessing_stub
        sklearn_stub.metrics = metrics_stub
        sys.modules["sklearn"] = sklearn_stub
        sys.modules["sklearn.preprocessing"] = preprocessing_stub
        sys.modules["sklearn.metrics"] = metrics_stub

    try:
        import chronos
    except Exception as exc:
        return {
            "status": "blocked",
            "stage": "import chronos",
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        pipeline = chronos.Chronos2Pipeline.from_pretrained(
            str(ROOT / model_name),
            local_files_only=True,
            device_map="cpu",
        )
        sample = next(iter(make_synthetic_series().values()))
        embeds, loc_scale = pipeline.embed([sample], batch_size=1, context_length=len(sample))
        model = pipeline.model
        capture_layers = sorted({0, len(model.encoder.block) // 2, len(model.encoder.block) - 1})
        captured: dict[str, torch.Tensor] = {}
        handles = []

        def hook_for(layer_idx: int):
            def hook(_mod, _inp, out):
                hidden = out[0] if isinstance(out, tuple) else out.hidden_states
                captured[f"layer_{layer_idx}"] = hidden.detach().cpu()

            return hook

        for idx in capture_layers:
            handles.append(model.encoder.block[idx].register_forward_hook(hook_for(idx)))

        with torch.no_grad():
            context = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
            encoder_outputs, *_ = model.encode(context=context, num_output_patches=1)

        for handle in handles:
            handle.remove()

        return {
            "status": "ok",
            "model_name": model_name,
            "patch_len": model.chronos_config.input_patch_size,
            "num_layers": len(model.encoder.block),
            "embedding_shape": list(embeds[0].shape),
            "direct_encode_last_hidden_shape": list(encoder_outputs[0].shape),
            "captured_layer_shapes": {k: list(v.shape) for k, v in captured.items()},
            "loc_scale_len": len(loc_scale),
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "model_name": model_name,
            "stage": f"load/embed {model_name}",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    series = make_synthetic_series()
    patches, labels = patch_bank(series)
    summary = {
        "synthetic_dataset": {
            "motifs": list(series.keys()),
            "num_series": len(series),
            "series_length": len(next(iter(series.values()))),
            "patch_len": 32,
        },
        "motif_discovery": nearest_patch_summary(patches, labels),
        "timesfm_2_5": timesfm_extract(series),
        "chronos_2_small": chronos_probe("chronos-2-small"),
        "chronos_2": chronos_probe("chronos-2"),
    }

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "smoke_temporal_primitives_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
