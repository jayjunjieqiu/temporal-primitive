"""Cross-architecture shared-pool sampling + unified representation dispatch.

跨架构泛化附录的基础设施。核心设计（见对应 cross-architecture appendix 计划）：

* **共享池**：三个模型吃 **同一批 512 点真实时间窗**（同一段绝对时间跨度），各自按
  自己的 patch_len 切 —— Bolt 16→32 patch、TimesFM 32→16 patch、MOMENT 8→64 patch。
  512 是 MOMENT 的硬约束（定长），正好当公共时间跨度；Bolt(max ctx 2048) / TimesFM
  都能吃下。唯一变量是模型本身 → apples-to-apples 的 intrinsic-signature 对照。
* **去 OOD 化**：只比"每个模型各自内部随深度的上下文化签名形状"，不比绝对数值、不比
  cluster 身份、不声称泛化到训练外（常见 benchmark 对 MOMENT 全是 in-distribution）。
* **统一 dispatch**：把三个已验证的抽取砖块（``extract_bolt_representations`` /
  ``extract_timesfm_layers`` / ``extract_moment_representations``）挂到同一入口，返回
  统一的 ``{tokenizer?, layer_{i}...}`` -> (N, num_patches, d_model)。

约定：从仓库根目录调用（``ROOT`` 在 sys.path 上其它 scripts 才可 import）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_second_pilot_discovery import (  # noqa: E402
    DATA_ROOT,
    robust_z,
    sample_windows,
    select_domain_balanced_indices,
)

# 共享时间跨度：MOMENT 定长 512；Bolt / TimesFM 都能吃下。三模型同窗、各自切 patch。
SHARED_WINDOW_LEN = 512

# 跨架构对照的三个模型。layers = shallow / mid / deepest（相对深度对齐，非绝对层号）。
MODELS: dict[str, dict[str, Any]] = {
    "chronos_bolt": {
        "kind": "bolt",
        "family": "encoder-decoder",
        "objective": "autoregressive token",
        "patch_len": 16,
        "num_patches": SHARED_WINDOW_LEN // 16,  # 32
        "d_model": 768,
        "layers": [0, 6, 11],  # 12 encoder blocks
    },
    "timesfm_2_5": {
        "kind": "timesfm",
        "family": "decoder-only",
        "objective": "autoregressive",
        "patch_len": 32,
        "num_patches": SHARED_WINDOW_LEN // 32,  # 16
        "d_model": 1280,
        "layers": [0, 10, 19],  # 20 stacked transformer layers
    },
    "moment_1_large": {
        "kind": "moment",
        "family": "encoder-only",
        "objective": "masked reconstruction",
        "patch_len": 8,
        "num_patches": SHARED_WINDOW_LEN // 8,  # 64
        "d_model": 1024,
        "layers": [0, 12, 23],  # 24 encoder blocks
    },
}


def build_shared_pool(
    data_root: Path = DATA_ROOT,
    window_len: int = SHARED_WINDOW_LEN,
    windows_per_dataset: int = 200,
    seed: int = 13,
    max_per_domain: int | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """采一批跨域 512 点窗，供三个模型共享。

    Parameters
    ----------
    windows_per_dataset : 每个数据集采样窗数（短于 window_len 的数据集自动跳过）。
    max_per_domain      : 若给定，做 domain-balanced 下采样（复用 pilot 逻辑）。
    normalize           : 每窗 robust-z（各模型仍会再做自己的 instance norm）。

    Returns
    -------
    (windows, metadata, dataset_summary)
        windows : (N, window_len) float32，三个模型共享的同一批输入。
        metadata: 每窗 dataset/domain/frequency/node/start。
    """
    windows, metadata, dataset_summary = sample_windows(
        data_root=data_root,
        context_len=window_len,
        windows_per_dataset=windows_per_dataset,
        seed=seed,
    )

    if max_per_domain is not None:
        idx = select_domain_balanced_indices(metadata, max_per_domain=max_per_domain, seed=seed)
        windows = windows[idx]
        metadata = [metadata[i] for i in idx]
        for new_id, m in enumerate(metadata):
            m["window_id"] = new_id

    if normalize:
        windows = np.stack([robust_z(w) for w in windows]).astype(np.float32)

    return windows, metadata, dataset_summary


def _bolt_extractor(windows: np.ndarray, layers: list[int], batch_size: int) -> dict[str, np.ndarray]:
    from scripts.chronos_bolt_backbone import extract_bolt_representations

    return extract_bolt_representations(windows, batch_size=batch_size, layers=layers)


def _moment_extractor(windows: np.ndarray, layers: list[int], batch_size: int) -> dict[str, np.ndarray]:
    from scripts.moment_backbone import extract_moment_representations

    return extract_moment_representations(windows, batch_size=batch_size, layers=layers)


def _timesfm_extractor(windows: np.ndarray, layers: list[int], batch_size: int) -> dict[str, np.ndarray]:
    # 复用归档 pilot 路径里已验证的 TimesFM 抽取（layers 取自 MODEL_SPECS["timesfm_2_5"]）。
    from scripts.run_second_pilot_discovery import MODEL_SPECS, extract_timesfm_layers

    MODEL_SPECS["timesfm_2_5"]["layers"] = layers
    return extract_timesfm_layers(windows, batch_size)


_EXTRACTORS: dict[str, Callable[[np.ndarray, list[int], int], dict[str, np.ndarray]]] = {
    "bolt": _bolt_extractor,
    "moment": _moment_extractor,
    "timesfm": _timesfm_extractor,
}


def extract_for_model(
    model_key: str,
    windows: np.ndarray,
    batch_size: int = 64,
    layers: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """统一入口：对共享池窗口跑指定模型，返回 {layer_{i}...} -> (N, num_patches, d_model)。"""
    if model_key not in MODELS:
        raise KeyError(f"unknown model_key {model_key!r}; choices: {list(MODELS)}")
    spec = MODELS[model_key]
    resolved_layers: list[int] = list(spec["layers"] if layers is None else layers)
    windows = np.asarray(windows, dtype=np.float32)
    return _EXTRACTORS[spec["kind"]](windows, resolved_layers, batch_size)


def extract_all_models(
    windows: np.ndarray,
    batch_size: int = 64,
    model_keys: list[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """对同一批共享池窗口顺序跑三个模型（每个跑完释放显存）。"""
    model_keys = model_keys or list(MODELS)
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in model_keys:
        out[key] = extract_for_model(key, windows, batch_size=batch_size)
    return out


if __name__ == "__main__":
    # smoke：搭一个小共享池（每集 2 窗），三个模型各跑一遍，打印形状 + 基本健全性。
    print(f"[smoke] building shared pool (window_len={SHARED_WINDOW_LEN}) from {DATA_ROOT}")
    win, meta, summary = build_shared_pool(windows_per_dataset=2, seed=13)
    domains = sorted({m["domain"] for m in meta})
    print(f"[smoke] pool: {win.shape} windows over {len(domains)} domains: {domains}")
    assert win.shape[1] == SHARED_WINDOW_LEN
    assert np.isfinite(win).all(), "non-finite in shared pool"

    for key, spec in MODELS.items():
        reps = extract_for_model(key, win, batch_size=16)
        shapes = {k: v.shape for k, v in reps.items()}
        print(f"[smoke] {key:>16s} ({spec['family']:>16s}): {shapes}")
        for k, v in reps.items():
            assert v.shape[0] == win.shape[0], f"{key}/{k} N mismatch"
            assert v.shape[1] == spec["num_patches"], f"{key}/{k} num_patches != {spec['num_patches']}"
            assert np.isfinite(v).all(), f"{key}/{k} non-finite reps"
    print("[smoke] OK — all three models share one 512-window pool; patch counts 32/16/64 as expected")
