"""Shared Chronos-Bolt representation extraction backbone.

迁移自 Chronos-2 archived pilot 的 clean 后继模型（见
`docs/99_chronos2_archive_and_chronos_bolt_pivot.md`）。Chronos-Bolt 的
`input_patch_embedding` 只吃 `[normalized patch values, patch mask]`，不含 time
encoding，因此是 pure value-only patch token —— 这正是相对 Chronos-2 的 clean 性质。

本模块给三个下游任务（forecasting probe / contextualization figure / main figure）
提供统一的 representation 提取：
- ``tokenizer``  : ``model.input_patch_embedding`` 输出（pre-transformer，value-only）
- ``layer_{i}``  : ``model.encoder.block[i]`` 的 hidden state（contextualized backbone）

复用本仓库 backbone 约定：从仓库根目录调用，运行时把 vendored chronos src 加进
sys.path，加载本地权重（local_files_only），用 forward hook 抓 hidden state，结束后
释放显存。
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHRONOS_SRC = ROOT / "external" / "chronos-forecasting" / "src"
BOLT_PATH = ROOT / "chronos-bolt-base"


def default_layers(num_blocks: int) -> list[int]:
    """shallow / mid / deepest 三档，和旧 Chronos-2 [0, 6, 11] 同构。"""
    return sorted({0, num_blocks // 2, num_blocks - 1})


def load_bolt_pipeline(path: Path | str = BOLT_PATH, device: str | None = None):
    """加载本地 Chronos-Bolt pipeline。"""
    if str(CHRONOS_SRC) not in sys.path:
        sys.path.insert(0, str(CHRONOS_SRC))
    import chronos

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = chronos.ChronosBoltPipeline.from_pretrained(
        str(path),
        local_files_only=True,
        device_map=device,
    )
    pipeline.model.eval()
    return pipeline


def bolt_model_info(pipeline) -> dict:
    model = pipeline.model
    return {
        "num_encoder_blocks": len(model.encoder.block),
        "patch_len": int(model.chronos_config.input_patch_size),
        "patch_stride": int(model.chronos_config.input_patch_stride),
        "d_model": int(model.model_dim),
        "use_reg_token": bool(model.chronos_config.use_reg_token),
        "context_length": int(model.chronos_config.context_length),
        "prediction_length": int(model.chronos_config.prediction_length),
    }


@torch.no_grad()
def extract_bolt_representations(
    windows: np.ndarray,
    batch_size: int = 128,
    layers: list[int] | None = None,
    include_tokenizer: bool = True,
    pipeline=None,
    keep_pipeline: bool = False,
) -> dict[str, np.ndarray]:
    """提取 tokenizer + 指定 encoder 层的 patch-level representation。

    Parameters
    ----------
    windows : (N, context_len) float array，已 normalize 的窗口（feature 0）。
    layers  : 要抓的 encoder block 索引；None -> default_layers。
    include_tokenizer : 是否额外返回 ``tokenizer`` （input_patch_embedding 输出）。

    Returns
    -------
    dict，键为 ``tokenizer`` / ``layer_{i}``，值为 (N, num_patches, d_model) 数组。
    每个窗口的 patch 数 = context_len // patch_len（不含 REG token，已切掉）。
    """
    owns_pipeline = pipeline is None
    if pipeline is None:
        pipeline = load_bolt_pipeline()
    model = pipeline.model

    num_blocks = len(model.encoder.block)
    if layers is None:
        layers = default_layers(num_blocks)
    patch_len = int(model.chronos_config.input_patch_size)

    layer_names = [f"layer_{idx}" for idx in layers]
    keys = (["tokenizer"] if include_tokenizer else []) + layer_names
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in keys}

    for start in range(0, len(windows), batch_size):
        batch = torch.tensor(
            windows[start : start + batch_size], dtype=torch.float32, device=model.device
        )
        # 每个窗口的 patch 数（左 padding 只在 context_len 不整除时发生；这里整除）
        num_patches = batch.shape[1] // patch_len

        captured: dict[str, torch.Tensor] = {}
        handles = []

        def hook_for(layer_idx: int):
            def hook(_mod, _inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                captured[f"layer_{layer_idx}"] = hidden.detach()

            return hook

        for layer_idx in layers:
            handles.append(model.encoder.block[layer_idx].register_forward_hook(hook_for(layer_idx)))

        # encode 返回 (encoder_last_hidden, loc_scale, input_embeds, attention_mask)
        _hidden, _loc_scale, input_embeds, _attn = model.encode(context=batch)

        for handle in handles:
            handle.remove()

        if include_tokenizer:
            chunks["tokenizer"].append(input_embeds[:, :num_patches].float().cpu().numpy())
        for idx, name in zip(layers, layer_names):
            chunks[name].append(captured[name][:, :num_patches].float().cpu().numpy())

    if owns_pipeline and not keep_pipeline:
        del pipeline, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {k: np.concatenate(parts, axis=0) for k, parts in chunks.items()}


if __name__ == "__main__":
    # smoke: 加载模型，跑一个小随机 batch，打印各 representation 形状
    print(f"[smoke] loading Chronos-Bolt from {BOLT_PATH}")
    pipe = load_bolt_pipeline()
    info = bolt_model_info(pipe)
    print("[smoke] model info:", info)

    rng = np.random.default_rng(0)
    ctx_len = 128
    dummy = rng.standard_normal((6, ctx_len)).astype(np.float32)
    reps = extract_bolt_representations(
        dummy, batch_size=4, pipeline=pipe, keep_pipeline=False
    )
    for k, v in reps.items():
        print(f"[smoke] {k:>10s}: {v.shape}")
    expected_patches = ctx_len // info["patch_len"]
    assert all(v.shape[1] == expected_patches for v in reps.values()), "patch count mismatch"
    print(f"[smoke] OK — {expected_patches} patches/window, d_model={info['d_model']}")
