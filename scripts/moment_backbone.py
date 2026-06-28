"""Shared MOMENT representation extraction backbone (encoder-only TSFM).

跨架构泛化实验里的 encoder-only 代表（见 docs cross-architecture appendix 计划）。
MOMENT-1-large 用 ``google/flan-t5-large`` 的 **encoder-only** backbone（24 层、
d_model=1024），masked-reconstruction 预训练目标 —— 与 Chronos-Bolt（encoder-decoder、
自回归 token）和 TimesFM（decoder-only、自回归）共同覆盖三种架构族 × 两种预训练目标。

与 ``chronos_bolt_backbone`` 对齐的统一接口：
- ``tokenizer``  : ``model.patch_embedding`` 输出（pre-transformer patch embedding）
- ``layer_{i}``  : ``model.encoder.block[i]`` 的 hidden state（contextualized backbone）

关键差异（务必注意）：
- MOMENT 要求 **定长 512** 输入，patch_len=8、非重叠 → 每窗 **64 个 patch**（无 reg token）。
  因此跨模型比较时，喂给 MOMENT 的窗口必须按 512 切，再与其它模型按绝对时间对齐。
- 加载走官方 ``momentfm`` 包的 ``MOMENTPipeline``（权重是它自定义的 patch-embedding + T5
  encoder 格式，裸 transformers T5 加载不上）；``from_pretrained`` 后必须 ``model.init()``。
- 输入是 ``(N, 1, 512)``（单变量当 1 channel）；MOMENT 内部用 RevIN 做 instance norm，
  与本仓库"喂 robust-z 窗口、模型再做自己的 norm"约定一致。
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MOMENT_PATH = ROOT / "moment-1-large"
MOMENT_SEQ_LEN = 512


def default_layers(num_blocks: int) -> list[int]:
    """shallow / mid / deepest 三档，和 Chronos-Bolt default_layers 同构。"""
    return sorted({0, num_blocks // 2, num_blocks - 1})


def load_moment_pipeline(
    path: Path | str = MOMENT_PATH,
    device: str | None = None,
    task_name: str = "reconstruction",
):
    """加载本地 MOMENT pipeline（已 init + eval）。"""
    from momentfm import MOMENTPipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MOMENTPipeline.from_pretrained(
        str(path),
        local_files_only=True,
        model_kwargs={"task_name": task_name},
    )
    model.init()
    model.to(device)
    model.eval()
    return model


def moment_model_info(model) -> dict:
    return {
        "num_encoder_blocks": len(model.encoder.block),
        "patch_len": int(model.patch_len),
        "seq_len": int(model.seq_len),
        "d_model": int(model.config.d_model),
        "transformer_type": "encoder_only",
    }


@torch.no_grad()
def extract_moment_representations(
    windows: np.ndarray,
    batch_size: int = 128,
    layers: list[int] | None = None,
    include_tokenizer: bool = True,
    model=None,
    keep_model: bool = False,
) -> dict[str, np.ndarray]:
    """提取 patch_embedding + 指定 encoder 层的 patch-level representation。

    Parameters
    ----------
    windows : (N, 512) float array，已 normalize 的定长窗口（feature 0）。
              MOMENT 要求 seq_len=512；非 512 会直接报错。
    layers  : 要抓的 encoder block 索引；None -> default_layers（[0, 12, 23]）。
    include_tokenizer : 是否额外返回 ``tokenizer``（patch_embedding 输出）。

    Returns
    -------
    dict，键为 ``tokenizer`` / ``layer_{i}``，值为 (N, num_patches, d_model) 数组。
    num_patches = seq_len // patch_len = 64。
    """
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 2 or windows.shape[1] != MOMENT_SEQ_LEN:
        raise ValueError(
            f"MOMENT 需要 (N, {MOMENT_SEQ_LEN}) 定长窗口，收到 {windows.shape}。"
            " 跨模型比较时请按 512 重新切窗（再与其它模型按绝对时间对齐）。"
        )

    owns_model = model is None
    if model is None:
        model = load_moment_pipeline()

    num_blocks = len(model.encoder.block)
    if layers is None:
        layers = default_layers(num_blocks)
    patch_len = int(model.patch_len)
    num_patches = MOMENT_SEQ_LEN // patch_len
    device = next(model.parameters()).device

    layer_names = [f"layer_{idx}" for idx in layers]
    keys = (["tokenizer"] if include_tokenizer else []) + layer_names
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in keys}

    def to_patch_dim(hidden: torch.Tensor) -> torch.Tensor:
        # encoder block -> (B, P, D)；patch_embedding 可能是 (B, C, P, D)，C=1 时压掉。
        if hidden.dim() == 4:
            hidden = hidden.reshape(hidden.shape[0], -1, hidden.shape[-1])
        return hidden

    for start in range(0, len(windows), batch_size):
        batch_np = windows[start : start + batch_size]
        x = torch.tensor(batch_np, dtype=torch.float32, device=device).unsqueeze(1)  # (B,1,512)
        mask = torch.ones((x.shape[0], MOMENT_SEQ_LEN), dtype=torch.float32, device=device)

        captured: dict[str, torch.Tensor] = {}
        handles = []

        def hook_for(name: str):
            def hook(_mod, _inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                captured[name] = to_patch_dim(hidden.detach())

            return hook

        if include_tokenizer:
            handles.append(model.patch_embedding.register_forward_hook(hook_for("tokenizer")))
        for idx, name in zip(layers, layer_names):
            handles.append(model.encoder.block[idx].register_forward_hook(hook_for(name)))

        model(x_enc=x, input_mask=mask)

        for handle in handles:
            handle.remove()

        for name in keys:
            chunks[name].append(captured[name][:, :num_patches].float().cpu().numpy())

    if owns_model and not keep_model:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {k: np.concatenate(parts, axis=0) for k, parts in chunks.items()}


if __name__ == "__main__":
    # smoke: 加载模型，跑一个小随机 batch，打印各 representation 形状
    print(f"[smoke] loading MOMENT from {MOMENT_PATH}")
    m = load_moment_pipeline()
    info = moment_model_info(m)
    print("[smoke] model info:", info)

    rng = np.random.default_rng(0)
    dummy = rng.standard_normal((6, MOMENT_SEQ_LEN)).astype(np.float32)
    reps = extract_moment_representations(dummy, batch_size=4, model=m, keep_model=False)
    for k, v in reps.items():
        print(f"[smoke] {k:>10s}: {v.shape}")
    expected_patches = MOMENT_SEQ_LEN // info["patch_len"]
    assert all(v.shape[1] == expected_patches for v in reps.values()), "patch count mismatch"
    assert all(v.shape[2] == info["d_model"] for v in reps.values()), "d_model mismatch"
    print(f"[smoke] OK — {expected_patches} patches/window, d_model={info['d_model']}")
