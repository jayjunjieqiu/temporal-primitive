"""Preview typical sampled time-series windows from curated Chronos *training*
datasets, grouped by macro domain.

用途：在把 primitive discovery 从 basicts 测试集迁到 Chronos in-distribution 训练数据之前，
先肉眼看一下我们挑的这些 (macro_domain, dataset) 采样出来的典型时序片段长什么样。

数据：/data/ts-datasets/chronos_datasets/<ds>/*.parquet（每行一条 series：target/timestamp/id）。
采样：每个数据集随机抽 N 条 series、各取一个长度 L 的 finite 窗口，robust-z normalize。
画图：每行一个数据集（左侧色块=macro domain），每列一个随机窗口。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.chronos_training_data import (  # noqa: E402
    DOMAIN_COLORS,
    TRAINING_DATASETS as CURATED,
    load_series_arrays,
)


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = np.std(x)
    if scale < 1e-8:
        scale = 1.0
    z = (x - med) / scale
    return np.clip(z, -10, 10)


def sample_windows_from_series(series_list, n_windows, length, rng, max_tries=400):
    """从一组 series 里随机抽 n_windows 个长度 length 的 finite 窗口，robust-z 后返回。"""
    out = []
    tries = 0
    n_series = len(series_list)
    if n_series == 0:
        return out
    while len(out) < n_windows and tries < max_tries:
        tries += 1
        s = series_list[rng.integers(n_series)]
        if s.shape[0] < length:
            continue
        start = int(rng.integers(0, s.shape[0] - length + 1))
        w = s[start : start + length].astype(np.float64)
        finite = np.isfinite(w)
        if finite.mean() < 0.95:
            continue
        if not finite.all():  # 插值少量 NaN
            idx = np.arange(length)
            w = np.interp(idx, idx[finite], w[finite])
        if np.std(w) < 1e-9:  # 跳过常数窗口
            continue
        out.append(robust_z(w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-per-dataset", type=int, default=5)
    ap.add_argument("--length", type=int, default=192)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "outputs/figures/training_preview/training_dataset_samples.png"),
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ncol = args.windows_per_dataset
    rows = CURATED

    fig_h = 1.05 * len(rows)
    fig_w = 1.7 * ncol + 2.2
    fig, axes = plt.subplots(
        len(rows), ncol, figsize=(fig_w, fig_h), squeeze=False
    )
    fig.subplots_adjust(left=0.20, right=0.985, top=0.965, bottom=0.025, hspace=0.45, wspace=0.18)

    for ri, (domain, ds, disp) in enumerate(rows):
        color = DOMAIN_COLORS.get(domain, "#999999")
        series = load_series_arrays(ds)
        wins = sample_windows_from_series(series, ncol, args.length, rng)
        n_series = len(series)
        for ci in range(ncol):
            ax = axes[ri][ci]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#cccccc")
            if ci < len(wins):
                ax.plot(wins[ci], color=color, lw=1.0)
                ax.set_ylim(-3.5, 3.5)
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        color="#bbbbbb", transform=ax.transAxes)
        # 左侧 domain 色块 + 数据集名
        ax0 = axes[ri][0]
        ax0.text(
            -0.62, 0.5, disp, transform=ax0.transAxes,
            ha="right", va="center", fontsize=9, fontweight="bold",
        )
        ax0.text(
            -0.62, 0.04, f"{domain} · n={n_series}", transform=ax0.transAxes,
            ha="right", va="center", fontsize=6.5, color="#666666",
        )
        # domain 色条
        fig.patches.append(
            Rectangle(
                (0.012, axes[ri][0].get_position().y0),
                0.012, axes[ri][0].get_position().height,
                transform=fig.transFigure, color=color, clip_on=False,
            )
        )

    fig.suptitle(
        f"Curated Chronos training datasets — typical robust-z windows (L={args.length})",
        fontsize=11, y=0.99,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[preview] saved -> {out}")
    print(f"[preview] {len(rows)} datasets x {ncol} windows, length={args.length}")


if __name__ == "__main__":
    main()
