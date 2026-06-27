"""Supplementary figure: examples of the rule-based motif labels (v0).

为论文 supplementary 生成「rule-based motif label 示例」图：每个 motif class 画一个
**真实训练 patch** 的代表性例子——直接从主图提取缓存里取出 layer-0 raw patch，用与正文
Fig. 4b ``Predefined motif labels`` 完全相同的 label_patch（patch_len = 16，对应 Chronos-Bolt）
打标签，再取每类置信度最高的那个 patch 作为示例。若某类在缓存里没有实例，则回退到仓库自带的
synthetic_patch 生成器（图注会标注 ``synthetic``）。

从仓库根目录运行：
    .venv/bin/python scripts/supp_motif_label_examples.py
默认输出到论文目录 /data/junjieqiu/TSFM_NatRevComp/fig/fig_motif_labels.pdf（+ 同名 .png 预览）。
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_bolt_main_figure import flatten_patches  # noqa: E402
from scripts.explore_motif_taxonomy import label_patch, synthetic_patch  # noqa: E402

PATCH_LEN = 16  # Chronos-Bolt
CACHE = ROOT / "outputs/figures/bolt_main_figure/.cache/extract_train_wpd200_val150_ctx128_seed47_layers0-11.pkl"

# 展示顺序 + 人类可读名（与 Fig. 4b legend 一致）
DISPLAY = [
    ("flat_low_information", "Flat / low information"),
    ("trend", "Trend"),
    ("oscillation", "Oscillation"),
    ("impulse_spike", "Impulse spike"),
    ("burst_event", "Burst event"),
    ("level_shift", "Level shift"),
    ("volatility_shift", "Volatility shift"),
    ("intermittent", "Intermittent"),
    ("mixed_uncertain", "Mixed / uncertain"),
]

# 每类一组合理的生成参数（amplitude, noise, alignment）；在其上扫 seed 直到 label_patch 判回该类
PARAMS = {
    "flat_low_information": (0.05, 0.01, 0.5),
    "trend": (2.2, 0.06, 0.5),
    "oscillation": (2.0, 0.05, 0.3),
    "impulse_spike": (5.0, 0.04, 0.5),
    "burst_event": (2.4, 0.05, 0.5),
    "level_shift": (2.2, 0.05, 0.5),
    "volatility_shift": (2.4, 0.05, 0.5),
    "intermittent": (2.2, 0.05, 0.5),
    "mixed_uncertain": (2.2, 0.05, 0.5),
}

INK = "#2b3038"
LINE = "#34506e"
FILL = "#cdd8e6"
ACCENT = "#B5403F"

FS_TITLE = 13
FS_TICK = 9


def synth_example(label: str, max_seeds: int = 400) -> np.ndarray:
    """回退：生成一个被 label_patch 判为 `label` 的合成 patch（扫 seed 求第一个命中）。"""
    amp, noise, align = PARAMS[label]
    for seed in range(max_seeds):
        rng = np.random.default_rng(seed)
        patch = synthetic_patch(label, PATCH_LEN, amp, noise, align, rng)
        if label_patch(patch, PATCH_LEN).label == label:
            return patch.astype(float)
    raise RuntimeError(f"no verified synthetic example for {label} within {max_seeds} seeds")


def collect_real_examples() -> dict[str, dict]:
    """从主图提取缓存取 layer-0 raw patch，用 label_patch 打标签，返回每类置信度最高的样本。"""
    if not CACHE.exists():
        raise SystemExit(f"extraction cache not found: {CACHE}\n先跑 build_bolt_main_figure.py 生成缓存。")
    with open(CACHE, "rb") as fh:
        b = pickle.load(fh)
    disc_z, disc_meta, disc_reps, patch_len = b["disc_z"], b["disc_meta"], b["disc_reps"], b["patch_len"]
    _emb, raw, _meta = flatten_patches(disc_reps["layer_0"], disc_z, disc_meta, patch_len)
    best: dict[str, dict] = {}
    for i in range(len(raw)):
        res = label_patch(raw[i], patch_len)
        cur = best.get(res.label)
        if cur is None or res.confidence > cur["conf"]:
            best[res.label] = {"patch": np.asarray(raw[i], dtype=float), "conf": float(res.confidence)}
    return best


def find_example(label: str, real: dict[str, dict]) -> tuple[np.ndarray, bool]:
    """优先用真实样本；缺失则回退合成。返回 (patch, is_synthetic)。"""
    if label in real:
        return real[label]["patch"], False
    return synth_example(label), True


def render(out_pdf: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "axes.edgecolor": "#9aa1ab",
        "axes.linewidth": 0.7,
    })
    real = collect_real_examples()
    fig, axes = plt.subplots(3, 3, figsize=(8.6, 6.2))
    t = np.arange(PATCH_LEN)
    for ax, (label, nice) in zip(axes.ravel(), DISPLAY):
        y, is_synth = find_example(label, real)
        color = ACCENT if label in ("impulse_spike", "mixed_uncertain") else LINE
        ax.plot(t, y, "-", color=color, lw=2.0, zorder=3)
        ax.fill_between(t, y, np.min(y), color=FILL, alpha=0.45, zorder=1)
        title = f"{nice} (synthetic)" if is_synth else nice
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", color=INK, pad=6)
        ax.set_xticks([0, PATCH_LEN - 1])
        ax.set_xticklabels(["0", str(PATCH_LEN - 1)], fontsize=FS_TICK)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.margins(x=0.02, y=0.18)
        for spn in ("top", "right"):
            ax.spines[spn].set_visible(False)
    fig.supxlabel("Time step within patch", fontsize=11, color=INK)
    fig.supylabel("Raw value", fontsize=11, color=INK)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[supp motif] saved -> {out_pdf}")
    print(f"[supp motif] preview -> {out_pdf.with_suffix('.png')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("/data/junjieqiu/TSFM_NatRevComp/fig/fig_motif_labels.pdf"))
    args = ap.parse_args()
    render(args.out)


if __name__ == "__main__":
    main()
