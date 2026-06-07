"""Task 3 figure — Chronos-Bolt forecasting probe: backbone vs tokenizer。

读取 run_bolt_forecasting_probe.py 的两个 canonical summary（horizon 16 / 64，mean-pool,
MLP probe, 4400 windows），画 depth curve：MASE 与 R² 随 representation（raw → tokenizer
→ encoder layers）变化。clean Chronos-Bolt evidence。

从仓库根目录运行：
    .venv/bin/python scripts/plot_bolt_forecasting_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "outputs" / "bolt_forecasting_probe"
FIG_DIR = ROOT / "outputs" / "figures" / "bolt_forecasting_probe"

# x 轴顺序：tokenizer (pre-transformer) -> encoder layers (contextualized)
# raw_last_patch 只在报告表格里作 AR 锚点，不进折线图（用户要求）。
REP_ORDER = ["tokenizer", "layer_0", "layer_3", "layer_6", "layer_9", "layer_11"]
REP_LABEL = {
    "tokenizer": "tokenizer\n(input embed)",
    "layer_0": "enc L0",
    "layer_3": "enc L3",
    "layer_6": "enc L6",
    "layer_9": "enc L9",
    "layer_11": "enc L11",
}
HORIZONS = [
    ("horizon16", "H=16", "#1f77b4"),
    ("horizon64", "H=64", "#d62728"),
]


def load(tag: str) -> dict:
    path = PROBE_DIR / f"bolt_forecasting_probe_{tag}_mean_mlp_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def series(summary: dict, metric: str) -> list[float]:
    return [summary["results"][r][metric] for r in REP_ORDER]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {tag: load(tag) for tag, _, _ in HORIZONS}

    xs = list(range(len(REP_ORDER)))
    xlabels = [REP_LABEL[r] for r in REP_ORDER]
    tok_x = REP_ORDER.index("tokenizer")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # --- MASE panel ---
    ax = axes[0]
    for tag, label, color in HORIZONS:
        ax.plot(xs, series(data[tag], "mase"), "-o", color=color, label=label, lw=2, ms=6)
    ax.axhline(1.0, ls="--", color="gray", lw=1)
    ax.text(0.02, 1.0, "persistence (MASE=1)", color="gray", fontsize=8, va="bottom", transform=ax.get_yaxis_transform())
    ax.axvspan(tok_x + 0.5, len(REP_ORDER) - 0.5, color="#2ca02c", alpha=0.06)
    ax.set_ylabel("MASE  (↓ better)")
    ax.set_title("Forecast error vs representation")

    # --- R2 panel ---
    ax = axes[1]
    for tag, label, color in HORIZONS:
        ax.plot(xs, series(data[tag], "r2"), "-o", color=color, label=label, lw=2, ms=6)
    ax.axhline(0.0, ls="--", color="gray", lw=1)
    ax.axvspan(tok_x + 0.5, len(REP_ORDER) - 0.5, color="#2ca02c", alpha=0.06)
    ax.set_ylabel("R²  (↑ better)")
    ax.set_title("Explained variance vs representation")

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.axvline(tok_x + 0.5, color="#2ca02c", ls=":", lw=1)
        ax.legend(loc="best", fontsize=9, title="future horizon")
        ax.grid(True, axis="y", alpha=0.25)

    n = data["horizon16"]["config"]["n_examples"]
    fig.suptitle(
        "Chronos-Bolt-base forecasting probe — contextualized backbone > tokenizer\n"
        f"frozen representation → MLP probe → predict genuine future (mean-pool over context; "
        f"{n} windows; green = transformer-contextualized)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "bolt_forecasting_probe_depth_curve.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] saved -> {out}")


if __name__ == "__main__":
    main()
