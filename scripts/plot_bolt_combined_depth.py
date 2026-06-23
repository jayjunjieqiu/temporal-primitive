"""合并 depth 图：forecasting skill（probe）+ confounder decodability + within-context similarity。

三个面板共享同一条深度横轴（tokenizer → encoder L1/L4/L7/L10/L12），等大并排：
  左   = forecast skill（RelMAE vs persistence；来自 forecasting probe，docs/13）
  中   = confounder decodability（10-NN probe accuracy；docs/14 §2.1）
  右   = within-context patch similarity（centered cosine；docs/14 §2.1）

读已生成的 summary JSON，不重算。注意：左面板（forecasting）在 basicts 上算，中/右（contextualization）
在训练子集上算——三者都是"随 depth 变化"的独立指标，横轴含义一致，但不是同一份数据（图注已注明）。

从仓库根目录运行：
    .venv/bin/python scripts/plot_bolt_combined_depth.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/temporal_primitive_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FC_DIR = ROOT / "outputs" / "bolt_forecasting_probe"
CTX_JSON = ROOT / "outputs" / "bolt_contextualization" / "bolt_contextualization_training_summary.json"
OUT = ROOT / "outputs" / "figures" / "bolt_main_figure" / "main_E_useful_contextualized_depth.png"

REP_ORDER = ["tokenizer", "layer_0", "layer_3", "layer_6", "layer_9", "layer_11"]
REP_LABEL = {"tokenizer": "tokenizer\n(input embed)", "layer_0": "enc L1", "layer_3": "enc L4",
             "layer_6": "enc L7", "layer_9": "enc L10", "layer_11": "enc L12"}
CONF_COLOR = {"domain": "#1f77b4", "frequency": "#ff7f0e", "position": "#2ca02c"}


def relmae(summary: dict) -> list[float]:
    return [summary["results"][r].get("relmae", summary["results"][r].get("mase")) for r in REP_ORDER]


def main() -> None:
    fc16 = json.loads((FC_DIR / "bolt_forecasting_probe_horizon16_mean_mlp_summary.json").read_text())
    fc64 = json.loads((FC_DIR / "bolt_forecasting_probe_horizon64_mean_mlp_summary.json").read_text())
    ctx = json.loads(CTX_JSON.read_text())

    xs = list(range(len(REP_ORDER)))
    xlabels = [REP_LABEL[r] for r in REP_ORDER]
    tok_x = 0
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # --- 左：forecast skill (RelMAE) ---
    ax = axes[0]
    ax.plot(xs, relmae(fc16), "-o", color="#1f77b4", lw=2, ms=5, label="H=16")
    ax.plot(xs, relmae(fc64), "-o", color="#d62728", lw=2, ms=5, label="H=64")
    ax.axhline(1.0, ls="--", color="gray", lw=1)
    ax.text(0.02, 1.0, "persistence (=1)", color="gray", fontsize=8, va="bottom",
            transform=ax.get_yaxis_transform())
    ax.set_ylabel("RelMAE vs persistence  (↓ better)")
    ax.set_title("Forecast skill vs depth\nRelMAE (<1 beats naive)", fontsize=10)
    ax.legend(fontsize=8, title="future horizon")

    # --- 中：confounder decodability (probe accuracy) ---
    ax = axes[1]
    for conf in ["domain", "frequency", "position"]:
        ys = [ctx["results"][r]["knn_probe_acc"][conf] for r in REP_ORDER]
        ax.plot(xs, ys, "-o", color=CONF_COLOR[conf], lw=2, ms=5, label=conf)
        ax.axhline(ctx["chance"][conf], ls=":", color=CONF_COLOR[conf], lw=1, alpha=0.6)
    ax.set_ylabel("10-NN probe accuracy  (↑ more decodable)")
    ax.set_title("Confounder decodability vs depth\n(dotted = chance)", fontsize=10)
    ax.legend(fontsize=8, title="confounder")

    # --- 右：within-context similarity ---
    ax = axes[2]
    same = [ctx["results"][r]["within_context_sim"]["same_context"] for r in REP_ORDER]
    diff = [ctx["results"][r]["within_context_sim"]["different_context"] for r in REP_ORDER]
    ax.plot(xs, same, "-o", color="#d62728", lw=2, ms=5, label="same context\n(diff-position patches)")
    ax.plot(xs, diff, "-o", color="#7f7f7f", lw=2, ms=5, label="different context\n(random patches)")
    ax.fill_between(xs, diff, same, color="#d62728", alpha=0.12)
    ax.set_ylabel("centered cosine similarity")
    ax.set_title("Within-context patch similarity vs depth", fontsize=10)
    ax.legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.axvline(tok_x + 0.5, color="#2ca02c", ls=":", lw=1)
        ax.axvspan(tok_x + 0.5, len(REP_ORDER) - 0.5, color="#2ca02c", alpha=0.06)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Chronos-Bolt — representations become useful & contextualized with depth\n"
        "forecast skill (left; basicts) + confounder decodability (middle) + within-context similarity "
        "(right; training data); x = depth (tokenizer → encoder layers)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT, dpi=150)
    print(f"[combined] saved -> {OUT}")


if __name__ == "__main__":
    main()
