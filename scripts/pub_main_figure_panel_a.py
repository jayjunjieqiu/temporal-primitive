"""Publication panel a — "useful & contextualized with depth" 三联图（SVG）。

源数据同 plot_bolt_combined_depth.py（不重算，读已生成 summary JSON）。出版版差异：
  - SVG 矢量输出，文字保留为可编辑 text（svg.fonttype='none'），方便 PPT 里改字；
  - 字体整体放大；
  - 不画顶部 suptitle / 总描述（caption 由用户在 PPT 自己写）；
  - 文案收紧（标题、轴标签、图例）。

三个子图共享深度横轴（tokenizer → encoder L1/L4/L7/L10/L12）：
  左 forecast skill（RelMAE）· 中 context decodability（k-NN probe）· 右 within-context coherence。

从仓库根目录运行：
    .venv/bin/python scripts/pub_main_figure_panel_a.py
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
OUT_DIR = ROOT / "figure_projects" / "pub_main_figure"

REP_ORDER = ["tokenizer", "layer_0", "layer_3", "layer_6", "layer_9", "layer_11"]
REP_LABEL = {"tokenizer": "Tokenizer", "layer_0": "L1", "layer_3": "L4",
             "layer_6": "L7", "layer_9": "L10", "layer_11": "L12"}
CONF_COLOR = {"domain": "#1f77b4", "frequency": "#ff7f0e", "position": "#2ca02c"}

# 出版字号
FS_TITLE = 17
FS_LABEL = 16
FS_TICK = 13
FS_LEGEND = 13
FS_ANNOT = 12


def relmae(summary: dict) -> list[float]:
    return [summary["results"][r].get("relmae", summary["results"][r].get("mase")) for r in REP_ORDER]


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",       # 文字保留为可编辑 text（PPT 里可改）
        "axes.linewidth": 1.0,
    })

    fc16 = json.loads((FC_DIR / "bolt_forecasting_probe_horizon16_mean_mlp_summary.json").read_text())
    fc64 = json.loads((FC_DIR / "bolt_forecasting_probe_horizon64_mean_mlp_summary.json").read_text())
    ctx = json.loads(CTX_JSON.read_text())

    xs = list(range(len(REP_ORDER)))
    xlabels = [REP_LABEL[r] for r in REP_ORDER]
    tok_x = 0
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 3.8))

    # --- 左：forecast skill (RelMAE) ---
    ax = axes[0]
    ax.plot(xs, relmae(fc16), "-o", color="#1f77b4", lw=2.4, ms=7, label="H = 16")
    ax.plot(xs, relmae(fc64), "-o", color="#d62728", lw=2.4, ms=7, label="H = 64")
    ax.set_ylabel("Prediction error  ↓", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_LEGEND, title="Forecast horizon", title_fontsize=FS_LEGEND,
              loc="upper right", framealpha=0.9)

    # --- 中：confounder decodability (probe accuracy) ---
    ax = axes[1]
    for conf in ["domain", "frequency", "position"]:
        ys = [ctx["results"][r]["knn_probe_acc"][conf] for r in REP_ORDER]
        ax.plot(xs, ys, "-o", color=CONF_COLOR[conf], lw=2.4, ms=7, label=conf.capitalize())
    ax.set_ylabel("k-NN probe accuracy  ↑", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_LEGEND, title="Attribute", title_fontsize=FS_LEGEND,
              loc="lower right", framealpha=0.9)

    # --- 右：within-context similarity（双 y 轴：同/异上下文各自刻度，便于各看各的趋势）---
    ax = axes[2]
    same = [ctx["results"][r]["within_context_sim"]["same_context"] for r in REP_ORDER]
    diff = [ctx["results"][r]["within_context_sim"]["different_context"] for r in REP_ORDER]
    SAME_C, DIFF_C = "#d62728", "#595959"
    ax2 = ax.twinx()
    ax2.patch.set_visible(False)
    ax.plot(xs, same, "-o", color=SAME_C, lw=2.4, ms=7)
    ax2.plot(xs, diff, "--s", color=DIFF_C, lw=2.4, ms=6)
    # 两个 y 轴各用对应线的颜色标注，明确告诉读者这是两个不同刻度
    ax.set_ylabel("Same-context similarity", fontsize=FS_LABEL, color=SAME_C)
    ax2.set_ylabel("Different-context similarity", fontsize=FS_LABEL, color=DIFF_C)
    ax.tick_params(axis="y", labelcolor=SAME_C)
    ax2.tick_params(axis="y", labelcolor=DIFF_C, labelsize=FS_TICK)
    ax.spines["left"].set_color(SAME_C)
    ax2.spines["right"].set_color(DIFF_C)
    ax2.spines["left"].set_visible(False)
    ax2.grid(False)
    # 不再需要 legend：左右 y 轴已用对应线的颜色标注（红=same、灰=different）

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=FS_TICK)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.axvline(tok_x + 0.5, color="#2ca02c", ls=":", lw=1.2)
        ax.axvspan(tok_x + 0.5, len(REP_ORDER) - 0.5, color="#2ca02c", alpha=0.06)
        ax.grid(True, axis="y", alpha=0.25)

    # 三联横轴含义相同，整行底部只写一次（各子图刻度保留）。双轴图在右端，右标签贴图边、
    # 不占图间空隙，三联用默认 tight_layout 即均匀，无需额外补偿。
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.supxlabel("Depth   (Tokenizer → encoder layer)", fontsize=FS_LABEL, y=0.03)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / "panel_a_depth.svg"
    png = OUT_DIR / "panel_a_depth.png"   # 仅供肉眼校对
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print(f"[panel a] saved -> {svg}")
    print(f"[panel a] preview -> {png}")


if __name__ == "__main__":
    main()
