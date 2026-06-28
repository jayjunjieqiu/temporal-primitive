"""跨架构泛化附录两张图的共享 style / 术语 —— 与 pub_main_figure / pub_ood_figure 统一。

统一项（硬约束，见各 pub_* 脚本）：
- 不画 suptitle（caption 用户在 PPT 自写）。
- 层号 1-based；横轴 "Depth (Tokenizer → ...)"；attribute 用 Domain/Frequency/Position。
- 轴标签措辞与 panel_a 一致："k-NN probe accuracy ↑" 等。
- SVG 矢量 + 可编辑文字（svg.fonttype='none'）；同款放大字号。
- 域配色复用 chronos_training_data.DOMAIN_COLORS（按 macro_domain）；cluster 标 C1..Cn。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.chronos_training_data import DOMAIN_COLORS  # noqa: E402  (re-export)

# 出版字号（同 pub_main_figure/panel_a）
FS_TITLE = 17
FS_LABEL = 16
FS_TICK = 13
FS_LEGEND = 13
FS_ANNOT = 12

# 三个架构族的线色：取自仓库 palette 风格、与 DOMAIN_COLORS / CONF_COLOR 都不撞色。
MODEL_COLOR = {
    "chronos_bolt": "#B5403F",     # brick red（house ACCENT_RED）— encoder-decoder
    "timesfm_2_5": "#2F6E8F",      # deep teal                    — decoder-only
    "moment_1_large": "#C98A2B",   # ochre                        — encoder-only
}
# 架构族标注（en-dash，与正文一致）
MODEL_LABEL = {
    "chronos_bolt": "Chronos-Bolt (encoder–decoder)",
    "timesfm_2_5": "TimesFM (decoder-only)",
    "moment_1_large": "MOMENT (encoder-only)",
}
MODEL_ORDER = ["chronos_bolt", "timesfm_2_5", "moment_1_large"]

ATTR_LABEL = {"domain": "Domain", "frequency": "Frequency", "position": "Position"}

# basicts 细域 → macro_domain（统一到 DOMAIN_COLORS / 主图·OOD 图的域命名与配色）
BASICTS_TO_MACRO = {
    "electricity consumption": "Energy",
    "electricity transformer temperature": "Energy",
    "traffic flow": "Traffic",
    "traffic speed": "Traffic",
    "road occupancy rates": "Traffic",
    "weather": "Environment",
    "Beijing air quality": "Environment",
    "exchange rate": "Finance",
    "illness data": "Health",
    "simulated Gaussian data": "Synthetic",
    "simulated pulse data": "Synthetic",
}


def macro_domain(basicts_domain: str) -> str:
    return BASICTS_TO_MACRO.get(basicts_domain, "Other")


def macro_color(basicts_domain: str) -> str:
    return DOMAIN_COLORS.get(macro_domain(basicts_domain), DOMAIN_COLORS["Other"])


def apply_house_rc() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "axes.linewidth": 1.0,
    })
