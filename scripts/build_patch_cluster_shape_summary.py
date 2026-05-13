#!/usr/bin/env python3
"""Build a Chinese annotated summary image of clustered patch shapes."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "outputs/second_pilot_discovery_summary.json"
PANEL_PATH = ROOT / "outputs/figures/second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_prototype_panel.png"
OUT_PATH = ROOT / "outputs/poster_assets/timesfm_layer10_cluster_patch_shape_summary_cn.png"

COLORS = {
    "ink": "#172026",
    "muted": "#53636d",
    "line": "#d4dee4",
    "paper": "#f6f8fa",
    "panel": "#ffffff",
    "green": "#3a7a38",
    "teal": "#1f8a7a",
    "amber": "#b36b18",
    "red": "#b34545",
    "blue": "#2f6f9f",
    "gray": "#7b878e",
    "soft_green": "#eef8ef",
    "soft_teal": "#eaf6f4",
    "soft_amber": "#fff3df",
    "soft_red": "#faeeee",
    "soft_blue": "#eaf3f8",
}

INTERPRETATIONS = {
    0: ("脉冲/零值模拟簇", "Pulse 主导，zero ratio 极高；更像 synthetic artifact，不进入 taxonomy。", "artifact"),
    1: ("低信息交通占用池", "多为 road occupancy / 60min，形状偏平或弱事件；域/频率风险较高。", "confounded"),
    2: ("大型非平稳混合池", "traffic speed + ETT + traffic flow，包含 mixed / level-shift；太宽泛，需拆分。", "pool"),
    3: ("首 patch 噪声/事件池", "p0 占绝对多数，混有 Gaussian；形状有噪声和事件感，但 position 风险强。", "artifact"),
    4: ("首 patch 位置假象", "跨域、像 level-shift/trend，但 patch_index=0 占 100%；核心 negative control。", "artifact"),
    5: ("下降/平滑转移池", "c5 内部可拆出 strong_falling 与 smooth_falling；是 v1 主候选来源。", "candidate"),
    6: ("首 patch 强转移池", "p0 占 100%，斜率强；形状像 transition，但仍主要是 position-bound。", "artifact"),
    7: ("Illness 周频 level-shift", "Illness 和 weekly frequency 主导；可解释，但偏域/频率 artifact。", "confounded"),
    8: ("上升/恢复转移池", "c8 在 p1/p2/p3 出现，回到原空间是 rising/recovery；是 v1 候选来源。", "candidate"),
    9: ("首 patch 低信息/事件池", "p0 + 60min 主导，混合 flat / burst；更像 context/frequency-mediated pool。", "confounded"),
    10: ("Gaussian 噪声簇", "simulated Gaussian 主导；用于 synthetic artifact control，不命名为概念。", "artifact"),
    11: ("60min 事件/用电池", "Electricity / Traffic / ETT 且 frequency=60；有活动形态但 cadence confounding 强。", "confounded"),
}

STATUS_STYLE = {
    "candidate": ("候选概念来源", COLORS["green"], COLORS["soft_green"]),
    "pool": ("宽泛混合池", COLORS["blue"], COLORS["soft_blue"]),
    "confounded": ("混杂主导", COLORS["amber"], COLORS["soft_amber"]),
    "artifact": ("负例/假象", COLORS["red"], COLORS["soft_red"]),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    return ImageFont.truetype(path, size=size)


def rounded(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, outline: str | None = None, radius: int = 18) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline)


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], fnt: ImageFont.FreeTypeFont, fill: str, width: int, line_gap: int = 8) -> int:
    x, y = xy
    lines: list[str] = []
    for para in text.split("\n"):
        current = ""
        for ch in para:
            trial = current + ch
            if draw.textlength(trial, font=fnt) <= width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = ch
        if current:
            lines.append(current)
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def top_values(items: list[dict], k: int = 2) -> str:
    return " / ".join(f"{item['value']}({item['count']})" for item in items[:k])


def contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value <= prev + 1:
            prev = value
        else:
            runs.append((start, prev))
            start = prev = value
    runs.append((start, prev))
    return runs


def reinforce_subplot_borders(crop: Image.Image) -> Image.Image:
    """Make the matplotlib axes boxes survive downsampling in the poster image."""
    arr = np.asarray(crop.convert("RGB"))
    dark = (arr[:, :, 0] < 80) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)

    y_runs = contiguous_runs(np.where(dark.sum(axis=1) > crop.width * 0.65)[0])
    x_runs = contiguous_runs(np.where(dark.sum(axis=0) > crop.height * 0.45)[0])
    if len(y_runs) < 2 or len(x_runs) < 8:
        return crop

    y_top = (y_runs[0][0] + y_runs[0][1]) // 2
    y_bottom = (y_runs[-1][0] + y_runs[-1][1]) // 2
    x_lines = [(a + b) // 2 for a, b in x_runs[:8]]

    out = crop.copy()
    draw = ImageDraw.Draw(out)
    for left, right in zip(x_lines[0::2], x_lines[1::2]):
        draw.rectangle((left, y_top, right, y_bottom), outline=(0, 0, 0), width=4)
    return out


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    clusters = summary["models"]["timesfm_2_5"]["layers"]["layer_10"]["domain_balanced"]["clusters"]
    clusters_by_id = {int(c["cluster"]): c for c in clusters}

    proto = Image.open(PANEL_PATH).convert("RGB")
    source_row_h = proto.height / 12.0

    width = 2480
    header_h = 260
    row_h = 276
    pad = 34
    label_w = 650
    crop_w = width - label_w - pad * 3
    height = header_h + row_h * 12 + pad * 2
    canvas = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(canvas)

    title_font = font(54, True)
    subtitle_font = font(26)
    h_font = font(28, True)
    body_font = font(22)
    small_font = font(18)
    tiny_font = font(16)

    draw.text((pad, 34), "TimesFM-2.5 layer_10 聚类后的 patch 形态总览", font=title_font, fill=COLORS["ink"])
    subtitle = (
        "主汇报组：domain-balanced second pilot，KMeans K=12。"
        "每行展示一个 embedding cluster 的代表性原始 patch，并用中文标注其可能含义与混杂风险。"
    )
    draw_wrapped(draw, subtitle, (pad, 108), subtitle_font, COLORS["muted"], width - pad * 2)

    legend_x, legend_y = pad, 188
    legend = [
        ("候选概念来源", COLORS["green"], COLORS["soft_green"]),
        ("宽泛混合池", COLORS["blue"], COLORS["soft_blue"]),
        ("混杂主导", COLORS["amber"], COLORS["soft_amber"]),
        ("负例/假象", COLORS["red"], COLORS["soft_red"]),
    ]
    for label, color, fill in legend:
        rounded(draw, (legend_x, legend_y, legend_x + 190, legend_y + 40), fill, color, 20)
        draw.text((legend_x + 18, legend_y + 8), label, font=tiny_font, fill=color)
        legend_x += 210

    for cid in range(12):
        y = header_h + cid * row_h
        c = clusters_by_id[cid]
        title, desc, status = INTERPRETATIONS[cid]
        status_label, status_color, status_fill = STATUS_STYLE[status]

        rounded(draw, (pad, y + 8, width - pad, y + row_h - 12), COLORS["panel"], COLORS["line"], 18)
        rounded(draw, (pad + 20, y + 26, pad + 150, y + 72), status_fill, status_color, 23)
        draw.text((pad + 42, y + 35), f"c{cid}", font=h_font, fill=COLORS["ink"])
        draw.text((pad + 166, y + 35), title, font=h_font, fill=COLORS["ink"])
        draw.text((pad + 20, y + 86), status_label, font=small_font, fill=status_color)
        draw_wrapped(draw, desc, (pad + 20, y + 116), body_font, COLORS["ink"], label_w - 60, line_gap=4)

        meta_y = y + 178
        taxonomy = top_values(c["top_taxonomy_labels"], 3)
        domain = top_values(c["top_domains"], 2)
        patch = top_values(c["top_patch_indices"], 2)
        draw.text((pad + 20, meta_y), f"n={c['size']} | taxonomy: {taxonomy}", font=tiny_font, fill=COLORS["muted"])
        draw.text((pad + 20, meta_y + 26), f"domain: {domain} | patch: {patch}", font=tiny_font, fill=COLORS["muted"])

        row_top = int(cid * source_row_h)
        row_bottom = int((cid + 1) * source_row_h)
        crop = proto.crop((0, row_top, proto.width, row_bottom))
        crop = reinforce_subplot_borders(crop)
        crop.thumbnail((crop_w, row_h - 120), Image.Resampling.LANCZOS)
        crop_x = label_w + pad * 2
        crop_y = y + (row_h - crop.height) // 2
        canvas.paste(crop, (crop_x, crop_y))

    footer = "读法：c5/c8 是最适合继续构建 taxonomy v1 的模型内生 transition 概念来源；c4/c6/c3 等说明形状看起来像 transition 的 cluster 也可能只是 patch position/context artifact。"
    draw_wrapped(draw, footer, (pad, height - pad - 56), small_font, COLORS["muted"], width - pad * 2)

    canvas.save(OUT_PATH)
    print(OUT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
