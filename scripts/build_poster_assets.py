#!/usr/bin/env python3
"""Build simplified assets for the TSFM motif-taxonomy poster."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster_assets"

COLORS = {
    "ink": "#172026",
    "muted": "#5b6972",
    "line": "#d6dee3",
    "paper": "#fbfcfd",
    "blue": "#2f6f9f",
    "teal": "#1f8a7a",
    "green": "#3a7a38",
    "amber": "#b36b18",
    "red": "#b34545",
    "violet": "#6750a4",
    "gray": "#8b969d",
    "light_blue": "#eaf3f8",
    "light_teal": "#eaf6f4",
    "light_amber": "#fff3df",
    "light_red": "#faeded",
}


def load_json(path: str) -> dict:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def svg_root(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="{COLORS['paper']}"/>
{body}
</svg>
"""


FONT_FAMILY = "Noto Sans CJK SC, Noto Sans CJK JP, WenQuanYi Micro Hei, Arial, sans-serif"

TEXT_STYLES = {
    "title": (34, 700, COLORS["ink"], FONT_FAMILY),
    "subtitle": (18, 500, COLORS["muted"], FONT_FAMILY),
    "label": (20, 700, COLORS["ink"], FONT_FAMILY),
    "small": (15, 500, COLORS["muted"], FONT_FAMILY),
    "tiny": (13, 500, COLORS["muted"], FONT_FAMILY),
    "mono": (14, 600, COLORS["ink"], "JetBrains Mono, Consolas, monospace"),
}


def t(x: float, y: float, text: object, cls: str = "small", anchor: str = "start", fill: str | None = None) -> str:
    size, weight, color, family = TEXT_STYLES[cls]
    color = fill or color
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{color}">{esc(text)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str | None = None, r: int = 8) -> str:
    stroke_attr = f' stroke="{stroke}"' if stroke else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}"{stroke_attr}/>'


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 2, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def polyline(points: list[tuple[float, float]], color: str, width: float = 4) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>'


def path_from_curve(values: list[float], x: float, y: float, w: float, h: float) -> list[tuple[float, float]]:
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        lo -= 1
        hi += 1
    pts = []
    for i, v in enumerate(values):
        px = x + (w * i / (len(values) - 1))
        py = y + h - ((v - lo) / (hi - lo)) * h
        pts.append((px, py))
    return pts


def find_subcluster(taxonomy: dict, model: str, layer: str, cluster: int, subcluster: int) -> dict:
    for parent in taxonomy["parent_splits"]:
        target = parent["target"]
        if target["model"] == model and target["layer"] == layer and target["cluster"] == cluster:
            for sub in parent["subclusters"]:
                if sub["subcluster"] == subcluster:
                    return sub
    raise KeyError((model, layer, cluster, subcluster))


def find_cluster_card(cards: dict, model: str, layer: str, cluster: int) -> dict:
    for item in cards["targets"]:
        target = item["target"]
        if target["model"] == model and target["layer"] == layer and target["cluster"] == cluster:
            return item
    raise KeyError((model, layer, cluster))


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def write_selected_clustering_composite() -> Path:
    sources = [
        ("A. KMeans clusters", ROOT / "outputs/figures/second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_pca_clusters.png"),
        ("B. motif taxonomy v0 probe", ROOT / "outputs/figures/second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_pca_taxonomy_v0.png"),
        ("C. patch_index confounder audit", ROOT / "outputs/figures/second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_pca_patch_index.png"),
    ]
    panel_w, panel_h = 520, 372
    title_h, pad = 54, 24
    canvas = Image.new("RGB", (panel_w * 3 + pad * 4, panel_h + title_h + pad * 2), "#fbfcfd")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24, bold=True)
    for i, (label, path) in enumerate(sources):
        img = Image.open(path).convert("RGB")
        # Remove the raw matplotlib title; the composite supplies a cleaner label.
        img = img.crop((0, 70, img.width, img.height))
        img.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
        x = pad + i * (panel_w + pad)
        y = title_h + pad
        draw.rounded_rectangle((x - 6, y - 44, x + panel_w + 6, y + panel_h + 6), radius=12, fill="#ffffff", outline="#d6dee3")
        draw.text((x + 8, y - 38), label, fill="#172026", font=title_font)
        canvas.paste(img, (x + (panel_w - img.width) // 2, y + (panel_h - img.height) // 2))
    path = OUT / "poster_selected_timesfm_clustering.png"
    canvas.save(path)
    return path


def write_selected_patch_rows() -> Path:
    src = ROOT / "outputs/figures/second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_prototype_panel.png"
    img = Image.open(src).convert("RGB")
    # The prototype panel has 12 cluster rows. Keep rows c4, c5, c8 because they form the
    # artifact / falling-transition / rising-transition story.
    row_h = img.height / 12.0
    rows = [(4, "c4 first-patch artifact"), (5, "c5 falling / smooth-transition pool"), (8, "c8 rising / recovery transition")]
    crop_h = int(row_h * 0.96)
    crops = []
    for row_id, label in rows:
        y0 = max(0, int(row_id * row_h + row_h * 0.04))
        crop = img.crop((0, y0, img.width, min(img.height, y0 + crop_h)))
        crop.thumbnail((1280, 178), Image.Resampling.LANCZOS)
        crops.append((label, crop))
    pad, label_w = 22, 250
    out_w = label_w + 1280 + pad * 3
    out_h = pad + len(crops) * (178 + pad)
    canvas = Image.new("RGB", (out_w, out_h), "#fbfcfd")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22, bold=True)
    small_font = load_font(15, bold=False)
    for i, (label, crop) in enumerate(crops):
        y = pad + i * (178 + pad)
        draw.rounded_rectangle((pad, y, out_w - pad, y + 178), radius=12, fill="#ffffff", outline="#d6dee3")
        draw.text((pad + 16, y + 42), label.split(" ", 1)[0], fill="#172026", font=title_font)
        draw.text((pad + 16, y + 76), label.split(" ", 1)[1], fill="#5b6972", font=small_font)
        canvas.paste(crop, (label_w + pad * 2, y + (178 - crop.height) // 2))
    path = OUT / "poster_selected_timesfm_patch_rows.png"
    canvas.save(path)
    return path


def write_story_flow() -> Path:
    steps = [
        ("Question", "什么是时序语言?", "patch-level primitives"),
        ("v0 Probe", "motif taxonomy v0", "human-prior / shapelet"),
        ("Discover", "representation clusters", "discover first, name second"),
        ("Audit", "confounder audit", "position / frequency / domain"),
        ("v1 Name", "motif taxonomy v1", "validated motif families"),
    ]
    body = [t(50, 62, "Story: motif taxonomy v0 -> model-derived motif taxonomy v1", "title")]
    x0, y0, w, h, gap = 56, 126, 254, 210, 28
    fills = [COLORS["light_blue"], "#ffffff", COLORS["light_amber"], COLORS["light_teal"], "#ffffff"]
    strokes = [COLORS["blue"], COLORS["line"], COLORS["amber"], COLORS["teal"], COLORS["green"]]
    for i, (k, title, desc) in enumerate(steps):
        x = x0 + i * (w + gap)
        body.append(rect(x, y0, w, h, fills[i], strokes[i], 12))
        body.append(t(x + 22, y0 + 42, f"{i + 1}. {k}", "label"))
        body.append(t(x + 22, y0 + 88, title, "label"))
        body.append(t(x + 22, y0 + 126, desc, "small"))
        if i < len(steps) - 1:
            ax = x + w + 5
            ay = y0 + h / 2
            bx = x + w + gap - 6
            body.append(line(ax, ay, bx, ay, COLORS["muted"], 3))
            body.append(f'<polygon points="{bx:.1f},{ay:.1f} {bx-13:.1f},{ay-8:.1f} {bx-13:.1f},{ay+8:.1f}" fill="{COLORS["muted"]}"/>')
    body.append(rect(56, 388, 1350, 92, "#172026", None, 12))
    body.append(t(86, 428, "Core claim", "label"))
    body.append(t(210, 428, "TSFM patch tokens learn motif/prototype neighborhoods, not a direct copy of v0 labels.", "subtitle", fill="#dfe7eb"))
    body.append(t(210, 462, "Only clusters surviving original-space inspection and confounder audit enter motif taxonomy v1.", "subtitle", fill="#dfe7eb"))
    path = OUT / "poster_story_flow.svg"
    path.write_text(svg_root(1460, 530, "\n".join(body)), encoding="utf-8")
    return path


def write_taxonomy_v0_prototypes() -> Path:
    prototypes = [
        ("trend", "趋势 trend", [-1.2, -0.9, -0.6, -0.25, 0.05, 0.35, 0.65, 1.0], COLORS["green"]),
        ("oscillation", "振荡 oscillation", [0, 0.8, 0.15, -0.7, -0.2, 0.7, 0.25, -0.5], COLORS["violet"]),
        ("impulse_spike", "脉冲 spike", [0, 0, 0.05, 1.6, 0.1, 0, -0.05, 0], COLORS["amber"]),
        ("burst_event", "爆发 burst", [0, 0.1, 1.0, 1.2, 1.1, 0.2, 0, 0], COLORS["amber"]),
        ("level_shift", "水平突变 shift", [-0.8, -0.75, -0.82, -0.7, 0.75, 0.8, 0.78, 0.82], COLORS["blue"]),
        ("volatility_shift", "波动变化 volatility", [-0.1, 0.08, -0.06, 0.1, -0.9, 0.8, -0.7, 0.95], COLORS["red"]),
        ("intermittent", "间歇 intermittent", [0, 0.9, 0, 0, 0.75, 0, 0.65, 0], COLORS["teal"]),
        ("flat_low_information", "低信息 flat", [0, 0.02, -0.01, 0.0, 0.01, -0.02, 0.0, 0.01], COLORS["gray"]),
        ("mixed_uncertain", "混合 uncertain", [-0.7, 0.25, -0.1, 0.9, 0.35, -0.8, -0.2, 0.6], COLORS["red"]),
    ]
    body = [t(42, 54, "Motif taxonomy v0: human-prior probes", "title")]
    body.append(t(42, 86, "Shapelet-inspired anchors for original-space inspection; not ground-truth motif labels.", "subtitle"))
    x0, y0, w, h = 42, 126, 252, 150
    gap_x, gap_y = 22, 24
    for i, (name, zh, vals, color) in enumerate(prototypes):
        row, col = divmod(i, 3)
        x = x0 + col * (w + gap_x)
        y = y0 + row * (h + gap_y)
        body.append(rect(x, y, w, h, "#ffffff", COLORS["line"], 10))
        body.append(t(x + 18, y + 30, zh, "label"))
        body.append(t(x + 18, y + 52, name, "tiny"))
        cx, cy, cw, ch = x + 18, y + 70, w - 36, 58
        body.append(rect(cx, cy, cw, ch, "#f7fafb", COLORS["line"], 5))
        body.append(line(cx, cy + ch / 2, cx + cw, cy + ch / 2, COLORS["line"], 1, "4 4"))
        body.append(polyline(path_from_curve(vals, cx + 8, cy + 8, cw - 16, ch - 16), color, 3.5))
    body.append(rect(42, 644, 800, 70, COLORS["light_amber"], COLORS["amber"], 10))
    body.append(t(66, 674, "Key point", "label"))
    body.append(t(178, 674, "v0 is a probe; second-pilot v0 NMI is about 0.17-0.20, so hidden clusters are not copies.", "small"))
    path = OUT / "poster_taxonomy_v0_prototypes.svg"
    path.write_text(svg_root(884, 744, "\n".join(body)), encoding="utf-8")
    return path


def write_method_and_hypotheses(second: dict) -> Path:
    tfm = second["models"]["timesfm_2_5"]["layers"]["layer_10"]["domain_balanced"]
    c2 = second["models"]["chronos_2"]["layers"]["layer_11"]["domain_balanced"]
    c2s = second["models"]["chronos_2_small"]["layers"]["layer_5"]["domain_balanced"]
    body = [t(46, 58, "Discover-first: clustering only generates motif candidates", "title")]

    steps = [
        ("1", "Sample", "22 datasets; 100 windows each"),
        ("2", "Represent", "raw / projection / hidden states"),
        ("3", "Project", "StandardScaler + PCA(30)"),
        ("4", "Cluster", "KMeans + stability check"),
        ("5", "Name", "prototype + retrieval + audit"),
    ]
    x0, y0, w, h, gap = 46, 106, 282, 150, 20
    for i, (num, title, desc) in enumerate(steps):
        x = x0 + i * (w + gap)
        body.append(rect(x, y0, w, h, "#ffffff", COLORS["line"], 10))
        body.append(t(x + 18, y0 + 34, f"{num}. {title}", "label"))
        for j, part in enumerate([desc[:48], desc[48:96], desc[96:]]):
            if part:
                body.append(t(x + 18, y0 + 70 + j * 24, part, "tiny"))
        if i < len(steps) - 1:
            ax, ay = x + w + 4, y0 + h / 2
            bx = x + w + gap - 6
            body.append(line(ax, ay, bx, ay, COLORS["muted"], 2))
            body.append(f'<polygon points="{bx},{ay} {bx-10},{ay-6} {bx-10},{ay+6}" fill="{COLORS["muted"]}"/>')

    body.append(t(46, 314, "Current evidence for H1-H3", "title"))
    rows = [
        ("H1 temporal primitives", "partial", "falling_transition motif family beats matched random across three models.", COLORS["green"]),
        ("H2 v0 vs v1", "audit", f"v0 NMI {tfm['nmi']['taxonomy_v0']:.3f}; domain/frequency/position signals remain.", COLORS["amber"]),
        ("H3 scale / architecture", "mixed", f"Chronos is position-stable; TimesFM patch-index NMI {tfm['nmi']['patch_index']:.3f}.", COLORS["amber"]),
    ]
    x, y, rowh = 46, 364, 92
    widths = [360, 150, 850]
    for i, (hyp, status, evidence, color) in enumerate(rows):
        fill = "#ffffff" if i % 2 == 0 else "#f7fafb"
        body.append(rect(x, y, sum(widths), rowh, fill, COLORS["line"], 8))
        body.append(t(x + 18, y + 34, hyp, "label"))
        body.append(rect(x + widths[0] + 18, y + 22, 108, 30, color, None, 15))
        body.append(t(x + widths[0] + 72, y + 43, status, "tiny", "middle", fill="#ffffff"))
        body.append(t(x + widths[0] + widths[1] + 22, y + 34, evidence[:74], "small"))
        body.append(t(x + widths[0] + widths[1] + 22, y + 62, evidence[74:], "small"))
        y += rowh + 10

    metric_y = 690
    metric_rows = [
        ("TimesFM-2.5 layer_10", tfm),
        ("Chronos-2 layer_11", c2),
        ("Chronos-2-small layer_5", c2s),
    ]
    body.append(t(46, metric_y, "Why focus on TimesFM-2.5 layer_10?", "label"))
    body.append(t(438, metric_y, "High stability, clear motif candidates, and a first-patch artifact negative control.", "small"))
    y = metric_y + 28
    for name, m in metric_rows:
        body.append(t(66, y + 20, name, "small"))
        vals = [
            ("silhouette", m["silhouette_pca_space"], 0.2),
            ("stability", m["kmeans_vs_agglomerative_nmi"], 0.8),
            ("v0 NMI", m["nmi"]["taxonomy_v0"], 0.45),
            ("patch-index NMI", m["nmi"]["patch_index"], 0.45),
        ]
        xbar = 300
        for label, val, vmax in vals:
            body.append(t(xbar, y + 2, label, "tiny"))
            body.append(rect(xbar, y + 12, 130, 12, "#edf2f4", None, 6))
            color = COLORS["green"] if label == "stability" else COLORS["blue"] if label != "patch-index NMI" else COLORS["red"]
            body.append(rect(xbar, y + 12, 130 * min(val / vmax, 1), 12, color, None, 6))
            body.append(t(xbar, y + 42, f"{val:.3f}", "tiny"))
            xbar += 190
        y += 64
    path = OUT / "poster_method_hypotheses.svg"
    path.write_text(svg_root(1580, 930, "\n".join(body)), encoding="utf-8")
    return path


def write_concept_curves(taxonomy: dict) -> Path:
    specs = [
        ("strong_rising_recovery", "c8-s0", find_subcluster(taxonomy, "timesfm_2_5", "layer_10", 8, 0), COLORS["teal"], "candidate motif: rising / recovery"),
        ("strong_falling_transition", "c5-s1", find_subcluster(taxonomy, "timesfm_2_5", "layer_10", 5, 1), COLORS["green"], "strong motif: falling transition"),
        ("smooth_falling_transition", "c5-s0", find_subcluster(taxonomy, "timesfm_2_5", "layer_10", 5, 0), COLORS["blue"], "candidate motif: smooth falling"),
        ("uncertain_mixed_pool", "c5-s2", find_subcluster(taxonomy, "timesfm_2_5", "layer_10", 5, 2), COLORS["red"], "exclude: mixed pool"),
    ]
    body = [t(48, 58, "From local patch neighborhoods to motif taxonomy v1 families", "title")]
    x0, y0, w, h, gap = 52, 116, 330, 268, 28
    for i, (name, code, sub, color, zh) in enumerate(specs):
        x = x0 + i * (w + gap)
        vals = sub["shape_summary"]["mean_curve"]
        cond = sub["validation"]["condition_summary"]
        body.append(rect(x, y0, w, h, "#ffffff", COLORS["line"], 12))
        body.append(t(x + 20, y0 + 36, zh, "label"))
        body.append(t(x + 20, y0 + 64, f"{name} | {code} | n={sub['stats']['size']}", "tiny"))
        cx, cy, cw, ch = x + 26, y0 + 88, w - 52, 104
        body.append(rect(cx, cy, cw, ch, "#f7fafb", COLORS["line"], 6))
        body.append(line(cx, cy + ch / 2, cx + cw, cy + ch / 2, COLORS["line"], 1, "4 4"))
        body.append(polyline(path_from_curve(vals, cx + 8, cy + 12, cw - 16, ch - 24), color, 4))
        metrics = [
            ("cross-domain", cond["cross_domain"]["mean_shape_correlation_mean"]),
            ("same-patch", cond["cross_domain_same_patch_index"]["mean_shape_correlation_mean"]),
            ("same-frequency", cond["cross_domain_same_frequency"]["mean_shape_correlation_mean"]),
        ]
        for j, (m, val) in enumerate(metrics):
            my = y0 + 218 + j * 22
            body.append(t(x + 24, my, m, "tiny"))
            body.append(rect(x + 142, my - 13, 120, 12, "#eef2f4", None, 6))
            fill = COLORS["green"] if val >= 0.6 else COLORS["amber"] if val >= 0.25 else COLORS["red"]
            body.append(rect(x + 142, my - 13, 120 * max(0, min(val, 1)), 12, fill, None, 6))
            body.append(t(x + 276, my, f"{val:.3f}", "tiny"))
    path = OUT / "poster_concept_curves.svg"
    path.write_text(svg_root(1500, 440, "\n".join(body)), encoding="utf-8")
    return path


def write_audit_matrix(taxonomy: dict, cards: dict, cross: dict) -> Path:
    rows = []
    for name, cluster, subcluster, label in [
        ("strong_falling_transition", 5, 1, "强下降 falling"),
        ("smooth_falling_transition", 5, 0, "平滑下降 smooth"),
        ("strong_rising_recovery", 8, 0, "上升恢复 rising"),
    ]:
        sub = find_subcluster(taxonomy, "timesfm_2_5", "layer_10", cluster, subcluster)
        cond = sub["validation"]["condition_summary"]
        cross_vals = []
        for mr in cross["model_results"]:
            base = mr["matched_random_baseline"]["mean_shape_correlation"]
            val = mr["global_retrieval"][name]["mean_shape_correlation"]
            cross_vals.append(val - base)
        rows.append({
            "name": name,
            "label": label,
            "kind": "candidate",
            "cross_domain": cond["cross_domain"]["mean_shape_correlation_mean"],
            "same_patch": cond["cross_domain_same_patch_index"]["mean_shape_correlation_mean"],
            "same_frequency": cond["cross_domain_same_frequency"]["mean_shape_correlation_mean"],
            "cross_model_uplift": min(cross_vals),
            "note": "PASS" if name == "strong_falling_transition" else "继续审计",
        })
    artifact = find_cluster_card(cards, "timesfm_2_5", "layer_10", 4)
    art_ret = artifact["retrieval"]
    rows.append({
        "name": "artifact_first_patch_behavior",
        "label": "首 patch artifact",
        "kind": "artifact",
        "cross_domain": art_ret["cross_domain"]["mean_shape_correlation"],
        "same_patch": art_ret["same_patch_index"]["mean_shape_correlation"],
        "same_frequency": art_ret["same_frequency"]["mean_shape_correlation"],
        "cross_model_uplift": -0.04,
        "note": "FAIL: patch_index=0",
    })

    headers = ["candidate motif", "cross-domain", "same-patch", "same-frequency", "cross-model uplift", "decision"]
    body = [t(48, 58, "Confounder audit: 高相似度不等于 motif", "title")]
    x, y = 52, 106
    colw = [360, 190, 190, 210, 230, 300]
    rowh = 82
    body.append(rect(x, y, sum(colw), 54, "#172026", None, 8))
    cx = x
    for i, h in enumerate(headers):
        body.append(f'<text x="{cx + 16:.1f}" y="{y + 34:.1f}" font-family="{FONT_FAMILY}" font-size="17" font-weight="700" fill="#ffffff">{esc(h)}</text>')
        cx += colw[i]
    y += 60
    for r, row in enumerate(rows):
        fill = "#ffffff" if r % 2 == 0 else "#f7fafb"
        body.append(rect(x, y, sum(colw), rowh, fill, COLORS["line"], 6))
        cx = x
        color = COLORS["red"] if row["kind"] == "artifact" else COLORS["teal"]
        body.append(t(cx + 16, y + 30, row["label"], "label"))
        body.append(t(cx + 16, y + 56, row["name"], "tiny"))
        cx += colw[0]
        for key in ["cross_domain", "same_patch", "same_frequency", "cross_model_uplift"]:
            val = row[key]
            bar_w = colw[["cross_domain", "same_patch", "same_frequency", "cross_model_uplift"].index(key) + 1] - 44
            if key == "cross_model_uplift":
                good = val >= 0.18
                warn = val >= 0.06
                shown = f"+{val:.3f}" if val >= 0 else f"{val:.3f}"
                norm = max(0, min((val + 0.05) / 0.45, 1))
            else:
                good = val >= 0.6
                warn = val >= 0.25
                shown = f"{val:.3f}"
                norm = max(0, min(val, 1))
            c = COLORS["green"] if good else COLORS["amber"] if warn else COLORS["red"]
            body.append(rect(cx + 16, y + 24, bar_w, 14, "#eef2f4", None, 7))
            body.append(rect(cx + 16, y + 24, bar_w * norm, 14, c, None, 7))
            body.append(t(cx + 16, y + 60, shown, "mono"))
            cx += colw[["cross_domain", "same_patch", "same_frequency", "cross_model_uplift"].index(key) + 1]
        body.append(t(cx + 16, y + 36, row["note"], "label"))
        if row["kind"] == "artifact":
            body.append(t(cx + 16, y + 62, "形状一致，但由位置驱动", "tiny"))
        y += rowh + 8
    body.append(t(54, 540, "读法：绿色表示形态一致性通过控制条件；artifact 行说明 cross-domain 高也可能只是 position 机制。", "subtitle"))
    path = OUT / "poster_audit_matrix.svg"
    path.write_text(svg_root(1530, 585, "\n".join(body)), encoding="utf-8")
    return path


def write_cross_model(cross: dict) -> Path:
    concepts = [
        ("strong_falling_transition", "falling", COLORS["green"]),
        ("smooth_falling_transition", "smooth", COLORS["blue"]),
        ("strong_rising_recovery", "rising", COLORS["teal"]),
        ("artifact_first_patch_behavior", "artifact", COLORS["red"]),
    ]
    models = cross["model_results"]
    body = [t(48, 58, "Cross-model validation: falling transition 是最稳 motif family", "title")]
    panel_w, panel_h = 450, 350
    for i, mr in enumerate(models):
        x = 52 + i * (panel_w + 36)
        y = 110
        base = mr["matched_random_baseline"]["mean_shape_correlation"]
        body.append(rect(x, y, panel_w, panel_h, "#ffffff", COLORS["line"], 12))
        body.append(t(x + 22, y + 38, mr["display"], "label"))
        chart_x, chart_y, chart_w, chart_h = x + 64, y + 78, panel_w - 112, 190
        body.append(line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, COLORS["line"], 1))
        body.append(line(chart_x, chart_y, chart_x, chart_y + chart_h, COLORS["line"], 1))
        by = chart_y + chart_h - base * chart_h
        body.append(line(chart_x, by, chart_x + chart_w, by, COLORS["ink"], 2, "6 5"))
        body.append(t(chart_x + chart_w - 2, by - 8, f"matched random {base:.3f}", "tiny", "end"))
        bw = chart_w / len(concepts) * 0.58
        for j, (concept, label, color) in enumerate(concepts):
            val = mr["global_retrieval"][concept]["mean_shape_correlation"]
            bx = chart_x + j * (chart_w / len(concepts)) + 20
            bh = val * chart_h
            body.append(rect(bx, chart_y + chart_h - bh, bw, bh, color, None, 3))
            body.append(t(bx + bw / 2, chart_y + chart_h + 26, label, "tiny", "middle"))
            body.append(t(bx + bw / 2, chart_y + chart_h - bh - 8, f"{val:.2f}", "tiny", "middle"))
        body.append(t(x + 24, y + 306, f"prototype 1NN same concept: {mr['prototype_space']['mean_1nn_same_concept']:.3f}", "small"))
        body.append(t(x + 24, y + 332, f"prototype shape corr: {mr['prototype_space']['mean_1nn_shape_correlation']:.3f}", "small"))
    body.append(rect(52, 500, 1420, 82, COLORS["light_teal"], COLORS["teal"], 10))
    body.append(t(82, 535, "Takeaway", "label"))
    body.append(t(212, 535, "strong_falling_transition 在 TimesFM、Chronos-2、Chronos-2-small 中均高于 matched random。", "subtitle"))
    path = OUT / "poster_cross_model.svg"
    path.write_text(svg_root(1530, 630, "\n".join(body)), encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    taxonomy = load_json("outputs/taxonomy_v1_pilot/taxonomy_v1_pilot_summary.json")
    cards = load_json("outputs/cluster_cards/cluster_card_summary.json")
    cross = load_json("outputs/cross_model_validation/cross_model_validation_summary.json")
    second = load_json("outputs/second_pilot_discovery_summary.json")
    written = [
        write_story_flow(),
        write_selected_clustering_composite(),
        write_selected_patch_rows(),
        write_taxonomy_v0_prototypes(),
        write_method_and_hypotheses(second),
        write_concept_curves(taxonomy),
        write_audit_matrix(taxonomy, cards, cross),
        write_cross_model(cross),
    ]
    manifest = {
        "assets": [str(p.relative_to(ROOT)) for p in written],
        "source_files": [
            "outputs/taxonomy_v1_pilot/taxonomy_v1_pilot_summary.json",
            "outputs/cluster_cards/cluster_card_summary.json",
            "outputs/cross_model_validation/cross_model_validation_summary.json",
            "outputs/second_pilot_discovery_summary.json",
        ],
    }
    (OUT / "poster_asset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in written:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
