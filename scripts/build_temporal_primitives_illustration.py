from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "temporal_primitives_illustration"
FIG = ROOT / "outputs" / "tsne_cluster_domain_reports" / "figures"


CANVAS_W = 4400
CANVAS_H = 3050
BG = "#f6f4ef"
INK = "#1f2933"
MUTED = "#667085"
SOFT = "#f9fafb"
BORDER = "#d5dbe5"
BLUE = "#2f6f9f"
GOLD = "#c9963e"
PURPLE = "#7564a7"
RED = "#bd5e5e"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


F_TITLE = font(72, True)
F_SUB = font(31)
F_PANEL = font(36, True)
F_LABEL = font(25, True)
F_SMALL = font(21)
F_TINY = font(17)
F_FOOT = font(20)


@dataclass(frozen=True)
class LayerSpec:
    name: str
    subtitle: str
    color: str
    map_file: str
    prototype_file: str
    total_clusters: int
    selected: tuple[tuple[int, str], ...]


LAYERS = (
    LayerSpec(
        "Projection / early",
        "local patch vocabulary",
        BLUE,
        "projection_tsne_kmeans_k6.png",
        "projection_tsne_kmeans_k6_center_nearest.png",
        6,
        (
            (2, "smooth rising family"),
            (3, "impulse-like family"),
            (5, "increasing trend family"),
        ),
    ),
    LayerSpec(
        "Layer 6 / middle",
        "contextual mixing, still structured",
        GOLD,
        "layer_6_tsne_kmeans_k10.png",
        "layer_6_tsne_kmeans_k10_center_nearest.png",
        10,
        (
            (0, "rising transition family"),
            (6, "falling transition family"),
            (7, "spike-like family"),
        ),
    ),
    LayerSpec(
        "Layer 11 / high-level",
        "stable but less physically direct",
        PURPLE,
        "layer_11_tsne_kmeans_k6.png",
        "layer_11_tsne_kmeans_k6_center_nearest.png",
        6,
        (
            (0, "mixed contextual family"),
            (4, "rising transition-like family"),
            (5, "mixed high-level family"),
        ),
    ),
)


def rounded_rect(draw: ImageDraw.ImageDraw, box, fill="#ffffff", outline=BORDER, radius=18, width=2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def paste_fit(canvas: Image.Image, img: Image.Image, box: tuple[int, int, int, int], border: str | None = BORDER) -> None:
    x0, y0, x1, y1 = box
    max_w = x1 - x0
    max_h = y1 - y0
    img = img.convert("RGB")
    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    px = x0 + (max_w - img.width) // 2
    py = y0 + (max_h - img.height) // 2
    canvas.paste(img, (px, py))
    if border:
        d = ImageDraw.Draw(canvas)
        d.rounded_rectangle([px - 2, py - 2, px + img.width + 2, py + img.height + 2], radius=10, outline=border, width=3)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, draw_font: ImageFont.ImageFont) -> str:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=draw_font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def crop_scatter(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def subplot_boxes(panel_path: Path, total_clusters: int, ncols: int = 4) -> list[list[tuple[int, int, int, int]]]:
    """Recover subplot boxes from the generated center-nearest panels.

    The source panels have a fixed matplotlib layout. We detect dark axis borders
    and use their dominant positions rather than relying on old screenshots.
    """
    img = Image.open(panel_path).convert("RGB")
    arr = np.asarray(img)
    dark = (arr[:, :, 0] < 110) & (arr[:, :, 1] < 120) & (arr[:, :, 2] < 130)
    row_counts = dark.sum(axis=1)
    col_counts = dark.sum(axis=0)

    y_lines = [i for i, c in enumerate(row_counts) if c > 0.75 * img.width]
    x_lines = [i for i, c in enumerate(col_counts) if c > 0.35 * img.height]

    def centers(lines: list[int]) -> list[int]:
        groups: list[list[int]] = []
        for line in lines:
            if not groups or line > groups[-1][-1] + 2:
                groups.append([])
            groups[-1].append(line)
        return [int(round((g[0] + g[-1]) / 2)) for g in groups]

    ys = centers(y_lines)
    xs = centers(x_lines)
    # Axis borders appear as pairs: left/right, top/bottom.
    y_pairs = [(ys[i], ys[i + 1]) for i in range(0, min(len(ys), total_clusters * 2), 2)]
    x_pairs = [(xs[i], xs[i + 1]) for i in range(0, min(len(xs), ncols * 2), 2)]
    if len(y_pairs) < total_clusters or len(x_pairs) < ncols:
        raise RuntimeError(f"Could not detect subplot boxes in {panel_path}")
    return [[(x0, y0, x1, y1) for x0, x1 in x_pairs[:ncols]] for y0, y1 in y_pairs[:total_clusters]]


def crop_curve(panel_path: Path, total_clusters: int, cluster_id: int, nearest_idx: int) -> Image.Image:
    img = Image.open(panel_path).convert("RGB")
    boxes = subplot_boxes(panel_path, total_clusters)
    x0, y0, x1, y1 = boxes[cluster_id][nearest_idx]
    pad_x = 10
    pad_y = 10
    return img.crop((max(0, x0 + pad_x), max(0, y0 + pad_y), min(img.width, x1 - pad_x), min(img.height, y1 - pad_y)))


def draw_curve_strip(
    canvas: Image.Image,
    panel_path: Path,
    total_clusters: int,
    cluster_id: int,
    box: tuple[int, int, int, int],
    color: str,
    n_examples: int = 4,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    gap = 12
    cell_w = (x1 - x0 - gap * (n_examples - 1)) // n_examples
    for i in range(n_examples):
        cell_x0 = x0 + i * (cell_w + gap)
        cell_x1 = cell_x0 + cell_w
        rounded_rect(draw, (cell_x0, y0, cell_x1, y1), fill="#ffffff", outline="#dfe4ec", radius=8, width=2)
        curve = crop_curve(panel_path, total_clusters, cluster_id, i)
        # Retain the zero-line and curve but remove tiny subplot titles.
        paste_fit(canvas, curve, (cell_x0 + 8, y0 + 8, cell_x1 - 8, y1 - 8), border=None)


def draw_layer_evidence_column(
    canvas: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    spec: LayerSpec,
) -> None:
    draw = ImageDraw.Draw(canvas)
    rounded_rect(draw, (x, y, x + w, y + h), fill="#ffffff", outline="#d4dae3", radius=22, width=3)
    draw.rounded_rectangle((x, y, x + w, y + 16), radius=8, fill=spec.color)
    draw.text((x + 34, y + 34), spec.name, font=F_PANEL, fill=INK)
    draw.text((x + 34, y + 82), spec.subtitle, font=F_SMALL, fill=MUTED)

    card_y = y + 132
    card_h = 265
    panel_path = FIG / spec.prototype_file
    for cid, note in spec.selected:
        rounded_rect(draw, (x + 28, card_y, x + w - 28, card_y + card_h), fill=SOFT, outline="#e1e7ef", radius=14, width=2)
        draw.text((x + 52, card_y + 22), f"C{cid}", font=font(31, True), fill=spec.color)
        draw.text((x + 115, card_y + 29), note, font=F_SMALL, fill=INK)
        draw.text((x + 52, card_y + 64), "visual descriptor; center-nearest examples", font=F_TINY, fill=MUTED)
        draw_curve_strip(
            canvas,
            panel_path,
            spec.total_clusters,
            cid,
            (x + 52, card_y + 98, x + w - 52, card_y + card_h - 28),
            spec.color,
        )
        card_y += card_h + 20


def build_composite() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((120, 70), "Model-learned temporal primitive-like structures", font=F_TITLE, fill=INK)
    draw.text(
        (124, 158),
        "Representative Chronos-2 pilot evidence from existing clustering results",
        font=F_SUB,
        fill=MUTED,
    )
    draw.text(
        (124, 212),
        "Claim boundary: learned primitive-like structures appear; their physical meaning is not fully resolved.",
        font=F_SMALL,
        fill="#4b5563",
    )

    draw.text((120, 318), "a", font=font(54, True), fill=INK)
    draw.text((170, 333), "Layer comparison in t-SNE representation space", font=F_PANEL, fill=INK)

    map_w = 1300
    map_h = 790
    map_y = 420
    for i, spec in enumerate(LAYERS):
        x = 120 + i * 1400
        rounded_rect(draw, (x, map_y, x + map_w, map_y + map_h), fill="#ffffff", outline="#d4dae3", radius=22, width=3)
        draw.rounded_rectangle((x, map_y, x + map_w, map_y + 16), radius=8, fill=spec.color)
        draw.text((x + 32, map_y + 34), spec.name, font=F_PANEL, fill=INK)
        draw.text((x + 32, map_y + 82), spec.subtitle, font=F_SMALL, fill=MUTED)
        paste_fit(canvas, crop_scatter(FIG / spec.map_file), (x + 70, map_y + 125, x + map_w - 55, map_y + map_h - 38), border=None)

    draw.text((120, 1295), "b", font=font(54, True), fill=INK)
    draw.text((170, 1310), "Representative KMeans-center nearest raw patches", font=F_PANEL, fill=INK)
    draw.text(
        (170, 1368),
        "Illustrative clusters are manually selected for readability; cluster IDs are not human-designed motif names.",
        font=F_SMALL,
        fill=MUTED,
    )

    col_y = 1440
    col_w = 1300
    col_h = 1050
    for i, spec in enumerate(LAYERS):
        draw_layer_evidence_column(canvas, 120 + i * 1400, col_y, col_w, col_h, spec)

    y = 2570
    rounded_rect(draw, (120, y, CANVAS_W - 120, y + 330), fill="#ffffff", outline="#d4dae3", radius=22, width=3)
    draw.text((165, y + 38), "Visual conclusion", font=font(38, True), fill=INK)
    message = (
        "The TSFM representation space is structured, not random. Early representations show neighborhoods that are easier to read as local patch patterns. "
        "Middle and high-level layers remain clustered, but their neighborhoods become more contextual and less directly mappable to hand-designed motifs."
    )
    wrapped = wrap_text(draw, message, CANVAS_W - 360, F_SUB)
    draw.multiline_text((165, y + 95), wrapped, font=F_SUB, fill="#384250", spacing=9)
    draw.text(
        (165, y + 262),
        "Use as representative archived Chronos-2 pilot evidence; clean follow-up should rerun the protocol with Chronos-Bolt.",
        font=F_FOOT,
        fill=RED,
    )

    out = OUT / "model_learned_temporal_primitives_composite.png"
    canvas.save(out, quality=96)
    return out


def build_prototype_only() -> Path:
    canvas = Image.new("RGB", (4200, 1780), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((100, 65), "Representative model-learned primitive families", font=font(60, True), fill=INK)
    draw.text(
        (104, 138),
        "KMeans-center nearest raw patches selected from existing t-SNE cluster results",
        font=F_SUB,
        fill=MUTED,
    )
    col_y = 230
    col_w = 1270
    col_h = 1400
    for i, spec in enumerate(LAYERS):
        draw_layer_evidence_column(canvas, 100 + i * 1360, col_y, col_w, col_h, spec)
    draw.text(
        (104, 1660),
        "These are representative examples, not final motif taxonomy labels. Cluster naming remains subject to DTW validation and confounder audit.",
        font=F_SMALL,
        fill=RED,
    )
    out = OUT / "model_learned_temporal_primitives_prototypes.png"
    canvas.save(out, quality=96)
    return out


def main() -> None:
    composite = build_composite()
    prototype_only = build_prototype_only()
    print(composite)
    print(prototype_only)


if __name__ == "__main__":
    main()
