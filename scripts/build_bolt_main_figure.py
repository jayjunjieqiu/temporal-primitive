"""Task 1 — Chronos-Bolt main figure（迁到 clean Bolt）。

老师反馈：main 图不要 first-difference / power-spectrum 行，只留 **raw patch stack**；加上
**最深层**；下面用**分 domain 的 prototype example** 那张。全部迁到 clean Chronos-Bolt。

产出 modular PNG（用户偏好手动拼接，不要自动合成整图）：
  - `bolt_patch_stack_cards_layer0.png`   : layer_0 (shallow) 的 raw-only patch-stack cards
  - `bolt_patch_stack_cards_layer11.png`  : layer_11 (deepest, contextualized) 的 raw-only cards
  - `bolt_domain_balanced_prototype_panel.png` : domain-balanced 聚类的 prototype example panel

方法（two-space principle，见 docs/00_narrative_rules.md §5.1）：在 representation space 用
StandardScaler → PCA(30) → KMeans(k) 生成候选 cluster；回到 original time-series space 用
z-normalized raw patch 展示形状。t-SNE 不参与（这里只要 cards/prototype，不要 atlas）。

注意：这些是 clean Chronos-Bolt 结构证据，但仍是 *candidate* cluster，不是命名好的 motif；
图注遵守 narrative rules（不要把 raw KMeans cluster 直接叫 motif）。

从仓库根目录运行：
    .venv/bin/python scripts/build_bolt_main_figure.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/temporal_primitive_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.chronos_bolt_backbone import extract_bolt_representations, load_bolt_pipeline  # noqa: E402
from scripts.chronos_training_data import DOMAIN_COLORS, sample_training_windows  # noqa: E402
from scripts.explore_motif_taxonomy import LABELS as MOTIF_LABELS  # noqa: E402
from scripts.explore_motif_taxonomy import label_patch  # noqa: E402
from scripts.run_prior_guided_probe_sanity_check import macro_domain  # noqa: E402
from scripts.run_second_pilot_discovery import (  # noqa: E402
    DATA_ROOT,
    robust_z,
    sample_windows,
    select_domain_balanced_indices,
)

OUTPUT_DIR = ROOT / "outputs" / "figures" / "bolt_main_figure"

# validation 时必须剔除的 basicts 数据集，两类原因：
#  (1) 在 Chronos 预训练集内（模型本就见过）：Electricity≈monash_electricity_hourly、
#      BeijingAirQuality≈monash_kdd_cup_2018（见 docs/16）。
#  (2) 与我们 discovery 训练子集**同一份数据**（泄漏）：Traffic≡monash_traffic（862 series/17544）、
#      ExchangeRate≡exchange_rate（8 series/7588）。
VALIDATION_EXCLUDE = {"Electricity", "BeijingAirQuality", "Traffic", "ExchangeRate", "BLAST"}


def z_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x - float(np.mean(x))) / max(float(np.std(x)), eps)).astype(np.float32)


def flatten_patches(
    layer_emb: np.ndarray, windows_z: np.ndarray, window_meta: list[dict[str, Any]], patch_len: int
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n, num_patches, _ = layer_emb.shape
    emb_list, raw_list, meta_list = [], [], []
    for i in range(n):
        for p in range(num_patches):
            raw = windows_z[i, p * patch_len : (p + 1) * patch_len]
            if raw.shape[0] < patch_len:
                continue
            emb_list.append(layer_emb[i, p])
            raw_list.append(raw)
            meta_list.append(
                {
                    "dataset": window_meta[i]["dataset"],
                    "domain": window_meta[i].get("domain"),
                    # 训练数据 meta 自带 macro_domain；basicts 走 domain->macro 映射
                    "macro_domain": window_meta[i].get("macro_domain")
                    or macro_domain(window_meta[i].get("domain")),
                    "patch_index": p,
                }
            )
    return np.stack(emb_list).astype(np.float32), np.stack(raw_list).astype(np.float32), meta_list


def cluster_pca(emb: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """StandardScaler -> PCA(30) -> KMeans。返回 (labels, centers_in_pca, pca_coords)。"""
    Xs = StandardScaler().fit_transform(emb)
    Xp = PCA(n_components=min(30, Xs.shape[1]), random_state=seed).fit_transform(Xs)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xp)
    return km.labels_, km.cluster_centers_, Xp


def cluster_pca_fit(emb: np.ndarray, k: int, seed: int):
    """同 cluster_pca，但额外返回 fit 好的 (scaler, pca, km)，供把 validation patch 投到同一
    discovery 空间做 cross-space 泛化检验。返回 (labels, centers, pca_coords, scaler, pca, km)。"""
    scaler = StandardScaler().fit(emb)
    Xs = scaler.transform(emb)
    pca = PCA(n_components=min(30, Xs.shape[1]), random_state=seed).fit(Xs)
    Xp = pca.transform(Xs)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xp)
    return km.labels_, km.cluster_centers_, Xp, scaler, pca, km


def render_raw_cards(
    layer_name: str,
    labels: np.ndarray,
    centers: np.ndarray,
    pca_coords: np.ndarray,
    raw_patches: np.ndarray,
    meta: list[dict[str, Any]],
    k: int,
    top_n: int,
    out_path: Path,
) -> dict[str, Any]:
    """每个 cluster 一张 raw patch-stack 卡（center-nearest top_n，z-normalized imshow）。

    标题下加一根 100% 归一化的 domain-composition 横条，反映**整个 cluster**（不是 top_n）
    的 macro_domain 构成，避免"cluster=单一 domain"的误读。
    """
    fig = plt.figure(figsize=(2.05 * k, 4.1))
    # bottom 留一点空间放 ticker(time/0 7 15) + 单排 legend，二者之间留小间隙、不重叠
    gs = fig.add_gridspec(2, k, height_ratios=[0.085, 1.0], hspace=0.04, wspace=0.18,
                          top=0.90, bottom=0.20)
    card_info = []
    seen_domains: set[str] = set()
    # 先收集每簇 center-nearest 的 z-stack，求一个**共享** vmax，使所有卡片同一色标、colorbar 可读
    per_cluster: list[tuple[np.ndarray, np.ndarray] | None] = []
    all_abs: list[np.ndarray] = []
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            per_cluster.append(None)
            continue
        dist = np.linalg.norm(pca_coords[idx] - centers[cid], axis=1)
        order = np.argsort(dist)[: min(top_n, len(idx))]
        z = np.stack([z_normalize(raw_patches[i]) for i in idx[order]])
        per_cluster.append((idx, z))
        all_abs.append(np.abs(z).ravel())
    vmax = float(np.percentile(np.concatenate(all_abs), 97)) if all_abs else 2.5

    im = None
    for cid in range(k):
        bar_ax = fig.add_subplot(gs[0, cid])
        ax = fig.add_subplot(gs[1, cid])
        if per_cluster[cid] is None:
            bar_ax.axis("off")
            ax.axis("off")
            continue
        idx, z = per_cluster[cid]
        im = ax.imshow(z, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks([0, 7, 15])
        ax.set_yticks([])
        ax.tick_params(labelsize=6, length=2)
        ax.set_xlabel("time", fontsize=7)
        if cid == 0:
            ax.set_ylabel("rank (center→far)", fontsize=7.5)
        for sp in ax.spines.values():
            sp.set_color("#1f2933")
            sp.set_linewidth(0.8)

        # 整个 cluster 的 domain 构成（100% 堆叠条，按占比降序）
        comp = Counter(meta[i]["macro_domain"] for i in idx)
        total = sum(comp.values())
        ordered = sorted(comp.items(), key=lambda kv: -kv[1])
        left = 0.0
        for dom, cnt in ordered:
            frac = cnt / total
            bar_ax.barh(0, frac, left=left, height=1.0,
                        color=DOMAIN_COLORS.get(dom, DOMAIN_COLORS["Other"]), edgecolor="white", lw=0.3)
            left += frac
            seen_domains.add(dom)
        bar_ax.set_xlim(0, 1)
        bar_ax.set_ylim(-0.5, 0.5)
        bar_ax.axis("off")
        bar_ax.set_title(f"C{cid + 1}  (n={len(idx)})", fontsize=8, pad=2)
        card_info.append(
            {"cluster": f"C{cid + 1}", "size": int(len(idx)),
             "domain_composition": {d: round(c / total, 3) for d, c in ordered}}
        )

    # 颜色条：告诉读者颜色代表多少数值。每个 patch 单独 z-normalize（均值0、单位σ），色标统一为 ±vmax σ
    if im is not None:
        cax = fig.add_axes((0.915, 0.30, 0.008, 0.45))
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("z-normalized value (per patch, σ)", fontsize=7.5)
        cbar.ax.tick_params(labelsize=7)

    handles = [
        Line2D([0], [0], marker="s", ls="", ms=7, color=DOMAIN_COLORS[d], label=d)
        for d in DOMAIN_COLORS if d in seen_domains
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.115), title="macro domain composition",
               title_fontsize=8.5, columnspacing=1.6, handletextpad=0.5)
    fig.suptitle(
        f"Chronos-Bolt {layer_name} — raw patch-stack candidate clusters (k={k}, center-nearest {top_n})",
        fontsize=10, y=0.97,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"output": str(out_path), "clusters": card_info}


def render_cluster_maps(
    layer_clusters: dict[str, tuple[np.ndarray, np.ndarray]],
    v0_labels: np.ndarray,
    domain_labels: np.ndarray,
    k: int,
    seed: int,
    perplexity: float,
    out_path: Path,
) -> dict[str, Any]:
    """representation atlas（中间 plate）：每行一个 depth，三列同一套 t-SNE 点、不同着色——
    左=模型 KMeans cluster，中=human motif taxonomy v0（shapelet probe，*不是* ground truth），
    右=macro domain（confounder audit：模型 cluster 有没有被 domain 带跑）。

    KMeans 在 PCA(30) space 完成；t-SNE 只做 visualization（不参与聚类）。
    """
    names = list(layer_clusters.keys())
    clu_cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")
    motif_cmap = plt.get_cmap("tab10")
    motif_color = {lab: motif_cmap(i % 10) for i, lab in enumerate(MOTIF_LABELS)}
    # mixed_uncertain 是 probe 的"兜底"类、占比大，淡化成浅灰画在底层，让真正 fired 的 motif 突出
    DIM_MOTIFS = {"mixed_uncertain": "#cfcfcf"}
    motif_color.update(DIM_MOTIFS)
    motif_draw_order = [m for m in MOTIF_LABELS if m in DIM_MOTIFS] + [m for m in MOTIF_LABELS if m not in DIM_MOTIFS]
    seen_domains = [d for d in DOMAIN_COLORS if d in set(domain_labels.tolist())]

    fig, axes = plt.subplots(len(names), 3, figsize=(15.5, 4.6 * len(names)), squeeze=False)
    info: dict[str, Any] = {}
    for row, name in enumerate(names):
        labels, pca_coords = layer_clusters[name]
        xy = TSNE(n_components=2, perplexity=perplexity, init="pca",
                  random_state=seed, max_iter=1000).fit_transform(pca_coords)

        ax = axes[row, 0]  # 模型 KMeans cluster
        for cid in range(k):
            m = labels == cid
            ax.scatter(xy[m, 0], xy[m, 1], s=5, color=clu_cmap(cid % (10 if k <= 10 else 20)), alpha=0.6)
        ax.set_title(f"Chronos-Bolt {name}\nmodel-derived KMeans clusters (k={k})", fontsize=10)

        ax = axes[row, 1]  # human motif taxonomy v0
        for lab in motif_draw_order:
            m = v0_labels == lab
            if not m.any():
                continue
            dim = lab in DIM_MOTIFS
            ax.scatter(xy[m, 0], xy[m, 1], s=4 if dim else 6,
                       color=motif_color[lab], alpha=0.12 if dim else 0.75,
                       zorder=1 if dim else 2)
        ax.set_title(f"Chronos-Bolt {name}\nhuman motif taxonomy v0 (probe)", fontsize=10)

        ax = axes[row, 2]  # macro domain（confounder）
        for dom in seen_domains:
            m = domain_labels == dom
            ax.scatter(xy[m, 0], xy[m, 1], s=5, color=DOMAIN_COLORS[dom], alpha=0.6)
        ax.set_title(f"Chronos-Bolt {name}\nmacro domain (confounder)", fontsize=10)

        for col, ax in enumerate(axes[row]):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("t-SNE-1", fontsize=8)
            if col == 0:
                ax.set_ylabel("t-SNE-2", fontsize=8)
            for sp in ax.spines.values():
                sp.set_color("#1f2933")
                sp.set_linewidth(0.8)
        info[name] = {"n_points": int(len(labels))}

    def _pretty(lab: Any) -> str:  # 去掉粗糙的下划线命名（flat_low_information -> flat low information）
        return str(lab).replace("_", " ")

    clu_handles = [
        Line2D([0], [0], marker="o", ls="", ms=6, color=clu_cmap(c % (10 if k <= 10 else 20)), label=f"C{c + 1}")
        for c in range(k)
    ]
    motif_handles = [Line2D([0], [0], marker="o", ls="", ms=6, color=motif_color[l], label=_pretty(l)) for l in MOTIF_LABELS]
    dom_handles = [Line2D([0], [0], marker="o", ls="", ms=6, color=DOMAIN_COLORS[d], label=_pretty(d)) for d in seen_domains]
    # 每组 legend 放在对应列的正下方，双列排版省高度
    col_legends = [
        (0, clu_handles, f"model cluster (k={k})"),
        (1, motif_handles, "human motif v0 (probe)"),
        (2, dom_handles, "macro domain (confounder)"),
    ]
    for col, handles, title in col_legends:
        axes[-1, col].legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
            fontsize=7.5, title=title, title_fontsize=8.5, frameon=False,
            columnspacing=1.1, handletextpad=0.4, labelspacing=0.3,
        )
    fig.suptitle(
        "Representation atlas across depth — model clusters vs human motif taxonomy v0 vs macro domain\n"
        "(KMeans in PCA space; t-SNE for visualization only; v0 = shapelet-inspired probe, not ground truth)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"output": str(out_path), "panels": info, "perplexity": perplexity}


def render_prototype_panel(
    emb: np.ndarray,
    raw_patches: np.ndarray,
    meta: list[dict[str, Any]],
    k: int,
    seed: int,
    proto_per_cluster: int,
    max_per_domain: int,
    out_path: Path,
    layer_name: str,
) -> dict[str, Any]:
    """cross-domain prototype example panel：行=cluster，列=**不同 macro domain** 里离 cluster
    中心最近的最佳代表（line plot）。强调同一 shape family 跨域复用，而不是集中在某几个域。
    """
    sel = select_domain_balanced_indices(meta, max_per_domain=max_per_domain, seed=seed)
    labels, centers, pca_coords = cluster_pca(emb[sel], k, seed)
    raw_sel = raw_patches[sel]

    # 固定列：col0 = cluster prototype（与 generalization 图统一）；其后每列一个 macro domain
    present = {meta[int(sel[li])]["macro_domain"] for li in range(len(sel))}
    domain_order = [d for d in DOMAIN_COLORS if d != "Other" and d in present]
    ncol = 1 + len(domain_order)

    proto = np.zeros((k, raw_sel.shape[1]), dtype=np.float64)
    for c in range(k):
        m = labels == c
        if m.any():
            proto[c] = np.mean(np.stack([z_normalize(raw_sel[i]) for i in np.where(m)[0]]), axis=0)

    fig, axes = plt.subplots(k, ncol, figsize=(2.0 * ncol, 1.45 * k), squeeze=False)
    panel_info = []
    for cid in range(k):
        idx_local = np.where(labels == cid)[0]
        best_by_dom: dict[str, int] = {}
        if len(idx_local) > 0:
            dist = np.linalg.norm(pca_coords[idx_local] - centers[cid], axis=1)
            best_dist: dict[str, float] = {}
            for j, li in enumerate(idx_local):
                g = int(sel[li])
                dom = meta[g]["macro_domain"]
                if dom not in best_dist or dist[j] < best_dist[dom]:
                    best_dist[dom] = float(dist[j])
                    best_by_dom[dom] = g
        panel_info.append({"cluster": f"C{cid + 1}", "size": int(len(idx_local)),
                           "domains_present": sorted(best_by_dom.keys())})
        # col0 = cluster prototype（红，与 main_D 一致）
        ax = axes[cid, 0]
        ax.set_xticks([]); ax.set_yticks([])
        ax.plot(proto[cid], lw=1.6, color="#c0392b")
        ax.set_facecolor("#fbecea")
        ax.set_ylabel(f"C{cid + 1}", fontsize=9, rotation=0, labelpad=12, va="center")
        if cid == 0:
            ax.set_title("train prototype", fontsize=7, color="#c0392b", fontweight="bold")
        # domain 列：黑线 + 彩色 domain 表头（与 main_D 统一）
        for j, dom in enumerate(domain_order):
            ax = axes[cid, j + 1]
            ax.set_xticks([]); ax.set_yticks([])
            if cid == 0:
                ax.set_title(dom, fontsize=7, color=DOMAIN_COLORS[dom], fontweight="bold")
            if dom not in best_by_dom:
                ax.set_facecolor("#f6f6f6")
                continue
            ax.plot(z_normalize(raw_patches[best_by_dom[dom]]), lw=1.2, color="#1f2933")
    fig.suptitle(
        f"Chronos-Bolt {layer_name} — cross-domain prototypes (training)\n"
        f"(k={k}; col 0 = cluster prototype; columns = macro domain, center-nearest training patch per domain; "
        "blank = domain absent in cluster)",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"output": str(out_path), "clusters": panel_info, "domain_columns": domain_order}


def _zcorr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 1e-12 else 0.0


def render_generalization_panel(
    layer_name: str,
    scaler: Any,
    pca: Any,
    centers: np.ndarray,
    disc_labels: np.ndarray,
    disc_raw: np.ndarray,
    val_emb: np.ndarray,
    val_raw: np.ndarray,
    val_meta: list[dict[str, Any]],
    k: int,
    proto_per_cluster: int,
    out_path: Path,
    coh_thresh: float = 0.6,
    control_datasets: tuple[str, ...] = ("Gaussian", "Pulse"),
    noise_control: str = "Gaussian",
) -> dict[str, Any]:
    """泛化检验：把在**训练数据**上发现的 prototype，拿去检索**unseen** patch。

    - discovery prototype shape = 每簇 z-normalized discovery raw patch 的均值（original space）。
    - 每个 unseen patch：在 *representation space* 投到同一 discovery 空间 → 最近 cluster（rep-NN）。
    - **headline = shape coherence**：corr(patch, 其 rep-NN prototype) ≥ thresh 的比例。real held-out
      数据应当高，pure-noise control（Gaussian）应当≈0 → primitive 不是 artifact。这是主指标，因为它
      能干净地拒绝噪声。
    - 次要诊断 cross-space agreement（rep-NN == raw-shape-NN）**不**作 headline：它会被噪声蒙混
      （纯噪声在两个空间都稳定落进同一"通用 wiggle"簇，agreement 反而偏高），不能区分 control。
    - caveat：coherence 用 position-sensitive correlation，会**低估 shift-invariant 家族**（如
      impulse/Pulse：spike 位置随机，对固定位置 prototype 相关性低，但 representation 仍把它们正确
      归到 impulse 簇——见 panel）。故 Gaussian 是干净 negative control，Pulse 是被低估的 positive。
    - panel：行=discovery cluster，列=分配进该簇、来自不同 unseen 数据集、离中心最近的 patch。
    """
    # discovery 每簇的 prototype 形状（original space 均值）
    proto_shapes = np.zeros((k, disc_raw.shape[1]), dtype=np.float64)
    for c in range(k):
        idx = np.where(disc_labels == c)[0]
        if len(idx):
            proto_shapes[c] = np.mean(np.stack([z_normalize(disc_raw[i]) for i in idx]), axis=0)

    # validation patch：rep-space 分配 + shape-space 分配
    Xv = pca.transform(scaler.transform(val_emb))
    d_rep = np.linalg.norm(Xv[:, None, :] - centers[None, :, :], axis=2)  # [n, k]
    rep_assign = d_rep.argmin(axis=1)
    rep_dist = d_rep.min(axis=1)
    zval = np.stack([z_normalize(val_raw[i]) for i in range(len(val_raw))])
    d_shape = np.linalg.norm(zval[:, None, :] - proto_shapes[None, :, :], axis=2)
    shape_assign = d_shape.argmin(axis=1)

    agree = rep_assign == shape_assign
    coh = np.array([_zcorr(zval[i], proto_shapes[rep_assign[i]]) for i in range(len(zval))])
    is_coh = coh >= coh_thresh

    # 按数据集汇总（含 negative control 检查）
    datasets = sorted({m["dataset"] for m in val_meta})
    per_dataset = {}
    for ds in datasets:
        sel = np.array([i for i, m in enumerate(val_meta) if m["dataset"] == ds])
        per_dataset[ds] = {
            "n": int(len(sel)),
            "cross_space_agreement": round(float(np.mean(agree[sel])), 3),
            "shape_coherence": round(float(np.mean(is_coh[sel])), 3),
        }

    # headline：real held-out 的 shape coherence vs pure-noise control
    real_mask = np.array([m["dataset"] not in control_datasets for m in val_meta])
    real_coh = float(np.mean(is_coh[real_mask])) if real_mask.any() else 0.0
    noise_coh = per_dataset.get(noise_control, {}).get("shape_coherence")

    # 渲染：col 0 = 训练 prototype；其后**固定每列一个 held-out macro domain**，取该簇内该域里
    # rep-NN 离中心最近的 unseen patch。同一 domain 永远在同一列。
    val_domains = {m["macro_domain"] for m in val_meta}
    domain_order = [d for d in DOMAIN_COLORS if d != "Other" and d in val_domains]
    ncol = 1 + len(domain_order)
    fig, axes = plt.subplots(k, ncol, figsize=(2.0 * ncol, 1.45 * k), squeeze=False)
    panel_info = []
    for c in range(k):
        cand = np.where(rep_assign == c)[0]
        cand = cand[np.argsort(rep_dist[cand])]  # 按到中心距离升序
        best_by_dom: dict[str, int] = {}
        for i in cand:
            d = val_meta[int(i)]["macro_domain"]
            if d not in best_by_dom:  # cand 已排序 → 第一个即该域最近
                best_by_dom[d] = int(i)
        panel_info.append({"cluster": f"C{c + 1}", "n_assigned": int(len(cand)),
                           "domains_present": sorted(best_by_dom.keys())})
        # col 0：训练 prototype
        ax = axes[c, 0]
        ax.set_xticks([]); ax.set_yticks([])
        ax.plot(proto_shapes[c], lw=1.6, color="#c0392b", zorder=3)
        ax.set_ylabel(f"C{c + 1}", fontsize=9, rotation=0, labelpad=12, va="center")
        ax.set_facecolor("#fbecea")
        if c == 0:
            ax.set_title("train prototype", fontsize=7, color="#c0392b", fontweight="bold")
        # 其后各列：固定 domain
        for j, dom in enumerate(domain_order):
            ax = axes[c, j + 1]
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_title(dom, fontsize=7, color=DOMAIN_COLORS[dom], fontweight="bold")
            if dom not in best_by_dom:
                ax.set_facecolor("#f6f6f6")
                continue
            i = best_by_dom[dom]
            ax.plot(zval[i], lw=1.2, color="#1f2933", zorder=2)

    noise_txt = f"{noise_coh:.0%}" if noise_coh is not None else "n/a"
    fig.suptitle(
        f"Chronos-Bolt {layer_name} — generalization: train-discovered prototypes retrieve held-out patches\n"
        f"held-out shape coherence = {real_coh:.0%} (corr≥{coh_thresh} to discovered prototype) "
        f"vs pure-noise control = {noise_txt}; red = train prototype, black = unseen patch",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"output": str(out_path),
            "real_coherence": round(real_coh, 3),
            "noise_control_coherence": noise_coh,
            "cross_space_agreement_overall": round(float(np.mean(agree)), 3),
            "coherence_thresh": coh_thresh, "n_val_patches": int(len(zval)),
            "control_datasets": list(control_datasets),
            "per_dataset": per_dataset, "clusters": panel_info}


def render_nearest_panel(
    layer_name: str,
    scaler: Any,
    pca: Any,
    centers: np.ndarray,
    disc_labels: np.ndarray,
    disc_pca_coords: np.ndarray,
    disc_raw: np.ndarray,
    val_emb: np.ndarray,
    val_raw: np.ndarray,
    k: int,
    out_path: Path,
    n_each: int = 3,
) -> dict[str, Any]:
    """不走 cross-domain、只按距离：每簇一行 = [聚类中心原型 | n 个最近**训练** patch | n 个最近
    **held-out** patch]（都按 representation-space 到 cluster 中心的距离取最近）。蓝=训练、绿=held-out。
    """
    proto = np.zeros((k, disc_raw.shape[1]), dtype=np.float64)
    for c in range(k):
        idx = np.where(disc_labels == c)[0]
        if len(idx):
            proto[c] = np.mean(np.stack([z_normalize(disc_raw[i]) for i in idx]), axis=0)

    Xv = pca.transform(scaler.transform(val_emb))
    d_rep = np.linalg.norm(Xv[:, None, :] - centers[None, :, :], axis=2)
    rep_assign = d_rep.argmin(axis=1)
    rep_dist = d_rep.min(axis=1)
    zval = np.stack([z_normalize(val_raw[i]) for i in range(len(val_raw))])

    ncol = 1 + 2 * n_each
    fig, axes = plt.subplots(k, ncol, figsize=(2.0 * ncol, 1.45 * k), squeeze=False)
    info = []
    for c in range(k):
        didx = np.where(disc_labels == c)[0]
        dnear = didx[np.argsort(np.linalg.norm(disc_pca_coords[didx] - centers[c], axis=1))[:n_each]] if len(didx) else np.array([], int)
        vidx = np.where(rep_assign == c)[0]
        vnear = vidx[np.argsort(rep_dist[vidx])[:n_each]] if len(vidx) else np.array([], int)
        info.append({"cluster": f"C{c + 1}", "n_train": int(len(didx)), "n_val_assigned": int(len(vidx))})

        ax = axes[c, 0]
        ax.set_xticks([]); ax.set_yticks([])
        ax.plot(proto[c], lw=1.6, color="#c0392b")
        ax.set_facecolor("#fbecea")
        ax.set_ylabel(f"C{c + 1}", fontsize=9, rotation=0, labelpad=12, va="center")
        if c == 0:
            ax.set_title("cluster center", fontsize=7, color="#c0392b", fontweight="bold")
        for j in range(n_each):
            ax = axes[c, 1 + j]
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_title(f"train #{j + 1}", fontsize=7, color="#1f3a93", fontweight="bold")
            if j < len(dnear):
                ax.plot(z_normalize(disc_raw[dnear[j]]), lw=1.2, color="#1f3a93", zorder=2)
            else:
                ax.set_facecolor("#f6f6f6")
        for j in range(n_each):
            ax = axes[c, 1 + n_each + j]
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_title(f"held-out #{j + 1}", fontsize=7, color="#117a65", fontweight="bold")
            if j < len(vnear):
                ax.plot(proto[c], lw=0.9, color="#dddddd", zorder=1)
                ax.plot(zval[vnear[j]], lw=1.2, color="#117a65", zorder=2)
            else:
                ax.set_facecolor("#f6f6f6")

    fig.suptitle(
        f"Chronos-Bolt {layer_name} — by-distance nearest exemplars "
        f"(center | {n_each} nearest train | {n_each} nearest held-out)\n"
        "blue = training patches, green = held-out patches, grey = cluster-center reference",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"output": str(out_path), "clusters": info, "n_each": n_each}


def _extract(windows_raw, window_meta, layers, batch_size, pipe):
    """robust-z -> Bolt 提取 -> 返回 (windows_z, reps, patch_len)。pipe 复用，避免反复加载。"""
    windows_z = np.stack([robust_z(w) for w in windows_raw]).astype(np.float32)
    patch_len = int(pipe.model.chronos_config.input_patch_size)
    reps = extract_bolt_representations(
        windows_z, batch_size=batch_size, layers=layers, include_tokenizer=False,
        pipeline=pipe, keep_pipeline=True,
    )
    return windows_z, reps, patch_len


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-per-dataset", type=int, default=200,
                        help="discovery（训练数据）每个数据集采样窗口数")
    parser.add_argument("--val-windows-per-dataset", type=int, default=150,
                        help="validation（basicts held-out）每个数据集采样窗口数")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--card-layers", type=int, nargs="+", default=[0, 11])
    parser.add_argument("--prototype-layers", type=int, nargs="+", default=[0, 11])
    parser.add_argument("--generalization-layers", type=int, nargs="+", default=[0, 11])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--proto-per-cluster", type=int, default=6)
    parser.add_argument("--max-per-domain", type=int, default=400)
    parser.add_argument("--tsne-perplexity", type=float, default=40.0, help="cluster-map t-SNE perplexity (viz only)")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-cache", action="store_true", help="忽略提取缓存，强制重新跑 GPU 提取")
    parser.add_argument("--min-patch-std", type=float, default=0.15,
                        help="筛去近平直 patch：robust-z 窗口内 std < 此阈值的 patch 丢弃。"
                             "默认 0.15（开启，去掉 flat_low_information / 常数段的平直直线）；设 0 关闭。"
                             "论文须写明此预处理（见 docs/15 §5）。")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="输出文件名后缀，用于临时对比版本（如 _nofilter），避免覆盖 canonical 图")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sfx = args.out_suffix

    layers = sorted(set(args.card_layers) | set(args.prototype_layers) | set(args.generalization_layers))
    # 提取缓存：纯图形微调时跳过 GPU。key 跟"采样 + 提取"有关（与 k / 配色 / 版式无关）
    cache_path = args.out / ".cache" / (
        f"extract_train_wpd{args.windows_per_dataset}_val{args.val_windows_per_dataset}"
        f"_ctx{args.context_len}_seed{args.seed}_layers{'-'.join(map(str, layers))}.pkl"
    )
    if not args.no_cache and cache_path.exists():
        print(f"[main-fig] loading extraction cache -> {cache_path.name}")
        with open(cache_path, "rb") as fh:
            b = pickle.load(fh)
        (disc_z, disc_meta, disc_reps, val_z, val_meta_w, val_reps, patch_len) = (
            b["disc_z"], b["disc_meta"], b["disc_reps"],
            b["val_z"], b["val_meta_w"], b["val_reps"], b["patch_len"]
        )
    else:
        # discovery = Chronos in-distribution 训练子集
        print(f"[main-fig] sampling DISCOVERY (training) windows (per_dataset={args.windows_per_dataset})")
        disc_w, disc_meta, disc_summary = sample_training_windows(
            context_len=args.context_len, windows_per_dataset=args.windows_per_dataset, seed=args.seed
        )
        # validation = basicts held-out（剔除在训练集内的 Electricity/BeijingAirQuality）
        print(f"[main-fig] sampling VALIDATION (basicts held-out) windows (per_dataset={args.val_windows_per_dataset})")
        val_w_all, val_meta_all, _ = sample_windows(
            DATA_ROOT, context_len=args.context_len,
            windows_per_dataset=args.val_windows_per_dataset, seed=args.seed,
        )
        keep = [i for i, m in enumerate(val_meta_all) if m["dataset"] not in VALIDATION_EXCLUDE]
        val_w = val_w_all[keep]
        val_meta_w = [val_meta_all[i] for i in keep]
        print(f"[main-fig] discovery={len(disc_w)} train patches/windows | "
              f"validation={len(val_w)} held-out windows from "
              f"{len({m['dataset'] for m in val_meta_w})} datasets")

        pipe = load_bolt_pipeline()
        print(f"[main-fig] extracting Bolt layers {layers} (discovery + validation) ...")
        disc_z, disc_reps, patch_len = _extract(disc_w, disc_meta, layers, args.batch_size, pipe)
        val_z, val_reps, _ = _extract(val_w, val_meta_w, layers, args.batch_size, pipe)
        del pipe
        if not args.no_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump({"disc_z": disc_z, "disc_meta": disc_meta, "disc_reps": disc_reps,
                             "val_z": val_z, "val_meta_w": val_meta_w, "val_reps": val_reps,
                             "patch_len": patch_len, "disc_summary": disc_summary}, fh)
            print(f"[main-fig] cached extraction -> {cache_path.name}")

    summary: dict[str, Any] = {
        "model": "chronos-bolt-base", "patch_len": patch_len,
        "discovery_source": "chronos in-distribution training subset (16 datasets)",
        "validation_source": ("basicts held-out (excluded: Electricity/BeijingAirQuality in Chronos "
                              "pretraining; Traffic/ExchangeRate overlap discovery training subset)"),
        "config": vars(args) | {"out": str(args.out)},
        "cards": {}, "prototype_panel": {}, "generalization": {},
    }

    # 强制应用 VALIDATION_EXCLUDE（即使来自旧缓存）：剔除在 Chronos 预训练内 / 与 discovery 训练子集
    # 重叠的 basicts 数据集，避免泄漏。（缓存可能用旧的 exclude 采的，这里再过滤一次。）
    vkeep = [i for i, m in enumerate(val_meta_w) if m["dataset"] not in VALIDATION_EXCLUDE]
    if len(vkeep) < len(val_meta_w):
        dropped = sorted({m["dataset"] for m in val_meta_w if m["dataset"] in VALIDATION_EXCLUDE})
        val_z = val_z[vkeep]
        val_meta_w = [val_meta_w[i] for i in vkeep]
        val_reps = {kk: vv[vkeep] for kk, vv in val_reps.items()}
        print(f"[main-fig] validation exclude (post-cache) dropped {dropped}; "
              f"kept {len(vkeep)} windows from {len({m['dataset'] for m in val_meta_w})} datasets")

    # 各层 flatten（raw patch / meta 不随层变）
    disc_flat: dict[int, tuple] = {L: flatten_patches(disc_reps[f"layer_{L}"], disc_z, disc_meta, patch_len) for L in layers}
    val_flat: dict[int, tuple] = {L: flatten_patches(val_reps[f"layer_{L}"], val_z, val_meta_w, patch_len) for L in layers}

    # 可选：筛去近平直 patch（robust-z 窗口内 std < 阈值 = 几乎水平直线，flat_low_information / 常数段）
    if args.min_patch_std > 0:
        def _drop_flat(flat_by_layer):
            raw0 = flat_by_layer[layers[0]][1]
            keep = np.array([float(np.std(raw0[i])) >= args.min_patch_std for i in range(len(raw0))])
            out = {L: (e[keep], r[keep], [m[i] for i in range(len(m)) if keep[i]])
                   for L, (e, r, m) in flat_by_layer.items()}
            return out, keep
        disc_flat, keep_d = _drop_flat(disc_flat)
        val_flat, keep_v = _drop_flat(val_flat)
        summary["flat_filter"] = {"min_patch_std": args.min_patch_std,
                                  "disc_kept": int(keep_d.sum()), "disc_total": int(len(keep_d)),
                                  "val_kept": int(keep_v.sum()), "val_total": int(len(keep_v))}
        print(f"[main-fig] flat-filter std>={args.min_patch_std}: kept "
              f"{int(keep_d.sum())}/{len(keep_d)} disc, {int(keep_v.sum())}/{len(keep_v)} val patches")

    # discovery domain-balanced 子集（避免 Traffic 等高频域主导）
    meta0 = disc_flat[layers[0]][2]
    sel = select_domain_balanced_indices(meta0, max_per_domain=args.max_per_domain, seed=args.seed)
    summary["n_balanced_patches"] = int(len(sel))
    print(f"[main-fig] discovery domain-balanced subset: {len(sel)} patches")

    # 每层在 discovery balanced 子集上 fit（cards + generalization 共用同一聚类）
    layer_clusters: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fitted: dict[int, dict[str, Any]] = {}
    for L in sorted(set(args.card_layers) | set(args.generalization_layers)):
        emb, raw_patches, _meta = disc_flat[L]
        emb_b, raw_b = emb[sel], raw_patches[sel]
        labels, centers, pca_coords, scaler, pca, _km = cluster_pca_fit(emb_b, args.k, args.seed)
        fitted[L] = {"labels": labels, "centers": centers, "pca_coords": pca_coords,
                     "scaler": scaler, "pca": pca, "raw_b": raw_b}

    for L in args.card_layers:
        meta_b = [disc_flat[L][2][i] for i in sel]
        f = fitted[L]
        # 图里显示 1-based layer 号（Nature 习惯）= encoder block 索引 L + 1；代码/CLI/文件名仍 0-based
        layer_clusters[f"layer {L + 1}"] = (f["labels"], f["pca_coords"])
        out = args.out / f"main_A_cards_layer{L + 1}{sfx}.png"
        print(f"[main-fig] layer_{L}: rendering raw-only cards -> {out.name}")
        summary["cards"][f"layer_{L}"] = render_raw_cards(
            f"layer {L + 1}", f["labels"], f["centers"], f["pca_coords"], f["raw_b"], meta_b, args.k, args.top_n, out
        )

    # human motif taxonomy v0 标签（shapelet probe；只依赖 raw patch）+ macro domain（confounder）
    raw_b0 = disc_flat[layers[0]][1][sel]
    v0_labels = np.array([label_patch(raw_b0[i], patch_len).label for i in range(len(sel))])
    domain_labels = np.array([disc_flat[layers[0]][2][i]["macro_domain"] for i in sel])

    cmap_out = args.out / f"main_B_cluster_maps{sfx}.png"
    print(f"[main-fig] rendering cluster maps (cluster|motif v0|domain) ({list(layer_clusters)}) -> {cmap_out.name}")
    summary["cluster_maps"] = render_cluster_maps(
        layer_clusters, v0_labels, domain_labels, args.k, args.seed, args.tsne_perplexity, cmap_out
    )

    # cross-domain prototype（discovery 内，跨训练域）
    for Lp in args.prototype_layers:
        emb, raw_patches, meta = disc_flat[Lp]
        out = args.out / f"main_C_prototype_crossdomain_layer{Lp + 1}{sfx}.png"
        print(f"[main-fig] layer_{Lp}: rendering cross-domain prototype panel -> {out.name}")
        summary["prototype_panel"][f"layer_{Lp}"] = render_prototype_panel(
            emb, raw_patches, meta, args.k, args.seed, args.proto_per_cluster,
            args.max_per_domain, out, f"layer {Lp + 1}"
        )

    # ★ generalization：训练发现的 prototype 检索 unseen patch + cross-space 命中率
    for Lg in args.generalization_layers:
        f = fitted[Lg]
        v_emb, v_raw, v_meta = val_flat[Lg]
        out = args.out / f"main_D_generalization_heldout_layer{Lg + 1}{sfx}.png"
        print(f"[main-fig] layer_{Lg}: rendering generalization panel (unseen retrieval) -> {out.name}")
        info = render_generalization_panel(
            f"layer {Lg + 1}", f["scaler"], f["pca"], f["centers"], f["labels"], f["raw_b"],
            v_emb, v_raw, v_meta, args.k, args.proto_per_cluster, out,
        )
        summary["generalization"][f"layer_{Lg}"] = info

    # main_F：不走 cross-domain、只按距离的 nearest exemplars（center | 最近训练 | 最近 held-out）
    summary["nearest_panel"] = {}
    for Lg in args.generalization_layers:
        f = fitted[Lg]
        v_emb, v_raw, _vm = val_flat[Lg]
        out = args.out / f"main_F_nearest_exemplars_layer{Lg + 1}{sfx}.png"
        print(f"[main-fig] layer_{Lg}: rendering by-distance nearest panel -> {out.name}")
        summary["nearest_panel"][f"layer_{Lg}"] = render_nearest_panel(
            f"layer {Lg + 1}", f["scaler"], f["pca"], f["centers"], f["labels"],
            f["pca_coords"], f["raw_b"], v_emb, v_raw, args.k, out,
        )
        nc = info["noise_control_coherence"]
        nc_txt = f" vs noise control {nc:.0%}" if nc is not None else ""
        print(f"[main-fig]   layer_{Lg} held-out shape coherence = {info['real_coherence']:.0%}{nc_txt}")

    (args.out / "bolt_main_figure_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"[main-fig] saved modular PNGs + summary -> {args.out}")


if __name__ == "__main__":
    main()
