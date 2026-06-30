"""Step 3b — cross-architecture cluster coherence + cross-domain retrieval (two-space validation).

补齐第三个签名,凑齐论文 Fig.4 b/c/d 的跨架构对照。沿用 two-space distance principle:
在 *representation space* 用 PCA+KMeans 生成候选 primitive cluster,在 *original
time-series space* 验证(1)cluster 内部是否 shape-coherent,(2)cluster 是否跨域、
(3)retrieval 是否能跨域召回同形状邻居。三个架构族各自做,看签名是否复现。

去 OOD 化(见 cross-arch memory):共享池与各模型预训练有重叠(MOMENT 尤甚),因此
retrieval 只声称"表征空间支持跨域检索",不声称泛化到训练外。

产出(均房子风格:无 suptitle、SVG 可编辑、术语对齐主图/OOD 图):
- cross_arch_cluster_retrieval_summary.json —— 每模型 coherence-lift / domain-entropy /
  retrieval cross-domain corr & diversity。
- panel_b_repmaps.svg —— 3×N 表征图:training-data-domains / learned-temporal-primitives
  两张 t-SNE + cluster cards + cross-domain retrieval(stack 第 2 行;附录唯一视觉主图)。
  (原 panel_c 量化 bars 已删除:Nature 偏科普,量化指标改写入 summary JSON 供 caption 引用。)

从仓库根目录运行:
    .venv/bin/python figure_projects/cross_arch_generalization/run_cross_arch_cluster_retrieval.py
    # 重绘(免重算): 追加 --replot
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/temporal_primitive_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

TAB10 = plt.get_cmap("tab10")
MODEL_SHORT = {
    "chronos_bolt": "Chronos-Bolt\n(encoder–decoder)",
    "timesfm_2_5": "TimesFM\n(decoder-only)",
    "moment_1_large": "MOMENT\n(encoder-only)",
}
# focal patch（card 的 medoid 原型 / retrieval 的 query）统一用中性黑——**不用 model 色也不用
# domain 色**，避免与底部 Domain 调色板撞色歧义（model 身份已由行标签给出）。members 用浅灰。
FOCAL_COLOR = "#2b3038"
MEMBER_GRAY = "#b9bcc2"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_arch_style import (  # noqa: E402
    DOMAIN_COLORS,
    FS_LABEL,
    FS_LEGEND,
    FS_TICK,
    FS_TITLE,
    MODEL_ORDER,
    apply_house_rc,
    macro_domain,
)
from scripts.build_bolt_main_figure import cluster_pca_fit  # noqa: E402
from scripts.build_cluster_cards import medoid_indices  # noqa: E402
from scripts.cross_arch_shared_pool import MODELS, SHARED_WINDOW_LEN, build_shared_pool, extract_for_model  # noqa: E402
from scripts.run_second_pilot_discovery import robust_z, select_domain_balanced_indices  # noqa: E402

OUT_DIR = ROOT / "figure_projects" / "cross_arch_generalization"
CACHE = OUT_DIR / ".cache" / "cluster_retrieval_cache.pkl"

# patch-length-fair：各模型 patch_len 不同(8/16/32),原生长度直接算相关会偏向长 patch。
# 度量(coherence / retrieval shape-corr)统一把 raw patch 线性插值重采样到 32 再算。
# 视觉 cluster cards 仍用原生长度(诚实展示各自粒度)。
COMMON_PATCH_LEN = 32


def _resample(patch: np.ndarray, length: int = COMMON_PATCH_LEN) -> np.ndarray:
    n = len(patch)
    if n == length:
        return np.asarray(patch, dtype=np.float32)
    return np.interp(np.linspace(0.0, 1.0, length), np.linspace(0.0, 1.0, n), patch).astype(np.float32)


def _pair_corr_mean(z: np.ndarray) -> float:
    """z: (M, L) 已 robust-z 的 patch；返回成对 Pearson 相关均值。"""
    if len(z) < 2:
        return 0.0
    c = np.corrcoef(z)
    iu = np.triu_indices(len(z), k=1)
    vals = c[iu]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else 0.0


def _domain_entropy(domains: np.ndarray) -> float:
    """归一化 Shannon 熵(0=单域,1=均匀跨域)。"""
    _, counts = np.unique(domains, return_counts=True)
    p = counts / counts.sum()
    h = -(p * np.log(p + 1e-12)).sum()
    return float(h / np.log(len(p))) if len(p) > 1 else 0.0


def analyze_model(model_key: str, windows: np.ndarray, window_meta: list[dict[str, Any]], args) -> dict[str, Any]:
    deepest = MODELS[model_key]["layers"][-1]
    patch_len = MODELS[model_key]["patch_len"]
    num_patches = MODELS[model_key]["num_patches"]
    reps = extract_for_model(model_key, windows, batch_size=args.batch_size, layers=[deepest])
    rep = reps[f"layer_{deepest}"]                       # (N, P, d)
    flat = rep.reshape(rep.shape[0] * num_patches, rep.shape[2])
    raw = windows.reshape(windows.shape[0], num_patches, patch_len).reshape(-1, patch_len)
    dom = np.array([macro_domain(window_meta[i // num_patches]["domain"]) for i in range(flat.shape[0])])
    win_of = np.repeat(np.arange(windows.shape[0]), num_patches)

    # patch 级 domain 均衡(聚类/检验子集)
    sel = select_domain_balanced_indices([{"domain": d} for d in dom], max_per_domain=args.max_per_domain, seed=args.seed)
    Xsel, raw_sel, dom_sel, win_sel = flat[sel], raw[sel], dom[sel], win_of[sel]
    z_sel = np.stack([robust_z(p) for p in raw_sel]).astype(np.float32)            # 原生长度(视觉 cards 用)
    z32_sel = np.stack([robust_z(_resample(p)) for p in raw_sel]).astype(np.float32)  # 重采样到 32(度量用,patch-length-fair)

    labels, _centers, coords, _sc, _pca, _km = cluster_pca_fit(Xsel, k=args.k, seed=args.seed)

    # cluster coherence lift + domain entropy（coherence 用 z32_sel：消除 patch_len 偏差）
    rng = np.random.default_rng(args.seed)
    per_cluster = []
    for c in range(args.k):
        idx = np.where(labels == c)[0]
        med = medoid_indices(coords, labels, c, args.card_n)
        coh = _pair_corr_mean(z32_sel[med])
        rand = z32_sel[rng.choice(len(z32_sel), size=min(args.card_n, len(z32_sel)), replace=False)]
        baseline = _pair_corr_mean(rand)
        ent = _domain_entropy(dom_sel[idx])
        per_cluster.append({"cluster": c, "size": int(len(idx)), "coherence": coh,
                            "baseline": baseline, "coherence_lift": coh - baseline,
                            "domain_entropy": ent, "n_domains": int(len(set(dom_sel[idx])))})

    # cross-domain retrieval：query → top-k 最近邻(排除同窗),统计跨域比例与跨域邻居形状相关
    q_idx = rng.choice(len(Xsel), size=min(args.n_queries, len(Xsel)), replace=False)
    xd_frac, xd_corr = [], []
    for qi in q_idx:
        d = np.linalg.norm(coords - coords[qi], axis=1)
        d[win_sel == win_sel[qi]] = np.inf            # 排除同窗 patch
        order = np.argsort(d)[: args.knn]
        is_xd = dom_sel[order] != dom_sel[qi]
        xd_frac.append(float(is_xd.mean()))
        if is_xd.any():
            qz = z32_sel[qi]  # 重采样到 32 后算相关(patch-length-fair)
            corrs = [np.corrcoef(qz, z32_sel[j])[0, 1] for j in order[is_xd] if np.std(z32_sel[j]) > 1e-8 and np.std(qz) > 1e-8]
            if corrs:
                xd_corr.append(float(np.nanmean(corrs)))

    metrics = {
        "mean_coherence_lift": float(np.mean([p["coherence_lift"] for p in per_cluster])),
        "mean_domain_entropy": float(np.mean([p["domain_entropy"] for p in per_cluster])),
        "retrieval_cross_domain_fraction": float(np.mean(xd_frac)),
        "retrieval_cross_domain_shape_corr": float(np.mean(xd_corr)) if xd_corr else float("nan"),
    }
    print(f"[xarch-cr] {model_key:>16}: coh-lift={metrics['mean_coherence_lift']:.3f} "
          f"dom-entropy={metrics['mean_domain_entropy']:.3f} "
          f"xd-frac={metrics['retrieval_cross_domain_fraction']:.3f} "
          f"xd-corr={metrics['retrieval_cross_domain_shape_corr']:.3f}")

    # ---- 视觉素材(子采样做 t-SNE)----
    sub = rng.choice(len(Xsel), size=min(args.tsne_n, len(Xsel)), replace=False)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=args.seed,
                metric="euclidean").fit_transform(coords[sub])
    # 存全部 cluster 的原生长度 medoid 形状 + 选择键(card 选择移到 figure，便于按需重选)
    cluster_cards_all = []
    for pc in per_cluster:
        med = medoid_indices(coords, labels, pc["cluster"], args.card_n)
        cluster_cards_all.append({"cluster": pc["cluster"], "n_domains": pc["n_domains"],
                                  "domain_entropy": pc["domain_entropy"],
                                  "coherence_lift": pc["coherence_lift"], "size": pc["size"],
                                  "z_members": z_sel[med], "patch_len": patch_len})
    # 一个跨域 retrieval 例子(选一个有跨域邻居的 query)
    retr = None
    for qi in q_idx:
        d = np.linalg.norm(coords - coords[qi], axis=1)
        d[win_sel == win_sel[qi]] = np.inf
        order = np.argsort(d)[: args.knn]
        is_xd = dom_sel[order] != dom_sel[qi]
        if is_xd.sum() >= 3:
            retr = {"q_z": z_sel[qi], "q_domain": dom_sel[qi], "patch_len": patch_len,
                    "nb_z": z_sel[order[is_xd][: args.retr_show]],
                    "nb_domains": dom_sel[order[is_xd][: args.retr_show]].tolist()}
            break

    return {"metrics": metrics, "per_cluster": per_cluster,
            "tsne": tsne, "tsne_dom": dom_sel[sub], "tsne_clu": labels[sub],
            "cluster_cards_all": cluster_cards_all, "retrieval": retr, "k": args.k}


# ----------------------------- figures -----------------------------
# 注：原 fig_quant（panel c：coherence-lift / domain-entropy / retrieval 量化 bars）已移除——
# Nature 风格偏科普，过多量化图表反而影响阅读；附录改为 panel a（signatures）+ panel b（视觉）。
# 量化指标仍写入 cross_arch_cluster_retrieval_summary.json，供 PPT caption 引用。
def _pick_cards(cards_all: list[dict[str, Any]], n_cards: int) -> list[dict[str, Any]]:
    """选 candidate primitive family(narrative §7):跨域(≥2 域,自动排除 synthetic 负控
    单域簇)且 shape-coherence 最高。避开"最跨域但 coherence≈0"的大杂簇(其 z-norm 均值
    会塌成直线、信息量低)。fallback:若没有跨域簇则退回按 coherence 排序。"""
    def _std0(c: dict[str, Any]) -> float:
        return float(np.std(np.asarray(c["z_members"])[0]))  # medoid(focal)的形状幅度
    xdom = [c for c in cards_all if c["n_domains"] >= 2]
    # 跳过近常数(flat)medoid:flat/level 在 z-norm 上 std≈0,不是可展示的"形状" primitive。
    pool = [c for c in xdom if _std0(c) > 0.25]
    if len(pool) < n_cards:                                   # 不够再放宽
        pool = xdom if len(xdom) >= n_cards else list(cards_all)
    return sorted(pool, key=lambda c: -c["coherence_lift"])[:n_cards]


def _spark(ax, z_members: np.ndarray, color: str) -> None:
    # z_members 由 medoid_indices 按"到簇中心距离"升序排列 → z_members[0] 是 medoid。
    # prototype 用 medoid 这个**真实中心样本**(不是均值:对相位错开的成员取均值会塌成
    # 直线、且不对应任何真实 patch)。其余成员画作浅灰背景,展示该 cluster 的形状散布。
    t = np.linspace(0, 1, z_members.shape[1])
    for z in z_members[1:]:
        ax.plot(t, z, color=MEMBER_GRAY, lw=0.6, alpha=0.55)
    ax.plot(t, z_members[0], color=color, lw=2.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#cfd4dc")


def fig_repmaps(per_model: dict[str, dict[str, Any]], out_svg: Path, n_cards: int, retr_show: int) -> None:
    apply_house_rc()
    keys = [m for m in MODEL_ORDER if m in per_model]
    ncol = 2 + n_cards + 1                                  # domains-map | primitives-map | cards | retrieval
    fig, axes = plt.subplots(len(keys), ncol, figsize=(3.0 * ncol, 3.0 * len(keys)))
    if len(keys) == 1:
        axes = axes[None, :]

    for r, mk in enumerate(keys):
        data = per_model[mk]
        xy = np.asarray(data["tsne"])
        # col0: representation map colored by training-data domain
        ax = axes[r, 0]
        for dom in sorted(set(data["tsne_dom"])):
            m = data["tsne_dom"] == dom
            ax.scatter(xy[m, 0], xy[m, 1], s=5, color=DOMAIN_COLORS.get(dom, DOMAIN_COLORS["Other"]), alpha=0.6, lw=0)
        # col1: representation map colored by model-derived cluster
        axc = axes[r, 1]
        clu = np.asarray(data["tsne_clu"])
        for c in sorted(set(clu.tolist())):
            m = clu == c
            axc.scatter(xy[m, 0], xy[m, 1], s=5, color=TAB10(c % 10), alpha=0.6, lw=0)
        for ax_ in (ax, axc):
            ax_.set_xticks([]); ax_.set_yticks([])
        # cards: candidate primitive family = 跨域(≥2 域) 且最 shape-coherent
        for i, card in enumerate(_pick_cards(data["cluster_cards_all"], n_cards)):
            axk = axes[r, 2 + i]
            _spark(axk, np.asarray(card["z_members"]), FOCAL_COLOR)
            axk.set_title(f"C{card['cluster'] + 1}", fontsize=FS_TICK)
        # retrieval：黑粗线=query patch;细线=其表征空间最近邻里来自**其它域**的 patch,
        # 按各自 domain 上色(复用底部 Domain 图例)——多种域色一眼即 cross-domain,无需 caption。
        # 文字量/位置与前两列统一:仅 row-0 顶部单标题,不写格内图例/底部 caption。
        axr = axes[r, ncol - 1]
        retr = data["retrieval"]
        if retr is not None:
            t = np.linspace(0, 1, len(retr["q_z"]))
            for z, ndom in zip(np.asarray(retr["nb_z"]), retr["nb_domains"]):
                axr.plot(np.linspace(0, 1, len(z)), z,
                         color=DOMAIN_COLORS.get(ndom, DOMAIN_COLORS["Other"]), lw=1.3, alpha=0.5)
            axr.plot(t, retr["q_z"], color=FOCAL_COLOR, lw=2.4)
        axr.set_xticks([]); axr.set_yticks([])
        for sp in axr.spines.values():
            sp.set_color("#cfd4dc")
        # 顶行写一次列标题(术语对齐主图 panel b)
        if r == 0:
            axes[r, 0].set_title("Training data domains", fontsize=FS_TITLE - 4)
            axes[r, 1].set_title("Model-derived pattern groups", fontsize=FS_TITLE - 4)
            axr.set_title("Cross-domain retrieval", fontsize=FS_TITLE - 4)
        # 行标签(最左,旋转):模型 + 聚类所用的最深层(1-based,与 cross_arch_style 守则一致)
        final_layer = MODELS[mk]["layers"][-1] + 1
        axes[r, 0].annotate(f"{MODEL_SHORT[mk]}\nlayer {final_layer}", xy=(-0.30, 0.5),
                            xycoords="axes fraction", rotation=90, ha="center", va="center",
                            fontsize=FS_LABEL - 3, fontweight="bold")
    # 底部两个图例：上=Domain(点;用于 col0 域图 + col5 邻居域色),
    # 下=黑色焦点线(card 的 medoid 原型 / retrieval 的 query) —— 灰色 members 不入图例(自明)。
    seen = sorted({d for mk in keys for d in set(per_model[mk]["tsne_dom"])})
    dom_handles = [Line2D([0], [0], marker="o", ls="", ms=8,
                          color=DOMAIN_COLORS.get(d, DOMAIN_COLORS["Other"]), label=d) for d in seen]
    line_handles = [Line2D([0], [0], color=FOCAL_COLOR, lw=2.4, label="prototype / query patch")]
    fig.tight_layout(rect=(0, 0.072, 1, 1))
    fig.legend(handles=dom_handles, loc="lower center", ncol=len(seen), fontsize=FS_LEGEND - 2,
               title="Domain", title_fontsize=FS_LEGEND - 1, frameon=False,
               bbox_to_anchor=(0.5, 0.018))
    fig.legend(handles=line_handles, loc="lower center", ncol=1, fontsize=FS_LEGEND - 2,
               frameon=False, bbox_to_anchor=(0.5, -0.004))
    out_png = out_svg.with_suffix(".png")
    out_pdf = out_svg.with_suffix(".pdf")  # vector PDF for direct paper inclusion (fig_cross_arch)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.12)
    print(f"[xarch-cr] saved -> {out_svg} (+ .png, .pdf)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows-per-dataset", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-per-domain", type=int, default=600)
    ap.add_argument("--k", type=int, default=6,
                    help="KMeans k；固定为 6 与 pub_main_figure_fullrep 对齐,三模型同 k 才可比")
    ap.add_argument("--card-n", type=int, default=16)
    ap.add_argument("--n-cards", type=int, default=2)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--retr-show", type=int, default=6)
    ap.add_argument("--tsne-n", type=int, default=3000)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    if args.replot and CACHE.exists():
        per_model = pickle.load(open(CACHE, "rb"))
        print(f"[xarch-cr] replot from cache {CACHE.name}")
    else:
        print(f"[xarch-cr] building shared pool (per_dataset={args.windows_per_dataset})")
        windows, window_meta, _ = build_shared_pool(
            window_len=SHARED_WINDOW_LEN, windows_per_dataset=args.windows_per_dataset, seed=args.seed)
        print(f"[xarch-cr] pool: {windows.shape}")
        per_model = {mk: analyze_model(mk, windows, window_meta, args) for mk in MODELS}
        pickle.dump(per_model, open(CACHE, "wb"))

    summary = {
        "experiment": "cross-architecture cluster coherence + cross-domain retrieval (step 3b)",
        "method_note": ("Late-layer patch reps per model -> PCA(30)+KMeans(k); validate in original "
                        "space: cluster shape-coherence lift over random, cluster domain-entropy, "
                        "cross-domain retrieval fraction & shape-corr. Shared 512 pool, patch-level "
                        "domain-balanced. PATCH-LENGTH-FAIR: raw patches (8/16/32) linearly resampled to "
                        f"L={COMMON_PATCH_LEN} before any shape-correlation (coherence & retrieval shape-corr), "
                        "so the metric is not biased toward longer patches; visual cluster cards keep native "
                        "patch length. Coherence uses the DEEPEST layer — for Chronos-Bolt the final layer "
                        "reorganizes by forecasting role (paper Fig.4), so its shape-coherence is partial by "
                        "design, not a defect. Pool overlaps each model's pretraining (esp. MOMENT) — "
                        "retrieval = 'representation supports cross-domain retrieval', NOT OOD generalization."),
        "config": vars(args),
        "models": {mk: {"family": MODELS[mk]["family"], "metrics": per_model[mk]["metrics"],
                        "per_cluster": per_model[mk]["per_cluster"]} for mk in per_model},
    }
    (OUT_DIR / "cross_arch_cluster_retrieval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[xarch-cr] saved summary -> {OUT_DIR / 'cross_arch_cluster_retrieval_summary.json'}")

    fig_repmaps(per_model, OUT_DIR / "panel_b_repmaps.svg", args.n_cards, args.retr_show)


if __name__ == "__main__":
    main()
