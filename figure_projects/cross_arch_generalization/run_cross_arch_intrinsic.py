"""Step 3 — cross-architecture intrinsic-signature comparison (Bolt / TimesFM / MOMENT).

跨架构泛化附录的第一组真实结果。在**同一批 512 点共享池窗口**上,对三个架构族各自
计算两个最敏感的 intrinsic 上下文化签名,看签名形状是否随深度跨架构一致复现:

1. **Contextualization probe** —— 每层做 10-NN probe accuracy,预测 confounder
   (domain / frequency / position)。随深度越来越可解码 = backbone 把 system-level
   信息重新注入 local patch。指标函数直接复用 run_bolt_contextualization_training。
2. **Selective convergence** —— centered-cosine,同窗(同 context 不同位置)patch 相似度
   vs 跨窗(不同 context 随机)patch 相似度。gap 随深度增大 = contextualization 的签名。

横轴 = **相对深度**(tokenizer=0,各模型末层=1),三个模型三条线对齐叠加。
方法论守则(见 cross-arch memory):只比"各模型各自的深度趋势形状",不比绝对数值、
不比 cluster 身份、不声称泛化到训练外。position 的 chance=1/num_patches 各模型不同
(patch 数 32/16/64),图中按模型画各自 chance 虚线。

从仓库根目录运行:
    .venv/bin/python figure_projects/cross_arch_generalization/run_cross_arch_intrinsic.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/temporal_primitive_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.cross_arch_shared_pool import (  # noqa: E402
    MODELS,
    SHARED_WINDOW_LEN,
    build_shared_pool,
    extract_for_model,
)
from scripts.run_bolt_contextualization_training import (  # noqa: E402
    knn_accuracy,
    within_context_similarity,
)
from scripts.run_second_pilot_discovery import select_domain_balanced_indices  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_arch_style import (  # noqa: E402
    ATTR_LABEL,
    FS_LABEL,
    FS_LEGEND,
    FS_TICK,
    FS_TITLE,
    MODEL_COLOR,
    MODEL_LABEL,
    MODEL_ORDER,
    apply_house_rc,
)

OUT_DIR = ROOT / "figure_projects" / "cross_arch_generalization"

# 每个模型在自己深度上等距取 5 层(含首末);见脚本顶部 np.linspace 推导。
PROBE_LAYERS: dict[str, list[int]] = {
    "chronos_bolt": [0, 3, 6, 8, 11],   # 12 blocks
    "timesfm_2_5": [0, 5, 10, 14, 19],  # 20 layers
    "moment_1_large": [0, 6, 12, 17, 23],  # 24 blocks
}
NUM_BLOCKS: dict[str, int] = {"chronos_bolt": 12, "timesfm_2_5": 20, "moment_1_large": 24}


def patch_labels(window_meta: list[dict[str, Any]], num_patches: int) -> dict[str, np.ndarray]:
    """(window, patch) 顺序展开的 domain / frequency / position 标签。"""
    dom, freq, pos = [], [], []
    for meta in window_meta:
        fm = meta.get("frequency_minutes")
        for p in range(num_patches):
            dom.append(str(meta["domain"]))
            freq.append("none" if fm is None else str(fm))
            pos.append(p)
    return {"domain": np.asarray(dom), "frequency": np.asarray(freq), "position": np.asarray(pos)}


def rel_depth(model_key: str, rep: str) -> float:
    """相对深度:tokenizer=0,encoder/transformer block i -> (i+1)/num_blocks。"""
    if rep == "tokenizer":
        return 0.0
    layer_idx = int(rep.split("_")[1])
    return (layer_idx + 1) / NUM_BLOCKS[model_key]


def analyze_model(model_key: str, windows: np.ndarray, window_meta: list[dict[str, Any]], args) -> dict[str, Any]:
    layers = PROBE_LAYERS[model_key]
    reps = extract_for_model(model_key, windows, batch_size=args.batch_size, layers=layers)
    num_patches = MODELS[model_key]["num_patches"]
    labels = patch_labels(window_meta, num_patches)

    # patch 级 domain-balanced 子集(probe 用,避免高频域主导)
    patch_meta = [{"domain": d} for d in labels["domain"]]
    sel = select_domain_balanced_indices(patch_meta, max_per_domain=args.max_per_domain, seed=args.seed)
    freq_ok = labels["frequency"][sel] != "none"  # frequency probe 剔除 synthetic

    def code(arr: np.ndarray) -> np.ndarray:
        return np.unique(arr, return_inverse=True)[1]

    rep_order = (["tokenizer"] if "tokenizer" in reps else []) + [f"layer_{L}" for L in layers]
    out: dict[str, Any] = {}
    for rep in rep_order:
        emb = reps[rep]  # (N, P, d)
        flat = emb.reshape(emb.shape[0] * emb.shape[1], emb.shape[2])
        Xp = PCA(n_components=min(30, flat.shape[1]), random_state=args.seed).fit_transform(
            StandardScaler().fit_transform(flat[sel])
        )
        acc = {
            "domain": round(knn_accuracy(Xp, code(labels["domain"][sel]), args.knn), 4),
            "frequency": round(knn_accuracy(Xp[freq_ok], code(labels["frequency"][sel][freq_ok]), args.knn), 4),
            "position": round(knn_accuracy(Xp, code(labels["position"][sel]), args.knn), 4),
        }
        sim = within_context_similarity(emb, args.global_pairs, args.seed)
        out[rep] = {"rel_depth": round(rel_depth(model_key, rep), 4),
                    "knn_probe_acc": acc, "within_context_sim": sim}
        print(f"[xarch] {model_key:>16} {rep:<10} d={out[rep]['rel_depth']:.2f} "
              f"dom={acc['domain']:.3f} freq={acc['frequency']:.3f} pos={acc['position']:.3f} "
              f"conv-gap={sim['gap']:.3f}")

    chance = {
        "domain": float(np.bincount(code(labels["domain"][sel])).max() / sel.size),
        "frequency": float(np.bincount(code(labels["frequency"][sel][freq_ok])).max() / max(1, int(freq_ok.sum()))),
        "position": 1.0 / num_patches,
    }
    return {"rep_order": rep_order, "num_patches": num_patches, "chance": chance, "results": out}


def _model_keys(per_model: dict[str, dict[str, Any]]) -> list[str]:
    return [m for m in MODEL_ORDER if m in per_model]


def make_figure(per_model: dict[str, dict[str, Any]], out_svg: Path) -> None:
    """房子风格:不画 suptitle(caption 用户自写)、SVG 可编辑、术语对齐 panel_a。"""
    apply_house_rc()
    keys = _model_keys(per_model)
    # 与 panel_b_repmaps 共享公共宽度（vertical-stack 对齐，~1090pt tight-bbox）。
    fig, axes = plt.subplots(1, 4, figsize=(15.15, 4.0))

    for ax, conf in zip(axes[:3], ["domain", "frequency", "position"]):
        for mk in keys:
            data = per_model[mk]
            xs = [data["results"][r]["rel_depth"] for r in data["rep_order"]]
            ys = [data["results"][r]["knn_probe_acc"][conf] for r in data["rep_order"]]
            ax.plot(xs, ys, "-o", color=MODEL_COLOR[mk], lw=2.4, ms=7, label=MODEL_LABEL[mk])
            ax.axhline(data["chance"][conf], ls=":", color=MODEL_COLOR[mk], lw=1.1, alpha=0.55)
        ax.set_title(ATTR_LABEL[conf], fontsize=FS_TITLE)
        ax.set_ylabel("k-NN probe accuracy  ↑", fontsize=FS_LABEL)

    ax = axes[3]
    for mk in keys:
        data = per_model[mk]
        xs = [data["results"][r]["rel_depth"] for r in data["rep_order"]]
        ys = [data["results"][r]["within_context_sim"]["gap"] for r in data["rep_order"]]
        ax.plot(xs, ys, "-o", color=MODEL_COLOR[mk], lw=2.4, ms=7)
    ax.axhline(0.0, ls=":", color="#777777", lw=1.1)
    ax.set_title("Within-context coherence", fontsize=FS_TITLE)
    ax.set_ylabel("Same − different similarity  ↑", fontsize=FS_LABEL)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.grid(True, axis="y", alpha=0.25)

    # 架构图例：stack 三面板共享同一组架构颜色，故只在 panel a 写一次，放底部横排
    # （图形级，不挤占任何子图；dotted = per-model chance 由 caption 说明）。
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=MODEL_COLOR[mk], lw=2.4, marker="o", ms=7,
                      label=MODEL_LABEL[mk]) for mk in keys]

    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.supxlabel("Relative depth   (Tokenizer → final layer)", fontsize=FS_LABEL, y=0.10)
    fig.legend(handles=handles, loc="lower center", ncol=len(keys), fontsize=FS_LEGEND,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    out_png = out_svg.with_suffix(".png")
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print(f"[xarch] saved -> {out_svg}")
    print(f"[xarch] preview -> {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows-per-dataset", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-per-domain", type=int, default=400)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--global-pairs", type=int, default=400000)
    ap.add_argument("--max-per-domain-pool", type=int, default=None,
                    help="可选:对共享池做 window 级 domain 均衡(默认不做,probe 内部已 patch 级均衡)")
    ap.add_argument("--replot", action="store_true",
                    help="只从已存 summary JSON 重绘图(房子风格打磨用,免重算)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.replot:
        summary = json.loads((OUT_DIR / "cross_arch_intrinsic_summary.json").read_text(encoding="utf-8"))
        make_figure(summary["per_model"], OUT_DIR / "panel_a_intrinsic_signatures.svg")
        return

    print(f"[xarch] building shared pool (window_len={SHARED_WINDOW_LEN}, per_dataset={args.windows_per_dataset})")
    windows, window_meta, dataset_summary = build_shared_pool(
        window_len=SHARED_WINDOW_LEN,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
        max_per_domain=args.max_per_domain_pool,
    )
    domains = sorted({m["domain"] for m in window_meta})
    print(f"[xarch] pool: {windows.shape} over {len(domains)} domains: {domains}")

    per_model: dict[str, dict[str, Any]] = {}
    for mk in MODELS:
        per_model[mk] = analyze_model(mk, windows, window_meta, args)

    summary = {
        "experiment": "cross-architecture intrinsic contextualization signatures (step 3)",
        "shared_pool": {"window_len": SHARED_WINDOW_LEN, "n_windows": int(windows.shape[0]),
                        "domains": domains, "windows_per_dataset": args.windows_per_dataset,
                        "seed": args.seed},
        "models": {mk: {"family": MODELS[mk]["family"], "objective": MODELS[mk]["objective"],
                        "patch_len": MODELS[mk]["patch_len"], "num_patches": MODELS[mk]["num_patches"],
                        "probe_layers": PROBE_LAYERS[mk]} for mk in MODELS},
        "method_note": ("Shared 512-point pool; each model patchifies natively (32/16/64). "
                        "10-NN probe accuracy (PCA-30 + StandardScaler, patch-level domain-balanced) for "
                        "domain/frequency/position; within-context centered-cosine gap for selective "
                        "convergence. Compare qualitative depth-trend shape across families, NOT absolute "
                        "values. Pool overlaps each model's pretraining (esp. MOMENT) — no OOD claim."),
        "config": vars(args),
        "per_model": per_model,
        "dataset_summary": dataset_summary,
    }
    (OUT_DIR / "cross_arch_intrinsic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[xarch] saved summary -> {OUT_DIR / 'cross_arch_intrinsic_summary.json'}")

    make_figure(per_model, OUT_DIR / "panel_a_intrinsic_signatures.svg")


if __name__ == "__main__":
    main()
