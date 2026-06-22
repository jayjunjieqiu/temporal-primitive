"""Task 3 — Chronos-Bolt forecasting probe: backbone vs tokenizer.

老师反馈：用特征"回归预测未来"来证明 backbone 后的 representation 比 tokenizer 更有用
（不是分类 / SVM，而是 forecasting probe）。

本脚本支持两种 probe（``--mode``）：

1. ``horizon`` （默认，对齐 Chronos-Bolt 本职 forecasting，撑主线 1 "TSFM 有用"）
   - 对整段 context 做 pooling 得到一个序列级 representation：
       tokenizer  : input_patch_embedding 输出 pooled（bag-of-patch，无 patch 间交互）
       layer_{L}  : encoder.block[L] hidden pooled（attention 已让 patch 互相 contextualize）
   - 用 frozen Ridge linear probe 预测**窗口之后真正的未来 H 步**（genuine future）。
   - 归一化只用 context 的 median/MAD（不泄漏未来）。MASE 基线 = persistence
     （context 最后一个值重复 H 步）。
   - 公平点：tokenizer 与 backbone 都 pool 了整段 context，差别只在 backbone 经过
     attention 做了 contextualization，因此 backbone 赢 = contextualization 对预测有用。

2. ``next_patch`` （诊断 / 负例，撑主线 2 "随层越来越抽象"）
   - 每个 patch 位置的 representation 线性解码紧接的下一个 patch 原始值。
   - 已知结果：layer_0 最好，越深越差——深层 hidden 为 decoder cross-attention 服务，
     是 contextualized/abstract 表示，不为"线性还原下一 patch 原始值"优化。该负例须保留。

注意：Chronos-Bolt 是 archived Chronos-2 pilot 的 clean 后继模型，input token 不含 time
encoding（见 docs/99_...）。结果可作为 clean evidence。

从仓库根目录运行：
    .venv/bin/python scripts/run_bolt_forecasting_probe.py                 # horizon
    .venv/bin/python scripts/run_bolt_forecasting_probe.py --mode next_patch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.chronos_bolt_backbone import extract_bolt_representations, load_bolt_pipeline  # noqa: E402
from scripts.run_prior_guided_probe_sanity_check import macro_domain  # noqa: E402
from scripts.run_second_pilot_discovery import DATA_ROOT, sample_windows  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "bolt_forecasting_probe"


def robust_z_with(x: np.ndarray, loc: float, scale: float, clip: float = 10.0) -> np.ndarray:
    """用给定 loc/scale（context 统计量）做 z-score 并 clip（避免泄漏未来）。

    用 mean/std（与 Chronos-Bolt 内部 instance_norm 同款）而非 median/MAD：近常数
    context 的 MAD 可能为 0，会把 target 放大到天文数字；clip 到 ±clip 防 flat-context
    病态窗口（intermittent 序列）主导 MAE，这也是 Chronos/TimesFM 的常规做法。
    """
    z = (x - loc) / scale
    return np.clip(z, -clip, clip).astype(np.float32)


def context_stats(ctx: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    loc = float(np.mean(ctx))
    scale = float(np.std(ctx))
    return loc, max(scale, eps)


def pool_rep(rep_window: np.ndarray, mode: str) -> np.ndarray:
    """对一个窗口的 patch 维做 pooling。rep_window: (num_patches, dim)。"""
    if mode == "mean":
        return rep_window.mean(axis=0)
    if mode == "last":
        return rep_window[-1]
    raise ValueError(f"unknown pooling: {mode}")


# ---------------------------------------------------------------------------
# horizon mode
# ---------------------------------------------------------------------------
def build_horizon_dataset(
    full_windows: np.ndarray,
    context_len: int,
    horizon: int,
    reps: dict[str, np.ndarray],
    window_meta: list[dict[str, Any]],
    pooling: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """每个窗口一个样本：pooled context representation -> 真实未来 H 步。"""
    rep_names = list(reps.keys())
    X_lists: dict[str, list[np.ndarray]] = {"raw_last_patch": [], **{r: [] for r in rep_names}}
    y_list: list[np.ndarray] = []
    naive_list: list[np.ndarray] = []
    win_ids: list[int] = []
    macros: list[str] = []
    patch_len = context_len // reps[rep_names[0]].shape[1]

    for i in range(len(full_windows)):
        ctx = full_windows[i, :context_len]
        fut = full_windows[i, context_len : context_len + horizon]
        med, mad = context_stats(ctx)
        fut_z = robust_z_with(fut, med, mad)
        ctx_z = robust_z_with(ctx, med, mad)

        X_lists["raw_last_patch"].append(ctx_z[-patch_len:])  # linear-AR 锚点
        for r in rep_names:
            X_lists[r].append(pool_rep(reps[r][i], pooling))
        y_list.append(fut_z)
        naive_list.append(np.full(horizon, float(ctx_z[-1]), dtype=np.float32))
        win_ids.append(int(window_meta[i]["window_id"]))
        macros.append(macro_domain(window_meta[i].get("domain")))

    X_by_rep = {k: np.asarray(v, dtype=np.float32) for k, v in X_lists.items()}
    return (
        X_by_rep,
        np.asarray(y_list, dtype=np.float32),
        np.asarray(naive_list, dtype=np.float32),
        np.asarray(win_ids),
        np.asarray(macros),
    )


# ---------------------------------------------------------------------------
# next_patch mode (diagnostic / negative)
# ---------------------------------------------------------------------------
def build_next_patch_dataset(
    windows_z: np.ndarray,
    reps: dict[str, np.ndarray],
    window_meta: list[dict[str, Any]],
    patch_len: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_windows, ctx_len = windows_z.shape
    num_patches = ctx_len // patch_len
    rep_names = list(reps.keys())
    X_lists: dict[str, list[np.ndarray]] = {"raw_patch": [], **{r: [] for r in rep_names}}
    y_list: list[np.ndarray] = []
    naive_list: list[np.ndarray] = []
    win_ids: list[int] = []
    macros: list[str] = []

    for i in range(n_windows):
        macro = macro_domain(window_meta[i].get("domain"))
        for p in range(num_patches - 1):
            cur = windows_z[i, p * patch_len : (p + 1) * patch_len]
            nxt = windows_z[i, (p + 1) * patch_len : (p + 2) * patch_len]
            X_lists["raw_patch"].append(cur)
            for r in rep_names:
                X_lists[r].append(reps[r][i, p])
            y_list.append(nxt)
            naive_list.append(np.full(patch_len, float(cur[-1]), dtype=np.float32))
            win_ids.append(int(window_meta[i]["window_id"]))
            macros.append(macro)

    X_by_rep = {k: np.asarray(v, dtype=np.float32) for k, v in X_lists.items()}
    return (
        X_by_rep,
        np.asarray(y_list, dtype=np.float32),
        np.asarray(naive_list, dtype=np.float32),
        np.asarray(win_ids),
        np.asarray(macros),
    )


def make_probe(kind: str, seed: int):
    if kind == "ridge":
        return RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0))
    if kind == "mlp":
        # 匹配 Bolt 自身的非线性 head；early-stopping 防止 768-dim 过拟合
        return MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-3,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=seed,
        )
    raise ValueError(f"unknown probe: {kind}")


def evaluate_rep(
    X: np.ndarray,
    y: np.ndarray,
    naive: np.ndarray,
    train_mask: np.ndarray,
    macros: np.ndarray,
    probe_kind: str = "ridge",
    seed: int = 47,
) -> dict[str, Any]:
    scaler = StandardScaler().fit(X[train_mask])
    Xs = scaler.transform(X)
    probe = make_probe(probe_kind, seed)
    probe.fit(Xs[train_mask], y[train_mask])
    pred = probe.predict(Xs[~train_mask])

    y_te, naive_te, macro_te = y[~train_mask], naive[~train_mask], macros[~train_mask]
    mae = float(mean_absolute_error(y_te, pred))
    mae_naive = float(mean_absolute_error(y_te, naive_te))
    result = {
        "n_train": int(train_mask.sum()),
        "n_test": int((~train_mask).sum()),
        "alpha": float(getattr(probe, "alpha_", float("nan"))),
        "mae": mae,
        "mae_naive_persistence": mae_naive,
        # RelMAE = MAE / persistence-MAE（同 horizon、out-of-sample skill ratio）。注意这**不是**
        # 教科书 MASE（后者用 in-sample 1-step naive 标度）；"mase" 键仅为向后兼容的别名。
        "relmae": float(mae / mae_naive) if mae_naive > 0 else float("nan"),
        "mase": float(mae / mae_naive) if mae_naive > 0 else float("nan"),
        "r2": float(r2_score(y_te, pred)),
        "per_macro_domain": {},
    }
    for macro in sorted(set(macro_te.tolist())):
        m = macro_te == macro
        if m.sum() < 10:
            continue
        mm = float(mean_absolute_error(y_te[m], pred[m]))
        mn = float(mean_absolute_error(y_te[m], naive_te[m]))
        result["per_macro_domain"][macro] = {
            "n_test": int(m.sum()),
            "mae": mm,
            "relmae": float(mm / mn) if mn > 0 else float("nan"),
            "mase": float(mm / mn) if mn > 0 else float("nan"),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["horizon", "next_patch"], default="horizon")
    parser.add_argument("--probe", choices=["ridge", "mlp"], default="mlp")
    parser.add_argument("--windows-per-dataset", type=int, default=60)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=16, help="horizon mode: 预测未来步数")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 6, 11])
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sample_len = args.context_len + (args.horizon if args.mode == "horizon" else 0)
    print(f"[probe] mode={args.mode} sampling windows (per_dataset={args.windows_per_dataset}, len={sample_len})")
    full_windows, window_meta, dataset_summary = sample_windows(
        DATA_ROOT,
        context_len=sample_len,
        windows_per_dataset=args.windows_per_dataset,
        seed=args.seed,
    )
    print(f"[probe] sampled {len(full_windows)} windows across {len(dataset_summary)} datasets")

    # 只把 context 部分喂给模型；归一化用 context 统计量
    context_raw = full_windows[:, : args.context_len]
    context_z = np.stack(
        [robust_z_with(c, *context_stats(c)) for c in context_raw]
    ).astype(np.float32)

    print(f"[probe] extracting Bolt representations (layers={args.layers}) ...")
    pipe = load_bolt_pipeline()
    patch_len = int(pipe.model.chronos_config.input_patch_size)
    reps = extract_bolt_representations(
        context_z,
        batch_size=args.batch_size,
        layers=args.layers,
        include_tokenizer=True,
        pipeline=pipe,
        keep_pipeline=False,
    )

    if args.mode == "horizon":
        X_by_rep, y, naive, win_ids, macros = build_horizon_dataset(
            full_windows, args.context_len, args.horizon, reps, window_meta, args.pooling
        )
        rep_order = ["raw_last_patch", "tokenizer"] + [f"layer_{L}" for L in args.layers]
    else:
        X_by_rep, y, naive, win_ids, macros = build_next_patch_dataset(
            context_z, reps, window_meta, patch_len
        )
        rep_order = ["raw_patch", "tokenizer"] + [f"layer_{L}" for L in args.layers]
    print(f"[probe] built {len(y)} examples; target_dim={y.shape[1]}")

    rng = np.random.default_rng(args.seed)
    uniq = np.unique(win_ids)
    test_windows = set(
        rng.choice(uniq, size=int(round(len(uniq) * args.test_frac)), replace=False).tolist()
    )
    train_mask = np.array([w not in test_windows for w in win_ids])

    results = {
        rep: evaluate_rep(X_by_rep[rep], y, naive, train_mask, macros, args.probe, args.seed)
        for rep in rep_order
    }

    summary = {
        "model": "chronos-bolt-base",
        "mode": args.mode,
        "probe": args.probe,
        "note": "clean successor to archived Chronos-2 pilot; input token is value-only (no time encoding)",
        "config": {
            "windows_per_dataset": args.windows_per_dataset,
            "context_len": args.context_len,
            "horizon": args.horizon if args.mode == "horizon" else None,
            "pooling": args.pooling if args.mode == "horizon" else None,
            "patch_len": patch_len,
            "layers": args.layers,
            "seed": args.seed,
            "test_frac": args.test_frac,
            "n_windows": int(len(full_windows)),
            "n_examples": int(len(y)),
            "metric_note": ("RelMAE = MAE / persistence-MAE (same horizon, out-of-sample skill "
                            "ratio; NOT textbook in-sample MASE). 'mase' key kept as legacy alias."),
        },
        "results": results,
    }
    tag = args.mode if args.mode == "next_patch" else f"horizon{args.horizon}_{args.pooling}"
    out_json = args.out / f"bolt_forecasting_probe_{tag}_{args.probe}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== Forecasting probe [{args.mode}, probe={args.probe}] (lower=better) ===")
    print(f"{'representation':<16}{'dim':>6}{'MAE':>9}{'RelMAE':>9}{'R2':>8}")
    for rep in rep_order:
        r = results[rep]
        print(f"{rep:<16}{X_by_rep[rep].shape[1]:>6}{r['mae']:>9.4f}{r['relmae']:>9.4f}{r['r2']:>8.3f}")
    print(f"\n[probe] saved -> {out_json}")


if __name__ == "__main__":
    main()
