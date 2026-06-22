"""Task 2 figure — Chronos-Bolt contextualization。

读取 run_bolt_contextualization.py 的 summary，画两面板：
- 左：layer-wise NMI（cluster label vs macro_domain / frequency / patch_index）。
- 右：local vs global patch cosine similarity（同窗口 vs 跨窗口），阴影=gap。

随 representation 深度（tokenizer → encoder layers），position/frequency NMI 上升、
local−global gap 由 ~0 转正 → contextualization 的 clean 证据。

从仓库根目录运行：
    .venv/bin/python scripts/plot_bolt_contextualization.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "bolt_contextualization" / "bolt_contextualization_summary.json"
FIG_DIR = ROOT / "outputs" / "figures" / "bolt_contextualization"

REP_LABEL = {
    "tokenizer": "tokenizer\n(input embed)",
    "layer_0": "enc L1",
    "layer_3": "enc L4",
    "layer_6": "enc L7",
    "layer_9": "enc L10",
    "layer_11": "enc L12",
}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    reps = summary["rep_order"]
    res = summary["results"]
    xs = list(range(len(reps)))
    xlabels = [REP_LABEL.get(r, r) for r in reps]
    tok_end = 0.5  # tokenizer 在 index 0；其后为 transformer-contextualized 区

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # --- NMI panel ---
    ax = axes[0]
    nmi_specs = [
        ("macro_domain", "domain (macro)", "#1f77b4"),
        ("frequency", "frequency / cadence", "#ff7f0e"),
        ("patch_index", "position (patch idx)", "#2ca02c"),
    ]
    for key, label, color in nmi_specs:
        ys = [res[r]["nmi"][key] for r in reps]
        ax.plot(xs, ys, "-o", color=color, label=label, lw=2, ms=6)
    ax.set_ylabel("NMI(cluster, confounder)")
    ax.set_title("Confounder information absorbed vs depth")
    ax.legend(loc="upper left", fontsize=9)

    # --- similarity panel ---
    # 研究问题：同一 context 下不同位置 patch 的相似度是否随 depth 上升。
    # 用 centered cosine（每层减全局均值方向），去掉"深层空间整体散开"的 confounder。
    ax = axes[1]
    local = [res[r]["similarity"]["centered"]["local_mean"] for r in reps]
    glob = [res[r]["similarity"]["centered"]["global_mean"] for r in reps]
    ax.plot(xs, local, "-o", color="#d62728", label="same context\n(diff-position patches)", lw=2, ms=6)
    ax.plot(xs, glob, "-o", color="#7f7f7f", label="different context\n(random patches)", lw=2, ms=6)
    ax.fill_between(xs, glob, local, where=[a >= b for a, b in zip(local, glob)],
                    color="#d62728", alpha=0.12)
    ax.set_ylabel("centered cosine similarity")
    ax.set_title("Within-context patch similarity vs depth")
    ax.legend(loc="upper left", fontsize=8)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.axvspan(tok_end, len(reps) - 0.5, color="#2ca02c", alpha=0.05)
        ax.axvline(tok_end, color="#2ca02c", ls=":", lw=1)
        ax.grid(True, axis="y", alpha=0.25)

    n = summary["config"]["n_windows"]
    fig.suptitle(
        "Chronos-Bolt-base — representations become contextualized with depth\n"
        f"value-only tokenizer → encoder layers acquire position/cadence; within-context patches "
        f"grow more similar with depth ({n} windows; green = transformer-contextualized)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "bolt_contextualization_depth_curve.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] saved -> {out}")


if __name__ == "__main__":
    main()
