"""把 main figure（main_*.png）+ OOD-transfer 结果（ood_*.png）打包成 advisor zip。

命名规则：main 交付图一律 `main_X_...png`（A cards / B cluster maps / C prototype / D generalization /
E useful+contextualized / F nearest-exemplars）；OOD-transfer 图 `ood_*.png`。zip 里的名字 == 文件夹里的
名字。临时对比图（带 _FILTERED / _nofilter 后缀）不进 zip。

从仓库根目录运行：
    .venv/bin/python scripts/assemble_main_figure_zip.py
"""
from __future__ import annotations

import glob
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures" / "bolt_main_figure"
OOD_DIR = ROOT / "outputs" / "figures" / "bolt_ood_transfer"
ZIP = ROOT / "outputs" / "figures" / "chronos_bolt_raw_figures_3tasks.zip"
EXCLUDE_TOKENS = ("_FILTERED", "_nofilter")  # 临时对比图后缀，不进交付 zip


def main() -> None:
    main_files = sorted(glob.glob(str(FIG_DIR / "main_*.png")))
    ood_files = sorted(glob.glob(str(OOD_DIR / "ood_*.png")))
    files = [f for f in main_files + ood_files
             if not any(tok in Path(f).name for tok in EXCLUDE_TOKENS)]
    if not main_files:
        raise SystemExit("no main_*.png found — run build_bolt_main_figure.py + plot_bolt_combined_depth.py first")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path(f).name)
    print(f"[zip] {ZIP.name} ({len(files)} figures: {len(main_files)} main + {len(ood_files)} ood):")
    for f in files:
        print("  ", Path(f).name)


if __name__ == "__main__":
    main()
