"""Chronos *training* 数据（in-distribution）的采样后端，供 primitive discovery 使用。

背景（见 docs/16 + memory discovery-on-training-data-pivot）：2026-06-21 起，primitive discovery
改在 Chronos 真正训过的数据上做（in-distribution，representation 最可信），basicts 测试集降级为
泛化 validation。本模块提供 discovery 侧的 curated 训练子集采样。

数据：/data/ts-datasets/chronos_datasets/<ds>/*.parquet，每行一条 series（值列名不固定：
target / power_mw / ...，自动探测）。每个数据集手工标 macro_domain（训练数据无 desc.json）。

入选规则（域无关，写进 doc）：(a) 在 Chronos 训练语料内；(b) series 长度 ≥ context(128)；
(c) 窗口内非退化。monash_hospital 因 (b) 排除（仅 84 点）；covid_deaths 是退化 cumulative ramp，
保留但当 mini negative-control 如实展示。
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CHRONOS_DATA = Path("/data/ts-datasets/chronos_datasets")

# curated 训练数据集：(macro_domain, dataset_relpath, 显示名)。covid_deaths = 退化 ramp（negative-control）。
TRAINING_DATASETS: list[tuple[str, str, str]] = [
    ("Energy", "monash_australian_electricity", "AU electricity"),
    ("Energy", "solar_1h", "Solar (1h)"),
    ("Energy", "wind_farms_hourly", "Wind farms (1h)"),
    ("Traffic", "monash_traffic", "Traffic (Monash)"),
    ("Traffic", "monash_pedestrian_counts", "Pedestrian counts"),
    ("Traffic", "taxi_30min", "Taxi (30min)"),
    ("Environment", "monash_weather", "Weather (Monash)"),
    ("Environment", "ushcn_daily", "USHCN climate (daily)"),
    ("Environment", "monash_temperature_rain", "Temperature-rain"),
    ("Finance", "exchange_rate", "Exchange rate"),
    ("Finance", "monash_fred_md", "FRED-MD (macro)"),
    ("Retail/Web", "m5", "M5 (retail)"),
    ("Retail/Web", "dominick", "Dominick (retail)"),
    ("Retail/Web", "wiki_daily_100k", "Wikipedia views"),
    ("Health", "monash_covid_deaths", "COVID deaths (cumulative)"),
    ("Synthetic", "training_corpus/kernel_synth_1m", "KernelSynth (synthetic)"),
]

# 全图统一 macro_domain 调色板（discovery 训练域 + validation basicts 域都覆盖）
DOMAIN_COLORS: dict[str, str] = {
    "Traffic": "#4C72B0",
    "Energy": "#DD8452",
    "Environment": "#55A868",
    "Finance": "#C44E52",
    "Retail/Web": "#8172B3",
    "Health": "#CCB974",
    "Synthetic": "#937860",          # KernelSynth（training, in-distribution）
    "Synthetic control": "#937860",  # 自制 Gaussian/Pulse（validation negative control）
    "Other": "#999999",
}


def _detect_value_column(parquet_file: str) -> str | None:
    """值列 = 第一个 list<float/double> 类型且不是 timestamp 的列。"""
    schema = pq.ParquetFile(parquet_file).schema_arrow
    for field in schema:
        t = str(field.type)
        if t.startswith("list<") and "timestamp" not in t:
            return field.name
    return None


def load_series_arrays(ds_relpath: str, max_rows: int = 6000) -> list[np.ndarray]:
    """读该数据集第一个 parquet 分片的前 max_rows 条 series（足够采样）。"""
    files = sorted(glob.glob(str(CHRONOS_DATA / ds_relpath / "*.parquet")))
    if not files:
        return []
    col = _detect_value_column(files[0])
    if col is None:
        return []
    df = pd.read_parquet(files[0], columns=[col], engine="pyarrow")
    if len(df) > max_rows:
        df = df.iloc[:max_rows]
    return [np.asarray(v, dtype=np.float64) for v in df[col].values]


def _interp_nans(w: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(w)
    if finite.mean() < 0.95:
        return None
    if not finite.all():
        idx = np.arange(w.shape[0])
        w = np.interp(idx, idx[finite], w[finite])
    return w


def sample_training_windows(
    context_len: int,
    windows_per_dataset: int,
    seed: int,
    datasets: list[tuple[str, str, str]] | None = None,
    max_tries_factor: int = 200,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """从 curated 训练子集采样窗口，接口对齐 run_second_pilot_discovery.sample_windows。

    返回 (windows[raw, 未 robust-z], metadata, dataset_summary)。metadata 含 dataset / domain /
    macro_domain（直接给定，下游 flatten 不再走 basicts macro_domain 映射）。
    """
    rng = np.random.default_rng(seed)
    rows = datasets if datasets is not None else TRAINING_DATASETS
    windows: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for macro, ds, disp in rows:
        series = load_series_arrays(ds)
        usable = [s for s in series if s.shape[0] >= context_len]
        accepted = 0
        tries = 0
        max_tries = windows_per_dataset * max_tries_factor
        seen: set[tuple[int, int]] = set()
        n_usable = len(usable)
        while accepted < windows_per_dataset and tries < max_tries and n_usable > 0:
            tries += 1
            si = int(rng.integers(n_usable))
            s = usable[si]
            start = int(rng.integers(0, s.shape[0] - context_len + 1))
            key = (si, start)
            if key in seen:
                continue
            seen.add(key)
            w = s[start : start + context_len].astype(np.float64)
            w = _interp_nans(w)
            if w is None or float(np.nanstd(w)) < 1e-6:
                continue
            windows.append(w.astype(np.float32))
            metadata.append(
                {
                    "window_id": len(metadata),
                    "dataset": disp,
                    "dataset_path": ds,
                    "domain": macro,
                    "macro_domain": macro,
                    "start": start,
                    "context_len": context_len,
                }
            )
            accepted += 1
        summary.append(
            {"dataset": disp, "dataset_path": ds, "macro_domain": macro,
             "n_series": len(series), "n_usable": n_usable,
             "accepted_windows": accepted, "status": "ok" if accepted else "empty"}
        )

    return np.stack(windows).astype(np.float32), metadata, summary
