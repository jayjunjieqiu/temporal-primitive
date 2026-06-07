# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 这个仓库是什么

一个**研究仓库**，不是产品代码。它研究 Time Series Foundation Models (TSFMs) 的 patch
token 到底学到了什么，并把问题讲成一个 **model-derived motif taxonomy discovery
protocol**。主流程是：从 heterogeneous cross-domain time series 采样 patch → 提取 TSFM
hidden representations → 在 representation space 做 clustering → 回到 original
time-series space 验证这些 cluster → 审计 domain / frequency / position confounders。
交付物是 markdown reports（`docs/`）和 figures（`outputs/`、`figure_projects/`），
不是一个可发布的 library。

绝大多数文档用中文写，并保留关键英文技术词（`TSFM`、`patch token`、`motif taxonomy`、
`prototype`、`controlled retrieval` 等）。改写文档时请保持这种中英混合风格。

## 动手前必读的两份文档

1. **`docs/00_narrative_rules.md`**（以及指向它的 `AGENTS.md`）—— 所有 proposal、report、
   figure caption、实验解释都必须遵守的写作与术语体系。这些规则是硬约束，不是文风偏好：
   - **双层 taxonomy 叙事。** `motif taxonomy v0` 是 human-prior / shapelet-inspired 的
     *probe*（不是 ground truth）；`model-derived motif taxonomy v1` 是从 representation
     space 发现的候选 cluster（是 *pilot*，不是 final taxonomy）。两者绝不能混为一谈。
   - 绝不能把一个 raw KMeans cluster 直接写成 "motif"。只有经过 original-space inspection
     + DTW-aware controlled retrieval + domain/frequency/position confounder audit
     之后，它才能称为 `candidate motif/prototype family`（见该文档 §7）。
   - 绝不宣称已经发现完整的 "temporal language" 或 final taxonomy。`temporal primitives`
     必须指向 `patch-level temporal primitives` / `motif prototypes`。
   - 负例（如 TimesFM first-patch artifact、synthetic Gaussian cluster）必须主动展示，
     不能隐藏。

2. **`docs/99_chronos2_archive_and_chronos_bolt_pivot.md`** —— 当前研究路线。自
   2026-05-20 路线转向起，**所有 Chronos-2 layer-wise 结果都已归档**为历史 diagnostic
   材料，原因是 Chronos-2 的 `projection` / `input_patch_embedding` token 混入了
   `[time encoding, normalized patch values, patch mask]`，因此它不是 pure value-only
   patch token。后续 clean analysis 的默认模型是 **Chronos-Bolt**。请把报告 `11_`、
   `12_`、`90_`–`98_` 以及当前 `figure_projects/` 的素材都当作 archived pilot evidence，
   并在写作时明确标注。

## 环境与运行

- Python **3.13**，由 **uv** 管理。文档中统一使用的解释器是 `.venv/bin/python`。
  所有命令都从**仓库根目录**（`/data/junjieqiu/temporal-primitive`）运行。
- 这台服务器的 shell 带有 HTTP proxy，会让包和模型下载失败。执行 `uv sync` 或任何
  Hugging Face 下载时**显式 unset 所有 proxy**：
  ```bash
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync
  ```
  默认 PyPI index 是 Tsinghua 镜像；`sklearn` 的 import 由 `scikit-learn` 提供
  （不要额外添加 deprecated 的 `sklearn` 包）。
- **Vendored editable 依赖**（gitignored，单独 clone）：`external/chronos-forecasting`
  和 `external/timesfm`，通过 `[tool.uv.sources]` 接入。脚本在 `import chronos` 之前还会
  在运行时把 `external/chronos-forecasting/src` 加进 `sys.path`。
- **本地模型权重**（gitignored）：`chronos-2/`、`chronos-2-small/`、
  `timesfm-2.5-200m-pytorch/`。用 `hf-mirror` 的 `hfd.sh` 在 unset proxy 的情况下下载，
  完整流程见 `docs/00_local_model_download.md`。注意 `chronos-2-small` 的 HF repo id 是
  `autogluon/chronos-2-small`。
- **`/data` 是共享卷，经常把空间和 inode 用尽。** 重跑大任务前先清理可再生 scratch
  （`__pycache__`、过期的 `outputs/` 中间产物）；卷满时即使很小的写入也会以 `ENOSPC`
  失败。

### 验证（没有 pytest）

这里的"测试"指语法检查 + smoke run：
```bash
.venv/bin/python -m py_compile scripts/<changed>.py        # 本仓库使用的语法关卡
.venv/bin/python scripts/smoke_temporal_primitives.py      # 端到端 smoke
.venv/bin/python scripts/feasibility_smoke.py              # 模型加载 / 提取 smoke
```

## 代码架构

**脚本既是独立 CLI 入口，又互相作为 package import。** `scripts/` 下每个文件都有
`argparse` + `if __name__ == "__main__"`，但其中一些会 `from scripts.X import ...`，
并依赖 `sys.path.insert(0, str(ROOT))`。因此：**必须从仓库根目录调用**
（`.venv/bin/python scripts/foo.py ...`），不要 `cd scripts/` 再跑。仓库没有安装成
package；`scripts/` 之所以可 import，只是因为 ROOT 在 path 上。

共享 backbone 模块（请复用，不要重复实现）：
- **`scripts/run_second_pilot_discovery.py`** —— 基础。定义 `MODEL_SPECS`
  （每个模型的 path / `patch_len` / 要探测的 `layers`）、`DATA_ROOT`、`sample_windows`、
  `robust_z`、`select_domain_balanced_indices`。多数其他脚本建立在它之上。
- **`scripts/run_chronos_multilayer_cluster_validation.py`** —— Chronos hidden-state
  提取、`macro_domain` 分组（synthetic Gaussian/Pulse = negative-control 组）、
  `MACRO_DOMAIN_ORDER` / `MODEL_PATH`。
- **`scripts/explore_motif_taxonomy.py`** —— `label_patch` 与 v0 probe detectors。
- **`scripts/build_cluster_cards.py`** —— cluster-card / domain-balanced clustering 辅助函数。

**模型 representation 提取模式：** 用 vendored pipeline 加载本地权重
（`chronos.Chronos2Pipeline.from_pretrained(path, local_files_only=True)`），然后用
**forward hook** 挂在 `model.encoder.block[layer_idx]` 上抓取各层 hidden state，projection
token 则取自 `model.input_patch_embedding`。结束后务必释放显存
（`del pipeline, model; gc.collect(); torch.cuda.empty_cache()`）。

**数据流程：** `DATA_ROOT = /data/junjieqiu/datasets/basicts_datasets`。每个数据集目录有
`desc.json`（`shape = [T, nodes, features]`、`domain`、`frequency (minutes)`）和
`data.dat`（一个 float32 **memmap**）。`sample_windows` 在 feature 0 上随机抽取
`(node, start)` 窗口，插值 NaN（<95% finite 的窗口丢弃），并做 robust-z normalization。
`patch_len` 在 Chronos 是 16，在 TimesFM 是 32。

**方法 = two-space distance principle**（`docs/00_narrative_rules.md` §5.1）：在
*representation space* 用 Euclidean geometry（PCA → KMeans → nearest-neighbor）
**生成候选**；在 *original time-series space* 用 DTW-aware geometry **验证**一个
neighborhood 是否对应 coherent shape family。只靠 Euclidean clustering 永远不能命名 motif。

**Configs**（`configs/`）是 YAML，不是代码：`motif_taxonomy_v0.yaml`（probe detectors +
thresholds）、`model_derived_taxonomy_v1_pilot.yaml`（candidate inventory，带 inclusion
rules 和显式的 `negative_control` / exclusion 标签）。

## Outputs 与 figures

- `outputs/**` **默认 gitignored**，再由 `.gitignore` 里一长串 `!`-allowlist 只重新纳入
  compact summary JSON 和 report-linked 证据图。当你想让某个新产物被 git 跟踪时，要加一条
  对应的 `!outputs/...` 例外；否则它会被静默忽略。大块数组 / 中间产物保持 untracked。
- **`figure_projects/reference_style_spectral_illustration/`** 把汇报图拆成独立模块
  （Module 2.1 representation atlas、2.2 patch-stack exemplar cards、2.3 cluster
  descriptor grid）—— 用户偏好手动拼接的模块化 PNG，而不是一张自动合成的整图。精确复现
  命令见它的 `README.md` 和 `docs/plan.md`，当前状态见 `docs/101_claude_code_handoff.md`。
  - 素材是**带版本号**的；Module 2.2 当前是 **`v4`** —— 不要再回头用 `v2`/`v3`。
  - GPU t-SNE（`--reducer cuml_tsne`）通过**单独的 conda env `rapids-tsne`**（RAPIDS/cuML）
    运行，由 `cuml_tsne_helper.py` 调用 —— 它**不属于** uv venv。KMeans 在 **PCA space**
    计算，t-SNE 只用于 visualization（图注里要写明）。cuML 的 nearest-neighbor warning
    是已知且无害的。

## 文档命名规则（`docs/`）

`00_*` 规则/setup · `01_*` proposal · `02_*` feasibility · `03_*`–`10_*` discovery 与
validation 历史 · `11_*`–`12_*` 归档的 Chronos-2 主证据 + method ablation ·
`80_*` appendix / weak-label sanity check · `90_*`–`98_*` meeting 与 PPT 材料（已归档）·
`99_*` 路线转向决策 · `100_*`+ figure 计划与 handoff。先从 `docs/README.md` 看索引地图。
