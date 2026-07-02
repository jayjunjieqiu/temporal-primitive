# 论文结果复现指南（Paper Reproduction）

论文：**《Decoding Dynamical Systems: Foundation Models for Time Series and Beyond》**（Nature
Reviews Computing，已投稿）。写作稿仓库：`/data/junjieqiu/TSFM_NatRevComp`
（`CastleLiang/TSFM_NatRevComp`）。

本文件说明如何从本代码仓库复现论文里**由代码生成的经验性 figure / table**。主线模型是
**Chronos-Bolt-base**（clean 路线，见 `docs/99`）；发布版主图用的是 **fullrep 变体**（在标准化后的
完整 768 维 representation 上聚类、不做 PCA，`--cluster-space full --k 6`，见 `docs/17`）。

> 注意：论文里的 `fig_intro` / `fig_tokenization` / `fig_methodology` / `fig_eval` 是**概念示意图**
> （手绘 / 排版而成），不由代码生成，不在本指南范围内。早期做过的 **OOD analysis 在最终版被移出论文**
> （见写作稿 `docs/16` 讨论与最终 supplement），相关脚本（`pub_ood_figure.py`、`run_bolt_ood_transfer.py`）
> 仍在仓库里，但**不是提交版的 artifact**，复现论文时可忽略。

---

## 0. 论文 artifact → 脚本 → 产物 速查表

| 论文位置 | Artifact | 生成脚本 | 产物 |
| --- | --- | --- | --- |
| 正文 **Fig. 4** | `fig_exp.pdf`（representation analysis）| `pub_main_figure_panel_b/c/d.py` | 3 个 SVG panel（PPT 手动拼）|
| Supp **Fig. 1**（§1）| `fig_cross_arch.pdf`（cross-model）| `cross_arch_generalization/run_cross_arch_intrinsic.py` + `run_cross_arch_cluster_retrieval.py` | `panel_b_repmaps.pdf` 直出 + panel a SVG |
| Supp **Table 1**（§2）| discovery datasets（16 数据集 / 7 域）| `chronos_training_data.py`（定义）| 列表（无需运行）|
| Supp **Table 2**（§3）| motif label 判定阈值 | `explore_motif_taxonomy.py`（定义）| 阈值（无需运行）|
| Supp **Fig. 2**（§3）| `fig_motif_labels.pdf`（motif 示例）| `supp_motif_label_examples.py` | PDF + PNG |
| Supp **Table 3**（§4）| retrieval statistics | `run_bolt_retrieval_quant.py` | `bolt_retrieval_quant_summary.json` |

> ⚠️ **script 名和 Fig.4 panel 有一位偏移**（历史遗留）：`panel_b.py`→Fig.4 **panel a**、
> `panel_c.py`→**panel b**、`panel_d.py`→**panel c**。`pub_main_figure_panel_a.py` 是**已停用**的
> 深度三联图，不属于 Fig.4。别搞错。

---

## 1. 环境准备

所有命令**从仓库根目录**（`/data/junjieqiu/temporal-primitive`）运行，解释器统一是 `.venv/bin/python`。

- **Python 3.13 + uv**。装依赖（这台机器带 proxy，会让下载失败，必须显式 unset）：
  ```bash
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync
  ```
- **cross-model 附录额外需要 `momentfm`**（MOMENT backbone）：`uv pip install --no-deps momentfm`。
- **本地模型权重**（gitignored，用 `hf-mirror` 的 `hfd.sh` 下载，流程见 `docs/00_local_model_download.md`）：
  - `chronos-bolt-base/`（主图 + 检索 + 跨架构 里的 Chronos-Bolt）
  - `timesfm-2.5-200m-pytorch/`（跨架构 TimesFM，需 `external/timesfm/src` 在 path 上）
  - `moment-1-large/`（跨架构 MOMENT，走 `momentfm`）
- **GPU**：只有「抽取 representation」这一步（含 cross-arch）需要 GPU；其余 panel/表格脚本都是**纯 CPU 读缓存**。
- t-SNE 用的是 CPU `sklearn.manifold.TSNE`——**Fig.4 / cross-arch 都不需要那个单独的 `rapids-tsne`
  conda env**（那是归档的 reference-style illustration 项目才用的，见 `docs/101`）。

---

## 2. 共享前置：GPU 抽取缓存（Fig.4 / Supp Fig.2 / Table 3 都依赖）

主图三 panel、supp motif 图、retrieval 统计脚本**本身都不加载模型**，而是读同一个 GPU 抽取缓存：

```
outputs/figures/bolt_main_figure/.cache/extract_train_wpd200_val150_ctx128_seed47_layers0-11.pkl
```

它由 `build_bolt_main_figure.py`（唯一在 GPU 上加载 Chronos-Bolt 权重的步骤）生成。缓存文件名按
`extract_train_wpd{wpd}_val{val}_ctx{ctx}_seed{seed}_layers{lo}-{hi}.pkl` 模板，**必须用默认 flag 生成
才能对上名字**：

```bash
.venv/bin/python scripts/build_bolt_main_figure.py \
  --windows-per-dataset 200 --val-windows-per-dataset 150 \
  --card-layers 0 11 --prototype-layers 0 11 --generalization-layers 0 11 \
  --k 8 --top-n 24 --proto-per-cluster 6 --max-per-domain 400 --tsne-perplexity 40
```

- 需要 GPU + `chronos-bolt-base/`。缓存存在时会直接复用；加 `--no-cache` 强制重抽。
- 缓存在默认 `--k 8 / --seed 47 / --context-len 128` 下建立；**panel 脚本会自己在缓存 embedding 上
  用 `--k 6` 重新聚类**，所以 k=6 是在 panel 脚本上设，不是这里。
- panel 脚本里写死、不暴露成 CLI 的常量：`SEED=47`、`PERPLEXITY=40.0`、`MIN_PATCH_STD=0.15`、
  `MAX_PER_DOMAIN=400`、`LAYERS=[0,11]`（显示为 Layer 1 / Layer 12）。

---

## 3. 正文 Fig. 4（`fig_exp.pdf`）— Chronos-Bolt representation analysis

三个 panel 都用 `--cluster-space full --k 6` 跑，产物落在 `figure_projects/pub_main_figure_fullrep/`：

```bash
# Fig.4 panel a — Layer-1/12 t-SNE atlas（domain / rule-based motif label / model-derived group）
.venv/bin/python scripts/pub_main_figure_panel_b.py --cluster-space full --k 6

# Fig.4 panel b — k=6 pattern groups 的 raw-patch cards（Layer 12）
.venv/bin/python scripts/pub_main_figure_panel_c.py --cluster-space full --k 6

# Fig.4 panel c — retrieval audit：unseen held-out query → 最近训练 patch
.venv/bin/python scripts/pub_main_figure_panel_d.py --cluster-space full --k 6   # 可选 --n-queries 5
```

产物（SVG = 文字可编辑矢量，供 PPT；PNG = proof；**不逐 panel 出 PDF**）：
- `figure_projects/pub_main_figure_fullrep/panel_a_cluster_maps.{svg,png}`
- `figure_projects/pub_main_figure_fullrep/panel_b_cards_layer12.{svg,png}`
- `figure_projects/pub_main_figure_fullrep/panel_c_retrieval.{svg,png}`
- t-SNE 坐标缓存（gitignored，自动）：`.panel_b_tsne_cache_full_k6.npz`

flag 语义：`--cluster-space full` = `PCA_DIM=None`（直接在标准化后的 768 维上 KMeans，不做 PCA）并把
输出目录切到 `pub_main_figure_fullrep`；`--k 6` = KMeans 簇数。**`fig_exp.pdf` 本身不由脚本生成——三个
SVG 在 PowerPoint 里手动拼接**（atlas 在上，cards + retrieval 同排在下），导出为 PDF。排版约定见 `docs/17` §1。

---

## 4. Supp Fig. 1（§1，`fig_cross_arch.pdf`）— cross-model comparison

两个独立 panel，在 PPT 里上下叠成一张竖图（panel a 在上、panel b 在下）。两者都读
`cross_arch_shared_pool.py` 建的**共享 512-point 真实窗口池**，对 Chronos-Bolt / TimesFM / MOMENT 三个
backbone 各跑一遍（**需要 GPU + 三套权重 + `momentfm`**）。

```bash
# （可选）基建 smoke：建共享池 + 分发三个模型
.venv/bin/python scripts/cross_arch_shared_pool.py

# panel a — intrinsic signatures（contextualization 10-NN probe + selective convergence）
.venv/bin/python figure_projects/cross_arch_generalization/run_cross_arch_intrinsic.py \
    --windows-per-dataset 200 --seed 13          # 加 --replot 从缓存 JSON 重画

# panel b — representation maps（domains / learned primitives / cards / cross-domain retrieval）
.venv/bin/python figure_projects/cross_arch_generalization/run_cross_arch_cluster_retrieval.py \
    --windows-per-dataset 200 --seed 13 --k 6    # 加 --replot 从 .cache pickle 重画
```

- 三模型 layers：Bolt `[0,6,11]`、TimesFM `[0,10,19]`、MOMENT `[0,12,23]`；KMeans **k=6**（与主图对齐）。
- `cluster_retrieval` 关键默认：`--seed 13 --k 6 --batch-size 64 --max-per-domain 600 --knn 10
  --n-queries 300 --card-n 16 --n-cards 2 --retr-show 6 --tsne-n 3000`。
- 产物（在 `figure_projects/cross_arch_generalization/`）：
  - `panel_a_intrinsic_signatures.{svg,png}` + `cross_arch_intrinsic_summary.json`
  - `panel_b_repmaps.{svg,png,pdf}` —— **`.pdf` 直出**，就是可直接进论文的 `fig_cross_arch` 矢量图
    + `cross_arch_cluster_retrieval_summary.json`（caption 里的定量指标来源）
  - 重画缓存：`.cache/cluster_retrieval_cache.pkl`；共享样式模块：`cross_arch_style.py`（被 import，不单独跑）

---

## 5. Supp Fig. 2（§3，`fig_motif_labels.pdf`）— rule-based motif label 示例

```bash
.venv/bin/python scripts/supp_motif_label_examples.py            # 可选 --out <路径>
```

- 前置：同第 2 节的抽取缓存（读其中 `layer_0` 的 raw patch；某类缺样本才回退到合成 patch）。用
  `explore_motif_taxonomy.label_patch`（`patch_len=16`）给每个 patch 打标签，取每类置信度最高的真实样本。CPU。
- **默认输出写进写作稿仓库**：`/data/junjieqiu/TSFM_NatRevComp/fig/fig_motif_labels.pdf`（+ `.png` 预览）。
  想输出到别处用 `--out`。

---

## 6. Supp Table 1（§2）— discovery datasets（16 数据集 / 7 域）

**只是定义，无需运行**。见 `scripts/chronos_training_data.py` 里的 `TRAINING_DATASETS`（16 行
`(macro_domain, dataset_relpath, display_name)`，7 个 macro-domain：Energy / Traffic / Environment /
Finance / Retail-Web / Health / Synthetic），parquet 根目录 `/data/ts-datasets/chronos_datasets/`。纳入
规则见模块 docstring：(a) 在 Chronos 训练语料内；(b) series 长度 ≥ context(128)；(c) 非退化窗口。
`covid_deaths` = 退化 cumulative ramp，作 mini negative-control 保留。每数据集采样节奏在
`DATASET_FREQ_MINUTES`。

可选：`.venv/bin/python scripts/preview_training_datasets.py --windows-per-dataset 6 --length 192`
→ `outputs/figures/training_preview/`（采样窗口的可视化预览）。

---

## 7. Supp Table 2（§3）— rule-based motif label 判定阈值

**只是定义，无需运行**。阈值在 `scripts/explore_motif_taxonomy.py`：`score_detectors()`（判据）、
`extract_features()`（特征）、`label_patch()`（仲裁）；9 个标签在 `LABELS`。所有特征在 **robust
$z$-normalized patch** 上算（`robust_z` = MAD×1.4826；`|z|>2` = active、`|z|>3` = strong），flat detector
额外用 raw std/range。核心阈值：

| 标签 | 判据 |
| --- | --- |
| flat / low information | `raw_std < 0.08` 且 `raw_range < 0.30` |
| trend | `abs_slope ≥ 0.75`、`trend_r2 ≥ 0.70`、无 impulse |
| oscillation | `spectral_ratio ≥ 0.42`、`dominant_cycles ≥ 1`、`zero_crossings ≥ 3` |
| impulse spike | `max_robust_z ≥ 4.0`、strong-active ≤ 2、longest active run ≤ 2 |
| burst event | active_ratio ∈ [0.15,0.60]、longest_run/L ≥ 0.12、无 impulse |
| level shift | `mean_change_score ≥ 1.45`、`trend_r2 ≤ 0.86` |
| volatility shift | `std_ratio ≥ 2.2`、`mean_change_score ≤ 1.25` |
| intermittent | active_ratio ∈ [0.08,0.40]、active_runs ≥ 2、longest_ratio ≤ 0.20、无 impulse |
| mixed / uncertain | 无 detector ≥ 0.55，或 top-two non-flat 差 ≤ 0.18 |

fire 阈值 = score ≥ 0.55。可选重跑校准 JSON/混淆图：`.venv/bin/python scripts/explore_motif_taxonomy.py`
→ `outputs/motif_taxonomy_exploration_summary.json` + `outputs/figures/motif_taxonomy_confusion_patch{16,32}.png`。

---

## 8. Supp Table 3（§4）— retrieval statistics

```bash
.venv/bin/python scripts/run_bolt_retrieval_quant.py
```

- 默认（论文里所有数字都用这套）：`--layers 0 11`（显示为 Layer 1 / Layer 12）、`--k 8`、
  `--min-patch-std 0.15`、`--max-per-domain 400`、`--seed 47`、`--knn 10`、`--topn 50`、`--rand-pairs 40000`。
- 前置：同一抽取缓存（脚本 glob `outputs/figures/bolt_main_figure/.cache/extract_train_*.pkl` 取最新）。
  会重新套用 `VALIDATION_EXCLUDE`（`Electricity, BeijingAirQuality, Traffic, ExchangeRate, BLAST`）并剔除
  合成 control（`Gaussian, Pulse`），检索在 **PCA(30)** 空间。CPU，不加载模型。
- 产物：`outputs/bolt_ood_transfer/bolt_retrieval_quant_summary.json`——含每层 `n_query_patches`（≈16,453）、
  `coherence_retrieved_top1`（0.69 / 0.39）、`coherence_raw_patch_ceiling_top1`（0.88）、
  `coherence_random_baseline`（−0.005）、`cross_domain_retrieval_rate_top1`（92% / 93%）、bootstrap 95% CI、
  以及 `significance_wilcoxon_pvalue`（representation vs random / vs raw ceiling）。这是 Table 3 所有数字的唯一来源。

---

## 9. 需要注意的坑

- **发布主图 = `--cluster-space full --k 6`（fullrep）**，不是 `docs/15` 里记的旧版 canonical `k=8` +
  PCA(30)。提交版 `fig_exp.pdf` 用 k=6 / full，panel 排版规范以 `docs/17` §1 为准。
- **script 名 vs Fig.4 panel 有一位偏移**（`panel_b→a / panel_c→b / panel_d→c`；`panel_a.py` 是停用的深度图）。
- **`fig_exp.pdf` 和 `fig_cross_arch` 的最终拼接不同**：主图三 panel 在 PPT 手动拼；cross-arch 的
  `panel_b_repmaps.pdf` 是脚本直出的矢量 PDF。
- **motif 图默认写进写作稿仓库**（`TSFM_NatRevComp/fig/`），不是本仓库。
- **OOD analysis 已从最终论文移除**：`pub_ood_figure.py` / `run_bolt_ood_transfer.py` 仍在，但不属于提交版；
  复现论文时忽略。
- `outputs/**` 默认 gitignored，只有 `.gitignore` allowlist 里的 compact summary JSON / report-linked 图会被
  跟踪；大数组和中间缓存保持 untracked。
