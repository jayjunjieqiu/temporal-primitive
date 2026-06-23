# Chronos-Bolt main figure：训练数据上发现 primitive + held-out 上验证泛化

更新时间：2026-06-21
模型：**Chronos-Bolt-base**（clean 路线）
脚本：`scripts/build_bolt_main_figure.py` · `scripts/chronos_bolt_backbone.py` · `scripts/chronos_training_data.py`

## 0. 本版相对上一版的改动（discovery-on-training pivot）

上一版在 basicts **测试集**上做 clustering。本版按 train→discover / test→validate 重构：

- **discovery 改在 Chronos in-distribution 训练子集上做**（representation 在模型训过的分布上最可信）。
- **basicts 测试集降级为泛化 validation**（剔除在训练集内的 Electricity / BeijingAirQuality，见 docs/16）。
- **新增 generalization 模块**：把训练数据上发现的 prototype 拿去检索 held-out patch，量化"primitive 是否
  在模型没见过的数据上重现"。
- 双层对比（layer_0 + layer_11）在 cards / cluster-maps / prototype / generalization 四块都保留。

**层号显示约定（2026-06-21）**：图里给人看的 layer 号用 **1-based（layer 1–12，Nature 习惯）**=
encoder block 索引 + 1。**代码 / CLI 参数（`--card-layers 0 11`）/ reps key / 源文件名仍是 0-based**
（block 索引，对应 `model.encoder.block[i]`）。即 figure 的 "layer 1" = block 0 = 最浅，"layer 12" =
block 11 = 最深。zip 里的 arcname 也用 1-based（`..._layer1.png` / `..._layer12.png`）。

## 1. discovery 数据（curated 训练子集，16 个）

来源 `/data/ts-datasets/chronos_datasets/`（parquet，每行一条 series，值列自动探测），清单与
macro_domain 见 `scripts/chronos_training_data.py:TRAINING_DATASETS`，肉眼预览见
`outputs/figures/training_preview/training_dataset_samples.png`。

| macro domain | 数据集 |
| --- | --- |
| Energy | AU electricity · Solar(1h) · Wind farms(1h) |
| Traffic | Monash traffic · Pedestrian · Taxi(30min) |
| Environment | Weather(Monash) · USHCN climate · Temperature-rain |
| Finance | Exchange rate · FRED-MD |
| Retail/Web | M5 · Dominick · Wikipedia |
| Health | COVID deaths（cumulative，退化 ramp，当 mini negative-control 如实展示） |
| Synthetic | KernelSynth（in-distribution 合成） |

**入选规则（域无关，对所有域一视同仁）**：(a) 在 Chronos 训练语料内；(b) series 长度 ≥ context(128)；
(c) 窗口内非退化。据此 `monash_hospital` 因 (b) 排除（仅 84 点月度）；Health 在本语料只剩退化的
`covid_deaths`——这是个 finding，不是挑域。`weatherbench_hourly` 是嵌套多变量目录，换成 `ushcn_daily`。

## 2. validation 数据（basicts held-out）

basicts 22 个数据集，剔除 Electricity / BeijingAirQuality（在 Chronos 训练集内）后的 **20 个 held-out**，
覆盖 PEMS×4 / METR-LA / PEMS-BAY / CA / GBA / GLA / SD（交通）、ETTh/ETTm（能源）、Weather、ExchangeRate、
Illness（Health，discovery 侧没有的域 → 纯泛化），外加自制 **Gaussian / Pulse** 作 negative control。

## 3. 产出（modular PNG，用户偏好手动拼接）

命名规则：交付图统一 `main_X_...png`（文件夹与 advisor zip 同名，由 `scripts/assemble_main_figure_zip.py`
打包；带 `_FILTERED` 后缀的是临时对比图，不进 zip）。layer 号 1-based 显示（= encoder block + 1）。
都在 `outputs/figures/bolt_main_figure/`：

- `main_A_cards_layer{1,12}.png` —— **discovery cards**：每簇 center-nearest top-24 raw patch 堆叠
  （z-normalized imshow，右侧 colorbar 标 ±σ 色标），标题下一根 100% **domain-composition 横条**反映整簇
  macro_domain 构成（跨域混合）。
- `main_B_cluster_maps.png` —— **representation atlas**（2×3）：每行一个 depth，三列同一套 t-SNE 点、不同着色：
  模型 KMeans cluster | human motif taxonomy v0（shapelet probe，*非* ground truth）| macro domain（confounder
  audit）。每列下方各自 2 列 legend。KMeans 在 PCA(30) space，t-SNE 仅可视化。
- `main_C_prototype_crossdomain_layer{1,12}.png` —— **cross-domain prototype**：行=cluster，列=不同训练
  域里离中心最近的最佳代表，强调同一 shape family 跨域复用。
- `main_D_generalization_heldout_layer{1,12}.png` —— **★ 泛化检验**：行=训练发现的 prototype（第 1 列红=prototype
  形状），后续列=被分配进该簇、来自**不同 held-out 数据集**的 unseen patch（黑），灰线=prototype 参照。
- `main_E_useful_contextualized_depth.png` —— **useful & contextualized**（合并图，3 等分面板）：forecast skill
  （RelMAE）| confounder decodability | within-context similarity，共享深度横轴。脚本
  `scripts/plot_bolt_combined_depth.py`。

可选筛除近平直 patch：`--min-patch-std 0.15 --out-suffix _FILTERED`（cards/prototype 的临时对比版）。

`k=8`，seed=47，context=128，patch_len=16。discovery 每数据集 200 窗口（domain-balanced ≤400/域，约 2800 patch），
validation 每数据集 150 窗口（20 数据集，约 24000 patch）。

## 4. generalization 指标（这是新主线证据）

**主指标 = held-out shape coherence**：每个 unseen patch 在 *representation space* 投到 discovery 的
StandardScaler→PCA→KMeans 空间、分配到最近 cluster（rep-NN），再算它与该簇 prototype 形状的相关系数；
coherence = corr ≥ 0.6 的比例。

| 层 | real held-out coherence | pure-noise control (Gaussian) |
| --- | --- | --- |
| layer_0 | **34%** | **0.8%** |
| layer_11 | 29% | 0.7% |

- **34% vs 0.8% ≈ 40× 分离**：在模型没见过的数据上，三分之一的 patch 能高相关地落到某个训练发现的
  prototype 上，而纯高斯噪声几乎为零 → 这些 prototype 是真实可复用的结构，不是聚类 artifact。
- **layer_0 > layer_11**：shallow 层 cluster 是 shape family（泛化检索高）；深层 cluster 被 context/domain
  重组，shape 检索掉到 29%——与 docs/14（contextualization 随 depth 上升）一致。

**为什么不用 cross-space agreement（rep-NN cluster == raw-shape-NN cluster）做 headline**：它会被噪声蒙混——
Gaussian 噪声在 representation 与 raw 两个空间都稳定落进同一个"通用 wiggle"簇，agreement 反而高达 0.30
（比 real 数据还高），无法区分 negative control。故只把它作次要诊断（写在 summary JSON），不作主指标。

**caveat（必须如实写）**：coherence 用 position-sensitive correlation，会**低估 shift-invariant 家族**。
Pulse（impulse spike，位置随机）coherence 仅 ~7%，但 panel 里能看到 Pulse patch **确实被 representation
归进 impulse 簇 C1**——说明模型表示比 raw-L2 更 shift-invariant，是指标低估而非泛化失败。所以干净的
negative control 是 **Gaussian（纯噪声）**，Pulse 是被指标低估的 positive。

## 5. 观察（含必须如实展示的对比）

- **layer_0 cards / prototype 是 shape-coherent 候选 cluster**：可见 impulse spike（C1）、上升/下降 ramp、
  U 形 valley、高频振荡、flat/intermittent 等家族，且 domain-composition 条显示每簇跨多个训练域。
- **layer_11 明显更"杂"**：深层 cluster 按 context/domain 重组，raw patch 形状不再一致——不是 bug，是
  contextualization。generalization panel 在 layer_11 检索质量下降同样印证。
- **negative control**：discovery 侧 COVID（退化 ramp，塌成 ramp 簇）+ KernelSynth；validation 侧 Gaussian
  coherence ≈ 0（纯噪声不匹配任何 primitive）、Pulse 落入 impulse 簇（被指标低估的 positive）。

叙事边界（`docs/00_narrative_rules.md`）：这些仍是 clean Chronos-Bolt 的 **candidate** cluster / prototype
family，generalization 提供了 controlled-retrieval + negative-control 证据，但**不能**直接称为已命名 motif；
完整命名仍需 DTW-aware retrieval + domain/frequency/position confounder audit。

## 6. 复现

```bash
.venv/bin/python scripts/build_bolt_main_figure.py \
  --windows-per-dataset 200 --val-windows-per-dataset 150 \
  --card-layers 0 11 --prototype-layers 0 11 --generalization-layers 0 11 \
  --k 8 --top-n 24 --proto-per-cluster 6 --max-per-domain 400 --tsne-perplexity 40

# 训练数据集采样预览（discovery 数据选型依据）
.venv/bin/python scripts/preview_training_datasets.py --windows-per-dataset 6 --length 192
```

GPU 提取结果缓存在 `.cache/extract_train_*.pkl`（纯改图/改指标时命中缓存，跳过 GPU）；`--no-cache` 强制重提。
交付图打包：`.venv/bin/python scripts/assemble_main_figure_zip.py`（folder 与 zip 同名 `main_*.png`）。

## 7. 预处理与 caveats（**论文须写明**）

1. **平直直线筛除（默认开启）**：聚类/卡片/prototype/generalization **默认丢弃近平直 patch**——robust-z
   窗口内 std < **0.15** 的 patch（`flat_low_information` / 常数段，画出来是无信息的水平直线）。约丢弃
   ~13% discovery patch。**论文方法部分必须写明这一筛除规则**（它会影响 cluster 组成与覆盖率）。
   未筛除的对照快照在 `outputs/figures/bolt_main_figure_archive/`（`--min-patch-std 0` 复现）。
2. **validation 泄漏剔除**：basicts 测试集里剔除了 4 个数据集——`Electricity`/`BeijingAirQuality`（在 Chronos
   预训练集内）、`Traffic`/`ExchangeRate`（与 discovery 训练子集 monash_traffic/exchange_rate **同一份数据**，
   见 docs/16）。剔除后 held-out coherence 33%→（仍远高于噪声 1%）。
3. **prototype(main_C) 与 generalization(main_D) 已统一版式**：col 0 = cluster prototype（红），其后**每列固定一个
   macro domain**（同域永远同列、彩色表头），main_C 列内是训练代表、main_D 是 held-out 代表。
4. **main_F（by-distance nearest）**：col0 聚类中心 | 3 最近训练 patch（蓝）| 3 最近 held-out patch（绿），
   只按距离、不强制跨域。

层号显示 1-based（layer N = encoder block N−1）；交付图统一 `main_X_...png`（A cards / B cluster-maps /
C prototype / D generalization / E useful+contextualized / F nearest-exemplars）。

## 8. OOD transfer 案例研究（见 `scripts/run_bolt_ood_transfer.py`、`outputs/figures/bolt_ood_transfer/`）

"可迁移泛化"：离 train(≈pretraining) 分布最远（OOD）的 held-out patch 也能被训练发现的 primitive 解释。
OOD 分 = kNN 距离(test→train) ÷ train 自身典型近邻距离（每层各自归一化）。产出 overlap t-SNE、OOD 排名
（Gaussian/Pulse 最 OOD，作 sanity）、case study、归属饼图（OOD patch 的最近 prototype 跨全部 cluster/domain；
top-25%-OOD patch 中 27% 仍有 shape-corr≥0.6 的 training prototype）。

**层深解读**：OOD 排名在浅层(L1) vs 深层(L12) 呈"两头向中间挤"——真实 held-out（交通 SD/GLA/CA/GBA、
Weather）**深层更 OOD**，合成 Gaussian/Pulse **深层更不 OOD**。原因是浅层表征 ≈ 通用形状（跨数据集共享 →
held-out 不 OOD）、深层表征 ≈ 数据集/context 身份（held-out 是新身份 → 更 OOD）；噪声在深层抓不到 context、
塌向 bulk → 变不 OOD。这与"浅层 reusable primitive / 深层 contextualized"一致。

**★ caveat（必须写明，且是真实局限）**：OOD 是相对**我们 16 个 discovery 数据集**算的，**不是相对 Chronos
全量预训练分布**。16 个数据集**稀疏覆盖**深层的 context 流形，深层"洞"多，会让 held-out 的深层 OOD 被**抬高**
——所以"深层更 OOD"里**有相当一部分是参照欠采样 artifact，而非纯内禀 contextualization**，二者用当前稀疏参照
分不开。**因此 OOD-transfer 的可迁移结论锚定在浅层(L1，primitive 所在、欠采样影响最小)更稳；深层 OOD 升高
不单独当 contextualization 证据**（contextualization 已有直接证据：docs/14 的 domain decodability / within-context
similarity / 本文 generalization coherence 0.34→0.29）。要把 OOD 当 headline，需先把 train 参照扩到接近预训练
覆盖度（更多 chronos_datasets）再重算，见 §8 末。

若要消除欠采样混淆：把 OOD 的 train 参照从 16 个扩到 chronos_datasets 里尽量多的 named 数据集（仅 train 侧多做
一次 GPU 提取），重算排名；深层 OOD 升高若明显缩小则证实为欠采样所致。本轮未做（OOD 为支撑性分析，非 main claim）。
