# Chronos-Bolt contextualization：representation 随层变得 contextualized

更新时间：2026-06-22
模型：**Chronos-Bolt-base**（clean 路线；见 `docs/99_chronos2_archive_and_chronos_bolt_pivot.md`）
主脚本：`scripts/run_bolt_contextualization_training.py`（训练数据、probe accuracy 版，**当前主图**）
对照脚本：`scripts/run_bolt_contextualization.py` · `scripts/plot_bolt_contextualization.py`（旧 NMI 版，appendix）

advisor 三条主线里"**随层 contextualized**"的 clean 证据。两个 layer-wise 量随 representation
深度（`tokenizer` → encoder 12 层）变化。

## 1. 两个指标

- **confounder decodability（probe accuracy，主指标，取代 NMI）**：对每个 confounder
  （`macro_domain` / `frequency` / `position`）做 **k-NN probe**——从每层表征（PCA(30)）直接预测该
  confounder 标签，看准确率随深度怎么变。这是**直接的 decodability**，不经 KMeans。
  - **为什么取代 NMI**：旧指标是 NMI(KMeans 簇标签, confounder)，经 KMeans + 固定 k；深层连续体被
    任意切分，使 domain NMI 出现**中层达峰后回落的 artifact**（见 §2b / §4）。probe accuracy 无此问题：
    domain/frequency/position 三者都**单调上升后饱和**。NMI 版降为 appendix 对照（§2.2）。
- **within-context patch similarity**（研究问题）：**同一 context 下不同位置的 patch
  representation 之间的相似度，会不会随 depth 增加而增加？** 同一窗口内不同位置的 patch 当
  *same-context*，跨窗口随机 patch 当 *different-context*，比较余弦相似度。
  - ⚠️ **绝对 cosine 会被 confound**：深层表示空间整体散开，所有相似度（含 different-context）
    一起下降，掩盖真实趋势。因此用 **centered cosine**（每层先减全局均值方向）度量同 context 耦合
    **本身**随深度的变化。

设置（主图）：Chronos in-distribution 训练子集（16 数据集，与 main figure 一致），每数据集 200 窗口、
`context_len=128`、`patch_len=16`、seed=47、全 12 层 + tokenizer；probe = 10-NN、domain-balanced ≤400/域。
（旧 NMI 对照版在 basicts 22 数据集 × 100 窗口上算。）

## 2. 结果

层号约定：图中 x 轴 `enc L1…L12` 用 1-based（Nature 习惯）= encoder block 索引 + 1；代码/CLI 仍 0-based
（`enc L1` = block 0、`enc L12` = block 11）。

### 2.1 主图：confounder decodability（probe accuracy）+ within-context similarity

证据图：`outputs/figures/bolt_contextualization/bolt_contextualization_probe_depth.png`（训练数据；
左 = 三个 confounder 的 10-NN probe accuracy，右 = within-context centered-cosine similarity）。

| representation | domain acc | frequency acc | position acc | same-ctx sim |
| --- | --- | --- | --- | --- |
| tokenizer | 0.49 | 0.67 | 0.21 | 0.04 |
| enc L1 (block 0) | 0.62 | 0.75 | 0.29 | 0.16 |
| enc L4 | 0.77 | 0.84 | 0.32 | 0.25 |
| enc L7 | 0.81 | 0.86 | 0.38 | 0.24 |
| enc L10 | 0.82 | 0.86 | 0.50 | 0.23 |
| enc L12 (block 11) | 0.80 | 0.85 | **0.59** | 0.24 |

（chance：domain≈0.14、frequency≈0.2–0.3、position=1/8=0.125；frequency probe 剔除 synthetic——无 cadence）

结论：

1. **三个 confounder 的 decodability 都随深度单调上升后饱和**，无 NMI 版那种回落。最大跃升都在
   tokenizer → 前几层 encoder（attention 注入 context）。
2. **position 最干净、且持续上升到最深层**（0.21 → 0.59，chance 0.125）：value-only tokenizer 几乎不含
   位置信息，**位置完全是经 encoder 逐层注入**——contextualization 的招牌信号。
3. **domain / frequency 早期猛涨、中后段饱和**（domain 0.49→0.82、frequency 0.67→0.86）。
4. **同一 context 下不同位置 patch 的 centered 相似度随深度上升**（0.04 → ~0.24，答研究问题：会），
   different-context 始终≈0。

### 2.2 对照（appendix）：旧 NMI 版（basicts，经 KMeans，存在 artifact）

证据图：`outputs/figures/bolt_contextualization/bolt_contextualization_depth_curve.png`。**保留作对照**——
注意 domain NMI 在中层达峰后回落，是 deep-layer KMeans 切连续体 + frequency 轴挤占的 **artifact**（直接
probe accuracy 无此问题，见 §2.1 / §2b）。

NMI（confounder absorption）：

| representation | NMI domain | NMI frequency | NMI position |
| --- | --- | --- | --- |
| tokenizer (input embed) | 0.123 | 0.145 | 0.005 |
| enc layer_0 | 0.129 | 0.156 | 0.003 |
| enc layer_3 | 0.164 | 0.204 | 0.004 |
| enc layer_6 | 0.165 | 0.230 | 0.009 |
| enc layer_9 | 0.154 | 0.257 | 0.012 |
| enc layer_11 | 0.143 | 0.262 | **0.021** |

within-context similarity（centered cosine；raw 见 summary JSON）：

| representation | same-context (centered) | different-context (centered) | gap |
| --- | --- | --- | --- |
| tokenizer (input embed) | 0.007 | 0.014 | −0.006 |
| enc layer_0 | 0.085 | 0.010 | +0.075 |
| enc layer_3 | 0.156 | 0.003 | +0.153 |
| enc layer_6 | 0.162 | 0.001 | +0.161 |
| enc layer_9 | 0.166 | 0.001 | +0.165 |
| enc layer_11 | **0.168** | 0.001 | **+0.167** |

结论：

1. **position NMI 单调上升（0.005 → 0.021，约 4×）**。这是最干净的 contextualization 信号：
   Chronos-Bolt 的 tokenizer 是 **value-only**（无 time encoding），所以本身几乎不含位置信息
   （NMI≈0.005）；**位置信息完全是经 encoder 的 attention 逐层注入的**。
2. **frequency / cadence NMI 单调上升（0.145 → 0.262）**：深层越来越编码采样频率这种
   context/domain-level 属性，而非单个 patch 的局部形状。
3. **domain NMI 在中层达峰（0.12 → 0.16，layer_6）** 后略降。
4. **同一 context 下不同位置 patch 的相似度随 depth 单调上升（答研究问题：会）**。centered
   same-context similarity 从 tokenizer 的 **0.007 升到 layer_11 的 0.168**，而 different-context
   始终 ≈0（随机 patch 去掉 shared component 后无关）。即 tokenizer 几乎没有 context 耦合
   （value-only），**encoder 逐层把"同 context 的 patch 互相靠拢"这件事建立起来**。
   - 注意：若看**未 centered 的绝对** cosine，same-context 相似度反而先降后升（0.77→0.47→0.62），
     因为深层整个空间散开、所有相似度一起降；这会掩盖真实趋势，故以 centered 为准（见 §4 边界）。

NMI 与 within-context similarity 一致指向同一结论，且**最大变化都发生在 layer_0 → layer_3**——
这与 forecasting probe
（`docs/13_`）里 forecasting 价值的最大跳变层完全吻合：**attention 前两三层是 contextualization
真正发生的地方**。

### 2b. 直接的 macro-domain 分离度（有监督，不经 KMeans）

证据图：`outputs/figures/bolt_contextualization/bolt_domain_separation_depth.png`（脚本
`scripts/run_bolt_domain_separation.py`，全 12 层 + tokenizer）。

上面的 domain NMI 经 KMeans，会被 **deep-layer 连续体切分**干扰。这里**直接用 macro_domain 标签**在每层
PCA(30) 空间量"同域 patch 是否聚在一起"，两个互补、且都不依赖 KMeans 的指标（左右两面板）：

- **10-NN macro-domain accuracy（局部）**：一个 patch 的 10 个最近邻里同域占比，对所有 patch 平均。
- **Calinski-Harabasz（全局）**：类间/类内方差比——即 **inertia 的有监督、可跨层比较的版本**（裸 inertia
  无监督、依赖每层尺度，不可比）。
- silhouette(domain) 作对照，全程 ≈0（macro domain 是重叠多模流形，silhouette 不适合，故只当 foil）。

| representation | 10-NN domain acc | Calinski-Harabasz |
| --- | --- | --- |
| tokenizer | 0.49 | 25 |
| enc L1 (block 0) | 0.62 | 47 |
| enc L4 | 0.77 | 56 |
| enc L7 | 0.81 | 62 |
| enc L10 | 0.81 | 60 |
| enc L12 (block 11) | 0.80 | 66 |

（7 个 macro domain，majority baseline ≈ 0.14）

结论：**两个指标都随深度单调上升后饱和**（acc 0.49→~0.82，CH 25→~60+），最大跃升在 tokenizer→前几层，
与上面 NMI / similarity 的跳变层一致——是贯穿 12 层的**普遍规律**。

这条直接度量还**澄清了 §2 里 domain NMI 在 layer_6 达峰后略降的疑点**：那个"下降"是 deep-layer KMeans 在
切连续体、cluster 标签本身变得不稳/不再对齐 domain 的 **artifact**，而表征里**同域 patch 实际上一直越聚
越紧**。即 domain 组织随深度持续增强，没有中层回落。（这也正是 deep-layer cluster-maps 里 domain 列看着
更"聚"、而 shape primitive 反而留在浅层的原因。）

## 3. 复现

```bash
# §2.1 主图：probe accuracy（训练数据，复用 §2b 全层提取缓存，无需 GPU）
.venv/bin/python scripts/run_bolt_contextualization_training.py

# §2b 直接 macro-domain 分离度（全 12 层 + tokenizer；首次跑会提取并缓存到 .cache）
.venv/bin/python scripts/run_bolt_domain_separation.py

# §2.2 对照：旧 NMI 版（basicts）
.venv/bin/python scripts/run_bolt_contextualization.py \
  --windows-per-dataset 100 --layers 0 3 6 9 11 --k 10
.venv/bin/python scripts/plot_bolt_contextualization.py
```

## 4. 边界

- KMeans / NMI 受 k 与 seed 影响；趋势（随层方向）稳健，但绝对值不应过度解读。
- **similarity 必须看 centered（或 gap），不能看绝对 cosine**：深层表示空间整体散开会让所有
  cosine 一起下降，掩盖 within-context 耦合的真实上升趋势。报告主结论用 centered cosine；raw
  与 ratio 一并存于 summary JSON 供审计。
- 这些是 representation-geometry 层面的 contextualization 证据，配合 `docs/13_` 的
  functional（forecasting）证据一起看更完整。
