# Cluster Card Review: 数据/模型中心 TSFM Patch Concept Discovery

## 1. 本轮目标与结论

本轮目标不是定义最终 taxonomy，而是把 second pilot 中的自然簇转化为可人工审阅的 **cluster cards** 和 **controlled retrieval diagnostics**，从而判断哪些 cluster 更像真实 temporal concepts，哪些更可能是 `domain`、`frequency`、`patch_index`、scale 或 synthetic-data artifact。

核心结论：

- 当前最值得继续追踪的候选 concept 是 `Chronos-2 layer_11 c6` 和 `TimesFM-2.5 layer_10 c8`。二者都呈现较明显的 transition-like / directional change 形态，并且在 `cross_domain` retrieval 下仍保留一定视觉一致性。
- `Chronos-2 layer_11` 比 `Chronos-2 layer_6` 更适合作为下一步 model-derived taxonomy v1 的 Chronos 侧起点，因为它同时有较高稳定性、低 patch-index confounding，以及更清晰的 transition-like 候选簇。
- `TimesFM-2.5 layer_10` 的 cluster stability 最高，但必须非常谨慎处理 patch-index confounding。`c8` 是有希望的候选概念，`c4` 则是一个非常清楚的 patch-position artifact negative control。
- `taxonomy-v0` 只能作为解释性 probe。自然簇并不等价于 `trend / level_shift / mixed_uncertain` 等先验标签，而是更像把多种局部非平稳变化合并成较粗的 transition family。

生成脚本：

```bash
.venv/bin/python scripts/build_cluster_cards.py --windows-per-dataset 100 --domain-balanced-patches 700 --batch-size 96 --top-k 10
```

主要输出：

- `scripts/build_cluster_cards.py`
- `outputs/cluster_cards/cluster_card_summary.json`
- `outputs/cluster_cards/cards/`
- `outputs/cluster_cards/retrieval/`

## 2. 方法简述

本轮沿用 second pilot 的设置：非 BLAST 的 22 个 BasicTS datasets，每个 dataset 抽取 `100` 个长度为 `128` 的 windows，使用 `domain_balanced` patch subset 重新构建 PCA + KMeans cluster。分析对象限定为：

- `Chronos-2 layer_11 domain_balanced`
- `Chronos-2 layer_6 domain_balanced`
- `TimesFM-2.5 layer_10 domain_balanced`

每个 cluster card 包含：

- model, layer, cluster id, cluster size
- top datasets / domains / frequencies / patch indices
- `taxonomy-v0` distribution
- raw statistics: `std`, `range`, `abs_slope`, `zero_ratio`
- cluster mean curve and variance band
- medoid patches
- 128-step context with the patch highlighted
- explicit confounder warnings

Controlled retrieval 包含五种条件：

- `unrestricted`
- `same_patch_index`
- `same_frequency`
- `cross_domain`
- `same_domain`

每个 retrieval condition 记录 `mean_shape_correlation`、`taxonomy_v0_agreement`、`domain_diversity`、`frequency_diversity`、`patch_index_diversity` 和 `mean_embedding_distance`。

## 3. 候选 Temporal Concepts

### 3.1 `Chronos-2 layer_11 c6`: `high_variation_transition_like`

Card:

- `outputs/cluster_cards/cards/chronos_2_layer_11_c6_high_variation_transition_like.png`

Retrieval:

- `outputs/cluster_cards/retrieval/chronos_2_layer_11_c6_high_variation_transition_like_retrieval.png`



关键信息：

- cluster size: `645`
- layer stability: `0.642`
- silhouette: `0.154`
- warnings: `no severe single-factor warning`
- top domains: `exchange rate` 144, `illness data` 110, `traffic flow` 97
- top frequencies: `1440` 144, `5` 123, `10080.0` 110
- top patch indices: `p7` 134, `p6` 117, `p5` 110, `p4` 67
- top taxonomy-v0: `mixed_uncertain` 241, `level_shift` 187, `trend` 82
- raw stats: `abs_slope_mean=1.069`, `zero_ratio_mean=0.004`

视觉解释：

这个 cluster 的 medoids 多数是明显的局部跃迁、局部上升、局部下降或 plateau transition。128-step context 中，高亮 patch 往往位于更长趋势或 regime movement 的一段，而不是孤立噪声。cluster mean curve 本身不特别尖锐，但 medoid 和 retrieval 更说明模型把这些 patch 放在相近 representation 区域。

Controlled retrieval:

| condition | mean shape corr | taxonomy-v0 agreement | domain diversity | frequency diversity | patch-index diversity |
|---|---:|---:|---:|---:|---:|
| `unrestricted` | 0.552 | 0.100 | 4 | 4 | 5 |
| `same_patch_index` | 0.411 | 0.300 | 5 | 4 | 1 |
| `same_frequency` | 0.220 | 0.100 | 1 | 1 | 6 |
| `cross_domain` | 0.328 | 0.300 | 3 | 3 | 6 |
| `same_domain` | 0.220 | 0.100 | 1 | 1 | 6 |

判断：

这是当前 Chronos 侧最强候选 concept。它不是一个纯 `level_shift` 或纯 `trend`，更像 `directional transition / local nonstationary transition`。它通过了初步 `same_patch_index` 和 `cross_domain` 检查，但 cross-domain shape correlation 仍只是中等，后续需要用多个 medoids 做 cluster-level retrieval，避免单一 query 偏差。

临时命名建议：`directional_transition`

### 3.2 `TimesFM-2.5 layer_10 c8`: `timesfm_transition_like`

Card:

- `outputs/cluster_cards/cards/timesfm_2_5_layer_10_c8_timesfm_transition_like.png`

Retrieval:

- `outputs/cluster_cards/retrieval/timesfm_2_5_layer_10_c8_timesfm_transition_like_retrieval.png`



关键信息：

- cluster size: `683`
- layer stability: `0.740`
- silhouette: `0.177`
- warnings: `no severe single-factor warning`
- top domains: `traffic flow` 206, `exchange rate` 135, `weather` 117
- top frequencies: `15` 196, `1440` 135, `10` 117
- top patch indices: `p2` 244, `p1` 236, `p3` 203
- top taxonomy-v0: `level_shift` 248, `mixed_uncertain` 230, `trend` 112
- raw stats: `abs_slope_mean=1.251`, `zero_ratio_mean=0.005`

视觉解释：

这个 cluster 的 mean z-patch 是平滑上升形态，medoids 中既有 traffic speed 的 rising/plateau transition，也有 Weather 的 hump-like transition。full context 显示高亮 patch 常位于 U-shape 回升、level recovery 或上升尾段。它比 `Chronos-2 layer_11 c6` 的 cluster mean 更干净，但由于 TimesFM 在 second pilot 中已有明显 patch-index confounding，不能仅凭视觉形态直接命名为 concept。

Controlled retrieval:

| condition | mean shape corr | taxonomy-v0 agreement | domain diversity | frequency diversity | patch-index diversity |
|---|---:|---:|---:|---:|---:|
| `unrestricted` | 0.408 | 0.600 | 3 | 2 | 2 |
| `same_patch_index` | 0.317 | 0.800 | 3 | 3 | 1 |
| `same_frequency` | 0.380 | 0.600 | 1 | 1 | 2 |
| `cross_domain` | 0.393 | 0.700 | 4 | 3 | 3 |
| `same_domain` | 0.380 | 0.600 | 1 | 1 | 2 |

判断：

这是 TimesFM 侧最强候选 concept。它在 `cross_domain` 下仍有 `0.393` 的 mean shape correlation 和较高 taxonomy-v0 agreement，视觉上也能看到多个 retrieved patches 保持 rising / transition 形态。主要风险是 patch index 只覆盖 `p1/p2/p3`，没有 `p0`，说明它可能是 “middle/late context transition” 而不是完全 position-invariant 的 temporal concept。

临时命名建议：`smooth_rising_transition`

## 4. 弱候选与不稳定候选

### 4.1 `Chronos-2 layer_11 c1`: `transition_like_cross_domain`

Card:

- `outputs/cluster_cards/cards/chronos_2_layer_11_c1_transition_like_cross_domain.png`

Retrieval:

- `outputs/cluster_cards/retrieval/chronos_2_layer_11_c1_transition_like_cross_domain_retrieval.png`

关键信息：

- cluster size: `1245`
- warnings: `no severe single-factor warning`
- top taxonomy-v0: `mixed_uncertain` 442, `level_shift` 381, `trend` 157
- patch-index distribution relatively spread: `p2` 218, `p3` 208, `p0` 189, `p1` 176

Controlled retrieval 的 mean shape correlation 较弱：

- `unrestricted`: `0.093`
- `same_patch_index`: `0.168`
- `same_frequency`: `0.146`
- `cross_domain`: `0.050`

判断：

这个 cluster 规模大、混杂警告少，但 retrieval 不支持它是一个清晰 shape-level concept。更可能是一个宽泛的 “nonstationary / transition-ish supercluster”，内部包含多种变化形态。它可以作为 taxonomy v1 的上位候选，但暂时不宜直接命名为具体 temporal primitive。

临时命名建议：`broad_nonstationary_transition_pool`

### 4.2 `Chronos-2 layer_6 c2`: `midlayer_transition_like`

Card:

- `outputs/cluster_cards/cards/chronos_2_layer_6_c2_midlayer_transition_like.png`

Retrieval:

- `outputs/cluster_cards/retrieval/chronos_2_layer_6_c2_midlayer_transition_like_retrieval.png`

关键信息：

- cluster size: `712`
- warnings: `no severe single-factor warning`
- top domains: `illness data` 148, `traffic flow` 145, `weather` 141
- top taxonomy-v0: `mixed_uncertain` 244, `level_shift` 190, `trend` 91

Controlled retrieval:

- `unrestricted`: `0.199`
- `same_patch_index`: `0.209`
- `same_frequency`: `0.436`
- `cross_domain`: `-0.007`

判断：

这个 cluster 在 `same_frequency` 条件下形状相似度反而最高，但 `cross_domain` 近乎失效。这说明它很可能是中层 representation 中的 frequency/domain-mediated structure，而不是跨域 temporal concept。它仍有研究价值，因为它提示 Chronos 中层可能编码 cadence 或 domain-style。

临时命名建议：`frequency_mediated_transition_like`

### 4.3 `TimesFM-2.5 layer_10 c5`: `timesfm_smooth_transition_like`

Card:

- `outputs/cluster_cards/cards/timesfm_2_5_layer_10_c5_timesfm_smooth_transition_like.png`

Retrieval:

- `outputs/cluster_cards/retrieval/timesfm_2_5_layer_10_c5_timesfm_smooth_transition_like_retrieval.png`

关键信息：

- cluster size: `667`
- warnings: `no severe single-factor warning`
- top domains: `traffic flow` 220, `weather` 115, `Beijing air quality` 93
- top taxonomy-v0: `mixed_uncertain` 229, `level_shift` 224, `trend` 140

Controlled retrieval 的 `mean_shape_correlation` 全部为 `0.000`。这不是说它一定没有意义，而是说明当前 medoid query 可能落在 near-flat / low-variance 或 heterogeneous 子簇上，导致 shape correlation 诊断失效。

判断：

不建议把它作为第一批 candidate temporal concept。它可能是 TimesFM 中层的 heterogeneous transition pool，也可能需要多 query 或 cluster 内二次聚类后才有解释性。

临时命名建议：`heterogeneous_smooth_transition_pool`

## 5. 明确 Artifacts / Negative Controls

### 5.1 `Chronos-2 layer_6 c7`: `gaussian_noise_artifact`

Card:

- `outputs/cluster_cards/cards/chronos_2_layer_6_c7_gaussian_noise_artifact.png`

Retrieval:

- `outputs/cluster_cards/retrieval/chronos_2_layer_6_c7_gaussian_noise_artifact_retrieval.png`

关键信息：

- cluster size: `795`
- warnings: `single-domain dominated`, `single-frequency dominated`
- top domain: `simulated Gaussian data` 700
- top frequency: `None` 700
- top taxonomy-v0: `mixed_uncertain` 565, `volatility_shift` 96, `intermittent` 61

判断：

这是一个很好的 negative control。它说明 cluster 可以非常稳定地捕捉 synthetic Gaussian distribution，而这不应被解释为自然 temporal primitive。后续如果保留 simulated datasets，需要在报告里单独标记 simulated-domain clusters，避免污染真实数据 taxonomy。

临时命名建议：`synthetic_gaussian_artifact`

### 5.2 `TimesFM-2.5 layer_10 c4`: `timesfm_patch_position_artifact`

Card:

- `outputs/cluster_cards/cards/timesfm_2_5_layer_10_c4_timesfm_patch_position_artifact.png`

Retrieval:

- `outputs/cluster_cards/retrieval/timesfm_2_5_layer_10_c4_timesfm_patch_position_artifact_retrieval.png`


关键信息：

- cluster size: `294`
- warnings: `patch-position dominated`
- top patch index: `p0` 294 / 294
- top taxonomy-v0: `level_shift` 136, `mixed_uncertain` 93, `trend` 36

Controlled retrieval 很有迷惑性：

- `unrestricted`: mean shape corr `0.896`, patch-index diversity `1`
- `cross_domain`: mean shape corr `0.895`, patch-index diversity `1`
- `same_patch_index`: mean shape corr `0.896`, patch-index diversity `1`

判断：

这个 cluster 视觉上和 retrieval 上都可能看起来很一致，但一致性几乎完全和 `patch_index=0` 绑定。因此它是 TimesFM position artifact 的强证据。后续 TimesFM 分析必须做 same-position、position-stratified 或 position-residualized 检查，否则会把 “first patch behavior” 错误命名成 temporal concept。

临时命名建议：`first_patch_position_artifact`

## 6. 哪些概念通过 Controlled Retrieval？

当前可以分三档：

**较强候选**

| cluster | temporary concept | 理由 |
|---|---|---|
| `Chronos-2 layer_11 c6` | `directional_transition` | `same_patch_index` 和 `cross_domain` 仍有中等 shape coherence；无严重单因素警告；medoids/context 有清楚 transition 形态 |
| `TimesFM-2.5 layer_10 c8` | `smooth_rising_transition` | `cross_domain` retrieval 仍保持 `0.393` shape corr 和 `0.700` taxonomy-v0 agreement；cluster mean 和 medoids 较清晰 |

**弱候选 / 上位池**

| cluster | temporary concept | 主要问题 |
|---|---|---|
| `Chronos-2 layer_11 c1` | `broad_nonstationary_transition_pool` | cluster 大但 retrieval shape coherence 弱，可能是宽泛 supercluster |
| `Chronos-2 layer_6 c2` | `frequency_mediated_transition_like` | same-frequency 强，cross-domain 弱，像 frequency/domain-mediated structure |
| `TimesFM-2.5 layer_10 c5` | `heterogeneous_smooth_transition_pool` | 当前 medoid retrieval 失效，需要多 query 或二次聚类 |

**Artifact / negative control**

| cluster | artifact type | 证据 |
|---|---|---|
| `Chronos-2 layer_6 c7` | synthetic Gaussian artifact | `simulated Gaussian data` 占 700 / 795 |
| `TimesFM-2.5 layer_10 c4` | patch-position artifact | `patch_index=0` 占 294 / 294 |

## 7. 对模型与层的判断

### Chronos-2

`Chronos-2 layer_11` 是目前最适合作为 Chronos 侧 taxonomy v1 起点的层。它在 second pilot 中有不错的 stability (`0.642`) 和低 patch-index confounding，并且本轮 `c6` 给出了可解释的 transition-like cluster。

`Chronos-2 layer_6` 更适合用来研究中层混杂：它可能编码 frequency/domain-style，同时也包含 temporal structure。它不应作为最终 concept 命名的首选层。

### TimesFM-2.5

`TimesFM-2.5 layer_10` 的 clustering stability 最高 (`0.740`)，因此非常值得研究。但它的 patch-position confounding 是核心风险。`c8` 显示 TimesFM 中层确实可能有清晰 transition-like concept；`c4` 同时证明同一层也能产生强 position artifact。

因此，TimesFM 下一步不应简单扩大 KMeans，而应做 position-aware analysis。

## 8. 下一步建议

我建议下一步做 **cluster-level controlled retrieval + position/frequency-aware audit**，仍然不要定义最终 taxonomy。

具体下一步：

1. 对 `Chronos-2 layer_11 c6` 和 `TimesFM-2.5 layer_10 c8` 做 multi-query retrieval：每个 cluster 取 `10-20` 个 medoids / high-confidence members，而不是只用单一 medoid query。
2. 对每个 query 计算 `unrestricted`, `same_patch_index`, `same_frequency`, `cross_domain`, `cross_domain_same_patch_index`, `cross_domain_same_frequency` 的平均结果。
3. 对 TimesFM 做 position-stratified clustering：分别在 `p0/p1/p2/p3` 内聚类，再看是否仍出现 `smooth_rising_transition`。
4. 对 Chronos 做 frequency-stratified retrieval：尤其区分 `5/15/60/1440/10080` minutes，避免把 cadence-specific transition 当成通用 concept。
5. 对 `Chronos-2 layer_11 c1` 和 `TimesFM-2.5 layer_10 c5` 做 cluster 内二次聚类，判断它们是不是多个 concept 的混合池。
6. 暂时把 taxonomy v1 写成候选 concept inventory，而不是 closed taxonomy。首批只放入通过 controlled retrieval 的概念，例如：
   - `directional_transition`
   - `smooth_rising_transition`
   - `broad_nonstationary_transition_pool` as supercluster / uncertain
   - `frequency_mediated_transition_like` as confounded concept
   - `synthetic_gaussian_artifact`
   - `first_patch_position_artifact`

## 9. 当前 Go / No-Go

**Go，但只进入 controlled validation，不进入最终 taxonomy claim。**

本轮已经完成从 PCA cluster 到 time-series space 解释的第一轮闭环：我们不仅能看到 cluster 在 representation space 中的结构，还能回到 raw patch、z-normalized patch、full context 和 controlled retrieval 中判断它到底像概念还是 artifact。

最重要的研究判断是：TSFM patch representation 中确实存在可命名的自然结构，但命名必须经过 confounder audit。`Chronos-2 layer_11 c6` 和 `TimesFM-2.5 layer_10 c8` 是下一轮 model-derived taxonomy v1 的主候选；`Chronos-2 layer_6 c7` 和 `TimesFM-2.5 layer_10 c4` 应作为 negative controls 保留在论文实验设计里。
