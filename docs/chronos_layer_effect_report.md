# Chronos-2 Layer Effect Report: 聚类和 motif 空间如何随层变化

## 0. 导师真正想回答的问题

这份报告后续要服务的不是“多画一些 cluster 图”，而是 Yuxuan Liang 老师在 meeting 中反复追问的几个机制问题：

1. **Single patch 是否仍然有信息**：一个 patch 被放进 Chronos-2 的 whole context sentence 之后，是否还保留相对独立的 local temporal information？
2. **不同层在学什么**：`projection`、`layer_0`、`layer_6`、`layer_11` 是否分别对应 pre-contextual patch token、early local vocabulary、contextual mixing、more contextualized representation？
3. **早层是不是更适合看 motif**：老师的直觉是 early layers 更保留 spike、oscillation、trend 等 local information；middle layers 开始融合 context；top layer 更全局。这个直觉需要用 evidence 支撑，而不是只凭可视化印象。
4. **聚类中心在原空间长什么样**：老师希望使用 KMeans center 作为 cluster center，并用离 center 最近的真实 raw patches 作为 examples，而不是随便挑样本。
5. **是否存在跨领域语义**：nearest examples 不能只来自 1-2 个 dataset。需要检查同一个 cluster center 是否能在多个 macro-domain 中找到可信原空间 patch。
6. **K 是否合理**：K 的选择必须有定量依据，不能只靠经验公式或哪张图好看。
7. **prior-guided motif 的边界**：human-prior motif 可以帮助解释，但不是 ground truth；KMeans cluster 不能直接用 prior-guided motif 名字命名。
8. **主证据和诊断证据要分开**：好看的图如果 confounder 高，只能作为 diagnostic/failure case；主报告图必须通过稳定性、跨领域性、原空间一致性和 confounder audit。

因此，下一版报告应围绕一个更严格的问题组织：

> Chronos-2 的 patch representation 在 projection、early、middle、late layers 中，是否保留 local temporal primitives，并如何把这些 primitives 重组为 contextualized cross-domain temporal concepts？

## 1. 这次只回答一个问题

老师的直觉是对的：如果我们想知道 **single patch 是否仍然保留局部信息**，Chronos-2 的 early layers 应该优先看。  
这份短报告只看 `Chronos-2`，不混 TimesFM。

这里的阅读方式也按照老师建议来：**用 KMeans center 作为 cluster center，再用离 center 最近的点做 example**。  
所以下面的 original-space 图，不是“随便挑的样本”，而是围绕 cluster center 的 nearest examples。

更具体地说：

- 先在每个 representation 上做 `StandardScaler -> PCA -> KMeans`，用它作为 cluster candidate generator。
- KMeans cluster label 只写成 `C0, C1, ...`，不提前命名。
- 每个 cluster 的中心使用 `kmeans.cluster_centers_`。
- 原空间展示的是离该 center 最近的 raw patches。
- `trend / level shift / spike` 这类词只能作为人工后验解释或 prior-guided audit probe，不能写成 cluster 自己的名字。

每个 representation 还额外画配对图：左边是 KMeans cluster，右边是在同一二维坐标上按 `prior-guided motif probe` 上色。  
这张图的用途是审计二者是否对齐，而不是把 prior-guided label 当成 cluster ground truth。

### 1.1 为什么新增 t-SNE view

老师指出只用 PCA 二维图来展示 representation space，不太符合领域里做 embedding visualization 的常见汇报习惯。  
所以本报告新增一套 `t-SNE` view，并保留原来的 PCA 图作为 reference。

这里要特别说明一个边界：

- KMeans cluster、silhouette、stability、NMI、center-nearest examples 的计算仍然基于 `StandardScaler -> PCA(max 30 dims)` 空间。
- `t-SNE` 只用于二维可视化，不作为新的聚类依据。
- `t-SNE` 输入是上述 PCA clustering space，`perplexity=40`，`init=pca`，`random_state=47`。
- 因此，t-SNE 图回答的是“这些 KMeans labels 在非线性二维 view 里是否仍有局部结构”，而不是重新定义 cluster。

### 1.2 为什么加入 macro-domain view

老师提醒的一个关键点是：只看每个 cluster center 的 nearest 4，可能会被少数 dataset 主导。  
例如某个 cluster 的最近 4 个样本都来自 `Traffic` 或 `ETT`，视觉上看起来很一致，但我们不知道它到底是一个跨领域的 local motif，还是一个 dataset/domain artifact。

所以这里新增一个 **macro-domain nearest view**：对每个 KMeans cluster center，不再只取全局最近的 4 个点，而是在每个 macro-domain 内各找一个离 center 最近的 raw patch。  
它回答的是：

> 如果这个 cluster center 真的是一个相对通用的 temporal concept，那么在 Traffic / Energy / Environment / Finance / Health 这些不同应用领域里，是否都能找到相似的原空间 patch？

当前 macro-domain 分组如下。这个分组不是最终 taxonomy，只是为了做跨领域诊断：

| macro-domain | source domains |
|---|---|
| `Traffic` | `traffic flow`, `traffic speed`, `road occupancy rates` |
| `Energy` | `electricity consumption`, `electricity transformer temperature` |
| `Environment` | `weather`, `Beijing air quality` |
| `Finance` | `exchange rate` |
| `Health` | `illness data` |
| `Synthetic control` | `simulated Gaussian data`, `simulated pulse data` |

macro-domain 图的读法：

- 每一行是一个 KMeans cluster：`C0, C1, ...`。
- 每一列是一个 macro-domain。
- 每个小图是在该 macro-domain 内，距离当前 KMeans center 最近的 raw patch。
- 距离是在对应 representation 的 `StandardScaler -> PCA` 空间里计算的，不是在原始曲线空间里计算的。
- 黑色实线边框：这个 patch 本身也被 KMeans 分到了该行 cluster。
- 灰色虚线边框：它是该 macro-domain 里离该 center 最近的 patch，但 KMeans 分到了另一个 cluster。
- 红色虚线边框：不仅被分到另一个 cluster，而且距离超过该 cluster 内距离的 90% 分位数，属于弱匹配。

这张图不替代 nearest 4；它是一个 confounder audit。nearest 4 负责看 cluster center 附近最典型的样子，macro-domain view 负责看这个中心是否能跨领域复现。

## 2. 原空间里先有 motif 了吗？

有。raw-space 本身就已经出现了比较清楚的局部形态族，这说明我们不是在模型里“硬造”概念。

PCA reference：

![Chronos raw-space PCA cluster vs prior probe](../outputs/chronos_layer_effect/figures/chronos_layer_effect_raw_patch_cluster_vs_prior_probe.png)

t-SNE view：

![Chronos raw-space t-SNE cluster vs prior probe](../outputs/chronos_layer_effect/figures/chronos_layer_effect_raw_patch_tsne_cluster_vs_prior_probe.png)

![Chronos raw-space center-nearest patches](../outputs/chronos_layer_effect/figures/chronos_layer_effect_raw_patch_center_nearest_patches.png)

![Chronos raw-space macro-domain nearest patches](../outputs/chronos_layer_effect/figures/chronos_layer_effect_raw_patch_macro_domain_nearest_patches.png)

这张图是直接在 raw patch 空间聚类后，每个 KMeans center 最近的原始 patch。它不使用 prior-guided motif 名字。  
从视觉上看，原空间里已经存在一些可解释的 shape family，例如：

- `level_shift`
- `trend`
- `burst`
- `oscillation`
- `flat_low_information`
- `mixed_uncertain`

所以更准确的说法不是“模型发明了 motif”，而是：

> 原空间里本来就有一些 shape family，Chronos 的不同层是在重新组织这些 family。

### 2.1 最早的 token/projection 也还保留局部信息

PCA reference：

![Chronos projection cluster vs prior probe](../outputs/chronos_layer_effect/figures/chronos_layer_effect_projection_cluster_vs_prior_probe.png)

t-SNE view：

![Chronos projection t-SNE cluster vs prior probe](../outputs/chronos_layer_effect/figures/chronos_layer_effect_projection_tsne_cluster_vs_prior_probe.png)

![Chronos projection center-nearest patches](../outputs/chronos_layer_effect/figures/chronos_layer_effect_projection_center_nearest_patches.png)

![Chronos projection macro-domain nearest patches](../outputs/chronos_layer_effect/figures/chronos_layer_effect_projection_macro_domain_nearest_patches.png)

`chronos_proj_with_time` 仍然主要是在整理局部 patch vocabulary，而不是把它变成全局语义。  
它已经比 raw patch 更接近模型内部表示，但还没有进入深层那种强 contextualization。

## 3. 早层、 中层、深层的差别

### 3.1 `layer_0`：最像 local patch vocabulary

PCA reference：

![Chronos layer 0 cluster vs prior probe](../outputs/chronos_layer_effect/figures/chronos_layer_effect_layer_0_cluster_vs_prior_probe.png)

t-SNE view：

![Chronos layer 0 t-SNE cluster vs prior probe](../outputs/chronos_layer_effect/figures/chronos_layer_effect_layer_0_tsne_cluster_vs_prior_probe.png)

![Chronos layer 0 center-nearest patches](../outputs/chronos_layer_effect/figures/chronos_layer_effect_layer_0_center_nearest_patches.png)

![Chronos layer 0 macro-domain nearest patches](../outputs/chronos_layer_effect/figures/chronos_layer_effect_layer_0_macro_domain_nearest_patches.png)

`layer_0` 还比较保留局部形状，cluster 里已经能看到清楚的 motif-like 组团，但还没有被强烈地 context 化。  
从 second pilot 的统计看，`layer_0` 的 `patch-index NMI` 很低，说明它没有明显被 patch 位置支配。

macro-domain view 对 `layer_0` 尤其重要：它显示许多 cluster center 可以在真实应用领域中找到同 cluster 的近邻，而不是只在单一数据集内成立。  
在全部 90 个 macro-domain cell 中，`layer_0` 有 75 个 cell 的 nearest patch 仍然属于同一个 KMeans cluster；如果只看真实领域、暂时排除 `Synthetic control`，比例是 68/75。  
这说明 `layer_0` 比较适合作为 Chronos-native local patch vocabulary 的起点。

### 3.2 `layer_6`：开始融合 context

PCA reference：

![Chronos layer 6 cluster vs prior probe](../outputs/chronos_layer_effect/figures/chronos_layer_effect_layer_6_cluster_vs_prior_probe.png)

t-SNE view：

![Chronos layer 6 t-SNE cluster vs prior probe](../outputs/chronos_layer_effect/figures/chronos_layer_effect_layer_6_tsne_cluster_vs_prior_probe.png)

![Chronos layer 6 center-nearest patches](../outputs/chronos_layer_effect/figures/chronos_layer_effect_layer_6_center_nearest_patches.png)

![Chronos layer 6 macro-domain nearest patches](../outputs/chronos_layer_effect/figures/chronos_layer_effect_layer_6_macro_domain_nearest_patches.png)

`layer_6` 的 cluster 仍然存在，但已经开始更明显地混入 domain / frequency 信号。  
这一步更像是把 local patch vocabulary 重新编排成更上下文化的 representation。

和 `projection` / `layer_0` 相比，`layer_6` 的 nearest examples 仍然能看到 trend、level shift、spike-like 或 low-information patches，但同一 cluster 内的原空间形态更容易变宽：一些 cluster 不再只对应一个很干净的 local motif，而更像把多个局部形态按 context role、cadence 或 domain-style 合并到一起。

### 3.3 `layer_11`：更强的 contextualized representation

PCA reference：

![Chronos layer 11 cluster vs prior probe](../outputs/chronos_layer_effect/figures/chronos_layer_effect_layer_11_cluster_vs_prior_probe.png)

t-SNE view：

![Chronos layer 11 t-SNE cluster vs prior probe](../outputs/chronos_layer_effect/figures/chronos_layer_effect_layer_11_tsne_cluster_vs_prior_probe.png)

![Chronos layer 11 center-nearest patches](../outputs/chronos_layer_effect/figures/chronos_layer_effect_layer_11_center_nearest_patches.png)

![Chronos layer 11 macro-domain nearest patches](../outputs/chronos_layer_effect/figures/chronos_layer_effect_layer_11_macro_domain_nearest_patches.png)

到 `layer_11`，cluster 更稳定，但也更 contextualized。  
它不是简单地“更像 motif”，而是更容易混入 domain-style、cadence 和更长上下文信息。

`layer_11` 的 prototype panel 适合和 `layer_0` 对照看：有些 cluster 仍能回到原空间解释成 transition-like / trend-like patches，但 cluster 内部异质性更强。因此深层更适合回答“模型如何把 local vocabulary 重组成 contextualized motif family”，而不是直接回答“单个 patch 自身长什么样”。

## 4. 一个很短的数值总结

下面这组数值是最适合跟老师讲的版本：

| representation | silhouette | stability | NMI patch-index | NMI domain | NMI frequency | NMI prior-guided probe |
|---|---:|---:|---:|---:|---:|---:|
| `raw_patch` | 0.167 | 0.407 | 0.003 | 0.027 | 0.018 | 0.059 |
| `projection` | 0.106 | 0.554 | 0.006 | 0.192 | 0.173 | 0.116 |
| `layer_0` | 0.101 | 0.590 | 0.005 | 0.294 | 0.248 | 0.195 |
| `layer_6` | 0.146 | 0.627 | 0.006 | 0.445 | 0.380 | 0.200 |
| `layer_11` | 0.154 | 0.642 | 0.010 | 0.402 | 0.338 | 0.171 |

这个表最重要的读法是：

1. `layer_0` 已经有局部语义，但仍然相对干净。
2. `layer_6` 和 `layer_11` 开始更明显吸收 context / domain / frequency。
3. 如果目标是看“单个 patch 自己有没有语义”，`layer_0` 比深层更合适。
4. `NMI prior-guided probe` 只是说明 KMeans cluster 和 prior-guided motif probe 的一致程度；它不是 cluster 命名来源。

新增 macro-domain view 后，还可以补充一个更直观的跨领域诊断：

| representation | same-cluster macro-domain cells | weak-match cells | same-cluster rate, real domains only |
|---|---:|---:|---:|
| `raw_patch` | 45/90 | 40/90 | 41/75 |
| `projection` | 73/90 | 2/90 | 64/75 |
| `layer_0` | 75/90 | 7/90 | 68/75 |
| `layer_6` | 62/90 | 13/90 | 59/75 |
| `layer_11` | 58/90 | 9/90 | 55/75 |

这里的 `same-cluster macro-domain cells` 指：对某个 cluster center 和某个 macro-domain，找到的最近 patch 仍然被 KMeans 分到同一个 cluster。  
这个指标不是最终评价指标，但它很适合解释给老师听：`layer_0` 的 local vocabulary 不仅视觉上干净，而且跨真实领域更容易找到同 cluster 的近邻；中后层虽然 cluster stability 更高，但跨领域 nearest view 中虚线更多，说明 representation 已经更明显混入 domain/style/context。

## 5. 这说明了什么

我们现在可以更稳地说：

- `raw patch` 说明原空间已经有 motif-like shape family。
- `layer_0` 更像 local vocabulary。
- `layer_6` 更像 contextual mixing。
- `layer_11` 更像 contextualized concept space。

所以老师说的那句“early layers 更保留 local info”是有证据支撑的，而且对 Chronos-2 尤其适合。

## 6. 我们接下来该怎么做

下一步不应该再只看一个 cluster 图，也不应该只追求“画得更好看”。应该按老师关心的问题，把实验升级为一个可复现的 multi-layer validation：

1. 覆盖 `projection`、`layer_0`、`layer_6`、`layer_11`，不要只看 `layer_0`。
2. 把当前 `100 windows/dataset` pilot 扩展到更稳的 `500 windows/dataset`，必要时做 `1000 windows/dataset` sanity check。
3. 从 source-domain balanced 改成 macro-domain balanced，并限制 macro-domain 内单个 dataset 的最大贡献。
4. 用 K sweep 选择 final K setting，综合 silhouette、Calinski-Harabasz、Davies-Bouldin、seed stability、KMeans vs Agglomerative NMI、cluster size、confounder NMI 和 original-space interpretability。
5. 以 KMeans center 为锚点，系统导出 center-nearest raw patch examples。
6. 生成 confidence-filtered macro-domain examples：没有可信 match 的 cell 留空或标记 no confident match，不再强行展示弱样本。
7. 把 prior-guided motif 作为 audit probe，检查 model-derived clusters 和 human-prior motif 是否错位，但不把它作为 ground truth。
8. 把主证据图和 diagnostic/failure-case 图分开：主证据必须稳定、跨领域、原空间可解释、confounder 风险低。

一句话版本：

> 如果我们要证明 single patch 仍然带着局部语义，需要看 projection 和 early layer；如果我们要解释 context 如何重组 motif，则必须同时看 layer_6 和 layer_11，并用严格的 K selection、macro-domain validation 和 original-space evidence 支撑结论。
