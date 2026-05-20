# Distance Metric Ablation Report: DTW vs Euclidean for Chronos-2 Patch Concepts

## 1. Advisor Question

这份报告回答一个很具体的问题：Chronos-2 的 model-derived motif clusters 在 original time-series space 中看起来不够 coherent，是否部分来自 distance function / prototype selection metric 不合适。

最新 verdict：Euclidean/KMeans cluster label 不能作为最终 motif taxonomy。它只能作为 TSFM representation geometry 的 diagnostic baseline / candidate neighborhood sampler；真正进入 `model-derived motif taxonomy v1` 的 candidate motif/prototype family，必须通过 warping-aware original-space validation，尤其是 DTW-aware prototype selection、controlled retrieval 和 confounder audit。

本报告采用 **two-space distance principle**：在 `Chronos-2 representation space` 中用 Euclidean geometry 发现模型内部的 patch-token neighborhoods；在 `original time-series space` 中用 DTW geometry 验证这些 neighborhoods 是否对应 coherent temporal shapes。换句话说，Euclidean 用来回答“模型认为哪些 token 相近”，DTW 用来回答“这些 token 回到原空间后是否同形”。

## 2. Method

- fixed model: `Chronos-2`
- representation-space candidate generation: `StandardScaler -> PCA(max 30 dims) -> KMeans` with Euclidean geometry
- K selection remains a representation-space operating-point choice, not a DTW clustering objective
- original-space validation metrics: constrained DTW as the primary shape-coherence metric; z-normalized raw Euclidean and `1 - |correlation|` as diagnostic controls
- DTW setting: Sakoe-Chiba radius `2`; radius sensitivity `1/2/3` is recorded in the summary JSON
- prototype figures use shape-eligible patches when a cluster is not flat-dominated; low-information / near-flat patches are retained only for flat-dominated diagnostic clusters
- zero-reference guide lines are removed from all patch waveform panels, so horizontal curves represent actual low-information patches rather than plotting guides
- external weak motif labels are excluded from the main evidence because the current deterministic probe is not reliable enough for paper-level claims

## 3. Visual Evidence I: Embedding Cluster Maps

每张图从左到右分别是 KMeans cluster、cluster-level DTW gain、confounder risk。DTW gain 为正表示该 cluster 在 DTW 下相对 raw Euclidean 更紧；confounder risk 越高，越不应把该 cluster 命名为 motif/prototype family。

![projection_k6 embedding audit](../outputs/distance_metric_ablation/figures/embedding_cluster_audit_projection_k6.png)

![layer_0_k6 embedding audit](../outputs/distance_metric_ablation/figures/embedding_cluster_audit_layer_0_k6.png)

![layer_6_k6 embedding audit](../outputs/distance_metric_ablation/figures/embedding_cluster_audit_layer_6_k6.png)

![layer_11_k6 embedding audit](../outputs/distance_metric_ablation/figures/embedding_cluster_audit_layer_11_k6.png)

![layer_6_k10 embedding audit](../outputs/distance_metric_ablation/figures/embedding_cluster_audit_layer_6_k10.png)

## 4. Visual Evidence II: Prototype Selection Comparison

每一行展示一个 representation-space cluster；四列分别使用 representation center、raw Euclidean medoid、correlation medoid 和 DTW medoid 选择 prototype patches。前一列回答模型空间中心是什么，后三列回答原空间用不同距离看会得到什么 prototype。所有 clusters 都展示，不做 cherry-picking。为避免横线型 low-information patches 污染非 flat clusters，本版对非 flat-dominated cluster 使用 shape-eligible subset；flat-dominated cluster 仍保留为 diagnostic。

![projection_k6 prototype comparison](../outputs/distance_metric_ablation/figures/prototype_metric_comparison_projection_k6.png)

![layer_0_k6 prototype comparison](../outputs/distance_metric_ablation/figures/prototype_metric_comparison_layer_0_k6.png)

![layer_6_k6 prototype comparison](../outputs/distance_metric_ablation/figures/prototype_metric_comparison_layer_6_k6.png)

![layer_11_k6 prototype comparison](../outputs/distance_metric_ablation/figures/prototype_metric_comparison_layer_11_k6.png)

![layer_6_k10 prototype comparison](../outputs/distance_metric_ablation/figures/prototype_metric_comparison_layer_6_k10.png)

## 5. Quantitative Evidence

下图按 cluster 汇总 original-space distance diagnostics。ratio 是 intra-cluster distance / matched random baseline distance；越低说明该 metric 下 cluster 越紧。DTW ratio 是 motif/prototype 命名时更重要的 gate；raw Euclidean / correlation 用于说明距离选择是否改变解释。

![distance metric heatmap](../outputs/distance_metric_ablation/figures/distance_metric_heatmap.png)

| setting | cluster | DTW gain ↑ | raw ratio ↓ | DTW ratio ↓ | low-info ↓ | macro diversity ↑ | confounder ↓ | label |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `layer_6_k10` | C3 | 0.195 | 0.706 | 0.511 | 0.000 | 2 | 0.989 | `confounded` |
| `layer_6_k6` | C1 | 0.178 | 0.922 | 0.743 | 0.000 | 5 | 0.649 | `dtw_benefited` |
| `layer_6_k10` | C2 | 0.139 | 0.595 | 0.456 | 0.000 | 4 | 0.726 | `dtw_benefited` |
| `layer_0_k6` | C3 | 0.118 | 1.013 | 0.895 | 0.000 | 5 | 0.903 | `confounded` |
| `layer_6_k10` | C0 | 0.082 | 0.750 | 0.668 | 0.014 | 5 | 0.427 | `dtw_benefited` |
| `projection_k6` | C0 | 0.066 | 0.904 | 0.839 | 0.000 | 6 | 0.452 | `unchanged` |
| `layer_6_k6` | C2 | 0.052 | 0.806 | 0.754 | 0.014 | 5 | 0.375 | `unchanged` |
| `layer_11_k6` | C0 | 0.045 | 0.570 | 0.525 | 0.006 | 6 | 0.254 | `unchanged` |

## 6. Retrieval Comparison

同一个 query patch 分别用 representation Euclidean、raw Euclidean、correlation 和 DTW 做 retrieval。若 DTW 找到的 neighbors 更同形但与 representation retrieval overlap 很低，说明 cluster 可能包含多个 raw-shape subfamilies。

![retrieval comparison](../outputs/distance_metric_ablation/figures/retrieval_metric_comparison_examples.png)

## 7. Failure Cases

DTW 不是无条件更好。下面展示 DTW over-warping、representation-near but raw-DTW-far、raw-DTW-near but representation-far 三类风险。

![DTW failure cases](../outputs/distance_metric_ablation/figures/dtw_failure_cases.png)

| case | rep distance | DTW distance | shape corr |
|---|---:|---:|---:|
| DTW over-warping | 11.889 | 0.000 | 0.067 |
| Rep-near raw-DTW-far | 11.295 | 2.565 | 0.282 |
| Raw-DTW-near rep-far | 35.917 | 0.180 | 0.945 |

## 8. Final Answer

当前 ablation 支持一个更明确的结论：Euclidean representation clustering 不能作为最终 motif taxonomy 机制。它可以揭示 Chronos-2 patch-token representation geometry，但会把 time-shifted / phase-shifted / locally warped shapelet-like patterns 处理得不够稳健。本轮共有 `3` 个 cluster 被标记为 `dtw_benefited`，但也有 `7` 个 cluster 存在较高 confounder risk。

因此，后续路线不是抛弃 Euclidean，而是明确分工：Euclidean/KMeans 保留为 representation-space neighborhood discovery；DTW-aware original-space validation 升级为命名 candidate motif/prototype family 的必要条件。对于 spike / burst / oscillation 等局部错位敏感 motif，优先使用 DTW medoid 和 DTW controlled retrieval 做证据；同时保留 failure cases，防止 DTW over-warping 被误读成真实 temporal concept。
