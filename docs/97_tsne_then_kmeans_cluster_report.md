# t-SNE 降维后 KMeans 聚类结果与原型图

> 这份报告只看 `projection`、`layer_6` 和 `layer_11`。流程是先对 Chronos-2 representation 做 `StandardScaler + PCA precompression`，再用 t-SNE 降到 2D，并在这个 2D t-SNE space 中做 KMeans。也就是说，这份报告回答的是“t-SNE 可视化空间里能否形成直观 clusters”。

## 1. 方法说明

实际流程：

```text
Chronos-2 representation
-> StandardScaler
-> PCA precompression up to 50 dimensions
-> t-SNE to 2D
-> KMeans in 2D t-SNE space
-> t-SNE scatter visualization
-> t-SNE-space KMeans-center nearest raw patches as examples
```

K setting 使用每层当前最佳值：`projection K=6`，`layer_6 K=10`，`layer_11 K=6`。`layer_6` 用 `K=10` 是因为 per-layer K search 指向更细的 middle-layer split。

注意：t-SNE 强调局部邻域可视化，不保留严格全局距离。因此这份报告主要用于 human inspection 和 presentation，不替代高维 representation-space stability / NMI / DTW validation。

## Projection

- K: `6`
- clustered patches: `5100`
- t-SNE perplexity: `50.0`

![Projection t-SNE KMeans](../outputs/tsne_cluster_domain_reports/figures/projection_tsne_kmeans_k6.png)

![Projection center-nearest prototypes](../outputs/tsne_cluster_domain_reports/figures/projection_tsne_kmeans_k6_center_nearest.png)

读法：上图是 t-SNE 2D space，颜色是 KMeans labels；下图是每个 cluster 中离 t-SNE-space KMeans center 最近的 raw patches。cluster ID 只表示可视化空间中的 neighborhood，不是 motif 名字。

## Layer 6

- K: `10`
- clustered patches: `5100`
- t-SNE perplexity: `50.0`

![Layer 6 t-SNE KMeans](../outputs/tsne_cluster_domain_reports/figures/layer_6_tsne_kmeans_k10.png)

![Layer 6 center-nearest prototypes](../outputs/tsne_cluster_domain_reports/figures/layer_6_tsne_kmeans_k10_center_nearest.png)

读法：上图是 t-SNE 2D space，颜色是 KMeans labels；下图是每个 cluster 中离 t-SNE-space KMeans center 最近的 raw patches。cluster ID 只表示可视化空间中的 neighborhood，不是 motif 名字。

## Layer 11

- K: `6`
- clustered patches: `5100`
- t-SNE perplexity: `50.0`

![Layer 11 t-SNE KMeans](../outputs/tsne_cluster_domain_reports/figures/layer_11_tsne_kmeans_k6.png)

![Layer 11 center-nearest prototypes](../outputs/tsne_cluster_domain_reports/figures/layer_11_tsne_kmeans_k6_center_nearest.png)

读法：上图是 t-SNE 2D space，颜色是 KMeans labels；下图是每个 cluster 中离 t-SNE-space KMeans center 最近的 raw patches。cluster ID 只表示可视化空间中的 neighborhood，不是 motif 名字。

## 结论边界

- 这些图可以说明 Chronos-2 representation space 中存在稳定 neighborhood。
- `projection` 更接近 local patch token vocabulary。
- `layer_6 K=10` 用来观察 middle layer 的更细 contextual substructure。
- `layer_11` 更稳定但不一定更适合直接命名 raw-shape motif。
- 是否能称为 candidate motif/prototype family，还需要 original-space DTW validation 和 domain/frequency/position confounder audit。
