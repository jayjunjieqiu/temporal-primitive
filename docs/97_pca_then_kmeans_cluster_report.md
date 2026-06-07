# PCA-space KMeans 聚类结果与原型图

> 这份报告只看 `projection`、`layer_6` 和 `layer_11`。流程是先对 Chronos-2 representation 做 `StandardScaler + PCA(max 30 dims)`，再在 PCA space 中做 KMeans。图上的散点只显示 PC1/PC2，因此二维图是可视化，不是完整聚类空间。

## 1. 方法说明

实际流程：

```text
Chronos-2 representation
-> StandardScaler
-> PCA up to 30 dimensions
-> KMeans in PCA space
-> PC1/PC2 scatter visualization
-> KMeans-center nearest raw patches as examples
```

K setting 使用每层当前最佳值：`projection K=6`，`layer_6 K=10`，`layer_11 K=6`。`layer_6` 用 `K=10` 是因为 per-layer K search 指向更细的 middle-layer split。

注意：PCA 图更适合解释线性主方向和全局结构；是否能称为 motif/prototype family，仍需 DTW 和 confounder audit。

## Projection

- K: `6`
- clustered patches: `5100`

![Projection PCA KMeans](../outputs/pca_cluster_domain_reports/figures/projection_pca_kmeans_k6.png)

![Projection center-nearest prototypes](../outputs/pca_cluster_domain_reports/figures/projection_pca_kmeans_k6_center_nearest.png)

读法：上图是 PCA space 的前两维投影，颜色是 KMeans labels；下图是每个 cluster 中离 KMeans center 最近的 raw patches。cluster ID 只表示 representation-space neighborhood，不是 motif 名字。

## Layer 6

- K: `10`
- clustered patches: `5100`

![Layer 6 PCA KMeans](../outputs/pca_cluster_domain_reports/figures/layer_6_pca_kmeans_k10.png)

![Layer 6 center-nearest prototypes](../outputs/pca_cluster_domain_reports/figures/layer_6_pca_kmeans_k10_center_nearest.png)

读法：上图是 PCA space 的前两维投影，颜色是 KMeans labels；下图是每个 cluster 中离 KMeans center 最近的 raw patches。cluster ID 只表示 representation-space neighborhood，不是 motif 名字。

## Layer 11

- K: `6`
- clustered patches: `5100`

![Layer 11 PCA KMeans](../outputs/pca_cluster_domain_reports/figures/layer_11_pca_kmeans_k6.png)

![Layer 11 center-nearest prototypes](../outputs/pca_cluster_domain_reports/figures/layer_11_pca_kmeans_k6_center_nearest.png)

读法：上图是 PCA space 的前两维投影，颜色是 KMeans labels；下图是每个 cluster 中离 KMeans center 最近的 raw patches。cluster ID 只表示 representation-space neighborhood，不是 motif 名字。

## 结论边界

- 这些图可以说明 Chronos-2 representation space 中存在稳定 neighborhood。
- `projection` 更接近 local patch token vocabulary。
- `layer_6 K=10` 用来观察 middle layer 的更细 contextual substructure。
- `layer_11` 更稳定但不一定更适合直接命名 raw-shape motif。
- 是否能称为 candidate motif/prototype family，还需要 original-space DTW validation 和 domain/frequency/position confounder audit。
