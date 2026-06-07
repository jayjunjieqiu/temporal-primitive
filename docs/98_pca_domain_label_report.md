# PCA 降维后直接打上 domain 标签的诊断图

> 这份报告不做聚类命名，只问一个 confounder 问题：Chronos-2 representation 降维后的几何结构，是不是明显被 domain/source-domain 分开？

这里主图使用 `source_domain` 标签，因为它最接近数据集描述里的 domain；同时补充 `macro_domain` 标签，便于快速看 Traffic / Energy / Environment 等大类是否主导结构。

重要提醒：这些图只是 PC1/PC2 可视化。domain 混杂不能只靠肉眼判断，最终仍应结合 NMI、controlled retrieval 和 macro-domain evidence。

## Projection

### Source-domain labels

![Projection source-domain labels](../outputs/pca_cluster_domain_reports/figures/projection_pca_source_domain_labels.png)

### Macro-domain labels

![Projection macro-domain labels](../outputs/pca_cluster_domain_reports/figures/projection_pca_macro_domain_labels.png)

读法：如果同一颜色在 PC1/PC2 上形成大块分离，说明该层 representation 可能编码了较强 domain/source-domain information；如果颜色混合较多，说明该二维视图下 domain confounding 较弱。但这不是充分证据，仍需看 NMI 指标。

## Layer 6

### Source-domain labels

![Layer 6 source-domain labels](../outputs/pca_cluster_domain_reports/figures/layer_6_pca_source_domain_labels.png)

### Macro-domain labels

![Layer 6 macro-domain labels](../outputs/pca_cluster_domain_reports/figures/layer_6_pca_macro_domain_labels.png)

读法：如果同一颜色在 PC1/PC2 上形成大块分离，说明该层 representation 可能编码了较强 domain/source-domain information；如果颜色混合较多，说明该二维视图下 domain confounding 较弱。但这不是充分证据，仍需看 NMI 指标。

## Layer 11

### Source-domain labels

![Layer 11 source-domain labels](../outputs/pca_cluster_domain_reports/figures/layer_11_pca_source_domain_labels.png)

### Macro-domain labels

![Layer 11 macro-domain labels](../outputs/pca_cluster_domain_reports/figures/layer_11_pca_macro_domain_labels.png)

读法：如果同一颜色在 PC1/PC2 上形成大块分离，说明该层 representation 可能编码了较强 domain/source-domain information；如果颜色混合较多，说明该二维视图下 domain confounding 较弱。但这不是充分证据，仍需看 NMI 指标。

## 对 TSFM 设计的启发

- 如果 early layer 的 PCA map 更按 local shape 而不是 domain 分开，它更适合作为 `local patch vocabulary` 的观察窗口。
- 如果 middle/top layer 的 PCA map 更受 domain 或 macro-domain 影响，说明 contextual representation 可能混入 domain/cadence information。
- 新 TSFM 设计里需要显式区分 `shared temporal commonality` 和 `domain-specific heterogeneity`，否则 prototype family 可能退化成 domain labels。
