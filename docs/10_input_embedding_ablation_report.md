# Input Embedding Ablation: pre-transformer token 是否更适合研究 patch vocabulary？

## 1. 为什么做这个 ablation

我们前面的聚类主要使用 selected transformer layer hidden states。这个表示已经经过 attention/contextualization，因此 cluster 可能同时编码 patch 形状、位置、频率、domain-style 和上下文角色。本轮 ablation 的目标是把“patch 自身的词汇表示”和“上下文化后的时序概念”分开。

## 2. 对照的 representation

- `raw_z_patch`: raw patch 做 robust z-normalization 后直接聚类；这是形状本身的 baseline。
- `chronos_proj_with_time`: Chronos `_prepare_patched_context` 后经 `input_patch_embedding`；包含 per-observation time encoding、patch value、mask。
- `chronos_proj_time_zeroed`: 将 Chronos patch input 中的 time encoding 通道置零后再过 `input_patch_embedding`；不是官方前向路径，只作为 position-source diagnostic。
- `chronos_hidden`: Chronos encoder selected layer output，已经过 transformer contextualization。
- `timesfm_tokenizer`: TimesFM running RevIN 后的 patch 经 tokenizer projection；进入 `stacked_xf` 前，无显式 absolute position embedding，但 running stats 是顺序相关的。
- `timesfm_hidden`: TimesFM selected transformer layer output，已经过 causal self-attention 和 RoPE。

特别注意：pre-transformer 并不自动等于 position-free。Chronos-2 的 projection 输入显式拼接了 time encoding；TimesFM-2.5 的 tokenizer 前没有显式 absolute position embedding，但 running RevIN 是顺序相关的。

## 3. 运行设置

- windows per dataset: `40`
- context length: `128`
- domain-balanced patches per domain: `300`
- seed: `47`

## 4. 主要指标

| model | representation | silhouette | stability | NMI taxonomy-v0 | NMI patch-index | NMI domain | NMI frequency |
|---|---|---:|---:|---:|---:|---:|---:|
| Chronos-2-small layer_5 | `raw_z_patch` | 0.216 | 0.543 | 0.026 | 0.004 | 0.019 | 0.016 |
| Chronos-2-small layer_5 | `chronos_proj_with_time` | 0.109 | 0.537 | 0.156 | 0.005 | 0.224 | 0.163 |
| Chronos-2-small layer_5 | `chronos_proj_time_zeroed` | 0.117 | 0.536 | 0.153 | 0.006 | 0.222 | 0.160 |
| Chronos-2-small layer_5 | `chronos_hidden` | 0.187 | 0.628 | 0.174 | 0.007 | 0.340 | 0.260 |
| Chronos-2 layer_11 | `raw_z_patch` | 0.216 | 0.543 | 0.026 | 0.004 | 0.019 | 0.016 |
| Chronos-2 layer_11 | `chronos_proj_with_time` | 0.121 | 0.505 | 0.109 | 0.006 | 0.174 | 0.142 |
| Chronos-2 layer_11 | `chronos_proj_time_zeroed` | 0.124 | 0.567 | 0.109 | 0.006 | 0.180 | 0.143 |
| Chronos-2 layer_11 | `chronos_hidden` | 0.153 | 0.616 | 0.183 | 0.006 | 0.379 | 0.300 |
| TimesFM-2.5 layer_10 | `raw_z_patch` | 0.866 | 0.731 | 0.013 | 0.005 | 0.011 | 0.009 |
| TimesFM-2.5 layer_10 | `timesfm_tokenizer` | 0.026 | 0.335 | 0.105 | 0.005 | 0.148 | 0.134 |
| TimesFM-2.5 layer_10 | `timesfm_hidden` | 0.159 | 0.791 | 0.176 | 0.300 | 0.409 | 0.369 |

指标总览图：

![Input embedding ablation metrics](../outputs/input_embedding_ablation/figures/input_embedding_ablation_metric_overview.png)

## 5. 这个结果回答了什么

- 对 TimesFM-2.5，你的担心基本成立：`timesfm_tokenizer` 的 patch-index NMI 是 `0.005`，而 `timesfm_hidden` 是 `0.300`。这说明前面看到的 TimesFM position confounding 主要来自 transformer contextualization / causal attention / RoPE，而不是 tokenizer projection 本身。
- 但 `timesfm_tokenizer` 的 stability 只有 `0.335`，明显低于 hidden layer 的 `0.791`。所以 tokenizer 更干净，但结构也更弱；它适合做 patch vocabulary baseline，不足以单独替代 hidden-state analysis。
- 对 Chronos-2，`chronos_proj_with_time` 与 `chronos_proj_time_zeroed` 的 cluster ARI 是 `0.904`，说明在本轮 128-step windows 上，显式 time encoding 对 projection-level cluster 影响不大。Chronos 的 patch-index NMI 在 projection 和 hidden 中都很低。
- Chronos-2 hidden layer 的 domain/frequency NMI 从 projection 的 `0.174` / `0.142` 升到 `0.379` / `0.300`。这提示 transformer 层会强化 domain/cadence-style，而不只是强化人类可见 shape。

## 6. 关键图像证据

### 6.1 TimesFM-2.5: tokenizer vs layer_10

Tokenizer PCA by cluster：

![TimesFM tokenizer clusters](../outputs/input_embedding_ablation/figures/timesfm_2_5_timesfm_tokenizer_pca_clusters.png)

Tokenizer PCA by patch_index：

![TimesFM tokenizer patch index](../outputs/input_embedding_ablation/figures/timesfm_2_5_timesfm_tokenizer_pca_patch_index.png)

Layer_10 PCA by cluster：

![TimesFM hidden clusters](../outputs/input_embedding_ablation/figures/timesfm_2_5_timesfm_hidden_pca_clusters.png)

Layer_10 PCA by patch_index：

![TimesFM hidden patch index](../outputs/input_embedding_ablation/figures/timesfm_2_5_timesfm_hidden_pca_patch_index.png)

### 6.2 Chronos-2: projection with time vs time-zeroed vs layer_11

![Chronos projection with time](../outputs/input_embedding_ablation/figures/chronos_2_chronos_proj_with_time_pca_clusters.png)

![Chronos projection time-zeroed](../outputs/input_embedding_ablation/figures/chronos_2_chronos_proj_time_zeroed_pca_clusters.png)

![Chronos hidden layer](../outputs/input_embedding_ablation/figures/chronos_2_chronos_hidden_pca_clusters.png)

## 7. 初步结论

1. `raw_z_patch` 和 `projection/tokenizer` 应该作为后续所有 concept discovery 的必要 baseline。若某个 cluster 在 raw/proj 中已经存在，它更像 patch-shape vocabulary；若只在 hidden layer 中出现，它才更像 contextualized temporal concept。
2. pre-transformer token 不应被直接称为无位置问题。Chronos 的官方 projection 明确吃入 time encoding；TimesFM 的 running normalization 也会让前后 patch 的统计分布不同。
3. 对导师问题的更严谨表述应改为：TSFM 的时序语言至少有两层，第一层是 local patch vocabulary，第二层是 transformer contextualized temporal grammar。

## 8. 下一步建议

下一步不要抛弃 hidden states，而是把 taxonomy/concept 发现改成双通道：先在 pre-transformer projection 上发现 local vocabulary，再追踪这些 vocabulary 在 hidden layers 中如何合并、分裂或变成 position/context artifact。

具体建议：对 TimesFM c5/c8 和 negative control c4，补做 raw/proj/hidden 的 cluster lineage：看同一批 patch 在 projection cluster 和 layer_10 cluster 之间的转移矩阵。