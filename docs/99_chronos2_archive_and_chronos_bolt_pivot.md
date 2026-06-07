# Chronos-2 证据归档与 Chronos-Bolt 转向说明

> 日期：2026-05-20  
> 状态：研究路线转向记录。  
> 结论：此前围绕 `Chronos-2` 的 layer-wise clustering、projection / hidden-state analysis、PCA/t-SNE visualization 和 DTW audit 统一归档为历史探索材料；后续默认使用 `Chronos-Bolt` 作为 Chronos family 的主模型。

## 1. 为什么要归档 Chronos-2 证据

我们原本希望用 `Chronos-2` 的 `projection`、`layer_0`、`layer_6`、`layer_11` 来回答：

- single patch 是否仍保留 independent local temporal information；
- early / middle / late layers 如何从 local patch vocabulary 走向 contextualized representation；
- KMeans center-nearest patches 回到 original time-series space 后是否能解释为 candidate motif/prototype family。

但进一步检查 Chronos-2 tokenizer / input patch embedding 后发现，`projection` 并不是只由 raw patch value 或 normalized patch value 得到。Chronos-2 的 `input_patch_embedding` 输入是：

```text
[time encoding, normalized patch values, patch mask]
```

具体流程是：

```text
raw context
-> instance normalization / scaling
-> patching, patch size = 16
-> concatenate time encoding, normalized patch values, patch mask
-> residual MLP input patch embedding
-> d_model = 768 patch token
```

也就是说，Chronos-2 的 `projection` token 已经混入了 explicit time encoding 和 mask information。对于 forecasting 这是合理设计，但对我们当前的机制问题会带来解释混淆：我们想问的是 **patch token 是否从 local value shape 中学习到 motif/prototype vocabulary**，而 Chronos-2 projection 本身已经不是一个干净的 value-only patch representation。

因此，之前关于 `projection` 的结论不能再被表述为“只来自 single raw patch value 的 local vocabulary”。更谨慎的说法是：

> Chronos-2 projection represents a time-encoded, mask-aware, normalized patch token, not a pure value-only patch embedding.

这个差异足够重要，所以我们决定把 Chronos-2 主线归档。

## 2. 哪些材料归档

以下报告和素材保留为历史探索材料，不再作为后续主线证据：

- `docs/11_chronos_layer_effect_main_report.md`
- `docs/12_distance_metric_ablation_report.md`
- `docs/94_weekly_advisor_ppt_draft.md`
- `docs/95_dynamical_systems_story_ppt_draft.md`
- `docs/96_notion_dynamical_story_report.md`
- `docs/97_tsne_then_kmeans_cluster_report.md`
- `docs/98_tsne_domain_label_report.md`
- `docs/97_pca_then_kmeans_cluster_report.md`
- `docs/98_pca_domain_label_report.md`
- `outputs/chronos_multilayer_validation/`
- `outputs/distance_metric_ablation/`
- `outputs/tsne_cluster_domain_reports/`
- `outputs/pca_cluster_domain_reports/`

这些材料仍然有价值，因为它们记录了：

- layer-wise representation audit 的流程；
- K selection、center-nearest examples、macro-domain filtered examples 的图表模板；
- two-space distance principle；
- DTW original-space validation 的必要性；
- domain/frequency/position confounder audit 的重要性。

但它们不再用于支撑 “Chronos-2 projection 是纯 local patch vocabulary” 这类结论。

## 3. 后续为什么改用 Chronos-Bolt

后续默认主线改为 `Chronos-Bolt`，原因是它更适合我们当前问题：

1. **更接近 patch-value based tokenization 问题。**  
   我们需要一个更干净的 patch token representation 来问：模型是否从 local patch value 中学习 shapelet-like motif/prototype family。

2. **避免 explicit time encoding 混淆 projection 层解释。**  
   Chronos-2 的 projection token 包含 time encoding；这会让 projection-level clustering 同时反映 value shape、time position / encoding、mask 等因素。

3. **保留 Chronos family 的可比性。**  
   Chronos-Bolt 仍属于 Chronos family，适合延续我们已有的模型加载、patch extraction、hidden state audit 和 original-space validation 流程。

4. **更适合回答老师的问题。**  
   老师关心的是 single patch 是否独立保留 local information。Chronos-Bolt 更适合作为下一步验证这个问题的主模型。

## 4. 新默认研究问题

后续报告默认把问题改写为：

> In Chronos-Bolt, do patch-value-based token representations preserve local temporal information, and how are these local motif/prototype candidates reorganized across transformer layers?

中文讲法：

> 在 Chronos-Bolt 中，patch-value-based token representation 是否保留 single-patch local temporal information？这些 local motif/prototype candidates 又如何在 hidden layers 中被重组为 contextualized motif/prototype families？

## 5. 新实验路线

后续应优先重跑以下分析，模型改为 `Chronos-Bolt`：

1. **Model loading and representation extraction audit**  
   确认 Chronos-Bolt 的 tokenizer / input patch embedding 输入到底包含哪些信息：normalized patch values、mask、是否有 explicit time encoding、是否有 reg token。

2. **Layer-wise representation extraction**  
   选择 tokenizer/projection、early layer、middle layer、top layer。

3. **Dimensionality visualization**  
   同时画 PCA 和 t-SNE：
   - clustering view；
   - source-domain / macro-domain label view；
   - patch-index / frequency confounder view。

4. **KMeans center-nearest raw patch examples**  
   按老师要求，用 KMeans center 作为 center，nearest points 作为 examples。

5. **Original-space validation**  
   用 DTW-aware validation 判断哪些 clusters 可以作为 candidate motif/prototype family。

6. **Transition-aware hypothesis**  
   如果 Chronos-Bolt 中 early layer 证据更干净，再继续讨论 prototype state transition geometry 是否能指导 TSFM 设计。

## 6. 对旧结论的改写边界

旧结论中仍可保留：

- `Chronos-2` hidden representations 存在可聚类结构；
- higher layers 更 contextualized；
- KMeans center-nearest + original-space inspection 是有效证据形式；
- DTW validation 对 original-space motif/prototype naming 很重要；
- domain/frequency/position confounder audit 必须保留。

旧结论中需要撤回或弱化：

- 不再把 Chronos-2 `projection` 解释为 pure value-only local patch vocabulary；
- 不再把 Chronos-2 layer-wise结果作为后续主线；
- 不再基于 Chronos-2 projection 图直接推断 single raw patch independent information；
- 不再用 Chronos-2 作为默认模型设计启发的唯一依据。

一句话总结：

> Chronos-2 results are archived as a useful diagnostic pilot, but Chronos-Bolt becomes the default model for the next clean test of patch-level temporal primitive learning.

