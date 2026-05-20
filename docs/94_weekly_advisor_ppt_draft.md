# 本周导师汇报 PPT 草稿：Chronos-2 Patch Token 到底学到了什么？

> 用途：本周给 Yuxuan Liang 老师汇报的 PPT 文案、图片选择和排版建议。  
> 当前主线：`Chronos-2 only`；从 `projection / layer_0 / layer_6 / layer_11` 看 patch representation 如何从 local patch vocabulary 变成 contextualized representation。  
> 核心口径：我们提出的是 `model-derived motif/prototype family discovery protocol`，不是 final motif taxonomy。

## 0. 汇报总策略

### 一句话版本

> 按老师建议，我们本周把实验收敛到 Chronos-2：在 representation space 中用 Euclidean geometry 发现 patch-token neighborhoods，在 original time-series space 中用 DTW-aware validation 检查这些 neighborhoods 是否仍保留可解释的 local temporal information，并比较 `projection -> layer_0 -> layer_6 -> layer_11` 的层间变化。

### 本周相对上一周的变化

- 从 `TimesFM / Chronos / Chronos-small` 的 broad pilot 收敛为 `Chronos-2 only`。
- 从 prior-guided motif labels 作为解释锚点，转为把它移出主证据链。
- 从“cluster 看起来像 motif”升级为 `representation-space discovery + original-space DTW validation`。
- 从挑选好看的 cluster，改成展示 all-cluster evidence、K selection、confounder audit 和 failure cases。

### 汇报时不要主动过度 claim

- 不说：我们已经发现了最终 `motif taxonomy`。
- 要说：我们得到了一套更严谨的 `model-derived motif/prototype family discovery protocol`，并形成 Chronos-2 pilot evidence。
- 不说：Euclidean 聚类是 motif taxonomy。
- 要说：Euclidean/KMeans 是 representation-space candidate generator；DTW 是 original-space motif/prototype naming gate。

## 1. 推荐页序

建议 12 页主线 + 若干 backup。每页只回答一个问题。

## Slide 1. Opening Question

### 屏幕文字

**What do Chronos-2 patch tokens learn?**

From local patch vocabulary to contextualized motif/prototype families

### 排版建议

- 白底，左侧大标题，右侧画简单流程：
  `raw series -> patch -> projection -> layer 0 -> layer 6 -> layer 11`
- 页脚小字：`Frozen Chronos-2 representation analysis; no fine-tuning.`

### 口头讲法

> 老师上次建议我们先聚焦 Chronos-2，并回答 single patch 是否仍然保留 local information。本周我把问题收敛成 layer-wise mechanism study：不同层的 patch token representation 到底保留了什么，又从什么时候开始融合 context。

## Slide 2. Advisor Questions and Our Framing

### 屏幕文字

**Advisor questions**

- Does a single patch token still preserve local temporal information?
- Are early layers more local than middle/top layers?
- Can representation clusters be interpreted in original time-series space?
- Is poor visual coherence caused by the distance metric?

**Our framing**

`representation-space discovery` + `original-space DTW validation`

### 排版建议

- 左侧：老师问题，使用四个 question blocks。
- 右侧：two-space principle 小图：
  - `Chronos representation space: Euclidean / KMeans`
  - `Original time-series space: DTW / prototype validation`

### 口头讲法

> 这里我不直接说 cluster 就是 motif。我们先问模型空间里哪些 patch token 近，再问这些邻域回到原空间后是否仍然同形。

## Slide 3. Experimental Protocol

### 屏幕文字

**Chronos-2-only layer-wise validation**

- Representations: `projection`, `layer_0`, `layer_6`, `layer_11`
- Data: BasicTS non-BLAST datasets
- Sampling: macro-domain balanced
- Context length: `128`; patch length: `16`
- No fine-tuning, no weight modification

### 推荐图片

可手动画一个 4-stage pipeline。若要放现成图，优先使用：

![Layer comparison summary](../outputs/chronos_multilayer_validation/figures/layer_comparison_summary.png)

但这张图更适合 Slide 5；Slide 3 可以保持简洁。

### 口头讲法

> 我们这次不是 full benchmark，而是 mechanism-oriented analysis。为了避免 traffic/weather 这类大域主导，主分析使用 macro-domain balanced sampling。

## Slide 4. K Is a Representation-Space Operating Point

### 屏幕文字

**K selection is not silhouette-only.**

We jointly check:

- cluster separation and size
- KMeans seed stability
- KMeans vs Agglomerative agreement
- domain / frequency / patch-index confounding
- original-space inspectability

### 推荐图片

![K selection summary](../outputs/chronos_multilayer_validation/figures/k_selection_summary.png)

### 排版建议

- 图占页面 70-78% 宽度。
- 右侧放结论框：
  - `shared K = 6`
  - `layer_6 specific K = 10`
  - `K is for representation neighborhoods, not final taxonomy`

### 口头讲法

> 这里 K=6 不是说世界上只有 6 类 motif，而是为了四层横向比较的 shared operating point。layer_6 的 per-layer search 指向 K=10，说明 middle layer 可能存在更细的 contextual substructure。

## Slide 5. Layer-Wise Validation: What Changes Across Layers?

### 屏幕文字

**Early layers are more local; middle/top layers mix more context and confounders.**

Metrics:

- `stability ↑`: consistency across KMeans seeds
- `agg NMI ↑`: agreement with AgglomerativeClustering
- `macro/frequency NMI ↓`: lower confounder risk
- `high-conf macro rate ↑`: cross-domain prototype support

### 推荐图片

![Layer comparison summary](../outputs/chronos_multilayer_validation/figures/layer_comparison_summary.png)

### 排版建议

- 图占页面 80% 宽度。
- 下方用一句中文 caption：`projection/layer_0 更像 local patch vocabulary；layer_6/layer_11 更稳定，但更容易吸收 domain/frequency/context-style information。`

### 口头讲法

> 这个图是回答老师“early layer 是否更好”的第一层证据。早层不是所有指标都最高，但它们的 confounder 更低，更接近 single-patch local vocabulary。中高层稳定性很高，但 macro/frequency NMI 上升，说明它们更可能混入 context-style 或 cadence-style 信息。

## Slide 6. What Do Early-Layer Clusters Look Like?

### 屏幕文字

**Projection and layer 0 preserve interpretable local patch neighborhoods.**

### 推荐图片

![Projection center nearest](../outputs/chronos_multilayer_validation/figures/projection_main_center_nearest.png)

![Layer 0 center nearest](../outputs/chronos_multilayer_validation/figures/layer_0_main_center_nearest.png)

### 排版建议

- 建议拆成两页：
  - Slide 6A：projection all clusters；
  - Slide 6B：layer_0 all clusters。
- 如果必须一页，左右各放一张，缩小 caption，不要叠太多文字。

### 口头讲法

> 这页是回到原空间看 cluster 到底长什么样。我们没有 cherry-pick，K=6 时 C0 到 C5 都展示。projection/layer_0 里有一些 cluster 的 nearest patches 在原空间可解释，说明 single patch information 仍然在 representation 里。

## Slide 7. What Happens After Context Mixing?

### 屏幕文字

**Layer 6 and layer 11 show stronger contextual mixing.**

### 推荐图片

![Layer 6 center nearest](../outputs/chronos_multilayer_validation/figures/layer_6_main_center_nearest.png)

![Layer 11 center nearest](../outputs/chronos_multilayer_validation/figures/layer_11_main_center_nearest.png)

### 排版建议

- 建议拆成两页：
  - Slide 7A：layer_6 K=6 all clusters；
  - Slide 7B：layer_11 K=6 all clusters。
- 每页只放一张主图 + 两条结论，不要再加 heatmap。

### 口头讲法

> layer_6/layer_11 的 cluster 不是完全没有 local shape，但很多 cluster 变成 weak diagnostic 或 confounded diagnostic。这支持老师的 intuition：middle/top layers 开始融合 context，不能再把 hidden cluster 简单解释为 single raw-shape motif。

## Slide 8. Layer 6 Needs a Finer Split

### 屏幕文字

**Layer 6 prefers a finer representation-space split.**

- shared setting: `K=6`
- layer-specific check: `K=10`
- interpretation: possible contextual subfamilies, not final taxonomy

### 推荐图片

![Layer 6 K10 center nearest](../outputs/chronos_multilayer_validation/figures/layer_6_k10_center_nearest.png)

### 排版建议

- 全页放图。
- 右上角放 `K=10 is a layer-specific split check`。

### 口头讲法

> 这页回答“为什么 layer_6 最好的 K 不是 6”。K=10 能把 layer_6 的部分结构拆细，但同时也出现更多 weak/confounded clusters，所以我们不能把它直接叫 taxonomy。

## Slide 9. Why Raw Euclidean Is Not Enough

### 屏幕文字

**Distance matters in original space.**

- Euclidean in representation space: candidate neighborhood discovery
- DTW in original space: shapelet-like prototype validation
- Raw Euclidean / correlation: diagnostic controls

### 推荐图片

![Distance metric heatmap](../outputs/distance_metric_ablation/figures/distance_metric_heatmap.png)

### 排版建议

- 图占页面 78-85%。
- 标题里直接写：`Two-space distance principle`。

### 口头讲法

> 我们不是要用 DTW 替代 representation-space Euclidean。正确分工是：模型空间用 Euclidean 看 Chronos 认为哪些 token 近；原空间用 DTW 看这些 token 是否是 time-shift / phase-shift 后仍然同形的 shapelet-like pattern。

## Slide 10. Prototype Selection Under Different Metrics

### 屏幕文字

**Same cluster, different original-space prototype metric.**

Columns:

- representation-center nearest
- raw Euclidean medoid
- correlation medoid
- DTW medoid

### 推荐图片

主讲优先选择两页：

![Layer 0 prototype comparison](../outputs/distance_metric_ablation/figures/prototype_metric_comparison_layer_0_k6.png)

![Layer 6 K10 prototype comparison](../outputs/distance_metric_ablation/figures/prototype_metric_comparison_layer_6_k10.png)

### 排版建议

- 不要把两张硬塞一页；每张做一页。
- 用红框或 PowerPoint 标注强调某一行，但不要隐藏其它 rows。

### 口头讲法

> 这页的重点不是说 DTW 永远最好，而是说明同一个 representation cluster 回到原空间后，prototype selection metric 会改变我们看到的 shape evidence。对 time-shifted spike/burst/oscillation 类 pattern，DTW 更合理。

## Slide 11. Controlled Retrieval and Failure Cases

### 屏幕文字

**DTW helps, but it is not a free lunch.**

- DTW can recover shifted local shapes.
- It can also over-warp unrelated patches.
- Failure cases are part of the evidence package.

### 推荐图片

![Retrieval comparison](../outputs/distance_metric_ablation/figures/retrieval_metric_comparison_examples.png)

![DTW failure cases](../outputs/distance_metric_ablation/figures/dtw_failure_cases.png)

### 排版建议

- 建议拆成两页：
  - Slide 11A：retrieval comparison；
  - Slide 11B：failure cases。
- failure case 页一定保留，老师通常会认可这种保守性。

### 口头讲法

> 如果 DTW 找到更同形的 patch，但 representation retrieval overlap 很低，说明 Chronos hidden representation 可能编码了 context family，而不是单一 raw-shape motif。相反，如果 DTW over-warping，也不能把它误读成真实 temporal concept。

## Slide 12. Final Answer This Week

### 屏幕文字

**Current answer**

1. Single patch information is still visible in `projection` and `layer_0`.
2. `layer_6` and `layer_11` are more stable but more contextualized/confounded.
3. Representation-space clusters need original-space DTW validation before motif naming.
4. Current evidence supports a discovery protocol, not a final motif taxonomy.

**Next step**

Multi-seed resampling stability + DTW-aware controlled retrieval for candidate clusters

### 排版建议

- 用四个 conclusion blocks。
- 最后一行写 `Claim boundary`：`candidate motif/prototype families only after DTW + confounder audit.`

### 口头讲法

> 我觉得这周最稳的结论不是“发现了某几类 motif”，而是方法论上清楚了：Chronos-2 representation space 的 cluster 可以作为 candidate generator，但最终是否是 motif/prototype family，需要 DTW-aware original-space validation 和 confounder audit。

## 2. Backup Slides

### Backup A. Metric Definitions

| metric | meaning | direction | how to explain |
|---|---|---|---|
| `silhouette ↑` | cluster 内紧密、cluster 间分离 | 越高越好 | 不能单独选 K |
| `stability ↑` | 不同 KMeans seeds 的 NMI | 越高越好 | 稳定不等于语义正确 |
| `agg NMI ↑` | KMeans 与 Agglomerative labels 一致性 | 越高越好 | 检查算法依赖 |
| `macro NMI ↓` | cluster 与 macro-domain 的 mutual information | 越低越好 | 高值提示 domain confounding |
| `frequency NMI ↓` | cluster 与 cadence/frequency 的 mutual information | 越低越好 | 高值提示 frequency confounding |
| `high-conf macro rate ↑` | cluster 在多个 macro-domain 的可信 match | 越高越好 | 检查跨域复现性 |
| `DTW ratio ↓` | intra-cluster DTW / random baseline DTW | 越低越好 | 原空间 shape coherence gate |
| `DTW gain ↑` | DTW 相对 raw Euclidean 的改善 | 越高越好 | 说明 warping-aware metric 更合适 |

### Backup B. Macro-Domain Evidence

推荐图片：

![Projection macro-domain filtered](../outputs/chronos_multilayer_validation/figures/projection_main_macro_domain_filtered.png)

![Layer 0 macro-domain filtered](../outputs/chronos_multilayer_validation/figures/layer_0_main_macro_domain_filtered.png)

![Layer 6 macro-domain filtered](../outputs/chronos_multilayer_validation/figures/layer_6_main_macro_domain_filtered.png)

![Layer 11 macro-domain filtered](../outputs/chronos_multilayer_validation/figures/layer_11_main_macro_domain_filtered.png)

口头讲法：

> macro-domain view 不是强行每格都展示 nearest，而是 confidence-filtered。空白或 weak cell 本身就是 evidence，说明该 cluster 可能不是 cross-domain motif family。

### Backup C. Why Prior-Guided Motif Probe Is Not Main Evidence

屏幕文字：

- prior-guided labels are weak probes, not ground truth
- deterministic rules are sensitive to thresholds and mixed patches
- current paper-level claim should rely on model-derived clusters + DTW validation

口头讲法：

> prior-guided motif taxonomy 仍然是我们圈子能理解的语言，但这版 PPT 不把它放主证据链。它可以作为 appendix sanity check，不能用来命名主 cluster。

### Backup D. What If Teacher Asks Why Not Cluster With DTW Directly?

推荐回答：

> 我们的问题分成两个空间。representation space 中没有时间轴，Chronos hidden vector 的几何关系自然是 Euclidean / cosine 这类向量空间距离；DTW 只适用于 original time-series space。直接用 DTW 聚 raw patches 会回答“原始形状有哪些类”，但不能回答“Chronos token representation 学到了什么”。所以我们保留 Euclidean for representation discovery，使用 DTW for original-space validation。

### Backup E. What If Teacher Asks Whether Stability Is Enough?

推荐回答：

> 现在 representation-space clustering 的 seed stability 很高，例如 shared K 下 projection/layer_0/layer_6/layer_11 的 seed NMI 大约在 0.97 到 0.99。但这只是 initialization stability。真正 paper-level 还需要 data resampling stability：换一批 sampled windows 后，同一类 prototype family 是否还能出现；DTW-benefited clusters 是否仍然 benefited；confounder risk 是否稳定。

## 3. 本周 PPT 中建议主放的图片

| slide | figure | role |
|---|---|---|
| Slide 4 | `../outputs/chronos_multilayer_validation/figures/k_selection_summary.png` | 解释 K 选择 |
| Slide 5 | `../outputs/chronos_multilayer_validation/figures/layer_comparison_summary.png` | 解释层间变化 |
| Slide 6 | `../outputs/chronos_multilayer_validation/figures/projection_main_center_nearest.png` | early/local evidence |
| Slide 6 | `../outputs/chronos_multilayer_validation/figures/layer_0_main_center_nearest.png` | early/local evidence |
| Slide 7 | `../outputs/chronos_multilayer_validation/figures/layer_6_main_center_nearest.png` | contextual mixing evidence |
| Slide 7 | `../outputs/chronos_multilayer_validation/figures/layer_11_main_center_nearest.png` | top-layer evidence |
| Slide 8 | `../outputs/chronos_multilayer_validation/figures/layer_6_k10_center_nearest.png` | layer-specific split |
| Slide 9 | `../outputs/distance_metric_ablation/figures/distance_metric_heatmap.png` | distance audit |
| Slide 10 | `../outputs/distance_metric_ablation/figures/prototype_metric_comparison_layer_0_k6.png` | metric changes prototypes |
| Slide 10 | `../outputs/distance_metric_ablation/figures/prototype_metric_comparison_layer_6_k10.png` | metric changes prototypes |
| Slide 11 | `../outputs/distance_metric_ablation/figures/retrieval_metric_comparison_examples.png` | controlled retrieval |
| Slide 11 | `../outputs/distance_metric_ablation/figures/dtw_failure_cases.png` | conservative failure cases |

## 4. 老师可能会问的问题

### Q1. 为什么这周只用 Chronos-2？

答：这是按 meeting 后老师建议收敛问题。TimesFM 和 Chronos-small 已经证明 broad setting 有结构也有 artifact；本周更重要的是把 mechanism question 做扎实：single patch information 是否保留、不同层如何改变 representation、cluster 是否能回到原空间解释。

### Q2. 早层是不是一定更好？

答：不绝对。早层更适合看 local patch vocabulary，因为 confounder 较低，原空间 nearest 更直接；但 middle/top layers 更稳定，也可能编码 contextualized temporal commonalities。区别是：早层适合 local motif inspection，中高层需要更严格的 DTW validation 和 confounder audit。

### Q3. K=6 和 layer_6 K=10 到底哪个对？

答：它们回答不同问题。`K=6` 是 shared operating point，用于 layer-wise comparison；`layer_6 K=10` 是 layer-specific split check，说明 layer_6 可能有更细 substructure。两者都不是 final taxonomy K。

### Q4. 为什么不能直接用 prior-guided motif labels？

答：prior-guided motif taxonomy 是有用语言，但当前 deterministic labels 对 mixed/uncertain patches 和阈值敏感；它更适合作为 sanity check，不适合 paper-level ground truth。本周主证据改用 model-derived clusters + original-space DTW validation。

### Q5. 现在能不能说 Chronos 学到了时序语言？

答：可以把 `temporal language` 作为研究问题和叙事框架，但不能说已经完整发现。更稳妥的说法是：我们看到 Chronos-2 patch-token space 中存在可审计的 candidate motif/prototype families，并建立了从 representation neighborhoods 到 original-space validation 的 discovery protocol。

### Q6. 结果稳定性够吗？

答：初始化稳定性够高，但 data resampling stability 还没完成。因此当前是 pilot-level evidence；下一步需要多 seed sampling，检查 cluster matching、prototype stability 和 DTW validation stability。

## 5. 结尾建议

最后一页建议明确把下一步写成：

1. Run multi-seed data resampling stability.
2. For stable clusters, run DTW-aware controlled retrieval.
3. Separate local patch vocabulary from contextualized prototype family.
4. Only then propose `model-derived motif taxonomy v1 pilot`.

这样收束会比较符合老师的关注点：不是为了画漂亮图，而是把 `TSFM shared temporal knowledge` 的证据链一步步做扎实。

