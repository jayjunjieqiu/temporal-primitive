# Yuxuan Liang 语境下的项目叙事与术语规则

## 1. 目标

这份规则用于统一本仓库所有 proposal、poster、report 和 figure caption 的写法。当前项目应讲成一个 **TSFM representation interpretability / model-derived motif taxonomy discovery** 问题。

关键修正：`motif taxonomy`、`temporal primitives`、`时序语言` 都可以使用，而且是我们圈子里能读懂的语言；但要采用双层叙事，避免把 human-prior taxonomy 当作 ground truth。

## 2. 主叙事

推荐一句话：

> 本项目研究 patch-based TSFMs 是否在 heterogeneous cross-domain time series 中学习到可迁移的 patch-level temporal primitives，并通过 representation clustering、controlled retrieval、confounder audit 和 original-space inspection，从 human-prior motif taxonomy v0 走向 model-derived motif taxonomy v1。

英文短句：

> We diagnose what patch tokens in TSFMs learn by discovering and auditing model-derived motif/prototype families in the patch-token representation space.

## 3. 双层 Motif Taxonomy

- `motif taxonomy v0`：human-prior / shapelet-inspired probe taxonomy。它来自传统 Time Series Data Mining 语言，例如 trend、oscillation、spike、burst、level shift、intermittent、flat、mixed。
- `model-derived motif taxonomy v1`：从 TSFM representation clusters 出发，经 original-space inspection、controlled retrieval、domain/frequency/position confounder audit 后得到的候选 taxonomy。

写作时要明确：v0 是 probe，不是 ground truth；v1 是 pilot，不是 final taxonomy。

## 4. 术语表

| 推荐术语 | 使用方式 |
| --- | --- |
| `motif taxonomy v0` | human-prior probe taxonomy，用于解释和校准 |
| `model-derived motif taxonomy v1` | 我们的候选发现结果 |
| `patch-level temporal primitives` | 描述 TSFM patch token 可能编码的局部时序原语 |
| `motif prototype / prototype patch` | 回到原空间解释 cluster 形态 |
| `shapelet-like local pattern` | 连接传统 TSDM / ShapeX / shapelet 语言 |
| `candidate motif family` | 经初步审计的 cluster |
| `model-derived motif cluster` | hidden-space 中的候选聚类 |
| `temporal language / 时序语言` | 可作副标题或引入句，连接 TSFM token space 与 motif taxonomy |
| `confounder audit` | domain/frequency/position/scale/raw-statistics 审计 |
| `position confounding score` | 给 patch-index NMI 的解释性名称 |

避免用法：

- 不要把 `taxonomy-v0` 写成 ground truth。
- 不要把 KMeans cluster 直接写成 motif。
- 不要把 `model-derived motif taxonomy v1` 写成 final taxonomy。

## 5. 推荐叙事结构

1. **Background：TSFM/STFM 的目标是跨域泛化。**  
   从 `cross-domain time series learning`、`data heterogeneity`、`shared representations`、`OOD generalization` 进入，同时指出 motif taxonomy 是解释 shared representations 的自然语言。

2. **Gap：性能有效，但 token-level mechanism 不清楚。**  
   当前 TSFM 文献强调 architecture、pre-training、adaptation、scaling，但 patch token 内部学到的 motif/prototype structure 缺乏系统诊断。

3. **Question：什么是 TSFM 的时序语言？**  
   问题写成：raw patch、tokenizer/projection、hidden state 分别编码什么？它们是 domain identity、frequency、position，还是 cross-domain motif primitives？

4. **Method：discover-first, name-second。**  
   先从真实多域数据抽取 patch，提取 TSFM representation，做 clustering/retrieval，再通过 original-space inspection 和 confounder audit 命名 candidate motif/prototype families。

5. **Evidence：从 local vocabulary 到 contextualized motif family。**  
   用 `raw -> tokenizer/projection -> hidden` lineage 讲清楚：早层更像 local patch vocabulary，hidden 层可能把多个 local patterns 重组为 contextualized motif families。

6. **Risk Control：必须展示负例。**  
   TimesFM `first-patch artifact` 这类结果要主动展示，说明我们不会把 position artifact 误读成 motif family。

7. **Contribution：motif taxonomy discovery protocol。**  
   当前贡献应写成 `model-derived motif taxonomy discovery protocol`、`motif taxonomy v1 pilot`、跨模型/跨层诊断结果。

## 5.1 Distance Principle

本项目采用 **two-space distance principle**：

- 在 `TSFM representation space` 中，用 Euclidean geometry / KMeans / nearest-neighbor 分析模型内部认为哪些 `patch tokens` 相近。这里的目标是发现 `representation neighborhoods`，不是直接定义 motif。
- 在 `original time-series space` 中，用 DTW-aware geometry 验证这些 neighborhoods 是否对应可解释的 `shapelet-like local patterns`。这里的目标是检查 time shift、phase shift、local warping 后仍是否保持 prototype coherence。
- 因此，`representation-space Euclidean clustering` 是 candidate generator；`original-space DTW validation` 是 motif/prototype family 命名的必要 gate。
- raw Euclidean 和 correlation 可以作为 diagnostic controls，但不应替代 DTW 来判断 spike、burst、oscillation 等局部错位敏感 pattern 的原空间相似性。

推荐句式：

> We use Euclidean geometry to discover neighborhoods in Chronos patch-token representation space, and DTW geometry to validate whether those neighborhoods correspond to coherent temporal shapes in original time-series space.

## 6. 和老师已有工作的连接

### FM4TS / TSFM Survey

- 本项目补足 methodology-centric TSFM taxonomy 之外的 representation-level diagnosis。
- 现有 survey 整理 architecture、pre-training、adaptation、data modality；我们进一步问这些设计在 patch-token space 中形成了什么 motif/prototype structure。

### UniTime / Cross-Domain Learning

- UniTime 强调 cross-domain learning 中既有 temporal commonality，也有 domain confusion。
- 我们的 cluster/retrieval 分析可以解释为 TSFM token space 的 domain confusion audit：好的 motif family 应该保留 temporal commonality，同时不过度泄漏 domain identity、frequency 或 patch position。

### Time-LLM / Reprogramming / Prototype

- Time-LLM 把 time series patch reprogram 到 text prototype representations，说明 patch 可以通过 prototype space 与 foundation model 知识对齐。
- 我们做相反方向的解释：观察 frozen TSFM 内部是否自发形成 patch-level motif prototype families。

### Time-FFM / Heterogeneity

- Time-FFM 强调 data heterogeneity 下的 global encoder 和 local prediction heads。
- 我们检查 frozen TSFM 的 global token representation 是否真的捕获跨域 motif knowledge，还是主要编码 domain-specific heterogeneity。

### Scaling Laws

- `Chronos-2-small -> Chronos-2` 是模型规模/能力变化下 motif organization 的 scaling diagnosis。
- `TimesFM-2.5` 是 architecture contrast，用来判断发现是否是 Chronos-specific。

### ShapeX / Shapelet-Driven Explanation

- 解释 patch 原空间形态时，优先用 `shapelet-like segment`、`motif prototype`、`prototype patch`、`subsequence-level explanation`。
- `motif taxonomy v0` 可被称为 `shapelet/prototype-inspired probe taxonomy`。

## 7. 结果解释规则

一个 cluster 只有同时满足下面条件，才可以作为 `candidate motif/prototype family`，进入 `model-derived motif taxonomy v1 pilot`：

- original-space medoid / nearest neighbors 形态可解释；
- DTW-aware controlled retrieval 后仍能保持相似形态；
- 不被单一 domain、frequency 或 patch position 主导；
- external weak motif labels / motif taxonomy v0 只能作为 appendix-level sanity check，不进入 paper-level 主证据链；
- lineage 上能看到 hidden 层相对 raw/tokenizer/projection 的重组；
- 尽可能通过 cross-model sanity check。

推荐句式：

> 该 cluster 可以作为候选 motif/prototype family，因为它不是单一 local patch vocabulary 的延续，而是在 hidden 层汇聚多个 source clusters，并且在 controlled retrieval 和原空间检查中保持形态一致。

负例写法：

> 形态上看似一致，但主要由 position/domain/frequency confounding 驱动，不能进入 motif taxonomy v1。

## 8. 图表语言

图内文字采用中英混合：

- 关键技术词保留英文：`motif taxonomy v0`、`model-derived motif taxonomy v1`、`prototype family`、`confounder audit`、`controlled retrieval`。
- 解释性短语可以中文：`人类先验`、`模型内生`、`负例控制`、`回到原空间`。
- 如果要生成 PNG，优先使用 Noto CJK 字体，避免中文渲染丢失。

## 9. 标题建议

推荐标题：

- `Discovering Model-Derived Motif Taxonomy in Time Series Foundation Models`
- `What Is the Temporal Language of TSFMs? From Human-Prior Motif Taxonomy to Model-Derived Taxonomy`
- `From Local Patch Vocabulary to Model-Derived Motif Taxonomy in TSFMs`

中文标题：

- `什么是 TSFM 的时序语言？从 Human-Prior Motif Taxonomy 到 Model-Derived Taxonomy`
- `面向时间序列基础模型的 Model-Derived Motif Taxonomy Discovery`
- `从局部 Patch Vocabulary 到模型内生 Motif Taxonomy`

## 10. 推荐贡献表述

本项目当前阶段的贡献应写成：

1. 提出一个面向 patch-based TSFMs 的 `model-derived motif taxonomy discovery protocol`。
2. 从 heterogeneous cross-domain datasets 中构建 patch bank，分析 raw patch、tokenizer/projection、hidden states 的 motif organization。
3. 通过 clustering、controlled retrieval、lineage tracing 和 confounder audit，区分 motif/prototype families 与 domain/frequency/position artifacts。
4. 形成一个 `model-derived motif taxonomy v1 pilot`，为后续 TSFM mechanism study、cross-domain generalization diagnosis 和 shapelet/prototype-based explanation 提供基础。

## 11. 参考来源

- Yuxuan Liang homepage: https://yuxuanliang.com/
- Yuxuan Liang Research page: https://yuxuanliang.com/research/
- CityMind Lab publications: https://citymind.top/publications/
- `Foundation Models for Time Series Analysis: A Tutorial and Survey`: https://arxiv.org/abs/2403.14735
- `Position Paper: What Can Large Language Models Tell Us about Time Series Analysis`: https://arxiv.org/abs/2402.02713
- `UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting`: https://arxiv.org/abs/2310.09751
- `Time-LLM: Time Series Forecasting by Reprogramming Large Language Models`: https://openreview.net/forum?id=Unb5CVPtae
- `Time-FFM: Towards LM-Empowered Federated Foundation Model for Time Series Forecasting`: https://arxiv.org/abs/2405.14252
- `Foundation Models for Spatio-Temporal Data Science: A Tutorial and Survey`: https://arxiv.org/abs/2503.13502
- `Towards Neural Scaling Laws for Time Series Foundation Models`: see CityMind Lab publication list, ICLR 2025.
- `ShapeX: Shapelet-Driven Post Hoc Explanations for Time Series Classification Models`: see CityMind Lab publication list, NeurIPS 2025.
