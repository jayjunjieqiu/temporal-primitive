# 研究提案：TSFM 的 Model-Derived Motif Taxonomy Discovery

## 题目

**What Is the Temporal Language of TSFMs? From Human-Prior Motif Taxonomy to Model-Derived Taxonomy**

中文工作标题：**什么是 TSFM 的时序语言？从 Human-Prior Motif Taxonomy 到 Model-Derived Taxonomy**

## 1. 背景与动机

Yuxuan Liang 老师团队长期关注 Spatio-Temporal Data Mining、Time Series Analysis、Urban Computing 和 Foundation Models。相关工作中反复出现的核心问题是：在交通、能源、气象、环境、金融等 heterogeneous time series domains 之间，模型如何学习可迁移的 shared representations，并在 cross-domain / OOD 场景中保持 generalization。

Chronos-2、Chronos-2-small 和 TimesFM-2.5 这类 patch-based TSFMs 会先把时间序列切成 patch token，再通过 foundation model 做预测。它们在 forecasting 上有效，但 token-level mechanism 仍然不清楚：

> TSFM 的 patch token representation 到底编码了什么？是局部数值形状、domain identity、frequency/position shortcut，还是一套可跨域复用的 patch-level temporal primitives？

本项目把这个问题讲成一个 **model-derived motif taxonomy discovery** 问题：我们先用传统 Time Series Data Mining 语言构造 `motif taxonomy v0` 作为 human-prior probe，再从 TSFM representation space 出发，通过 clustering、controlled retrieval、confounder audit 和 original-space inspection，归纳候选 `model-derived motif taxonomy v1`。

## 2. 核心研究问题

1. **Temporal language**：patch-based TSFMs 是否形成了一套可解释的“时序语言”，即 patch-level temporal primitives / motif prototypes？
2. **v0 vs v1**：human-prior motif taxonomy v0 与 TSFM hidden-space 中自然形成的 model-derived motif taxonomy v1 是一一对应、部分重叠，还是明显错位？
3. **Representation lineage**：raw patch、tokenizer/projection embedding 和 transformer hidden state 分别更像 local patch vocabulary，还是 contextualized motif family？
4. **Confounder vs motif**：embedding clusters 中的结构是由 motif commonality 驱动，还是主要来自 domain identity、frequency、scale、raw statistics 或 patch position？
5. **Scaling / architecture effect**：`Chronos-2-small -> Chronos-2` 的规模变化，以及 `Chronos` vs `TimesFM` 的架构差异，会如何改变 motif organization？

## 3. 假设

### H1. TSFMs 学到可跨域复用的 patch-level temporal primitives

如果 TSFM 真的具备 cross-domain generalization 能力，那么它的 patch-token space 中应存在跨数据域复用的 motif/prototype families。它们可能表现为 trend、oscillation、spike、burst、level shift、intermittent、transition-like patterns 等，但不一定与 human-prior taxonomy v0 一一对应。

### H2. Hidden layers 会把 local patch vocabulary 重组为 contextualized motif families

tokenizer/projection 层更可能保留局部形状和数值模式，即 local patch vocabulary；更深的 hidden state 可能结合上下文，把多个 local vocabulary buckets 聚合成更抽象的 contextualized motif families。这个假设必须通过 `raw -> tokenizer/projection -> hidden` lineage tracing 验证。

### H3. Motif organization 受 scale、architecture 和 confounders 共同影响

`Chronos-2-small`、`Chronos-2` 和 `TimesFM-2.5` 可能形成不同的 motif organization。更大的模型不一定简单地产生“更纯”的 motif taxonomy；它也可能更强地编码 domain/frequency/context。我们需要同时报告 motif evidence 和 confounder evidence。

## 4. 模型与数据

### 4.1 模型

- `Chronos-2-small`：作为 Chronos family 中更小/更早的 scale baseline，用于观察 motif organization 的 scaling behavior。
- `Chronos-2`：作为更强的 Chronos TSFM，用于检查 patch-level temporal primitives 是否更稳定。
- `TimesFM-2.5`：作为 architecture contrast，用于判断发现是否是 Chronos-specific。

### 4.2 数据

主要使用 `/data/junjieqiu/datasets/basicts_datasets` 中除 `BLAST` 外的数据集，覆盖 traffic flow/speed、electricity、weather、air quality、exchange rate、illness、simulated series 等多类 domains。

当前 second pilot 设置：

- 22 个 non-BLAST datasets；
- 每个数据集抽取 100 个 128-step windows；
- 用 domain-balanced sampling 构建 patch bank；
- 不 fine-tune、不修改权重、不跑下游迁移实验。

## 5. 方法路线：Discover First, Name Second

本项目不再从人工 motif taxonomy 出发单向验证模型，而是采用双层 taxonomy 路线：

1. 构造 `motif taxonomy v0`：human-prior / shapelet-inspired probe taxonomy。
2. 从真实多域数据中抽取 patch bank。
3. 提取 `raw patch`、`tokenizer/projection`、selected `hidden states`。
4. 在 representation space 中做 PCA/KMeans、nearest-neighbor retrieval 和 stability check，筛选 candidate motif clusters。
5. 对每个 cluster 做 original-space inspection：查看 medoid patches、nearest neighbors、full context。
6. 对每个 cluster 做 confounder audit：检查 dataset/domain/frequency/patch-index/raw statistics。
7. 对候选 cluster 做 controlled retrieval：cross-domain、same patch-index、same frequency、cross-model 等条件。
8. 只有通过审计的 cluster，才进入 `model-derived motif taxonomy v1 pilot`。

## 6. Motif Taxonomy v0：Human-Prior Probe

`motif taxonomy v0` 的作用不是提供 ground truth，而是提供弱语义锚点，让我们可以解释和校准原空间 patch shape。

建议 v0 类别：

- `trend`
- `oscillation`
- `impulse_spike`
- `burst_event`
- `level_shift`
- `volatility_shift`
- `intermittent`
- `flat_low_information`
- `mixed_uncertain`

这些类别连接传统 Time Series Data Mining 中的 motif discovery、shapelet、Matrix Profile、prototype-based explanation。它们能帮助我们问：模型 cluster 是否复刻了 human-prior motif taxonomy，还是形成了更上下文化的 motif families？

## 7. 具体分析模块

### 7.1 Representation extraction

对每个 patch 提取：

- `raw_z_patch`：z-normalized 原始 patch，作为 shapelet-like raw baseline。
- `tokenizer/projection embedding`：进入 transformer 前的 local patch vocabulary。
- selected `hidden states`：contextualized token representation。

重点比较：

- `raw -> tokenizer/projection`：模型如何编码局部形状；
- `tokenizer/projection -> hidden`：transformer 是否进行了 contextual reorganization；
- `hidden -> motif family`：哪些 cluster 能被解释为 model-derived motif/prototype families。

### 7.2 Representation clustering

我们使用 `StandardScaler -> PCA(max 30 dims) -> KMeans` 做 exploratory clustering。K 值用于控制探索粒度，不直接等于最终 taxonomy 类别数。

聚类结果必须和以下审计指标一起报告：

- silhouette / clustering stability；
- NMI with dataset/domain/frequency/patch_index；
- NMI with motif taxonomy v0 labels；
- cluster size 和 domain coverage；
- original-space medoid / prototype shape。

### 7.3 Controlled retrieval

给定 query patch 或 prototype patch，在 embedding bank 中检索 nearest neighbors，并设置控制条件：

- unrestricted retrieval；
- same patch-index only；
- same frequency only；
- cross-domain only；
- cross-model retrieval；
- matched random baseline。

输出包括：

- top-k shape similarity；
- domain/frequency diversity；
- motif taxonomy v0 agreement as weak probe；
- visual coherence；
- failure cases。

### 7.4 Representation lineage cards

对关键 cluster 追踪 `raw -> tokenizer/projection -> hidden` 的路径，回答：

- hidden cluster 是否只是单一 raw/tokenizer cluster 的延续？
- 是否由多个 local patch vocabulary buckets 汇入？
- 汇入后的 original-space patch 是否仍有一致 motif prototype？
- 是否存在 position/domain/frequency artifact？

TimesFM `c8/c5/c4` 和 Chronos `c6/c1` 的 lineage cards 是当前最重要的证据形式。

## 8. 当前初步发现

### 8.1 TSFM representation space 确实存在 motif-level 结构

Second pilot 显示，多域真实 patch 在 selected hidden layers 中形成可复现结构。但这些结构不是 human-prior motif taxonomy v0 的简单复刻，而是同时混合 motif commonality、domain/frequency information 和 position/context effects。

### 8.2 TimesFM-2.5 layer_10 给出最清楚的正负例

- `c8`：rising / recovery motif family candidate。它不是单一 tokenizer cluster 延续，而是由多个 source clusters 汇入，且原空间 medoid 形态较一致。
- `c5`：falling / smooth transition motif pool。形态清楚，但内部仍需 split。
- `c4`：first-patch artifact negative control。形态上看似一致，但 `patch_index=0` 占 100%，因此不能进入 motif taxonomy v1。

这个负例很关键：它说明我们的方法不是“看到聚类就命名 motif”，而是会识别 position confounding。

### 8.3 Chronos-2 与 TimesFM 呈现不同的 confounder profile

Chronos-2 hidden layer 的 patch-index confounding 较弱，但 domain/frequency encoding 更强。这说明模型间差异本身就是研究对象：TSFM 的 motif taxonomy 不是脱离 architecture 和 pre-training recipe 存在的。

### 8.4 tokenizer/projection 与 hidden state 承担不同角色

TimesFM tokenizer 的 patch-index NMI 很低，更像 local patch vocabulary；TimesFM hidden layer 的 stability 更高，但 patch-index NMI 明显上升。Chronos-2 hidden layer patch-index NMI 仍低，但 domain/frequency NMI 上升。由此可见，做 clustering 时不能简单说“哪一层最好”，而要区分 local vocabulary analysis 和 contextual motif taxonomy discovery。

## 9. 评估协议

我们把 cluster 解释为 candidate motif/prototype family 的最低标准设为：

1. original-space medoid / nearest neighbors 有可解释形态；
2. controlled retrieval 后仍保持形态一致；
3. 不被单一 domain、frequency 或 patch position 主导；
4. motif taxonomy v0 作为 weak probe 能提供解释，但不要求 label purity；
5. lineage 上能看到 hidden 层相对 raw/tokenizer/projection 的重组；
6. 尽可能通过 cross-model sanity check。

如果 cluster 视觉上清楚但由 position/domain/frequency 解释，则标记为 artifact / negative control。

## 10. 预期贡献

本项目当前阶段的贡献应写成：

- 一个面向 patch-based TSFMs 的 `model-derived motif taxonomy discovery protocol`；
- 一个 heterogeneous cross-domain patch bank 和 frozen TSFM representation extraction pipeline；
- 一套区分 motif/prototype family 与 domain/frequency/position artifact 的 controlled validation framework；
- 一组从模型内生 cluster 出发的 candidate motif taxonomy v1 evidence；
- 对 TSFM cross-domain generalization mechanism 的经验诊断：local patch vocabulary 如何在 hidden layers 中被上下文化重组为 motif families。

## 11. 范围控制

第一阶段不做：

- fine-tuning；
- full downstream transfer-performance comparison；
- prompt/covariate engineering；
- 大规模数据下载；
- 把 motif taxonomy v1 宣称为 final taxonomy。

第一阶段只做 frozen representation analysis、controlled retrieval、cluster interpretation 和 confounder audit。

## 12. 下一步里程碑

1. 用当前写作规则重写 poster 和主要 reports，让叙事对齐 `motif taxonomy v0 -> model-derived motif taxonomy v1`。
2. 构建 `model-derived motif taxonomy v1 evidence table`：以 TimesFM `c8/c5` 为正例、`c4` 为负例，整合 lineage、retrieval、cross-model validation。
3. 围绕 `falling_transition` motif family 做更平衡的 prototype bank、merge test、direction-flip control。
4. 补做 Chronos-native motif discovery，避免 TimesFM-derived taxonomy 直接迁移到 Chronos。
5. 在更大采样规模上验证 candidate motif families 的稳定性。

## 13. 相关参考

- Yuxuan Liang homepage: https://yuxuanliang.com/
- Yuxuan Liang research page: https://yuxuanliang.com/research/
- CityMind Lab publications: https://citymind.top/publications/
- Chronos: https://arxiv.org/abs/2403.07815
- Chronos-2: https://arxiv.org/abs/2510.15821
- Chronos repository: https://github.com/amazon-science/chronos-forecasting
- TimesFM: https://arxiv.org/abs/2310.10688
- TimesFM repository: https://github.com/google-research/timesfm
- Foundation Models for Time Series Analysis: https://arxiv.org/abs/2403.14735
- Position Paper: What Can Large Language Models Tell Us about Time Series Analysis: https://arxiv.org/abs/2402.02713
- UniTime: https://arxiv.org/abs/2310.09751
- Time-LLM: https://openreview.net/forum?id=Unb5CVPtae
- Time-FFM: https://arxiv.org/abs/2405.14252
- Foundation Models for Spatio-Temporal Data Science: https://arxiv.org/abs/2503.13502
- Matrix Profile XXII: https://arxiv.org/abs/2009.07907
- Motiflets: https://arxiv.org/abs/2206.03735
- HIME: https://arxiv.org/abs/1802.04883
- A Framework for Guided Time Series Motif Discovery: https://www.kdd.org/kdd2017/papers/view/a-framework-for-guided-time-series-motif-discovery
