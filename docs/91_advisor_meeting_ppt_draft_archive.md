# PPT 草稿：TSFM 的 Model-Derived Motif Taxonomy Discovery

> 用途：明天给 Yuxuan Liang 老师汇报。  
> 建议时长：15-20 分钟。  
> 主线：**不是“我们已经定义好了 taxonomy”，而是“我们用 TSFM representation 反向发现 motif taxonomy，并审计哪些是真 motif、哪些是 artifact”。**

## 0. 汇报总策略

### 一句话版本

> 我们想回答：patch-based TSFMs 的 patch token 到底学到了什么时序语言？初步结果显示，hidden space 不是简单复刻 human-prior motif taxonomy v0，而是形成了一些 model-derived motif families；其中 `falling_transition` 最稳，但 position/domain/frequency confounding 必须显式审计。

### 叙述顺序

1. 先接老师熟悉的问题：TSFM / STFM 的 cross-domain generalization 需要 shared temporal knowledge。
2. 再引出缺口：forecasting 有效，但 patch token 学到了什么还不清楚。
3. 提出双层 taxonomy：v0 是 human-prior motif taxonomy，v1 是 model-derived motif taxonomy。
4. 展示方法：discover-first, name-second。
5. 先展示最直观的图：embedding clusters + 原空间 patch shape。
6. 再展示严谨性：lineage tracing、confounder audit、negative control。
7. 最后展示跨模型验证：TimesFM-derived motif 能不能转移到 Chronos。
8. 收束到下一步：domain-balanced prototype bank + Chronos-native discovery。

### 不要主动过度 claim

- 不要说：我们发现了最终 motif taxonomy。
- 要说：我们得到 `model-derived motif taxonomy v1 pilot`，并建立了一套审计 protocol。
- 不要说：cluster 就是 motif。
- 要说：cluster 只有通过 original-space inspection、controlled retrieval、confounder audit 后才进入 candidate motif family。

---

# Slide 1. Title

## 屏幕文字

**What Is the Temporal Language of TSFMs?**  
**From Human-Prior Motif Taxonomy to Model-Derived Taxonomy**

Patch-based TSFMs: `Chronos-2-small`, `Chronos-2`, `TimesFM-2.5`

## 版式建议

- 左侧：大标题，占 60%。
- 右侧：一个小流程图：`raw patch -> patch token -> hidden state -> motif taxonomy v1`。
- 不放复杂图。

## 口头讲法

> 我这次想讲的不是 forecasting performance，而是 TSFM 的内部表征：patch token 到底学到了什么。我们先用 human-prior motif taxonomy v0 做 probe，再从模型 representation 里反向发现 model-derived motif taxonomy v1。

---

# Slide 2. Why This Matters to TSFM / STFM

## 屏幕文字

**TSFMs aim to learn shared temporal knowledge across heterogeneous domains.**

- Cross-domain generalization is central to TSFM / STFM.
- But token-level mechanism is under-explored.
- Patch token is the natural unit for diagnosis.

## 版式建议

- 三个横向 block：
  - `heterogeneous domains`
  - `patch tokens`
  - `shared temporal knowledge`
- 右下角放一句：`Question: what do patch tokens learn?`

## 口头讲法

> 老师之前很多工作都关心 cross-domain、heterogeneity、shared representation。TSFM 的 patch token 是一个很自然的切入口，因为所有数值序列最终都要变成 token 进入 foundation model。

---

# Slide 3. Core Framing: v0 -> v1

## 屏幕文字

**Two-layer motif taxonomy framing**

- `motif taxonomy v0`: human-prior / shapelet-inspired probe
- `model-derived motif taxonomy v1`: discovered from TSFM representation space
- Goal: discover which motifs are real, contextualized, and transferable

## 推荐图片


## 版式建议

- 左侧放上图。
- 右侧放 v0/v1 对比表：

| | v0 | v1 |
|---|---|---|
| source | human prior / TSDM | TSFM hidden clusters |
| role | weak probe | candidate discovery |
| status | not ground truth | not final taxonomy |

## 口头讲法

> 这里我们恢复 motif taxonomy 语言。v0 是我们熟悉的 trend、spike、shift 等 human-prior taxonomy；但它不是 ground truth。真正想要的是从 TSFM hidden space 里反过来发现 v1。

---

# Slide 4. Method: Discover First, Name Second

## 屏幕文字

**A cluster is not a motif until it survives audits.**

Pipeline:

1. Build cross-domain patch bank
2. Extract raw / projection / hidden representations
3. PCA + KMeans as candidate generator
4. Inspect original-space prototypes
5. Audit domain / frequency / patch-index confounders
6. Controlled retrieval + cross-model validation

## 版式建议

- 用横向流程图。
- 每个步骤最多 3-5 个词。
- 页脚加一句红色提示：`KMeans label != final motif`

## 口头讲法

> 关键是 discover-first, name-second。KMeans 只是给候选，不给结论。真正命名 motif family 需要回到原空间看 patch 长什么样，还要排除 domain、frequency、position confounding。

---

# Slide 5. Data and Models

## 屏幕文字

**Pilot setting**

- Data: 22 BasicTS non-BLAST datasets
- Sampling: 100 windows per dataset, 128-step context
- Models: `Chronos-2-small`, `Chronos-2`, `TimesFM-2.5`
- No fine-tuning, no weight modification

## 版式建议

- 左侧：数据设置。
- 右侧：模型设置。
- 底部：`frozen representation analysis only`

## 口头讲法

> 这是一个 representation analysis，不是训练实验。我们先控制规模，避免把问题做成 full benchmark。BLAST 太大，先排除。

---

# Slide 6. First Evidence: Hidden Space Has Structure

## 屏幕文字

**TimesFM layer_10 has stable structure, but not a copy of v0.**

- KMeans clusters show clear neighborhoods.
- Motif taxonomy v0 colors do not align one-to-one.
- Patch-index coloring reveals position confounding.

## 推荐图片


## 版式建议

- 全页放图，标题和 3 个 bullet 放在上方。
- 讲的时候按 A/B/C 三列依次读。

## 口头讲法

> 左边说明 hidden space 确实有结构。中间说明这个结构不是 v0 taxonomy 的简单复刻。右边说明我们必须小心，TimesFM hidden layer 有 position confounding。

---

# Slide 7. Most Important Visual: What Do Clusters Look Like?

## 屏幕文字

**Return to original time-series space**

- `c8`: rising / recovery motif candidate
- `c5`: falling / smooth-transition motif pool
- `c4`: first-patch artifact negative control

## 推荐图片


## 版式建议

- 全页大图。
- 右侧加三个标签：
  - green: `candidate motif`
  - blue: `motif pool`
  - red: `artifact`

## 口头讲法

> 这是我觉得最应该给老师看的图。聚类不是抽象点云，我们能回到原空间看到 patch shape。c8/c5 有 motif 形态，c4 也很像一个 motif，但它其实是 position artifact。

---

# Slide 8. Lineage: Hidden Motif Is Not Just Raw Shape

## 屏幕文字

**Representation lineage: raw -> tokenizer/projection -> hidden**

Evidence for contextualization:

- hidden cluster aggregates multiple tokenizer/source clusters
- original-space shape remains interpretable
- not every visually coherent cluster is valid

## 推荐图片


## 版式建议

- 左侧 70% 放图。
- 右侧 30% 放解释：
  - `multi-source aggregation`
  - `contextualized motif family`
  - `candidate for v1`

## 口头讲法

> 这页回答 H2。hidden cluster 不是简单复述 raw patch shape，而是把多个 tokenizer/source clusters 汇聚起来。这个更像 contextualized motif family。

---

# Slide 9. Negative Control: Why Confounder Audit Is Necessary

## 屏幕文字

**Visually coherent does not mean motif.**

TimesFM `c4`:

- shape looks coherent
- cross-domain sources exist
- but `patch_index=0` is 100%
- therefore: artifact, not motif taxonomy v1

## 推荐图片


## 版式建议

- 图放左侧。
- 右侧用红色框强调：`Do not name this as motif`

## 口头讲法

> 这页是防守页。我们主动展示负例，说明我们不是看到 shape 一致就命名 motif。c4 是 first-patch artifact，这个负例能保护我们的结论。

---

# Slide 10. From Cluster to Taxonomy v1 Families

## 屏幕文字

**Internal split gives cleaner motif families**

Current v1 pilot:

- `strong_falling_transition`: primary
- `smooth_falling_transition`: secondary / merge candidate
- `strong_rising_recovery`: candidate
- `artifact_first_patch_behavior`: negative control

## 推荐图片


## 版式建议

- 上半页放图。
- 下半页放 v1 table。

## 口头讲法

> 原始 cluster 还太粗，所以我们做 internal split。现在最稳的是 strong_falling_transition；smooth falling 可能未来会和它 merge；rising recovery 有信号，但跨模型弱一些。

---

# Slide 11. Controlled Retrieval Audit

## 屏幕文字

**A motif family must survive controlled retrieval.**

Conditions:

- cross-domain
- same patch-index
- same frequency
- cross-model

## 推荐图片


## 版式建议

- 全页图。
- 讲之前先说：绿色不是“证明”，只是“通过某个控制条件”。

## 口头讲法

> 我们不只看 unrestricted nearest neighbors。一个 motif family 要在 cross-domain、same-patch、same-frequency 这些条件下仍然保持 shape coherence。artifact 行说明高相似度也可能来自位置机制。

---

# Slide 12. Cross-Model Validation

## 屏幕文字

**Does TimesFM-derived taxonomy transfer to Chronos?**

Main result:

- `strong_falling_transition` transfers best
- `strong_rising_recovery` has weaker Chronos global retrieval
- artifact does not become a valid motif

## 推荐图片


## 版式建议

- 左侧图。
- 右侧写一句结论：
  `falling_transition is the most defensible cross-model motif family`

## 口头讲法

> 这页回答“是不是 TimesFM-specific”。falling transition 在 TimesFM、Chronos-2、Chronos-2-small 里都高于 matched random，是目前最稳的跨模型 motif family。

---

# Slide 13. Current Takeaway

## 屏幕文字

**Current claim**

> TSFM hidden space is not a direct copy of human-prior motif taxonomy v0.  
> It forms model-derived motif/prototype families, but these families must be audited for domain, frequency, and position confounding.

**Best current motif family:** `falling_transition`

## 版式建议

- 极简页。
- 中间放大 claim。
- 底部放三列：
  - `v0 as probe`
  - `v1 as discovery`
  - `audit as guardrail`

## 口头讲法

> 所以目前不是 final taxonomy claim，而是一个 taxonomy discovery protocol 加上 v1 pilot evidence。最稳的现象是 falling transition。

---

# Slide 14. Next Steps

## 屏幕文字

**Next experiments before claiming final taxonomy**

1. Domain-balanced prototype bank
2. Merge test: `strong_falling` vs `smooth_falling`
3. Direction-flip control: rising vs falling
4. Chronos-native motif discovery
5. Larger-scale stability check

## 版式建议

- 五个 action items。
- 标出优先级：
  - P0: domain-balanced prototype bank
  - P0: Chronos-native discovery
  - P1: larger-scale stability

## 口头讲法

> 我建议下一步不是继续堆模型，而是修 prototype bank 的偏差，并做 Chronos-native discovery。这样才能判断 v1 taxonomy 是否稳。

---

# Backup Slide A. Representation Layer Choice

## 可能被问

> 为什么不用 tokenizer/projection 层做 clustering？hidden layer 会不会引入 position artifact？

## 回答

tokenizer/projection 和 hidden layer 回答的是不同问题：

- tokenizer/projection：更像 local patch vocabulary，适合看 patch shape encoding。
- hidden state：更像 contextualized motif family，适合看 TSFM 内部时序语言。
- TimesFM hidden layer 确实有 patch-index confounding，所以必须做 position-aware audit。
- Chronos hidden layer patch-index NMI 更低，但 domain/frequency encoding 更强。

## 可用图


---

# Backup Slide B. More Detailed Cross-Model Evidence

## 推荐图片

![Cross-model prototype curves](../outputs/cross_model_validation/figures/cross_model_prototype_curves.png)

![Cross-model prototype agreement](../outputs/cross_model_validation/figures/cross_model_prototype_space_agreement.png)

![Cross-model global retrieval](../outputs/cross_model_validation/figures/cross_model_global_retrieval_shape.png)

## 口头备用

> Prototype-space agreement 在 Chronos 中仍高于随机，但 silhouette 明显低于 TimesFM，说明 motif 在 Chronos 里更弱、更交叠。global retrieval 里 falling transition 最稳。

---

# 老师可能会问的问题与应对

## Q1. 这个工作到底是 taxonomy，还是 interpretability？

**建议回答：**

两者都有，但主贡献可以写成 `model-derived motif taxonomy discovery protocol`。taxonomy 是产物，interpretability 是方法。我们不是人工定义 taxonomy 后验证，而是用 representation analysis 反向发现 taxonomy。

## Q2. 为什么 v0 taxonomy 不能直接作为 ground truth？

**建议回答：**

v0 是 human-prior / TSDM 语言，适合做 weak semantic probe。但真实 TSFM hidden clusters 会跨越多个 v0 labels，例如 transition family 可能同时包含 trend、level shift、mixed。直接用 v0 purity 会低估模型学到的 contextual motifs。

## Q3. KMeans 的 K 怎么定？会不会结果依赖 K？

**建议回答：**

KMeans 只用于 candidate generation，不是最终类别数。我们用 stability、prototype inspection、controlled retrieval 和 confounder audit 来决定是否命名 motif。下一步可以补充 multi-K stability 和 hierarchical split。

## Q4. silhouette 不高，为什么还能说有结构？

**建议回答：**

silhouette 是全局几何指标，不适合单独决定 motif。我们看的是局部 neighborhood 是否稳定、原空间 prototype 是否一致、controlled retrieval 是否通过。尤其 TSFM hidden space 同时编码 domain/frequency/context，低 silhouette 不等于没有 motif structure。

## Q5. 这些 motif 是不是 domain/frequency artifact？

**建议回答：**

这是我们最重视的风险，所以每个候选都做 domain/frequency/position audit。TimesFM `c4` 就是负例：形态一致但 patch_index=0，所以不进入 taxonomy v1。`falling_transition` 更稳，是因为它在 cross-domain、same-patch、same-frequency 和 cross-model 中都更强。

## Q6. 为什么最强结果来自 TimesFM，而不是 Chronos？

**建议回答：**

这可能是 architecture effect。TimesFM layer_10 的 PCA structure 和 transition motifs 更清楚，但也有 position artifact。Chronos patch-index confounding 低，但 domain/frequency encoding 更强。我们不会直接把 TimesFM taxonomy 当成最终答案，所以计划补 Chronos-native discovery。

## Q7. TimesFM patch=32，Chronos patch=16，怎么比较？

**建议回答：**

当前 cross-model mapping 是把 TimesFM patch `p` 映射到 Chronos patch `2p` 和 `2p+1`。这对完整 rising recovery 可能不公平，因为 32-step shape 被切成两个 16-step half-patches。因此 rising recovery 在 Chronos global retrieval 较弱。下一步会做 Chronos-native 16-step prototypes。

## Q8. 为什么 falling transition 最稳？会不会只是 normalization 或 sign artifact？

**建议回答：**

目前 falling transition 在三模型中均高于 matched random，是最稳候选。但 sign/normalization artifact 还需要 direction-flip control。我们下一步会把 rising/falling 做配对测试，检查是否只是 sign inversion。

## Q9. 这个和 Time-LLM / prototype / shapelet 有什么关系？

**建议回答：**

Time-LLM 使用 prototype/reprogramming 把 time series patch 对齐到 language model representation。我们这里反过来做：不 reprogram，而是看 frozen TSFM 内部是否自发形成 motif prototype families。Shapelet 提供了 original-space explanation 的语言。

## Q10. 论文贡献怎么写最稳？

**建议回答：**

当前最稳贡献是三点：

1. A representation-level protocol for discovering model-derived motif taxonomy in patch-based TSFMs.
2. Evidence that hidden clusters are not direct copies of human-prior motif taxonomy v0.
3. A controlled validation framework separating motif families from domain/frequency/position artifacts.

## Q11. 下一步最应该做什么？

**建议回答：**

优先做两个：

1. Domain-balanced prototype bank，避免 weather / traffic 主导。
2. Chronos-native motif discovery，避免 TimesFM-derived taxonomy 直接迁移。

然后再做 merge test 和 direction-flip control。

---

# 明天汇报时的 3 个关键句

1. **“我们不是把 v0 taxonomy 当 ground truth，而是把它当 probe；真正要发现的是 model-derived motif taxonomy v1。”**
2. **“最重要的证据不是 PCA 点云，而是 cluster 回到 original time-series space 后的 prototype shape，以及 controlled retrieval 是否存活。”**
3. **“目前最稳的 motif family 是 `falling_transition`；但我们主动保留 `first-patch artifact` 作为 negative control，说明这套 protocol 能识别假 motif。”**

