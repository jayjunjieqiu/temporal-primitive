# 本周故事版 PPT 草稿：From Patch Tokens to Dynamical Prototype States

> 目标：参考 `Position: Why a Dynamical Systems Perspective is Needed to Advance Time Series Modeling`，把现有 Chronos-2 结果组织成一个更有故事性的导师汇报。  
> 当前阶段只写 story draft 和 figure material redraw plan，不直接生成最终图片。  
> 图内文字建议全英文；PPT 正文可以中文讲述并保留 English technical terms。

## 0. 这次 PPT 的核心故事

### 故事主线

现有 TSFM 很像在学习一种 `temporal pattern recognition`：把 time series 切成 patches，再把 patch tokens 放进 Transformer 中做上下文建模。但 dynamical systems perspective 提醒我们，真实 time series 不只是 patch pattern 的拼接，而是底层 system state 沿着某种 flow / transition law 演化的观测。

我们的 Chronos-2 结果正好提供了一个切入口：

1. `projection` 和 `layer_0` 仍能看到 local patch vocabulary；
2. `layer_6` 和 `layer_11` 更稳定，但也更容易混入 domain/frequency/context-style information；
3. representation-space clusters 用 Euclidean 能稳定产生 candidate neighborhoods；
4. 但回到 original time-series space 后，是否是 shapelet-like motif/prototype family，需要 DTW-aware validation；
5. 这说明当前 patch-based TSFM 可能学到了 local motifs 和 contextual neighborhoods，却未必显式学到 motif 之间的 dynamical transition law。

### 最终结论必须落到的可验证猜想

推荐把结论命名为：

> **Dynamical Prototype State Hypothesis for TSFMs**

中文讲法：

> TSFM 的 patch token 不应只被看作静态的 motif vocabulary，而应被看作对底层 dynamical state 的局部观测。更好的 TSFM 设计应同时学习：local motif prototypes、prototype-to-prototype transition structure，以及 regime-level context。这个猜想可以通过 layer-wise representation audit、DTW-aware prototype validation、transition graph stability 和 OOD/regime-shift forecasting 来验证。

更短的屏幕版：

> **A stronger TSFM should learn not only patch motifs, but the transition geometry among motif states.**

### 这不是空泛假设：我们已经有的证据链

- 早层保留 local information：`projection / layer_0` 的 all-cluster center-nearest patches 更容易回到 original space 解释。
- 中层开始重组：`layer_6` 的 per-layer K 倾向 `K=10`，说明 middle layer 可能存在更细 contextual substructure。
- 顶层更 contextualized：`layer_11` 稳定性高，但 raw-shape coherence 不一定更好。
- 距离原则已经清楚：representation space 用 Euclidean，original space 用 DTW。
- 稳健边界也清楚：现在还不能 claim final motif taxonomy；只能 claim 一个可验证的 design hypothesis。

## 1. 参考文章如何进入我们的故事

文章 `Position: Why a Dynamical Systems Perspective is Needed to Advance Time Series Modeling` 的关键点可以这样转译到我们的项目里：

- 时间序列通常来自 underlying dynamical system，而不是孤立 pattern 序列。
- 只看 short-term forecasting performance 可能掩盖模型是否理解了 long-term dynamics。
- Transformer-based TS models 更像 temporal pattern recognizers，缺乏自然的 time / flow representation。
- Dynamical Systems Reconstruction 强调 state space、attractor、transition、long-term statistics、tipping points 和 topological OOD。
- 对 TSFM 设计的启发：不要只扩大模型和数据，还要检查模型是否学到 dynamical state representation 和 transition structure。

我们不是要把 Chronos-2 改造成 DSR 模型，而是提出一个中间问题：

> patch-based TSFM 的 token space 里，是否已经出现了 dynamical state 的雏形？

这个问题比“cluster 像不像 motif”更有研究张力。

## 2. 建议 PPT 页序

建议主 PPT 10-12 页。每页讲一个递进问题，形成故事。

## Slide 1. Opening: The Missing Mechanism

### 屏幕文案

**What do TSFMs learn beyond forecasting?**

From patch-token pattern recognition to dynamical prototype states

### 故事功能

开场不要从实验设置开始，而是从 tension 开始：TSFMs 能 forecast，但我们不知道它们学到的是 static pattern dictionary，还是对 dynamical state 的某种 representation。

### 图片怎么画

重画一张横向 conceptual figure：

- 左侧：raw time series 被切成多个 patch tokens，标注 `patch tokens`。
- 中间：Transformer blocks，标注 `representation space`。
- 右侧分叉：
  - 上支：`static pattern dictionary`，画几个孤立 patch prototype；
  - 下支：`dynamical prototype states`，画 prototype nodes + arrows。
- 最右侧放 research question：`Which one is Chronos-2 closer to?`

参考现有图：不直接引用旧图；概念上参考 `docs/94_weekly_advisor_ppt_draft.md` 的 two-space principle。

### 口头讲法

> TSFM 的成功通常用 forecasting error 证明，但如果我们想指导下一代 TSFM 设计，就需要知道模型内部到底学到了什么。它是在记 patch pattern，还是已经开始形成类似 dynamical state 的结构？

## Slide 2. Why Dynamical Systems Perspective Changes the Question

### 屏幕文案

**Time series are observations of evolving systems.**

- A patch is a local observation, not the full state.
- A motif is useful only if we know how it transitions.
- OOD/regime shift is a change in dynamics, not just a new sample.

### 故事功能

把参考文章的核心观点转成我们的研究问题：真实 TS 的关键不是单个 patch，而是 state space 和 transition law。

### 图片怎么画

画一张上下对比图：

- 上半部分：`Pattern view`  
  一条 time series 被切成 patch A/B/C，箭头很弱或没有箭头，像 bag of motifs。
- 下半部分：`Dynamical view`  
  patch A/B/C 是 state-space trajectory 上的局部投影，画一个二维 state space，轨迹穿过几个 colored regions，每个 region 对应 prototype state。
- 右侧写：`Design question: should TSFMs learn transitions among prototype states?`

参考文章图意：state space / attractor / trajectory 的 conceptual style，可参考该文 Fig. 1 的 state-space trajectory 叙事，但不要复刻原图。

### 口头讲法

> 如果从 dynamical systems 角度看，patch motif 本身不是终点。真正重要的是这些 motif 是否是 state space 中可复现的局部状态，以及模型是否知道它们如何转移。

## Slide 3. Our Entry Point: Chronos-2 as a Frozen Microscope

### 屏幕文案

**We use Chronos-2 as a frozen microscope of patch-token representations.**

- No fine-tuning
- Four representation levels
- Macro-domain balanced patch bank
- Original-space validation

### 故事功能

把实验设置放到故事里：我们不是做 benchmark，而是用 frozen TSFM 做 mechanism diagnosis。

### 图片怎么画

重画一张简洁 protocol figure：

1. `Heterogeneous time series`
2. `Patch bank`
3. `Chronos-2 frozen encoder`
4. 四个出口：`projection`, `layer 0`, `layer 6`, `layer 11`
5. 两个 audit head：
   - `representation-space clustering`
   - `original-space DTW validation`

参考现有图：

- `../outputs/chronos_multilayer_validation/figures/layer_comparison_summary.png`
- `../outputs/distance_metric_ablation/figures/distance_metric_heatmap.png`

但本页应该重新画流程图，不直接放已有 metric 图。

### 口头讲法

> 我们把 Chronos-2 当成 frozen microscope，观察 patch token 从 projection 到 top layer 的演化。这样能避免训练过程带来的额外变量。

## Slide 4. Evidence I: The Local Vocabulary Is Still There

### 屏幕文案

**Early representations preserve local patch vocabulary.**

- `projection` and `layer_0` produce interpretable original-space neighborhoods.
- Single patch information is not erased immediately.

### 故事功能

回答老师的第一个问题：single patch 是否仍保留 local information。

### 图片怎么画

重画一张 `early-layer prototype panel`：

- 左列：`projection` 的 6 个 clusters，每行只画一个 cluster 的 KMeans-center nearest 4 patches。
- 右列：`layer 0` 的 6 个 clusters，同样每行 nearest 4 patches。
- 每行标 `C0-C5`，不要命名 motif。
- 用小圆点/浅色背景表示 cluster evidence tier：`candidate`, `weak`, `confounded`，但不要让图变复杂。
- 图内标题：
  - `Projection: local observation geometry`
  - `Layer 0: early patch vocabulary`

参考现有图：

- `../outputs/chronos_multilayer_validation/figures/projection_main_center_nearest.png`
- `../outputs/chronos_multilayer_validation/figures/layer_0_main_center_nearest.png`

重画目标：

- 比现有 all-cluster 图更紧凑；
- 每个 cluster 只保留 center-nearest examples；
- 线条更粗，字体更大；
- 不放中文，不放冗余 caption。

### 口头讲法

> 这说明 Chronos-2 的早层不是纯粹的 context abstraction。它仍然保留了 local observation geometry，因此 early layer 适合作为 local motif/prototype vocabulary 的观察窗口。

## Slide 5. Evidence II: Middle Layers Start to Mix Context

### 屏幕文案

**Middle-layer clusters are stable, but less purely local.**

- `layer_6` prefers a finer split (`K=10`).
- Some neighborhoods become contextual subfamilies.
- Confounder risk increases.

### 故事功能

把 layer_6 的 K=10 和 contextual mixing 讲成 dynamical transition 的线索，而不是“结果变差”。

### 图片怎么画

重画一张 `layer 6 split figure`，三栏：

1. 左：K selection mini plot，只显示 `layer_6` 的 K candidates，突出 `K=10`。
2. 中：`layer_6 K=6` 的 compressed cluster prototypes。
3. 右：`layer_6 K=10` 的 compressed cluster prototypes。

在图下方加一句英文小字：

`Finer split suggests contextual substructure, not a final taxonomy.`

参考现有图：

- `../outputs/chronos_multilayer_validation/figures/k_selection_summary.png`
- `../outputs/chronos_multilayer_validation/figures/layer_6_main_center_nearest.png`
- `../outputs/chronos_multilayer_validation/figures/layer_6_k10_center_nearest.png`

重画目标：

- 不需要展示所有 metric，只突出 `K=6 shared` vs `K=10 layer-specific`。
- prototype 图每个 cluster 只画 center + nearest examples，保持紧凑。

### 口头讲法

> layer_6 最有意思：它稳定，但 K=10 更合适。这可能说明 middle layer 不是简单保留 motif，而是在把 local patches 组织成更细的 contextual neighborhoods。这个现象是我们后面提出 design hypothesis 的关键。

## Slide 6. Evidence III: Top Layers Are More Contextualized

### 屏幕文案

**Top-layer neighborhoods are stable but harder to name as raw-shape motifs.**

- High seed stability
- More context/domain/cadence information
- Original-space coherence is uneven

### 故事功能

避免把 top layer 结果说成失败，而是说它回答的是另一个层级的问题：contextualized state/regime，不一定是 raw patch shape。

### 图片怎么画

重画一张 `layer comparison strip`：

- 四个 columns：`projection`, `layer 0`, `layer 6`, `layer 11`
- 每列三条 horizontal bars：
  - `seed stability ↑`
  - `confounder NMI ↓`
  - `raw-shape coherence ↑`
- 用颜色渐变表现：
  - early layers: local vocabulary
  - middle/top layers: contextual mixing

参考现有图：

- `../outputs/chronos_multilayer_validation/figures/layer_comparison_summary.png`

重画目标：

- 只保留故事需要的 3-4 个指标；
- 每个指标旁用 ↑/↓；
- 图中不出现下划线，用 `Layer 0` 而非 `layer_0`。

### 口头讲法

> 如果老师问为什么不用 top layer，我会说 top layer 不是没用，而是它已经不是单纯 local motif space。它可能更接近 contextualized representation，因此需要不同的验证方式。

## Slide 7. The Distance Problem: Motif Shape Lives in Original Space

### 屏幕文案

**The same neighborhood must be evaluated in two spaces.**

- Representation space: Euclidean geometry
- Original space: DTW geometry
- Motif naming requires original-space validation

### 故事功能

把 DTW vs Euclidean ablation 嵌入故事：不是技术细节，而是定义“什么证据才算 motif”的关键。

### 图片怎么画

重画一张 `two-space distance principle` 图：

- 左侧：hidden vectors 的 2D scatter，标注 `Euclidean neighborhood`。
- 中间：取出同一 cluster 的 patches。
- 右侧：original-space curves，用两种 bracket 标注：
  - `raw Euclidean may miss shifted shapes`
  - `DTW validates shape coherence`
- 下方放一条 verdict：
  `Euclidean discovers what the model groups; DTW tests whether the group is shapelet-like.`

参考现有图：

- `../outputs/distance_metric_ablation/figures/distance_metric_heatmap.png`
- `../outputs/distance_metric_ablation/figures/prototype_metric_comparison_layer_0_k6.png`

重画目标：

- 不直接放 heatmap；先画原则图。
- heatmap 可作为 backup。

### 口头讲法

> 这页是为了防止一个误解：我们不是说 Euclidean 错了。Euclidean 在 representation space 是合理的；DTW 在 original space 是合理的。两个空间回答不同问题。

## Slide 8. Evidence IV: DTW Reveals Which Neighborhoods Are Shapelet-Like

### 屏幕文案

**DTW separates shape-coherent neighborhoods from weak or confounded ones.**

- Some clusters are DTW-benefited.
- Some remain weak even under DTW.
- Some are confounded by domain/frequency/position.

### 故事功能

说明我们有审计机制，不只是看起来像就命名。

### 图片怎么画

重画一张 `DTW audit matrix`：

- 行：选择 8-10 个代表性 cluster setting，但要覆盖：
  - `projection K6`
  - `layer 0 K6`
  - `layer 6 K6`
  - `layer 6 K10`
  - `layer 11 K6`
- 列：
  - `DTW ratio ↓`
  - `DTW gain ↑`
  - `macro diversity ↑`
  - `confounder risk ↓`
  - `verdict`
- 最后一列用 `candidate`, `weak`, `confounded`，不要用 motif 名字。

参考现有图：

- `../outputs/distance_metric_ablation/figures/distance_metric_heatmap.png`
- `../outputs/distance_metric_ablation/cluster_metric_table.csv`

重画目标：

- 比当前 heatmap 更像 paper table-figure；
- metric label 使用英文 phrase，不用 code-style 下划线；
- 用 ↑/↓ 标方向。

### 口头讲法

> 有些 cluster 用 DTW 看确实更 coherent，说明 raw Euclidean 低估了它们的 shape similarity。但也有 cluster DTW 也救不回来，说明问题不是距离，而是 representation neighborhood 本身混杂。

## Slide 9. The Missing Object: Transition Geometry

### 屏幕文案

**Motifs are not enough. We need transitions among motif states.**

Current TSFM evidence:

- early layers: local prototype vocabulary
- middle layers: contextual subfamilies
- top layers: regime/context mixture

Missing test:

- prototype-to-prototype transition structure

### 故事功能

这是整套故事的转折点：从现有证据推导出“下一代设计需要什么”。

### 图片怎么画

重画一张 central figure：

- 左侧：三层结构金字塔
  - `Local motif prototypes`
  - `Contextual prototype families`
  - `Regime-level dynamics`
- 右侧：一个 directed graph
  - nodes 是 prototypes；
  - arrows 是 transition probabilities / flow directions；
  - 其中某些 arrows 跨 macro-domain 复现，画成实线；
  - confounded arrows 画成虚线。
- 中间放一句：
  `A useful temporal language should include both vocabulary and grammar.`

参考现有图：

- `layer_6 K10` 的 finer split；
- `two-space distance principle`；
- 不直接复用旧图，需新画 conceptual schematic。

### 口头讲法

> 如果我们把 motif 当作 temporal vocabulary，那么 dynamical systems perspective 告诉我们还缺 grammar：这些 prototype states 如何转移，哪些转移跨 domain 稳定，哪些只是 domain/cadence artifact。

## Slide 10. Design Hypothesis for TSFMs

### 屏幕文案

**Dynamical Prototype State Hypothesis**

A stronger TSFM should learn:

1. local motif prototypes
2. prototype-to-prototype transition geometry
3. regime-level context

This should improve OOD/regime-shift forecasting and long-term statistics.

### 故事功能

把结论明确变成能指导设计的 hypothesis。

### 图片怎么画

画一张 `design blueprint`：

- 左侧：current patch-based TSFM
  - `patch tokenizer`
  - `Transformer encoder`
  - `forecast head`
- 右侧：proposed DS-aware TSFM
  - `patch tokenizer`
  - `prototype state layer`
  - `transition regularizer / state-space consistency`
  - `regime-aware forecast head`
- 中间用箭头标注新增 inductive bias：
  - `DTW-validated prototype bank`
  - `transition graph consistency`
  - `long-term statistic regularization`

注意：这页不能写成已经实现，只写成 design conjecture。

### 口头讲法

> 这就是本周想给老师看的真正结论。我们的实验不是为了得到一个漂亮 cluster 图，而是提出一个可以指导 TSFM 设计的猜想：patch token space 应该显式组织成 dynamical prototype states，并学习它们之间的 transition geometry。

## Slide 11. How to Verify the Hypothesis

### 屏幕文案

**Verifiable predictions**

P1. Early-layer prototypes are more DTW-coherent than top-layer raw-shape clusters.  
P2. Stable transition graphs predict future patch states across domains.  
P3. DS-aware regularization reduces confounder-driven clusters.  
P4. Improvements appear in OOD/regime-shift and long-term statistic metrics, not only MSE.

### 故事功能

让 conclusion 可实验、可反驳，而不是空泛理论。

### 图片怎么画

重画一张 `hypothesis-to-test map`：

- 四行对应 P1-P4；
- 三列：
  - `measurement`
  - `expected evidence`
  - `failure mode`
- 示例：
  - P1 measurement: `DTW ratio, prototype stability`
  - P2 measurement: `transition graph NMI / Wasserstein`
  - P3 measurement: `domain/frequency/position NMI`
  - P4 measurement: `OOD forecasting, long-term spectral distance`

参考文章启发：

- long-term temporal/geometrical metrics；
- topological OOD / regime shift；
- DSR 强调长期统计和 state-space geometry。

### 口头讲法

> 这个 hypothesis 的好处是可验证：如果 transition graph 不稳定，或者对 OOD/regime-shift 没帮助，那就说明这个设计方向不成立。

## Slide 12. Closing: What We Learned This Week

### 屏幕文案

**Current answer**

- Chronos-2 early layers preserve local temporal information.
- Middle/top layers reorganize patches into contextual neighborhoods.
- Original-space DTW is needed before naming motif/prototype families.
- The next design target is transition-aware prototype state modeling.

**Claim boundary**

Discovery protocol and design hypothesis, not final taxonomy.

### 图片怎么画

可以复用 Slide 9 的 prototype transition graph，放成淡背景；前景放四条 conclusion。

### 口头讲法

> 这周我们把问题从“cluster 长得像不像 motif”推进到“TSFM 是否学到了 dynamical prototype states”。现在的证据支持一个更强的下一步：构建 transition-aware motif/prototype representation，并用 OOD/regime-shift 和 long-term statistic 评价它是否真的指导 TSFM 设计。

## 3. Backup Slides

### Backup A. Paper Reference

屏幕文字：

`Position: Why a Dynamical Systems Perspective is Needed to Advance Time Series Modeling`

Key messages:

- TS observations often arise from underlying dynamical systems.
- Short-term forecast error is not enough.
- Long-term statistics, attractor geometry, tipping points, and topological OOD matter.
- Transformer TSFMs may behave more like temporal pattern recognizers than dynamical reconstructors.

图片怎么画：

- 左侧放 paper title card；
- 右侧放 4 个 takeaway icons：
  - `state space`
  - `flow`
  - `attractor`
  - `regime shift`

### Backup B. Existing Evidence Table

屏幕文字：

| Evidence | Current result | Interpretation |
|---|---|---|
| Early layer prototypes | interpretable center-nearest patches | local vocabulary exists |
| Layer-wise metrics | high stability but rising confounders | context mixing |
| Layer 6 K search | K=10 preferred | finer contextual substructure |
| DTW ablation | some clusters DTW-benefited, others weak | original-space validation needed |
| Failure cases | DTW over-warping and representation/raw mismatch | no overclaiming |

图片怎么画：

- 做成 publication-style summary table；
- 每行配一个小 sparkline 或 icon；
- 不放大段文字。

### Backup C. If Teacher Asks “Is This Still Motif Taxonomy?”

推荐回答：

> 是，但现在的 motif taxonomy 不是静态标签表，而是更接近 dynamical taxonomy。第一层是 local motif prototypes，第二层是 contextual prototype families，第三层是 prototype transition / regime structure。我们仍然用 motif taxonomy 的语言，但把它从 pattern taxonomy 推进到 state-and-transition taxonomy。

### Backup D. If Teacher Asks “How Would This Change TSFM Architecture?”

推荐回答：

> 可以有三种轻量方向。第一，在 tokenizer/projection 后加入 prototype state bottleneck，让 patch token 对齐到可解释 prototype bank。第二，在训练中加入 transition consistency，让连续 patches 的 representation transition 可预测、跨域稳定。第三，加入 long-term statistic / spectral / regime-shift objective，避免只优化 short-term MSE。这些都不要求完全放弃 Transformer，而是给 patch-based TSFM 加 dynamical inductive bias。

### Backup E. If Teacher Asks “How to Evaluate?”

推荐回答：

> 评价要分两层。representation 层看 prototype stability、DTW coherence、transition graph stability、confounder NMI。task 层看 OOD/regime-shift forecasting、long-term statistic distance、spectral distance，以及是否能提前识别 regime transition。这样比单看 MSE 更接近 dynamical systems perspective。

## 4. 需要重画的 figure material 清单

| figure | purpose | reference | redraw instruction |
|---|---|---|---|
| Fig 1: static pattern vs dynamical prototype states | 开场问题 | conceptual only | 左侧 patch tokenizer，右侧分成 static dictionary / state graph |
| Fig 2: pattern view vs dynamical view | 引入 DS perspective | paper Fig. 1 style | 不复刻原图，画 state-space trajectory 与 patch observations |
| Fig 3: Chronos-2 frozen microscope | 方法流程 | current scripts/reports | frozen encoder 四层出口 + two audit heads |
| Fig 4: early-layer prototype panel | single patch local information | `projection_main_center_nearest.png`, `layer_0_main_center_nearest.png` | 紧凑重画 projection/layer 0 C0-C5 |
| Fig 5: layer 6 split | middle-layer contextual substructure | `k_selection_summary.png`, `layer_6_*center_nearest.png` | K=6 vs K=10 对照，不命名 motif |
| Fig 6: layer-wise metric strip | 层间变化 | `layer_comparison_summary.png` | 只保留 stability/confounder/coherence |
| Fig 7: two-space distance principle | 距离原则 | `prototype_metric_comparison_layer_0_k6.png` | hidden Euclidean vs original DTW schematic |
| Fig 8: DTW audit matrix | 原空间验证 | `distance_metric_heatmap.png`, CSV table | metric table-figure，带 ↑/↓ |
| Fig 9: motif vocabulary to transition grammar | 故事转折 | new conceptual | prototype nodes + transition arrows |
| Fig 10: DS-aware TSFM design hypothesis | 最终设计猜想 | new conceptual | current TSFM vs proposed transition-aware TSFM |
| Fig 11: hypothesis-to-test map | 可验证性 | new conceptual | P1-P4 measurement / evidence / failure mode |

## 5. 本周结论建议怎么说

最稳妥版本：

> 当前结果说明，Chronos-2 的 early patch-token representations 保留了可解释的 local temporal information；middle/top layers 会把这些 local motifs 重组为更 contextualized 的 neighborhoods。仅靠 representation-space Euclidean clustering 不能定义 motif taxonomy，必须用 original-space DTW validation 和 confounder audit。基于这个证据，我们提出一个可验证设计猜想：下一代 patch-based TSFM 不应只学习 static motif vocabulary，而应学习 dynamical prototype states and their transition geometry。

更有故事性的版本：

> 如果把 motif 看作 temporal vocabulary，那么 TSFM 还缺少 grammar。Dynamical systems perspective 告诉我们，grammar 就是 prototype states 之间的 transition law。我们的 Chronos-2 layer-wise evidence 暗示：早层有 vocabulary，中层开始形成 contextual subfamilies，但 transition geometry 还没有被显式建模。这个缺口可以变成下一步 TSFM 设计的机会。

## 6. 下一步实验建议

为了让这个故事从 PPT hypothesis 变成 paper evidence，建议下一步做：

1. **Prototype transition graph extraction**  
   从连续 windows 的 patch cluster sequence 中估计 transition matrix / directed graph。

2. **Layer-wise transition stability**  
   比较 `projection / layer_0 / layer_6 / layer_11` 的 transition graph 是否跨 seed、macro-domain、dataset 稳定。

3. **DTW-aware prototype state validation**  
   只有 DTW-coherent clusters 才进入 prototype graph；weak/confounded clusters 进入 diagnostic。

4. **OOD/regime-shift evaluation**  
   选择带 regime shift / abrupt transition 的 datasets 或 synthetic systems，检查 transition-aware features 是否能提前识别 state change。

5. **Design ablation**  
   尝试轻量 prototype bottleneck / transition regularizer，不改动大模型权重时可先做 post-hoc prediction head 或 retrieval-based transition predictor。

