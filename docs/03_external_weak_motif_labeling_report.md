# Motif Taxonomy 迭代报告

## 1. 结论摘要

本轮调研和小实验后的判断是：**不要直接沿用 proposal 里的六类 taxonomy，而应使用一个更保守、可操作的 v0 taxonomy**。原 proposal 里的 `trend / oscillation / spike / burst / regime shift / intermittent` 方向是对的，但正式进入 TSFM patch embedding 分析前，需要做三处修正：

1. `regime shift` 应拆成 `level_shift` 和 `volatility_shift`。这对应 time series data mining 里常见的 mean change 与 variance/volatility change 两类检测问题，操作化规则也不同。
2. `spike` 与 `burst` 应先作为 **event-like prototype soft labels**，不要在真实数据上过早做大规模硬标签。它们对 patch boundary、噪声和事件宽度非常敏感。
3. 必须加入 `flat_low_information` 和 `mixed_uncertain`。前者是低信息控制组，后者是保护标签质量的必要 fallback，不是失败类。

推荐的 taxonomy v0 已写入：

- `configs/motif_taxonomy_v0.yaml`

最小探索脚本已写入并跑通：

- `scripts/explore_motif_taxonomy.py`
- 输出：`outputs/motif_taxonomy_exploration_summary.json`
- 图：`outputs/figures/motif_taxonomy_confusion_patch16.png`
- 图：`outputs/figures/motif_taxonomy_confusion_patch32.png`

## 2. 文献与代码库调研结论

### 2.1 Time Series Data Mining 给我们的约束

**Matrix Profile / motif discovery**  
Matrix Profile 体系把 time series motif 定义成重复出现的相似 subsequence，也同时支持 discord、chains、snippets 等结构发现。它适合回答“哪些 patch 形状反复出现”，但它不会天然给出 `trend`、`spike` 这样的语义名。因此在本项目里，Matrix Profile 更适合作为 **prototype discovery / consistency check**，不是单独的 taxonomy 来源。

参考：

- STUMPY docs: https://stumpy.readthedocs.io/en/latest/
- Matrix Profile XXII: https://arxiv.org/abs/2009.07907
- Time Series Motif Discovery: A Comprehensive Evaluation: https://www.vldb.org/pvldb/vol18/p2226-boniol.pdf
- Motiflets: https://arxiv.org/abs/2206.03735
- LoCoMotif: https://arxiv.org/abs/2311.17582

**Shapelets**  
Shapelet literature 把 subsequence 当作能区分类别的局部原型。它适合我们后续做 `motif prototype bank`：先人工确认少量高置信原型，再向邻近 patch 传播标签。但 shapelet 本身通常需要监督标签或半监督类名，因此它更适合作为第二阶段工具。

参考：

- Shapelet transform: https://ueaeprints.uea.ac.uk/id/eprint/40201/1/LinesKDD2012.pdf
- Learning Time-Series Shapelets: https://www.cs.ucr.edu/~eamonn/shaplet.pdf
- aeon shapelet transforms: https://www.aeon-toolkit.org/en/latest/api_reference/transformations.html

**SAX / Bag-of-patterns**  
SAX、PAA 和 bag-of-patterns 能把 patch 变成符号词，适合做可解释的离散 token sanity check。它们不直接给语义标签，但能帮助我们检查“TSFM patch token 的邻近关系”是否和传统 symbolic pattern 一致。

参考：

- SAX paper: https://www.cs.ucr.edu/~eamonn/SAX.htm
- pyts docs: https://pyts.readthedocs.io/

**Change-point / segmentation**  
`level_shift` 与 `volatility_shift` 不应靠 generic clustering 来定义，而应靠 change-point 或分段统计量定义。`ruptures` 的 PELT/BinSeg 等方法正好对应这一类操作化。

参考：

- ruptures docs: https://centre-borelli.github.io/ruptures-docs/

**Subsequence clustering 的风险**  
传统文献对 subsequence clustering 有明确警告：直接聚类滑窗 subsequence 很容易得到不稳定或无意义的中心。因此我们可以用 `tslearn` / kMeans / DTW clustering 做诊断，但不能把它当作主标签生成器。

参考：

- Clustering of Time Series Subsequences is Meaningless: https://www.cs.ucr.edu/~eamonn/meaningless.pdf
- tslearn docs: https://tslearn.readthedocs.io/

### 2.2 TSFM 文献给我们的约束

Chronos / Chronos-2 / TimesFM 都把时间序列转成 patch/token 或 patch-like 表征，再通过 Transformer 建模。TSFM 综述和表示分析论文说明“模型内部是否形成可解释 temporal concepts”仍是开放问题，所以我们这项研究应把 taxonomy 设计成 **分析工具**，而不是假装存在一个公认的 TSFM patch 标签体系。

参考：

- Chronos: https://arxiv.org/abs/2403.07815
- Chronos-2: https://arxiv.org/abs/2510.15821
- TimesFM: https://arxiv.org/abs/2310.10688
- Foundation Models for Time Series Analysis: https://arxiv.org/abs/2403.14735
- A Survey of Time Series Foundation Models: https://arxiv.org/abs/2405.02358
- Large Language Models for Time Series: https://arxiv.org/abs/2402.01801
- Exploring Representations and Interventions in TSFMs: https://arxiv.org/abs/2409.12915

### 2.3 Yuxuan Liang 相关工作给我们的 framing

老师相关工作更强调 cross-domain / unified / foundation-model-style time series modeling，而不是给出一个现成 motif taxonomy。对本项目最重要的启发是：

1. taxonomy 应服务于跨域表征分析，而不是只服务于某个单一数据集。
2. patch label 不需要是最终预测任务标签；它可以是解释 TSFM 表征空间的 weak supervision。
3. 需要区分 domain identity 与 temporal primitive identity，这和 proposal 里的 H1/H2 是一致的。

参考：

- Yuxuan Liang research page: https://yuxuanliang.com/research/
- Time-LLM: https://arxiv.org/abs/2310.01728
- UniTime: https://arxiv.org/abs/2310.09751
- Time-FFM: https://arxiv.org/abs/2405.14252

## 3. 本地库可用性

本轮补装并记录到 `pyproject.toml` / `uv.lock` 的轻量依赖：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  uv pip install --python .venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple \
  ruptures pyts tslearn

uv lock
```

当前本地状态：

| Library | Version | 本地状态 | 用途 |
|---|---:|---|---|
| `stumpy` | 1.14.1 | `stumpy.stump` 跑通 | Matrix Profile motif/discord/prototype check |
| `ruptures` | 1.1.10 | `Pelt(model="l2")` 跑通 | `level_shift` / change-point 检测 |
| `pyts` | 0.13.0 | `SymbolicAggregateApproximation` 与 `BagOfWords` 跑通 | SAX / symbolic pattern sanity check |
| `aeon` | 1.4.0 | `RandomShapeletTransform` import OK | shapelet prototype bank 第二阶段使用 |
| `tslearn` | 0.8.1 | `TimeSeriesKMeans` 跑通 | 仅作 exploratory clustering diagnostic |

注意事项：

- `pyts` 中本地可用类名是 `BagOfWords`，不是 `BagOfPatterns`。
- `aeon.transformations.collection.shapelet_based.RandomShapeletTransform` 可用；之前假设的某些 Matrix Profile transformer 路径不能直接使用。
- `tslearn` 运行时提示未安装 `h5py`，但不影响当前 clustering smoke test；后续如果要保存 tslearn 模型再考虑安装。

## 4. Taxonomy v0

最终推荐的 v0 类别如下：

| Label | 操作化定义 | 16-step patch | 32-step patch | 建议用途 |
|---|---|---|---|---|
| `flat_low_information` | raw std/range 极低 | 只在低噪声稳定 | 只在低噪声稳定 | control label |
| `trend` | 线性拟合斜率大且 R2 高 | 稳定 | 稳定 | hard label |
| `oscillation` | FFT dominant component 明显且有足够 zero-crossing | 部分稳定 | 较稳定 | hard label, 16-step 降低置信 |
| `impulse_spike` | 1-2 个极端 robust z-score 点 | 弱 | 弱到中等 | soft prototype label |
| `burst_event` | 连续高能 active run | 弱 | 弱 | soft prototype label |
| `level_shift` | 分段均值突变分数高 | 部分稳定 | 较稳定 | hard label |
| `volatility_shift` | 分段方差比高且均值变化不强 | 部分稳定 | 部分稳定 | hard label with caution |
| `intermittent` | 多个分离 active runs | 弱 | 较稳定 | hard label mainly for 32-step |
| `mixed_uncertain` | 冲突、复合、边界切断或低置信 | 必须保留 | 必须保留 | fallback |

这比原 proposal 的改动是：

- 保留 `trend`、`oscillation`、`intermittent`。
- 把 `spike / burst` 改为 event-like soft prototypes。
- 把 `regime shift` 拆成 `level_shift` 与 `volatility_shift`。
- 新增 `flat_low_information` 与 `mixed_uncertain`。

## 5. 小实验结果

运行命令：

```bash
.venv/bin/python scripts/explore_motif_taxonomy.py
```

脚本生成 3,240 个 synthetic patches，覆盖：

- patch length: `16`, `32`
- motif family: 9 类
- noise: `0.02`, `0.08`, `0.16`
- amplitude: `0.7`, `1.2`, `2.0`
- alignment: `0.0`, `0.33`, `0.66`, `1.0`

核心诊断：

| Patch length | Non-uncertain coverage | Ambiguity rate | Accuracy including mixed | Raw NN true-label agreement |
|---:|---:|---:|---:|---:|
| 16 | 0.638 | 0.362 | 0.386 | 0.792 |
| 32 | 0.573 | 0.427 | 0.542 | 0.837 |

解释：

- raw patch nearest-neighbor agreement 在 `16` 和 `32` 都不低，说明 synthetic primitives 在原始形状空间里已经有一定结构。
- `32` 的 hard-label accuracy 明显高于 `16`，尤其是 `oscillation`、`intermittent`、`level_shift`。
- `16` 对短事件非常敏感：`intermittent` 容易退化成 `impulse_spike`，`burst_event` 容易被切断或误判为 volatility/event conflict。
- `flat_low_information` 在高噪声设定下不稳定，这不是 taxonomy 失败，而是说明 flat label 应只用于低 raw variance 控制样本。

按类别看：

- 稳定：`trend`。
- 较可用：`oscillation` at 32、`level_shift` at 32、`intermittent` at 32。
- 谨慎可用：`volatility_shift`，需要更多 change-point/variance detector 校准。
- 暂不建议大规模硬标：`impulse_spike`、`burst_event`。
- 必须保留：`mixed_uncertain`。

## 6. 推荐 labeling protocol

下一阶段不要直接“全量自动贴标签”。建议按下面流程：

1. 对每个 patch 保留 metadata：`series_id`, `domain`, `start`, `end`, `patch_len`, `model`, `layer`。
2. 对 patch 做 robust z-normalization，同时保留 raw std/range。
3. 先跑 deterministic detectors：
   - linear fit: `trend`
   - FFT / zero-crossing: `oscillation`
   - robust outlier/run statistics: `impulse_spike`, `burst_event`, `intermittent`
   - mean split score: `level_shift`
   - std ratio: `volatility_shift`
   - raw std/range: `flat_low_information`
4. 若多个 non-flat detector 同时触发且分数接近，标为 `mixed_uncertain`。
5. 用 `stumpy` 做 Matrix Profile prototype check，验证高置信 patch 是否在同类里有重复近邻。
6. 用 `ruptures` 对 `level_shift` / `volatility_shift` 候选做局部 change-point sanity check。
7. 人工检查每类 top prototypes，再建立 prototype bank。
8. 真实数据标签传播时使用 `mixed_uncertain` 保护边界样本，不强行贴硬标签。

## 7. 对 TSFM representation 分析的下一步

建议下一步进入“taxonomy v0 + synthetic/real pilot embedding”阶段：

1. 把 `scripts/explore_motif_taxonomy.py` 中的 detector 抽成可复用模块，例如 `scripts/motif_labeling.py`。
2. 生成一个小型 `patch_bank`：
   - synthetic high-confidence patches
   - 少量真实数据 patch
   - 每个 patch 都有 taxonomy v0 label、confidence、fired detectors 和 metadata
3. 对三个模型提取同一批 patch/series 的 selected layer hidden states：
   - `Chronos-2-small`: patch length `16`
   - `Chronos-2`: patch length `16`
   - `TimesFM-2.5`: patch length `32`
4. 先跑 A/B 分析的最小版本：
   - top-k retrieval motif agreement
   - domain-vs-motif nearest-neighbor confusion
   - PCA/UMAP visualization
   - cluster purity / NMI / ARI
5. 对 `impulse_spike` 和 `burst_event` 先汇报 parent family `event_like`，只有高置信 prototype 才看 subtype。

模型代码路径沿用 `docs/02_feasibility_model_loading_report.md` 已验证结果：

- Chronos-2: `external/chronos-forecasting/src/chronos/chronos2/model.py`
- Chronos-2 pipeline embed: `external/chronos-forecasting/src/chronos/chronos2/pipeline.py`
- TimesFM-2.5: `external/timesfm/src/timesfm/timesfm_2p5/timesfm_2p5_torch.py`

## 8. Go / No-Go

**Go，但限定为 pilot experiment。**

可以继续做：

- taxonomy v0 的 high-confidence synthetic calibration
- 少量真实数据 prototype discovery
- 三个 TSFM 的 patch representation retrieval / clustering pilot

暂时不要做：

- 对真实数据全量硬标签
- 把 `impulse_spike` 与 `burst_event` 当作稳定 subtype 大规模评估
- 直接用 subsequence clustering 生成最终标签
- 把 synthetic taxonomy 结果解释为真实世界语义 ground truth

本轮最重要的结论是：我们已经有一个可执行的 v0 labeling protocol，但它应被视为 **weak supervision + prototype bank**，不是封闭的人工本体论。
