# 数据/模型中心的 TSFM Patch Concept Discovery 报告

## 1. 军师判断

我同意这个新方向，而且认为它应该成为主线：**先从真实数据 patch 和 TSFM representation 出发发现自然簇，再回头命名这些簇的 temporal meaning**。这比直接套用 `trend / oscillation / spike / burst / level_shift` 这样的先验 taxonomy 更客观，也更符合“TSFM 的 patch token 到底学到了什么”这个问题。

但我不建议完全抛弃前一版 taxonomy。更稳的定位是：

- 数据/模型中心方法：负责发现候选概念簇。
- taxonomy v0：负责解释、命名、校准和发现混杂因素。
- synthetic motifs：只作为 probe，不作为真实世界 ground truth。

换句话说，研究范式应从 **knowledge-first taxonomy** 调整为 **discover-first, name-second**。

## 2. 为什么这个想法更强

原路线的问题是：我们先定义 motif taxonomy，再问模型 embedding 是否按这些标签聚类。这样容易被审稿人质疑：

1. taxonomy 是否只是人为选的？
2. 真实数据里的 patch 是否真的能被这些类别覆盖？
3. 如果聚类不好，是模型没学到，还是 taxonomy 不合适？

新路线更贴近 representation analysis：

1. 先从大量真实 patch 抽 TSFM embedding。
2. 在 embedding space 中做降维、聚类、nearest-neighbor 和 prototype mining。
3. 再检查每个簇的代表性 patch、来源域、频率、位置、raw statistics、taxonomy-v0 proxy label。
4. 最后命名稳定、跨域、跨模型复现的簇。

这能把核心问题变成：**TSFM 自己把哪些 patch 认为相似？这种相似性是 temporal primitive、domain identity、scale/frequency、context position，还是其他训练目标相关因素？**

## 3. 相关文献依据

这个方向和 broader representation interpretability / concept discovery 是一致的：

- `ACE: Automatic Concept-based Explanations` 用 activation clustering 自动发现 concept，再做人类命名和解释。我们的 patch embedding cluster 可以类比为 TSFM 的 temporal concept discovery。  
  https://arxiv.org/abs/1902.03129
- `TCAV` 用 concept activation vectors 测试模型内部是否对人类概念敏感。我们后续可以把发现的 patch clusters 转成 concept vectors，测试不同层的 sensitivity。  
  https://arxiv.org/abs/1711.11279
- `Network Dissection` 强调不要只看最终任务表现，而要量化 hidden representation 是否对应可解释概念。  
  https://arxiv.org/abs/1704.05796
- `UMAP` / PCA / clustering 是 exploratory representation analysis 的常用工具，但只能辅助发现结构，不能单独证明语义。  
  https://arxiv.org/abs/1802.03426

和 time series 相关的约束也很重要：

- Chronos、Chronos-2、TimesFM 都是 patch/token 风格 TSFM，本项目从 patch representation 入手是自然的。  
  https://arxiv.org/abs/2403.07815  
  https://arxiv.org/abs/2510.15821  
  https://arxiv.org/abs/2310.10688
- TSFM survey 把可解释性、跨域共享表示、foundation model 内部机制列为开放问题。  
  https://arxiv.org/abs/2403.14735  
  https://arxiv.org/abs/2405.02358
- 传统 time series data mining 对 naive subsequence clustering 有明确警告：聚类结果可能反映 boundary、offset、scale 或 trivial subsequence，而不是真正 motif。因此我们必须做混杂诊断。  
  https://www.cs.ucr.edu/~eamonn/meaningless.pdf

## 4. Pilot 实验

### 4.1 数据

数据来源：

- `/data/junjieqiu/datasets/basicts_datasets`
- 排除：`BLAST`

本轮 pilot 覆盖了 22 个非 BLAST 数据集：

- `BeijingAirQuality`
- `CA`
- `ETTh1`
- `ETTh2`
- `ETTm1`
- `ETTm2`
- `Electricity`
- `ExchangeRate`
- `GBA`
- `GLA`
- `Gaussian`
- `Illness`
- `METR-LA`
- `PEMS-BAY`
- `PEMS03`
- `PEMS04`
- `PEMS07`
- `PEMS08`
- `Pulse`
- `SD`
- `Traffic`
- `Weather`

抽样设置：

- 每个数据集抽 `12` 个 windows
- 每个 window 长度 `128`
- 只使用主数值特征，即 `feature=0`
- 每个 patch 同时记录 `dataset`, `domain`, `frequency`, `node`, `start/end`, `patch_index`, raw statistics, taxonomy-v0 proxy label

### 4.2 模型

已跑通三个模型：

| Model | Patch length | Window embedding shape | Patch embeddings |
|---|---:|---:|---:|
| `Chronos-2-small` | 16 | `[264, 8, 512]` | 2112 |
| `Chronos-2` | 16 | `[264, 8, 768]` | 2112 |
| `TimesFM-2.5` | 32 | `[264, 4, 1280]` | 1056 |

运行命令：

```bash
.venv/bin/python scripts/pilot_data_centered_discovery.py --windows-per-dataset 12 --batch-size 64
```

输出文件：

- `outputs/data_centered_discovery_summary.json`
- `outputs/figures/data_centered_chronos_2_small_pca_clusters.png`
- `outputs/figures/data_centered_chronos_2_small_pca_domains.png`
- `outputs/figures/data_centered_chronos_2_pca_clusters.png`
- `outputs/figures/data_centered_chronos_2_pca_domains.png`
- `outputs/figures/data_centered_timesfm_2_5_pca_clusters.png`
- `outputs/figures/data_centered_timesfm_2_5_pca_domains.png`

### 4.3 聚类诊断

本轮使用 `StandardScaler + PCA(20) + KMeans` 做最小 pilot。结果如下：

| Model | Silhouette | Dataset purity | Domain purity | Taxonomy-v0 purity | Patch-index purity | Frequency purity |
|---|---:|---:|---:|---:|---:|---:|
| `Chronos-2-small` | 0.186 | 0.221 | 0.438 | 0.487 | 0.165 | 0.521 |
| `Chronos-2` | 0.173 | 0.211 | 0.450 | 0.474 | 0.165 | 0.480 |
| `TimesFM-2.5` | 0.124 | 0.127 | 0.373 | 0.460 | 0.474 | 0.403 |

NMI 结果：

| Model | NMI dataset | NMI domain | NMI taxonomy-v0 | NMI patch-index | NMI frequency |
|---|---:|---:|---:|---:|---:|
| `Chronos-2-small` | 0.274 | 0.268 | 0.112 | 0.013 | 0.251 |
| `Chronos-2` | 0.276 | 0.273 | 0.116 | 0.014 | 0.235 |
| `TimesFM-2.5` | 0.155 | 0.146 | 0.078 | 0.266 | 0.160 |

## 5. 怎么解读这组结果

### 5.1 好消息

当前 clusters 并不是简单按 dataset 分开。三个模型的 `dataset purity` 都不高，尤其 `TimesFM-2.5` 只有 `0.127`。这说明用真实数据 patch 做跨域 concept discovery 是有空间的，不是天然会退化成 dataset classifier。

`Chronos-2-small` 和 `Chronos-2` 的 patch-index NMI 很低，说明在这个 pilot 里 Chronos 的 cluster 没有明显被 window 内 patch 位置支配。

### 5.2 风险信号

`TimesFM-2.5` 的 patch-index purity 和 NMI 明显更高：

- patch-index purity: `0.474`
- NMI patch-index: `0.266`

这提示 TimesFM 的 final patch embedding cluster 可能部分反映了 context position / decoding setup，而不一定都是 temporal primitives。后续做 TimesFM 时必须控制 patch position，例如：

- 只比较同一 patch index 的 embeddings
- 或把同一个 patch 放到多个 context positions 做 invariance test
- 或对 patch index 做 residualization / stratified clustering

另一个风险是 frequency purity 对 Chronos 也不低，说明 sampling frequency 或 domain cadence 可能影响 embedding organization。后续必须把 frequency 作为 confounder，而不是只看 motif。

### 5.3 对 taxonomy-v0 的反思

taxonomy-v0 purity 看起来接近 `0.46-0.49`，但 NMI 很低，且 `mixed_uncertain` 占比很高。因此不能说模型 embedding 已经按 taxonomy-v0 聚类。更客观的说法是：

> taxonomy-v0 可以作为解释簇的辅助探针，但目前还不能作为主监督标签。

这正好支持新路线：先发现自然簇，再对簇做 naming 和 taxonomy refinement。

## 6. 方法论建议

我建议把项目方法重构为三层：

### Layer 1: Discovery

从真实多域 patch 出发，提取 TSFM patch embeddings，做：

- PCA / UMAP
- KMeans / HDBSCAN
- nearest-neighbor graph
- prototype mining
- layer-wise comparison
- model-wise comparison

输出不是最终标签，而是 candidate concept clusters。

### Layer 2: Confounder Audit

每个 cluster 必须同时检查：

- dataset purity
- domain purity
- frequency purity
- patch-index purity
- raw mean/std/range
- missingness / zero ratio
- taxonomy-v0 proxy label distribution
- model/layer stability

只有跨数据集、跨域、低 position confounding、且形态可解释的 cluster，才值得命名为 temporal primitive。

### Layer 3: Naming and Validation

对稳定 cluster 做人工命名和验证：

- 选每个 cluster 的 medoid / nearest-to-centroid patches
- 可视化原始 patch 和上下文
- 用 Matrix Profile / shapelet / SAX / change-point detector 辅助解释
- 和 synthetic probes 做 nearest-neighbor comparison
- 形成 model-derived taxonomy v1

## 7. 对 proposal 的更新建议

原 proposal 的核心假设可以保留，但需要改变验证顺序：

旧顺序：

1. 先定义 temporal primitive taxonomy
2. 给 patch 贴标签
3. 检查 TSFM embedding 是否按标签聚类

新顺序：

1. 从真实多域 patch 提取 TSFM representations
2. 发现自然簇和 nearest-neighbor structure
3. 审计 domain/frequency/position/scale 混杂
4. 对稳定簇命名 temporal concepts
5. 用 taxonomy-v0 和 synthetic probes 做辅助解释

因此，研究问题也应改成：

> TSFM patch representation 中是否存在稳定、跨域、可解释的自然概念簇？这些簇更像 temporal primitives，还是更像 domain/frequency/position artifacts？

这个表述更强，也更不容易被“你们的 taxonomy 是拍脑袋”攻击。

## 8. 下一步实验

建议下一步不要急着跑全量，而是做一个更严谨的 second pilot：

1. 每个数据集至少 `100-300` 个 windows，但按 domain/frequency 做 balanced sampling。
2. 对三个模型抽 selected layers，而不仅是 final embeddings。
3. 对 TimesFM 做 patch-index 控制：
   - same-index clustering
   - cross-position invariance
   - patch-index residualization
4. 对 Chronos 检查 frequency confounding：
   - same-frequency subset
   - cross-frequency normalized comparison
5. 用 HDBSCAN 或 nearest-neighbor graph 替代单一 KMeans，检查 cluster 稳定性。
6. 每个候选 cluster 输出 prototype panel，供人工命名。
7. 把命名后的 cluster 和 taxonomy-v0 对齐，形成 `model_derived_taxonomy_v1`。

## 9. Go / No-Go

**Go，而且建议把它提升为主路线。**

但必须保留 confounder audit。没有这个 audit，数据/模型中心方法很容易把 position、frequency、scale 或 domain 当成“时序语言”。

当前最合理的论文叙事是：

> 我们不预设时序语言的词表，而是从 TSFM patch representation 中发现候选 temporal concepts；再通过跨域稳定性、混杂审计和传统 time series mining 工具解释这些概念。

这比单纯证明 `trend / spike / oscillation` 是否聚类更有探索性，也更像一篇分析性研究。
