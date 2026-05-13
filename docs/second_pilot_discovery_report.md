# Second Pilot: 数据/模型中心 TSFM Patch Concept Discovery

## 1. 核心结论

本轮 second pilot 支持把项目主线调整为 **discover-first, name-second**：先从真实多域 patch 的 TSFM representation 中发现自然簇，再对稳定、跨域、低混杂的簇进行命名。

但当前结果也给出一个很重要的警告：**现在还不应该直接把 cluster 命名成最终 temporal primitives**。原因是 cluster 结构确实存在，但仍明显受到 `domain`、`frequency`、`patch_index` 和 raw scale/statistics 的影响。这个发现本身很有价值，因为它让我们的研究问题从“模型是否按我们预设 taxonomy 聚类”升级为：

> TSFM patch representation 中的自然簇，到底是 temporal concept，还是 domain / frequency / position / scale artifact？

## 2. 实验设置

脚本：

- `scripts/run_second_pilot_discovery.py`

运行命令：

```bash
.venv/bin/python scripts/run_second_pilot_discovery.py \
  --windows-per-dataset 100 \
  --batch-size 96 \
  --domain-balanced-patches 700
```

数据：

- 数据根目录：`/data/junjieqiu/datasets/basicts_datasets`
- 排除：`BLAST`
- 覆盖非 BLAST 的 22 个数据集
- 每个数据集抽取 `100` 个 windows
- 每个 window 长度 `128`
- 总 windows: `2200`
- 主数值特征：`feature=0`

模型与层：

| Model | Patch length | Selected layers | Patch embeddings per layer |
|---|---:|---|---:|
| `Chronos-2-small` | 16 | `layer_0`, `layer_3`, `layer_5` | 17,600 |
| `Chronos-2` | 16 | `layer_0`, `layer_6`, `layer_11` | 17,600 |
| `TimesFM-2.5` | 32 | `layer_0`, `layer_10`, `layer_19` | 8,800 |

每个 layer 都做两组分析：

- `full_equal_per_dataset`: 每个 dataset 等量抽样，但 domain 不平衡。
- `domain_balanced`: 每个 domain 最多取 `700` 个 patch，用于缓解 traffic 类数据过多的问题。

输出：

- `outputs/second_pilot_discovery_summary.json`
- `outputs/second_pilot/second_pilot_discovery_summary.json`
- `outputs/figures/second_pilot/`

本轮生成 `90` 张图，包括 PCA scatter 和 prototype panels。

## 3. 关键指标

### 3.1 Chronos-2-small

| Layer | Split | Silhouette | Stability | NMI dataset | NMI domain | NMI taxonomy-v0 | NMI patch-index | NMI frequency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `layer_0` | full | 0.073 | 0.540 | 0.219 | 0.222 | 0.157 | 0.063 | 0.212 |
| `layer_0` | domain-balanced | 0.104 | 0.594 | 0.314 | 0.322 | 0.201 | 0.052 | 0.272 |
| `layer_3` | full | 0.110 | 0.534 | 0.328 | 0.300 | 0.129 | 0.065 | 0.326 |
| `layer_3` | domain-balanced | 0.144 | 0.626 | 0.432 | 0.441 | 0.199 | 0.006 | 0.395 |
| `layer_5` | full | 0.145 | 0.561 | 0.263 | 0.251 | 0.111 | 0.013 | 0.233 |
| `layer_5` | domain-balanced | 0.168 | 0.653 | 0.375 | 0.384 | 0.170 | 0.008 | 0.322 |

解读：

- deeper layer 的 silhouette 和 KMeans-vs-Agglomerative stability 更高。
- patch-index NMI 很低，说明 Chronos-2-small 在这个设置下没有明显被 patch position 支配。
- domain-balanced 后 dataset/domain NMI 反而上升，说明一些小 domain 的结构更容易被凸显，也说明“平衡抽样”不是自动消除 domain confounding。

### 3.2 Chronos-2

| Layer | Split | Silhouette | Stability | NMI dataset | NMI domain | NMI taxonomy-v0 | NMI patch-index | NMI frequency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `layer_0` | full | 0.078 | 0.523 | 0.186 | 0.202 | 0.148 | 0.005 | 0.176 |
| `layer_0` | domain-balanced | 0.101 | 0.590 | 0.284 | 0.294 | 0.195 | 0.005 | 0.248 |
| `layer_6` | full | 0.104 | 0.552 | 0.307 | 0.302 | 0.151 | 0.067 | 0.295 |
| `layer_6` | domain-balanced | 0.146 | 0.627 | 0.429 | 0.445 | 0.200 | 0.006 | 0.380 |
| `layer_11` | full | 0.127 | 0.571 | 0.282 | 0.282 | 0.115 | 0.013 | 0.257 |
| `layer_11` | domain-balanced | 0.154 | 0.642 | 0.392 | 0.402 | 0.171 | 0.010 | 0.338 |

解读：

- `Chronos-2` 和 `Chronos-2-small` 的趋势一致：中/深层 cluster 更稳定。
- patch-index confounding 仍然很低。
- `layer_6` 的 domain/frequency signal 最强，说明中层可能既编码 temporal structure，也编码 cadence/domain-style 信息。

### 3.3 TimesFM-2.5

| Layer | Split | Silhouette | Stability | NMI dataset | NMI domain | NMI taxonomy-v0 | NMI patch-index | NMI frequency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `layer_0` | full | -0.003 | 0.376 | 0.102 | 0.109 | 0.065 | 0.125 | 0.109 |
| `layer_0` | domain-balanced | -0.011 | 0.354 | 0.114 | 0.118 | 0.069 | 0.121 | 0.117 |
| `layer_10` | full | 0.184 | 0.735 | 0.352 | 0.345 | 0.169 | 0.260 | 0.350 |
| `layer_10` | domain-balanced | 0.177 | 0.740 | 0.396 | 0.417 | 0.198 | 0.264 | 0.365 |
| `layer_19` | full | 0.137 | 0.661 | 0.271 | 0.261 | 0.095 | 0.224 | 0.272 |
| `layer_19` | domain-balanced | 0.072 | 0.599 | 0.297 | 0.314 | 0.138 | 0.169 | 0.259 |

解读：

- `TimesFM-2.5 layer_10` 的 cluster 最稳定，silhouette 和 clustering-stability 都最高。
- 但它也有明显 patch-index confounding：`NMI patch-index ≈ 0.26`。
- 这意味着 TimesFM 的中层自然簇不应直接解释成 temporal primitives，必须先做 same-position 或 position-residualized 分析。

## 4. 重要发现

### 4.1 Cluster 结构真实存在

相比 first pilot，样本扩大到 `2200` windows 后，cluster structure 仍然存在。尤其：

- Chronos 深层 silhouette 上升。
- TimesFM 中层 cluster stability 很高。
- KMeans vs Agglomerative NMI 在中/深层明显高于浅层。

这说明 representation space 里确实有可分析结构，不是随机噪声。

### 4.2 不能直接命名 cluster

taxonomy-v0 的 NMI 始终不高，大多在 `0.06-0.20`。这说明当前自然簇不是简单地对应我们之前设定的 `trend / level_shift / volatility_shift / intermittent` 等标签。

这不是坏事。它恰恰说明新路线有意义：TSFM 的“时序语言”可能不是人类先验 taxonomy 的一一映射。

### 4.3 Chronos 与 TimesFM 的混杂结构不同

Chronos:

- patch-index NMI 很低。
- domain/frequency signal 更明显。
- 中/深层更稳定。

TimesFM:

- 中层最稳定。
- patch-index confounding 明显。
- final layer 的 position confounding 有所下降，但 cluster quality 也下降。

这说明不同 TSFM architecture 学到的 patch representation 可能不是同一种“语言结构”。这会成为论文里的一个很有价值的对比点。

### 4.4 Frequency 是必须控制的变量

Chronos 和 TimesFM 都显示出不低的 frequency signal。尤其在 `domain-balanced` split 下，frequency NMI 仍然明显。后续不能只说“跨域”，还要区分：

- same-frequency cross-domain
- cross-frequency same-domain
- cross-frequency cross-domain

否则很容易把采样频率/cadence 当成 temporal primitive。

## 5. 当前最值得追的候选方向

从 cluster summary 看，低混杂候选簇多数不是纯单一 taxonomy，而是下面几种组合：

1. `mixed_uncertain + level_shift + trend`  
   可能对应“局部非平稳变化”或“directional transition”。

2. `trend + level_shift`  
   可能对应模型把 smooth drift 和 abrupt level change 看成相近的 transition family。

3. `mixed_uncertain + volatility_shift + intermittent`  
   可能对应“eventful / sparse / noisy local activity”，但需要 prototype panel 人工检查。

4. `flat_low_information + mixed_uncertain + level_shift`  
   可能对应 low-information 或 near-zero segments，但也可能只是 scale/zero-ratio artifact。

这些名字还不能作为最终 taxonomy，只能作为 **candidate concept families**。

## 6. 对研究路线的更新

建议后续将研究路线明确分成四步：

1. **Representation discovery**  
   从真实多域 patch 抽 embedding，做 clustering / nearest-neighbor graph / prototype mining。

2. **Confounder audit**  
   每个 cluster 必须审计 `dataset`, `domain`, `frequency`, `patch_index`, raw scale, zero ratio, taxonomy-v0 proxy。

3. **Prototype naming**  
   人工查看 medoid patches 和上下文片段，只命名跨域稳定、低混杂的簇。

4. **Concept validation**  
   对命名后的 concepts 做：
   - same-position retrieval
   - same-frequency retrieval
   - cross-model agreement
   - layer-wise emergence
   - synthetic probe nearest-neighbor check

## 7. 下一步建议

我建议下一步不要继续盲目加大数据，而是做 **prototype inspection + controlled retrieval**：

1. 从本轮结果中选每个模型/层的候选低混杂 clusters。
2. 输出每个 cluster 的 medoid patch、上下文窗口和 nearest neighbors。
3. 人工给 cluster 写临时名字，例如：
   - `transition_like`
   - `eventful_sparse`
   - `low_activity`
   - `oscillatory_fragment`
4. 对每个临时 concept 跑 controlled retrieval：
   - same patch-index only
   - same frequency only
   - cross-domain only
5. 如果某些 concept 在控制后仍成立，再进入 `model_derived_taxonomy_v1`。

## 8. Go / No-Go

**Go，但下一步必须是 prototype inspection，不是 full-scale experiment。**

本轮已经证明：

- 多域真实 patch 可以端到端抽取三种 TSFM 的多层 representation。
- embedding space 有稳定结构。
- 不同模型/层的混杂模式不同。
- 先验 taxonomy 只能作为 probe，不能作为主标签来源。

因此项目应继续沿着数据/模型中心路线推进，但每个发现都要经过 confounder audit 和 prototype naming。
