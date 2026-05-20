# Chronos-2 Layer Effect Report: 聚类和 motif 空间如何随层变化

## 0. Advisor Question

这版报告回答 Yuxuan Liang 老师关心的机制问题：Chronos-2 的 `projection`、`layer_0`、`layer_6`、`layer_11` 是否保留 single patch 的 local temporal information，以及这些 local primitives 如何被 transformer layers 重组为 contextualized cross-domain temporal concepts。

由于当前 external weak motif labels 稳定性不足，本报告不再把它放入主证据链。所有 cluster 只写作 `C0, C1, ...`，主结论只依赖 representation geometry、original-space prototype、macro-domain evidence 和 confounder audit。

方法上采用 **two-space distance principle**：在 `Chronos-2 representation space` 中用 Euclidean geometry / KMeans 发现模型内部的 patch-token neighborhoods；回到 `original time-series space` 后，再用 DTW-aware validation 判断这些 neighborhoods 是否足以被命名为 shapelet-like motif/prototype family。本报告的 K 是 representation-space operating point，不是 DTW clustering 的 K。

## 1. Pilot Limitations

- old windows per dataset: `100`
- old context length / patch length: `128` / `16`
- old raw windows / estimated raw patches: `2200` / `17600`
- old clustered patches per representation: `{'raw_patch': 7700, 'projection': 7700, 'layer_0': 7700, 'layer_6': 7700, 'layer_11': 7700}`
- old K rule result: `{'raw_patch': 15, 'projection': 15, 'layer_0': 15, 'layer_6': 15, 'layer_11': 15}`

旧 macro-domain nearest 图是有价值的 diagnostic，但不适合作为主证据：它强制每个 cluster-domain cell 都找 nearest sample，因此即使某个 macro-domain 没有可信 match，也会出现视觉上很弱的曲线。

## 2. Improved Experimental Protocol

- model: `Chronos-2`
- representations: `projection, layer_0, layer_6, layer_11`
- windows per dataset: `500`
- selected balance mode: `macro_domain`
- max patches per macro-domain: `1500`
- max patches per dataset within macro-domain: `350`
- K candidates after coarse-to-fine search are recorded in `k_sweep_metrics.csv`。

K selection 不使用 silhouette-only，而是综合 seed stability、KMeans vs Agglomerative agreement、cluster size、confounder NMI、Davies-Bouldin 和 original-space evidence。需要注意：这里选择的是 representation-space K；motif/prototype 命名还需要后续 DTW-aware original-space validation。

主证据筛选也预先定义：cluster 不能太小，center-nearest raw patches 需要视觉一致，confidence-filtered macro-domain view 需要多个真实 macro-domain 的可信 match，同时不能明显被 single dataset、frequency 或 patch index 主导。后续进入 motif/prototype family 的候选，还必须通过 DTW medoid / DTW controlled retrieval。

## 3. K Selection Result

Recommended shared K: **`6`**

| representation | recommended per-layer K | top candidates |
|---|---:|---|
| `projection` | 6 | `[6, 7, 9, 8, 20]` |
| `layer_0` | 6 | `[6, 7, 20, 12, 18]` |
| `layer_6` | 10 | `[10, 9, 11, 16, 12]` |
| `layer_11` | 6 | `[6, 7, 9, 10, 8]` |

`layer_6` 的 per-layer K 倾向更细的划分，但本报告主图采用 shared K，是为了让 `projection -> layer_0 -> layer_6 -> layer_11` 的层间比较保持同一 operating point。这不是否认 `layer_6` 内部可能需要更细 taxonomy，而是把它留作下一步 layer-specific split analysis。

选择 shared K 的理由不是它在每个单项指标上都最优，而是它在四层中同时满足：seed stability 高、cluster size 不碎、confounder NMI 相对可控，并且可以生成可解释的 original-space evidence。

![K selection summary](../outputs/chronos_multilayer_validation/figures/k_selection_summary.png)

## 4. Layer-wise Validation Summary

| representation | K | silhouette ↑ | stability ↑ | agg NMI ↑ | macro NMI ↓ | frequency NMI ↓ | high-conf macro rate ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `projection` | 6 | 0.137 | 0.976 | 0.504 | 0.077 | 0.094 | 0.967 |
| `layer_0` | 6 | 0.088 | 0.968 | 0.491 | 0.092 | 0.105 | 0.933 |
| `layer_6` | 6 | 0.076 | 0.993 | 0.497 | 0.257 | 0.306 | 0.700 |
| `layer_11` | 6 | 0.170 | 0.992 | 0.572 | 0.169 | 0.186 | 0.933 |

Metric 读法：

| metric | 含义 | 方向 | 注意事项 |
|---|---|---|---|
| `silhouette ↑` | 样本到本 cluster 的紧密度相对其它 cluster 的分离度。 | 越高越好 | 不能单独用来选 K，因为高分可能来自过粗划分或 domain separation。 |
| `stability ↑` | 不同 KMeans random seeds 得到的 labels 的 NMI 平均值。 | 越高越好 | 表示 clustering 对初始化不敏感，但不等于语义正确。 |
| `agg NMI ↑` | KMeans labels 与 AgglomerativeClustering labels 的 NMI。 | 越高越好 | 表示 cluster structure 不太依赖单一聚类算法。 |
| `macro NMI ↓` | cluster labels 与 macro-domain labels 的 NMI。 | 通常越低越好 | 高值提示 domain confounding；若研究 domain-specific concept，则可作为警告而非直接否定。 |
| `frequency NMI ↓` | cluster labels 与采样频率/cadence labels 的 NMI。 | 通常越低越好 | 高值提示 frequency/cadence confounding。 |
| `high-conf macro rate ↑` | cluster × real macro-domain cell 中存在同 cluster 且距离中心足够近的比例。 | 越高越好 | 用于检查原空间 prototype 能否跨真实 macro-domain 复现，不含 Synthetic control。 |

![Layer comparison summary](../outputs/chronos_multilayer_validation/figures/layer_comparison_summary.png)

读法：`projection` 和 `layer_0` 更接近 local patch vocabulary；`layer_6` 与 `layer_11` 通常更稳定，但更容易吸收 domain/frequency/context-style 信息。

## 5. All-cluster Evidence Figures

为避免 cherry-picking，本节每个 representation 都展示 final shared K 下的全部 clusters。也就是说，`K=6` 时每层都展示 `C0-C5`。
质量闸门仍然保留，但它只用于解释每个 cluster 的证据强弱，不用于隐藏结果。

### projection

![projection center nearest](../outputs/chronos_multilayer_validation/figures/projection_main_center_nearest.png)

![projection macro-domain filtered](../outputs/chronos_multilayer_validation/figures/projection_main_macro_domain_filtered.png)

All clusters under this K setting:

| cluster | tier | size | score | macro domains | raw coherence | confounder risk | interpretation status |
|---|---|---:|---:|---:|---:|---:|---|
| `C0` | `main_evidence` | 560 | 0.886 | 4 | 0.739 | 0.452 | candidate concept |
| `C1` | `main_evidence` | 1422 | 1.000 | 5 | 1.000 | 0.255 | candidate concept |
| `C2` | `main_evidence` | 669 | 0.930 | 5 | 0.968 | 0.472 | candidate concept |
| `C3` | `main_evidence` | 683 | 0.937 | 5 | 0.950 | 0.449 | candidate concept |
| `C4` | `main_evidence` | 842 | 0.876 | 5 | 0.503 | 0.359 | candidate concept |
| `C5` | `diagnostic_weak` | 924 | 0.853 | 5 | 0.447 | 0.379 | weak diagnostic |

### layer_0

![layer_0 center nearest](../outputs/chronos_multilayer_validation/figures/layer_0_main_center_nearest.png)

![layer_0 macro-domain filtered](../outputs/chronos_multilayer_validation/figures/layer_0_main_macro_domain_filtered.png)

All clusters under this K setting:

| cluster | tier | size | score | macro domains | raw coherence | confounder risk | interpretation status |
|---|---|---:|---:|---:|---:|---:|---|
| `C0` | `main_evidence` | 878 | 0.979 | 5 | 0.912 | 0.343 | candidate concept |
| `C1` | `diagnostic_weak` | 1088 | 0.847 | 5 | 0.364 | 0.344 | weak diagnostic |
| `C2` | `diagnostic_weak` | 1086 | 0.827 | 5 | 0.278 | 0.208 | weak diagnostic |
| `C3` | `diagnostic_weak` | 196 | 0.477 | 3 | 0.300 | 0.903 | weak diagnostic |
| `C4` | `main_evidence` | 924 | 0.931 | 5 | 0.941 | 0.458 | candidate concept |
| `C5` | `main_evidence` | 928 | 0.939 | 5 | 0.951 | 0.446 | candidate concept |

### layer_6

![layer_6 center nearest](../outputs/chronos_multilayer_validation/figures/layer_6_main_center_nearest.png)

![layer_6 macro-domain filtered](../outputs/chronos_multilayer_validation/figures/layer_6_main_macro_domain_filtered.png)

All clusters under this K setting:

| cluster | tier | size | score | macro domains | raw coherence | confounder risk | interpretation status |
|---|---|---:|---:|---:|---:|---:|---|
| `C0` | `diagnostic_weak` | 881 | 0.671 | 1 | 0.566 | 0.262 | weak diagnostic |
| `C1` | `diagnostic_weak` | 539 | 0.669 | 4 | 0.258 | 0.649 | weak diagnostic |
| `C2` | `diagnostic_weak` | 1377 | 0.799 | 5 | 0.218 | 0.375 | weak diagnostic |
| `C3` | `diagnostic_weak` | 391 | 0.722 | 4 | 1.000 | 0.895 | weak diagnostic |
| `C4` | `diagnostic_confounded` | 721 | 0.440 | 2 | 0.458 | 0.979 | confounded diagnostic |
| `C5` | `diagnostic_weak` | 1191 | 0.824 | 5 | 0.267 | 0.301 | weak diagnostic |

### layer_11

![layer_11 center nearest](../outputs/chronos_multilayer_validation/figures/layer_11_main_center_nearest.png)

![layer_11 macro-domain filtered](../outputs/chronos_multilayer_validation/figures/layer_11_main_macro_domain_filtered.png)

All clusters under this K setting:

| cluster | tier | size | score | macro domains | raw coherence | confounder risk | interpretation status |
|---|---|---:|---:|---:|---:|---:|---|
| `C0` | `diagnostic_weak` | 1987 | 0.806 | 5 | 0.194 | 0.254 | weak diagnostic |
| `C1` | `main_evidence` | 468 | 0.821 | 4 | 0.884 | 0.647 | candidate concept |
| `C2` | `diagnostic_weak` | 421 | 0.755 | 4 | 1.000 | 0.831 | weak diagnostic |
| `C3` | `diagnostic_weak` | 924 | 0.812 | 5 | 0.306 | 0.392 | weak diagnostic |
| `C4` | `main_evidence` | 842 | 0.855 | 5 | 0.486 | 0.392 | candidate concept |
| `C5` | `main_evidence` | 458 | 0.817 | 5 | 0.904 | 0.664 | candidate concept |

## 6. Layer-specific K Check

shared K 用于层间对比；per-layer K 用于检查某一层内部是否存在更细的 model-derived motif/prototype family。这里不替换主结论，也不 cherry-pick：凡是补充的 K setting 都展示该 K 下的全部 clusters。

### layer_6 K=10

`layer_6` 的 per-layer K selection 指向 `K=10`，说明该层在 shared K 之外可能存在更细的 contextual substructure。shared `K=6` 仍然用于 `projection -> layer_0 -> layer_6 -> layer_11` 的横向比较；`K=10` 作为 layer-specific split check。

![layer_6 K10 center nearest](../outputs/chronos_multilayer_validation/figures/layer_6_k10_center_nearest.png)

![layer_6 K10 macro-domain filtered](../outputs/chronos_multilayer_validation/figures/layer_6_k10_macro_domain_filtered.png)

All clusters under this layer-specific K setting:

| cluster | tier | size | score | macro domains | raw coherence | confounder risk | interpretation status |
|---|---|---:|---:|---:|---:|---:|---|
| `C0` | `diagnostic_weak` | 935 | 0.767 | 5 | 0.192 | 0.427 | weak diagnostic |
| `C1` | `main_evidence` | 239 | 0.810 | 4 | 1.000 | 0.724 | candidate concept |
| `C2` | `diagnostic_weak` | 482 | 0.555 | 3 | 0.256 | 0.726 | weak diagnostic |
| `C3` | `diagnostic_confounded` | 179 | 0.260 | 0 | 0.333 | 0.989 | confounded diagnostic |
| `C4` | `diagnostic_weak` | 947 | 0.825 | 4 | 0.272 | 0.294 | weak diagnostic |
| `C5` | `diagnostic_weak` | 363 | 0.546 | 3 | 0.587 | 1.000 | weak diagnostic |
| `C6` | `diagnostic_weak` | 408 | 0.541 | 3 | 0.568 | 0.978 | weak diagnostic |
| `C7` | `diagnostic_weak` | 391 | 0.672 | 2 | 0.282 | 0.361 | weak diagnostic |
| `C8` | `main_evidence` | 556 | 0.927 | 5 | 0.694 | 0.340 | candidate concept |
| `C9` | `main_evidence` | 600 | 0.855 | 5 | 0.502 | 0.400 | candidate concept |

## 7. Diagnostic Evidence and Failure Cases

![Diagnostic failure cases](../outputs/chronos_multilayer_validation/figures/diagnostic_failure_cases.png)

这些 failure cases 是方法的安全阀：如果 cluster 视觉上有形态但 confounder 风险高、macro-domain match 弱或 cluster 太小，就不进入主结论。

## 8. Final Answer for Advisor

当前可以稳健声称：Chronos-2 的 patch representations 在不同层中确实保留并重组 local temporal information；但不同层承担的角色不同。`projection` / `layer_0` 更适合回答 single patch local vocabulary，`layer_6` / `layer_11` 更适合观察 contextual mixing 和 domain/cadence-style 的重组。

当前仍需谨慎：cluster 不是最终 taxonomy；K 是用于 exploratory concept discovery 的 operating point；macro-domain evidence 只能说明跨领域可复现性，不等于真实世界语义 ground truth。若进入 paper 阶段，还需要更大采样、多 seed 数据重采样和人工/领域知识审阅。
