# 上一周导师汇报 PPT 归档说明

> 归档日期：2026-05-20  
> 目的：把上一周围绕 `TimesFM / Chronos / prior-guided motif probe / cross-model validation` 的 PPT 素材标记为历史材料，避免和本周 `Chronos-2-only + layer effect + DTW original-space validation` 主线混用。

## 1. 归档范围

上一周 PPT 材料主要包括：

- `docs/90_advisor_meeting_ppt_material_pack.md`
- `docs/91_advisor_meeting_ppt_draft_archive.md`
- `docs/92_ppt_reproducibility_check.md`
- `outputs/ppt_raw_assets/`
- `outputs/ppt_assets/`
- `outputs/advisor_ppt_package/`

这些材料可以继续作为 backup 或历史记录，但不再作为本周汇报的主线。当前主证据应以：

- `docs/11_chronos_layer_effect_main_report.md`
- `docs/12_distance_metric_ablation_report.md`
- `outputs/chronos_multilayer_validation/`
- `outputs/distance_metric_ablation/`

为准。

## 2. 为什么需要归档

上一周 PPT 的叙事重点是：

- 从 `human-prior motif taxonomy` 到 `model-derived candidate motif families`；
- 对比 `TimesFM-2.5`、`Chronos-2`、`Chronos-2-small`；
- 使用 prior-guided motif probe 帮助解释 hidden-space clusters；
- 展示 cross-model validation 和 negative control。

meeting 后，老师给出的新方向已经更聚焦：

- `Use Chronos-2 only`；
- 看 single patch 是否仍然保留 independent local information；
- 重点比较 `projection / layer_0 / layer_6 / layer_11`；
- early layers 可能更保留 `spike / oscillation / local motif` 等 local information；
- 用 `KMeans center` 和 nearest points 回到 original space 看 cluster 是否有语义；
- 原空间解释需要考虑 DTW，因为 time-shifted shapelet-like patterns 不适合只用 raw Euclidean。

因此，本周不应继续用上一周的 TimesFM / cross-model 图作为主证据，否则容易让汇报问题发散。

## 3. 本周使用规则

上一周材料的使用边界：

- 可以作为 backup：解释我们为什么从 cross-model / prior-probe 转向 Chronos-only。
- 可以作为历史：说明项目路线如何从 broad pilot 收敛到 rigorous mechanism study。
- 不建议放入本周主 PPT：`TimesFM-2.5` triptych、cross-model validation、prior-guided motif probe alignment 等。
- 不应用 prior-guided labels 命名任何主报告 cluster。

本周主 PPT 的证据链：

1. `Chronos-2` 的多层 representation clustering；
2. macro-domain balanced sampling；
3. representation-space K selection；
4. all-cluster center-nearest raw patch inspection；
5. layer-wise metrics 与 confounder audit；
6. two-space distance principle；
7. DTW-aware original-space validation；
8. failure cases 和谨慎结论。

## 4. 对外口径

推荐说法：

> 上一周我们做了 broad pilot，确认 TSFM patch-token space 有可见结构，但也暴露了 prior-guided labels 和 cross-model transfer 的不稳定性。本周我们收敛到老师建议的 Chronos-2-only setting，系统比较 projection、early layer、middle layer 和 top layer，并用 DTW-aware original-space validation 检查哪些 representation neighborhoods 真的像 shapelet-like motif/prototype families。

避免说法：

- “上一周结果已经证明了 final taxonomy。”
- “prior-guided motif 是 ground truth。”
- “KMeans cluster 直接就是 motif。”
- “DTW 要替代 representation-space clustering。”

