# PPT 内容幻觉检测与代码可复现性检查

检查对象：`docs/ppt_material_pack_advisor_meeting.md` 及其引用的 `outputs/ppt_raw_assets/` 图片和配套 JSON。

检查时间：2026-05-12。

## 结论

当前 PPT 材料的核心叙事没有发现明显“无数据支撑”的 hallucination。Slide 5-12 的主要结论均能在现有 JSON summary 或 raw asset summary 中找到直接支撑。

但有一个重要可复现性风险：`scripts/build_ppt_raw_assets.py` 在本机可以运行，但依赖若干未纳入 git 的中间图片/summary，clean clone 后不能仅凭已提交文件重建全部 PPT raw assets。

## 内容一致性检查

### 通过项

- Slide 5 的 A/B/C panel 分别对应：
  - KMeans candidate clusters；
  - prior motif probe coloring；
  - patch index coloring。
  图片路径存在，语义与 `fig_timesfm_clustering_triptych_labeled.png` 一致。

- Slide 6 的 `domain-balanced display subset` 表述成立。`domain_balanced_falling_family_summary.json` 显示：
  - full prototype bank: 40；
  - display subset: 18；
  - per-domain cap: 5；
  - full bank 仍由 weather 主导，因此文案中保留 confounder risk 是合理的。

- Slide 7 的 `model-native representative examples` 表述成立。`model_native_candidate_patches_summary.json` 包含：
  - TimesFM-2.5 layer 10 cluster 5；
  - Chronos-2 layer 11 cluster 9；
  - Chronos-2-small layer 5 cluster 13。
  注意：它们是 native hidden-space clustering examples，不是 full controlled retrieval validation。

- Slide 8 的 “not every cluster is a motif family” 表述成立。`cluster_outcome_gallery_summary.json` 覆盖 clean candidate、broad mixed control、event-like control、position-confounded cluster、confounded cluster。

- Slide 11 的 lineage claim 成立。`input_embedding_ablation_summary.json` 显示：
  - TimesFM hidden patch-index NMI = 0.2998，position effect 明显；
  - Chronos-2 hidden patch-index NMI = 0.0064；
  - Chronos-2-small hidden patch-index NMI = 0.0069；
  - Chronos hidden states 的 domain/frequency NMI 明显高于 patch-index NMI。

- Slide 12 的 cross-model sanity check 成立。`cross_model_validation_summary.json` 显示 `strong_falling_transition` 在 TimesFM-2.5、Chronos-2、Chronos-2-small 中均高于 matched random baseline。

## 仍需谨慎的表述

- 不要说 “model-derived motif taxonomy 已经完成”。当前只能说 `model-derived candidate motif families` 或 `taxonomy v1 pilot evidence`。

- 不要说 Slide 7 已经证明 Chronos-native taxonomy。它只证明 Chronos-2 / Chronos-2-small 自身 hidden-space clustering 中存在可展示的 transition-like representatives。

- 不要把 Slide 6 说成 full domain-balanced prototype bank。它是 per-domain capped display subset，full bank 仍存在 weather skew。

- 不要用 high shape correlation alone 证明 motif。Slide 9/10 显示 first-patch artifact 也可能有高 shape coherence，因此必须同时讲 confounder audit。

## 可复现性检查

### 已通过

以下检查通过：

```bash
.venv/bin/python -m py_compile scripts/build_ppt_raw_assets.py scripts/build_ppt_assets.py scripts/run_cross_model_concept_validation.py scripts/run_input_embedding_ablation.py scripts/run_representation_lineage.py scripts/build_lineage_cards.py scripts/build_patch_cluster_shape_summary.py scripts/build_poster_assets.py
```

Markdown 图片链接检查通过：`docs/ppt_material_pack_advisor_meeting.md` 中 23 个本地图片/JSON 链接均存在。

核心 claim 的 JSON consistency check 通过，没有发现硬冲突。

### 可复现性风险

`scripts/build_ppt_raw_assets.py` 当前依赖以下未追踪但本机存在的中间文件：

- `outputs/figures/second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_pca_clusters.png`
- `outputs/figures/second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_pca_taxonomy_v0.png`
- `outputs/figures/second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_pca_patch_index.png`
- `outputs/figures/second_pilot/second_pilot_timesfm_2_5_layer_10_domain_balanced_prototype_panel.png`
- `outputs/figures/second_pilot/second_pilot_chronos_2_layer_11_domain_balanced_prototype_panel.png`
- `outputs/figures/second_pilot/second_pilot_chronos_2_small_layer_5_domain_balanced_prototype_panel.png`
- `outputs/input_embedding_ablation/input_embedding_ablation_summary.json`

这些文件被 `.gitignore` 的 `outputs/**` 规则忽略。也就是说：

- 当前仓库已经提交了最终 PPT raw assets，因此汇报使用不受影响；
- 但 clean clone 后，若只运行 `scripts/build_ppt_raw_assets.py`，会因为缺少中间文件而失败；
- 要完全复现，需要先运行 second pilot 和 input embedding ablation，或把这些中间 evidence figures / summaries 加入版本管理。

## 建议

明天汇报前可以使用当前 PPT raw assets。它们与文案基本一致。

若后续要把仓库交给别人复现，建议二选一：

1. 将 `outputs/figures/second_pilot/*.png` 和 `outputs/input_embedding_ablation/input_embedding_ablation_summary.json` 加入 git 例外。
2. 改写 `build_ppt_raw_assets.py`，使其只依赖已追踪的 JSON 和 raw patch arrays，而不是裁剪 second-pilot PNG。

当前更适合明天汇报的说法：

> We have reproducible final evidence assets for the meeting, but the raw-asset generation pipeline still depends on intermediate second-pilot figures that should be versioned or regenerated in the next cleanup.
