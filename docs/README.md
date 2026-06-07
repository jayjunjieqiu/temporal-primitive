# Documentation Index

本目录按研究阶段和证据层级重新命名。当前主线已经从 **Chronos-2-only representation interpretability** 转向 **Chronos-Bolt-based patch-token motif/prototype discovery**。

转向原因：进一步检查后发现，`Chronos-2` 的 `projection` / `input_patch_embedding` 输入并不是 pure value-only patch，而是 `[time encoding, normalized patch values, patch mask]`。这会混淆我们对 single-patch local information 的解释。因此 Chronos-2 相关结果统一归档为历史探索材料，后续默认使用 `Chronos-Bolt` 重新验证 patch-level temporal primitive / motif prototype 问题。

## Current Main Reports

- [99_chronos2_archive_and_chronos_bolt_pivot.md](99_chronos2_archive_and_chronos_bolt_pivot.md)  
  当前路线转向说明。解释为什么 Chronos-2 证据需要归档，以及为什么后续默认改用 Chronos-Bolt。

## Archived Chronos-2 Evidence

- [11_chronos_layer_effect_main_report.md](11_chronos_layer_effect_main_report.md)  
  已归档。回答 Chronos-2 的 `projection / layer_0 / layer_6 / layer_11` 是否保留 single-patch local information，以及这些 representation clusters 是否能回到原空间解释为 candidate motif/prototype families。由于 Chronos-2 projection 包含 explicit time encoding，不再作为后续主线证据。

- [12_distance_metric_ablation_report.md](12_distance_metric_ablation_report.md)  
  已归档的方法补充报告。说明 two-space distance principle：`representation space` 用 Euclidean/KMeans 做 candidate discovery，`original time-series space` 用 DTW 做 prototype validation 和 controlled retrieval audit。该方法原则保留，但 Chronos-2 结果不再作为主线。

## Project Setup And Rules

- [00_narrative_rules.md](00_narrative_rules.md)  
  仓库写作和术语规则。所有 proposal、report、figure caption 默认遵循这里的 Yuxuan Liang / CityMind Lab 叙事体系。

- [00_local_model_download.md](00_local_model_download.md)  
  本地模型权重下载默认流程，使用 `hf-mirror` + `hfd.sh`，并显式 unset HTTP proxy。

- [01_research_proposal.md](01_research_proposal.md)  
  当前研究 proposal。定位为 `model-derived motif taxonomy discovery protocol`，不是 final taxonomy claim。

- [02_feasibility_model_loading_report.md](02_feasibility_model_loading_report.md)  
  早期可行性报告：本地权重加载、hidden-state extraction、motif labeling prototype 的可行性检查。

## Discovery And Validation History

- [03_external_weak_motif_labeling_report.md](03_external_weak_motif_labeling_report.md)  
  早期 human-prior / weak motif label 的调研与操作化尝试。当前只作为 appendix-level background，不进入主证据链。

- [04_discover_first_strategy_report.md](04_discover_first_strategy_report.md)  
  从 prior-first 转向 discover-first, name-second 的策略报告。

- [05_second_pilot_discovery_report.md](05_second_pilot_discovery_report.md)  
  第二轮 data/model-centered pilot。证明 representation clusters 存在，但也暴露 domain/frequency/position confounding。

- [06_cluster_card_review_report.md](06_cluster_card_review_report.md)  
  cluster cards 与 controlled retrieval 的人工审阅报告。

- [07_cluster_level_controlled_validation_report.md](07_cluster_level_controlled_validation_report.md)  
  multi-query、position/frequency-aware 的 cluster-level controlled validation。

- [08_model_derived_taxonomy_pilot_report.md](08_model_derived_taxonomy_pilot_report.md)  
  早期 taxonomy v1 pilot，保留为历史材料。当前主线已收敛到 Chronos-2-only。

- [09_cross_model_validation_archive_report.md](09_cross_model_validation_archive_report.md)  
  TimesFM/Chronos cross-model validation 历史材料。当前不作为主线证据。

- [10_input_embedding_ablation_report.md](10_input_embedding_ablation_report.md)  
  input embedding / tokenizer-projection 与 hidden states 的对比分析。

## Appendix And Meeting Materials

- [93_last_week_ppt_archive_manifest.md](93_last_week_ppt_archive_manifest.md)  
  上一周 PPT 素材归档说明。把 TimesFM/cross-model/prior-probe 相关材料标记为历史和 backup，不再作为本周主线证据。

- [94_weekly_advisor_ppt_draft.md](94_weekly_advisor_ppt_draft.md)  
  已归档的导师汇报 PPT 草稿。主线曾切换为 Chronos-2-only layer-wise validation、two-space distance principle 和 DTW-aware original-space validation。

- [95_dynamical_systems_story_ppt_draft.md](95_dynamical_systems_story_ppt_draft.md)  
  已归档的故事版 PPT 草稿。参考 dynamical systems perspective，把 Chronos-2 结果组织成 `Dynamical Prototype State Hypothesis`，并给出逐页 figure redraw plan。

- [96_notion_dynamical_story_report.md](96_notion_dynamical_story_report.md)  
  已归档的 Notion 版周汇报报告。直接使用 Chronos-2 实验原图，回答 single patch local information、KMeans center-nearest examples 和新 TSFM 设计启发三个问题。

- [97_tsne_then_kmeans_cluster_report.md](97_tsne_then_kmeans_cluster_report.md)  
  已归档。t-SNE 降维后 KMeans 聚类结果与原型图报告。只看 Chronos-2 `projection`、`layer_6`、`layer_11`。

- [98_tsne_domain_label_report.md](98_tsne_domain_label_report.md)  
  已归档。t-SNE 降维后直接打上 source-domain / macro-domain 标签的诊断报告。

- [97_pca_then_kmeans_cluster_report.md](97_pca_then_kmeans_cluster_report.md)  
  已归档。PCA-space KMeans 聚类结果与原型图报告。作为 t-SNE 的线性全局结构对照。

- [98_pca_domain_label_report.md](98_pca_domain_label_report.md)  
  已归档。PCA 降维后直接打上 source-domain / macro-domain 标签的诊断报告。

- [80_external_weak_motif_probe_sanity_check.md](80_external_weak_motif_probe_sanity_check.md)  
  external weak motif probe 的 sanity check。当前结论是它不能作为 paper-level 主证据。

- [90_advisor_meeting_ppt_material_pack.md](90_advisor_meeting_ppt_material_pack.md)  
  上一周给导师汇报用的 PPT 素材包、排版建议和口头讲法。当前作为历史材料和 backup。

- [91_advisor_meeting_ppt_draft_archive.md](91_advisor_meeting_ppt_draft_archive.md)  
  旧 PPT 草稿归档。

- [92_ppt_reproducibility_check.md](92_ppt_reproducibility_check.md)  
  PPT 内容幻觉检测与代码可复现性检查归档。

注意：`90_*` 到 `98_*` 多为 meeting/archive materials，部分引用的旧图片来自 ignored `outputs/` 目录，clean clone 中可能不存在。当前路线决策以 `99_*` 为准；后续主证据应围绕 Chronos-Bolt 重新生成。

## Naming Rule

- `00_*`: repository rules and setup notes
- `01_*`: proposal
- `02_*`: feasibility
- `03_*` to `10_*`: discovery and validation history
- `11_*` to `12_*`: archived Chronos-2 main evidence and method ablation
- `80_*`: appendix / weak-label sanity checks
- `90_*`: meeting and PPT materials
- `99_*`: route pivot / archive decisions
