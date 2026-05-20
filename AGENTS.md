# Repository Writing Rules

本仓库所有 proposal、poster、report、figure caption 和实验解释，默认使用 `docs/00_narrative_rules.md` 中定义的话语体系。

核心原则：

- 用 Yuxuan Liang / CityMind Lab 熟悉的 `Time Series Foundation Models (TSFMs)`、`Spatio-Temporal Foundation Models (STFMs)`、`cross-domain generalization`、`data heterogeneity`、`shared representations`、`temporal commonalities`、`motif taxonomy`、`prototype`、`shapelet-driven explanation`、`scaling laws`、`OOD generalization` 来讲故事。
- 本项目采用双层 taxonomy 叙事：`motif taxonomy v0` 是 human-prior / shapelet-inspired probe，`model-derived motif taxonomy v1` 是我们从 TSFM representation space 中发现并审计的候选结果。
- `temporal primitives` 可以恢复使用，但应指向 `patch-level temporal primitives`、`motif prototypes` 或 `shapelet-like local patterns`；不要把它写成未经验证的最终语义。
- `时序语言 / temporal language` 可以作为引入句、副标题或讨论框架，用来连接 motif taxonomy 与 TSFM token space；避免直接宣称已经发现完整“时序语言”。
- 写中文报告时保留关键英文技术词，例如 `TSFM`、`STFM`、`patch token`、`tokenization`、`representation`、`motif taxonomy`、`prototype`、`shapelet`、`cross-domain`、`OOD`、`domain confusion`、`position confounding`、`controlled retrieval`。
- 对 cluster 的解释必须谨慎：先称为 `candidate motif/prototype family` 或 `model-derived motif cluster`。只有经过 original-space inspection、controlled retrieval、domain/frequency/position confounder audit 后，才可以临时进入 `motif taxonomy v1`。
- 本项目的主贡献表述是：提出一个面向 TSFM patch tokens 的 `model-derived motif taxonomy discovery protocol`，并用它诊断 patch-level shared temporal knowledge 与 confounder artifacts。
- 不要直接宣称发现了最终 taxonomy。当前更稳妥的表述是：给出 `model-derived motif taxonomy v1 pilot` 的候选证据。

写作前先阅读：

- `docs/00_narrative_rules.md`
- `docs/01_research_proposal.md`
- `docs/05_second_pilot_discovery_report.md`
- `docs/06_cluster_card_review_report.md`
- `docs/07_cluster_level_controlled_validation_report.md`
- `docs/11_chronos_layer_effect_main_report.md`
