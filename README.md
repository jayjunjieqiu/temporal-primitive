# Temporal Primitive / TSFM Patch Representation Study

This repository studies what patch tokens in Time Series Foundation Models (TSFMs) learn. The current active direction is **Chronos-2-only representation interpretability** with a two-space validation protocol:

- use Euclidean geometry in `Chronos-2 representation space` to discover model-derived patch-token neighborhoods;
- use DTW-aware validation in `original time-series space` to decide whether those neighborhoods correspond to coherent motif/prototype families.

## Start Here

- Documentation index: [docs/README.md](docs/README.md)
- Writing and terminology rules: [docs/00_narrative_rules.md](docs/00_narrative_rules.md)
- Research proposal: [docs/01_research_proposal.md](docs/01_research_proposal.md)
- Current main report: [docs/11_chronos_layer_effect_main_report.md](docs/11_chronos_layer_effect_main_report.md)
- Distance-method ablation: [docs/12_distance_metric_ablation_report.md](docs/12_distance_metric_ablation_report.md)

## Current Claim Level

The current evidence supports a **model-derived motif/prototype discovery protocol** and several candidate observations about Chronos-2 layer behavior. It does not claim a final motif taxonomy.
