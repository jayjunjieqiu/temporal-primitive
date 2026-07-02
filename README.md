# Temporal Primitive / TSFM Patch Representation Study

This repository studies what patch tokens in Time Series Foundation Models (TSFMs) learn. It backs the
Review article *"Decoding Dynamical Systems: Foundation Models for Time Series and Beyond"* (Nature
Reviews Computing, submitted). The main analysis uses **Chronos-Bolt** patch representations with a
two-space validation protocol:

- use Euclidean geometry in the **representation space** to discover model-derived patch-token neighbourhoods;
- return to the **original time-series space** to check whether those neighbourhoods correspond to coherent motif/prototype families.

> Route note: the project pivoted from the earlier Chronos-2-only line to Chronos-Bolt (see
> [docs/99](docs/99_chronos2_archive_and_chronos_bolt_pivot.md)); the Chronos-2 reports (`11_`, `12_`,
> `90_`–`98_`) are archived historical material.

## Start Here

- **Reproduce the paper's figures/tables → [docs/18_paper_reproduction.md](docs/18_paper_reproduction.md)**
- Documentation index: [docs/README.md](docs/README.md)
- Writing and terminology rules: [docs/00_narrative_rules.md](docs/00_narrative_rules.md)
- Model-weight download: [docs/00_local_model_download.md](docs/00_local_model_download.md)
- Main-figure report: [docs/15_bolt_main_figure_report.md](docs/15_bolt_main_figure_report.md) · publication panels: [docs/17_publication_figures.md](docs/17_publication_figures.md)

## Current Claim Level

The evidence supports a **model-derived motif/prototype discovery protocol** and illustrative
observations about Chronos-Bolt patch representations. It does not claim a final motif taxonomy: clusters
remain **candidate motif/prototype families** until audited (original-space inspection + DTW-aware
controlled retrieval + domain/frequency/position confounder audit, see
[docs/00_narrative_rules.md](docs/00_narrative_rules.md) §7).
