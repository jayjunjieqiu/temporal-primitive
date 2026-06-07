# Reference-style spectral illustration

This folder contains the modular drawing plan and generated assets for the reference-style spectral illustration.

- `docs/plan.md`: current design specification.
- `scripts/draw_21_central_representation_atlas.py`: draws Module 2.1 / Module A.
- `scripts/cuml_tsne_helper.py`: runs GPU t-SNE inside the `rapids-tsne` conda env.
- `assets/central_representation_atlas_layer0_clean.png`: generated clean central representation atlas.

The current assets use archived Chronos-2 pilot evidence. The final clean version should be rerun with Chronos-Bolt.

## Current Module 2.1 Command

Default aligned K=6 figure:

```bash
python figure_projects/reference_style_spectral_illustration/scripts/draw_21_central_representation_atlas.py \
  --windows-per-dataset 100 \
  --seed 47 \
  --batch-size 128 \
  --selection-mode source_domain_balanced \
  --max-per-source-domain 700 \
  --max-tsne-points 7700 \
  --pca-dim 30 \
  --k 6 \
  --tsne-perplexity 40 \
  --tsne-max-iter 1000 \
  --reducer cuml_tsne \
  --cuml-init random \
  --cluster-space pca
```

Old-result aligned K=15 comparison:

```bash
python figure_projects/reference_style_spectral_illustration/scripts/draw_21_central_representation_atlas.py \
  --windows-per-dataset 100 \
  --seed 47 \
  --batch-size 128 \
  --selection-mode source_domain_balanced \
  --max-per-source-domain 700 \
  --max-tsne-points 7700 \
  --pca-dim 30 \
  --k 15 \
  --tsne-perplexity 40 \
  --tsne-max-iter 1000 \
  --reducer cuml_tsne \
  --cuml-init random \
  --cluster-space pca \
  --output figure_projects/reference_style_spectral_illustration/assets/central_representation_atlas_layer0_aligned_k15_pca_cluster_cuml_tsne.png \
  --summary-output figure_projects/reference_style_spectral_illustration/assets/central_representation_atlas_layer0_aligned_k15_pca_cluster_cuml_tsne_summary.json
```

Implementation notes:

- `cuml_tsne` uses RAPIDS/cuML in conda env `rapids-tsne`.
- The main script still handles sampling, cached Chronos-2 `layer_0` embeddings, KMeans, and Matplotlib drawing.
- The GPU helper only receives PCA-preprocessed embeddings and writes the t-SNE coordinates back to cache.
- Current aligned full run uses 7700 cached patches and takes about 19-22 seconds end to end after embedding cache exists; cuML's t-SNE fit itself is about 1 second.
- KMeans is computed in PCA space by default (`--cluster-space pca`); t-SNE is visualization only. This matches the earlier layer-effect plots more closely than clustering directly in t-SNE space.
- Intermediate outputs for Module 2.2 / 2.3 are saved under `assets/intermediates/<figure-stem>/`, including raw patches, PCA coordinates, t-SNE coordinates, cluster labels, center-nearest indices, and metadata.
- `perplexity=40` is retained for old-result alignment. The current RAPIDS/cuML version still prints a nearest-neighbor warning even with `n_neighbors > 3 * perplexity`; the actual recorded helper parameter is saved in each summary JSON.

## Current Module 2.2 Command

Module 2.2 uses the aligned K=6 intermediate outputs from Module 2.1. It does not rerun Chronos-2.

```bash
python figure_projects/reference_style_spectral_illustration/scripts/draw_22_patch_stack_exemplar_cards.py --version-tag v4
```

Outputs:

- `assets/patch_stack_exemplar_cards_layer0_k6/patch_stack_exemplar_cards_layer0_k6_selected_v4.png`: selected illustrative patch-stack evidence plate.
- `assets/patch_stack_exemplar_cards_layer0_k6/patch_stack_exemplar_cards_layer0_k6_all_clusters_v4.png`: all-cluster patch-stack audit plate.
- `assets/patch_stack_exemplar_cards_layer0_k6/cards/C*_patch_stack_card_v4.png`: individual cards for manual assembly, including C6.
- `assets/patch_stack_exemplar_cards_layer0_k6/patch_stack_exemplar_cards_layer0_k6_summary_v4.json`: selected clusters, visual scores, examples, and metadata.
- `assets/patch_stack_exemplar_cards_layer0_k6/patch_stack_exemplar_cards_layer0_k6_data_v4.npz`: reusable arrays.

Interpretation notes:

- Module 2.2 is illustrative region evidence, not a final prototype grid.
- It automatically selects five visually informative clusters from C1-C6 using center-nearest coherence, shape energy, and spectral peakiness.
- Each selected card shows `Raw patch stack`, `First difference stack`, and `Power spectrum stack`.
- Rows are top-24 center-nearest examples in PCA clustering space.
- This avoids duplicating Module 2.3: 2.2 shows region-level sample texture; 2.3 summarizes all clusters with prototype descriptors.
- C6 is omitted from the selected illustrative plate because its visual score is low, but it is included in the all-cluster plate and individual card outputs.

## Current Module 2.3 Command

Module 2.3 summarizes all K=6 clusters. It should be used as the rigorous bottom descriptor/prototype grid.

```bash
python figure_projects/reference_style_spectral_illustration/scripts/draw_23_cluster_descriptor_grid.py
```

Outputs:

- `assets/cluster_descriptor_grid_layer0_k6.png`: all-cluster prototype/descriptor grid.
- `assets/cluster_descriptor_grid_layer0_k6_summary.json`: center-nearest metadata, cluster sizes, and top domains.

Interpretation notes:

- Each column is one KMeans cluster, C1-C6.
- Rows show z-normalized raw patch, first difference, and power spectrum.
- Thick lines are KMeans-center-nearest examples in PCA clustering space.
- Shaded bands are cluster-level IQR.
- C6 is retained even if visually weak/flat because Module 2.3 is the full cluster-level audit, not an illustrative selection.
