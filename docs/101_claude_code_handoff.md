# Claude Code Handoff: Reference-style Temporal Primitive Figure Project

更新时间：2026-06-07  
当前分支：`main`  
最近提交：`3fcf872 Add reference-style temporal primitive figures`

## 1. 当前任务背景

本仓库研究 TSFM patch-token representation 是否形成可解释的 `model-learned temporal primitive-like structures`。当前主线不是继续做完整 PPT，而是围绕 `reference_pic.jpg` 拆成模块化 figure，方便手动拼接成汇报图。

当前 figure project 位于：

```text
figure_projects/reference_style_spectral_illustration/
```

这个项目使用已有 `Chronos-2 layer_0` archived pilot 结果。注意：根据 `docs/00_narrative_rules.md` 和 `docs/99_chronos2_archive_and_chronos_bolt_pivot.md`，Chronos-2 结果只作为 archived diagnostic evidence，后续 clean analysis 应迁移到 Chronos-Bolt。

## 2. 重要叙事边界

写作和图注必须遵守：

```text
docs/00_narrative_rules.md
```

核心表述：

- 可以说 `model-learned temporal primitive-like structures`。
- 不要说已经发现最终 `temporal language` 或最终 `motif taxonomy`。
- `motif taxonomy v0` 是 human-prior / shapelet-inspired probe，不是 ground truth。
- 当前 figure 只说明 representation space 中存在 cluster-like structures，并能在 original time-series space 中找到一些 shape / spectral evidence。
- 对 Chronos-2 必须标注 archived pilot，因为它的 projection/input token 包含 time encoding，不是 pure value-only patch token。

## 3. 最近完成的内容

### 3.1 Module 2.1: Central Representation Atlas

脚本：

```text
figure_projects/reference_style_spectral_illustration/scripts/draw_21_central_representation_atlas.py
figure_projects/reference_style_spectral_illustration/scripts/cuml_tsne_helper.py
```

关键改动：

- 支持 GPU t-SNE：`--reducer cuml_tsne`。
- 使用 conda env：`rapids-tsne`。
- 严谨对齐旧图逻辑：

```text
Chronos-2 layer_0 embedding
-> StandardScaler
-> PCA(dim=30)
-> KMeans in PCA space
-> GPU t-SNE only for visualization
```

当前默认 aligned K=6 图：

```text
figure_projects/reference_style_spectral_illustration/assets/central_representation_atlas_layer0_clean.png
figure_projects/reference_style_spectral_illustration/assets/central_representation_atlas_layer0_clean_summary.json
```

旧结果对齐 K=15 图：

```text
figure_projects/reference_style_spectral_illustration/assets/central_representation_atlas_layer0_aligned_k15_pca_cluster_cuml_tsne.png
figure_projects/reference_style_spectral_illustration/assets/central_representation_atlas_layer0_aligned_k15_pca_cluster_cuml_tsne_summary.json
```

汇报模块用 K=6 图：

```text
figure_projects/reference_style_spectral_illustration/assets/central_representation_atlas_layer0_aligned_k6_pca_cluster_cuml_tsne.png
figure_projects/reference_style_spectral_illustration/assets/central_representation_atlas_layer0_aligned_k6_pca_cluster_cuml_tsne_summary.json
```

复现命令见：

```text
figure_projects/reference_style_spectral_illustration/README.md
```

注意：

- `perplexity=40` 是为了和旧图对齐。
- 当前 RAPIDS/cuML 会打印 nearest-neighbor warning，即使 summary 记录 `n_neighbors=121`，这是已知环境行为。
- t-SNE 坐标只做 visualization；cluster labels 来自 PCA space。

### 3.2 Module 2.2: Patch-stack Exemplar Cards

用户指出旧版 2.2 画 raw line cards 会和 2.3 重复，而且 1D power spectrum 太 trivial。已改成 patch-stack style evidence。

脚本：

```text
figure_projects/reference_style_spectral_illustration/scripts/draw_22_patch_stack_exemplar_cards.py
```

当前推荐输出使用 `v4`，不要再读旧 `v2/v3`：

```text
figure_projects/reference_style_spectral_illustration/assets/patch_stack_exemplar_cards_layer0_k6/patch_stack_exemplar_cards_layer0_k6_selected_v4.png
figure_projects/reference_style_spectral_illustration/assets/patch_stack_exemplar_cards_layer0_k6/patch_stack_exemplar_cards_layer0_k6_all_clusters_v4.png
figure_projects/reference_style_spectral_illustration/assets/patch_stack_exemplar_cards_layer0_k6/patch_stack_exemplar_cards_layer0_k6_summary_v4.json
figure_projects/reference_style_spectral_illustration/assets/patch_stack_exemplar_cards_layer0_k6/patch_stack_exemplar_cards_layer0_k6_data_v4.npz
```

单卡输出：

```text
figure_projects/reference_style_spectral_illustration/assets/patch_stack_exemplar_cards_layer0_k6/cards/C1_patch_stack_card_v4.png
...
figure_projects/reference_style_spectral_illustration/assets/patch_stack_exemplar_cards_layer0_k6/cards/C6_patch_stack_card_v4.png
```

设计含义：

- 2.2 是 illustrative region evidence，不是 prototype summary。
- 每张卡展示：

```text
Raw patch stack | First difference stack | Power spectrum stack
```

- 每行是 PCA clustering space 中离 KMeans center 最近的 top-24 patches。
- `selected_v4` 自动选择 visual score 前 5 个 cluster，当前省略 C6。
- `all_clusters_v4` 和 `cards/C6...v4.png` 包含 C6。

关于 C6：

- C6 看起来空白/弱，不是缓存或绘图错误。
- 它的 center-nearest set 大量是 near-flat / low-information patches，z-normalization 后接近 0。
- 因此 selected illustrative plate 省略 C6，但 all-cluster audit 和单卡中保留。

### 3.3 Module 2.3: Cluster Descriptor Grid

脚本：

```text
figure_projects/reference_style_spectral_illustration/scripts/draw_23_cluster_descriptor_grid.py
```

输出：

```text
figure_projects/reference_style_spectral_illustration/assets/cluster_descriptor_grid_layer0_k6.png
figure_projects/reference_style_spectral_illustration/assets/cluster_descriptor_grid_layer0_k6_summary.json
```

设计含义：

- 2.3 是严谨的 all-cluster prototype / descriptor grid。
- 每列一个 cluster：C1-C6。
- 三行分别是：

```text
z-normalized raw patch
first difference
power spectrum
```

- 粗线是 PCA-space KMeans center-nearest example。
- 阴影是 cluster-level IQR。
- C6 即使弱也保留，因为 2.3 是完整 audit，不是 illustrative selection。

## 4. 重要操作细节

### 4.1 图片缓存问题

用户提醒：重复读取同一路径图片可能看到缓存旧图。检查图片时尽量：

- 输出新文件名，如 `v4`、`v5`；
- 或者确认 viewer 不读缓存；
- 不要反复用同一路径判断新图。

### 4.2 已清理内容

为了完成 commit，清理过：

```text
figure_projects/reference_style_spectral_illustration/cache/
figure_projects/reference_style_spectral_illustration/assets/intermediates/
outputs/ 下若干 ignored generated outputs
```

这些不在 git 中。若需要重跑 2.1/2.2/2.3，脚本会重新生成所需 intermediate/cache。

### 4.3 磁盘空间

当前 `/data` 非常满。最近检查只有约 `128M` 可用：

```text
/dev/md0  11T  11T  128M  100% /data
```

后续运行大脚本前必须先清理空间，尤其避免重新生成大量 embedding cache。

不要随意删除模型权重，除非用户明确同意。当前模型权重目录在 `.gitignore` 中：

```text
chronos-2/
chronos-2-small/
timesfm-2.5-200m-pytorch/
```

### 4.4 Git 状态

当前最近提交：

```bash
git log --oneline -1
# 3fcf872 Add reference-style temporal primitive figures
```

提交时没有加入：

- `figure_projects/**/cache/**`
- `figure_projects/**/assets/intermediates/**`
- old v2/v3 scratch figures
- smoke figures
- model weights

相关 `.gitignore` 已更新。

## 5. 推荐 Claude Code 接手步骤

### Step 1: 先读这些文件

```text
docs/00_narrative_rules.md
docs/100_reference_style_spectral_illustration_plan.md
figure_projects/reference_style_spectral_illustration/README.md
figure_projects/reference_style_spectral_illustration/docs/plan.md
```

### Step 2: 不要先重跑实验

优先检查已经提交的图：

```text
central_representation_atlas_layer0_clean.png
patch_stack_exemplar_cards_layer0_k6_selected_v4.png
patch_stack_exemplar_cards_layer0_k6_all_clusters_v4.png
cluster_descriptor_grid_layer0_k6.png
```

### Step 3: 如果继续画图

建议下一步做：

1. 生成 Module 2.4 callout link overlay，把 2.1 central map 和 2.2 cards 连接起来。
2. 或者制作一个 final assembly canvas，手动排版：

```text
top / side: selected_v4 patch-stack cards
center: central_representation_atlas_layer0_clean.png
bottom: cluster_descriptor_grid_layer0_k6.png
```

但要注意：用户之前明确说不想要过度 HTML/PPT 化的整图，更偏好模块化原始图片，方便手动拼。

### Step 4: 如果要改 2.2

当前 2.2 selected plate 省略 C6；如果用户要求 all clusters，则用：

```text
patch_stack_exemplar_cards_layer0_k6_all_clusters_v4.png
```

不要直接说 C6 缺失。正确解释是：

> C6 是 low-information / near-flat region，因此不适合作为 illustrative selected card，但它在 all-cluster audit 和 2.3 中保留。

## 6. 复现命令

### 6.1 Module 2.1 aligned K=6

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

### 6.2 Module 2.2 v4

```bash
python figure_projects/reference_style_spectral_illustration/scripts/draw_22_patch_stack_exemplar_cards.py --version-tag v4
```

### 6.3 Module 2.3

```bash
python figure_projects/reference_style_spectral_illustration/scripts/draw_23_cluster_descriptor_grid.py
```

## 7. 已知风险

- 当前 figure 基于 Chronos-2 archived pilot，不应作为最终 clean mechanism claim。
- t-SNE geometry 受 seed / backend / perplexity 影响，不能把 t-SNE 边界当作真实几何边界。
- KMeans 在 PCA space 中做，t-SNE 只是 view；这点必须在图注或说明中保留。
- 2.2 selected plate 是 illustrative，不是完整 evidence。
- 2.3 才是 all-cluster descriptor audit。
- C6 weak/flat 是结果，不是图没画出来。
- `/data` 空间极低，重跑前必须清理。
