# Chronos-Bolt main figure：raw-only patch-stack cards + domain-balanced prototype panel

更新时间：2026-06-07  
模型：**Chronos-Bolt-base**（clean 路线）  
脚本：`scripts/build_bolt_main_figure.py` · `scripts/chronos_bolt_backbone.py`

advisor 反馈：main 图**不要 first-difference / power-spectrum 行，只留 raw patch stack**；
**加最深层**；下面用**分 domain 的 prototype example** 那张。已整迁到 clean Chronos-Bolt。

modular PNG（用户偏好手动拼接，不自动合成整图），都在
`outputs/figures/bolt_main_figure/`：

- `bolt_patch_stack_cards_layer0.png` —— layer_0（shallow）的 raw-only patch-stack cards。
- `bolt_patch_stack_cards_layer11.png` —— layer_11（deepest）的 raw-only cards。
- `bolt_cluster_maps.png` —— **中间 plate**（2×2）：每行一个 depth（layer_0 / layer_11），
  **左=模型 KMeans cluster，右=human motif taxonomy v0**（shapelet-inspired probe，9 类，
  *不是* ground truth）。左右共享同一套 t-SNE 点，只着色不同，方便对比"模型 cluster vs 人工
  motif"。KMeans 在 PCA(30) space 完成，t-SNE（perplexity=40）只做可视化（不参与聚类）。
  可见 impulse_spike 小岛在模型 cluster 与 v0 motif 两侧对应；中心大片被 v0 标为
  mixed_uncertain（probe 粗粒度，如实展示）。
- `bolt_cross_domain_prototype_panel_layer0.png` —— layer_0 的 **cross-domain** prototype
  example panel：每行一个 cluster，每列是一个**不同 macro domain** 里离 cluster 中心最近的最佳
  代表。强调同一 shape family 跨域复用（如 U 形/振荡/下降/impulse 各跨 5–6 个域）。
- `bolt_cross_domain_prototype_panel_layer11.png` —— layer_11 版（contextualized）。

`k=8`（见下节 sweep 结论）。建议的 main 图排版（modular 手动拼）：上=patch-stack cards
（layer_0 / layer_11），中=`bolt_cluster_maps.png`（两 depth 聚类 atlas），下=cross-domain
prototype panel（layer_0 / layer_11）。

### k 选择（额外 sweep，见 `scripts/sweep_bolt_cluster_k.py`）

`outputs/figures/bolt_main_figure/k_sweep/` 扫了 k∈{4,5,6,8,10,12}。数据内在结构是一大团
连续 mass + 几个离群小岛（synthetic impulse 等）：k≤6 的 cluster map 干净但 shape 类型偏少；
k≥10 在硬切连续区、map 过碎。**选 k=8**：能额外暴露 flat/intermittent、U-rise、down-trend、
impulse 等更多 shape 类型（代价是 1–2 个 continuum"杂"簇）。cross-domain prototype 选择让每个
shape 家族跨多个域展示，正好配合 k=8 暴露的丰富类型。

### cross-domain prototype 选择逻辑

在 domain-balanced 子集上 PCA→KMeans 后，对每个 cluster：按 `macro_domain` 分组，每个域取该域
内离 cluster 中心最近的 patch，再按距离排序取前 N 个**不同域**。这样每行展示同一 shape 在不同
领域的实现（reusable primitive 跨域证据），而不是集中在该形状最常见的那几个域。domain-specific
的 cluster（如 impulse 主要在 Synthetic）会自然只显示较少列——如实反映。

## 1. 方法

two-space principle（`docs/00_narrative_rules.md` §5.1）：representation space 里
StandardScaler → PCA(30) → KMeans(k=8) 生成候选 cluster；回到 original time-series space
用 z-normalized raw patch 展示。**cards 与 prototype 都在 domain-balanced 子集上聚类**
（每个 source domain ≤ 400 patch，共 4400），否则 Traffic 这类高频 domain 会主导每个
cluster、淹没 shape 结构。

- patch-stack card：每个 cluster 取 center-nearest top-24 raw patch，z-normalize 后 imshow
  堆叠（行=rank，列=patch 内时间）。只有 raw 一行。标题下加一根 100% 归一化的
  **domain-composition 横条**，反映**整个 cluster**（非 top-24）的 macro_domain 构成 +
  共享 legend——明确展示 cluster 是跨域混合的，避免"cluster=单一 domain"误读（早期版本只印
  top-24 众数 domain，会误导，已弃用）。
- cross-domain prototype panel：每个 cluster 按 macro domain 分组、每域取 center-nearest 最佳
  代表，取前若干个**不同域**，画 z-normalized line plot（行=cluster，列=不同域的代表）。

设置：22 数据集各 120 窗口（2640 窗口），`context_len=128`，`patch_len=16`，k=8，seed=47。

## 2. 观察（含必须如实展示的对比）

- **layer_0 cards 是 shape-coherent 的候选 cluster**：能看到 monotonic ramp / level-shift
  （C1 下降 step、C6 上升 ramp）、impulse spike（C3，全是 synthetic pulse = negative
  control）等形状家族。这是"model-learned temporal primitive-like structures"在 shallow 层
  的证据（主线 3）。
- **layer_11 cards 的 raw patch stack 明显更"杂"**：这不是 bug。深层 cluster 已按
  **context / domain** 重组（C1/C2/C5≈Environment、C4≈Traffic、C3/C6≈Synthetic），所以
  同一 cluster 内的 raw patch 形状不再一致。这与 `docs/14_`（contextualization：深层 NMI 与
  域/位置对齐上升）完全一致——**深层是 contextualized cluster，不是 shape family**。
  （C3 在两层都保持干净的 synthetic-impulse 形状，是 negative-control 组的稳定特征。）
- **domain-balanced prototype panel（layer_11）每行一个可解释原型**：下降趋势 / 平坦-
  intermittent / impulse spike（synthetic）/ 上升趋势 / 峰 / 高频振荡。适合作 main 图下半部。

叙事边界（narrative rules）：这些是 clean Chronos-Bolt 的**候选** cluster，**不能**直接称为
motif；只有经过 original-space inspection + DTW-aware controlled retrieval + domain/frequency/
position confounder audit 后才是 `candidate motif/prototype family`。synthetic 组（C3）必须
作为 negative control 主动展示。

## 3. 复现

```bash
.venv/bin/python scripts/build_bolt_main_figure.py \
  --windows-per-dataset 120 --card-layers 0 11 --prototype-layers 0 11 \
  --k 8 --top-n 24 --proto-per-cluster 6 --max-per-domain 400 --tsne-perplexity 40

# k sweep（额外探索）
.venv/bin/python scripts/sweep_bolt_cluster_k.py --k-list 4 5 6 8 10 12
```

## 4. 待定（需 advisor 确认的排版选择）

- layer_11 的 raw patch-stack cards 因 contextualization 而 shape-incoherent。main 图里这一排
  是当作"shallow shape vs deep context"的对比保留，还是只放 layer_0 cards + prototype panel，
  取决于 main 图想强调哪条主线。脚本两种都能出（`--card-layers`）。
- k=8 已选定（见上节 sweep 结论）；`--k` 可继续调。
- k=8 时 patch-stack card 是一排 8 张、偏宽；如需更紧凑可改成 2 行布局。
