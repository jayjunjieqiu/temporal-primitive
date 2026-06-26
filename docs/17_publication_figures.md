# 发布版 figures（NC 子刊）：main figure + OOD figure

更新时间：2026-06-26
模型：**Chronos-Bolt-base**（clean 路线）
变体：**fullrep**（完整 768 维 representation 上聚类，**不做 PCA**，`--cluster-space full --k 6`）

这两张图是给 NC 子刊投稿用的**发布级 figure**，在 PowerPoint 里**手动拼接**。每个 panel 都是
独立的、**文字可编辑的 SVG**（`svg.fonttype='none'`，sans-serif = Arial/Helvetica/DejaVu Sans），
字号偏大、**不带顶部 suptitle**（caption 在 PPT / 正文里写），科普性优先。所有 panel **共用一套
字号方案**、按各自最终摆放尺寸渲染，再整图统一缩放到版面宽度 —— 避免差异化缩放导致字号看起来不一致。

> 复刻这两张图所需的 cache 与 backbone 见 `docs/15`（main figure report）与
> `scripts/build_bolt_main_figure.py`。本文件只记录**发布版排版与配色约定**。

---

## 1. Main figure（`figure_projects/pub_main_figure_fullrep/`）

脚本：`scripts/pub_main_figure_panel_a.py` · `_panel_b.py` · `_panel_c.py` · `_panel_d.py`

| panel | 文件 | 内容 |
| --- | --- | --- |
| a | `panel_a_depth.*` | 三联深度曲线：forecast skill（RelMAE，H=16/64）· confounder probe accuracy（domain/frequency/position）· within-context similarity（same vs different context 双轴） |
| b | `panel_b_cluster_maps.*` | representation atlas（Layer 1 / Layer 12 两行 × 三列 t-SNE）：列 = Training data domains / Predefined motif labels / Learned temporal primitives；三个 legend 统一放到**图最上方**、列标题作为各 legend 框的**居中加粗标题** |
| c | `panel_c_cards_layer12.*` | Layer 12 的 6 个 candidate primitive family 的 patch-stack cluster cards（2 行 × 3 列，muted diverging heatmap `MUTED_DIV`），顶部 domain-composition legend |
| d | `panel_d_retrieval.*` | held-out → training retrieval：5 个代表性 **unseen test patch**（红 filled sparkline）→ 各自所在 cluster 的最近训练 patch（跨 domain，domain-colored 标签） |

排版（PPT）：a 通栏在最上，b 在中、c+d 同排（c 左较窄、d 右较宽）。

### panel d 行标约定
每行左侧两行右对齐：**shape 名**（粗体，如 *Gradual rise*）+ 下方分层标注 —— 极小号浅灰
eyebrow `Unseen dataset` + 稍大深灰**数据集名**（如 Weather / ETTh2 / CA / ETTh1 / ETTm2）。
表头 `Unseen test patch`（砖红）已说明 unseen 语义，行内不再重复 "unseen ·" 前缀。

---

## 2. OOD figure（`figure_projects/pub_ood_figure_fullrep/`）

脚本：`scripts/pub_ood_figure.py`，复现：

```bash
.venv/bin/python scripts/pub_ood_figure.py --cluster-space full --k 6
```

OOD score = held-out patch 到训练 patch 的 **mean kNN 距离 / 训练内 median neighbour 距离**，
在 **fullrep（标准化后的完整 768 维）空间**计算。t-SNE 仅用于 panel a 可视化（坐标缓存到
gitignored 的 `.ood_tsne_*.npz`），聚类本身在 representation space。

| panel | 文件 | 内容 |
| --- | --- | --- |
| a | `panel_a_overlap.*` | Layer 1 / Layer 12 两张 t-SNE：训练 patch（灰）+ held-out patch（按 OOD score 着色），共用右侧 colorbar `OOD score`；无 legend、无轴标签 |
| b | `panel_b_attribution.*` | 最 OOD 的 held-out patch 的两张 donut：**Nearest training primitive**（C1..C6）+ **Nearest training domain**；%在环内白字、类别名在外圈、极小扇区只留外名（避免拥挤重叠） |
| c | `panel_c_casestudy.*` | 个案研究：最 OOD 的 held-out patch（砖红）→ 各自最近的 3 个训练 primitive（filled sparkline，`C·domain` 标签 domain-colored），右侧高瘦 |
| d | `panel_d_ranking.*` | 各 held-out 数据集按 OOD 程度排名（Layer 1 vs Layer 12 分组竖柱），全幅底部条带；legend 放图**正下方**居中、横排两项、无标题 |

**排版（PPT）**：左上 a 叠 b、右上 c（高瘦）、底部 d 通栏。
（注意：c = 个案研究、d = ranking。早期版本编号相反，2026-06-26 互换。）

---

## 3. 配色系统（两图共用，muted seaborn-deep register）

刻意保持**统一的暖色克制调性**，三套色各司其职、不串义：

- **暖色（暖灰 → seaborn 橙 → 砖红）= OOD 程度**：OOD colormap（panel a）高端 = 砖红
  `#B5403F`；panel c 的 test patch、panel d 的 Layer 12 也用同一砖红。
  → 砖红 `#B5403F` 是贯穿全图的**高端 / 重点色**。
- **暖色浅→深梯度 = encoder depth**（OOD panel d）：浅暖褐 `#E8C4A0`（Layer 1）→ 砖红
  `#B5403F`（Layer 12）；落在 OOD colormap 同一条暖轴上，"更深 = 更暖 = Layer 12 更 OOD" 语义自洽。
  （曾用冷蓝梯度，因与上方暖调割裂，2026-06-26 改暖。）
- **seaborn-deep（muted）= 类别身份**：domain 用 `DOMAIN_COLORS`（Traffic `#4C72B0` / Energy
  `#DD8452` / Environment `#55A868` / Finance `#C44E52` / Retail-Web `#8172B3` / Health `#CCB974`
  / Synthetic `#937860`）；cluster C1..C6 用 seaborn-deep 列表（`SNS_DEEP`，比 tab10 更克制，
  大色块下不刺眼）。main figure 的 cluster 仍用 tab10——两图同一 C 号是「同色相、OOD 更柔」，
  读者不会混淆；若需跨图像素级一致可把 main 也切到 seaborn-deep（暂未做）。

main figure heatmap 用 `MUTED_DIV`（muted blue–warm white–brick red），红端与上述砖红同族。

---

## 4. 必须如实展示 / 写进 caption 的点（narrative rules）

- **synthetic control 用 `*` 标注**：OOD panel d 的 `Gaussian *` / `Pulse *` 是自制
  negative-control（不该被任何 primitive 虚假匹配）；`*` 含义在 caption / 正文写明，不在图里堆字。
- **OOD 是相对 16 个 discovery 子集、不是相对全量预训练**：ETT / Traffic / Weather / Illness
  等经典 benchmark 大概率在 Chronos-Bolt 预训练里见过（Electricity / BeijingAirQuality 正因此被
  `VALIDATION_EXCLUDE` 剔除，见 `docs/16`）。所以这里 "unseen / OOD" 严格指 **discovery 没见过**，
  不是模型没见过。若 OOD 升级为 main result，需把训练参照扩到 40–50+ chronos_datasets 重算
  （见 `docs/15 §8`、`docs/99` 路线说明）。
- **没有真正 top-level unseen domain**：basicts 测试集每个 macro-domain（Traffic / Energy /
  Environment / Finance / Health）都能在训练 6 域里找到对应。**最接近 "near-unseen domain" 的是
  Illness（Health）**：训练侧 Health 只有退化的 `covid_deaths`（cumulative ramp，当 negative-control），
  所以 discovery 子集里没有正常的健康类 seasonal prototype——Illness 本质上是 clustering 没真正
  覆盖过的域。要一个真正全新的 top-level domain 需引入 basicts/chronos 之外的数据（如生理信号
  ECG/EEG、地震、天文光变曲线）。
- **coverage ≈ 0.31**：~31% 的最 OOD held-out patch 与某个已知训练 primitive 形状一致
  （centered-cosine corr ≥ 0.6）——这是个**量化命中率**，不是 "完整 temporal language"。
- 这些 cluster 仍是 **candidate primitive / prototype families**，不是已命名的 motif；命名需经
  original-space inspection + DTW-aware controlled retrieval + confounder audit（见 `docs/00_narrative_rules.md §7`）。
