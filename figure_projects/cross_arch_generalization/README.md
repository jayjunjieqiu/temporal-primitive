# Cross-architecture generalization（跨架构泛化附录）

为论文补强"单模型 Chronos-Bolt"短板：同一套 discovery/contextualization protocol，换到
**三种架构族 × 两种预训练目标**上，看关键 intrinsic 签名是否随深度跨架构复现。

| 模型 | 架构族 | 预训练目标 | patch_len | num_patches(512) | d_model | layers |
|---|---|---|---|---|---|---|
| Chronos-Bolt | encoder-decoder | autoregressive token | 16 | 32 | 768 | 0,6,11 |
| TimesFM-2.5 | decoder-only | autoregressive | 32 | 16 | 1280 | 0,10,19 |
| MOMENT-1-large | encoder-only | masked reconstruction | 8 | 64 | 1024 | 0,12,23 |

## 方法守则（硬约束）
- **共享池 + 绝对时间对齐**：三模型吃同一批 **512 点真实时间窗**，各自切 patch。512 是
  MOMENT 硬约束，正好当公共时间跨度。
- **去 OOD 化**：不逐模型重定义 ID/OOD、不下训练集。MOMENT 的 Time-series Pile 训了所有
  常见 benchmark（ETT/Electricity/Traffic/Weather/Exchange/ILI + 整个 Monash），不存在干净
  OOD 集。共享池与各模型预训练有重叠（MOMENT 尤甚）→ **不作任何泛化-到-训练外的声明**。
- **只比签名形状**：比"各模型各自随深度的趋势形状"，不比绝对数值、不比 cluster 身份。
- **不能声称"与 patch size 无关"**：patch_len 和架构/目标完全混淆，只能说"跨架构/目标/
  粒度的普遍性"。

## 复现
```bash
# 基建 smoke（共享池 + 三模型 dispatch）
.venv/bin/python scripts/cross_arch_shared_pool.py

# Step 3 — 两个 intrinsic 签名（contextualization 10-NN + selective convergence）
.venv/bin/python figure_projects/cross_arch_generalization/run_cross_arch_intrinsic.py \
    --windows-per-dataset 200 --seed 13          # 重绘: 追加 --replot

# Step 3b — 第三签名：cluster 相干 + 跨域 retrieval（two-space 验证）
.venv/bin/python figure_projects/cross_arch_generalization/run_cross_arch_cluster_retrieval.py \
    --windows-per-dataset 200 --seed 13          # 重绘: 追加 --replot
```

前置：`scripts/moment_backbone.py`（需 `uv pip install --no-deps momentfm`，见
`docs/00_local_model_download.md` §2.05）、`scripts/cross_arch_shared_pool.py`、
`figure_projects/cross_arch_generalization/cross_arch_style.py`（共享术语/配色）。

## 排版方案（panel 序号 + 对齐）
房子风格：每个面板是独立 `panel_<letter>_*.svg/.png`，在 PPT 里**手动拼接**（同主图/OOD 图）。
本附录拼成**单列 vertical stack a/b**，两面板**共享公共宽度**（tight-bbox ≈ 1090 pt，以
panel b 的方格表征图为锚，panel a 调 `figsize` 宽度对齐）：

```
panel a  intrinsic signatures   (1×4 lines,  figsize 15.15×4.0)  ← 顶：随深度签名（quant）
panel b  representation maps     (3×5 grid,   figsize 15.0×9.0)   ← 底：视觉主体（atlas+cards+retrieval）
```

> 原 panel c（cluster/retrieval 量化 bars）已删除：Nature 偏科普，过多量化图表影响阅读；
> 量化指标（coherence-lift / domain-entropy / retrieval fraction & shape-corr）仍写入
> `cross_arch_cluster_retrieval_summary.json`，供 PPT caption 直接引用。

panel b 每行左侧标注该模型**聚类所用的最深层**（1-based：Chronos-Bolt L12 / TimesFM L20 /
MOMENT L24）。**KMeans 固定 k=6**（与 `pub_main_figure_fullrep` 对齐；三模型同 k 才可比）。
列术语对齐主图 panel b：`Training data domains`（域上色）、`Learned temporal primitives`
（model-derived cluster 上色）、`Cross-domain retrieval`。**不使用 “atlas” 一词。**
域命名与主图/OOD 图统一：合成负控标 **`Synthetic`**（非 “Synthetic control”）。

## 产物（均房子风格：无 suptitle、SVG 可编辑、术语对齐主图/OOD 图）
- `panel_a_intrinsic_signatures.svg/.png` + `cross_arch_intrinsic_summary.json`
  —— 1×4：domain/frequency/position decodability（k-NN probe）+ within-context coherence，
  横轴相对深度，三模型三条线。
- `panel_b_repmaps.svg/.png` —— **3×N 视觉主图**：行=模型（左侧标各自聚类最深层），列=
  `Training data domains` 表征图 / `Learned temporal primitives` 表征图 / 2 cluster cards
  （**candidate primitive family**：跨域 ≥2 域且 shape-coherence 最高，原生 patch 长度；
  **粗线 = medoid 原型**＝表征空间里最靠近簇中心的真实 patch，**不是均值**；灰线 = 其余中心成员）/
  `Cross-domain retrieval`（黑粗线 = query patch；细线 = 其表征空间最近邻里**来自其它域**的
  patch，**按各自 domain 上色**＝复用底部 Domain 图例，多种域色即 cross-domain 自明；文字量/位置
  与前两列统一：仅顶部单标题，无格内图例/底部 caption）。
- `cross_arch_cluster_retrieval_summary.json` —— 量化指标（cluster shape-coherence lift /
  cluster 跨域熵 / retrieval 跨域比例 & 形状相关），不再画成图（原 panel c 已删），供 caption 引用。

## patch-length-fair coherence（重要方法点）
各模型 patch_len 不同（8/16/32）。原始 patch 直接算 Pearson 相关会**偏向长 patch**（32 点比
8 点稳）。因此所有 **shape-correlation 度量**（cluster coherence、retrieval shape-corr）在算
相关前，把 raw patch **线性插值重采样到统一长度 L=32**，消除 patch_len 偏差。视觉 cluster
cards 仍用**原生长度**（诚实展示各自粒度）。见 `run_cross_arch_cluster_retrieval.py` 的
`_resample` / `COMMON_PATCH_LEN`。

## 结论（full scale，200 窗/集）
**干净复现的两个签名：**
- **contextualization decodability**：domain/frequency 三族都"随深度上升、末层回落"。
- **selective convergence**：三族都从 tokenizer≈0 随深度增大。MOMENT（双向）单调升到最深层、
  无末层回落 —— 诚实差异，对应其 masked-reconstruction / 无"末层转向预测角色"。

**第三签名 = 部分复现 + 架构差异（诚实展示，勿当成"三族一样"）：**
- ✅ 跨域 retrieval：三族 cross-domain 邻居形状相关全为正（patch-length-fair 后可比）。
- ✅ 合成负控（Gaussian/Pulse）在三族表征图里都自成孤岛 —— negative control 该分开就分开。
- ⚠️ cluster shape-coherence：用最深层；Chronos-Bolt 末层按 forecasting role 重组（论文 Fig.4），
  故其 shape-coherence 偏 partial，**不是缺陷**，与论文叙事自洽。
- ⚠️ cluster 跨域熵：Bolt/MOMENT 高（cluster 跨域），**TimesFM 偏低、更按域聚集** —— 真实架构差异。

**禁止的声明**：不能说"与 patch size 无关"（patch_len 与架构/目标混淆），只能说"跨架构/目标/
粒度的普遍性"。retrieval 不作"泛化到训练外"声明（共享池与各模型预训练重叠）。
