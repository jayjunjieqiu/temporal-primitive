# Reference-style spectral illustration plan

## 1. 目标

这份文档总结当前讨论形成的画图方案。目标不是机械复刻 `reference_pic.jpg`，而是借用它的证据结构，画出一张更适合本项目的图：

> TSFM patch-token representation space 中存在 `model-learned temporal primitive-like structures`；这些结构不是 human-designed motif taxonomy 的直接投影，而是需要通过 original-space inspection、spectral evidence、DTW-aware validation 和 confounder audit 来解释的 model-derived candidate motif/prototype families。

当前阶段使用已有 `Chronos-2` pilot 结果作为 archived diagnostic evidence。后续 clean test 应按 `docs/99_chronos2_archive_and_chronos_bolt_pivot.md` 迁移到 `Chronos-Bolt`。

## 2. Reference 图的有效成分

`reference_pic.jpg` 可以拆成四类视觉成分。

### 2.1 Central representation atlas

图中央是一个 2D embedding / latent space map。点按类别着色，周围的 evidence panels 通过虚线连接到 embedding map 的局部区域。

它的作用不是给出最终 taxonomy，而是展示：latent space 里存在可见的 cluster-like structure。

对应到本项目：

- 使用 `t-SNE` 或 `UMAP` 展示 TSFM patch representation space。
- 点按 `KMeans cluster` 着色。
- 标出 `KMeans center` 或 selected cluster region。
- 主要 layer 建议使用 `layer_6` 或 `layer_7` 这类 middle layer，因为导师关心 early/middle/top layer 如何保留或混合 local temporal information。

### 2.2 Surrounding spectral-shape exemplars

参考图上半部分围绕 central map 的一圈小图，不应理解为 cluster prototype。它们更像从 embedding region 中抽出的 representative samples，并以 Fourier-domain / spectral-domain 形式展示。

换句话说：

- 它们是 region-level exemplar evidence；
- 它们通过虚线连接到 embedding space 的具体区域；
- 它们用于说明某个 latent region 对应的样本具有可读的 frequency / texture / spectral signatures；
- 它们不是整个 cluster 的平均原型，也不是最终 taxonomy label。

对应到本项目，不应再简单画 raw patch prototype cards。更合理的是画：

```text
Representative cluster instance:
raw patch | low-frequency reconstruction | high-frequency residual | 1D power spectrum
```

如果版面需要简化，可以退化为：

```text
raw patch | 1D power spectrum
```

### 2.3 Bottom prototype / descriptor grid

参考图底部的 `Power spectrum versus frequency` 和 `Phase spectrum versus frequency` 更像真正的 family-level descriptor / prototype summary。它不是单个样本，而是对若干类别或区域的系统化频域总结。

对应到本项目，底部应该画：

- `cluster-level median power spectrum`;
- `IQR band` 或 variance band;
- `DTW medoid raw patch` 或 `KMeans-center nearest patch`;
- 可选的 `first-difference spectrum`，用于 spike、burst、transition 等局部变化模式。

### 2.4 Callout links

参考图用浅灰虚线把 central map 的局部区域和周围 evidence panels 连起来。这是它的关键阅读逻辑：周围 evidence 来自 representation map 的具体位置，而不是单独挑选的漂亮样本。

对应到本项目，虚线最好作为单独的透明 overlay 输出，等最终 PPT / figure 排版确定后再画，避免反复对齐。

## 3. 为什么本项目不做 2D Fourier transform

我们的基本对象是 1D time-series patch：

```text
patch = [x1, x2, ..., x16]  或  [x1, x2, ..., x32]
```

它只有一个自然时间轴。因此最自然、最严谨的频域分析是：

- `1D FFT`;
- `power spectrum`;
- `phase spectrum`;
- `first-difference spectrum`;
- `autocorrelation`;
- 在更长 context 上可考虑 `STFT` 或 wavelet。

参考图能做 `2D Fourier transform`，是因为它处理的是 CT / image-like data：

```text
image = H x W
```

若把本项目的 top-k patches 堆叠成矩阵再做 2D FFT，技术上可行，但解释不稳：

- 横轴是 within-patch time；
- 纵轴是 top-k nearest order；
- 这个 order 不是物理时间、空间位置或 sensor channel；
- 改变 top-k 排序会改变 2D FFT 结果。

因此当前主图不应使用 2D FFT。更稳妥的设计是模仿 reference 的频域证据结构，但数学对象改为 1D spectral evidence。

推荐句式：

> We follow the visual logic of Fourier-domain evidence, but use 1D spectral signatures because TSFM patches are one-dimensional temporal segments rather than two-dimensional images.

## 4. 最终应画的模块

### Module A: Central t-SNE representation map

用途：展示 model representation space 中存在 cluster-like structure。

建议内容：

- `Chronos-2 layer_6, K=10` 作为 archived pilot 主图；
- 点按 KMeans cluster 着色；
- 标出 selected clusters 的 centers；
- 不使用 prior-guided motif labels；
- 图内文字使用英文；
- cluster 只命名为 `C0`, `C1`, ...

建议尺寸：

- `2200 x 1600 px`;
- 适合放在 PPT 主图中间。

### Module B: Surrounding spectral-shape exemplar cards

用途：展示 selected cluster regions 回到 original time-series space 后具有可读的 shape/spectrum evidence。

建议每个 card 内容：

```text
Ck
raw patch | low-frequency reconstruction | high-frequency residual | 1D power spectrum
```

简化版：

```text
Ck
raw patch | 1D power spectrum
```

候选 selected clusters：

- `C0`: rising transition-like region;
- `C6`: falling transition-like region;
- `C7`: spike-like region;
- `C2`: flat / low-information region;
- `C3`: noisy / high-variation region;
- `C4` 或 `C5`: mixed transition-like region。

注意：

- 这些 descriptor 只能作为 visual descriptor，不能作为 final motif taxonomy label；
- 每个 card 应基于 `KMeans center-nearest` 或 `DTW medoid` 样本；
- 如果 top-1 example 不清楚，应展示 top-5 center-nearest examples 的一致性，而不是只挑一个漂亮样本。

建议尺寸：

- 单个 card: `760 x 360 px`;
- 一组 cards: `3600 x 1200 px`。

### Module C: Bottom cluster spectral prototype grid

用途：提供 family-level / cluster-level prototype summary，对应 reference 图底部的 spectrum grids。

建议每列对应一个 selected cluster，每格包含：

- `median power spectrum`;
- `IQR band`;
- `DTW medoid raw patch` 或 `KMeans-center nearest patch`;
- 可选 `first-difference spectrum`。

建议不要把 `phase spectrum` 作为主证据。patch length 16 较短，phase 解释可能不稳。若需要二级证据，可放 appendix 或 diagnostic。

建议尺寸：

- `3600 x 1000 px`。

### Module D: Layer progression strip

用途：回答导师的问题：不同 layer 如何改变 patch representation，single patch / early representation 是否更保留 local temporal information。

建议内容：

- `projection / early`;
- `layer_6 / middle`;
- `layer_11 / high-level`;
- 每张图使用同一 visual style；
- 下方短说明：
  - `Projection / early: more local and shape-readable`;
  - `Layer 6 / middle: structured but contextualized`;
  - `Layer 11 / high-level: stable but less physically direct`。

注意：

- 由于 Chronos-2 projection token 包含 time encoding，这部分必须标注为 archived pilot evidence；
- clean conclusion 应在 Chronos-Bolt 上复验。

建议尺寸：

- `3200 x 900 px`。

### Module E: Callout link overlay

用途：把 Module A 的 selected cluster centers 和 Module B 的 spectral-shape cards 连起来。

建议：

- 单独输出透明背景 PNG / SVG；
- 使用浅灰虚线；
- 不使用箭头；
- 等 PPT 最终排版确定后再生成。

建议尺寸：

- 与最终 PPT 画布一致，例如 `3840 x 2160 px`。

## 5. 推荐最终拼接结构

### Version A: Reference-style main figure

适合导师汇报主图。

```text
           Spectral card C0          Spectral card C7
                    \                  /
                     \                /
Spectral card C6 -- Central layer-6 map -- Spectral card C4
                     /                \
                    /                  \
           Spectral card C5       Spectral card mixed/noisy

Bottom: cluster spectral prototype grid
```

这版最接近 reference 图的视觉语法。

### Version B: Paper-style multi-panel figure

适合后续报告或论文草图。

```text
a. Layer progression strip
b. Central layer-6 representation atlas
c. Surrounding or adjacent spectral-shape exemplars
d. Cluster-level spectral prototype grid
```

这版更适合写成正式 figure caption。

## 6. 推荐结论边界

主图可以支持的表述：

- TSFM patch representation space 中存在 stable cluster-like structures；
- 这些 structures 在 original time-series space 中对应可重复的 shape / spectral signatures；
- early representation 更接近 local patch vocabulary；
- middle layer 仍保留 structure，但更 contextualized；
- 这些可以称为 `model-learned temporal primitive-like structures` 或 `candidate motif/prototype families`。

主图不能直接支持的表述：

- 我们已经发现最终 `temporal language`；
- KMeans clusters 就是 final motif taxonomy；
- human-prior motif taxonomy 是 ground truth；
- Chronos-2 projection 能干净回答 pure value-only patch token 的 single-patch local information。

推荐 caption 句式：

> Representative archived Chronos-2 pilot evidence. Model-derived cluster regions in patch-token representation space show recurring shape and 1D spectral signatures, suggesting primitive-like temporal structures learned from heterogeneous time series. These structures are not final human-designed motif labels and require DTW-aware validation and confounder audit.

## 7. 下一步画图任务

建议先做三个最小模块，避免再次画成过度复杂的 composite figure：

1. `Module A`: `layer_6 K=10` central t-SNE map，标出 selected cluster centers。
2. `Module B`: selected clusters 的 spectral-shape exemplar cards，每个 cluster 至少 top-5 center-nearest examples。
3. `Module C`: cluster-level median power spectrum + IQR band + DTW medoid / KMeans-center nearest patch。

完成这三项后，再根据 PPT 版面决定是否生成 `Module D` layer progression strip 和 `Module E` callout overlay。

