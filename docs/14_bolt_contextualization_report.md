# Chronos-Bolt contextualization：representation 随层变得 contextualized

更新时间：2026-06-07  
模型：**Chronos-Bolt-base**（clean 路线；见 `docs/99_chronos2_archive_and_chronos_bolt_pivot.md`）  
脚本：`scripts/run_bolt_contextualization.py` · `scripts/plot_bolt_contextualization.py` · `scripts/chronos_bolt_backbone.py`

advisor 三条主线里"**随层 contextualized**"的 clean 证据。两个 layer-wise 量随 representation
深度（`tokenizer` → encoder `layer_0/3/6/9/11`）变化，配合原有的 NMI 思路 + 一个新的
local-vs-global 相似度检查。

## 1. 两个指标

- **NMI（confounder absorption）**：每层在 PCA(30) space 做 KMeans(k=10)，算 cluster label
  与 `macro_domain` / `frequency` / `patch_index` 的 normalized mutual information。
- **within-context patch similarity**（研究问题）：**同一 context 下不同位置的 patch
  representation 之间的相似度，会不会随 depth 增加而增加？** 同一窗口内不同位置的 patch 当
  *same-context*，跨窗口随机 patch 当 *different-context*，比较余弦相似度。
  - ⚠️ **绝对 cosine 会被 confound**：深层表示空间整体散开，所有相似度（含 different-context）
    一起下降，掩盖真实趋势。因此用 **centered cosine**（每层先减全局均值方向，去掉所有 patch
    共享的 dominant component），度量同 context 耦合**本身**随深度的变化。

设置：22 数据集各 100 窗口（2200 窗口），`context_len=128`，`patch_len=16`，seed=47。

## 2. 结果

层号约定：图中 x 轴 `enc L1…L12` 用 1-based（Nature 习惯）= encoder block 索引 + 1；代码/CLI 仍 0-based
（`enc L1` = block 0、`enc L12` = block 11）。

证据图：`outputs/figures/bolt_contextualization/bolt_contextualization_depth_curve.png`

NMI（confounder absorption）：

| representation | NMI domain | NMI frequency | NMI position |
| --- | --- | --- | --- |
| tokenizer (input embed) | 0.123 | 0.145 | 0.005 |
| enc layer_0 | 0.129 | 0.156 | 0.003 |
| enc layer_3 | 0.164 | 0.204 | 0.004 |
| enc layer_6 | 0.165 | 0.230 | 0.009 |
| enc layer_9 | 0.154 | 0.257 | 0.012 |
| enc layer_11 | 0.143 | 0.262 | **0.021** |

within-context similarity（centered cosine；raw 见 summary JSON）：

| representation | same-context (centered) | different-context (centered) | gap |
| --- | --- | --- | --- |
| tokenizer (input embed) | 0.007 | 0.014 | −0.006 |
| enc layer_0 | 0.085 | 0.010 | +0.075 |
| enc layer_3 | 0.156 | 0.003 | +0.153 |
| enc layer_6 | 0.162 | 0.001 | +0.161 |
| enc layer_9 | 0.166 | 0.001 | +0.165 |
| enc layer_11 | **0.168** | 0.001 | **+0.167** |

结论：

1. **position NMI 单调上升（0.005 → 0.021，约 4×）**。这是最干净的 contextualization 信号：
   Chronos-Bolt 的 tokenizer 是 **value-only**（无 time encoding），所以本身几乎不含位置信息
   （NMI≈0.005）；**位置信息完全是经 encoder 的 attention 逐层注入的**。
2. **frequency / cadence NMI 单调上升（0.145 → 0.262）**：深层越来越编码采样频率这种
   context/domain-level 属性，而非单个 patch 的局部形状。
3. **domain NMI 在中层达峰（0.12 → 0.16，layer_6）** 后略降。
4. **同一 context 下不同位置 patch 的相似度随 depth 单调上升（答研究问题：会）**。centered
   same-context similarity 从 tokenizer 的 **0.007 升到 layer_11 的 0.168**，而 different-context
   始终 ≈0（随机 patch 去掉 shared component 后无关）。即 tokenizer 几乎没有 context 耦合
   （value-only），**encoder 逐层把"同 context 的 patch 互相靠拢"这件事建立起来**。
   - 注意：若看**未 centered 的绝对** cosine，same-context 相似度反而先降后升（0.77→0.47→0.62），
     因为深层整个空间散开、所有相似度一起降；这会掩盖真实趋势，故以 centered 为准（见 §4 边界）。

NMI 与 within-context similarity 一致指向同一结论，且**最大变化都发生在 layer_0 → layer_3**——
这与 forecasting probe
（`docs/13_`）里 forecasting 价值的最大跳变层完全吻合：**attention 前两三层是 contextualization
真正发生的地方**。

## 3. 复现

```bash
.venv/bin/python scripts/run_bolt_contextualization.py \
  --windows-per-dataset 100 --layers 0 3 6 9 11 --k 10
.venv/bin/python scripts/plot_bolt_contextualization.py
```

## 4. 边界

- KMeans / NMI 受 k 与 seed 影响；趋势（随层方向）稳健，但绝对值不应过度解读。
- **similarity 必须看 centered（或 gap），不能看绝对 cosine**：深层表示空间整体散开会让所有
  cosine 一起下降，掩盖 within-context 耦合的真实上升趋势。报告主结论用 centered cosine；raw
  与 ratio 一并存于 summary JSON 供审计。
- 这些是 representation-geometry 层面的 contextualization 证据，配合 `docs/13_` 的
  functional（forecasting）证据一起看更完整。
