# Chronos-Bolt forecasting probe：contextualized backbone 比 tokenizer 更有用

更新时间：2026-06-07  
模型：**Chronos-Bolt-base**（clean 路线默认模型；见 `docs/99_chronos2_archive_and_chronos_bolt_pivot.md`）  
脚本：`scripts/run_bolt_forecasting_probe.py` · `scripts/plot_bolt_forecasting_probe.py` · `scripts/chronos_bolt_backbone.py`

这是 advisor 三条主线里"**TSFM 有用**"+"**随层 contextualized**"两条的 clean 证据。
是一个 **forecasting probe**（不是分类 / SVM）：用 frozen representation 回归预测真正的未来。

## 1. 为什么用 Chronos-Bolt（clean 性质）

Chronos-Bolt 的 `input_patch_embedding`（我们称的 "tokenizer"）只吃
`[normalized patch values, patch mask]`，**不含 time encoding**——对全观测 patch，mask
恒为 1，所以本质是 pure value-only patch token。这正是相对 archived Chronos-2 pilot
（其 projection token 混入 time encoding）的 clean 改进，因此本结果可作为 clean evidence。

- backbone：T5 encoder，12 层（`encoder.block[0..11]`），`d_model=768`，`patch_len=16`。
- 两种 representation 对照：
  - `tokenizer`：`input_patch_embedding` 输出（pre-transformer，无 patch 间交互）。
  - `layer_{L}`：`encoder.block[L]` hidden state（attention 已让 patch 互相 contextualize）。

## 2. Probe 设计（horizon-from-context）

- 跨 22 个 basicts 数据集采样窗口（每个 200 个，共 **4400** 窗口），`context_len=128`。
- 归一化只用 **context** 的 mean/std（z-score + clip ±10，和 Bolt 内部 instance_norm 同款；
  避免泄漏未来，也避免近常数 context 的病态放大）。
- representation 对整段 context 做 **mean-pool** 得到序列级向量。
  - 公平点：tokenizer 与 backbone 都 pool 了整段 context，差别只在 backbone 经过
    attention 做了 contextualization → backbone 赢 = contextualization 对预测有用。
- frozen **MLP probe**（`MLPRegressor(256,128)`，early-stopping）预测**窗口之后真正的未来
  H 步**（genuine future，H∈{16, 64}）。用 MLP 而非线性 probe，是为匹配 Bolt 自身的
  非线性 forecast head（见 §5 的负例诊断）。
- 评估：held-out 窗口（按 window 划分，避免同窗口泄漏），report **MASE**（基线 =
  persistence，context 最后值重复）与 **R²**。锚点 `raw_last_patch` = context 最后 16 个
  归一化值（linear-AR 参照）。

## 3. 主结果

证据图：`outputs/figures/bolt_forecasting_probe/bolt_forecasting_probe_depth_curve.png`

**H = 16**

| representation | MASE ↓ | R² ↑ |
| --- | --- | --- |
| raw_last_patch (AR 锚点) | 0.894 | 0.493 |
| tokenizer (input embed) | 1.100 | 0.360 |
| enc layer_0 | 0.970 | 0.461 |
| enc layer_3 | 0.837 | 0.552 |
| enc layer_6 | 0.853 | 0.546 |
| enc layer_9 | 0.839 | 0.554 |
| **enc layer_11** | **0.831** | **0.555** |

**H = 64**

| representation | MASE ↓ | R² ↑ |
| --- | --- | --- |
| raw_last_patch (AR 锚点) | 0.889 | 0.286 |
| tokenizer (input embed) | 0.927 | 0.249 |
| enc layer_0 | 0.858 | 0.287 |
| enc layer_3 | 0.787 | 0.392 |
| enc layer_6 | 0.789 | 0.400 |
| enc layer_9 | 0.784 | 0.398 |
| **enc layer_11** | **0.782** | **0.401** |

三个结论，对应三条主线：

1. **TSFM 有用（主线 1）**：backbone representation 的 MASE < 1（赢过 persistence 基线），
   并且赢过 `raw_last_patch` 这个 raw-AR 读出（H=16: R² 0.555 vs 0.493；H=64: 0.401 vs
   0.286）。即学到的表示比"原始最近值"更能预测未来。
2. **随层 contextualized（主线 2）**：误差从 tokenizer → encoder 层显著下降，**最大跳变发生在
   layer_0 → layer_3**（正是 attention 层注入 contextualization 之处），之后在 layer_6–11
   plateau。深度本身带来 forecasting 价值。
3. **backbone > tokenizer**：两个 horizon、**每个 macro domain** 下，encoder 层都赢 tokenizer
   （H=16 per-domain：layer_11 在 Energy/Environment/Finance/Health/Traffic/Synthetic
   全面低于 tokenizer 的 MASE）。

一个值得点出的细节：H=16 时 **tokenizer 反而比 raw_last_patch 差**（MASE 1.10 vs 0.89）。
tokenizer 是对每个 patch 的有损非线性 embedding、再 mean-pool，丢掉了 raw 保留的精确最近值；
**只有经过 attention（encoder）后，表示才恢复并反超**。这本身就是"contextualization 才是
usefulness 来源"的直接证据。

## 4. 复现

```bash
.venv/bin/python scripts/run_bolt_forecasting_probe.py \
  --mode horizon --horizon 16 --pooling mean --probe mlp \
  --windows-per-dataset 200 --layers 0 3 6 9 11
.venv/bin/python scripts/run_bolt_forecasting_probe.py \
  --mode horizon --horizon 64 --pooling mean --probe mlp \
  --windows-per-dataset 200 --layers 0 3 6 9 11
.venv/bin/python scripts/plot_bolt_forecasting_probe.py
```

## 5. 诚实记录的负例与陷阱（narrative rules §负例必须主动展示）

得到上面的 clean 结果之前，踩过几个坑，保留下来以免后人误判：

- **linear probe 会得到相反结论。** 如果用 Ridge（线性）probe 预测原始未来值，会看到
  `raw_last_patch` 最好、backbone 全是负 R²、且越深越差。原因：从 Bolt encoder hidden 到
  未来值的映射本质非线性（T5 decoder + ResidualBlock head + 反归一化），线性读出对学到的
  抽象表示极不公平。**该 metric 测的是"未来原始值能否被线性解码"，不是"表示是否有用"。**
  这条负例其实正面支撑主线 2（深层越抽象）。
- **next-patch（每个 patch 位置解码紧接的下一个 patch）模式下，layer_0 最好、越深越差。**
  同样因为深层 hidden 为 decoder cross-attention 服务、是 contextualized/abstract 表示，
  不为"线性还原下一 patch 原始值"优化。`--mode next_patch` 可复现该诊断。
- **小样本是伪影来源。** horizon 模式每个窗口只产生 1 个样本；只用 ~900 训练样本喂 768-dim
  representation 时，MLP 过拟合、方差大，会让 raw（16-dim）显得更稳更好。加到 4400 窗口后，
  backbone 的优势才稳定显现。
- **归一化必须 clip。** 近常数 context（intermittent / flat 序列）的 MAD≈0，会把归一化后的
  target 放大到天文数字、主导 MAE。改用 mean/std z-score 并 clip ±10 解决。

## 6. 边界与后续

- 这是 **short/medium horizon、point-forecast、frozen-probe** 的证据；不等于 Chronos-Bolt
  端到端 forecasting accuracy 的完整评测。
- 短 horizon 上 recency 很强，结论依赖足够的样本量与非线性 probe；写作时不要把它讲成"任何
  设定下 backbone 都碾压"。
- 可选的黄金标准补充：用 Bolt 真实 `predict()` 的 forecasting metric（WQL/MASE）对比"完整
  模型 vs 绕过 encoder"，更贴近"TSFM 有用"的端到端叙事（本轮未做）。
