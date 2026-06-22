# 测试数据集 × Chronos 预训练成员身份（zero-shot 依据）

更新时间：2026-06-21

用途：判断我们用的测试数据哪些在 Chronos / Chronos-Bolt 的**训练集内**，哪些在**训练集外**，
为"TSFM 有用"（`docs/13_`）这条主线的 zero-shot 表述提供依据。

## 0. 角色（2026-06-21 discovery-on-training pivot）

primitive discovery 已迁到 Chronos **in-distribution 训练数据**上做（curated 16 数据集，见
`scripts/chronos_training_data.py` 与 `docs/15`），basicts 测试集**降级为泛化 validation**。所以本表
现在的作用是：**确认 basicts 测试集里哪些可以当干净的 held-out 泛化证据**——结论是除 Electricity /
BeijingAirQuality 外的 20 个都在训练集外，可用作 validation（main figure 的 generalization 模块即用它们）。
下面是详细判定。

## 1. 我们定义的 domain

本仓库只定义 **6 个 macro domain**（`MACRO_DOMAIN_DEFINITIONS`，见
`scripts/run_prior_guided_probe_sanity_check.py`），数据集 `desc.json` 里的原始 `domain`
字符串（traffic flow / traffic speed / road occupancy rates / electricity consumption /
electricity transformer temperature / weather / Beijing air quality / exchange rate /
illness data / simulated Gaussian / simulated pulse）被归并进这 6 个：

- Traffic / Energy / Environment / Finance / Health / Synthetic control

## 2. 成员身份表

判定原则：只要在 Chronos 论文里属于**训练集外**（无论是显式 held-out / zero-shot，还是根本不在
语料里），统一记"否"。只有真正用于训练（in-domain）才记"是"。

| macro domain | 数据集 | 在 Chronos 训练集 |
| --- | --- | --- |
| Traffic | CA, GBA, GLA, SD, PEMS03, PEMS04, PEMS07, PEMS08, METR-LA, PEMS-BAY, Traffic | 否 |
| Energy | Electricity | 是 |
| Energy | ETTh1, ETTh2, ETTm1, ETTm2 | 否 |
| Environment | BeijingAirQuality | 是 |
| Environment | Weather | 否 |
| Finance | ExchangeRate | 否 |
| Health | Illness | 否 |
| Synthetic control | Gaussian, Pulse | 否 |

## 3. 结论

- **只有 2 个数据集在 Chronos 训练集内**：
  - `Electricity` ≈ `monash_electricity_hourly`（in-domain）
  - `BeijingAirQuality` ≈ `monash_kdd_cup_2018`（KDD Cup 2018 北京+伦敦空气质量，in-domain）
- 其余 **20 个全部在训练集外**，可作 zero-shot 证据。其中：
  - `Traffic`（= `monash_traffic`）、`ExchangeRate`（= `exchange_rate`）虽然在 chronos_datasets
    集合目录里，但属于 Chronos **官方 zero-shot held-out**，未用于训练，故记"否"。
  - 10 个时空交通（PEMS×4 / METR-LA / PEMS-BAY / CA / GBA / GLA / SD）、4 个 ETT、Weather（Jena）、
    Illness、2 个合成（Gaussian/Pulse）完全不在 Chronos 语料。
- **写 zero-shot claim 时剔除 `Electricity` 和 `BeijingAirQuality`。**

## 4. 判定依据与 caveat

- 训练 vs held-out 用的是 chronos-forecasting 自带的官方清单：
  `external/chronos-forecasting/scripts/evaluation/results/chronos-bolt-base-in-domain.csv`
  与 `…-zero-shot.csv`；预训练语料目录在 `/data/ts-datasets/chronos_datasets/`。
- **目录里有该数据集 ≠ 用于训练**：chronos_datasets 集合同时存放训练数据和 zero-shot 评测数据
  （`exchange_rate`、`monash_traffic` 即反例）。`training_corpus/` 子目录只含合成增强
  （`kernel_synth_1m`、`tsmixup_10m`）。
- 数据集对应关系按**名称 + 领域**判定（如 Electricity↔monash_electricity_hourly、
  Traffic↔monash_traffic）。如需 100% 确证，可进一步比对序列数量 / 采样频率 / 数值范围做
  内容级核对。
