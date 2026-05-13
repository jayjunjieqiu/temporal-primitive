# 可行性报告：Temporal Primitive Patch 标注与 TSFM Hidden-State 抽取

## 1. 核心判断

**可以进入下一阶段实现，但还不建议直接跑正式实验。**

目前几个关键可行性问题已经得到正向验证：

- 本地 `TimesFM-2.5` 权重可以从当前工作区加载。
- 本地 `Chronos-2-small` 和 `Chronos-2` 权重可以从当前工作区加载。
- 三个模型都可以抽取 patch-level final embedding 和 selected layer hidden states。
- 可以构造一个小型 synthetic motif calibration set。
- 最小 patch-level motif-neighbor prototype 可以端到端跑通。

剩下的主要问题不是模型本身不可行，而是**实验环境还不够干净**。当前 Python 环境可以支持 smoke test，但还不适合作为正式实验环境。

## 2. 本地资源检查

本地已有模型权重和配置：

- `chronos-2-small/config.json`
- `chronos-2-small/model.safetensors`
- `chronos-2/config.json`
- `chronos-2/model.safetensors`
- `timesfm-2.5-200m-pytorch/config.json`
- `timesfm-2.5-200m-pytorch/model.safetensors`

为了检查官方实现，已克隆代码库：

- `external/chronos-forecasting`
- `external/timesfm`

本次没有修改任何模型权重。

## 3. 环境状态

当前运行环境：

- Python: `3.13.9`
- PyTorch: `2.11.0+cu130`
- CUDA 可用: yes
- GPU: NVIDIA A800 80GB
- 已安装 `accelerate==1.2.1`，用于支持 Hugging Face `device_map` 加载

smoke test 后的包状态：

- `torch`: 可用
- `transformers`: 可用，但版本高于 Chronos 声明的 `<5` 要求
- `accelerate`: 可用
- `chronos`: 未全局安装，通过 `PYTHONPATH=external/chronos-forecasting/src` 使用
- `timesfm`: 未全局安装，通过 `PYTHONPATH=external/timesfm/src` 使用
- `sklearn`: 未安装
- `stumpy`: 未安装
- `aeon`: 未安装

重要说明：`scikit-learn` 安装时在默认镜像和 PyPI 下载 wheel 阶段都出现卡住。为了完成 smoke test，脚本里临时放了一个很小的 `sklearn` import stub，让 Chronos 源码在不使用 categorical covariates 的情况下可以导入并抽 embedding。这个做法只适合可行性验证，不能作为正式实验环境。

## 4. 模型代码路径

### 4.1 Chronos-2-small 与 Chronos-2

官方源码路径：

- model implementation: `external/chronos-forecasting/src/chronos/chronos2/model.py`
- pipeline implementation: `external/chronos-forecasting/src/chronos/chronos2/pipeline.py`
- dataset/input preparation: `external/chronos-forecasting/src/chronos/chronos2/dataset.py`

本地加载方式：

```python
sys.path.insert(0, "external/chronos-forecasting/src")
import chronos

pipeline = chronos.Chronos2Pipeline.from_pretrained(
    "chronos-2-small",  # or "chronos-2"
    local_files_only=True,
    device_map="cpu",
)
```

final embedding 抽取：

```python
embeds, loc_scale = pipeline.embed([series], batch_size=1, context_length=len(series))
```

layer-wise hidden state 抽取：

```python
model = pipeline.model
handle = model.encoder.block[layer_idx].register_forward_hook(hook)
encoder_outputs, *_ = model.encode(context=context, num_output_patches=1)
handle.remove()
```

相关内部流程：

- `_prepare_patched_context(...)`
- `self.patch(...)`
- `self.input_patch_embedding(...)`
- optional `[REG]` token insertion
- future masked output patch insertion
- `self.encoder.block[...]`

patch/token 细节：

- `Chronos-2-small`: patch length `16`, `6` encoder layers, hidden size `512`
- `Chronos-2`: patch length `16`, `12` encoder layers, hidden size `768`
- 对 128-step 输入和一个 masked output patch，hidden sequence length 为 `10`：`8` 个 context patches + `1` 个 `[REG]` token + `1` 个 future patch token

### 4.2 TimesFM-2.5

官方源码路径：

- model implementation: `external/timesfm/src/timesfm/timesfm_2p5/timesfm_2p5_torch.py`
- normalization utilities: `external/timesfm/src/timesfm/torch/util.py`
- transformer block: `external/timesfm/src/timesfm/torch/transformer.py`

本地加载方式：

```python
sys.path.insert(0, "external/timesfm/src")
import timesfm

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "timesfm-2.5-200m-pytorch",
    torch_compile=False,
    local_files_only=True,
)
```

layer-wise hidden state 抽取：

```python
module = model.model
handle = module.stacked_xf[layer_idx].register_forward_hook(hook)
(input_embeds, final_embeds, point_head, quantile_head), _ = module(
    normed_inputs,
    patched_masks,
)
handle.remove()
```

相关内部流程：

- 将原始序列 reshape 成 patch length `32`
- 通过 `util.update_running_stats` 计算 running patch normalization
- 应用 `util.revin`
- `module.tokenizer(...)`
- `module.stacked_xf[...]`

patch/token 细节：

- `TimesFM-2.5`: patch length `32`, `20` transformer layers, hidden size `1280`

## 5. Motif 标签构造可行性

目前没有看到现成的 TSFM patch-level motif label 数据集。更实际的路线仍然是：

1. 先构造带可控 motif 标签的 synthetic calibration set。
2. 在真实数据上用 motif discovery / shapelet 方法生成弱监督 prototype bank。
3. 人工命名少量高置信 prototype。
4. 用最近邻或轻量分类器把标签传播到更多 patch。
5. 对模糊或复合片段保留 `mixed/uncertain` 类。

可用方法候选：

- **STUMPY / Matrix Profile**：适合寻找重复 subsequence、motif pairs、discords、chains、snippets，也支持 multidimensional extensions。官方文档里有 `stumpy.stump(...)`、`stumpy.mstump(...)` 和 GPU variants。
- **aeon transforms**：适合使用 `MatrixProfile`、`MatrixProfileTransformer`，以及 `RandomShapeletTransform`、`RandomDilatedShapeletTransform`、`SAST`、`RSAST` 等 shapelet-based transforms。
- **naive z-normalized patch nearest-neighbor baseline**：足够透明，适合在引入更重的 motif libraries 之前作为 sanity check。

本次 smoke prototype 使用第三种轻量 baseline，主要是为了避开当前环境中 `stumpy` / `aeon` 尚未安装的问题。

## 6. Smoke Prototype

已创建脚本：

- `scripts/smoke_temporal_primitives.py`

输出文件：

- `outputs/smoke_temporal_primitives_summary.json`

运行命令：

```bash
python scripts/smoke_temporal_primitives.py
```

脚本内容：

- 构造六类 synthetic motifs: `trend`, `oscillation`, `spike`, `burst`, `regime_shift`, `intermittent`
- 用 patch length `32` 做 motif-neighbor sanity check
- 计算 z-normalized patch nearest neighbors
- 从本地权重加载 `TimesFM-2.5`，并 hook layers `0`, `10`, `19`
- 从本地权重加载 `Chronos-2-small`，并 hook layers `0`, `3`, `5`
- 从本地权重加载 `Chronos-2`，并 hook layers `0`, `6`, `11`

本次 smoke test 结果：

- synthetic motif series: `6`
- synthetic length: `128`
- patch-neighbor label agreement: `0.625`
- `TimesFM-2.5` final embedding shape: `[6, 4, 1280]`
- `TimesFM-2.5` captured layer shapes: `[6, 4, 1280]`
- `Chronos-2-small` final embedding shape: `[1, 10, 512]`
- `Chronos-2-small` captured layer shapes: `[1, 10, 512]`
- `Chronos-2` final embedding shape: `[1, 10, 768]`
- `Chronos-2` captured layer shapes: `[1, 10, 768]`

这里的 patch-neighbor agreement 只是 sanity signal，不是正式评估结果，也不应该作为研究发现使用。

## 7. 阻塞点与风险

### 阻塞点 A: 依赖环境

原始环境可以跑 prototype，但不够干净：

- Chronos 声明依赖 `transformers>=4.41,<5`，当前环境的 `transformers` 版本更高。
- `scikit-learn` 缺失，且安装尝试在下载阶段卡住。
- `stumpy` 和 `aeon` 缺失。
- smoke script 中 Chronos import 曾依赖一个临时 `sklearn` stub。

这一点已经通过 `uv` 管理的项目 `.venv` 得到缓解。

`.venv` 当前状态：

- `torch 2.11.0+cu130` 可用，`cuda` 可用
- `transformers 4.57.6` 可用
- `scikit-learn 1.7.2` 可用
- `stumpy 1.14.1` 可用
- `aeon 1.4.0` 可用
- `chronos` 和 `timesfm` 可通过本地源码使用
- `smoke_temporal_primitives.py` 在 `.venv` 下已重新跑通

建议在正式实验前继续使用这个 project environment，并避免回到原始全局环境。

当前 `.venv` 采用 Python `3.13.9`，原因是它可以通过 `--system-site-packages` 复用已有的 CUDA PyTorch。长期如果要做更干净的独立环境，可以再评估 Python `3.12` + CUDA PyTorch wheel 的方案。

建议环境目标：

- Python `3.13.9` for current `.venv`
- `torch`
- `transformers>=4.49,<5`
- `chronos-forecasting` from `external/chronos-forecasting` or PyPI
- `timesfm` from `external/timesfm`
- `scikit-learn>=1.6,<2`
- `stumpy`
- `aeon`
- `numpy`, `pandas`, `scipy`, `matplotlib`, `umap-learn`

### 阻塞点 B: motif taxonomy 仍需要操作化定义

proposal 中的标签适合作为研究标签，但正式实验需要明确判定规则：

- spike 和 burst 的边界是什么
- trend 至少需要持续多久
- regime shift 和 trend 如何区分
- composite motifs 如何标注
- amplitude / scale normalization 如何处理

### 风险 C: patch length mismatch

Chronos 使用 patch length `16`，TimesFM-2.5 使用 patch length `32`。

建议：

- 至少在 `16` 和 `32` 两种粒度上构造 motif 标签。
- 每个 patch 都保留原始时间索引。
- 同时评估 native-patch setting 和 resampled/common-window setting。

## 8. 推荐下一步

**现在不建议继续做泛化式文献调研。**

下一步应该进入 implementation-focused 阶段：

1. 创建一个干净、可复现的实验环境。
2. 把模型抽取逻辑封装成 reusable adapters：
   - `Chronos2Extractor`
   - `TimesFM25Extractor`
3. 把 synthetic motif generation 拆成独立 dataset module。
4. 添加真实 motif backend：
   - first choice: STUMPY / Matrix Profile
   - second choice: aeon MatrixProfile and shapelet transforms
5. 跑一个稍大的 synthetic benchmark，验证 motif 标签在 patch length `16` 和 `32` 下是否稳定。

建议的 immediate implementation target：

- 每个 motif 抽取 `N=100` 条 synthetic series。
- 保存 patch metadata，字段包括：`series_id`, `domain`, `motif`, `start`, `end`, `patch_len`, `model`, `layer`。
- 用统一结构保存 embeddings，例如 `.npz`，或者 parquet metadata + 单独 tensor files。

## 9. 参考资料

- Chronos official repository: https://github.com/amazon-science/chronos-forecasting
- TimesFM official repository: https://github.com/google-research/timesfm
- STUMPY documentation: https://stumpy.readthedocs.io/en/latest/index.html
- aeon transformations documentation: https://www.aeon-toolkit.org/en/latest/api_reference/transformations.html

## 10. uv 复现命令

创建 `.venv`：

```bash
uv venv .venv --python /data/junjieqiu/miniforge3/envs/tools/bin/python --system-site-packages --clear
```

安装依赖时，建议显式取消 `.bashrc` 里的日本代理，再走阿里云源：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  uv pip install --python .venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple \
  -e external/chronos-forecasting -e external/timesfm \
  'transformers>=4.49,<5' 'accelerate>=0.34,<2' 'scikit-learn>=1.6,<2' stumpy aeon umap-learn
```

当前依赖也已写入 `pyproject.toml`，并生成了 `uv.lock`：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  uv lock --index-url https://mirrors.aliyun.com/pypi/simple
```

重跑 smoke test：

```bash
.venv/bin/python scripts/smoke_temporal_primitives.py
```
