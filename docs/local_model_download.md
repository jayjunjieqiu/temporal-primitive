# 本地模型权重下载默认流程

本仓库默认使用 `hf-mirror` 提供的 `hfd.sh` 脚本下载 Hugging Face 模型权重。服务器 `.bashrc` 里可能配置了 HTTP proxy；下载模型权重时默认绕开这些 proxy，并显式使用 `HF_ENDPOINT=https://hf-mirror.com`。

## 1. 下载 hfd.sh

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  wget -q -O /tmp/hfd.sh https://hf-mirror.com/hfd/hfd.sh

chmod +x /tmp/hfd.sh
```

## 2. 下载三个 TSFM baseline 权重

在仓库根目录 `/data/junjieqiu/temporal-primitive` 下运行：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  HF_ENDPOINT=https://hf-mirror.com \
  /tmp/hfd.sh amazon/chronos-2 \
  --local-dir chronos-2 \
  --tool aria2c -x 8 -j 4
```

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  HF_ENDPOINT=https://hf-mirror.com \
  /tmp/hfd.sh autogluon/chronos-2-small \
  --local-dir chronos-2-small \
  --tool aria2c -x 8 -j 4
```

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  HF_ENDPOINT=https://hf-mirror.com \
  /tmp/hfd.sh google/timesfm-2.5-200m-pytorch \
  --local-dir timesfm-2.5-200m-pytorch \
  --tool aria2c -x 8 -j 4
```

注意：`Chronos-2-small` 的 Hugging Face repo id 是 `autogluon/chronos-2-small`，不是 `amazon/chronos-2-small`。

## 3. 验证

```bash
test -f chronos-2/config.json
test -f chronos-2/model.safetensors
test -f chronos-2-small/config.json
test -f chronos-2-small/model.safetensors
test -f timesfm-2.5-200m-pytorch/config.json
test -f timesfm-2.5-200m-pytorch/model.safetensors
du -sh chronos-2 chronos-2-small timesfm-2.5-200m-pytorch
```

参考大小：

- `chronos-2/`: 约 456M
- `chronos-2-small/`: 约 107M
- `timesfm-2.5-200m-pytorch/`: 约 883M

## 4. Git policy

这三个目录是本地模型权重目录，已在 `.gitignore` 中整体忽略：

- `chronos-2/`
- `chronos-2-small/`
- `timesfm-2.5-200m-pytorch/`

不要把模型权重提交到 GitHub。

## 5. uv 环境恢复注意事项

本服务器的 shell 可能默认带有 HTTP proxy。恢复 `.venv` 时也建议显式取消 proxy，否则 `uv sync` 可能卡在包下载阶段。

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
  uv sync
```

`sklearn` 的 Python import 由 `scikit-learn` 包提供。本仓库 `pyproject.toml` 已经依赖 `scikit-learn`，不要额外添加 deprecated 的 `sklearn` PyPI 包。
