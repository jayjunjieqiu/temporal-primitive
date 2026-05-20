# Recovery Notes

本目录由 Codex 会话日志中的 `apply_patch` 记录重放恢复得到。

恢复时间：2026-05-13

## 已恢复

- proposal、中文 reports、PPT material markdown
- configs
- scripts
- AGENTS.md、pyproject.toml、.gitignore

## 未完全恢复

- external 代码库：`external/chronos-forecasting`、`external/timesfm`
- 运行产物与图片：`outputs/`
- `.venv`
- 原始 `.git` 历史

## 本地权重恢复

本地模型权重已重新下载到：

- `chronos-2/`
- `chronos-2-small/`
- `timesfm-2.5-200m-pytorch/`

默认下载方式见 `docs/00_local_model_download.md`。核心原则是使用 `hf-mirror` 的 `hfd.sh`，并在下载命令中显式：

- `HF_ENDPOINT=https://hf-mirror.com`
- `unset`/`env -u` 所有 `http_proxy`、`https_proxy`、`HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`、`all_proxy`

其中 `Chronos-2-small` 的 repo id 是 `autogluon/chronos-2-small`。

## 验证状态

- 已对关键脚本执行 `python -m py_compile`，语法检查通过。
- 模型权重已恢复。
- `external/chronos-forecasting` 和 `external/timesfm` 已重新 clone。
- `.venv` 已通过无 proxy 的 `uv sync` 恢复；`sklearn` import 由 `scikit-learn` 提供。
- PPT 图片重生成仍需要重跑中间 `outputs/`。

## 建议

先把该恢复目录作为新的工作副本使用，并尽快重新配置远程仓库或备份。
