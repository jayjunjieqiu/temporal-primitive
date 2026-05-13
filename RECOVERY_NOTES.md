# Recovery Notes

本目录由 Codex 会话日志中的 `apply_patch` 记录重放恢复得到。

恢复时间：2026-05-13

## 已恢复

- proposal、中文 reports、PPT material markdown
- configs
- scripts
- AGENTS.md、pyproject.toml、.gitignore

## 未完全恢复

- 本地模型权重目录：`chronos-2/`、`chronos-2-small/`、`timesfm-2.5-200m-pytorch/`
- external 代码库：`external/chronos-forecasting`、`external/timesfm`
- 运行产物与图片：`outputs/`
- `.venv`
- 原始 `.git` 历史

## 验证状态

- 已对关键脚本执行 `python -m py_compile`，语法检查通过。
- 由于 outputs 和模型权重未恢复，PPT 图片重生成与 TSFM 实验脚本需要重新准备依赖后再运行。

## 建议

先把该恢复目录作为新的工作副本使用，并尽快重新配置远程仓库或备份。
