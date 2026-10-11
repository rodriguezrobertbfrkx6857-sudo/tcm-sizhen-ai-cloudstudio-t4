# Qiora × AMD ROCm × Cloud Studio T4 — 本地阶段交接包

截至 2026-10-10，本包是**离线审查与准备产物**，不是已部署的 ModelScope/T4 仓库变更，也不是任何新模型成果。

## 6 份正式交付

1. `CLOUD_GPU_RESOURCE_REVIEW.md` — 项目与两张 GPU 对照、已核实/待实测状态。
2. `AMD_ROCM_MIGRATION_PLAN.md` — 独立 AMD 迁移方案、PyTorch/ROCm/Ultralytics/LLM 依赖门槛。
3. `GPU_EXPERIMENT_PLAN.md` — 100h 条件预算、优先级、实验假设与停止条件。
4. `CLOUD_AGENT_HANDOFF.md` — AMD/T4 云端 Agent 可复制的 **仅预检** 操作。
5. `CLOUD_ASSET_TRANSFER.md` — GitHub 代码与模型/数据分离、SHA、下载回传和中断恢复。
6. `README.md` — 交付索引与限制。

附：`scripts/amd_rocm_probe.py`、`amd_visual_smoke.py`、`asset_verify.py`、`prepublish_guard.py`、`rocm_env_isolated.sh`，配置及单元测试。

## 本地 CPU 测试

```bash
python -m unittest discover -s tests -v
python scripts/amd_rocm_probe.py --workspace . --out outputs/local_readonly_probe.json
```

这两条不会下载模型、连接云、执行优化器更新。第二条可能显示无 ROCm GPU，属于预期的本机环境状态。

## 重要执行约束

- **没有推送/修改公开 GitHub 仓库**；建议下一步由本地 Codex 创建 `codex/modelscope-rocm-preflight-20261010` 新分支，审核后提交本包合适脚本/文档。
- **没有开启云资源，没有使用免费额度，也没有运行 CUDA/ROCm 数值实验**。
- 用户给出的 AMD 192GB/100h 与 T4 配额均未实际登录验证。
- 本包不会安装任何 wheel，不会覆盖原系统环境；T4 的 `requirements-cu126.txt` 禁止在 ROCm 上执行。
- 完整项目 Windows 工作树不可直接在当前执行环境访问；项目状态依据历史用户交接和实读 GitHub 文档，不能宣称已审核 Windows 此刻全部文件。
- Qiora Pro 仍未通过综合模型初版验收，Max 没有完成正式训练；权重及 TEST2 不在本 ZIP 内。

## 下一步建议

**先以最少机时启动 AMD 的环境只读查询；若 ROCm 确实可用，运行小张量 finite 前后向。与此同时 T4 若有额度，跑六类 YOLO 可信权重最小 CUDA 推理。** 两条测试都通过并保证持久存储和实例外回传后，才单独批准较大模型和训练资产迁移。
