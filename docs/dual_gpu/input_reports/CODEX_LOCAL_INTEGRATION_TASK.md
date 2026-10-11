# 交给本地 Codex 的阶段整合指令

请在 `D:\TCM-Sizhen-AI` 中只读审查完整的本地项目最新状态、最后训练/评测报告和已保存模型，再阅读当前交接包所有 Markdown、配置、脚本和测试。

**本轮边界：不启动魔搭/腾讯云 GPU，不执行 GPU 训练、不下载大型模型，不接触 TEST2，不覆盖旧权重和冻结实验。** 只能在独立工作分支整合公开且不含隐私的 AMD ROCm 预检代码，保护原 T4 分支与原项目工作树。

顺序：

1. 对照 `CLOUD_GPU_RESOURCE_REVIEW.md`，补齐只有 Windows 本地才能核验的最新仓库、权重、数据与状态；发现报告不符时以真实文件为准并记录差异。
2. 将 `scripts`、`configs` 的独立 AMD 组件部署到一个新工作目录/独立 git 分支，绝对不要在 AMD 执行原 T4 的 `requirements-cu126.txt` / `scripts/bootstrap.sh`。
3. 运行 `python -m unittest discover -s tests -v` 与只读探测器；检查合成/假数据时不得触发网络下载。
4. 用 `prepublish_guard.py` 检查准备发布的 git 文件并人工复核，不上传资产、医学数据、私钥、token 和历史权重；原 T4 默认分支不改。
5. 若准备推送 AMD 新分支，记录具体 commit、修改清单和发布 SHA；先获得用户确认远程发布变更再执行。
6. 根据 `CLOUD_AGENT_HANDOFF.md` 输出分开的 AMD / T4 最小云端执行单，要求用户手动启动平台，再有限执行设备探测与合成数值测试。
7. 根据真实机器测到的 GPU gfx、内存、存储和速度，更新 `GPU_EXPERIMENT_PLAN.md`；不要将 100h 作为训练授权。
8. 完成阶段报告、代码清单、测试日志、已准备好/尚阻塞列表，标记 `CLOUD_GPU_PREFLIGHT_REQUIRES_USER_ACTION`。任何真实模型训练需要独立批准。

优先确保用户后续可以：**Windows Codex 审核 → GitHub 公开代码分支 → 两台云机器匿名拉取代码 → 经授权私传模型 → 云端受控验证 → 结果回传 Windows**。若发现公开代码不能安全发布或云平台持久性未知，明确 HOLD，不伪称通过。
