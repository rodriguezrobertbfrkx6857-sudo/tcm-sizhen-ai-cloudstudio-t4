# 本阶段实际验证记录

日期：2026-10-10（本文由工具执行形成）。

## 执行与未执行边界

- 使用 GitHub 已连接账户只读核验公开仓库默认分支、HEAD 及目录树，读取关键 README、环境与模型资产清单、CPU 对照与 DEPLOYMENT_STATUS。
- 使用可访问的历史上传交接文档对照 Qiora Pro / KER 最近 CPU 修复状态与 TongueAI 视觉历史记录。
- 阅读 AMD、PyTorch、Ultralytics 官方文档，对 ROCm Python 语义与轮子兼容性进行外部核对。
- 未访问用户 Windows `D:`，未读取本地真实权重 payload，未开启云 GPU、未安装 wheel、未使用云配额、未修改 GitHub 仓库。

## 实际本地脚本测试

- `python -m unittest discover -s tests -v`：**5项 PASS**，含资产 SHA 正确/错误用例、路径越界防护、profile 禁训练、无 Git 仓库阻断、CPU only probe。
- `python scripts/amd_visual_smoke.py --device cpu`：合成 ConvNet 推理 **PASS**，torch 版本 `2.10.0+cpu`、HIP `null`、无需模型下载、无 optimizer 更新。这是工具环境而非魔搭环境。
- `python scripts/amd_rocm_probe.py --workspace .`：已调用（只读），工具执行环境不是 AMD ROCm；不产生硬件通过结论。

## GitHub 实际发现

- 默认分支：`codex/cloudstudio-t4-20261011`。
- HEAD：`3374e235077b5f02d4de5bcb1c2444362a1837c5`。
- 43 个 Git 树条目（包括目录）。
- `requirements-cu126.txt`/原 bootstrap 会安装 CUDA 12.6 PyTorch；已在新方案中明确禁止 AMD 复用。
- 当前六类 YOLO11n 权重 SHA pinned，但权重仅在用户本地，不在公开 GitHub。

## 待用户实际云端确认事项

GPU 名称、gfx target、是否单卡独享及192GB显存实际可用数、ROCm/PyTorch构建、`/mnt/workspace` 容量及保留规则、Notebook超时及100h计费真实规则、Agent功能、权重安全传输通路、数值forward/backward、模型加载和恢复验证。以上均不能从截图或 CPU 测试中填入 PASS。
