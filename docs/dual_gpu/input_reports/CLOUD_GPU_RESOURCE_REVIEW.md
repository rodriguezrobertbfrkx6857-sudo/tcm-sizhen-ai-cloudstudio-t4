# CM-Sizhen-AI / TongueAI — 双云 GPU 资源与项目审查

**日期：2026-10-10；性质：离线审查，非云端实测。**

## 证据边界

- **已实际读取**：用户此前上传的 `FINAL_HANDOFF_FOR_CHATGPT(20261010-142314).md`、`FINAL_HANDOFF_FOR_CHATGPT(20261011-014304).md`，以及其他可见项目交接摘要；使用 GitHub 连接读取公开仓库、分支、README、清单、发布状态、迁移报告和部署脚本。
- **用户提供、尚未远端复核**：魔搭提供 192GB 显存、200GB RAM、8 核、ROCm 7.2.3 / Python 3.12、剩余约 100 小时；腾讯云 T4 16GB、32GB RAM、8核、每运行小时消耗1.2机时。
- **本阶段未访问**：Windows `D:\TCM-Sizhen-AI` 的完整工作树、魔搭/腾讯实时实例、GPU运行时和云账单。不能以历史交接代替实时状态，也不能以公开仓库代表完整中医项目。

## 公开 GitHub 仓库核验

- URL：https://github.com/rodriguezrobertbfrkx6857-sudo/tcm-sizhen-ai-cloudstudio-t4
- 默认分支（通过 GitHub 连接核实）：`codex/cloudstudio-t4-20261011`
- 审查时该分支 HEAD：`3374e235077b5f02d4de5bcb1c2444362a1837c5`
- 仓库为 `public`，非归档；不是完整的 CM-Sizhen-AI / Qiora 仓库。
- 检索了 43 项 Git 树记录（文件和目录）。包含 `cloudstudio_tongue/`、`configs/`、`scripts/`、`tests/`、部署文档、SHA 清单。
- `docs/DEPLOYMENT_STATUS.json` 显示 **本地 CPU 原始输出一致性已通过，云端 CUDA 尚未测、未上传模型权重、未上传数据、未启动云实例，Pro/Max训练为 false**。
- `requirements-cu126.txt` 固定 PyTorch 2.8.0 / torchvision 0.23.0、CUDA 12.6 wheel。**严禁在 ROCm 上复用**；`scripts/bootstrap.sh` 也明确安装 CUDA wheel，必须独立分流。
- YOLO 六类当前模型：`5,445,914` 字节，SHA256 `34ce40cd05168ff0cd637ee2e9d0dc7657dee1da47321be3bc37ad60492818b4`；权重未公开上传。旧参考组件完整五文件约 225.38MiB。
- 原来源 YOLO VAL：TRAIN 4,222、VAL 896，mAP50 ~0.5365 / mAP50-95 ~0.4896；这是既有训练记录，非 T4/AMD 实测、非患者独立或临床结果。
- **仓库未提供统一开源许可证**，分离的工程贡献/第三方许可不能据公开可见性推定拥有商业再分发权。

## 项目研究线状态（来自历史上传交接，不是本次云端验证）

| 模块 | 有依据的当前阶段结论 | 本次是否实际核验 GPU |
|---|---|---|
| TongueAI 六类 YOLO11n | 有本地训练权重与原始输出 CPU 对照证据；临床 Gold 无 | 否 |
| DINOv2-small 旧二分类 | 可供基线参考；已有 CPU 对照；不能当六类模型替代 | 否 |
| ResNet18-UNet ROI 参考 | 历史分割权重与 CPU 对照；不用于病灶诊断 | 否 |
| Qiora BF16 | 历史 54 万医疗 QA 领域适配后的合并模型，已完成过外部恢复证明；大权重未包含在公开迁移仓库 | 否 |
| Qiora Pro / KER | Domain / Evidence 研究权重存在；未通过 Pro 初版能力验收。最近 compact-v3 修复属于 CPU 研究，GPU 后端未接入 | 否 |
| 学习 Router / Max | Router 仍未形成通过门槛的训练成品；Max 只有 CPU 研究设计 | 否 |
| 闻诊 / 切诊 / 多模态 | 以概念/初期研究为主，不得宣称完成或临床有效 | 否 |

上述“当前”截至可读取的交接版本，不保证 Windows 项目此刻完全未更新；Codex 接手后必须先只读复核新文件与阶段状态。

## 双资源实际分工建议

| 资源 | 优先级 P0 / P1 | 待实测后 P2 | 不建议 |
|---|---|---|---|
| **腾讯云 T4 16GB** | 已有视觉组件 CUDA preflight、YOLO 真实权重合成输入推理、旧组件基础推理、DEV 对照 | 小 batch 视觉训练与回归评测 | 直接加载完整 BF16 8B Qiora 并训练 Pro、BF16 路线 |
| **魔搭 AMD 192GB（声明容量）** | GPU 型号/gfx/ROCm/PyTorch、磁盘持久性、短小数值测试 | Qiora BF16 8B 推理、LoRA、小规模视觉与多模态比较 | 未识别 GPU 前声称 MI300X；未经验证部署 CUDA wheel、FlashAttention/CUDA-only kernels |
| **Windows 本地** | Codex 主仓、设计、代码审阅、隐私及 SHA 清单、标准训练/评测身份管理 | 小型 CPU 检查与实例外恢复 | 向公共 GitHub 上传患者图片/病历/受限数据/私钥 |

## 对显存和容量的审慎估计

- Qiora BF16 8B 光权重约 16GB，历史合并母体的模型文件约 16.4GB；**训练还需要激活、优化器、模型副本/缓存、并行内核等，不以权重大小等同 VRAM 需求**。
- T4 16GB 理论显存小于完整 8B BF16权重及运行开销的实用需求，不作为 Pro 主训练资源。
- AMD 192GB（用户界面显示）可能有较大余量，但未知 GPU 名称、是否独享、实际剩余显存、GPU 利用率、驱动、存储配额、断点保留期、吞吐；本阶段一律标记 UNVERIFIED。
- 魔搭相关公开教程提到 `/mnt/workspace` 下 100GB 持久空间，**这是平台教程示例，不能据此确认本次 AMD 192GB 实例也有 100GB**。使用 `df -h`、挂载检查与停止/再开保留政策确认。

## 即刻决策

**完成本地只读准备后，首次云端操作仅做环境和合成输入检查，不下载大权重、不启动正式训练。**
首个实测顺序建议：AMD 环境探测 → T4 YOLO 小权重可重复推理（如果 T4 有额度）→ AMD 可信 YOLO 小权重验证 → 资源许可充足后再单独批准 Qiora BF16 传输与加载。

## 一手出处

- 用户仓库：上述 GitHub URL，分支 `codex/cloudstudio-t4-20261011`；具体参考 `README.md`、`docs/DEPLOYMENT_STATUS.json`、`docs/MIGRATION_REVIEW.md`、`docs/CURRENT_YOLO_TRAINING_SUMMARY.json`。
- AMD ROCm 7.2.3兼容矩阵：https://rocm.docs.amd.com/en/docs-7.2.3/compatibility/compatibility-matrix.html
- PyTorch HIP 行为：https://docs.pytorch.org/docs/stable/notes/hip.html
- Ultralytics AMD 使用：https://docs.ultralytics.com/integrations/amd
- 魔搭 Notebook 公开教程：https://github.com/modelscope/ms-cookbook/blob/main/content/chapters/chapter-04.md
