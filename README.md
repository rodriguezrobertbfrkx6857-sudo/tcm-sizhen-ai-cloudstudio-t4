# Qiora / TongueAI · AMD ROCm 主环境本地整合

本分支准备在 AMD 上完成舌象视觉、Qiora 语言模型可行性、评测与备份。腾讯云 T4 保留为独立备用分支，不要求两云同时运行。

**状态：Pro ROCm 训练后端已加入；本次在 Windows 验证，未执行新的 AMD 8B 训练。用户已报告母体 BF16 云端推理成功，Pro 尚未验收。**

- [Pro 8B 训练、配对 DEV、恢复命令](docs/AMD_PRO_TRAINING.md)

Pro 的 `qiora_rocm.train` 使用私有 revision_03 数据及 SHA 清单，真实更新 Domain/Evidence；Router 暂不训练。数据、权重和凭据不在公开仓库。旧视觉说明与预检证据保留为历史范围，新训练范围以上述操作单为准。

- [AMD 简洁云端操作单](docs/dual_gpu/CLOUD_QUICKSTART.md)
- [真实状态、任务与语言模型可行性设计](docs/dual_gpu/AMD_TASK_PLAN.md)
- [源代码及原生 Qiora 复制 SHA](docs/dual_gpu/SOURCE_CODE_PROVENANCE.json)
- [免责声明](DISCLAIMER.md) / [许可说明](LICENSE_NOTICE.md)

默认不安装依赖、不下载权重、不启动训练。AMD 的 `scripts/bootstrap.sh` 已禁用 CUDA 安装；只读探测使用 `python3 scripts/amd_rocm_probe.py --workspace /mnt/workspace --out outputs/01_inventory.json`。短小 GPU 测试及模型传输需用户先批准。

本地完整 CPU 测试（已有 torch、torchvision、transformers、peft 等环境时）：

```bash
python scripts/validate_local.py --out /path/to/new_cpu_validation_run
```

微型随机 Qwen3 使用复制的原生 Domain/Evidence LoRA + Sequence-level Router，只验证工程结构；不能当成 Qiora 8B、已训练专家或临床能力证明。Case仍inactive，Semantic未启用。母体14文件16,397,459,696字节；权重与数据不在Git，TEST2不在本分支。

**T4用户请另行克隆原 `codex/cloudstudio-t4-20261011` 分支。** 下文为原视觉迁移包介绍及历史 CPU 证据，其中 T4 环境安装命令仅适用于原分支。

---

# TongueAI · 舌象视觉研究与 Cloud Studio T4 部署

CM-Sizhen-AI / TongueHerb（岐黄智镜）的独立视觉组件部署准备版本，面向 NVIDIA T4 16GB 的环境检查、模型加载、推理和评测复现。公开工程代码与验证记录；不包含原始图像、医疗记录、模型权重或访问凭据。

> **研究用途：尚未完成临床验证，不可用于诊断、处方或治疗决策。输出可能错误；“健康舌”只是来源数据的类别名称，不表示医学上的健康结论。请先阅读 [免责声明](DISCLAIMER.md) 和 [许可说明](LICENSE_NOTICE.md)。**

## 实际功能与验证状态

| 组件 | 功能 | 当前状态 |
|---|---|---|
| YOLO11n（首选） | 六类舌象外观检测及可视化 | 本地 CPU 原始输出一致性通过；T4 CUDA 待验证 |
| DINOv2-small 旧参考组件 | 白苔 / 黄苔二分类 | 本地 CPU 与原组件输出一致；云端待验证 |
| ResNet18-UNet 旧参考组件 | 舌体区域 mask | 本地 CPU 与原组件输出一致；不作病灶判断 |
| 工程工具 | 资源检查、SHA 核验、DEV 评测、合成输入保存恢复测试 | 本地合同测试 6 项通过；真实 T4 测试待执行 |

六类来源标签为健康舌、红舌、瘦舌、白苔舌、黄苔舌、黑苔舌。当前 YOLO 来源训练记录为 TRAIN 4,222 / VAL 896，历史来源 VAL mAP50 0.5365、mAP50–95 0.4896。**这些不是本次云端复测成绩，也不是患者独立或临床验证结果。** 见 [真实训练摘要](docs/CURRENT_YOLO_TRAINING_SUMMARY.json)。

## 项目范围

首选入口是当前网站使用的 **YOLO11n 六类舌象模型**，其640像素、conf0.25、IoU0.5条件保持不变。另提供旧 DINOv2-small 舌苔白/黄二分类、ResNet18-UNet 舌体 mask 的可移植加载、推理、DEV 评测及两步合成输入训练测试。保留原模型 SHA 与来源绑定，不重新训练原模型。舌体分割和全图分类分别运行，本包未重建网站、四诊合参或对话模型。Pro/Max 训练入口没有纳入迁移包。

**准备状态：本地代码与权重 CPU 验证完成后见 docs 中真实记录；Cloud Studio CUDA 和指定依赖组合仍需云端验证。** 不宣称临床能力或云端已经部署。

- [Cloud Studio 部署步骤](docs/CLOUD_STUDIO_DEPLOY.md)
- [真实项目审查及资源计划](docs/MIGRATION_REVIEW.md)
- [后续权重及数据迁移](docs/ASSET_TRANSFER_PLAN.md)
- `configs/assets_manifest.json`：5 个必要资产的相对目标路径、字节数与 SHA。
- `configs/t4_profile.json`：224/384 分辨率、batch 1、两步上限及资源检查。
- `cloudstudio_tongue/`：路径无关加载器与命令入口。
- `reference/train_roi_legacy.py`：原 ROI 训练源码，只供追溯；不能用来绕过原数据/授权 Gate。

## 快速开始

公开代码可直接克隆，权重与数据另行授权、单独迁移：

```bash
git clone --branch codex/cloudstudio-t4-20261011 https://github.com/rodriguezrobertbfrkx6857-sudo/tcm-sizhen-ai-cloudstudio-t4.git
cd tcm-sizhen-ai-cloudstudio-t4
```

环境检查：
```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
python -m cloudstudio_tongue.preflight --require-gpu --output outputs/preflight_01.json
python -m unittest discover -s tests -v
```
此时没有模型权重，不能运行真实模型。按资产清单单独迁移后再运行推理或合成 smoke；没有自动下载、后台训练或自动租赁。

当前YOLO模型另需 `pip install -r requirements-yolo.txt`，资产清单是 `configs/yolo_assets_manifest.json`。只迁移这一份约5.19MiB的权重就可验证当前六类组件，不必先迁移225MiB的旧组件资产。

## 资源与研究边界

目标环境为 Linux x86_64、Python 3.11、PyTorch 2.8.0 / CUDA 12.6。T4 使用 FP32 或 FP16，不采用 BF16。建议至少 30 GiB 存储；预检要求 20 GiB 空闲磁盘与 6 GiB 可用显存。推理约 2–4 GiB、小规模视觉训练约 4–8 GiB **仅为规划估计，尚非 T4 实测值**。

本次没有正式视觉训练或 Pro/Max 训练，没有改动原模型与历史结果。两步合成标签更新只验证工程链路，不表示医学能力提升。本包的模型只提供来源受限的研究视觉输出，不能据此判断证候、疾病或处方。未知/未标注类别不转成阴性。

## AMD Pro、Max 和 Vision 训练入口

Pro 使用 [AMD_PRO_TRAINING.md](docs/AMD_PRO_TRAINING.md)。独立 Max reading 研究与六类视觉训练见 [AMD_MAX_VISION.md](docs/AMD_MAX_VISION.md)。新增模块不会修改 `qiora_rocm` 的训练代码身份；正在运行的任务保持固定版本。Max layer Router 目前未训练，Vision 与 Max 的真实 AMD 训练结果尚待云端验证。34项CPU工程测试已通过，不能代替真实GPU能力评测。

## 隐私、许可与贡献

欢迎提交不含隐私的工程问题和复现记录。Issue、PR 和日志中请勿提交舌像、病例、身份信息、API Key、SSH 私钥或账户凭据。现有模型许可用途保持为非商业研究；公开迁移代码不会扩大底层数据或权重许可。仓库当前未授予统一开源许可，第三方内容分别遵循其原有许可，具体见 [LICENSE_NOTICE.md](LICENSE_NOTICE.md)。
