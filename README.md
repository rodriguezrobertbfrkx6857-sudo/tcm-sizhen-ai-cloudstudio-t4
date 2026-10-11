# TongueAI · Cloud Studio T4 迁移包

CM-Sizhen-AI / TongueHerb 的独立视觉组件部署准备版本。私有代码仓库；不包含原始图像、医疗记录、模型权重或访问凭据。

首选入口是当前网站使用的 **YOLO11n 六类舌象模型**，其640像素、conf0.25、IoU0.5条件保持不变。另提供旧 DINOv2-small 舌苔白/黄二分类、ResNet18-UNet 舌体 mask 的可移植加载、推理、DEV 评测及两步合成输入训练测试。保留原模型 SHA 与来源绑定，不重新训练原模型。舌体分割和全图分类分别运行，本包未重建网站、四诊合参或对话模型。Pro/Max 训练入口没有纳入迁移包。

**准备状态：本地代码与权重 CPU 验证完成后见 docs 中真实记录；Cloud Studio CUDA 和指定依赖组合仍需云端验证。** 不宣称临床能力或云端已经部署。

- [Cloud Studio 部署步骤](docs/CLOUD_STUDIO_DEPLOY.md)
- [真实项目审查及资源计划](docs/MIGRATION_REVIEW.md)
- [后续权重及数据迁移](docs/ASSET_TRANSFER_PLAN.md)
- `configs/assets_manifest.json`：5 个必要资产的相对目标路径、字节数与 SHA。
- `configs/t4_profile.json`：224/384 分辨率、batch 1、两步上限及资源检查。
- `cloudstudio_tongue/`：路径无关加载器与命令入口。
- `reference/train_roi_legacy.py`：原 ROI 训练源码，只供追溯；不能用来绕过原数据/授权 Gate。

快速查看：
```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
python -m cloudstudio_tongue.preflight --require-gpu --output outputs/preflight_01.json
python -m unittest discover -s tests -v
```
此时没有模型权重，不能运行真实模型。按资产清单单独迁移后再运行推理或合成 smoke；没有自动下载、后台训练或自动租赁。

当前YOLO模型另需 `pip install -r requirements-yolo.txt`，资产清单是 `configs/yolo_assets_manifest.json`。只迁移这一份约5.19MiB的权重就可验证当前六类组件，不必先迁移225MiB的旧组件资产。

本包的模型只提供来源受限的研究视觉输出，不能据此判断证候、疾病或处方。未知/未标注类别不转成阴性。现有模型许可用途保持为非商业研究；迁移代码不会扩大底层数据或权重许可。
