# 腾讯云 Cloud Studio T4 部署步骤

## 1. 建立工作空间并同步代码

由用户在腾讯云选择高性能 GPU 工作空间，核实实际设备、计费规则和存储持久性。官方文档列举通常为 T4、16GB 显存的规格，但实际分配必须用 nvidia-smi 核验，不能凭套餐名认为已拿到 T4。本阶段未开通或操作腾讯云实例。

本代码仓库按用户授权公开，可以直接克隆，不需要个人访问 token。权重和数据仍需另行授权并使用独立传输通路；不要把凭据写入仓库、克隆 URL 或终端命令。工作空间可能在停止/重建后丢失临时文件，权重与新结果另存持久存储。使用前请阅读根目录 DISCLAIMER.md 和 LICENSE_NOTICE.md。

```bash
git clone --branch codex/cloudstudio-t4-20261011 https://github.com/rodriguezrobertbfrkx6857-sudo/tcm-sizhen-ai-cloudstudio-t4.git
cd tcm-sizhen-ai-cloudstudio-t4
```

## 2. 安装环境与检查

建议 Linux x86_64、Python 3.11（支持3.10–3.12）、PyTorch 2.8.0 + torchvision 0.23.0 的 CUDA 12.6 wheel。主机驱动由腾讯云提供，建议至少对应 CUDA 12.6 的 R560 驱动，并通过实际 kernel 测试；不在工作空间自行安装/替换主机驱动。nvidia-smi 的 CUDA Version 表示驱动支持上限，torch.version.cuda 表示 wheel 运行时，两者不必相等。无需为运行 wheel 单独安装完整 CUDA 编译工具链。

```bash
nvidia-smi
bash scripts/bootstrap.sh
source .venv/bin/activate
python -m cloudstudio_tongue.preflight --require-gpu --output outputs/preflight_01.json
python -m unittest discover -s tests -v
```

bootstrap 不下载模型、不启动训练。前置检查要求 T4、sm_75 支持、至少6GiB可用显存、至少20GiB空闲磁盘且无其他 compute 进程；如果不满足，报告资源问题，不停止别人任务。首次 preflight 只核资源，不能替代 CUDA 数值验证。T4 使用 FP16 或 FP32，禁用 BF16/TF32 方案；不迁移 Qiora 8B BF16 训练到16GB配置。

## 3. 分开迁移权重

**首选当前六类 YOLO11n：** 安装 `python -m pip install -r requirements-yolo.txt` 后，只迁移 `assets/yolo/best.pt`（5,445,914字节）。如 import cv2 报缺 libGL，使用平台包含libGL的Python镜像或由用户在自己的工作空间安装libgl1，不修改宿主驱动。Ultralytics 8.3.200会按本地已知SHA先核验可信checkpoint，拒绝任意第三方pickle；不自动下载YOLO基础权重。包内只保存清单，不保存payload。

```bash
python -m cloudstudio_tongue.yolo_infer --assets assets --image /path/to/authorized_image.jpg --device cuda --out outputs/yolo_infer_01
```
保持当前网站imgsz640、conf0.25、iou0.5、class-aware NMS、FP32条件；JSON含六类舌象和检测框，overlay用本地Pillow绘制，不下载字体。健康舌是原来源标签，不是医学健康结论。

真实YOLO来源VAL评测（仅目录型本地YAML，不允许download、外部路径或锁定TEST）：
```bash
python -m cloudstudio_tongue.yolo_evaluate --assets assets --data-root /path/to/authorized_dataset --data-yaml /path/to/authorized_dataset/data.yaml --confirm-authorized-data --device cuda --out outputs/yolo_val_01
```
来源验证不等于患者独立测试。此命令未在本阶段对真实图像执行，未上传任何图像。下面旧组件是可选参考路线。

若还需要旧二分类/分割组件，经用户确认后把其资产放入独立的 assets 目录（Git忽略）：
```
assets/g1/best.pt
assets/roi/roi_selected.pt
assets/dino/model.safetensors
assets/dino/config.json
assets/dino/preprocessor_config.json
```
通过控制台上传、私有对象存储或授权文件传输。不要上传本地整盘、.env、SSH 私钥、缓存或病人照片。本包始终校验大小和 SHA，拒绝缺失/不匹配文件；所有模型严格 state_dict 加载，weights_only=True，不执行下载或旧母体合并。

## 4. 最小 CUDA 工程测试（非正式训练）

```bash
python -m cloudstudio_tongue.smoke --assets assets --device cuda --steps 2 --out outputs/t4_smoke_01
```
实际冻结 DINO 编码器+ROI 进行合成输入推理；仅内存中 G1 分类头最多2次合成标签 SGD 更新，用新临时文件检查保存/加载/optimizer/RNG。FP16编码器/ROI、FP32头，报告实际loss、耗时和峰值显存。原权重只读且SHA保持。合成结果不是医学准确度，生成的 head checkpoint 标记 NOT_RELEASE，不能部署为新研究模型。首次 CUDA 失败不得写为通过。

## 5. 真实模型推理

使用已获准研究的非隐私示例或用户单独批准的输入：
```bash
python -m cloudstudio_tongue.infer --assets assets --image /path/to/authorized_image.jpg --device cuda --out outputs/infer_01
```
输出两类概率与384x384全帧mask。mask可以按全帧拉伸叠回原图；不据此做病灶测量。推理默认FP32以保持原CPU参考；FP16只在smoke中测。不能把本包的独立组件输出宣称为完整旧网站或完整研究报告管线。

## 6. DEV 评测

没有随仓库上传数据。只读通过资格和隐私检查的DEV manifest；每行结构为：id、split='DEV'、use_authorized=true、相对 image、可选相对 mask、可选 coating_label='white_coating'/'yellow_coating'。无标签或UNKNOWN只统计排除，不映射到阴性。split、来源/主体隔离仍需实际数据清单证明。

```bash
python -m cloudstudio_tongue.evaluate --assets assets --data-root /path/to/authorized_dev --dev-manifest /path/to/authorized_dev_manifest.jsonl --confirm-authorized-data --device cuda --out outputs/dev_01
```
输出已知二分类条件准确率、混淆矩阵和已给定mask Dice，非临床验证。TEST2禁止由本入口读取；没有独立测试成绩。本阶段不执行正式训练。未来正式视觉训练需额外确认合格数据、预算、持久备份和恢复计划，不能直接从两步smoke扩大训练。

## 官方环境依据

- [腾讯云 Cloud Studio 购买指南](https://cloud.tencent.com/document/buy-guide/1039/33503)
- [腾讯云 CPU/GPU 算力切换](https://cloud.tencent.com/document/product/1039/131806)
- [PyTorch 官方版本安装矩阵](https://docs.pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA 12.6 发布说明](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/)
- [NVIDIA T4 compute capability](https://developer.nvidia.com/cuda/gpus)

本地CPU验证环境与以上云端依赖固定版本并非同一环境；真实日志记录本地版本，不能据本地通过宣称云端组合已验证。
