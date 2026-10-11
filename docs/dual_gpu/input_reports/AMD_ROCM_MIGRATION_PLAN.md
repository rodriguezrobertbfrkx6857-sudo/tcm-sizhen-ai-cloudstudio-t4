# AMD ROCm 迁移与兼容性计划

**仅为本地准备与目标云端预检方案，尚未在魔搭 GPU 上执行。**

## 1. 与 T4 完全隔离

- 不编辑、覆盖原 `codex/cloudstudio-t4-20261011` 分支；新代码建议由本地 Codex 在另一个分支（例如 `codex/modelscope-rocm-preflight-20261010`）审阅后再发布。**本交接没有创建/推送远程分支。**
- **不能执行** T4 原仓库的 `bash scripts/bootstrap.sh` 或 `pip install -r requirements-cu126.txt`，因为它会安装 NVIDIA CUDA wheel。
- 优先使用预装 ROCm/PyTorch，任何缺依赖先只读记录；只有经确认且在隔离的 `.venv-rocm` 中才安装非 Torch 用户态依赖。禁止覆盖 ROCm 栈、宿主驱动、系统 Python。
- AMD Python 代码里 `torch.device('cuda')` 是正确的 HIP 接口；`torch.version.hip` 非空才说明 PyTorch 为 ROCm 构建。CUDA 字样不等于实际 NVIDIA。

## 2. 识别真实机器与镜像

首先在魔搭 Notebook / Code Editor 的终端执行以下只读命令（未执行前 GPU 型号仍未知）：

```bash
uname -a
python3 --version
command -v rocminfo && rocminfo | grep -E 'Name:|gfx[0-9]+' | head -80
command -v rocm-smi && rocm-smi --showproductname --showmeminfo vram
ls -ld /dev/kfd /dev/dri 2>/dev/null || true
python3 -c 'import torch; print(torch.__version__, torch.version.hip, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_GPU")'
df -h /mnt/workspace /tmp 2>/dev/null || true
mount | grep -E '/mnt/workspace|/dev/kfd' || true
```

运行附带脚本 `python3 scripts/amd_rocm_probe.py --workspace /mnt/workspace --out outputs/amd_inventory.json`。确认权限后经用户批准再添加 `--gpu-smoke` 做微型 tensor 前后向，并记录峰值显存和 finite loss。默认探测器不会进行 GPU 运算。

如果 `rocminfo` 不可用、`torch.version.hip is None`、`torch.cuda.is_available()` 为假、没有 `/dev/kfd` 权限，记录 FAIL 并停止该 GPU 路线，不能用 CPU 结果假冒加速成功。

## 3. 依赖兼容性分级

| 组件 | 初步判断 | 放行方法 |
|---|---|---|
| PyTorch + ROCm / torchvision | ROCm 官方支持；目标镜像版本未验证 | 直接读取 torch/torchvision、gfx、有限张量前后向。若不匹配，另建环境并按 AMD 7.2.3 对应 wheel 文档，禁止覆盖原镜像 |
| Ultralytics YOLO11n | 官方支持 ROCm 环境训练与推理，个别环境 AMP 可能有问题 | 合成 64×64 ConvNet → 本地可信 SHA YOLO11n 640 推理 → 若 AMP 出错改 FP32 并注明 |
| DINOv2/ResNet18-UNet | 主要由 torch/torchvision 组成，兼容性合理但未验证 | 先最小 batch 推理，再比较相同预处理输出 |
| Transformers + PEFT + Accelerate | 一般 PyTorch 路径有可行性，Qwen 具体算子要测 | 单独记录 tokenizer、SDPA/eager、LoRA 注入和 checkpoint 恢复，不引入第三方量化作为前置依赖 |
| `flash_attn` / Triton / Triton kernels | ROCm/GPU gfx/版本差异显著 | 先使用 PyTorch SDPA/eager，只有明确实测通过后启用优化实现 |
| `bitsandbytes` QLoRA | 新版支持部分 ROCm 架构，但不是所有预装环境都保证一致 | 默认关闭；先测试版本、gfx与小模型 4bit 参数数值与反向，再允许选为备选 |
| vLLM | 有 ROCm 支持路线，但需匹配镜像/架构/算子版本 | 推理独立实验，不能取代 HF+PEFT 训练验证 |
| 自定义 CUDA `.so` / cuDNN / NCCL | 不可直接复用 AMD | 标记 AMD_BLOCKED，单独考虑 HIP 或 RCCL，优先避开 |

**重要**：按照 ROCm 7.2.3 官方矩阵，Ubuntu 22.04 与多种 AMD CDNA/RDNA GPU 可用，但支持依赖于 **GPU/gfx target、镜像、驱动和 PyTorch 构建** 四者匹配。官方 wheel 文档为 https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/README.html；不要在未经验证的实例上复制最新 ROCm 10.x 的安装命令。

## 4. 最小测试门槛（递进，不是连续自动训练）

- **A0：纯只读环境检查**：保存 GPU name、gfx、ROCm/PyTorch、可用显存、设备数量、`/mnt/workspace` 持久性证据；输出 `amd_inventory.json`。
- **A1：小型数值测试**：一次有限矩阵前后向、Conv2d 推理，确认 `torch.version.hip` 非空、有限结果、峰值显存；可经批准的非训练测试。
- **A2：YOLO 可信权重短推理**：仅用此前本地 SHA 一致的 `yolo/best.pt` 与合成图片，固定 640、conf0.25、IoU0.5，不使用临床图片。
- **A3：框架兼容试验**：如果 A2 通过，再进行 Transformers、PEFT、BF16、SDPA、微型可恢复 state_dict 的数值测试。只做小张量/小模型，不下载 8B 基础权重。
- **A4：正式模型加载候选**：独立授权传输 Qiora BF16（约16.4GB）及必要 adapter，并检查 SHA、持久化容量、定量吞吐和备份回读，才能进行后续 Pro 训练讨论。

## 5. 重要工程措施

- 为 T4 与 AMD 分别维持需求文件/镜像清单；不跨设备复用 `pip freeze` 当作锁定文件。
- 严格禁止 `torch.cuda.empty_cache()` 作为修复 OOM 的替代方案；应记录峰值、batch/seq/fp16/BF16、内存碎片及真实过程。
- 不默认 BF16 有效；仅在实际设备、torch 数值测试通过后使用。T4 优先 FP32/FP16。
- ModelScope Code Editor/Notebook 可能有自动休眠；必须测试可持久路径、关机后文件是否保留、GPU 小时计算规则。不把 `/mnt/workspace` 自动当成 192GB 可用磁盘。
- Qiora-KER compact Evidence 仍需新 GPU 后端接入，**不以大显存可用表示 Pro 训练入口已经就绪**。

## 6. 本包脚本

- `scripts/amd_rocm_probe.py`：只读收集环境，显式 `--gpu-smoke` 才执行有限 GPU 数值测试。
- `scripts/amd_visual_smoke.py`：默认 CPU 合成 ConvNet，选择设备和可信 SHA 时可验证现有 YOLO11n 权重；无下载、无正式训练。
- `scripts/rocm_env_isolated.sh`：只读检查或显式创建 `--system-site-packages` 的隔离 venv，复用已安装 ROCm Torch，不安装 CUDA wheel。
- `configs/amd_rocm_profile.json`：禁训练、禁模型下载的初始状态。

## 7. 参考依据

- AMD ROCm 7.2.3 PyTorch：https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.3/install/3rd-party/pytorch-install.html
- AMD 7.2.3 兼容矩阵：https://rocm.docs.amd.com/en/docs-7.2.3/compatibility/compatibility-matrix.html
- PyTorch HIP 接口：https://docs.pytorch.org/docs/stable/notes/hip.html
- Ultralytics AMD：https://docs.ultralytics.com/integrations/amd
- bitsandbytes ROCm：https://huggingface.co/docs/bitsandbytes/installation
- Transformers SDPA：https://huggingface.co/docs/transformers/attention_interface
