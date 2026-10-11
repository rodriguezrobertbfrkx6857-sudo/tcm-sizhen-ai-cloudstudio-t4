# 云端 Agent 交接：AMD Notebook / NVIDIA T4

**执行阶段：环境确认与最小测试；无训练授权、无大型模型下载授权。** 本指令只供用户启动的已获权限实例使用；不要求云端 Agent 登录你的 GitHub 主账号，也不要求发送 SSH 私钥。

## AMD 端：严格按顺序执行

### A. 会话起始（平台 Notebook / Code Editor）

1. 在用户自己开启的魔搭 Notebook 中确认剩余机时、是否正在计费、停止保留规则，以及 Code Editor 内是否真的有可用 Agent。不要推测 Agent 一定存在。
2. 选择一个独立工作目录，首先验证 `/mnt/workspace` 是否挂载、可写，以及容量/关机保留政策。确认后才考虑保存任何模型。
3. 公共代码使用匿名 HTTPS 克隆，不在云端配置 GitHub 登录凭据：

```bash
git clone --branch codex/cloudstudio-t4-20261011 \
  https://github.com/rodriguezrobertbfrkx6857-sudo/tcm-sizhen-ai-cloudstudio-t4.git
cd tcm-sizhen-ai-cloudstudio-t4
git rev-parse HEAD
```

注意：**此时的原仓库仍是 T4 专用，禁止运行 `scripts/bootstrap.sh`**。先由本地 Codex 将本交接包的 `scripts/amd_*.py`、`scripts/rocm_env_isolated.sh`、`configs/amd_rocm_profile.json` 放到独立 AMD 分支或人工授权文件传输路径；不能假定 GitHub 已出现 AMD 分支。本交接不修改任何远程仓库。

### B. 资源查询与 PyTorch 只读测试

```bash
python3 --version
command -v rocminfo && rocminfo | grep -E 'Name:|gfx[0-9]+' | head -60
command -v rocm-smi && rocm-smi --showproductname --showmeminfo vram
df -h /mnt/workspace /tmp 2>/dev/null || true
python3 scripts/amd_rocm_probe.py --workspace /mnt/workspace \
  --out outputs/AMD_READONLY_PROBE.json
```

允许的第二步（需用户已批准在该实例上做短小 GPU 计算）：

```bash
python3 scripts/amd_rocm_probe.py --gpu-smoke \
  --out outputs/AMD_TINY_NUMERIC_SMOKE.json
python3 scripts/amd_visual_smoke.py --device cuda \
  --out outputs/AMD_SYNTHETIC_VISUAL_SMOKE.json
```

只有 `torch.version.hip` 非空、GPU 计算 finite、显存报告一致，并能回传 JSON 后才标记 `ROCM_NUMERIC_SMOKE_PASS`；这不是 YOLO 权重推理通过，更不是 Pro 训练通过。

### C. 可选真实 YOLO 小权重验证

用户通过**私有/授权传输**提供 `yolo/best.pt`，先执行：

```bash
python3 scripts/asset_verify.py --root assets \
 --manifest configs/research_asset_manifest_template.json \
 --out outputs/AMD_ASSET_SHA.json
python3 scripts/amd_visual_smoke.py --device cuda \
 --model assets/yolo/best.pt \
 --sha256 34ce40cd05168ff0cd637ee2e9d0dc7657dee1da47321be3bc37ad60492818b4 \
 --out outputs/AMD_YOLO_SYNTHETIC.json
```

上述操作要求 ROCm 版 Torch 和已确认可运行的 `ultralytics`，**不要为安装它自动覆盖原 PyTorch 或切换到 CUDA 镜像**。只有资产校验/兼容性都过关才执行。没有安全依赖安装条件则记录阻碍并停止。

### D. AMD 端阶段退出条件

提供：真实 GPU 名称、gfx 目标、HIP/PyTorch 版本、显存、各数值步骤耗时/峰值、持久盘容量和是否可跨会话保留、合成输出 JSON、实际依赖冲突、Agent 实际能力，以及最小结果的本地回传 SHA。**禁止启动 Qiora 8B 下载/训练**；Pro G0 应是用户另行授权的新阶段。

## 腾讯 T4 端

与 AMD 分开执行。继续遵循原仓库 `docs/CLOUD_STUDIO_DEPLOY.md`，前提是用户在 Cloud Studio 控制台确认工作空间和机时。先 `nvidia-smi`，再使用 T4 的 CUDA 12.6 环境要求；Cloud Studio Agent 不需 GitHub 帐号登录，拉取公共代码即可。完成 preflight 后，优先只上传当前六类 YOLO11n 5.19MiB SHA 固定权重，用合成输入执行同预处理、同输出契约的 CUDA 推理。其他旧 DINO/ROI 和真实 DEV 数据后置。

云端 T4 CLI 与 AMD CLI 不混用。T4 不加载 Qiora BF16 8B 完整母体训练。

## 云端 Agent 工作边界

- Agent 可以进行环境探测、有限依赖诊断、代码/文件 SHA 核验、最小推理与 JSON 报告生成；**不得**自动安装/替换宿主驱动、抢占别的计算任务、下载大型基础模型、升级 Torch 造成原环境损坏。
- 不准复制真实病人图片、私密病例或受限源数据到 GitHub / issue / 日志；对外报告只给匿名统计与不敏感 SHA。
- 如果云端只提供交互式 Notebook 而没有可长期运行的 Agent，就使用短时手动测试，不能声称无人值守任务可以持续。
- 只有用户明确开启正式训练、给出预算/时间与数据准入后，才转入独立训练指令。

## 最终反馈格式

```json
{
 "platform":"ModelScope-AMD or CloudStudio-T4",
 "timestamp_utc":"...",
 "gpu_name":"ACTUAL_PROBE_OR_UNKNOWN",
 "gpu_arch":"ACTUAL_GFX_OR_SM_OR_UNKNOWN",
 "torch":"...",
 "torch_hip_or_cuda":"...",
 "visible_vram_bytes":null,
 "persistent_mount_verified":false,
 "tiny_numeric_smoke":"NOT_RUN/PASS/FAIL",
 "trusted_yolo_smoke":"NOT_RUN/PASS/FAIL",
 "files_backed_up_and_read_back":false,
 "formal_optimizer_updates":0,
 "next_gate":"USER_REVIEW_REQUIRED"
}
```
