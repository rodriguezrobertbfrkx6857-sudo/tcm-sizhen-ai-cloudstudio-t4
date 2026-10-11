# AMD 主环境：云端最小操作单

本地已准备，**新分支尚未推送**。以下匿名克隆命令必须在用户批准发布且远端分支出现后执行。此页不授予 GPU 计算、模型上传或训练权限。

## 1. 拉公开代码，不登录 GitHub

用户先手动启动魔搭环境，并确认实际配额与持久存储规则：

```bash
git clone --branch codex/modelscope-rocm-preflight-20261011 https://github.com/rodriguezrobertbfrkx6857-sudo/tcm-sizhen-ai-cloudstudio-t4.git qiora-amd
cd qiora-amd
git rev-parse HEAD
python3 scripts/prepublish_guard.py --root .
python3 scripts/amd_rocm_probe.py --workspace /mnt/workspace --out outputs/01_inventory.json
```

不要运行 CUDA requirements 或 T4 bootstrap。AMD 分支的 bootstrap 已主动阻断自动安装。首选平台预装 ROCm Torch；缺依赖先记录，不安装/覆盖驱动或 Torch。`rocm_env_isolated.sh --create-venv` 仅在已有 HIP Torch 的条件下创建隔离 venv，不装 wheel。

192GB 是用户提供的规格，不能视为真实可用显存，也不能据此推断 GPU 型号。报告记录 HIP、gfx、设备、实际 free/total 显存、磁盘、设备权限和进程列表；不会把磁盘或显存混为一谈。`persistent_mount_verified=false` 必须经过平台政策与跨会话保留测试后另立证明，不能因目录可写就改为 true。

## 2. 获准后才做短小数值验证

先人工查看 `rocm-smi --showpids`，确认没有其他计算任务；该确认标志不是自动进程占用证明。脚本不会停止任何任务。

```bash
python3 scripts/amd_rocm_probe.py --workspace /mnt/workspace --gpu-smoke --confirm-no-other-compute --out outputs/02_numeric.json
python3 scripts/amd_visual_smoke.py --device cuda --confirm-no-other-compute --out outputs/03_visual_synthetic.json
python3 -m qiora_rocm.feasibility --device cuda --confirm-no-other-compute --attention eager --out outputs/04_tiny_qiora_eager.json
python3 -m qiora_rocm.feasibility --device cuda --confirm-no-other-compute --attention sdpa --out outputs/05_tiny_qiora_sdpa.json
```

以上只有小张量或随机微型 Qwen3 的有限正反向，无优化器更新、不下载模型、不训练 Pro。BF16 作为独立后续测试：前面通过且用户批准时，将最后一条增加 `--precision bf16` 并使用新的输出文件。BF16 结果不能由 CPU FP32 结果替代。

## 3. AMD 上运行现有舌象组件

用户通过私有传输提供已核准的 `assets/yolo/best.pt`，不经公开 GitHub：

```bash
python3 scripts/asset_verify.py --root assets --manifest configs/research_asset_manifest_template.json --out outputs/06_yolo_sha.json
python3 scripts/amd_visual_smoke.py --device cuda --confirm-no-other-compute --model assets/yolo/best.pt --sha256 34ce40cd05168ff0cd637ee2e9d0dc7657dee1da47321be3bc37ad60492818b4 --out outputs/07_yolo_synthetic.json
```

保持 imgsz640/conf0.25/IoU0.5/FP32/class-aware NMS。无真实患者图像、无临床能力结论。Ultralytics 缺失或与 ROCm 不兼容时先记录阻塞，不让 pip 自动替换 Torch。

旧 DINO/ROI 仅在必要时单独迁移五项资产，用 HIP 守卫包装已有相同加载器：

```bash
python3 -m qiora_rocm.legacy_visual --assets assets --device cuda --confirm-no-other-compute --out outputs/08_legacy_synthetic.json
```

真实图像评测后续单独确认数据和版本；使用相同权重、预处理、DEV、评分与后处理，跨后端差异使用预先设定容差，不要求逐位相等。

## 4. Qiora 8B 可行性：后续独立步骤

微型 Qwen3 测试不能证明 8B 可用。权重传输、全量加载和 GPU 预算获准后，将本地现有 BF16 母体 14 文件私传至 `assets/qiora_parent/`：

```bash
python3 -m qiora_rocm.feasibility --verify-parent-only assets/qiora_parent --parent-manifest configs/qiora_parent_assets_manifest.json --out outputs/09_parent_sha.json
```

这条只读 SHA，不加载模型。`qiora_rocm.local_model.load_verified_parent` 已准备 local-files-only 的只读模型加载接口，默认拒绝未批准调用；它保持原双专家与 Router 架构，从冻结母体初始化未训练专家，不能称为训练好的 Pro。完整 8B 加载、推理、实际显存、吞吐、原生 GPU RNG 和恢复仍未验证。

正式训练后端还需接入 revision_03 compact 协议、V3 分轨评测、预算/锁与实例外备份，不能直接复制旧 CUDA 控制器在 AMD 上开训。当前 READY_FOR_GPU_TRAINING=false，TEST2 保持封存。

## 5. 回传与保留

输出 JSON 和 `.sha256` 私下下载至 Windows；重新读文件、核验 SHA，再审查内容。输出路径不可重复写入，失败也保留。先用非敏感小文件验证上传→独立下载→SHA及跨会话保留。模型 checkpoint 还需真实重载生成，不能由传输成功替代。

## T4 备用

T4 只作为备用，不是完成主任务的必要资源。另建目录匿名克隆 `codex/cloudstudio-t4-20261011`，遵循其 `docs/CLOUD_STUDIO_DEPLOY.md`；不在 AMD 目录安装 CUDA wheel。原 T4 分支未改。
