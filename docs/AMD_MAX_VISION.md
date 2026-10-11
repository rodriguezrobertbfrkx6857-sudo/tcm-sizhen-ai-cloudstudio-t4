# Max 和 Vision：独立研究入口

本提交不改变 `qiora_rocm` 的 Pro 训练代码或恢复协议。Pro 运行中不要更新代码目录；完成一个恢复边界并退出后才切换提交。工程可并行准备，默认 GPU 作业仍顺序执行，避免抢占 Pro。没有自动租赁、自动训练或后台占用 GPU。

## 当前真实状态

- Pro 已发布，实际 AMD 训练结果由云端日志与新 checkpoint 决定。
- Max 是独立 `qiora_max_rocm`：Domain + Evidence + reading LoRA，prompt-only layer Router 已实现但没有效用标签，保持禁用且零更新。不是已验收 Max。
- reading 数据来自现有真实证据：185 TRAIN、9 revealed DEV，作品/事实簇无重叠；108,742 输入 token、9,815 监督 token，最长738。监督是原文抽取，不是新医学 Gold，也不能代替复杂推理训练。不得把9条 DEV 声称独立验收。
- Vision 是六类外观 YOLO11n：4,222 TRAIN、896 source DEV。原始图片/标签不公开，患者独立性未知，不是临床验证。
- CPU 小模型测试验证 reading 真实梯度更新、冻结参数、恢复后下一更新一致；真实8B/AMD训练仍待验证。Vision 已做真实历史权重 SHA + CPU 前向、全部私有资产回读；新 ROCm 训练和 NMS 仍待云端运行。

## Max（Pro 的新 D/E 权重已经保存并评测后）

私有目录上传：`D:\Qiora-AMD-Upload\max_reading_01` → `data/max_reading_01`。不可提交 GitHub。

先设置云端真实权重路径及其 SHA，不能使用占位文件：
```bash
export DCP=outputs/pro_d_01/step_000059.pt
export ECP=outputs/pro_e_01/step_000064.pt
export DSHA=$(sha256sum "$DCP" | cut -d' ' -f1)
export ESHA=$(sha256sum "$ECP" | cut -d' ' -f1)
python -m qiora_max_rocm.train --parent assets/qiora_parent \
  --domain "$DCP" --domain-sha "$DSHA" --evidence "$ECP" --evidence-sha "$ESHA" \
  --data-dir data/max_reading_01 --run-dir outputs/max_r_01 --backup-dir backups_stage \
  --max-steps 32 --stop-after 2 --max-seconds 3600 --confirm-research-training
```
检查两步的 loss、梯度、冻结参数、备份回读后，在同一窗口显式恢复：
```bash
export RCP=outputs/max_r_01/step_000002.pt
export RSHA=$(sha256sum "$RCP" | cut -d' ' -f1)
python -m qiora_max_rocm.train --parent assets/qiora_parent \
  --domain "$DCP" --domain-sha "$DSHA" --evidence "$ECP" --evidence-sha "$ESHA" \
  --data-dir data/max_reading_01 --run-dir outputs/max_r_01 --backup-dir backups_stage \
  --max-steps 32 --stop-after 32 --max-seconds 3600 --resume "$RCP" --resume-sha "$RSHA" --confirm-research-training
```
最多32个研究更新，累积4，128行，不为了用满GPU增加循环。显存准入32GiB只是保守工程条件，真实训练峰值待测。使用相同D/E权重公平比较Base/Pro/reader/Max固定组合，另有 `pro_matched` 保持与Max完全相同的D/E系数(.4/.4)而关闭reading，用来分离新增专家与系数变化的收益：
```bash
export RCP=outputs/max_r_01/step_000032.pt
export RSHA=$(sha256sum "$RCP" | cut -d' ' -f1)
python -m qiora_max_rocm.evaluate --parent assets/qiora_parent \
  --domain "$DCP" --domain-sha "$DSHA" --evidence "$ECP" --evidence-sha "$ESHA" \
  --reader "$RCP" --reader-sha "$RSHA" --data-dir data/max_reading_01 --out outputs/max_read_dev_01
```
真实实例外下载后，在完整母体可用的 Windows/CPU 环境运行（加载约32GiB FP32母体，另留内存余量）：
```bash
python -m qiora_max_rocm.restore_model --parent assets/qiora_parent \
  --domain "$DCP" --domain-sha "$DSHA" --evidence "$ECP" --evidence-sha "$ESHA" \
  --reader "$RCP" --reader-sha "$RSHA" --device cpu --out MAX_EXTERNAL_RELOAD.json --confirm-full-reload
```
数据/代码/父权重身份变化、非有限 loss/梯度、备份失败或时间用尽均停止；保留已保存 checkpoint。Max结论只能为研究分支结果，不能绕过 Pro 科学 Gate 或声称 Router 有效。

## TongueAI Vision

使用独立 Python 环境保留 AMD 官方 torch/torchvision，不安装 NVIDIA torch wheel。另装 `requirements-yolo.txt`；不要在正在运行的 Pro 环境修改依赖。

私有 `D:\Qiora-AMD-Upload\vision_six` 上传到 `data/vision_six`。历史 `best.pt` 放到 `assets/yolo/best.pt`，SHA 为 `34ce40cd05168ff0cd637ee2e9d0dc7657dee1da47321be3bc37ad60492818b4`。无需上传旧视觉大权重。

```bash
python -m cloudstudio_tongue.rocm_train --mode train \
  --weights assets/yolo/best.pt --weights-sha 34ce40cd05168ff0cd637ee2e9d0dc7657dee1da47321be3bc37ad60492818b4 \
  --data-dir data/vision_six --out outputs/vision_01 --backup-dir backups_vision \
  --epochs 3 --batch 8 --max-seconds 3600 --confirm-private-research
```
FP32、AMP关闭、640px、AdamW；每epoch保存且备份SHA回读，3epoch是首轮建议。预计4–8GiB仅为规划值，AMD实测未完成。原生Ultralytics恢复含optimizer/EMA，但半精度保存及原生数据顺序意味着**不保证逐位一致恢复**；不冒称完整RNG恢复。异常只从已经备份的未strip中间checkpoint显式 `--resume`，原配置/窗口不变，不能从最终已strip权重冒称续训。中途任意batch故障可能损失当前epoch，已保存epoch保留。

DEV对照用实际新文件SHA：
```bash
export VCP=outputs/vision_01/training/weights/best.pt
export VSHA=$(sha256sum "$VCP" | cut -d' ' -f1)
python -m cloudstudio_tongue.rocm_train --mode evaluate \
  --weights "$VCP" --weights-sha "$VSHA" --baseline assets/yolo/best.pt \
  --baseline-sha 34ce40cd05168ff0cd637ee2e9d0dc7657dee1da47321be3bc37ad60492818b4 \
  --data-dir data/vision_six --out outputs/vision_dev_01 --batch 8 --confirm-private-research
```
输出mAP等真实差值，不称患者独立或诊断能力。下载实际新权重后CPU真实重载：
```bash
python -m cloudstudio_tongue.rocm_train --mode restore --weights "$VCP" --weights-sha "$VSHA" \
  --data-dir data/vision_six --out VISION_EXTERNAL_RELOAD.json --confirm-private-research
```
`backups_stage`/`backups_vision`若在同一实例内，只是校验过的暂存；必须下载到 Windows 或经过确认的独立持久存储，并真实重新加载，才能宣布实例外恢复通过。
