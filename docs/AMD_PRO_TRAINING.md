# AMD Pro 8B：真实训练入口

本页取代“仅 preflight”的训练范围限制；旧 T4 分支不变。云端 gfx942 / ROCm 7.2、BF16/SDPA 和母体推理成功是用户提供的结果，本次 Windows 工程测试未冒充 ROCm 实测。

母体只读：`/mnt/workspace/qiora-amd/assets/qiora_parent/`。不重新合并、不下载、不覆盖。第18–35层 q_proj/v_proj，rank8/alpha16/dropout0.05，BF16母体、FP32 LoRA/AdamW，microbatch1、累积4、无梯度检查点、无自动混合精度切换。Domain/Evidence各自 one-hot 激活；Case/Semantic关闭，Router不更新。首次使用 SDPA，无需重跑基础 GPU 探测。

## 代码和私有数据

```bash
cd /mnt/workspace/qiora-amd
git pull --ff-only origin codex/modelscope-rocm-preflight-20261011
# 仅补充缺失的用户态依赖；不替换 ROCm torch。现有兼容环境可跳过。
python -m pip install --no-deps -r configs/requirements-pro.txt
```

六个 JSONL 通过私有渠道放入 `data/pro_revision03/`，文件名/SHA在 `configs/pro_revision03_data_manifest.json`。它们没有进入 GitHub。

- D_REPAIR 原256行保留；显式采样清单剔除22个完全相同缺证监督，使用234行、227个记录事实簇，最多59更新。缺证独立问法只有2种，边界覆盖不足仍是研究限制。
- E_COMPACT 256行、256记录事实簇，六类文字关系；最多64更新。不是医学语义 Gold，系统绑定来源不算模型引用能力。
- 两个历史控制组各64行只作参照，当前 CLI 优先主线，没有自动执行控制组。
- 已揭示 DEV：48 compact +172旧面板。TRAIN/DEV主/附加事实簇无重叠；历史语义污染未知。禁止使用封存 TEST2。

## 先做2步真实更新及恢复

备份目录是第二份经过SHA回读的复制目标，**路径不同不证明实例外持久性**。以下 `backups_stage` 只是中途暂存；每个重要 checkpoint 应下载到 Windows，并进行下述全模型重载。若已有真实实例外挂载，可把 `--backup-dir` 指向该挂载。

```bash
python -m qiora_rocm.train \
  --parent assets/qiora_parent --data-dir data/pro_revision03 \
  --expert domain --max-steps 59 --stop-after 2 --save-every 8 \
  --max-seconds 7200 --run-dir outputs/pro_d_01 \
  --backup-dir backups_stage --confirm-training
```

检查 `outputs/pro_d_01/UPDATES_*.jsonl`、`LATEST.json` 和 `RESTORE_*.json`：有限loss、非零梯度、冻结参数版本不变、optimizer/scheduler/RNG重载、真实logits一致。GPU实测峰值和tokens/s就在日志。初始化步0不是训练；随机 tiny CPU fixture不是8B训练。

```bash
D_SHA=$(python -c 'import json; print(json.load(open("outputs/pro_d_01/LATEST.json"))["checkpoint_sha256"])')
python -m qiora_rocm.train \
  --parent assets/qiora_parent --data-dir data/pro_revision03 \
  --expert domain --max-steps 59 --stop-after 16 --save-every 8 \
  --max-seconds 7200 --run-dir outputs/pro_d_01 --backup-dir backups_stage \
  --resume outputs/pro_d_01/step_000002.pt --resume-sha "$D_SHA" --confirm-training
```

新代码/数据/配置/依赖版本/采样顺序会改变恢复身份；运行期间不要 `git pull` 或改配置。上限59从首次调用起冻结，`--stop-after`只是暂停边界；改成16或59可以接续，窗口不重置。暂停点不一定是8的倍数，也会保存。

## 阶段DEV与继续

```bash
D_SHA=$(cat outputs/pro_d_01/step_000016.sha256)
python -m qiora_rocm.dev_eval --parent assets/qiora_parent \
  --data-dir data/pro_revision03 --domain outputs/pro_d_01/step_000016.pt \
  --domain-sha "$D_SHA" --out outputs/dev_d16_01
```

这是真实GPU生成，母体/Domain同提示、同greedy条件，legacy上限512、compact192。报告程序判分 choice/general/compact；自然语言及旧契约保留 NEEDS_REVIEW，**不替代历史评分器**。原始输出不被Rule拒答替换，未审阅 UNKNOWN 不能自动PASS。确证新增缺证补引文、错误作品支持或一般能力严重退化，停止扩大该分支。出现不确定结果先审阅逐题输出，不能靠 loss 选择。报告不自动触发后续训练。

若审阅允许，以上训练命令换 `--stop-after 59 --resume outputs/pro_d_01/step_000016.pt --resume-sha "$D_SHA"`，其他冻结参数保持相同。证据专家独立从母体初始化，先2步，再16步审阅，最多64：

```bash
python -m qiora_rocm.train --parent assets/qiora_parent --data-dir data/pro_revision03 \
  --expert evidence --max-steps 64 --stop-after 2 --save-every 8 \
  --max-seconds 7200 --run-dir outputs/pro_e_01 --backup-dir backups_stage --confirm-training
# 接续时必须给当前完整checkpoint及其SHA；步骤与Domain相同。
```

两专家得到审阅后，用同一组权重比较 Base、Domain、Evidence、fixed(.5/.5)、rule(有来源.2/.8，其他.8/.2)：

```bash
D_SHA=$(cat outputs/pro_d_01/step_000059.sha256)
E_SHA=$(cat outputs/pro_e_01/step_000064.sha256)
python -m qiora_rocm.dev_eval --parent assets/qiora_parent --data-dir data/pro_revision03 \
  --domain outputs/pro_d_01/step_000059.pt --domain-sha "$D_SHA" \
  --evidence outputs/pro_e_01/step_000064.pt --evidence-sha "$E_SHA" --out outputs/dev_pair_01
```

评测失败保留逐题输出；同命令加 `--resume-eval` 可只补未完成的题，条件变更拒绝恢复。报告包括已知配对分母、UNKNOWN、绝对/相对增益、事实簇bootstrap CI和新增失败样本。没有真实双向专家效用标签，因此暂不提供学习式Router训练或独立验收PASS。

## 停止、备份、恢复

使用 `kill -TERM <trainerPID>`，不使用 `kill -9`：下一完整更新边界保存。OOM/非有限梯度/备份失败时只认最后完整checkpoint，`FAILURE_*.json`保留错误；不得把部分optimizer状态保存成可恢复模型。持久日志不删除。强制崩溃后的未提交更新日志保留在旧attempt，不当作新模型完成步数。

`assets/QIORA_TRAINER.lock`防重复训练/评测。崩溃可能留锁，只在核实PID不存在、没有GPU任务且有完整断点后人工移除；程序不自动抢锁。独立实例外复制至少包含完整run目录、冻结代码commit、原母体14文件（Windows已经存在，不重复传回）。从AMD工作台下载 `outputs/pro_d_01`/`outputs/pro_e_01` 到 Windows。不要把这些目录提交GitHub。

Windows完整母体+新专家实际重载生成（PowerShell，SHA用实际下载的 `.sha256`）：

```powershell
$dSha = (Get-Content 'D:/qiora-backup/pro_d_01/step_000059.sha256').Trim()
$eSha = (Get-Content 'D:/qiora-backup/pro_e_01/step_000064.sha256').Trim()
python -m qiora_rocm.restore_model --device cpu --confirm-full-reload `
  --parent 'D:/TCM-Sizhen-AI/artifacts/models/qiora_pro_v1_rebuild_20261009_01/parent_rebuilt/merged_hf' `
  --domain 'D:/qiora-backup/pro_d_01/step_000059.pt' --domain-sha $dSha `
  --evidence 'D:/qiora-backup/pro_e_01/step_000064.pt' --evidence-sha $eSha `
  --out 'D:/qiora-backup/ACTUAL_FULL_RESTORE_01.json'
```

CPU重载使用FP32母体，建议至少40GiB可用RAM。该证明必须实际执行才成立，不证明CUDA/HIP RNG实际恢复。云端训练resume另行验证HIP原生set/get，不假定RNG字节长度。

## 预算、阻塞与真实状态

首轮训练建议：D2→16→59，E2→16→64，LR分别2e-5/5e-5。每微批实际输入最长为D664/E810 token；理论上限2048。母体用户实测15.36GiB；训练初估25–60GiB，**非AMD实测**。步0/每8步/停止点保存，并写备份SHA回读；空间不足、窗口（默认各分支2小时）到期、非有限loss/梯度、母体变动、数据泄漏、恢复身份不符立即停止。无效GPU负载不运行。

当前：CPU更新/续训测试已实现，实际数据检查通过；本次尚未启动AMD训练，没有新8Bcheckpoint。私有数据还需迁移，实际实例外备份/全8B新专家恢复须执行。DEV不能提供独立Pro初版验收，Max/Vision独立准备，不改变本训练结构。
