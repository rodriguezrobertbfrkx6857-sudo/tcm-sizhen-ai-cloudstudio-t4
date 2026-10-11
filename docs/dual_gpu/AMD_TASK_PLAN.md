# AMD 为主的任务与部署设计

根据用户最新调整：舌象视觉、Qiora 语言模型、评测和后续研究优先在 AMD 完成；T4 保留为可选备用，不要求两云并行。

## 当前真实状态

Windows 主仓为 master，已有大量历史未提交修改，未整体暂存或改变；公开迁移仓库是独立代码包，不代表完整医疗项目。T4 分支基线为 `3374e235077b5f02d4de5bcb1c2444362a1837c5`。新 AMD 本地分支在另一个 worktree，不修改该分支。

最新 Qiora-KER 真实 GPU 轮次为 Domain64、Evidence64、对照16+16，总160更新，Router0；Pro未通过科学验收。revision_03 CPU 修复已完成，但新紧凑监督/评分仍缺实际GPU后端接入，状态GPU_REQUIRED_STOP。13个研究checkpoint与14个母体文件重新SHA回读通过，母体16,397,459,696字节（约15.27GiB），不重新合并历史医疗LoRA。历史最终QA训练行数为543,627，不重新训练。恢复证明是已有历史证据，本次只读SHA回读不称为新的全母体重载。

TongueAI当前首选为YOLO11n六类外观模型，权重5,445,914字节；未完成患者独立或医生Gold临床验证。旧DINO/ROI为参考组件。现有网页/API和非语言模块不随本包完整迁移，也不宣称已完成AMD部署。

## 三处职责

| 环境 | 任务 | 状态 |
|---|---|---|
| Windows | 源码审查、SHA冻结、本地CPU测试、永久备份和验收记录 | 本轮执行 |
| 魔搭AMD | 环境、数值、视觉、语言模型可行性；后续有条件的Pro训练和评测 | 仅准备入口，云端未执行 |
| 腾讯T4 | 原视觉部署备用路线 | 分支保留；无必需任务 |

不是多机分布式训练，也没有自动抢占/迁移其他任务。公开Git只放脚本、配置草案、文档、合成测试；母体、专家权重、真实数据和凭据经单独私有通路。

## 有界推进顺序

1. AMD只读探测，确认HIP/gfx/权限/真实free显存/存储/进程；用户报告的192GB和100小时不视为实测或预算授权。
2. 获准有限GPU测试后，运行FP32小张量→合成视觉→微型原生Qiora eager/SDPA；单独测BF16。无新优化器更新。
3. 可信YOLO私传及SHA，AMD合成输入推理；必要时旧组件对照。公开代码已有CPU证明，AMD数值仍待验证。
4. 验证持久存储及实例外PUT→独立GET→SHA；统计每个阶段实际资源。母体约15.27GiB，FP32加载理论权重体积约30.5GiB，运行和训练还需额外内存。初次只读FP32加载接口要求至少32GiB真实free显存，只是最低保护，不保证容量够用；BF16经过实测才能采用。存储建议至少60GiB真实free作为母体、回读副本、专家、环境与日志的初始规划，再按实际checkpoint扩大。
5. 权重传输及8B加载批准后，冻结全部14文件SHA，local_files_only验证真实加载/固定非医疗生成、吞吐和恢复；不从HF下载原Qwen替代Qiora母体。
6. 训练仍需新预算、单控制器锁、ROCm资源守卫、revision_03数据/评分接入、真实标签与长度检查、有限前后向和原生RNG/optimizer/scheduler恢复。未具备这些条件前 READY_FOR_GPU_TRAINING=false。
7. 有实际非退化收益才继续专家整合和Router效用研究，不为利用率凑更新，不降低Gate。TEST2不在本包内，不打开或hash。

## 已实现和待验证

已实现：HIP区分、流式SHA及目录/符号链接检查、种子合成测试、原生双专家/Router微型正反向、无更新保存恢复、只读母体加载接口、不可覆盖报告与SHA、包含暂存和未跟踪文件的发布检查。复制的原生Qiora七份代码逐字节不变，Case仍inactive，Semantic未启用。

CPU已证明：小模型冻结/梯度、Case权重0、恢复logits与CPU RNG一致、可信YOLO无网络合成输入推理、现有权重SHA、合同与安全检查。只有空optimizer初始状态恢复，没有用新增更新制造“充分训练恢复”证明。

待GPU：真实gfx/ROCm版本兼容、BF16、8B加载、GPU恢复、吞吐、跨会话持久性、实际专业能力增益。生产V3训练后端、GPU成本授权和独立验收均未就绪；本轮不标ProAccepted。

## 官方依据（2026-10-11查阅）

- [PyTorch HIP语义](https://docs.pytorch.org/docs/2.14/notes/hip.html)：ROCm复用torch.cuda API，使用torch.version.hip区分构建。
- [ROCm 7.2.3矩阵](https://rocm.docs.amd.com/en/docs-7.2.3/compatibility/compatibility-matrix.html)：按实际GPU/gfx、OS、驱动及框架版本匹配，不直接假定192GB对应某型号。
- [Transformers attention interface](https://huggingface.co/docs/transformers/attention_interface)：eager/SDPA作为初始比较，优化内核另行实测。
- [Ultralytics AMD说明](https://docs.ultralytics.com/integrations/amd)：可行路径不是本项目实际AMD通过证明。

原输入技术报告存于input_reports，保持原文作为来源记录；本文件以实际Windows状态和用户AMD主环境的新要求修正旧双云分工。
