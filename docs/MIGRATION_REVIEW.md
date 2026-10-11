# 本地真实项目审查与迁移范围（2026-10-11）

公开发布补充：本仓库首次以 private 方式建立，随后按用户明确指令转为 public，补充 README、DISCLAIMER.md 和 LICENSE_NOTICE.md，并重新核验发布文件及历史。下文私有仓库描述保留为初次准备时的历史记录；权重、数据与凭据仍不公开，科学验证结果不因可见性调整而改变。

项目实际名称为TCM-Sizhen-AI，用户称CM-Sizhen-AI/TongueAI。原工作树在master（b1d7dba），没有Git remote，存在大量已有未提交改动。此迁移使用独立目录和私有仓库，原工作树不checkout、不reset、不整体add；原源码、权重、实验报告保持原样。

真实注册表中舌诊基础G1 A0冻结复用SHA b715351507ebc6a1000ed4551e02622f37b5e9cac18bce2af3313e17cb3923b3；R1-M分割SHA 68cd4aa8f3813ec306ad5e1d6c02b3077a1fe8f599401b5776e4c70d0184a32e。全部5个必要资产本阶段本地SHA核验通过。旧Windows加载器有绝对路径、历史manifest和CPU-only约束；新增移植层严格加载相同tensor、保留checkpoint身份，数值是否一致以ACTUAL_CPU_PARITY.json为准。

进一步审阅当前serve_tongueherb_v1与本地tongue/TRAIN_RESULT后确认，当前网站已实际使用2026-10-06六类YOLO11n，训练结果COMPLETED、进程exit0，本地best.pt SHA 34ce40cd05168ff0cd637ee2e9d0dc7657dee1da47321be3bc37ad60492818b4、大小5,445,914字节。因此这份YOLO模型是首选迁移入口，旧二分类/ROI为可选参考。来源VAL历史mAP50约0.5365、mAP50-95约0.4896，4222TRAIN/896VAL，无患者独立性或医生Gold。新迁移适配器以实际CPU原始输出一致性记录为准，不把读到旧分数视为重测。本阶段未将旧注册表改写为新模型，也未改当前网站。

历史基础G1任务只区分来源标注的白/黄舌苔，其他类别不能强行归类。旧内部DEV macro-F1约0.8203、balanced accuracy约0.8506；R1-M历史publisher-mask DEV Dice约0.9875。这些是历史来源受限成绩，本阶段未重跑真实图像DEV，不能作为腾讯云新成绩或临床有效性。Local病灶/新全局模型另有EXPERIMENTAL/DEFERRED状态，本包不冒充成熟功能。

最近Qiora-KER云轮次完成D64/E64/两对照16共160更新、Router0，科学审阅未通过Pro验收；随后本地修复为GPU_REQUIRED_STOP。Pro/Max和8B BF16母体不在本迁移训练范围。原8B BF16仅权重就接近/超过T4可用显存，还需要激活和训练状态，不能沿用大显存云配置。本包没有这些训练入口。

本阶段Windows进程只读快照未发现匹配train/gpu_cli/cached_pipeline的进程；看到备份/恢复辅助及其他Python进程，未终止或修改。未连接旧云服务器，不能把历史“自有GPU已退出”当作新的远端实时状态。云端是否另有任务交给部署时preflight实际核验。

T4 16GB规划：G1输入224、ROI输入384、batch1，FP32推理预计2–4GiB，单个小视觉模型训练预计4–8GiB；此范围尚未T4实测，不承诺固定吞吐。小测试优先用合成输入和冻结编码器，避免下载医疗数据。至少6GiB真实空闲显存、无其他compute进程、20GiB空闲磁盘后再执行，报告真实峰值后决定后续batch。T4不采用BF16/TF32，平台驱动由云端负责。

代码包：新路径无关模型加载/预处理、CLI推理、mask评测、二分类DEV评测、两步合成工程训练、环境清单、依赖草案、只读原ROI训练源码与文档。不包含完整FastAPI/Next.js四诊网站、医疗对话模型和大规模数据工程。迁移先满足舌诊基础组件可运行，后续全站要独立审阅依赖和患者数据路径。

GitHub：目标账号rodriguezrobertbfrkx6857-sudo已核实管理员权限；目标独立仓库tcm-sizhen-ai-cloudstudio-t4在准备时尚不存在，计划新建private，feature分支codex/cloudstudio-t4-20261011。不更改已有公开仓库的可见性或内容。仅白名单文本源码/配置/文档进入提交，凭据、.env、权重、缓存、图像和数据目录排除；敏感扫描是启发式检查加人工选取代码，不宣称万能隐私识别。
