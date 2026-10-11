# 后续资产迁移清单

当前仓库与交接ZIP均不含模型payload、图片、标签数据、大型缓存或个人医疗资料。必要5个文件合计236,330,819字节（约225.38MiB）；完整逐文件大小和SHA见 configs/assets_manifest.json。本地原路径保存在项目交付目录的 LOCAL_MIGRATION_INVENTORY_DO_NOT_UPLOAD.json，不进入Git。

**首选当前网站模型另列清单：** YOLO11n六类best.pt为5,445,914字节（约5.19MiB），SHA/英文内部类别顺序见yolo_assets_manifest.json。当前六类入口只需要这一个文件；上述5个文件是旧组件路线。如果两套都迁移，总计241,776,733字节（约230.58MiB）。YOLO权重包含序列化模块，必须先核验绑定SHA且仅加载本地可信训练产物。

1. G1 A0 完整 DINOv2-small 二分类checkpoint：白/黄苔来源受限研究任务。
2. R1-M完整 ResNet18-UNet checkpoint：舌体分割，不是病灶 detector。
3. 原DINOv2-small基础safetensors、architecture config、processor config：保持来源和预处理指纹。完整 G1 checkpoint 已含encoder，因此不重新合并或训练。

ROI完整checkpoint含ResNet编码器，无需为推理额外上传ResNet18初始化权重；原初始化文件只有在未来经批准的重新训练中才需要。原训练代码与权重元数据SHA在SOURCE_CODE_PROVENANCE.json记录；加载器不会因为云路径改变而修改checkpoint。

首轮优先只上传YOLO资产，旧五个资产按需上传，使用合成RGB做CUDA测试。随后按需提供授权DEV图像和mask/已知标签、来源与split清单；图片数量、真实容量和迁移权限当前未冻结，不编造已准备完整数据。原ROI的1,658 TRAIN/384 DEV是历史publisher-mask协议，不代表有权立即把原图上传腾讯云。当前六类来源原始5978张约1.4GB，历史准入4222TRAIN/896VAL；这些数据同样不自动上传，患者独立性未证实。

不迁移Qiora BF16母体、Pro/Max checkpoint、历史54万QA、Shizhen全量、Huatuo全量、网站患者上传目录或租赁实例连接凭据。正式训练后将新checkpoint和SHA回存实例外，而不是仅放工作空间。

存储规划：代码不足1MiB；当前资产约225MiB；环境下载/安装估计6–10GiB、临时测试/结果预留2–5GiB，要求20GiB空闲，建议30GiB以上；若增加正式图像数据建议50GiB以上，并按实际数据容量调整。环境/显存数字是规划范围，不是云端实测。
