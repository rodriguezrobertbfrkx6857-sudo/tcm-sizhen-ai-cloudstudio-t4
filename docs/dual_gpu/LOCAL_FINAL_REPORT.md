# Qiora AMD 主环境本地整合最终报告

状态：CLOUD_GPU_PREFLIGHT_REQUIRES_USER_ACTION。已完成本地准备，AMD真实设备及正式训练仍未验证；READY_FOR_GPU_TRAINING=false。

按用户最新要求，后续舌象视觉、Qiora语言模型可行性、评测和备份优先在AMD完成。腾讯T4只保留为备用，原分支及其文件不变。

实际完成：安全导入交接包；审阅最新GPU失败研究与revision_03 CPU修复；14个母体文件16,397,459,696字节及13个checkpoint重新SHA回读；D/E64可信payload的optimizer/scheduler/RNG/order/offset字段检查；独立AMD worktree；修复HIP/可信YOLO/资产/发布/报告边界；复制七份原生Qiora实现且源SHA不变；只读母体加载接口与部署流程。

测试：最终23项CPU合同/安全测试全部通过；eager与SDPA随机微型Qwen3原生双专家/Router梯度、冻结参数、Case权重0、CPU保存恢复logits及RNG一致；真实SHA固定YOLO和旧DINO/ROI在合成输入CPU推理通过；YOLO验证阻断网络socket并测试NMS；两份bash脚本语法通过。Windows中间symlink防护以OS谓词mock验证，Linux真实symlink测试尚待云端。零新增优化器更新，没有安装依赖、下载模型或读取TEST2。

本地读取不是新8B重载证明；微型Qwen3不是已训练Pro；空optimizer的初始状态恢复不是充分训练后的恢复；T4原有2步合成测试属于上一阶段，不计为本轮更新。Pro仍未验收，Router历史真实更新0，医生Gold0。未经GPU验证不声称ROCm/BF16/实际吞吐/训练能力通过。

代码：package；分支codex/modelscope-rocm-preflight-20261011；远端尚未推送。云操作：package/docs/dual_gpu/CLOUD_QUICKSTART.md。任务设计：AMD_TASK_PLAN.md。最终Gate：GATE_STATUS.json。所有公开候选均是文本代码、配置、文档与非敏感摘要，不含权重/数据/凭据。

后续：用户批准推送后才能匿名拉取新分支；用户手动开启AMD并批准限定预检，核真实HIP/gfx/显存/持久性/实例外通道，再单独批准母体私传与8B加载。正式Pro训练还需revision_03 V3后端接入和新的有界GPU预算。100小时与192GB仍是未核实用户规格，不是执行授权。
