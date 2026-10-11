# 本地整合审查

## 输入与保护范围

指定handoffs目录最初不存在，从用户提供ZIP按安全相对路径解包，ZIP CRC通过；FILE_MANIFEST列出的17个文件大小与SHA全部匹配。全部技术报告、配置、脚本和测试已审阅。原文保存于input_reports，原ZIP及解包目录保留。

Windows根项目已有大量未提交历史改动，未整体暂存、重置或切分支。公开迁移仓库的T4 worktree保持原commit和文件；新AMD代码在独立worktree和codex/modelscope-rocm-preflight-20261011本地分支。无推送、无云连接、无模型上传、无新增优化器更新、无TEST2读取或hash。

## 真实修正

| 输入脚本问题或不足 | 本轮处理 | 验证 |
|---|---|---|
| visual的cuda选择只检查GPU可见，不区分NVIDIA/HIP | AMD入口要求HIP、/dev/kfd读写、真实free显存和显式计算确认 | CPU伪NVIDIA用例被拒绝；真实HIP待测 |
| 调用方提供任意SHA并不意味着pickle可信 | YOLO固定已知训练权重SHA、文件大小及六类映射 | 错SHA在反序列化前拒绝；实际可信权重CPU推理通过 |
| 合成ConvNet未固定seed，空YOLO图片可能不经过NMS | 固定seed并增加三个合成框的NMS检查 | 重复输出一致，NMS选中[0,2] |
| asset_verify大小可缺失、未检查重复及中间symlink | 必须有合法bytes/64位hex SHA，阻断目录越界、重复和符号链接 | 正/负合同测试；Windows无symlink权限时用OS谓词mock，Linux真实链接待测 |
| prepublish只查tracked工作文件，可能遗漏untracked与index旧内容 | 同查未跟踪候选、工作树、暂存blob；超大文本与敏感后缀阻断 | 暂存secret但工作树已clean的用例仍被拒绝；不打印secret值 |
| 报告可能覆盖已有结果，异常时难区分测试状态 | 原子exclusive报告、SHA回读、FAIL保留 | Windows跨平台换行故障修复；不可覆盖与失败报告测试通过 |
| ROCm venv未先拒绝CPU/CUDA Torch | 创建前要求预装HIP Torch；AMD bootstrap阻断CUDA安装 | bash语法通过；Linux实际venv创建待测 |
| 没有Qiora原生结构可行性入口 | 复制七份原模型代码（SHA不变），新增随机微型Qwen3测试和只读母体加载函数 | eager/SDPA CPU均通过；8B及ROCm待测 |

首轮测试暴露报告写入的Windows CRLF校验错误、library调用时argv[0]不是文件的问题，失败证据在本地保留；最终报告以修复后退出码、JSON与SHA为准。原生模型源文件、历史权重与历史评分没有修改。

## 性能与依赖边界

本地视觉/合同环境：Python3.13、Torch2.9.1+cpu、torchvision0.24.1+cpu、Transformers4.57.1、Ultralytics8.3.200。原生Qiora微型测试使用另一个现有本地环境：Torch2.9.1+cpu、Transformers5.16.1、PEFT0.21.2。没有更改这两个环境或安装新依赖。

这些版本不是云端lockfile。云端优先读取平台预装ROCm栈，以GPU/gfx、驱动、Torch/torchvision兼容性确定环境；任何可选依赖安装需在隔离venv中另行检查，禁止按本地pip freeze覆盖云端Torch。FlashAttention、bitsandbytes、vLLM不作为前置要求。

已核实母体与研究checkpoint的SHA及D/E64真实payload字段：64steps、offset256、optimizer72项状态、220个adapter/router tensors、Python/Torch/GPU RNG和scheduler存在。检查字段不是执行恢复，更不是医学能力验收。历史失败与Router0保持真实。

阶段最终状态 CLOUD_GPU_PREFLIGHT_REQUIRES_USER_ACTION。下一步先批准新分支公开推送，再由用户手动开启AMD，按CLOUD_QUICKSTART做限定预检；正式训练另行授权和完成新协议工程接入。
