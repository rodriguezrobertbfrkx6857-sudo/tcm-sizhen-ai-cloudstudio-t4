# 云端代码、权重、数据与训练成果传输方案

## 公共 GitHub 代码与受限资产彻底分离

**代码路径**：Windows Codex → 新独立 AMD 分支（人工审阅后推送）→ 云端匿名 HTTPS clone。现有 T4 分支是 `codex/cloudstudio-t4-20261011`；本次并未创建或推送新 AMD 分支。

**资产路径**：Windows 或合规的持久对象存储 → 用户授权的私有传输机制/平台上传 → 云端 `assets/` 目录 → 与预冻结大小 + SHA256 比对。不会使用公开 GitHub LFS 存放权重，除非另行明确授权且许可允许。

**成果返回**：云端 checkpoint / 评测 JSON → 用户许可的私有目标或安全下载 → Windows `D:\TCM-Sizhen-AI\artifacts\...` 新版本目录 → 独立 GET → SHA256 → 真实载入生成。不得把“上传成功”当作“可恢复”。

## 资产分级和首次迁移建议

| 等级 | 内容 | 推荐传输条件 |
|---|---|---|
| 0 | 公共源码、无私密测试工具 | 匿名 HTTPS clone，提交前运行 `prepublish_guard.py` |
| 1 | 当前 YOLO 六类 5,445,914 字节 | 首次可选小资产，私有上传，固定 SHA |
| 2 | 旧 G1/DINO/ROI 参考五文件约 225.38MiB | 仅在 T4/AMD 小模型端到端推理需要时 |
| 3 | Qiora BF16 母体历史 14 文件，约 16.4GB | AMD 显存/持久盘/令牌/模型许可与 Pro G0 任务另行批准后 |
| 4 | Domain/Evidence adapter、训练优化器状态、数据集 | 分别做许可/隐私/实验身份与 SHA 审查，不把 54万历史QA 重新上传为默认步骤 |
| 5 | 原始舌图、医疗记录、病人资料 | 不能进入公共仓库；无特定授权和去标识化/最小化依据时 HOLD |

小资产参考 SHA：当前 YOLO `34ce40cd05168ff0cd637ee2e9d0dc7657dee1da47321be3bc37ad60492818b4`。详细五项旧组件 SHA 与文件名在公共仓库 `configs/assets_manifest.json`。

## SHA 校验步骤（只读，未上传的文件不算已验证）

```bash
# 将本包 configs/research_asset_manifest_template.json 放到工作目录后：
python3 scripts/asset_verify.py --root assets \
 --manifest configs/research_asset_manifest_template.json \
 --out outputs/assets_SHA_verified.json

# 单文件通用校验（有资产时才执行）：
sha256sum assets/yolo/best.pt
```

`asset_verify.py` 会拒绝不存在的文件、错误大小、SHA 不一致和目录越界。测试时使用小型合成 fixture；**不是检查过用户的真实 Windows 大模型权重**。

## 下行与上行通道选择

- ModelScope Notebook 手动上传/下载、授权对象存储签名链接或用户控制的 SFTP/SCP 都可以；以云平台实际支持情况为准。不要把私有对象存储密钥写入命令日志、GitHub、截图或交接 ZIP。
- 第一次只做 **1 个非敏感小文件** 的上传、独立下载及 SHA，记录是否能跨 Notebook 重启保留。
- 若平台持久路径空间不足：仅放正在运行的最小模型和 checkpoint，训练数据用经过批准的共享卷或对象存储；不要从磁盘容量不足硬转为不断覆盖唯一备份。
- Checkpoint 建议包含 `model/adapter`、`optimizer`、`scheduler`、`rng`、`step`、`data_offset/order`、冻结配置/代码SHA、eval 摘要；如果只回传 adapter 无法无歧义恢复训练，必须明确标记 `INFERENCE_ONLY`。
- CPU 端真实加载与生成证明由 Windows 后续验证；不能声称云端网络连通测试代替独立 CPU 加载。

## 开源仓库发布前门槛

1. 本地 Codex 在**新分支**审阅文件，明确 LICENSE/第三方来源和授权范围。
2. 对变更文件运行 `python scripts/prepublish_guard.py --root <checkout>`；拒绝任何 `.pt/.safetensors/.jpg/.csv/.env`、密钥、数据集、图片、缓存等路径。此扫描是保守门槛，不是隐私万能证明；还需人工检查 Git 历史。
3. 只提交可公开的脚本、JSON 草案、操作文档、合成单元测试；不要提交真实训练数据、秘密 URI 或模型输入输出含个人信息的样本。
4. 云端用只读 clone，选择精确分支 / commit 固定身份；不要求 GitHub 用户账号登录。
5. 不把历史实验指标改写成“已在 AMD 或 T4 实测”。

## 断点/实例丢失策略

- 任何达到验收条件的 checkpoint 先复制至实例外，再通过独立读取和 SHA 与恢复证明验证；未经验证时不要依赖一个工作空间文件。
- 定期核查云平台停机/释放对 `/mnt/workspace`、临时目录、任务中断的真实策略。
- 工作空间意外消失时，从 Windows 保留的母体+adapter+状态恢复，创建 **新 instance ID、新输出目录与 lineage**，不能套用旧机器的绝对路径/运行窗口/旧 resume nonce。
