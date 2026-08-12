# 0006 将 Prompt 和 RAG 参数放在代码外

## 状态

Accepted

## 背景

RAG 应用需要频繁调整：

- Prompt 文案。
- 向量库 collection。
- TopK。
- 文档切分大小。
- 文件类型白名单。
- 模型供应商。

如果全部写死在代码中，每次调整都需要修改业务代码，风险较高。

## 决策

将配置拆到代码外：

- Prompt 放在 `backend/app/prompt/*.txt`。
- Chroma 和切分参数放在 `backend/app/config/chroma.yaml`。
- 模型供应商和密钥放在 `.env`。
- 通过 `prompt_loader.py`、`config_handler.py`、`factory.py` 加载。

## 备选方案

- 全部写死在 Python 代码中。
- 使用数据库管理 Prompt 和 RAG 参数。
- 使用专门配置中心。

## 后果

正面影响：

- Prompt 和 RAG 参数更容易调整。
- 业务代码更干净。
- 不同环境可以用不同 `.env`。
- 后续可以平滑迁移到配置中心或管理后台。

负面影响：

- 配置文件和代码之间需要保持命名一致。
- 缺少配置校验时，运行期才暴露错误。
- Prompt 变更缺少版本管理流程。

## 后续约束

- 关键配置应增加启动时校验。
- Prompt 变更应记录原因和效果。
- 生产环境密钥不能提交到仓库。

