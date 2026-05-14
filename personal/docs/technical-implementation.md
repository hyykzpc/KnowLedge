# 技术实现

本文从代码实现角度解释当前项目的关键技术点。

## 1. FastAPI 应用实现

入口文件：

```text
backend/main.py
```

核心职责：

- `FastAPI()` 创建应用。
- `include_router` 注册 API。
- `CORSMiddleware` 允许跨域。
- `RateLimitMiddleware` 做全局限流。
- `register_exception_handlers(app)` 注册异常处理。
- `startup_event` 初始化 MySQL、会话管理器、Redis、Reranker。
- `shutdown_event` 关闭 Redis。

当前注册的路由：

| Router | 来源 | 作用 |
| --- | --- | --- |
| `chat_router` | `app.router.chat` | AI/RAG/会话/向量库/重排。 |
| `health_router` | `app.router.health` | 健康检查。 |
| `user_router` | `app.router.user` | FastAPI 侧用户详情代理。 |

## 2. API 请求和响应模型

位置：

```text
backend/app/schemas/models.py
```

| Pydantic 模型 | 作用 |
| --- | --- |
| `QueryRequest` | Agent 查询请求，包含 `session_id` 和 `query`。 |
| `RAGRequest` | RAG 查询请求。 |
| `SessionResponse` | 会话历史响应。 |
| `AgentStep` | Agent 中间步骤。 |
| `AgentResponse` | Agent 完整响应。 |
| `RAGResponse` | RAG 响应。 |
| `ReorderRequest` | 重排请求。 |
| `ReorderResponse` | 重排响应。 |

统一成功响应：

```text
backend/app/core/success_response.py
```

异常处理：

```text
backend/app/core/failed_response.py
backend/app/core/failed_response_register.py
```

## 3. 模型工厂实现

位置：

```text
backend/app/utils/factory.py
```

实现了三个工厂概念：

| 类 | 作用 |
| --- | --- |
| `ChatModelFactory` | 根据 `LLM_TYPE` 创建聊天模型。 |
| `EmbedModelFactory` | 根据 `EMBED_MODEL_TYPE` 创建 Embedding 模型。 |
| `DashScopeEmbeddingsWrapper` | 把 DashScope Embedding 包装成 LangChain `Embeddings` 接口。 |

支持的聊天模型：

| `LLM_TYPE` | 实现 |
| --- | --- |
| `OLLAMA` | `ChatOllama` |
| `ALIYUN` | `ChatTongyi` |

支持的 Embedding：

| `EMBED_MODEL_TYPE` | 实现 |
| --- | --- |
| `OLLAMA` | `OllamaEmbeddings` |
| `ALIYUN` | `DashScopeEmbeddingsWrapper` |

全局实例：

```python
chat_model = ChatModelFactory().generator()
embed_model = EmbedModelFactory().generator()
```

设计收益：

- RAG 和 Agent 不直接关心模型供应商。
- 通过环境变量切换模型。
- 后续添加 OpenAI、Azure、vLLM 等供应商时改动集中。

## 4. RAG 实现

核心文件：

```text
backend/app/rag/rag_service.py
backend/app/rag/vector_store.py
backend/app/rag/reorder_service.py
backend/app/rag/text_spliter.py
backend/app/utils/file_handler.py
```

### 4.1 文档加载

`file_handler.py` 支持：

| 函数 | 文件类型 |
| --- | --- |
| `pdf_loader` | PDF |
| `txt_loader` | TXT |
| `markdown_loader` | Markdown |
| `ppt_loader` | PPT/PPTX |
| `word_loader` | DOCX，需要后续验证 |

### 4.2 文本切分

`AsyncTextSplitter` 包装 LangChain 的 `RecursiveCharacterTextSplitter`。

配置来自：

```text
backend/app/config/chroma.yaml
```

核心参数：

- `chunk_size`
- `chunk_overlap`
- `separators`

### 4.3 向量库

`VectorStoreService` 使用：

```python
Chroma(
    collection_name=...,
    embedding_function=embed_model,
    persist_directory=...
)
```

入库时会：

- 计算文件 MD5。
- 跳过重复文件。
- 加载文档。
- 切分文档。
- 写入 `user_id` metadata。
- 调用 `self.vectors_store.add_documents(document)`。

### 4.4 混合检索

当前检索组合：

```text
Chroma similarity retriever
  +
BM25Retriever
  ↓
EnsembleRetriever
```

动态权重策略：

| 查询特征 | 权重倾向 |
| --- | --- |
| 长查询 | 增加向量检索权重。 |
| 短查询 | 增加 BM25 权重。 |
| 词密度高 | 适当增加 BM25 权重。 |

### 4.5 HyDE

`RagService.generate_hypothetical_document()` 会先让 LLM 根据用户问题生成一段假想文档，再用假想文档做检索。

意义：

- 用户问题通常很短，和文档片段的向量相似度不一定高。
- 假想文档更接近知识库文本形态，有利于召回。

### 4.6 重排

`ReorderService` 使用：

```text
sentence_transformers.CrossEncoder
Qwen/Qwen3-Reranker-0.6B
```

输入：

```text
[(query, doc1), (query, doc2), ...]
```

输出：

```text
[
  {"document": "...", "similarity": 0.92},
  {"document": "...", "similarity": 0.81}
]
```

### 4.7 摘要生成

`RagService` 使用 LangChain Runnable：

```python
self.prompt_template | self.chat_model | StrOutputParser()
```

当前实现：

- 先对前几个文档分别摘要。
- 如果有多个摘要，再合并生成最终摘要。
- 每次调用设置超时，避免长时间阻塞。

## 5. Agent 实现

核心文件：

```text
backend/app/agent/agent.py
backend/app/agent/agent_tools.py
backend/app/agent/agent_middleware.py
```

`AgentFactory` 负责：

- 读取默认工具。
- 读取系统 Prompt。
- 根据 `LLM_TYPE` 创建模型。
- 创建 `ChatPromptTemplate`。
- 调用 `create_tool_calling_agent`。
- 包装为 `AgentExecutor`。

默认工具：

| 工具 | 能力 |
| --- | --- |
| `rag_summary_tools` | RAG 查询。 |
| `reorder_documents_tools` | 文档重排。 |
| `get_user_info_tools` | 获取用户信息。 |
| `get_weather_tools` | 天气。 |
| `what_time_is_now` | 当前时间。 |

流式输出：

```text
get_agent_stream_response()
  ↓
yield "data: {...}\n\n"
```

前端按 SSE 格式解析：

```text
data: {"type": "response", "content": "..."}
data: {"type": "done", "session_id": "..."}
data: {"type": "error", "content": "..."}
```

## 6. 会话存储实现

核心文件：

```text
backend/app/services/database_session_manager.py
backend/app/models/chat_history.py
backend/app/db/db_config.py
```

数据库连接：

- 使用 SQLAlchemy async engine。
- MySQL 驱动是 `aiomysql`。
- `init_db()` 在启动时执行 `Base.metadata.create_all`。

会话写入：

```text
add_message(session_id, user_id, user_message, assistant_message)
```

行为：

- 如果会话不存在，创建会话。
- 如果会话属于其他用户，返回 403。
- 用户消息和助手消息分别写入 `chat_messages`。
- 默认会话标题由用户消息前 30 个字符生成。

会话读取：

```text
get_history(session_id, user_id)
```

行为：

- 查询当前用户的会话。
- 按创建时间读取消息。
- 将连续的 user/assistant 消息组织为 `(user_msg, assistant_msg)`。

## 7. JWT 和认证实现

Django 侧：

```text
DjangoUserService/apps/user/authentications.py
```

FastAPI 侧：

```text
backend/app/utils/auth_utils.py
```

Django 负责：

- 登录注册后签发 token。
- 刷新 token。
- 登出时拉黑 token。
- DRF 请求认证。

FastAPI 负责：

- 解析 Bearer Token。
- 用同一密钥校验 token。
- 检查 Redis 中是否存在黑名单 key。
- 从 payload 获取 `user_id`。

## 8. Redis 实现

FastAPI Redis：

```text
backend/app/db/redis_config.py
```

用途：

- 限流。
- 用户信息缓存。
- Token 黑名单检查。

Django Redis：

```text
DjangoUserService/settings.py
DjangoUserService/apps/utils/cache_utils.py
```

用途：

- Django cache。
- 用户信息缓存。
- Token 黑名单。

注意：

- FastAPI Redis 配置当前写死为 `localhost:6379 db=3`。
- Django Redis 来自环境变量 `REDIS_CACHE_URL`。
- 后续生产化建议统一配置来源和 key 前缀。

## 9. 前端流式渲染实现

核心文件：

```text
front/src/views/AIChat.vue
```

主要步骤：

1. 用户点击发送。
2. 前端把用户消息追加到 `messages`。
3. 追加一个空的 assistant 消息作为占位。
4. 使用 `fetch` 请求 `/api/agent/query/stream`。
5. 读取 `response.body.getReader()`。
6. 用 `TextDecoder` 解码。
7. 按 `data: ` 行解析 JSON。
8. 将 `response.content` 逐字追加到最后一条 assistant 消息。
9. 收到 `done` 后保存 `session_id` 并跳转 `/aichat/{session_id}`。

Markdown 渲染：

- `marked`
- `marked-highlight`
- `highlight.js`
- `DOMPurify`

## 10. Django 用户服务实现

核心文件：

```text
DjangoUserService/apps/user/models.py
DjangoUserService/apps/user/serializers.py
DjangoUserService/apps/user/views.py
DjangoUserService/apps/user/authentications.py
```

用户模型：

- 继承 `AbstractBaseUser`。
- 主键 `uuid`。
- 登录字段 `email`。
- 额外字段：`telephone`、`gender`、`bio`、`avatar`、`status`。

API View：

| View | 作用 |
| --- | --- |
| `LoginView` | 登录。 |
| `RegisterView` | 注册。 |
| `ResetPasswordView` | 重置密码。 |
| `TokenRefreshView` | 刷新 token。 |
| `UserDetailView` | 用户详情。 |
| `UserUpdateView` | 更新用户。 |
| `UserLogOutView` | 登出。 |

缓存：

- `cache_user_info` 装饰器缓存用户详情。
- `clear_user_cache` 在用户信息变化后清理。

## 11. 配置实现

配置来源：

| 类型 | 位置 |
| --- | --- |
| 环境变量 | `.env` |
| Python 依赖 | `backend/pyproject.toml`、`DjangoUserService/pyproject.toml` |
| 前端依赖 | `front/package.json` |
| Chroma 配置 | `backend/app/config/chroma.yaml` |
| Prompt 配置 | `backend/app/prompt/*.txt` |

建议：

- 环境变量用于密钥、数据库地址、模型供应商。
- YAML 用于 RAG 参数、向量库参数、Prompt 类型。
- Prompt 文件独立管理，避免写死在代码中。

## 12. 当前实现的可改进点

| 点 | 建议 |
| --- | --- |
| `ChatService` 偏大 | 拆成 Agent、RAG、Vector、Session 多个应用服务。 |
| 知识文件同步入库 | 改成后台任务，提供索引状态查询。 |
| Redis 配置分散 | FastAPI 侧改为读取环境变量。 |
| 前端 API 路径不统一 | 统一使用 `apiConfig.baseURL` 和 `userBaseURL` 或明确代理规则。 |
| DOCX 加载 | 使用 `UnstructuredWordDocumentLoader` 或专门 docx loader。 |
| Chroma 多用户隔离 | 按用户/知识库建立 collection 或增加严格 metadata filter。 |
| 文档编码 | 统一 UTF-8，修复 README/API 文档乱码。 |

