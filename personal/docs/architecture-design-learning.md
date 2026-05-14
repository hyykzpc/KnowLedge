# 架构设计学习笔记

> 目标：结合当前 `LangChain-RAG-FastAPI-Service` 项目目录，理解项目采用的技术架构、模块职责、实现方式，以及这些设计和常见架构模式之间的关系。

## 1. 如何阅读这个项目

这个项目不是一个单后端项目，而是一个前后端分离、多后端服务协作的 RAG 应用。可以从三个运行单元开始理解：

```text
LangChain-RAG-FastAPI-Service/
├── front/              # Vue 3 前端，负责页面、路由、状态管理、SSE 流式展示
├── backend/            # FastAPI + LangChain AI 服务，负责 RAG、Agent、会话、向量库
├── DjangoUserService/  # Django REST 用户服务，负责注册、登录、JWT、用户资料、头像上传
├── docs/               # 文档
├── images/             # README 使用的截图资源
└── README.md           # 项目说明，当前终端显示存在编码乱码
```

推荐阅读顺序：

1. 先看 `front/src/config/api.js`，理解前端调用了哪些 API。
2. 再看 `backend/main.py`，理解 FastAPI 启动、路由、中间件和生命周期。
3. 看 `backend/app/router/chat.py` 和 `backend/app/router/chat_service.py`，理解 AI 服务的请求入口和业务编排。
4. 看 `backend/app/rag/`，理解文档处理、切分、向量化、检索、重排、摘要。
5. 看 `backend/app/agent/`，理解 Agent 如何调用工具和流式输出。
6. 看 `DjangoUserService/apps/user/`，理解用户认证和 JWT 如何生成、校验、拉黑。
7. 看 `front/src/views/AIChat.vue` 和 `front/src/store/`，理解前端如何消费 SSE 和维护登录/会话状态。

## 2. 当前项目的总体架构

从 C4 模型看，当前项目可以这样理解：

```text
用户
  ↓
Vue 前端 front/
  ↓ REST / SSE
FastAPI AI 服务 backend/
  ├── LangChain Agent
  ├── RAG 检索增强生成
  ├── Chroma 向量库
  ├── MySQL 聊天会话库
  └── Redis 限流、缓存、Token 黑名单检查
  ↓
DjangoUserService 用户服务
  ├── 用户注册/登录/资料
  ├── JWT 生成和校验
  ├── 头像上传
  └── MySQL 用户库
```

它对应的架构类型：

| 视角 | 当前项目体现 |
| --- | --- |
| 前后端分离架构 | `front` 独立于后端，通过 HTTP API 调用服务。 |
| 多服务架构 | `backend` 和 `DjangoUserService` 是两个后端运行单元。 |
| 分层架构 | API 路由层、业务编排层、领域能力层、数据访问层分开。 |
| RAG 管线架构 | 文档加载 -> 切分 -> Embedding -> 向量库 -> 检索 -> 重排 -> LLM 摘要。 |
| Agent 工具调用架构 | Agent 通过工具函数调用 RAG、用户信息、时间、天气、重排等能力。 |
| 事件/流式交互 | FastAPI 使用 SSE 向前端返回逐字流式响应。 |

## 3. 模块职责地图

### 3.1 `front/`：前端展示层

`front` 是 Vue 3 + Vite 应用，主要职责是用户交互、页面路由、状态管理和 API 调用。

```text
front/
├── src/main.js              # Vue 应用入口
├── src/App.vue              # 根组件
├── src/router/index.js      # 页面路由
├── src/config/api.js        # API 路径配置
├── src/store/               # Pinia 状态管理
├── src/views/               # 页面
├── src/components/          # 公共组件
├── src/i18n/                # 国际化
└── package.json             # 前端依赖
```

关键模块：

| 模块 | 作用 | 架构位置 |
| --- | --- | --- |
| `src/views/AIChat.vue` | AI 聊天主页面，发送消息、消费 SSE、Markdown 渲染、代码高亮。 | 展示层 + 交互编排 |
| `src/views/Login.vue`、`Register.vue` | 登录注册页面。 | 展示层 |
| `src/views/Sessions.vue` | 会话列表页。 | 展示层 |
| `src/views/Profile.vue` | 用户资料和头像上传。 | 展示层 |
| `src/store/user.js` | 登录状态、JWT、用户信息、注册登录请求。 | 前端状态层 |
| `src/store/session.js` | 会话列表、当前会话、拉取/删除会话。 | 前端状态层 |
| `src/config/api.js` | 集中管理 API endpoint。 | 前端基础设施层 |
| `src/router/index.js` | 页面路由：`/aichat`、`/sessions`、`/profile` 等。 | 前端路由层 |

前端体现的架构思想：

| 架构概念 | 当前项目例子 |
| --- | --- |
| 组件化 | 每个页面是 Vue 组件，底部导航为 `TabBar.vue`。 |
| 状态集中管理 | 用户、会话、主题、语言分别用 Pinia store 管理。 |
| API 配置集中化 | `api.js` 避免 API 路径散落在所有组件里。 |
| 流式 UI | `AIChat.vue` 使用 `fetch` 读取 `ReadableStream`，按 SSE 消息逐步更新页面。 |

需要注意：

- `apiConfig` 定义了 `baseURL` 和 `userBaseURL`，但当前部分调用直接使用相对路径，例如 `/user/login/`、`/api/agent/query/stream`。这意味着开发环境通常依赖 Vite 代理或同源网关。
- 前端把 JWT 存入 `localStorage`，简单直接，但生产环境要评估 XSS 风险。

### 3.2 `backend/`：FastAPI AI/RAG 编排服务

`backend` 是项目的 AI 核心服务，负责聊天、RAG、Agent、会话、向量库和限流。

```text
backend/
├── main.py                  # FastAPI 应用入口
├── app/
│   ├── router/              # API 路由和业务编排入口
│   ├── rag/                 # RAG 检索增强生成
│   ├── agent/               # LangChain Agent 和工具
│   ├── services/            # 会话服务
│   ├── models/              # SQLAlchemy ORM 模型
│   ├── schemas/             # Pydantic 请求/响应模型
│   ├── db/                  # MySQL、Redis 连接
│   ├── cache/               # Redis 缓存装饰器
│   ├── core/                # 日志、响应、异常、限流
│   ├── utils/               # 配置、模型工厂、文件加载、路径工具
│   ├── config/              # YAML 配置
│   └── prompt/              # Prompt 模板
├── pyproject.toml           # Python 依赖
└── api.md                   # API 文档，当前终端显示存在编码乱码
```

#### 3.2.1 应用入口：`backend/main.py`

主要职责：

- 创建 `FastAPI()` 应用。
- 注册全局限流中间件 `RateLimitMiddleware`。
- 添加请求耗时响应头 `X-Process-Time`。
- 注册路由：`chat_router`、`health_router`、`user_router`。
- 配置 CORS。
- 注册统一异常处理器。
- 启动时初始化 MySQL、会话管理器、Redis、Reranker 模型。
- 关闭时断开 Redis。

对应架构：

| 概念 | 当前实现 |
| --- | --- |
| Composition Root | `main.py` 负责组装应用、路由、中间件和启动资源。 |
| Middleware | 限流、耗时统计、CORS。 |
| Lifecycle Hooks | `startup_event`、`shutdown_event` 管理外部资源。 |

#### 3.2.2 API 路由层：`backend/app/router/`

```text
backend/app/router/
├── chat.py          # AI/RAG/会话/向量库/重排 API
├── chat_service.py  # 路由背后的业务编排服务
├── user.py          # FastAPI 侧用户信息代理接口
└── health.py        # 健康检查
```

`chat.py` 是薄路由层，负责：

- 定义 API 路径。
- 解析请求模型。
- 注入依赖，例如当前用户 ID、限流、`ChatService`。
- 返回统一响应或 SSE。

主要 API：

| API | 作用 | 依赖 |
| --- | --- | --- |
| `POST /api/agent/query/stream` | Agent 流式问答。 | JWT、限流、`get_agent_stream_response` |
| `POST /api/rag/query` | 直接 RAG 问答。 | `ChatService.handle_rag_query` |
| `GET /api/session/{session_id}` | 查询会话历史。 | JWT、MySQL 会话 |
| `DELETE /api/session/{session_id}` | 删除会话。 | JWT、MySQL 会话 |
| `GET /api/sessions/{user_id}` | 查询用户会话列表。 | JWT、权限校验 |
| `POST /api/vector/add/single` | 上传单个知识文件到向量库。 | JWT、文件校验、Chroma |
| `POST /api/vector/add/multiple` | 批量上传知识文件到向量库。 | JWT、并发处理、Chroma |
| `DELETE /api/vector/clean` | 清理当前用户向量文档。 | JWT、Chroma metadata |
| `POST /api/reorder` | 对文本列表重排。 | Reranker |

`chat_service.py` 是业务编排层，负责把路由请求转发给具体能力：

| 方法 | 作用 |
| --- | --- |
| `handle_agent_query` | 取历史 -> 调用 Agent -> 保存消息。 |
| `handle_rag_query` | 创建 `RagService` 并执行 RAG 摘要。 |
| `handle_get_session` | 读取会话历史。 |
| `handle_get_user_sessions` | 校验用户只能看自己的会话。 |
| `handle_add_vector_single` | 校验文件大小/类型，写入向量库。 |
| `handle_add_vector_multiple` | 批量校验文件，并发写入向量库。 |
| `clean_user_upload` | 删除当前用户上传到 Chroma 的文档。 |
| `handle_reorder` | 调用 Reranker 服务。 |

对应架构：

| 概念 | 当前项目例子 |
| --- | --- |
| Controller / Router | `chat.py` |
| Application Service | `ChatService` |
| Dependency Injection | FastAPI `Depends(...)` |
| Facade | `ChatService` 对路由屏蔽 RAG、Agent、向量库、会话细节。 |

#### 3.2.3 RAG 模块：`backend/app/rag/`

```text
backend/app/rag/
├── rag_service.py       # RAG 主流程
├── vector_store.py      # Chroma、BM25、文档入库、检索器
├── text_spliter.py      # 文本切分
├── reorder_service.py   # CrossEncoder 重排
└── __init__.py
```

RAG 流程可以拆成两条：

文档入库流程：

```text
上传文件
  ↓
ChatService 校验文件大小和 MIME/后缀
  ↓
VectorStoreService.get_document()
  ↓
file_handler 加载 PDF/TXT/MD/PPTX/DOCX
  ↓
AsyncTextSplitter 切分文本
  ↓
添加 user_id metadata
  ↓
Chroma.add_documents()
  ↓
记录 MD5，避免重复入库
```

问答流程：

```text
用户问题
  ↓
RagService.generate_hypothetical_document()
  ↓
HyDE 生成假想文档
  ↓
VectorStoreService.get_retriever()
  ↓
Chroma 相似度检索 + BM25 关键词检索
  ↓
EnsembleRetriever 融合结果
  ↓
ReorderService 使用 CrossEncoder 重排
  ↓
选前 3 篇文档分别摘要
  ↓
合并摘要后再生成最终回答
```

关键实现：

| 模块 | 实现重点 |
| --- | --- |
| `VectorStoreService` | 使用 `langchain_chroma.Chroma` 持久化向量，支持 `BM25Retriever` 和 `EnsembleRetriever`。 |
| `AsyncTextSplitter` | 对 `RecursiveCharacterTextSplitter` 做异步封装。 |
| `RagService` | 使用 LangChain Runnable 管线：`PromptTemplate | chat_model | StrOutputParser`。 |
| `ReorderService` | 使用 `sentence_transformers.CrossEncoder`，本地加载 `Qwen/Qwen3-Reranker-0.6B`。 |
| `file_handler.py` | 支持 PDF、TXT、Markdown、PPTX、DOCX 等加载器。 |

对应架构：

| 架构概念 | 当前项目例子 |
| --- | --- |
| Pipeline Architecture | 文档入库和 RAG 查询都按阶段处理。 |
| Hybrid Search | 向量检索 + BM25 关键词检索。 |
| Retrieval-Rerank-Generate | 先粗召回，再重排，再生成。 |
| Strategy / Factory | 模型根据环境变量选择 Aliyun 或 Ollama。 |
| Metadata Isolation | 上传文档通过 `user_id` metadata 区分用户数据。 |

#### 3.2.4 Agent 模块：`backend/app/agent/`

```text
backend/app/agent/
├── agent.py             # AgentFactory、AgentExecutor、流式响应
├── agent_tools.py       # Agent 可调用工具
├── agent_middleware.py  # Agent/模型/工具调用日志 hook
└── __init__.py
```

`AgentFactory` 做了这些事情：

- 根据 `LLM_TYPE` 选择模型：`OLLAMA` 或 `ALIYUN`。
- 加载默认工具。
- 加载系统 Prompt。
- 创建 `create_tool_calling_agent(...)`。
- 包装成 `AgentExecutor`。

默认工具：

| 工具 | 作用 |
| --- | --- |
| `rag_summary_tools` | 调用 RAG 摘要能力。 |
| `reorder_documents_tools` | 对候选文档重排。 |
| `get_user_info_tools` | 获取用户信息。 |
| `get_weather_tools` | 获取天气信息。 |
| `what_time_is_now` | 获取当前时间。 |

Agent 流式响应：

```text
前端 POST /api/agent/query/stream
  ↓
FastAPI StreamingResponse
  ↓
get_agent_stream_response()
  ↓
读取 MySQL 历史会话
  ↓
AgentExecutor.astream()
  ↓
逐字符 yield SSE data
  ↓
保存用户消息和 AI 回复到 MySQL
```

对应架构：

| 架构概念 | 当前项目例子 |
| --- | --- |
| Tool-Using Agent | Agent 根据问题选择 RAG、时间、用户信息等工具。 |
| Factory Pattern | `AgentFactory` 封装模型、工具、Prompt 创建逻辑。 |
| Streaming Architecture | 后端用 SSE，前端逐块读取并渲染。 |
| Memory / Conversation State | `DatabaseSessionManager` 从 MySQL 读取历史对话。 |

#### 3.2.5 数据访问和状态：`backend/app/db/`、`services/`、`models/`

```text
backend/app/db/
├── db_config.py       # SQLAlchemy async MySQL engine/session
└── redis_config.py    # Redis async client

backend/app/services/
└── database_session_manager.py

backend/app/models/
└── chat_history.py
```

MySQL 模型：

| 表 | ORM 类 | 作用 |
| --- | --- | --- |
| `chat_sessions` | `ChatSession` | 会话 ID、用户 ID、标题、创建/更新时间。 |
| `chat_messages` | `ChatMessage` | 每条用户/助手消息，按 `session_id` 归属。 |

Redis 用途：

- FastAPI 全局限流。
- FastAPI 单接口限流。
- 查询 Django 用户信息的缓存。
- 校验 Django JWT 黑名单。

对应架构：

| 架构概念 | 当前项目例子 |
| --- | --- |
| Repository-ish Service | `DatabaseSessionManager` 封装会话读写。 |
| Unit of Work | SQLAlchemy session 生命周期管理。 |
| Cache Aside | 查用户信息时先读 Redis，未命中再请求 Django。 |
| Rate Limiting | Redis key 计数窗口。 |

#### 3.2.6 配置、Prompt 和基础设施：`backend/app/utils/`、`config/`、`prompt/`

```text
backend/app/utils/
├── factory.py          # LLM、Embedding 模型工厂
├── config.py           # 读取 YAML 配置
├── config_handler.py   # 配置加载函数
├── prompt_loader.py    # Prompt 文件加载
├── file_handler.py     # 文档加载
├── auth_utils.py       # FastAPI 侧 JWT 解码和用户信息代理
└── path_tool.py        # 路径工具

backend/app/config/
├── chroma.yaml
├── rag.yaml
├── agent.yaml
└── prompt.yaml

backend/app/prompt/
├── main_prompt.txt
├── rag_summarize.txt
├── reorder_prompt.txt
└── report_prompt.txt
```

重点：

- `factory.py` 通过环境变量 `LLM_TYPE`、`EMBED_MODEL_TYPE` 切换 Ollama 或阿里云 DashScope。
- `chroma.yaml` 配置 Chroma collection、持久化目录、检索数量、切分大小、允许的文件类型。
- `prompt/` 把 Prompt 从代码中抽出，方便调整。

对应架构：

| 架构概念 | 当前项目例子 |
| --- | --- |
| Configuration Externalization | 模型、数据库、Redis、向量库参数来自 `.env` 和 YAML。 |
| Factory Pattern | `ChatModelFactory`、`EmbedModelFactory`。 |
| Prompt as Resource | Prompt 文件独立保存，不直接写死在路由里。 |

### 3.3 `DjangoUserService/`：用户与文件服务

`DjangoUserService` 是独立 Django REST 服务，负责用户认证、用户资料、JWT 和头像上传。

```text
DjangoUserService/
├── manage.py
├── DjangoUserService/
│   ├── settings.py       # Django 配置
│   ├── urls.py           # 总路由
│   ├── celery.py         # Celery 配置
│   ├── asgi.py
│   └── wsgi.py
├── apps/
│   ├── user/             # 用户、JWT、资料
│   ├── file/             # 文件/头像上传
│   ├── utils/            # 缓存、限流工具
│   └── secret/
├── templates/
└── pyproject.toml
```

#### 3.3.1 用户模块：`apps/user/`

```text
apps/user/
├── models.py             # 自定义 User 模型
├── serializers.py        # DRF 序列化和校验
├── views.py              # 登录、注册、资料、登出等 API
├── urls.py               # user 路由
├── authentications.py    # JWT 认证和 Token 生成/刷新/拉黑
└── fatherClass.py        # 认证基类视图
```

主要 API：

| API | 作用 |
| --- | --- |
| `POST /user/register/` | 注册用户并返回 JWT。 |
| `POST /user/login/` | 登录并返回 JWT。 |
| `POST /user/reset-password/` | 修改密码，拉黑旧 Token，返回新 Token。 |
| `POST /user/refresh-token/` | 刷新 Token，拉黑旧 Token。 |
| `GET /user/detail/` | 查询当前用户资料。 |
| `PUT /user/update/` | 更新用户资料，清理缓存，返回新 Token。 |
| `POST /user/logout/` | 登出并拉黑 Token。 |

核心设计：

- 使用自定义 `User` 模型，主键为 `ShortUUIDField`。
- `JWTTokenGenerator` 生成包含 `user_id`、`username`、`email`、`exp`、`iat`、`jti` 的 token。
- Token 黑名单用 Redis cache 保存 `blacklist:{jti}`。
- 用户信息使用 cache-aside 缓存，`clear_user_cache` 在资料更新、密码更新、头像上传后清理。

对应架构：

| 架构概念 | 当前项目例子 |
| --- | --- |
| Identity Service | Django 服务专门负责用户身份。 |
| Token-Based Authentication | JWT Bearer Token。 |
| Cache Aside | `cache_user_info` 装饰器。 |
| Blacklist Token Revocation | Redis 保存 `jti` 黑名单。 |

#### 3.3.2 文件模块：`apps/file/`

```text
apps/file/
├── views.py          # UploadAPIView
├── serializers.py    # ImgSerializer
├── urls.py           # /file/upload/
└── models.py
```

作用：

- 接收头像图片。
- 写入 `MEDIA_ROOT/img/`。
- 更新当前用户 `avatar` 字段。
- 清理用户缓存。
- 返回图片 URL。

这个模块更接近“用户资料附件服务”，和 `backend` 里的知识文件上传不是同一类上传：

| 上传类型 | 所属服务 | 目的 |
| --- | --- | --- |
| 头像上传 | `DjangoUserService/apps/file` | 更新用户头像。 |
| 知识文件上传 | `backend/app/router/chat.py` + `VectorStoreService` | 文档向量化，参与 RAG 检索。 |

## 4. 当前项目的架构分层

可以把当前项目理解为两层分层架构叠加：整体系统层、每个服务内部层。

### 4.1 整体系统层

```text
表现层
  front/

API/服务层
  backend/ FastAPI AI 服务
  DjangoUserService/ Django 用户服务

AI 能力层
  LangChain Agent
  RAG
  Embedding
  Reranker
  LLM

数据层
  MySQL 用户库
  MySQL 聊天会话库
  Redis 缓存/限流/Token 黑名单
  Chroma 向量库
  本地文件/media/data
```

### 4.2 FastAPI 服务内部层

```text
Router 层
  app/router/chat.py

Application Service 层
  app/router/chat_service.py

Domain Capability 层
  app/rag/*
  app/agent/*

Infrastructure 层
  app/db/*
  app/models/*
  app/utils/factory.py
  app/utils/file_handler.py
  app/core/*
```

### 4.3 Django 服务内部层

```text
URL 路由层
  DjangoUserService/urls.py
  apps/user/urls.py
  apps/file/urls.py

View/API 层
  apps/user/views.py
  apps/file/views.py

Serializer/Validation 层
  apps/user/serializers.py
  apps/file/serializers.py

Domain/Auth 层
  apps/user/authentications.py
  apps/user/fatherClass.py

Model/Data 层
  apps/user/models.py
  MySQL

Infrastructure 层
  settings.py
  Redis cache
  media files
```

## 5. 当前项目的关键运行流程

### 5.1 登录流程

```text
front Login.vue / user store
  ↓ POST /user/login/
Django LoginView
  ↓ LoginSerializer 校验
User 模型查询
  ↓
JWTTokenGenerator.generate_token()
  ↓
返回 user + token
  ↓
front 保存 token 到 localStorage
```

### 5.2 AI 聊天流式问答流程

```text
front AIChat.vue
  ↓ POST /api/agent/query/stream
FastAPI chat.py
  ↓ get_current_user_id 解码 Django JWT
  ↓ get_agent_stream_response()
DatabaseSessionManager.get_history()
  ↓
AgentExecutor.astream()
  ↓ 工具调用：RAG / 用户信息 / 时间 / 天气 / 重排
  ↓
SSE data 持续返回前端
  ↓
DatabaseSessionManager.add_message()
```

### 5.3 RAG 查询流程

```text
POST /api/rag/query
  ↓
ChatService.handle_rag_query()
  ↓
RagService.rag_summary()
  ↓
HyDE 生成假想文档
  ↓
Chroma + BM25 检索
  ↓
CrossEncoder 重排
  ↓
PromptTemplate + LLM 生成摘要
```

### 5.4 知识文件上传流程

```text
POST /api/vector/add/single
  ↓
JWT 获取 user_id
  ↓
校验文件大小和类型
  ↓
临时保存上传文件
  ↓
file_handler 加载文档
  ↓
AsyncTextSplitter 切分
  ↓
写入 user_id metadata
  ↓
Chroma.add_documents()
```

### 5.5 用户资料缓存流程

```text
FastAPI 需要用户信息
  ↓
Redis 查询 user:{user_id}
  ↓ 未命中
Django /user/detail/
  ↓
写 Redis，TTL 3600 秒
```

## 6. 当前架构的优点和代价

### 优点

| 设计 | 收益 |
| --- | --- |
| 前后端分离 | 前端体验和后端能力可以独立开发。 |
| FastAPI 与 Django 分离 | AI/RAG 服务和用户服务职责清晰。 |
| LangChain 抽象 | 便于切换模型、Prompt、工具和 RAG 流程。 |
| Chroma 持久化 | 本地向量库上手简单，适合学习和原型。 |
| Redis 限流和缓存 | 简单解决热点请求、用户信息缓存、Token 黑名单。 |
| SSE 流式输出 | AI 回复体验更好，不必等待完整响应。 |
| 文档 metadata 带 `user_id` | 为多用户隔离提供基础。 |

### 代价

| 设计 | 代价 |
| --- | --- |
| 多后端服务 | 本地启动和部署复杂度高于单体。 |
| JWT 由 Django 签发，FastAPI 解码 | 两边必须共享 `SECRET_KEY` 和算法配置。 |
| Redis key 约定分散 | Django cache key 和 FastAPI 查询 key 需要保持一致。 |
| Chroma 本地持久化 | 扩展到分布式部署时需要重新设计向量库。 |
| `ChatService` 承担较多职责 | 后续功能增多时可能变成“上帝服务”。 |
| 前端部分 API 使用相对路径 | 部署时需要清楚代理或网关规则。 |
| 部分文档/注释编码异常 | 学习和维护成本上升，建议后续统一编码。 |

## 7. 它和常见架构设计的关系

### 7.1 单体架构

单体架构是把前后端外的所有后端能力放进一个应用中，例如：

```text
一个 Django 项目
├── user
├── chat
├── rag
├── vector
└── admin
```

如果本项目做成单体，可以把 FastAPI 的 AI/RAG 能力直接写成 Django app。

优点：

- 启动简单。
- 认证、数据库、缓存配置集中。
- 部署链路短。

缺点：

- AI/RAG 依赖重，可能拖慢普通用户服务。
- 流式响应、异步任务、模型加载对 Django 主服务压力较大。
- 业务边界不如现在清晰。

适合例子：

- 小型内部知识库系统。
- 学生项目或 MVP。
- 用户量小、功能集中、团队人数少的应用。

### 7.2 分层架构

分层架构把系统拆为表现层、业务层、数据层。当前项目内部已经大量使用这种模式。

典型结构：

```text
Controller / Router
  ↓
Service
  ↓
Repository / Model
  ↓
Database
```

当前项目例子：

```text
chat.py
  ↓
ChatService
  ↓
DatabaseSessionManager / RagService / VectorStoreService
  ↓
MySQL / Chroma / Redis
```

优点：

- 初学者容易理解。
- 模块职责清晰。
- 适合大多数 Web API。

缺点：

- 如果 Service 层膨胀，会出现大量业务逻辑堆积。
- 复杂领域逻辑可能被拆散在多个 Service 里。

### 7.3 微服务架构

微服务把系统拆成多个独立部署、独立拥有数据的服务。当前项目有微服务倾向，但还不是完整微服务体系。

当前项目的服务边界：

| 服务 | 职责 | 数据 |
| --- | --- | --- |
| `front` | 前端 UI | 浏览器状态 |
| `backend` | AI、RAG、会话、向量库 | 聊天 MySQL、Chroma、Redis |
| `DjangoUserService` | 用户、JWT、头像 | 用户 MySQL、media、Redis cache |

为什么说它“有微服务倾向”：

- 用户服务和 AI 服务独立进程。
- 两个后端可以分别启动。
- FastAPI 通过 HTTP 调用 Django 用户接口。

为什么还不是成熟微服务：

- JWT 密钥共享，服务间契约比较隐式。
- 没有服务发现、网关、统一配置中心、链路追踪体系。
- 数据一致性和跨服务错误处理还比较简单。

适合例子：

- 用户服务独立，AI 服务 GPU 资源独立部署。
- RAG 服务后续需要水平扩展。
- 不同团队分别维护用户系统和 AI 能力。

### 7.4 模块化单体

模块化单体是一个应用进程，但内部按业务模块严格隔离。

例子：

```text
app/
├── modules/
│   ├── user/
│   ├── chat/
│   ├── rag/
│   └── file/
└── shared/
```

如果当前项目改成模块化单体，可以合并 `backend` 和 `DjangoUserService`，但保持 `user`、`rag`、`chat` 模块边界。

优点：

- 部署比微服务简单。
- 模块边界比普通单体清楚。
- 适合从小项目逐渐演进。

缺点：

- AI 依赖和普通 Web 依赖会混在一个环境。
- 资源隔离弱，例如模型加载可能影响用户接口。

### 7.5 六边形架构 / 端口适配器架构

六边形架构强调核心业务不依赖外部技术，外部系统通过 Adapter 接入。

在当前项目中，可以这样类比：

```text
核心能力：
  RagService
  AgentFactory
  DatabaseSessionManager

外部适配器：
  FastAPI Router
  Chroma
  MySQL
  Redis
  DashScope/Ollama
  Django API
```

当前项目还没有完全实现六边形架构，因为很多核心类直接 import 具体基础设施，例如 `VectorStoreService` 直接使用 Chroma，`RagService` 直接依赖 `VectorStoreService`。

如果要往六边形架构演进，可以抽象接口：

```text
RetrieverPort
  ChromaRetrieverAdapter
  MilvusRetrieverAdapter
  ElasticsearchRetrieverAdapter

ChatModelPort
  DashScopeChatAdapter
  OllamaChatAdapter

UserInfoPort
  DjangoUserAdapter
```

适合例子：

- 你希望以后从 Chroma 切到 Milvus、pgvector 或 Elasticsearch。
- 你希望 LLM 供应商可插拔。
- 你希望核心 RAG 逻辑可以单元测试，不依赖真实外部服务。

### 7.6 Clean Architecture

Clean Architecture 和六边形架构类似，也强调依赖方向：外层依赖内层，内层不依赖外层。

理想结构：

```text
entities/
use_cases/
interface_adapters/
frameworks_drivers/
```

当前项目更接近实用分层架构，还没有严格的 use case 层和领域实体层。

如果套到本项目：

| Clean Architecture 层 | 当前对应 |
| --- | --- |
| Entities | `ChatSession`、`ChatMessage`、用户实体、文档片段概念 |
| Use Cases | 发送消息、上传知识文件、RAG 问答、登录注册 |
| Interface Adapters | FastAPI routers、Django views、Pydantic schemas、DRF serializers |
| Frameworks & Drivers | FastAPI、Django、SQLAlchemy、Chroma、Redis、LangChain |

适合例子：

- 长期维护的大型业务系统。
- 业务规则复杂，技术框架可能变化。
- 对测试隔离要求高。

### 7.7 CQRS

CQRS 把写操作和读操作分开。当前项目没有严格 CQRS，但已经有一些读写分离的影子：

| 写操作 | 读操作 |
| --- | --- |
| 上传知识文件到 Chroma | 从 Chroma/BM25 检索 |
| 保存聊天消息到 MySQL | 查询会话历史 |
| 更新用户资料 | 查询用户详情 |

如果要设计成 CQRS，可以这样：

```text
Command:
  UploadDocumentCommand
  SendMessageCommand
  UpdateUserProfileCommand

Query:
  GetSessionHistoryQuery
  SearchKnowledgeQuery
  GetUserProfileQuery
```

适合例子：

- 写入流程复杂，读取性能要求高。
- 要做搜索索引、缓存视图、异步投影。
- 会话列表、知识库检索、统计面板等读模型越来越多。

### 7.8 事件驱动架构

事件驱动架构通过事件解耦模块，例如“文档已上传”“用户已更新”“会话已创建”。

当前项目大多是同步调用：

```text
上传文件 -> 同步切分 -> 同步向量化 -> 同步写 Chroma
```

如果改成事件驱动：

```text
用户上传文件
  ↓
保存原始文件
  ↓ 发布 DocumentUploaded 事件
  ↓
后台 Worker 异步切分、Embedding、写入 Chroma
  ↓ 发布 DocumentIndexed 事件
  ↓
前端查询索引状态
```

适合例子：

- 文件很大，上传后不希望用户等待向量化完成。
- 需要任务重试、失败补偿、进度查询。
- Reranker 或 Embedding 计算很耗时。

本项目已有 Celery 配置，但当前主要用户流程没有大量使用异步任务。后续知识库入库非常适合迁移到 Celery/RQ/Arq 这类后台任务。

### 7.9 BFF 架构

BFF 是 Backend For Frontend，给特定前端提供专用后端聚合接口。

当前 `backend` 某种程度上承担了 AI 前端的 BFF：

- 前端只需要请求 `/api/agent/query/stream`。
- 后端内部处理 JWT、会话、Agent、RAG、工具调用。

但用户登录仍然直接打到 Django 的 `/user/login/`。

如果进一步 BFF 化，可以让前端只访问 FastAPI：

```text
front
  ↓
backend BFF
  ├── /api/auth/login -> 代理 Django
  ├── /api/user/detail -> 代理 Django
  ├── /api/agent/query/stream -> AI 服务
  └── /api/sessions -> 会话服务
```

优点：

- 前端只关心一个后端地址。
- 后端可以统一鉴权、错误格式、网关逻辑。

缺点：

- FastAPI 会承担更多代理职责。
- 如果边界不清，BFF 容易变成大杂烩。

### 7.10 RAG 专用架构

RAG 应用常见架构：

```text
Ingestion Pipeline
  Loader -> Splitter -> Embedding -> Vector Store

Query Pipeline
  Query Rewrite -> Retrieval -> Rerank -> Context Build -> Generate -> Cite
```

当前项目对应：

| RAG 阶段 | 当前实现 |
| --- | --- |
| Loader | `file_handler.py` |
| Splitter | `AsyncTextSplitter` |
| Embedding | `EmbedModelFactory`，Ollama 或 DashScope |
| Vector Store | `VectorStoreService` + Chroma |
| Keyword Search | BM25Retriever |
| Hybrid Retrieval | EnsembleRetriever |
| Query Rewrite | HyDE 假想文档 |
| Rerank | `ReorderService` + CrossEncoder |
| Generate | `RagService.chain` |
| Observability | `@traceable` 接入 LangSmith |

可改进方向：

- 增加引用来源返回，例如返回命中文档片段和 metadata。
- 增加文档状态表，记录上传、切分、入库、失败原因。
- 向量库按用户或知识库分 collection，强化隔离。
- 把入库任务异步化。
- 为 Prompt、检索参数、TopK、Rerank 参数增加管理接口。

## 8. 架构设计例子对比

### 8.1 当前设计：多服务 RAG 应用

```text
Vue
  ↓
FastAPI AI Service
  ├── Agent
  ├── RAG
  ├── Chat History
  └── Vector Store
  ↓
Django User Service
```

适合：

- AI 服务和用户服务想分开维护。
- AI 服务需要不同依赖、不同资源、不同扩展策略。
- 想学习 FastAPI + LangChain + Django 的组合。

### 8.2 简化设计：Django 单体 RAG

```text
Vue
  ↓
Django
  ├── user
  ├── chat
  ├── rag
  └── vector
```

适合：

- 初期快速上线。
- 用户量和知识库规模小。
- 团队熟悉 Django，不想维护两个后端。

### 8.3 AI 服务独立化设计

```text
Vue
  ↓
API Gateway
  ├── User Service
  ├── Chat Service
  ├── RAG Service
  └── File Indexing Worker
```

适合：

- 文件入库任务重。
- RAG 需要扩展多个 Worker。
- 多业务线共用一个 RAG 服务。

### 8.4 企业知识库设计

```text
Web App
  ↓
BFF
  ↓
Auth Service
Knowledge Service
Indexing Service
Retrieval Service
LLM Gateway
Audit Service
```

典型能力：

- 组织、角色、知识库权限。
- 文档版本管理。
- 异步索引任务和状态查询。
- 多向量库、多模型供应商。
- 审计日志和成本统计。

### 8.5 本地个人助手设计

```text
Desktop/Web UI
  ↓
FastAPI
  ├── Ollama
  ├── Chroma
  └── SQLite
```

适合：

- 单用户。
- 本地运行。
- 不需要 Django 用户系统。
- 不需要复杂部署。

## 9. 当前项目可以学习的设计点

### 9.1 为什么用户服务用 Django，AI 服务用 FastAPI

可能原因：

- Django/DRF 适合用户、权限、后台、ORM、Serializer。
- FastAPI 更适合异步 API、SSE、AI 服务编排。
- 两者拆开后，AI 服务可以独立安装 PyTorch、LangChain、Chroma、Reranker 等重依赖。

这是“按能力拆分服务”的设计。

### 9.2 为什么 RAG 模块单独放在 `app/rag`

RAG 是核心领域能力，不应该混在路由文件里。当前项目把以下能力集中在 `app/rag`：

- 检索。
- 文档切分。
- 向量入库。
- 重排。
- 摘要生成。

这让路由层保持较薄，也便于后续替换向量库或改检索策略。

### 9.3 为什么使用模型工厂

`factory.py` 通过环境变量选择模型：

```text
LLM_TYPE=ALIYUN 或 OLLAMA
EMBED_MODEL_TYPE=ALIYUN 或 OLLAMA
```

这对应 Factory Pattern。好处是业务代码不直接关心模型供应商：

```text
RagService -> chat_model
VectorStoreService -> embed_model
```

后续可以扩展：

- OpenAI。
- Azure OpenAI。
- vLLM。
- 本地 Hugging Face 模型。

### 9.4 为什么使用 SSE

AI 生成通常耗时较长。如果用普通 HTTP，用户必须等完整回答生成完成才看到结果。SSE 让后端可以持续推送：

```text
data: {"type": "response", "content": "你"}
data: {"type": "response", "content": "好"}
data: {"type": "done", "session_id": "..."}
```

适合：

- 单向流式输出。
- AI 打字机效果。
- 实现复杂度低于 WebSocket。

不适合：

- 双向实时通信。
- 多人协作编辑。
- 高频交互游戏。

### 9.5 为什么 Chroma 适合当前阶段

Chroma 优点：

- 本地持久化简单。
- 和 LangChain 集成方便。
- 适合学习和原型开发。

可能瓶颈：

- 多实例部署和高并发能力有限。
- 权限隔离、租户隔离需要自己设计。
- 大规模数据可能需要 Milvus、Qdrant、Weaviate、Elasticsearch、pgvector 等。

## 10. 代码中值得关注的风险点

这些不是要求马上修改，而是学习架构时需要识别的点。

| 风险 | 位置 | 原因 |
| --- | --- | --- |
| 文档/注释编码异常 | README、api.md、部分注释 | 影响团队学习和维护，建议统一 UTF-8。 |
| `ChatService` 职责偏多 | `backend/app/router/chat_service.py` | 同时处理 Agent、RAG、会话、上传、重排。 |
| 用户服务和 AI 服务共享 JWT 密钥 | `auth_utils.py`、Django settings | 部署时必须同步配置，密钥泄漏影响两个服务。 |
| Redis DB 配置不完全来自环境变量 | `backend/app/db/redis_config.py` | 当前写死 localhost、6379、db=3。 |
| 前端 API baseURL 使用不一致 | `front/src/config/api.js` 和 store/view | 部署依赖代理规则，排错成本高。 |
| 上传知识文件同步入库 | `handle_add_vector_single/multiple` | 大文件或模型慢时会阻塞请求。 |
| Word 文档加载实现可疑 | `file_handler.word_loader` | 当前使用 `TextLoader` 读取 docx，可能不适合真实 docx。 |
| Chroma 文档按 metadata 删除 | `delete_user_documents` | 依赖每个文档都正确写入 `user_id`。 |

## 11. 后续学习任务

可以按下面任务逐步深入：

1. 画一张系统上下文图：用户、前端、FastAPI、Django、MySQL、Redis、Chroma、LLM。
2. 跑通登录流程，记录前端、Django、Redis 中 token 的变化。
3. 跑通一次 `/api/agent/query/stream`，记录 SSE 消息格式。
4. 上传一个 txt 文件，观察 Chroma 持久化目录和 MD5 记录。
5. 追踪一次 RAG 查询，记录 HyDE、检索、重排、摘要每一步输入输出。
6. 把 `ChatService` 再拆分成 `AgentApplicationService`、`RagApplicationService`、`VectorApplicationService` 的设计草图。
7. 设计一个 ADR：为什么当前项目把用户服务和 AI 服务拆开。

## 12. 推荐补充的 ADR 文件

后续可以在 `docs/adr/` 下记录架构决策：

```text
docs/adr/
├── 0001-use-fastapi-for-ai-service.md
├── 0002-use-django-for-user-service.md
├── 0003-use-chroma-as-vector-store.md
├── 0004-use-sse-for-agent-streaming.md
└── 0005-use-hybrid-search-and-rerank-for-rag.md
```

ADR 模板：

```markdown
# 0001 标题

## 状态

Accepted

## 背景

为什么需要做这个决策。

## 决策

选择了什么方案。

## 备选方案

- 方案 A
- 方案 B
- 方案 C

## 后果

正面影响、负面影响、后续需要注意什么。
```

## 13. 参考资料

- C4 Model：用于从 System Context、Container、Component、Code 层次理解系统架构。<https://c4model.info/>
- arc42：用于组织架构文档，包括目标、约束、上下文、构建块、运行时视图、部署视图、质量要求、风险等。<https://arc42.org/overview>
- Diataxis：用于区分教程、指南、参考、概念解释。<https://nix.dev/contributing/documentation/diataxis>
- MADR / ADR：用于记录架构决策及其取舍。<https://adr.github.io/madr/>

