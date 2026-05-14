# 项目入门

## 1. 项目定位

当前项目是一个基于 `Vue 3 + FastAPI + Django REST + LangChain` 的 RAG 问答系统。它的核心能力是：

- 用户注册、登录、资料管理和头像上传。
- AI 聊天和会话管理。
- 上传知识文件，写入向量库。
- 使用 RAG 从知识库检索内容并生成回答。
- 使用 LangChain Agent 调用 RAG、时间、天气、用户信息、重排等工具。
- 前端通过 SSE 展示流式回答。

## 2. 三个主要运行单元

```text
LangChain-RAG-FastAPI-Service/
├── front/              # Vue 3 前端
├── backend/            # FastAPI AI/RAG 服务
└── DjangoUserService/  # Django 用户服务
```

| 运行单元 | 技术 | 主要职责 |
| --- | --- | --- |
| `front` | Vue 3、Vite、Pinia、Vue Router、Vant | 页面展示、用户交互、登录状态、聊天流式渲染。 |
| `backend` | FastAPI、LangChain、Chroma、SQLAlchemy、Redis | AI 聊天、RAG、Agent、会话、知识库向量化。 |
| `DjangoUserService` | Django、DRF、MySQL、Redis | 用户注册登录、JWT、用户资料、头像上传。 |

## 3. 先看哪些文件

建议按下面顺序阅读：

| 顺序 | 文件 | 读它的目的 |
| --- | --- | --- |
| 1 | `front/src/config/api.js` | 看前端会调用哪些接口。 |
| 2 | `front/src/views/AIChat.vue` | 看聊天页面如何发送请求和消费 SSE。 |
| 3 | `backend/main.py` | 看 FastAPI 如何启动、注册路由、中间件和生命周期事件。 |
| 4 | `backend/app/router/chat.py` | 看 AI/RAG/会话/向量库 API 入口。 |
| 5 | `backend/app/router/chat_service.py` | 看路由背后的业务编排。 |
| 6 | `backend/app/rag/rag_service.py` | 看 RAG 查询主流程。 |
| 7 | `backend/app/rag/vector_store.py` | 看文档入库和检索器构造。 |
| 8 | `backend/app/agent/agent.py` | 看 Agent 如何创建、调用工具、流式返回。 |
| 9 | `DjangoUserService/apps/user/views.py` | 看用户 API。 |
| 10 | `DjangoUserService/apps/user/authentications.py` | 看 JWT 生成、校验、刷新、拉黑。 |

## 4. 目录职责速查

### `front`

```text
front/src/
├── views/        # 页面组件
├── store/        # Pinia 状态
├── router/       # Vue Router
├── config/       # API 配置
├── components/   # 公共组件
└── i18n/         # 国际化
```

| 目录/文件 | 职责 |
| --- | --- |
| `views/AIChat.vue` | 聊天主界面，发送用户问题，逐步渲染 SSE 返回内容。 |
| `views/Sessions.vue` | 会话列表。 |
| `views/Login.vue`、`Register.vue` | 登录注册。 |
| `views/Profile.vue` | 用户资料和头像上传。 |
| `store/user.js` | 用户资料、登录状态、JWT 持久化。 |
| `store/session.js` | 会话列表、当前会话、删除会话。 |
| `config/api.js` | API endpoint 配置。 |

### `backend`

```text
backend/app/
├── router/     # FastAPI 路由和应用服务
├── rag/        # RAG 能力
├── agent/      # Agent 能力
├── services/   # 会话管理
├── models/     # SQLAlchemy ORM
├── schemas/    # Pydantic 模型
├── db/         # MySQL/Redis 连接
├── core/       # 日志、异常、响应、限流
├── utils/      # 配置、工厂、文件处理、认证辅助
├── config/     # YAML 配置
└── prompt/     # Prompt 模板
```

| 目录/文件 | 职责 |
| --- | --- |
| `main.py` | FastAPI 应用入口。 |
| `router/chat.py` | AI/RAG/会话/向量接口。 |
| `router/chat_service.py` | 路由层背后的业务编排。 |
| `rag/rag_service.py` | RAG 检索、重排、生成主流程。 |
| `rag/vector_store.py` | Chroma、BM25、混合检索、文档入库。 |
| `rag/reorder_service.py` | CrossEncoder 重排。 |
| `agent/agent.py` | Agent 创建和 SSE 流式输出。 |
| `agent/agent_tools.py` | Agent 可调用工具。 |
| `services/database_session_manager.py` | MySQL 会话历史读写。 |
| `utils/factory.py` | LLM 和 Embedding 模型工厂。 |

### `DjangoUserService`

```text
DjangoUserService/
├── DjangoUserService/  # Django 项目配置
└── apps/
    ├── user/           # 用户模块
    ├── file/           # 文件/头像上传
    ├── utils/          # 缓存、限流
    └── secret/         # 加密工具
```

| 目录/文件 | 职责 |
| --- | --- |
| `DjangoUserService/settings.py` | Django 配置、数据库、Redis cache、DRF。 |
| `DjangoUserService/urls.py` | 总路由，挂载 `user/` 和 `file/`。 |
| `apps/user/models.py` | 自定义用户模型。 |
| `apps/user/views.py` | 注册、登录、资料、登出、刷新 token。 |
| `apps/user/authentications.py` | JWT 认证和 token 管理。 |
| `apps/file/views.py` | 头像上传。 |

## 5. 技术栈清单

| 层级 | 技术 | 项目位置 |
| --- | --- | --- |
| 前端 | Vue 3、Vite、Pinia、Vue Router、Vant | `front/package.json` |
| AI API | FastAPI、Uvicorn、Pydantic | `backend/pyproject.toml` |
| RAG | LangChain、Chroma、BM25、CrossEncoder | `backend/app/rag/` |
| Agent | LangChain AgentExecutor、tool calling | `backend/app/agent/` |
| LLM | 阿里云 DashScope 或 Ollama | `backend/app/utils/factory.py` |
| 数据库 | MySQL、SQLAlchemy async | `backend/app/db/db_config.py` |
| 缓存 | Redis | `backend/app/db/redis_config.py`、`DjangoUserService/settings.py` |
| 用户服务 | Django、DRF、JWT | `DjangoUserService/` |

## 6. 本地启动顺序

从 README 和配置看，推荐按这个顺序启动依赖和服务：

1. MySQL。
2. Redis。
3. Ollama 或配置阿里云 DashScope。
4. `DjangoUserService` 用户服务。
5. `backend` FastAPI AI 服务。
6. `front` Vue 前端。

常见端口：

| 服务 | 端口 |
| --- | --- |
| FastAPI AI 服务 | `8000` |
| Django 用户服务 | `8001` |
| Vue 前端 | `3000` 或 Vite 默认端口 |
| MySQL | `3306` |
| Redis | `6379` |
| Ollama | `11434` |

## 7. 入门学习检查表

- [ ] 能说清楚 `front`、`backend`、`DjangoUserService` 分别负责什么。
- [ ] 能从 `AIChat.vue` 找到流式请求入口。
- [ ] 能从 `chat.py` 找到 `/api/agent/query/stream`。
- [ ] 能解释 `ChatService` 为什么是业务编排层。
- [ ] 能画出 RAG 文档入库流程。
- [ ] 能画出 RAG 查询流程。
- [ ] 能解释 Django JWT 如何被 FastAPI 校验。
- [ ] 能说出 Redis 在两个后端中分别承担什么职责。

