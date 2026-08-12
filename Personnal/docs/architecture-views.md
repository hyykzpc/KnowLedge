# 架构视图

本文用多个视角描述当前项目架构，帮助从整体到局部理解系统。

## 1. 系统上下文视图

```text
用户
  ↓ 浏览器
Vue 前端 front
  ↓ REST / SSE
FastAPI AI/RAG 服务 backend
  ├── MySQL：聊天会话
  ├── Redis：限流、缓存、JWT 黑名单检查
  ├── Chroma：向量库
  ├── DashScope/Ollama：LLM 和 Embedding
  └── DjangoUserService：用户资料和认证

DjangoUserService
  ├── MySQL：用户数据
  ├── Redis：用户缓存和 Token 黑名单
  └── media：头像文件
```

系统外部依赖：

| 依赖 | 用途 |
| --- | --- |
| DashScope | 阿里云模型服务，提供聊天模型和 Embedding。 |
| Ollama | 本地模型服务，提供聊天模型和 Embedding。 |
| ModelScope | 下载 `Qwen3-Reranker-0.6B`。 |
| LangSmith | RAG/Agent tracing。 |

## 2. 容器视图

| 容器 | 类型 | 技术 | 职责 |
| --- | --- | --- | --- |
| Web App | 前端 SPA | Vue 3、Vite | 页面展示、路由、状态、聊天流式渲染。 |
| AI Service | Web API | FastAPI、LangChain | Agent、RAG、会话、知识库。 |
| User Service | Web API | Django、DRF | 用户注册、登录、JWT、资料、头像。 |
| Chat DB | 数据库 | MySQL | 聊天会话和消息。 |
| User DB | 数据库 | MySQL | 用户账号和资料。 |
| Cache | 缓存 | Redis | 限流、缓存、Token 黑名单。 |
| Vector Store | 向量库 | Chroma | 文档向量和检索。 |
| Model Provider | 外部服务 | DashScope/Ollama | LLM 和 Embedding。 |

## 3. FastAPI 组件视图

```text
backend/main.py
  ↓ include_router
router/chat.py
  ↓ Depends
router/chat_service.py
  ├── agent/agent.py
  ├── rag/rag_service.py
  ├── rag/vector_store.py
  ├── rag/reorder_service.py
  └── services/database_session_manager.py
```

| 组件 | 角色 | 说明 |
| --- | --- | --- |
| `main.py` | 应用组装 | 注册中间件、路由、异常处理和启动资源。 |
| `chat.py` | API 入口 | 定义接口和依赖注入。 |
| `ChatService` | 应用服务 | 编排 Agent、RAG、会话、向量库。 |
| `RagService` | RAG 用例 | 完成 HyDE、检索、重排、摘要。 |
| `VectorStoreService` | 检索基础设施 | 负责 Chroma、BM25、文档入库。 |
| `ReorderService` | 重排能力 | 使用 CrossEncoder 对检索结果排序。 |
| `AgentFactory` | Agent 工厂 | 构造模型、工具、Prompt、AgentExecutor。 |
| `DatabaseSessionManager` | 会话仓储服务 | 读写 MySQL 中的聊天历史。 |

## 4. Django 组件视图

```text
DjangoUserService/urls.py
  ├── apps/user/urls.py
  │   ├── LoginView
  │   ├── RegisterView
  │   ├── UserDetailView
  │   ├── UserUpdateView
  │   └── UserLogOutView
  └── apps/file/urls.py
      └── UploadAPIView
```

| 组件 | 角色 | 说明 |
| --- | --- | --- |
| `settings.py` | 配置中心 | MySQL、Redis cache、DRF、CORS、media。 |
| `User` | 领域模型 | 自定义用户模型，主键为 ShortUUID。 |
| `JWTAuthentication` | 认证适配器 | 从 Bearer Token 解析用户。 |
| `JWTTokenGenerator` | Token 服务 | 生成、刷新、拉黑 JWT。 |
| `serializers.py` | 输入校验 | 登录、注册、重置密码、更新用户。 |
| `views.py` | API 层 | 用户接口实现。 |
| `UploadAPIView` | 文件接口 | 上传头像并更新用户 avatar。 |

## 5. 前端组件视图

```text
front/src/main.js
  ↓
App.vue
  ↓
router/index.js
  ├── AIChat.vue
  ├── Sessions.vue
  ├── Login.vue
  ├── Register.vue
  ├── My.vue
  ├── Profile.vue
  └── Settings.vue

Pinia store
  ├── user.js
  ├── session.js
  ├── theme.js
  └── language.js
```

| 组件 | 作用 |
| --- | --- |
| `AIChat.vue` | 发送聊天请求，消费 SSE，Markdown 渲染 AI 回复。 |
| `Sessions.vue` | 展示和管理会话。 |
| `user.js` | 登录、注册、登出、用户资料和 JWT。 |
| `session.js` | 会话列表、会话详情、删除会话。 |
| `api.js` | API endpoint 配置。 |

## 6. 数据视图

### 6.1 聊天数据

`backend/app/models/chat_history.py`

```text
chat_sessions
  id
  user_id
  title
  metadata
  created_at
  updated_at

chat_messages
  id
  session_id
  role
  content
  metadata
  created_at
```

关系：

```text
ChatSession 1 ── * ChatMessage
```

### 6.2 用户数据

`DjangoUserService/apps/user/models.py`

```text
user_service
  uuid
  username
  email
  telephone
  password
  is_active
  status
  gender
  bio
  date_joined
  last_login
  avatar
```

### 6.3 向量数据

`backend/app/rag/vector_store.py`

Chroma 存储：

- 文档切片内容。
- Embedding 向量。
- metadata，例如 `user_id`。

配置来源：

```text
backend/app/config/chroma.yaml
```

关键配置：

| 配置 | 含义 |
| --- | --- |
| `collection_name` | Chroma collection 名称。 |
| `persist_directory` | 向量库持久化目录。 |
| `k` | 检索数量。 |
| `chunk_size` | 文本切分大小。 |
| `chunk_overlap` | 文本重叠大小。 |
| `allow_knowledge_file_types` | 允许入库的知识文件类型。 |

## 7. 部署视图

开发环境可以理解为：

```text
localhost
├── front dev server
├── FastAPI backend :8000
├── DjangoUserService :8001
├── MySQL :3306
├── Redis :6379
├── Ollama :11434
└── Chroma local files
```

生产环境可能需要：

```text
Nginx / API Gateway
  ├── /            -> front 静态资源
  ├── /api/*       -> FastAPI backend
  ├── /user/*      -> DjangoUserService
  ├── /file/*      -> DjangoUserService
  └── /media/*     -> Django media
```

需要统一考虑：

- CORS 或同源代理。
- JWT 密钥同步。
- Redis DB 和 key 前缀约定。
- Chroma 持久化目录挂载。
- Reranker 模型文件挂载。
- LLM API key 配置。

## 8. 横切关注点

| 关注点 | 当前实现 |
| --- | --- |
| 鉴权 | Django 生成 JWT，FastAPI 解码 JWT。 |
| 限流 | FastAPI `RateLimitMiddleware` 和 `rate_limit` 依赖。 |
| 缓存 | Redis 缓存用户信息。 |
| 异常处理 | FastAPI 注册统一异常处理器。 |
| 日志 | `backend/app/core/logger_handler.py` 和 Django 日志。 |
| 配置 | `.env` + YAML。 |
| Prompt 管理 | `backend/app/prompt/*.txt`。 |
| 追踪 | `@traceable` 接入 LangSmith。 |

