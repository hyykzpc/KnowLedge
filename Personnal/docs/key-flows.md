# 关键流程

本文记录当前项目最重要的端到端流程。

## 1. 用户登录流程

```text
Login.vue / user.js
  ↓ POST /user/login/
DjangoUserService apps/user/views.py: LoginView
  ↓ LoginSerializer
User 模型校验账号密码
  ↓
JWTTokenGenerator.generate_token()
  ↓
返回 user + token
  ↓
front/src/store/user.js 保存 token 到 localStorage
```

关键文件：

- `front/src/store/user.js`
- `DjangoUserService/apps/user/views.py`
- `DjangoUserService/apps/user/serializers.py`
- `DjangoUserService/apps/user/authentications.py`

关键点：

- Token 是 Django 签发的 JWT。
- Token payload 包含 `user_id`、`username`、`email`、`exp`、`iat`、`jti`。
- 前端后续请求在 `Authorization: Bearer <token>` 中携带 token。

## 2. FastAPI 校验用户身份流程

```text
前端请求 /api/*
  ↓ Authorization: Bearer <token>
backend/app/utils/auth_utils.py
  ↓ decode_django_jwt()
使用 SECRET_KEY + ALGORITHM 解码
  ↓
检查 jti 是否在 Redis blacklist 中
  ↓
读取 payload.user_id
```

关键文件：

- `backend/app/utils/auth_utils.py`
- `backend/app/db/redis_config.py`
- `DjangoUserService/apps/user/authentications.py`

注意：

- Django 和 FastAPI 必须共享 JWT 密钥和算法。
- Django 拉黑 token 后，FastAPI 通过 Redis key 发现该 token 已失效。

## 3. AI 聊天流式问答流程

```text
front/src/views/AIChat.vue
  ↓ fetch('/api/agent/query/stream')
backend/app/router/chat.py
  ↓ query_stream()
get_current_user_id()
  ↓
get_agent_stream_response(query, session_id, user_id)
  ↓
DatabaseSessionManager.get_history()
  ↓
AgentExecutor.astream()
  ↓
SSE:
  data: {"type": "response", "content": "..."}
  data: {"type": "done", "session_id": "..."}
  ↓
DatabaseSessionManager.add_message()
```

关键文件：

- `front/src/views/AIChat.vue`
- `backend/app/router/chat.py`
- `backend/app/agent/agent.py`
- `backend/app/services/database_session_manager.py`

关键点：

- 前端使用 `ReadableStream` 逐块读取响应。
- 后端使用 `StreamingResponse` 返回 `text/event-stream`。
- 如果前端没有传 `session_id`，后端会创建新的 UUID。
- 对话完成后，后端把用户问题和 AI 回复保存到 MySQL。

## 4. Agent 工具调用流程

```text
AgentFactory.create_agent_executor()
  ↓
create_tool_calling_agent(chat_model, tools, prompt)
  ↓
AgentExecutor.astream()
  ↓
模型判断是否调用工具
  ├── rag_summary_tools
  ├── reorder_documents_tools
  ├── get_user_info_tools
  ├── get_weather_tools
  └── what_time_is_now
  ↓
工具返回 observation
  ↓
模型继续生成最终回答
```

关键文件：

- `backend/app/agent/agent.py`
- `backend/app/agent/agent_tools.py`
- `backend/app/agent/agent_middleware.py`
- `backend/app/prompt/main_prompt.txt`

关键点：

- Agent 模型由 `LLM_TYPE` 决定使用 Ollama 或阿里云。
- 工具函数把 RAG、重排、用户信息等能力暴露给 Agent。
- `intermediate_steps` 可用于观察工具调用过程。

## 5. RAG 查询流程

```text
POST /api/rag/query
  ↓
backend/app/router/chat.py
  ↓
ChatService.handle_rag_query()
  ↓
RagService.rag_summary()
  ↓
get_documents_and_summary()
  ↓
generate_hypothetical_document()
  ↓
retrieve_document()
  ↓
reorder_documents()
  ↓
PromptTemplate | chat_model | StrOutputParser
```

关键文件：

- `backend/app/router/chat.py`
- `backend/app/router/chat_service.py`
- `backend/app/rag/rag_service.py`
- `backend/app/rag/vector_store.py`
- `backend/app/rag/reorder_service.py`

关键点：

- 使用 HyDE 先生成假想文档，再用假想文档检索。
- 检索器是 Chroma 向量检索和 BM25 的融合。
- 重排器使用 CrossEncoder。
- 最多取前 3 个文档做摘要，再合并摘要生成最终回答。

## 6. 知识文件入库流程

```text
POST /api/vector/add/single
  ↓
chat.py add_vector_single()
  ↓
get_current_user_id()
  ↓
ChatService.handle_add_vector_single()
  ↓
校验文件大小、MIME、扩展名
  ↓
VectorStoreService.get_document(files=[file], user_id=user_id)
  ↓
写临时文件
  ↓
计算 MD5，检查是否重复
  ↓
file_handler 加载文档
  ↓
AsyncTextSplitter 切分
  ↓
写入 user_id metadata
  ↓
Chroma.add_documents()
  ↓
保存 MD5
```

关键文件：

- `backend/app/router/chat.py`
- `backend/app/router/chat_service.py`
- `backend/app/rag/vector_store.py`
- `backend/app/utils/file_handler.py`
- `backend/app/rag/text_spliter.py`
- `backend/app/config/chroma.yaml`

支持的文件类型：

- `.pdf`
- `.txt`
- `.md`
- `.pptx`
- `.docx`

注意：

- 当前上传入库是同步流程，大文件或模型慢时会影响请求耗时。
- DOCX 加载目前需要重点验证，代码中 `word_loader` 使用了 `TextLoader`。

## 7. 会话查询流程

```text
front/src/store/session.js
  ↓ GET /api/sessions/{user_id}
backend/app/router/chat.py
  ↓ get_user_sessions()
get_current_user_id()
  ↓
ChatService.handle_get_user_sessions()
  ↓
校验 user_id == current_user_id
  ↓
DatabaseSessionManager.get_user_sessions()
  ↓
MySQL chat_sessions
```

关键点：

- 用户只能查询自己的会话。
- 会话标题在 `add_message` 时从用户第一条消息截取前 30 个字符生成。

## 8. 用户资料缓存流程

```text
FastAPI 需要用户详情
  ↓
get_user_info_from_redis(user_id, credentials)
  ↓
Redis key :1:user:{user_id}
  ↓ 未命中
GET DjangoUserService /user/detail/
  ↓
写入 Redis，TTL 3600 秒
```

Django 侧：

```text
UserDetailView
  ↓
get_user_info()
  ↓
cache_user_info()
  ↓
Redis cache user:{user_id}
```

注意：

- 两边 key 前缀存在差异，FastAPI 查询 `:1:user:{user_id}`，Django cache 使用 `user:{user_id}`，具体取决于 Django Redis cache 的 key 前缀行为。
- 用户资料更新、密码更新、头像上传会调用 `clear_user_cache`。

## 9. Token 登出和拉黑流程

```text
front user.js logout()
  ↓ POST /user/logout/
Django UserLogOutView
  ↓
JWTTokenGenerator.blacklist_token(token)
  ↓
解析 token 获取 jti 和 exp
  ↓
Redis cache set blacklist:{jti}
```

FastAPI 后续校验：

```text
get_current_user_id()
  ↓
payload.jti
  ↓
Redis keys("*blacklist:{jti}")
  ↓
如果命中，返回 401
```

## 10. 限流流程

全局限流：

```text
main.py
  ↓
app.add_middleware(RateLimitMiddleware, limit=100, window=60)
```

接口级限流：

```text
@chat_router.post(...)
async def endpoint(..., _: None = Depends(rate_limit(limit=10, window=60)))
```

Redis key：

- `rate_limit:global:{client_ip}`
- `rate_limit:aichat:{client_ip}`

## 11. 头像上传流程

```text
Profile.vue
  ↓ POST /file/upload/
Django UploadAPIView
  ↓ JWTAuthentication
  ↓ ImgSerializer
  ↓ 写入 MEDIA_ROOT/img/
  ↓ 更新 user.avatar
  ↓ clear_user_cache(user.uuid)
  ↓ 返回 media URL
```

关键文件：

- `front/src/views/Profile.vue`
- `DjangoUserService/apps/file/views.py`
- `DjangoUserService/apps/file/serializers.py`
- `DjangoUserService/apps/user/authentications.py`

