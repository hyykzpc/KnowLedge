# 0001 拆分用户服务和 AI 服务

## 状态

Accepted

## 背景

当前项目包含两类差异很大的能力：

- 用户注册、登录、JWT、资料、头像上传。
- AI 聊天、Agent、RAG、向量库、模型调用、重排。

用户服务更适合 Django/DRF 的认证、模型、序列化和管理能力。AI 服务更适合 FastAPI 的异步接口、流式响应和 LangChain 编排。

## 决策

将系统拆成两个后端服务：

- `DjangoUserService`：负责用户、JWT、头像、用户缓存。
- `backend`：负责 FastAPI AI/RAG/Agent/会话/向量库。

前端 `front` 同时调用两个服务，或通过开发代理/API 网关转发。

## 备选方案

- 单个 Django 服务承载所有能力。
- 单个 FastAPI 服务承载所有能力。
- 更细粒度微服务：User、Chat、RAG、Indexing、LLM Gateway。

## 后果

正面影响：

- 用户服务和 AI 服务职责清晰。
- AI 服务可以独立安装 PyTorch、LangChain、Chroma、Reranker 等重依赖。
- AI 服务可以独立扩展和部署。
- Django 继续发挥用户认证和 ORM 生态优势。

负面影响：

- 本地启动复杂度增加。
- JWT 密钥和算法需要在两个服务之间保持一致。
- Redis key 和缓存约定需要跨服务维护。
- 前端或网关需要处理两个后端的路由。

## 后续约束

- 必须统一 JWT 的 `SECRET_KEY` 和 `ALGORITHM`。
- 必须统一 Token 黑名单 Redis key 规则。
- 需要明确前端代理规则，例如 `/api/*` 走 FastAPI，`/user/*` 和 `/file/*` 走 Django。

