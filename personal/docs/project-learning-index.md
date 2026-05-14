# 项目学习文档索引

这组文档用于系统学习当前项目，按“项目入门 + 架构视图 + 关键流程 + 技术实现 + 决策记录”组织。

## 推荐阅读顺序

1. [项目入门](./project-onboarding.md)
   - 先理解项目是什么、有哪些运行单元、从哪些文件开始读。
2. [架构视图](./architecture-views.md)
   - 用系统上下文、容器、组件、部署和数据视图理解整体结构。
3. [关键流程](./key-flows.md)
   - 跟踪登录、聊天、RAG、上传、缓存、限流等端到端流程。
4. [技术实现](./technical-implementation.md)
   - 深入 FastAPI、Django、Vue、LangChain、Chroma、Redis、MySQL 的实现方式。
5. [架构设计学习笔记](./architecture-design-learning.md)
   - 对照常见架构模式学习当前项目的设计取舍。
6. [ADR 决策记录](./adr/)
   - 阅读关键技术选择背后的原因和后果。

## 文档地图

| 文档 | 作用 | 适合回答的问题 |
| --- | --- | --- |
| `project-onboarding.md` | 项目入门 | 这个项目是什么？怎么读？每个目录做什么？ |
| `architecture-views.md` | 架构视图 | 系统如何拆分？模块如何协作？数据放在哪里？ |
| `key-flows.md` | 关键流程 | 一次请求从前端到后端经历了什么？ |
| `technical-implementation.md` | 技术实现 | 每个技术点在代码里如何实现？ |
| `architecture-design-learning.md` | 架构学习 | 当前设计对应哪些常见架构模式？ |
| `adr/*.md` | 决策记录 | 为什么这么设计？还有哪些替代方案？ |

## 快速定位

| 想了解 | 先看 |
| --- | --- |
| 前端如何请求 AI 流式接口 | `key-flows.md` 的“AI 聊天流式问答流程” |
| RAG 是怎么实现的 | `technical-implementation.md` 的“RAG 实现” |
| 用户服务为什么独立 | `adr/0001-split-user-service-and-ai-service.md` |
| 为什么用 SSE | `adr/0004-use-sse-for-streaming-chat.md` |
| 文件上传后怎么进入向量库 | `key-flows.md` 的“知识文件入库流程” |
| 项目有哪些风险点 | `architecture-design-learning.md` 的“代码中值得关注的风险点” |

