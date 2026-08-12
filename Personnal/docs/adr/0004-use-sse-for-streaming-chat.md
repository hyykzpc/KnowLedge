# 0004 使用 SSE 实现 AI 流式聊天

## 状态

Accepted

## 背景

AI 回复可能需要较长时间。如果等待完整回答后再返回，用户体验较差。项目需要一种简单的服务端到客户端流式推送方式。

## 决策

使用 Server-Sent Events：

- FastAPI 使用 `StreamingResponse`。
- 响应类型为 `text/event-stream`。
- 后端不断发送 `data: {...}\n\n`。
- 前端使用 `fetch` 和 `ReadableStream` 逐块读取。

当前事件类型：

- `response`：AI 输出内容。
- `done`：本轮结束，可能携带 `session_id`。
- `error`：错误信息。
- `step`：Agent 中间步骤，当前前端主要忽略。

## 备选方案

- 普通 HTTP，等待完整响应。
- WebSocket。
- Long Polling。
- GraphQL Subscription。

## 后果

正面影响：

- 实现简单。
- 适合 AI 单向输出。
- 浏览器原生支持流式读取。
- 比 WebSocket 更轻量。

负面影响：

- SSE 主要是服务端到客户端单向通信。
- 浏览器和代理对长连接有超时限制。
- 需要网关正确配置缓冲和超时。
- 前端需要自己处理 chunk、换行、JSON 解析和断流。

## 后续约束

- 生产环境网关应关闭响应缓冲或为 SSE 单独配置。
- 后端事件格式要保持稳定。
- 前端要处理 `error` 和异常断开。
- 如果未来需要双向实时交互，再评估 WebSocket。

