# 0002 使用 LangChain 实现 RAG 和 Agent

## 状态

Accepted

## 背景

项目需要实现：

- 多模型供应商支持。
- Prompt 管理。
- 文档检索增强生成。
- Agent 工具调用。
- 流式输出。

这些能力如果全部手写，会产生较多样板代码和适配成本。

## 决策

使用 LangChain 作为 AI 编排框架：

- `PromptTemplate | chat_model | StrOutputParser` 构建 RAG 生成链。
- `create_tool_calling_agent` 和 `AgentExecutor` 构建 Agent。
- `Chroma`、`BM25Retriever`、`EnsembleRetriever` 构建检索。
- `BaseTool` 风格工具暴露 RAG、重排、用户信息等能力。

## 备选方案

- 直接调用模型 API，不使用框架。
- 使用 LlamaIndex。
- 使用 Haystack。
- 自研 Agent 和 RAG 编排层。

## 后果

正面影响：

- 上手快，组件丰富。
- 容易接入 Chroma、BM25、Ollama、DashScope。
- Agent 工具调用和流式处理有现成抽象。
- 后续切换模型和 Prompt 的成本较低。

负面影响：

- LangChain 版本变化较快，升级风险较高。
- 抽象层较多，调试时需要理解框架行为。
- 某些复杂流程可能被框架约束。

## 后续约束

- 关键链路要保留清晰日志。
- 重要流程要写成项目自己的 service，而不是把所有逻辑写进 LangChain 表达式。
- 升级 LangChain 前应先验证 RAG、Agent、流式输出和工具调用。

