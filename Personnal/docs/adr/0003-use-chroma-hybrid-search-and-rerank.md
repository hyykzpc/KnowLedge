# 0003 使用 Chroma、混合检索和重排

## 状态

Accepted

## 背景

RAG 系统需要从知识文件中召回与问题相关的片段。单独使用向量检索容易漏掉关键词精确匹配，单独使用 BM25 又难以理解语义相似。

项目还需要在召回后提高上下文质量，减少无关片段进入 LLM。

## 决策

当前项目采用：

- Chroma 作为向量库。
- Chroma similarity retriever 负责语义召回。
- BM25Retriever 负责关键词召回。
- EnsembleRetriever 融合向量检索和 BM25。
- CrossEncoder Reranker 对候选文档重排。

## 备选方案

- 只用 Chroma 向量检索。
- 使用 Milvus、Qdrant、Weaviate、pgvector。
- 使用 Elasticsearch 做 BM25 和向量混合检索。
- 不做 Rerank，直接将 TopK 结果交给 LLM。

## 后果

正面影响：

- Chroma 本地持久化简单，适合学习和原型。
- 混合检索能兼顾语义和关键词。
- Rerank 能提高最终上下文质量。
- 结构清晰，便于后续替换向量库。

负面影响：

- Reranker 增加模型加载和推理成本。
- BM25 当前从文件重新构建，数据规模变大后可能影响性能。
- Chroma 本地部署在多实例场景下需要额外设计。
- 多用户隔离依赖 metadata，后续需要加强。

## 后续约束

- 上传文档必须写入 `user_id` metadata。
- 删除用户文档时必须按 `user_id` filter。
- 大规模数据时需要评估 Milvus、Qdrant、Elasticsearch 或 pgvector。
- Reranker 模型路径要作为部署配置显式管理。

