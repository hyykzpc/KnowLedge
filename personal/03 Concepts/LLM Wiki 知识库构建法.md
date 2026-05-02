---
title: LLM Wiki 知识库构建法
created: 2026-04-29
updated: 2026-04-29
type: concept
tags:
  - #概念
  - #PKM
  - #知识管理
status: seed
source:
  - 01 Sources/llm-wiki.md
related:
  - [[深度学习 MOC]]
---

# LLM Wiki 知识库构建法

## 核心摘要
- LLM Wiki 是一种区别于传统 RAG 的知识库构建模式
- 核心思想：LLM 增量构建并维护一个持久化 Wiki，而非每次查询时临时检索拼接
- 三层架构：Raw Sources（原始资料）→ Wiki（结构化笔记）→ Schema（规则文件）
- 三种操作：Ingest（摄取）、Query（查询）、Lint（健康检查）
- 关键文件：`index.md`（内容目录）+ `log.md`（时间日志）

## 解决的问题
- 传统 RAG 每次查询从头构建上下文，无知识积累
- Wiki 维护成本高（人工跟进），LLM 解决了维护瓶颈
- 知识难以在多次交互间持续沉淀

## 核心架构

### 三层分离
1. **Raw Sources**：不可变的原始资料（文章、论文、书摘）
2. **Wiki**：LLM 维护的 Markdown 笔记（概念页、实体页、MOC 等）
3. **Schema（CLAUDE.md）**：定义结构、命名、流程的规则文件

### 三种操作
- **Ingest**：资料 → 提取关键信息 → 更新/创建笔记 → 建立双链 → 更新 index/log
- **Query**：查 index → 读相关页 → 结构化回答 → 可沉淀为新笔记
- **Lint**：检查孤立页、断裂链接、过期信息、重复、矛盾

## 关键结论
- 人的职责：策展资料、引导分析、提出好问题
- LLM 的职责：提取、连接、更新、维护、日志
- Wiki 随每次摄取和查询变得更强（复利效应）
- Obsidian 是 IDE，Wiki 是代码库，LLM 是维护工程师

## 知识连接
- [[注意力机制与注意力模型]] — 本知识库的先行概念页
- [[EM 算法推导笔记]] — 本知识库的先行概念页
- [[深度学习 MOC]] — 主题索引

## 来源
- `01 Sources/llm-wiki.md`
