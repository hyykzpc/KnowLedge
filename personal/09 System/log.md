---
title: 系统演化日志
created: 2026-04-29
updated: 2026-05-02
type: log
tags:
  - #系统
  - #log
status: evergreen
---

# 系统演化日志

## 日志格式
每条记录：`[YYYY-MM-DD] <操作类型> | <标题>`

操作类型：`ingest` / `query` / `update` / `lint` / `decision`

---

## [2026-04-29] lint | 首次知识库健康检查
- 发现：无 `index.md`、无 `log.md`、无 `09 System/`
- 发现：MOC 缺失（深度学习主题已有 4 篇笔记）
- 发现：2 篇笔记缺少 frontmatter
- 发现：`notebook_13章_demo.md` 内容过少（6行），归档至 `99 Archive/`
- 发现：`llm-wiki.md` 位于 `99 Archive/`，应为 Sources 资料而非归档
- 创建：`09 System/index.md` 全局索引
- 创建：`09 System/log.md` 演化日志
- 创建：`05 MOC/深度学习 MOC.md`
- 更新：`02 Learning/注意力机制举例_第八章学习笔记追加.md` — 补充 frontmatter
- 更新：`99 Archive/notebook_13章_demo.md` — 移至 Archive
- 移动：`99 Archive/llm-wiki.md` → `01 Sources/llm-wiki.md`
- 创建：`03 Concepts/LLM Wiki 知识库构建法.md`

## [2026-04-29] ingest | 生成模型 第13章学习笔记
- Source: 99 Reference/nndl-book.pdf (第13章)
- Created: [[02 Learning/生成模型 - 第13章学习笔记]]
- Notes: VAE 完整推导（ELBO、重参数化、CVAE）、GAN（博弈论视角、JS 散度、模式崩溃）、W-GAN（Wasserstein 距离、Lipschitz 约束）。800+ 行，含对比表和关键数学推导。

## [2026-05-02] ingest | Transformer 多头注意力机制学习笔记
- Source: [[code/transformer.py]]
- Created: [[Transformer-代码实现解析-学习笔记]]
- Updated: [[05 MOC/深度学习 MOC.md]] — 新增笔记链接，标记 Transformer 完整笔记待办为已完成
- Updated: [[09 System/index.md]] — 新增条目
- Notes: 基于 transformer.py 代码生成，包含 SelfAttention（缩放点积注意力）和 MultiHeadAttention 的代码实现与详细数学原理注解，覆盖 Q/K/V 机制、缩放因子、多头拆分、残差连接与 LayerNorm。

## [2026-04-29] decision | 目录调整说明
- 创建 `09 System/` 存放 index.md 和 log.md
- 创建 `01 Sources/` 存放原始资料
- 创建 `03 Concepts/` 存放概念页
- `99 Reference/` 保留作为参考资料存放位置
