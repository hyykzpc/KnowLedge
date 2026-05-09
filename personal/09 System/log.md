---
title: 系统演化日志
created: 2026-04-29
updated: 2026-05-05
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

## [2026-05-05] update | Agent 开发面试 50 题标准答案 — 重写为口述版
- Updated: [[01 Notes/面试 - Agent开发 50题标准答案]] — 全部 50 题重写为面试者口述版，每题含「直接回答→展开要点→加分亮点」三层结构，便于背诵和临场发挥
- 原参考来源链接已保留，MOC 已同步

## [2026-05-05] ingest | Agent 开发面试 50 题标准答案（初版）
- Source: 网络检索汇总（DataWhale hello-agents、Awesome-LLM-Interview、Uplatz 50题、2025 大厂面经）
- Created: [[01 Notes/面试 - Agent开发 50题标准答案]] — 50 题标准答案，9 大模块
- Created: [[05 MOC/Agent 面试 MOC]] — Agent 面试话题索引
- Updated: [[09 System/index.md]] — 新增面试笔记条目与 MOC 条目

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

## [2026-05-05] ingest | 蔚来日常实习 Agent 开发面经
- Source: personal/面经/蔚来日常_agent开发_1.txt（用户提供）
- Moved: → `01 Sources/面试 - 蔚来日常 agent开发 题目.md`（原始题目）
- Created: [[01 Notes/面试 - 蔚来日常 Agent开发 Q&A]] — 19 题完整解答，覆盖 Agent/RAG/上下文管理/并发/数据库一致性/Java注解vs Python装饰器/C++智能指针与coredump/科研项目框架
- Updated: [[09 System/index.md]] — 新增面试笔记分类与原始资料条目
- Removed: personal/面经/（空目录已清理）


## [2026-05-09] ingest | 深度强化学习 第14章学习笔记
- Source: 99 Reference/nndl-book.pdf (第14章)
- Created: [[02 Learning/深度强化学习 - 第14章学习笔记]]
- Updated: [[09 System/index.md]] — 新增深度强化学习笔记条目
- Updated: [[05 MOC/深度学习 MOC.md]] — 新增笔记链接
- Notes: 覆盖 MDP、值函数与 Bellman 方程、DQN、SARSA 与 Q-Learning、Policy Gradient 完整推导（含对数导数技巧与基线降方差）、Actor-Critic 算法。680+ 行，含完整数学推导与详细注解。
## [2026-04-29] decision | 目录调整说明
- 创建 `09 System/` 存放 index.md 和 log.md
- 创建 `01 Sources/` 存放原始资料
- 创建 `03 Concepts/` 存放概念页
- `99 Reference/` 保留作为参考资料存放位置
