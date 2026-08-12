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
- Updated: [[面试 - Agent开发 50题标准答案]] — 全部 50 题重写为面试者口述版，每题含「直接回答→展开要点→加分亮点」三层结构，便于背诵和临场发挥
- 原参考来源链接已保留，MOC 已同步

## [2026-05-05] ingest | Agent 开发面试 50 题标准答案（初版）
- Source: 网络检索汇总（DataWhale hello-agents、Awesome-LLM-Interview、Uplatz 50题、2025 大厂面经）
- Created: [[面试 - Agent开发 50题标准答案]] — 50 题标准答案，9 大模块
- Created: [[Agent 面试 MOC]] — Agent 面试话题索引
- Updated: [[Personnal/09 System/index]] — 新增面试笔记条目与 MOC 条目

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
- Source: [[transformer.py]]
- Created: [[Transformer-代码实现解析-学习笔记]]
- Updated: [[深度学习 MOC]] — 新增笔记链接，标记 Transformer 完整笔记待办为已完成
- Updated: [[Personnal/09 System/index]] — 新增条目
- Notes: 基于 transformer.py 代码生成，包含 SelfAttention（缩放点积注意力）和 MultiHeadAttention 的代码实现与详细数学原理注解，覆盖 Q/K/V 机制、缩放因子、多头拆分、残差连接与 LayerNorm。

## [2026-05-05] ingest | 蔚来日常实习 Agent 开发面经
- Source: personal/面经/蔚来日常_agent开发_1.txt（用户提供）
- Moved: → `01 Sources/面试 - 蔚来日常 agent开发 题目.md`（原始题目）
- Created: [[面试 - 蔚来日常 Agent开发 Q&A]] — 19 题完整解答，覆盖 Agent/RAG/上下文管理/并发/数据库一致性/Java注解vs Python装饰器/C++智能指针与coredump/科研项目框架
- Updated: [[Personnal/09 System/index]] — 新增面试笔记分类与原始资料条目
- Removed: personal/面经/（空目录已清理）


## [2026-05-09] ingest | 深度强化学习 第14章学习笔记
- Source: 99 Reference/nndl-book.pdf (第14章)
- Created: [[02 Learning/深度强化学习 - 第14章学习笔记]]
- Updated: [[Personnal/09 System/index]] — 新增深度强化学习笔记条目
- Updated: [[深度学习 MOC]] — 新增笔记链接
- Notes: 覆盖 MDP、值函数与 Bellman 方程、DQN、SARSA 与 Q-Learning、Policy Gradient 完整推导（含对数导数技巧与基线降方差）、Actor-Critic 算法。680+ 行，含完整数学推导与详细注解。
## [2026-05-14] ingest | 期末复习专题 - 第3章 无监督学习
- Source: 神经网络复习/3无监督学习.pdf + 99 Reference/nndl-book.pdf (第9章)
- Created: [[02 Learning/期末复习 - 第3章 无监督学习]] — PCA 完整推导、稀疏编码、自编码器家族（AE/稀疏AE/堆叠AE/降噪AE/VAE）、聚类、密度估计
- Updated: [[Personnal/09 System/index]] — 第3章条目
- Notes: 对标作业2考察重点（PCA/稀疏编码/自编码器为★★★），密度估计/自监督/半监督为PPT覆盖内容

## [2026-05-14] ingest | 期末复习专题 - 第2章 机器学习概述
- Source: 神经网络复习/2机器学习概述.pdf + 99 Reference/nndl-book.pdf
- Created: [[02 Learning/期末复习 - 第2章 机器学习概述]] — ERM、梯度下降推导、MLE/MAP 对比、偏差-方差分解、正则化 L1 vs L2
- Updated: [[Personnal/09 System/index]] — 第2章状态更新
- Notes: 按应试导向编写，标注★★★优先级（ERM/MLE/过拟合/偏差-方差为核心考点）

## [2026-05-14] ingest | 期末复习专题 - 第1章 绪论
- Source: 神经网络复习/1绪论+-+上传版.pdf + 99 Reference/nndl-book.pdf
- Created: [[02 Learning/期末复习 - 第1章 绪论]] — AI 发展史、三大流派、表示学习、M-P 模型与感知机、通用近似定理、BP 推导、现代深度学习里程碑
- Updated: [[Personnal/09 System/index]] — 新增期末复习专题分类与条目
- Notes: PPT 内容与教材结合，包含数学推导（M-P 模型、BP 链式法则），按课程考试要求编排

## [2026-05-14] ingest | 期末复习专题 - 第4章 注意力机制与外部记忆
- Source: 神经网络复习/4注意力机制与外部记忆.pdf + 99 Reference/nndl-book.pdf (第8章)
- Created: [[02 Learning/期末复习 - 第4章 注意力机制与外部记忆]] — QKV 自注意力推导、多头注意力、Transformer 编码器/解码器架构、位置编码、外部记忆（NTM、Hopfield）、指针网络
- Updated: [[Personnal/09 System/index]] — 第4章条目
- Updated: [[深度学习 MOC]] — 新增期末复习条目
- Notes: 对标作业3（注意力6题）和作业4-2(1)（Transformer 结构），缩放点积推导 $\sqrt{d_k}$ 因子为必考推导

## [2026-05-14] ingest | 期末复习专题 - 第11章 概率图模型
- Source: 神经网络复习/11概率图模型.pdf + 99 Reference/nndl-book.pdf (第11章)
- Created: [[02 Learning/期末复习 - 第11章 概率图模型]] — PGM 表示（有向/无向图）、联合概率分解、条件独立性、EM 算法完整推导、GMM 的 EM、推断方法
- Updated: [[Personnal/09 System/index]] — 第11章条目
- Updated: [[深度学习 MOC]] — 新增条目
- Notes: EM 算法推导（ELBO 构造、E步/M步、收敛性证明）为★★★核心考点，GMM 计算题需会手算

## [2026-05-14] ingest | 期末复习专题 - 第13章 深度生成模型
- Source: 神经网络复习/13深度生成模型_yxy.pdf + 99 Reference/nndl-book.pdf (第13章)
- Created: [[02 Learning/期末复习 - 第13章 深度生成模型]] — 判别 vs 生成 AI、VAE（ELBO + 重参数化）、GAN（Minimax + 最优判别器推导）、W-GAN、扩散模型
- Updated: [[Personnal/09 System/index]] — 第13章条目
- Updated: [[深度学习 MOC]] — 新增条目
- Notes: 对标作业4全部内容，VAE 损失函数和 GAN Minimax 目标函数为★★★必考推导，10种以上判别模型列举为常考简答

## [2026-05-14] update | 期末复习全部 7 章完成 + 方法论沉淀
- Created: 期末复习 7 篇笔记（第1/2/3/4/11/13/14章）
- Created: [[期末复习笔记生成法]] — 复习笔记的系统生成方法记录
- Updated: [[Personnal/09 System/index]] — 所有期末复习条目 + 方法论条目
- Updated: [[深度学习 MOC]] — 7 条期末复习条目
- Notes: 复习专题完结，按教材章节脉络覆盖全部课程内容

## [2026-05-14] ingest | 期末复习专题 - 第14章 深度强化学习
- Source: 99 Reference/nndl-book.pdf (第14章)
- Created: [[02 Learning/期末复习 - 第14章 深度强化学习]] — MDP、Bellman 方程推导、DQN（目标网络+经验回放）、策略梯度定理完整推导（对数导数技巧+基线降方差）、Actor-Critic、SARSA vs Q-Learning 对比
- Updated: [[Personnal/09 System/index]] — 第14章条目
- Updated: [[深度学习 MOC]] — 新增条目
- Notes: 教材第14章内容，无直接对应 PPT 和作业，但 Bellman 方程推导和策略梯度定理为★★★核心考点

## [2026-04-29] decision | 目录调整说明
- 创建 `09 System/` 存放 index.md 和 log.md
- 创建 `01 Sources/` 存放原始资料
- 创建 `03 Concepts/` 存放概念页
- `99 Reference/` 保留作为参考资料存放位置
