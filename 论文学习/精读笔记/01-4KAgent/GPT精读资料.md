这是 **4KAgent 最像 Agent 的地方**。前面我们讲的：

- Perception Agent：判断“图像有什么问题”
- Planning：决定“先做什么后做什么”
- Restoration Agent：调用工具恢复

但如果没有 **Reflection + Rollback**，它其实只是一个：

> 自动调用多个模型的 pipeline。

真正让它变成 Agent 的，是：

\[ \boxed{ \text{执行} \rightarrow \text{反思} \rightarrow \text{失败恢复} } \]

也就是：

\[ \boxed{ Execution \rightarrow Reflection \rightarrow Rollback } \]

下面重点讲 Reflection，因为这是核心。

---

# 1. 为什么需要 Reflection？

假设 Planner 给出：

\[ P_I= [ Denoise, Deblur, SR ] \]

执行：

```
Input
 |
 ↓
Denoising Tool
 |
 ↓
Image 1
```

传统 pipeline 会直接：

```
Image1
 ↓
Deblur
 ↓
SR
```

但是问题：

**你怎么知道 Denoising 这一步真的有效？**

可能出现：

### 情况1：成功

输入：

```
严重噪声
```

输出：

```
干净图像
```

质量提升：

\[ Q(I_1)>Q(I_0) \]

继续。

---

### 情况2：失败

例如：

原图：

```
低光人像 + 轻微噪声
```

调用强 denoiser：

```
过度平滑
```

结果：

```
脸部纹理消失
```

虽然噪声下降，但是：

- 细节没了
- 人脸质量下降
- 语义损失

如果继续：

```
Deblur
 ↓
SR
```

后面只会放大错误。

所以 Agent 需要一个机制：

> 每完成一步，都检查一下“这一步有没有把事情搞坏”。

这个就是 Reflection。

---

# 2. Reflection 在 4KAgent 里面是什么？

很多人容易误解：

Reflection ≠ LLM 思考。

这里不是：

> ChatGPT 看图片然后说“我觉得不错”。

4KAgent 的 Reflection 是一个：

\[ \boxed{ \text{自动质量评估模块} } \]

它做的事情：

输入：

\[ \{I_1,I_2,...,I_N\} \]

多个 restoration expert 输出。

然后：

计算每个结果的质量分数：

\[ Q_s(I_i) \]

最后：

选择最高分。

---

也就是说：

Reflection 实际包含两层：

## 第一层：Expert 结果评价

例如一个 SR task：

输入：

\[ I_k \]

进入多个 SR 模型：

\[ \begin{cases} T_1(I_k)\\ T_2(I_k)\\ T_3(I_k) \end{cases} \]

得到：

```
Result A
Result B
Result C
```

Reflection：

```
A → score
B → score
C → score
```

选择：

\[ I_{best} = \arg\max_iQ_s(T_i(I_k)) \]

这就是 Q-MoE。

---

## 第二层：判断当前 Task 是否值得继续

选择最佳结果以后：

还要判断：

> 这个结果够不够好？

即：

\[ Q_s(I_k)>\eta? \]

如果：

\[ Q_s(I_k)>\eta \]

认为：

成功。

继续：

\[ I_k\rightarrow T_{k+1} \]

如果：

\[ Q_s(I_k)\leq\eta \]

认为：

失败。

触发：

\[ Rollback \]

---

所以：

Reflection 不只是：

> 选哪个模型。

还负责：

> 当前恢复步骤是否成功。

---

# 3. Reflection 的评分公式

论文定义：

\[ Q_s = H + \frac{Q_{nr}}4 \]

其中：

---

## (1) H：HPSv2

\[ H \]

表示：

Human Preference Score。

它回答：

> 人类更喜欢哪张图？

比如两个 SR 输出：

A:

```
纹理真实，但稍微模糊
```

B:

```
超级锐利，但出现假纹理
```

PSNR 可能喜欢 A。

但是人类可能喜欢 B。

所以引入 HPSv2。

---

## (2) \(Q_{nr}\)：无参考质量

论文组合：

\[ Q_{nr} = w_{NIQE} (1-\frac{Q_{NIQE}}{10}) + \sum_jw_jQ_j \]

其中：

包括：

- NIQE
- MUSIQ
- MANIQA
- CLIPIQA

作用：

衡量：

- 自然程度
- 清晰度
- 感知质量

---

所以：

Reflection 本质：

\[ \boxed{ \text{Human preference} + \text{Image quality} } \]

---

# 4. 一个具体例子理解 Reflection

假设当前任务：

\[ SR\times4 \]

输入：

\[ 512\times512 \]

有三个 SR expert：

---

### Expert 1：HAT

输出：

```
细节真实
```

评分：

\[ Q_s=0.78 \]

---

### Expert 2：DiffBIR

输出：

```
纹理漂亮
但是幻觉严重
```

评分：

\[ Q_s=0.85 \]

---

### Expert 3：SwinIR

输出：

```
比较平
```

评分：

\[ Q_s=0.72 \]

Reflection：

选择：

\[ I_k=DiffBIR \]

因为：

\[ 0.85=\max(0.78,0.85,0.72) \]

---

然后检查：

\[ 0.85>\eta \]

假设：

\[ \eta=0.5 \]

所以：

成功。

进入下一步。

---

# 5. 那 Rollback 是怎么触发的？

假设：

另一个情况：

三个模型结果：

```
Tool1: 0.35
Tool2: 0.41
Tool3: 0.38
```

最佳：

\[ Q_s=0.41 \]

但是：

\[ 0.41<0.5 \]

于是：

Reflection 输出：

```
Current restoration step failed
```

然后：

Rollback。

---

# 6. Rollback 回退什么？

不是回退像素。

很多人会误解：

> 把图片恢复回来？

不是。

它回退的是：

\[ \boxed{ \text{Task Planning} } \]

即：

撤销：

```
当前恢复决策
```

重新规划。

---

例如：

原计划：

```
Noise
 ↓
Deblur
 ↓
SR
```

执行 Noise Removal：

失败。

Rollback：

回到：

```
Noise
 ↓
Deblur
 ↓
SR
```

重新问 Planner：

> 刚才 Noise Removal 失败，现在怎么办？

Planner 收到：

- 原始 degradation
- 当前状态
- failure message

重新生成：

可能：

```
Deblur
 ↓
Noise Removal
 ↓
SR
```

---

# 7. Rollback 的输入是什么？

论文公式：

\[ P_I^{adj} = M_P(D_I,A_I^R,E,S_I) \]

几个变量：

---

### \(D_I\)

原始退化：

例如：

```
Noise
Blur
LR
```

---

### \(A_I^R\)

剩余任务：

例如：

已经完成：

```
Noise
```

剩余：

```
Deblur
SR
```

---

### \(E\)

经验：

告诉 Planner：

类似情况通常怎么处理。

---

### \(S_I\)

失败信息：

例如：

```
Denoising failed,
quality score below threshold
```

---

然后 Planner 重新生成：

\[ P_I^{adj} \]

---

# 8. 为什么这个机制很像 Agent？

传统 Restoration：

```
Input
 ↓
Network
 ↓
Output
```

没有反馈。

4KAgent：

```
Input

 ↓

Plan

 ↓

Action

 ↓

Observation

 ↓

Reflection

 ↓

New Decision
```

这其实对应 Agent 的经典循环：

\[ \boxed{ Observe \rightarrow Think \rightarrow Act \rightarrow Reflect } \]

---

# 9. 但是这里有一个值得批判的问题

Reflection 的最大问题：

它依赖：

\[ Q_s \]

这个评价函数。

也就是说：

Agent 认为：

\[ \text{高分} = \text{好结果} \]

但是：

这并不一定成立。

例如：

Diffusion SR：

可能生成：

```
漂亮的纹理
```

但是：

真实不存在。

IQA：

可能奖励它。

于是：

Reflection 可能选择：

\[ \text{视觉漂亮但错误} \]

这就是：

\[ \boxed{ Metric Hacking } \]

问题。

尤其医学影像、遥感：

这是严重风险。

---

# 10. 最后总结 Figure 2 里的 Reflection + Rollback

一句话：

> **Reflection 是一个“恢复结果质量审查器”，它利用 HPSv2 + IQA 对多个专家输出进行评分，选择最佳结果，并判断当前恢复步骤是否成功；如果质量低于阈值，Rollback 会把失败信息反馈给 Planner，让 Agent 重新安排恢复策略。**

流程：

\[ \boxed{ Input } \]

↓

\[ \boxed{ Multiple Restoration Experts } \]

↓

\[ \boxed{ Reflection: Q_s\text{评分} } \]

↓

两种情况：

成功：

\[ Q_s>\eta \]

↓

继续下一个 Task

失败：

\[ Q_s\leq\eta \]

↓

\[ \boxed{ Rollback } \]

↓

重新 Planning

所以 **Reflection 是 4KAgent 区别于普通 All-in-One Restoration 最大的地方之一**：它不是盲目执行一条固定 pipeline，而是在恢复过程中不断问：

> “刚才这一步真的把图变好吗？”

这也是后面理解它和 **AgenticIR、LLM-based Image Restoration** 区别的关键。