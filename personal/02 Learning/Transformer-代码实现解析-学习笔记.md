---
title: Transformer - 多头注意力机制学习笔记
created: 2026-05-02
updated: 2026-05-02
type: note
tags:
  - AI
  - 深度学习
  - 编程
status: developing
source:
  - [[code/transformer.py]]
related:
  - [[神经网络与深度学习 - 第8章学习笔记]]
  - [[注意力机制举例_第八章学习笔记追加]]
  - [[深度学习 MOC]]
---

# Transformer — 多头注意力机制

## 核心摘要
- 注意力机制的本质：**通过 Query 与 Key 的相似度对 Value 做加权求和**，让模型关注输入中更相关的部分。
- 缩放点积注意力（Scaled Dot-Product Attention）是 Transformer 的基础计算单元，公式为 $\text{softmax}(QK^T / \sqrt{d_k}) V$。
- 多头注意力（Multi-Head Attention）将模型拆分为多个子空间，每个头在不同表示子空间中独立计算注意力，最后拼接融合。
- 残差连接 + LayerNorm 保证深层网络训练稳定性。

## 解决的问题
- RNN/LSTM 的串行计算无法高效捕获长距离依赖。
- 单头注意力的表达能力有限，不同关系模式（语法/语义/位置）在同一个空间相互干扰。

---

## 1. SelfAttention：缩放点积注意力

### 类的初始化

```python
class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
```

**对应操作：** 定义了一个仅包含 dropout 和 softmax 的注意力计算单元，它不包含可学习参数。Softmax 的 `dim=-1` 表示在最后一个维度（即每个 Query 对所有 Key 的分数）上做归一化，使每个位置对所有位置的注意力权重之和为 1。

---

```python
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
```

**对应操作：** 从 Q 的最后一维取出 $d_k$（每个注意力头内部的特征维度）。例如如果 Q 的形状为 `(batch_size, n_heads, seq_len, 64)`，则 `d_k = 64`。

---

```python
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
```

**对应操作：** 这是**缩放点积注意力**的核心计算步骤。

它对应完整公式中的前半部分：

$$
\frac{QK^T}{\sqrt{d_k}}
$$

逐层拆解：
- `K.transpose(-2, -1)`：将 K 的最后两个维度交换，形状从 `(batch, n_heads, seq_len, d_k)` 变为 `(batch, n_heads, d_k, seq_len)`。这是为了做矩阵乘法时，K 的 key 维度与 Q 的 query 维度对齐。
- `torch.matmul(Q, K.transpose(-2, -1))`：Q 和 K^T 做矩阵乘法，结果 `scores` 的形状为 `(batch, n_heads, seq_len, seq_len)`，其中第 `[i, j]` 个元素表示第 `i` 个 query 位置和第 `j` 个 key 位置的点积相似度。
- ` / math.sqrt(d_k)`：除以 $\sqrt{d_k}$ 做缩放。**为什么要缩放？** 当 $d_k$ 较大时，点积的方差达到 $d_k$（假设 Q/K 各分量独立同分布、均值为 0、方差为 1），导致 softmax 输入值过大，进入梯度极小的饱和区域。除以 $\sqrt{d_k}$ 将方差拉回 1，保持梯度稳定。

---

```python
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
```

**对应操作：** 对注意力分数施加掩码。

具体机制：
- `mask` 的形状通常为 `(batch, 1, 1, seq_len)` 或 `(batch, 1, seq_len, seq_len)`，`mask == 0` 的位置表示不允许被关注。
- 将这些位置的分数设为 `-inf`，后续 softmax 会把 `-inf` 变为 0，使这些位置不参与注意力计算。
- **Padding Mask**：忽略补齐出来的无效 token。
- **Causal Mask**（上三角掩码）：解码器中令位置 $i$ 只能看到 $j \le i$ 的位置，防止信息泄露。

---

```python
        attn = self.softmax(scores)
```

**对应操作：** 在最后一个维度上做 softmax 归一化，将每行的分数转为概率分布。

对应公式中的：

$$
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

计算后，`attn` 的形状仍为 `(batch, n_heads, seq_len, seq_len)`，但每一行（每个 query 对所有 key 的注意力权重）和为 1。

---

```python
        attn = self.dropout(attn)
```

**对应操作：** 对注意力权重做 dropout，随机将部分注意力权重置 0。

这相当于在训练时随机削弱某些 token 之间的依赖关系，防止模型过度依赖特定位置的连接，起到正则化作用。

---

```python
        out = torch.matmul(attn, V)
```

**对应操作：** 用注意力权重对 Value 做加权求和，得到最终的上下文融合表示。

对应公式的最后一步：

$$
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- `attn` 的形状为 `(batch, n_heads, seq_len, seq_len)`，V 的形状为 `(batch, n_heads, seq_len, d_k)`。
- 对每个 query 位置：用该位置对所有 key 位置的注意力权重，加权求和所有 key 位置对应的 value 向量。
- 输出 `out` 的形状为 `(batch, n_heads, seq_len, d_k)`，表示每个 token 融合上下文信息后的表示。

---

```python
        return out, attn
```

**返回：**
- `out`：注意力加权后的输出，送入后续网络层。
- `attn`：注意力权重矩阵，可用于可视化分析（如 BERT 的 [CLS] token 关注了哪些词）。

---

## 2. MultiHeadAttention：多头注意力

### 类的初始化

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
```

**对应操作：** 配置多头注意力的超参数。

- `d_model`：输入输出的隐藏维度（如 512、768）。
- `n_heads`：注意力头数，`d_model` 必须能被 `n_heads` 整除。
- `d_k = d_model // n_heads`：每个注意力头的特征维度，满足 `n_heads * d_k = d_model`。

---

```python
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
```

**对应操作：** 定义三个线性投影层，将输入分别映射到 Query、Key、Value 空间。

对应多头注意力的公式：

$$
QW_i^Q \quad KW_i^K \quad VW_i^V
$$

为什么需要这三个投影？
- Q 用来表示"当前位置想查找什么信息"。
- K 用来表示"每个位置能提供什么匹配特征"。
- V 用来表示"真正被加权汇总的信息内容"。
- 将三者映射到不同空间，使它们可以各自学习最合适的特征表示。

输入和输出维度都保持为 `d_model`，是为了后续可以方便地 reshape 为 `n_heads` 个 `d_k` 维子空间。

---

```python
        self.fc = nn.Linear(d_model, d_model)
```

**对应操作：** 输出线性层，用于融合所有注意力头的信息。

对应公式：

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

其中 $W^O$ 就是这里的 `fc`。多个头的结果拼接后仍是 `d_model` 维，`fc` 负责将拼接结果做进一步的交叉融合。

---

```python
        self.attention = SelfAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
```

**对应操作：** 实例化三个组件：

- `self.attention`：前一节的缩放点积注意力计算单元，所有头复用同一个计算逻辑。
- `self.dropout`：输出 dropout，进一步正则化。
- `self.norm`：LayerNorm，对每个 token 的 `d_model` 维特征做归一化（均值为 0、方差为 1）。与 BatchNorm 不同，它在特征维度上归一化，不依赖 batch 大小，天然适配变长序列。

---

### 前向传播

```python
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
```

**对应操作：** 获取 batch 大小。`q` 的完整形状为 `(batch_size, seq_len, d_model)`。

**三种注意力模式的输入差异：**

| 模式 | q | k | v | 应用场景 |
|---|---|---|---|---|
| 编码器自注意力 | 编码器输入 | 编码器输入 | 编码器输入 | 编码器每层 |
| 解码器自注意力 | 解码器输入 | 解码器输入（带因果掩码） | 解码器输入 | 解码器每层（带 mask） |
| 编码器-解码器注意力 | 解码器输入 | 编码器输出 | 编码器输出 | 解码器的交叉注意力层 |

---

```python
        Q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
```

**对应操作：** 对 Q、K、V 做线性映射并拆分为多头。

维度变化过程（以 Q 为例）：
```
(batch, seq_len, d_model)
    → self.W_q → (batch, seq_len, d_model)                              # 线性映射
    → view     → (batch, seq_len, n_heads, d_k)                         # 拆分为 n_heads 份
    → transpose(1,2) → (batch, n_heads, seq_len, d_k)                   # 头维度提前
```

为什么需要 `transpose(1, 2)`？
- `view` 后的排列是 `(batch, seq_len, n_heads, d_k)`，即相邻的 `d_k` 属于同一个位置的不同头。
- `transpose(1, 2)` 后变为 `(batch, n_heads, seq_len, d_k)`，即每个头内部的 `seq_len` 维度连续排列。
- 这样 `SelfAttention.forward()` 中的 `matmul` 可以按头独立并行计算：每个头在自己的子空间内做完整的 scaled dot-product attention。

---

```python
        out, attn = self.attention(Q, K, V, mask)
```

**对应操作：** 在所有头上并行执行缩放点积注意力。

每个头独立执行：
$$
\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

此时 `out` 的形状为 `(batch, n_heads, seq_len, d_k)`，`attn` 为 `(batch, n_heads, seq_len, seq_len)`。

**为什么需要多头？** 单头注意力将所有关系压缩到同一空间。多头使不同头在不同子空间中学习不同类型的模式（如一个头关注语法关系，另一个头关注语义相似度），最后融合互补。

---

```python
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
```

**对应操作：** 将多头结果拼接回 `d_model` 维。

维度变化过程：
```
(batch, n_heads, seq_len, d_k)
    → transpose(1,2) → (batch, seq_len, n_heads, d_k)   # 恢复原始顺序
    → contiguous()   → (batch, seq_len, n_heads, d_k)   # 解决 transpose 后的内存不连续问题
    → view          → (batch, seq_len, d_model)         # 拼接为完整维度
```

`transpose` 操作在 PyTorch 中不改变内存布局，只是重新解释 shape。直接 `view` 会因内存不连续而报错，所以需要先 `contiguous()` 再 `view`。

---

```python
        out = self.fc(out)
        out = self.dropout(out)
```

**对应操作：** 拼接后的多头输出通过输出线性层 $W^O$ 做跨头融合，再用 dropout 正则化。

---

```python
        return self.norm(out + q), attn
```

**对应操作：** 残差连接 + LayerNorm。

- `out + q`：残差连接，将原始输入 q 与注意力输出逐元素相加。梯度可以通过短路路径直接回传，缓解深层网络的梯度消失问题。
- `self.norm(...)`：LayerNorm 对每个 token 独立做归一化。

返回：
- 归一化后的输出，形状为 `(batch_size, seq_len, d_model)`。
- 注意力权重 `attn`，可用于分析模型行为（如每个 token 关注了哪些位置）。

---

## 3. 注意力机制本质总结

一句话概括注意力机制的全流程：

> **以 Q 为查询，与 K 计算相似度得到一个概率分布（注意力权重），再用这个分布对 V 做加权平均。**

数学表达一个公式：

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

多头注意力将此过程在 `h` 个不同的子空间并行执行，最后融合：

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

---

## 关键结论

- 注意力机制的核心只有 **三步操作：点积 → 缩放 + Softmax → 加权求和**。
- 多头注意力通过拆分子空间扩展表达能力，计算量不变（每个头维度等比缩小）。
- 残差连接 + LayerNorm 是 Transformer 深度的保障。
- Q、K、V 来自不同输入时实现交叉注意力，来自同一输入时实现自注意力。

## 知识连接

- [[神经网络与深度学习 - 第8章学习笔记]] — 注意力机制的数学基础
- [[注意力机制举例_第八章学习笔记追加]] — 注意力机制 BP 数值例子
- [[code/self_attenation.py]] — 单头注意力的原始实现
- [[深度学习 MOC]] — 主题索引

## 后续问题

- [ ] Positional Encoding 的具体实现方式与原理
- [ ] Transformer 完整的编码器-解码器结构
- [ ] 不同注意力头是否真的学到了不同的关系模式（可视化验证）
