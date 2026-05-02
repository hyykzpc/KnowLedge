---
title: Transformer - 代码实现解析学习笔记
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

# Transformer — 代码实现解析

## 核心摘要
- Transformer 完全基于注意力机制，抛弃 RNN/CNN 的序列归纳偏置。
- 核心公式：$\text{Attention}(Q,K,V) = \text{softmax}(QK^T / \sqrt{d_k}) V$。
- 整体架构：Embedding + Positional Encoding → N× EncoderLayer → N× DecoderLayer → Linear 输出。
- 每个 EncoderLayer 含自注意力 + FFN；每个 DecoderLayer 含掩码自注意力 + 交叉注意力 + FFN。

---

## 1. SelfAttention：缩放点积注意力

```python
class SelfAttention(nn.Module):
```

定义一个不含可学习参数的注意力计算单元，后续由 MultiHeadAttention 调用。

---

```python
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
```

初始化 dropout 正则化和 softmax 层。`dim=-1` 表示在最后一个维度（每个 query 对所有 key 的分数）上做归一化，使每行权重和为 1。

---

```python
    def forward(self, Q, K, V, mask=None):
```

接收 Q、K、V 和可选的 mask。
**输入形状：** `Q, K, V` 均为 `(batch_size, n_heads, seq_len, d_k)`。

---

```python
        d_k = Q.size(-1)
```

取出每个注意力头的维度 $d_k$，用于后续缩放。

---

```python
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
```

**对应公式：** $\frac{QK^T}{\sqrt{d_k}}$。

- `K.transpose(-2, -1)`：交换 K 的最后两维，形状从 `(b, h, s, d_k)` → `(b, h, d_k, s)`，使 Q 和 K^T 可以矩阵相乘。
- `torch.matmul(Q, K^T)`：结果 `scores` 形状为 `(b, h, s, s)`，其中 `scores[h, i, j]` 表示第 h 头中位置 i（query）对位置 j（key）的点积相似度。
- ` / math.sqrt(d_k)`：缩放因子。**原理：** 若 Q、K 各分量独立同分布、均值为 0、方差为 1，点积的方差为 $d_k$。$d_k$ 越大，点积数值越大，softmax 越容易进入梯度极小的饱和区。缩放后方差回归 1，梯度更稳定。

---

```python
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
```

mask 中为 0 的位置设为 $-\infty$，后续 softmax 将其变为 0，使这些位置不参与注意力计算。两种用途：
- **Padding Mask**：忽略补齐的无效 token。
- **Causal Mask**（下三角，解码器用）：位置 i 只能看到 j ≤ i 的位置，防止未来信息泄露。

---

```python
        attn = self.softmax(scores)
```

在最后一个维度（所有 key 上）做 softmax，将分数转为概率分布。`attn` 形状不变：`(b, h, s, s)`，每行和为 1。

---

```python
        attn = self.dropout(attn)
```

对注意力权重做 dropout，随机将部分权重置 0。**原理：** 训练时削弱某些 token 间的依赖，防止过拟合，类似正则化。

---

```python
        out = torch.matmul(attn, V)
```

**对应公式：** $\text{softmax}(QK^T / \sqrt{d_k}) V$。

`attn` 与 V 相乘：注意力权重对 value 做加权求和。形状变化：`attn (b, h, s, s) × V (b, h, s, d_k) → out (b, h, s, d_k)`。每个 query 位置用对所有 key 的注意力权重，加权汇总所有 key 对应的 value。

---

```python
        return out, attn
```

返回：
- `out`：`(b, h, s, d_k)`，上下文融合后的表示。
- `attn`：`(b, h, s, s)`，注意力权重，可用于可视化分析。

---

## 2. MultiHeadAttention：多头注意力

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
```

定义多头注意力。`d_model` 是模型中每个位置的表示维度（如 512），`n_heads` 是注意力头数。`assert` 确保 `d_model` 能被 `n_heads` 整除。

---

```python
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
```

每个头分到的维度 $d_k = d_{model} / n_{heads}$，满足 `n_heads * d_k = d_model`。

---

```python
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
```

三个线性投影层，分别将输入映射到 Q、K、V 空间。**对应公式：** $QW_i^Q$, $KW_i^K$, $VW_i^V$。输入输出维度均保持 `d_model`，便于后续 reshape 拆多头。

为什么需要三个投影？Q 表示"我想找什么"，K 表示"我能匹配什么"，V 表示"我提供什么信息"。三个独立空间避免相互干扰。

---

```python
        self.fc = nn.Linear(d_model, d_model)
```

输出融合层。**对应公式：** $\text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$。多头输出拼接后仍是 `d_model` 维，`fc` 做跨头信息融合。

---

```python
        self.attention = SelfAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
```

注意力计算实例 + 输出 dropout + LayerNorm。LayerNorm 在特征维度（`d_model`）上做归一化，不依赖 batch 大小，天然适配变长序列。

---

```python
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
```

**输入形状：** `q, k, v` 均为 `(batch_size, seq_len, d_model)`。

三种注意力模式的 q/k/v 来源：

| 模式 | q | k/v | 场景 |
|---|---|---|---|
| 编码器自注意力 | 编码器输入 | 编码器输入 | 编码器每层 |
| 解码器自注意力 | 解码器输入 | 解码器输入（+ 因果掩码） | 解码器第一子层 |
| 交叉注意力 | 解码器输出 | 编码器输出（memory） | 解码器第二子层 |

---

```python
        Q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
```

**对应操作：** 线性映射 → 拆分为多头 → 重排维度。

```
W_q(q):           (batch, seq_len, d_model)           # 线性投影
.view(...):       (batch, seq_len, n_heads, d_k)      # 拆为 n_heads 份
.transpose(1,2):  (batch, n_heads, seq_len, d_k)      # 头维度提前
```

`transpose(1, 2)` 把 `n_heads` 提到 `seq_len` 前面，后续 `SelfAttention` 的 `matmul` 才能在每个头内独立并行。

---

```python
        K = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
```

K、V 的相同变换，维度变化同 Q。

---

```python
        out, attn = self.attention(Q, K, V, mask)
```

调用 SelfAttention，在所有头上并行计算。

**对应公式（每个头独立执行）：**
$$
\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

- `out` 形状：`(b, n_heads, seq_len, d_k)`
- `attn` 形状：`(b, n_heads, seq_len, seq_len)`

**多头原理：** 单头将所有关系塞入同一空间。多头让不同头在不同子空间关注不同类型的关系（如一头聚焦语法、一头聚焦语义），最后互补。

---

```python
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
```

拼接多头输出回 `d_model` 维。

```
out: (b, n_heads, seq_len, d_k)
    → transpose(1,2): (b, seq_len, n_heads, d_k)   # 恢复原始维度排列
    → contiguous():   同形状，内存重新连续            # view 需连续内存
    → view: (b, seq_len, d_model)                   # 拼接：n_heads*d_k = d_model
```

---

```python
        out = self.fc(out)
```

跨头融合。`fc` 将拼接后的 `(b, seq_len, d_model)` 再做一次线性变换，让不同头的信息交叉混合。

---

```python
        out = self.dropout(out)
```

输出前做 dropout 正则化。

---

```python
        return self.norm(out + q), attn
```

残差连接 + LayerNorm。

- `out + q`：**对应公式：** $\text{LayerNorm}(x + \text{Sublayer}(x))$。残差连接使梯度可直接短路回传，缓解深层网络梯度消失。
- `self.norm(...)`：逐 token 做 LayerNorm。
- 返回：输出 `(b, seq_len, d_model)` + `attn`（用于可视化）。

---

## 3. FeedForward：逐位置前馈网络

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
```

两层全连接，形状呈瓶颈状 `d_model → d_ff(宽) → d_model`。$d_{ff}$ 通常设为 $4 \times d_{model}$（原始论文默认 2048，对应 $d_{model}=512$）。

**原理：** Transformer 论文称其为"位置级前馈网络"（position-wise FFN），它对每个位置独立使用同一组参数做非线性变换。先升维 4 倍引入更多容量，ReLU 激活后再降回原维。

---

```python
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
```

输出 dropout + LayerNorm，与 MultiHeadAttention 一致。

---

```python
    def forward(self, x):
        out = self.fc2(self.dropout(torch.relu(self.fc1(x))))
```

**对应操作：** 升维 → ReLU → Dropout → 降维。

```
x: (batch, seq_len, d_model)
    → fc1 → (batch, seq_len, d_ff)    # 升维
    → relu → 同形状                    # 非线性
    → dropout → 同形状                 # 正则化
    → fc2 → (batch, seq_len, d_model) # 降维回 d_model
```

---

```python
        return self.norm(out + x)
```

残差连接 + LayerNorm。`out + x` 保留原始输入，`norm` 做特征维度归一化。

---

## 4. EncoderLayer：编码器层

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
```

编码器层由**两个子层**组成：多头自注意力 → 前馈网络。

**原理：** 编码器层的作用是让每个 token 通过自注意力聚合所有其他 token 的信息，再经 FFN 做非线性变换。每个子层遵循 $\text{LayerNorm}(x + \text{Sublayer}(x))$ 的标准模式。

---

```python
    def forward(self, src, src_mask=None):
        out, _ = self.self_attn(src, src, src, src_mask)
```

自注意力层：`q = k = v = src`，即每个位置关注编码器所有其他位置。`src_mask` 为 padding mask（可选）。

**维度：** `src: (b, s, d_model) → self_attn → out: (b, s, d_model)`。

---

```python
        out = self.ffn(out)
```

前馈网络层：对自注意力输出做非线性变换。

**维度：** `out: (b, s, d_model) → ffn → (b, s, d_model)`。

---

```python
        return out
```

返回编码后的表示，形状不变 `(batch, seq_len, d_model)`。

---

## 5. DecoderLayer：解码器层

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
```

解码器层由**三个子层**组成：掩码自注意力 → 交叉注意力 → 前馈网络。比编码器多了一个交叉注意力层。

---

```python
    def forward(self, tgt, memory, tgt_mask, mem_mask):
        out, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
```

**第一子层 — 掩码自注意力：** `q = k = v = tgt`，但 `tgt_mask` 是因果掩码（下三角），当前位置只能看到自身及之前位置，看不到未来 token。

**维度：** `tgt: (b, tgt_s, d_model) → self_attn → out: (b, tgt_s, d_model)`。

---

```python
        out, _ = self.cross_attn(out, memory, memory, mem_mask)
```

**第二子层 — 交叉注意力（编码器-解码器注意力）：** `q = out`（解码器上一层的输出），`k = v = memory`（编码器的最终输出）。

**原理：** 解码器每个位置作为 query，去关注编码器输出的所有位置，找到源语言中与当前解码位置最相关的信息。

**维度：** `out: (b, tgt_s, d_model), memory: (b, src_s, d_model) → cross_attn → out: (b, tgt_s, d_model)`。

---

```python
        out = self.ffn(out)
```

**第三子层 — 前馈网络：** 对交叉注意力输出做非线性变换。

**维度：** `out: (b, tgt_s, d_model) → ffn → (b, tgt_s, d_model)`。

```python
        return out
```

返回解码后的表示，形状不变。

---

## 6. PositionEncoding：位置编码

```python
class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
```

初始化位置编码矩阵 `pe: (max_len, d_model)`，每个位置存一个 $d_{model}$ 维向量。

**原理：** 自注意力没有位置概念（permutation invariant 或 set-like），需要注入位置信息。Transformer 使用固定频率正余弦编码而非可学习的嵌入。

---

```python
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
```

`position: (max_len, 1)`，即 `[[0], [1], [2], ..., [max_len-1]]`。每个值代表序列中的绝对位置。

---

```python
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
```

生成频率递减的除数项。`torch.arange(0, d_model, 2)` 取所有偶数下标 `[0, 2, 4, ..., d_model-2]`。

**对应公式（取 $i$ 为维度索引的一半）：**
$$
div\_term[i] = \exp\left(2i \cdot \frac{-\ln 10000}{d_{model}}\right) = \frac{1}{10000^{2i/d_{model}}}
$$

不同频率使不同维度对位置变化的敏感度不同——低频维度编码远距离信息，高频维度编码近距离信息。

---

```python
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
```

偶数索引填 $\sin$，奇数索引填 $\cos$。

**完整公式：**
$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
$$
$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

**为什么用正余弦交替？** 对任意偏移 $k$，$PE_{pos+k}$ 可表示为 $PE_{pos}$ 的线性变换，使模型能学到相对位置关系而非绝对位置。

---

```python
        pe = pe.unsqueeze(0)  # → (1, max_len, d_model)
        self.register_buffer('pe', pe)
```

`unsqueeze(0)` 增加 batch 维度，后续可通过广播加到任意 batch 的输入上。`register_buffer` 注册为持久化张量，随模型移动设备但不参与梯度更新。

---

```python
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
```

位置编码直接加在词嵌入上。`self.pe[:, :seq_len, :]` 根据输入的实际序列长度切片。

**维度变化：** `x: (b, s, d_model) + pe 广播 → (b, s, d_model)`。

---

## 7. Encoder：完整编码器

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionEncoding(d_model, max_len)
```

Token 嵌入和位置编码。`Embedding` 将词索引映射为稠密向量，形状 `(vocab_size, d_model)`。

---

```python
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
```

堆叠 `num_layers` 层 EncoderLayer。`nn.ModuleList` 让列表中的子层被正确注册为模型参数（普通 Python list 不会注册）。

---

```python
    def forward(self, src, src_mask=None):
        out = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
```

词嵌入 + 缩放。`src: (b, s)`（词索引）→ `embedding: (b, s, d_model)`。

乘 $\sqrt{d_{model}}$ 的原因：嵌入值和位置编码初始尺度不同，缩放后两者在同一量级，避免一方主导。

---

```python
        out = self.pos_embedding(out)
```

加位置编码。`out: (b, s, d_model) → 不变`。

---

```python
        for layer in self.layers:
            out = layer(out, src_mask)
        return out
```

逐层通过 N 个 EncoderLayer。每层输入输出形状均为 `(b, s, d_model)`。

最终输出称为 **memory**，形状 `(batch, src_seq_len, d_model)`，将传递给解码器的交叉注意力层作为 k/v。

---

## 8. Decoder：完整解码器

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionEncoding(d_model, max_len)
```

与编码器相同的嵌入和位置编码，处理目标语言序列。

---

```python
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
```

堆叠 `num_layers` 层 DecoderLayer。

---

```python
        self.fc_out = nn.Linear(d_model, vocab_size)
```

输出投影层。将 `d_model` 维表示映射到 `vocab_size` 维的 logits，即每个位置预测词表中每个词的分数。

---

```python
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        out = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
```

目标序列嵌入和缩放。`tgt: (b, tgt_s)` → `out: (b, tgt_s, d_model)`（乘 $\sqrt{d_{model}}$）。

---

```python
        out = self.pos_embedding(out)
```

加位置编码。

---

```python
        for layer in self.layers:
            out = layer(out, memory, tgt_mask, memory_mask)
```

逐层通过 N 个 DecoderLayer，每层接收 memory（编码器输出）和掩码。

`tgt_mask`：因果掩码，防止看到未来 token。
`memory_mask`：padding mask（可选），忽略源语言补齐 token。

---

```python
        return self.fc_out(out)
```

投影到词表 logits。`out: (b, tgt_s, d_model) → fc_out → (b, tgt_s, vocab_size)`。

---

## 9. Transformer：完整模型

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
```

完整 Transformer，默认参数与原始论文一致。`src_vocab` 和 `tgt_vocab` 可不同大小（例如中英文词汇量不同）。$d_{model}=512, n_{heads}=8, N=6, d_{ff}=2048$。

---

```python
        self.encoder = Encoder(src_vocab, d_model, n_heads, num_encoder_layers, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab, d_model, n_heads, num_decoder_layers, d_ff, dropout, max_len)
```

编码器处理源语言，解码器处理目标语言。两者共享同样的 `d_model, n_heads` 等配置，但词汇表可不同。

---

```python
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encoder(src, src_mask)
```

编码器将源语言序列转为上下文表示 memory。

```
src: (b, src_seq_len)
    → encoder → memory: (b, src_seq_len, d_model)
```

---

```python
        out = self.decoder(tgt, memory, tgt_mask, src_mask)
```

解码器基于 memory（源语言信息）和 tgt（目标语言序列）自回归生成输出。

```
tgt: (b, tgt_seq_len), memory: (b, src_seq_len, d_model)
    → decoder → out: (b, tgt_seq_len, tgt_vocab)
```

---

```python
        return out
```

返回形状 `(batch, tgt_seq_len, tgt_vocab)`，每个时间步上是目标词表的 logits。

---

## 10. generate_mask：因果掩码

```python
def generate_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask == 0
```

生成下三角因果掩码。

以 `size=4` 为例：
```
torch.triu(ones(4,4), diagonal=1).bool()
→ [[0, 1, 1, 1],      # 上三角为 True
   [0, 0, 1, 1],
   [0, 0, 0, 1],
   [0, 0, 0, 0]]

return mask == 0
→ [[1, 0, 0, 0],      # 下三角为 True
   [1, 1, 0, 0],
   [1, 1, 1, 0],
   [1, 1, 1, 1]]
```

`mask == 0` 将下三角（当前位置及之前）置 True，上三角（未来位置）置 False。与 SelfAttention 的 `masked_fill(mask == 0, -inf)` 配合时，未来位置权重为 0。

---

## 11. 使用示例

```python
src_vocab = 10000
tgt_vocab = 10000
model = Transformer(src_vocab, tgt_vocab)
```

实例化一个 Transformer，源和目标词汇量各 10000，其他参数用默认值（d_model=512, n_heads=8, N=6, d_ff=2048）。

---

```python
src = torch.randint(0, src_vocab, (32, 10))
tgt = torch.randint(0, tgt_vocab, (32, 10))
```

随机生成模拟数据：`src` 和 `tgt` 均为 `(batch=32, seq_len=10)`。

---

```python
tgt_mask = generate_mask(tgt.size(1)).to(tgt.device)
```

生成因果掩码，形状 `(10, 10)`，传到与 tgt 相同的设备。

---

```python
out = model(src, tgt, tgt_mask=tgt_mask)
print(out.shape)  # (32, 10, 10000)
```

前向传播。全流程维度传递：
```
src (32,10) → Encoder → memory (32,10,512)
tgt (32,10) → Decoder(memory) → (32,10,10000)
                      ↑
                 tgt_mask (10,10)
```

输出 `(batch=32, tgt_seq_len=10, tgt_vocab=10000)`，每个时间步预测整个词表的 logits。

---

## 完整架构一览

```
输入序列 (batch, src_seq_len)
    ↓ Embedding × √d_model
    ↓ Positional Encoding
┌──────────────────────────────┐
│  EncoderLayer × N            │
│    ├── MultiHead Self-Attn   │
│    │   └── q=k=v=src         │
│    ├── [+] Residual + LN     │
│    ├── FeedForward           │
│    └── [+] Residual + LN     │
└──────────┬───────────────────┘
           ↓ memory (b, src_s, d_model)
           │
目标序列 (batch, tgt_seq_len)  │
    ↓ Embedding × √d_model      │
    ↓ Positional Encoding       │
┌──────────────────────────────┐│
│  DecoderLayer × N           ││
│    ├── Masked Self-Attn     ││
│    │   └── q=k=v=tgt + mask ││
│    ├── [+] Residual + LN    ││
│    ├── Cross-Attn ←─────────┘│
│    │   └── q=dec, k/v=mem   │
│    ├── [+] Residual + LN    │
│    ├── FeedForward          │
│    └── [+] Residual + LN    │
└──────────┬───────────────────┘
           ↓
    Linear → (b, tgt_s, tgt_vocab)
```

---

## 关键结论

- **SelfAttention 核心三步：** Q×K^T → softmax → ×V，再加缩放和可选的掩码。
- **多头 = 子空间并行：** 每个头捕获不同模式，拼接后融合。
- **残差连接 + LayerNorm：** 让 N=6 甚至更深的 Transformer 可以稳定训练。
- **位置编码打破置换不变性：** 正余弦频率编码使模型可区分 token 顺序。
- **Encoder-Decoder 架构：** 编码器生成 memory，解码器自回归生成，交叉注意力桥接两端。

## 知识连接

- [[神经网络与深度学习 - 第8章学习笔记]] — 注意力机制的数学基础
- [[注意力机制举例_第八章学习笔记追加]] — 注意力机制 BP 数值例子
- [[code/self_attenation.py]] — 单头注意力的原始实现
- [[深度学习 MOC]] — 主题索引

## 后续问题

- [ ] Label Smoothing 在 Transformer 训练中的作用
- [ ] 推理时的自回归解码与 Beam Search
- [ ] Transformer 的參数量计算（Embedding、注意力、FFN）
