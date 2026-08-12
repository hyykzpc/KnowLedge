import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        # Q、K、V 的形状均为 (batch_size, n_heads, seq_len, d_k)。
        # 其中 d_k 是每个注意力头内部的特征维度，n_heads 表示并行注意力头数量。
        d_k = Q.size(-1)

        # 缩放点积注意力的核心公式：
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        #
        # 1. K.transpose(-2, -1) 将 K 的最后两个维度交换，
        #    形状从 (batch_size, n_heads, seq_len, d_k)
        #    变为 (batch_size, n_heads, d_k, seq_len)。
        # 2. Q 与 K^T 相乘后，得到每个 token 对所有 token 的相关性分数，
        #    scores 的形状为 (batch_size, n_heads, seq_len, seq_len)。
        # 3. 除以 sqrt(d_k) 是为了控制点积结果的方差。
        #    当 d_k 较大时，未缩放的点积容易变得很大，使 softmax 进入饱和区，
        #    导致梯度过小；缩放可以让训练更稳定。
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask 中为 0 的位置表示不允许被关注。
            # 将这些位置的分数设为 -inf 后，softmax 会把对应权重变成 0。
            # 常见用途：
            # 1. padding mask：避免关注补齐出来的无效 token；
            # 2. causal mask：在解码器中避免当前位置看到未来 token。
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 在最后一个维度上做 softmax，使每个 query 对所有 key 的注意力权重和为 1。
        attn = self.softmax(scores)
        # 对注意力权重做 dropout，相当于训练时随机削弱部分依赖关系，
        # 可以降低模型对某些 token 的过度依赖，起到正则化作用。
        attn = self.dropout(attn)

        # 用注意力权重对 V 做加权求和。
        # 输出 out 的形状仍为 (batch_size, n_heads, seq_len, d_k)，
        # 表示每个位置融合上下文信息后的表示。
        out = torch.matmul(attn, V)

        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        # d_model：Transformer 中每个位置的隐藏表示维度，
        # 例如常见的 512、768。输入 q/k/v 的最后一维都应该是 d_model。
        # n_heads：注意力头的数量。多头注意力会把 d_model 拆成 n_heads 份，
        # 让不同注意力头在不同子空间中学习不同类型的依赖关系。

        # d_model 必须能被 n_heads 整除，这样每个头才能分到相同维度 d_k。
        assert d_model % n_heads == 0
        # 每个注意力头分到的维度，满足 n_heads * d_k = d_model。
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # 将输入分别线性映射成 Query、Key、Value。
        # 这里输入和输出维度都保持为 d_model，是为了后续可以方便地 reshape
        # 成 n_heads 个头，每个头维度为 d_k。
        #
        # Q 用来表示“当前位置想查找什么信息”；
        # K 用来表示“每个位置能提供什么匹配特征”；
        # V 用来表示“真正被加权汇总的信息内容”。
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 多个头的输出拼接后，仍然是 d_model 维；
        # 再通过一个线性层做信息混合，得到最终的多头注意力输出。
        self.fc = nn.Linear(d_model, d_model)

        self.attention = SelfAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        # LayerNorm 放在残差连接之后，用来稳定每层输出的数值分布，
        # 缓解深层网络训练中的梯度不稳定问题。
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        # q、k、v 的常见形状为 (batch_size, seq_len, d_model)。
        # 自注意力中 q、k、v 通常来自同一个输入；
        # 编码器-解码器注意力中，q 来自解码器，k/v 来自编码器输出。
        batch_size = q.size(0)

        # 线性映射后，将 d_model 拆成 n_heads * d_k：
        # (batch_size, seq_len, d_model)
        # -> (batch_size, seq_len, n_heads, d_k)
        # -> (batch_size, n_heads, seq_len, d_k)
        #
        # transpose(1, 2) 把 n_heads 提到 seq_len 前面，
        # 这样每个注意力头都可以独立地计算一套注意力分布。
        Q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 在每个头上并行计算缩放点积注意力。
        # out:  (batch_size, n_heads, seq_len, d_k)
        # attn: (batch_size, n_heads, seq_len, seq_len)
        out, attn = self.attention(Q, K, V, mask)

        # 将多头结果重新拼接回 d_model 维。
        # transpose 后张量在内存中可能不连续，所以先调用 contiguous()，
        # 再用 view 重新整理形状：
        # (batch_size, n_heads, seq_len, d_k)
        # -> (batch_size, seq_len, n_heads, d_k)
        # -> (batch_size, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        # 输出线性层用于融合不同头的信息；dropout 用于正则化。
        out = self.fc(out)
        out = self.dropout(out)

        # 残差连接保留原始输入 q 的信息，使模型更容易训练深层结构；
        # LayerNorm 再对每个 token 的 d_model 维特征做归一化。
        # 返回值包含：
        # 1. 融合上下文后的表示，形状为 (batch_size, seq_len, d_model)；
        # 2. 注意力权重 attn，可用于分析每个 token 关注了哪些位置。
        return self.norm(out + q), attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # d_model：输入和输出的维度，通常与 MultiHeadAttention 的输出维度相同。
        # d_ff：前馈网络中间层的维度，通常比 d_model 大很多，例如 2048。
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # 前馈网络由两层全连接组成，中间使用 ReLU 激活函数。
        # 输入 x 的形状为 (batch_size, seq_len, d_model)。
        out =self.fc2(self.dropout(torch.relu(self.fc1(x))))

        # 残差连接和 LayerNorm 与 MultiHeadAttention 类似，
        # 保持输入信息并稳定数值分布。
        return self.norm(out + x)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
    def forward(self,src,src_mask=None):
        out,_ = self.self_attn(src,src,src,src_mask)
        out = self.ffn(out)
        return out

class DecoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model,n_heads,dropout)
        self.cross_attn = MultiHeadAttention(d_model,n_heads,dropout)
        self.ffn = FeedForward(d_model,d_ff,dropout)
    
    def forward(self,tgt,memory,tgt_mask,mem_mask):
        out,_ = self.self_attn(tgt,tgt,tgt,tgt_mask)
        out,_ = self.cross_attn(out,memory,memory,mem_mask)
        out = self.ffn(out)
        return out

class PostionEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + self.pe[:,:x.size(1),:]
        return x

class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,n_heads,num_layers,d_ff,dropout=0.1,max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos_embedding = PostionEncoding(d_model,max_len)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model,n_heads,d_ff,dropout) for _ in range(num_layers)])
        
    def forward(self,src,src_mask=None):
        out = self.embedding(src)*math.sqrt(self.embedding.embedding_dim)
        out = self.pos_embedding(out)
        for layer in self.layers:
            out = layer(out,src_mask)
        return out
    
class Decoder(nn.Module):
    def __init__(self,vocab_size,d_model,n_heads,num_layers,d_ff,dropout=0.1,max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos_embedding = PostionEncoding(d_model,max_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model,n_heads,d_ff,dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model,vocab_size)
        
    def forward(self,tgt,memory,tgt_mask=None,memory_mask=None):
        out = self.embedding(tgt)*math.sqrt(self.embedding.embedding_dim)
        out = self.pos_embedding(out)
        for layer in self.layers:
            out = layer(out,memory,tgt_mask,memory_mask)
        return self.fc_out(out)

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab,
                 tgt_vocab,
                 d_model=512,
                 n_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_ff=2048,
                 dropout=0.1,
                 max_len=5000):
        super().__init__()
        
        self.encoder = Encoder(
            src_vocab,
            d_model,
            n_heads,
            num_encoder_layers,
            d_ff,
            dropout,
            max_len
        )
        self.decoder = Decoder(
            tgt_vocab,
            d_model,
            n_heads,
            num_decoder_layers,
            d_ff,
            dropout,
            max_len
        )
    
    def forward(self,src,tgt,src_mask=None,tgt_mask=None):
        memory = self.encoder(src,src_mask)
        out = self.decoder(tgt,memory,tgt_mask,src_mask)
        return out
    
def generate_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask==0


src_vocab = 10000
tgt_vocab = 10000

model = Transformer(src_vocab,tgt_vocab)
src = torch.randint(0,src_vocab,(32,10))
tgt = torch.randint(0,tgt_vocab,(32,10))

tgt_mask = generate_mask(tgt.size(1)).to(tgt.device)  
out = model(src,tgt,tgt_mask=tgt_mask) 
print(out.shape)  # 输出形状应为 (32, 10, tgt_vocab)
