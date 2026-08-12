这次我按论文原文和作者官方实现重新精读。先给你一个最重要的结论：

> **GPP-LLIE 不是“让 VLM 生成增强图像”，而是让 VLM 充当一个低层视觉质量感知器，输出“哪里差、差到什么程度”的全局/局部先验；真正负责生成正常曝光图像的是后面的 latent diffusion Transformer。** [arXiv](https://arxiv.org/html/2412.20916v1)

所以整篇论文可以拆成两个完全不同的系统：

\[ \boxed{\text{VLM Prior Extraction}} \qquad+\qquad \boxed{\text{Prior-Guided Diffusion Enhancement}} \]

这也是理解 Figure 2 和 Figure 3 的关键。

---

# 一、先重新定位这篇论文到底在解决什么

作者认为传统 LLIE 最大的问题不只是“暗”，而是**面对真实世界不同照明条件时，模型不知道当前图像究竟应该增强多少、哪些区域应该重点增强**。这容易造成两类问题：

- 暗区域增强不足；
- 本来已经比较亮的区域又被增强，产生过曝光。

因此作者想让模型在恢复之前先有一个“视觉判断”：

> 这张图整体的 contrast / visibility / sharpness 到底怎么样？

以及：

> 左上角怎么样？右下角怎么样？哪些局部区域特别差？

这正是 VLM 被引入的原因。作者使用的不是普通 LLaVA，而是经过 Q-Instruct 低层视觉指令数据进一步训练的 LLaVA，使其具备更强的低层视觉属性判断能力。论文针对 LLIE 选择了三个属性：

\[ \boxed{ \text{Contrast},\quad \text{Visibility},\quad \text{Sharpness} } \]

而不是让 VLM 描述“图里有一个人、一棵树、一辆车”。 [arXiv](https://arxiv.org/html/2412.20916v1)

这一点和我们之前讨论的 Semantic-Sensitive VLM 方法差别很大：

\[ \text{语义 VLM：这是什么？} \]

而这篇：

\[ \text{GPP：这里的视觉质量怎么样？} \]

---

# 二、整篇论文最核心的两张图

实际上你理解这篇论文，只要吃透：

- **Figure 2：GPP 怎么从 VLM 得到**
- **Figure 3：GPP 怎么进入 diffusion network**

即可。

整个逻辑可以先压缩成：

```
                  ┌──────────── LLaVA / Q-Instruct ────────────┐
                  │                                            │
Low-light image ──┼── 全局评价 ─────→ Global Score S          │
       │          │                                            │
       │          └── 分块评价 ─────→ Local Quality Map M     │
       │
       │
       └── Encoder ────────────→ Low-light latent z_ll
                                      │
                                      ↓
Gaussian Noise z_T ──→ GPP-LLIE Transformer ──→ z_0 ──→ Decoder
                              ↑
                        S + M + z_ll
                                      │
                                      ↓
                             Enhanced Image
```

也就是说，**同一张低照度图像 \(I_{ll}\) 走两条路**：

\[ I_{ll} \rightarrow \begin{cases} \text{Encoder} \rightarrow z_{ll}\\ \text{VLM} \rightarrow S,M \end{cases} \]

最后三种信息：

\[ \boxed{z_{ll},\ S,\ M} \]

一起指导 diffusion 的逆扩散。 [arXiv](https://arxiv.org/html/2412.20916v1)

下面逐张图拆。

---

# 三、Figure 2：VLM 到底是怎么产生 GPP 的？

Figure 2 的标题本身就非常关键：

> fine-grained generative perceptual priors extraction from pre-trained VLMs.

它分为四步：

\[ (a)\rightarrow(b)\rightarrow(c)\rightarrow(d) \]

即：

\[ \text{Patchify} \rightarrow \text{VLM Assessment} \rightarrow \text{Quantification} \rightarrow \text{Global/Local Prior} \]

论文明确说明这一流程。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

## 3.1 第一步：不是只把整张图送进 VLM

输入：

\[ I_{ll} \]

首先保留完整图像，同时将它分成多个**互不重叠的 patch**。

例如假设分成 \(4\times4\)：

```
Original LL Image

┌────┬────┬────┬────┐
│ P1 │ P2 │ P3 │ P4 │
├────┼────┼────┼────┤
│ P5 │ P6 │ P7 │ P8 │
├────┼────┼────┼────┤
│ P9 │... │    │    │
├────┼────┼────┼────┤
│    │    │    │P16 │
└────┴────┴────┴────┘
```

于是出现两类输入。

### 全局输入

整张：

\[ I_{ll} \]

负责回答：

> 整张图的 visibility 怎么样？

---

### 局部输入

每个 patch：

\[ P_i \]

负责回答：

> 这个局部区域的 visibility 怎么样？

为什么必须做 local？

因为真实低照场景很可能：

```
左边：非常暗
中间：正常
右边：有灯，已经很亮
```

如果只有一个 global score：

\[ S=0.4 \]

网络只能知道：

> 整体偏差。

但不知道：

> 到底哪里差。

因此 Figure 2 的 patchification 本质上是在制造一个**空间质量描述**。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 四、第二步：VLM并不是识别物体，而是在“打分”

这是前面最容易误解的地方。

作者不是问：

> What's in this image?

而是设计 **Evaluation Command**，要求 LLaVA 针对指定低层视觉属性进行评价。

例如：

\[ A\in \{ \text{contrast}, \text{visibility}, \text{sharpness} \} \]

每一个 attribute 还配有相应 definition，帮助 VLM 理解“你到底让我判断什么”。 [arXiv](https://arxiv.org/html/2412.20916v1)

所以会分别问：

```
contrast 好不好？
visibility 好不好？
sharpness 好不好？
```

这三项不是同一件事情。

比如一张图可能：

\[ \text{visibility}=poor \]

但：

\[ \text{sharpness}=good \]

意思就是：

> 图虽然暗，但现有边缘并不模糊。

这时候增强网络应该：

**重点提亮，而不是疯狂锐化。**

这就是 GPP 比简单亮度图更有意思的地方。

---

# 五、Figure 2 最关键的一步：为什么不直接让 LLaVA 输出“3.5分”？

作者发现，直接使用 VLM 的自然语言答案并不稳定。

例如模型可能输出：

> The visibility of this image is poor.

但是：

“poor”

本身是一个离散 token。

而作者真正需要送进神经网络的是：

\[ \boxed{\text{连续数值}} \]

例如：

\[ 0.13 \]

或者：

\[ 0.78 \]

所以 Figure 2(c) 出现了：

## Sigmoid-based Quantification Strategy

作者关注两个相反 token：

\[ \text{good} \]

与

\[ \text{poor} \]

对应概率：

\[ \mathcal P_{\text{pos}} \]

和

\[ \mathcal P_{\text{neg}} \]

然后计算：

\[ \boxed{ S= \frac{1}{ 1+\exp \left( -\frac{ \mathcal P_{\text{pos}} - \mathcal P_{\text{neg}} }{\alpha} \right) } } \]

其中：

\[ \alpha=3 \]

。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 六、这个公式到底是什么意思？

不要先看 sigmoid，先看：

\[ \mathcal P_{\text{pos}} - \mathcal P_{\text{neg}} \]

假设 VLM 认为：

\[ P(good)=0.9 \]\[ P(poor)=0.1 \]

那么：

\[ P(good)-P(poor)>0 \]

说明质量好。

反过来：

\[ P(good)=0.1 \]\[ P(poor)=0.9 \]

那么：

\[ P(good)-P(poor)<0 \]

说明质量差。

Sigmoid 只是把它压到：

\[ (0,1) \]

中。

所以这里的先验不是：

> VLM说了一句话。

而是把 **VLM 的生成 token 分布转化成连续视觉质量信号**。

这才是 Figure 2 真正创新的地方。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 七、Global Prior \(S\) 到底是什么？

对整张图分别评价：

\[ S_{\text{contrast}} \]\[ S_{\text{visibility}} \]\[ S_{\text{sharpness}} \]

最后论文使用平均 global score 作为 global perceptual prior：

\[ \boxed{ S = \frac{ S_c+S_v+S_s }{3} } \]

论文说明三个属性被依次评价，并将平均 global score 用作感知先验。 [arXiv](https://arxiv.org/html/2412.20916v1)

所以最终：

\[ \boxed{S\in\mathbb R} \]

你可以把它理解成：

> **这张低照图从 VLM 感知角度看，整体“需要增强”的程度。**

当然严格说它是视觉质量评价，而不是直接的“增强强度”。

---

# 八、Local Prior \(M\) 又是什么？

对每个 patch 都执行类似过程。

比如 visibility：

```
┌────┬────┬────┐
│.12 │.18 │.50 │
├────┼────┼────┤
│.09 │.30 │.71 │
├────┼────┼────┤
│.16 │.42 │.83 │
└────┴────┴────┘
```

那么就形成一张：

\[ M_{\text{visibility}} \]

同理：

\[ M_{\text{contrast}} \]\[ M_{\text{sharpness}} \]

论文把三种局部质量图 concat，因此：

\[ \boxed{ M= \operatorname{Concat} ( M_c, M_v, M_s ) } \]

官方实现中的 local map 输入也是 3 通道，这与三个视觉属性对应。 [arXiv](https://arxiv.org/html/2412.20916v1)

所以：

\[ \boxed{ M\approx H_M\times W_M\times3 } \]

这就是 Figure 2(d) 最终输出的 **Local Quality Map**。

---

# 九、所以 Figure 2 一句话怎么理解？

它实际上是在干：

\[ \boxed{ \text{VLM} \rightarrow \text{Image Quality Assessor} } \]

而不是：

\[ \text{VLM} \rightarrow \text{Image Generator} \]

更准确的整个流程是：

\[ I_{ll} \xrightarrow{\text{LLaVA}} \text{视觉质量语言判断} \xrightarrow{\text{token probability}} \text{连续数值} \rightarrow \boxed{S,M} \]

这两个才叫：

**Generative Perceptual Priors**。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 十、现在进入最重要的 Figure 3：完整 GPP-LLIE 网络

Figure 3 第一眼很复杂，其实可以拆成三条横向流。

---

## 第一条：Normal-Light训练支路

训练时有 GT：

\[ I_{nl} \]

先经过 Encoder：

\[ \boxed{ z_{nl}^{0} = \mathcal E(I_{nl}) } \]

其中：

\[ z_{nl}^{0} \in \mathbb R^{ \frac Hf \times \frac Wf \times d } \]

也就是说，diffusion **不是直接在 RGB 空间工作**，而是在 latent feature space 工作。 [arXiv](https://arxiv.org/html/2412.20916v1)

然后对正常光 latent 做 forward diffusion：

\[ z_{nl}^{0} \rightarrow z_{nl}^{1} \rightarrow \cdots \rightarrow z_{nl}^{T} \]

最终逐渐变成噪声。

标准 diffusion 可以理解为：

\[ z_t = \sqrt{\bar\alpha_t}\,z_0 + \sqrt{1-\bar\alpha_t}\,\epsilon \]

其中：

\[ \epsilon\sim\mathcal N(0,I) \]

这里的训练目标相当于让网络学习：

> 给我一个被污染的正常光 latent，我怎么把它一步一步恢复回来？

论文明确说明 forward diffusion 作用在 normal-light latent \(z_{nl}^0\) 上。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 十一、第二条支路：Low-Light condition

同一对训练数据里的：

\[ I_{ll} \]

也经过 Encoder：

\[ \boxed{ z_{ll} = \mathcal E(I_{ll}) } \]

它的作用完全不同。

它**不参与 forward diffusion**。

它是：

\[ \boxed{\text{condition}} \]

告诉 diffusion：

> 你虽然从随机噪声开始生成正常光图像，但生成的内容必须对应这张低照度图。

否则 diffusion 可能生成：

> 一张很好看的正常曝光照片，

但和输入不是一张照片。

论文因此把 \(z_{ll}\) 作为逆扩散中的条件，用于保证 fidelity。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 十二、第三条支路：VLM Priors

同样的：

\[ I_{ll} \]

送入我们刚才 Figure 2 的 pipeline：

\[ I_{ll} \rightarrow \text{LLaVA} \rightarrow \boxed{S,M} \]

所以到这里，网络有三个条件：

\[ \boxed{ z_{ll} } \]

内容条件；

\[ \boxed{ S } \]

全局视觉质量条件；

\[ \boxed{ M } \]

局部空间视觉质量条件。

然后作者把它们全部送给：

\[ \epsilon_\theta \]

。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 十三、逆扩散过程到底怎么走？

Figure 3 底部有一个随机噪声：

\[ \hat z_{nl}^{T} \sim \mathcal N(0,I) \]

然后：

\[ \hat z_{nl}^{T} \rightarrow \hat z_{nl}^{T-1} \rightarrow \cdots \rightarrow \hat z_{nl}^{0} \]

但它不是无条件去噪。

每一步都是：

\[ \boxed{ \epsilon_\theta ( \hat z_{nl}^{t}, z_{ll}, S, M, t ) } \]

所以更直观地讲：

```
当前 noisy latent
       │
       ├── low-light latent：告诉你“内容是什么”
       │
       ├── global S：告诉你“整体质量多差”
       │
       └── local M：告诉你“哪里更差”
       ↓
 GPP-LLIE Transformer
       ↓
更干净、更正常曝光的 latent
```

最终：

\[ \hat z_{nl}^{0} \]

进入 Decoder：

\[ \boxed{ \hat I_{out} = \mathcal D(\hat z_{nl}^{0}) } \]

得到增强图像。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 十四、Figure 3 右边那个 GPP-LLIE Block 才是网络结构真正的核心

接下来重点看 Figure 3 右边。

它主要包含三个操作：

\[ \boxed{\text{GPP-LN}} \]\[ \boxed{\text{MSA}} \]\[ \boxed{\text{LPP-Attn}} \]

以及 Feed-Forward Network。

简单理解：

```
                   Global Prior S
                        ↓
                   GPP-LN
                        ↓
Current feature ──→ MSA ─────────────┐
                                     │
                   Local Prior M     │
                        ↓            │
                   LPP-Attn ←────────┤
                                     │
                   GPP-LN            │
                        ↓            │
                      FFN            │
                        ↓            │
                   output feature ←──┘
```

作者专门设计：

- global prior 控制 LayerNorm；
- local prior 控制 Attention。

这是整篇文章结构设计最漂亮的一点。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 十五、为什么 Global Prior 要进 LayerNorm？

作者提出：

## GPP-LN

普通 LayerNorm：

\[ \operatorname{LN}(z) \]

所有图像处理方式差不多。

但是现在有：

\[ S \]

它描述整张图的质量。

于是作者让：

\[ S \xrightarrow{\text{MLP}} \gamma,\beta \]

然后：

\[ \boxed{ z_{out} = \gamma(S) \odot \operatorname{LN}(z_{in}) + \beta(S) } \]

。 [arXiv](https://arxiv.org/html/2412.20916v1)

这句话非常重要：

> **S 并不是直接加到 feature 上，而是控制 feature 的归一化方式。**

---

# 十六、为什么这样比直接加 \(S\) 好？

假设两张图：

### 图 A

非常暗：

\[ S_A=0.15 \]

### 图 B

只是稍微暗：

\[ S_B=0.65 \]

如果网络完全一样处理：

\[ f(z) \]

那网络主要依赖训练集学到一个“平均增强策略”。

但现在变成：

\[ f(z|S) \]

因为：

\[ \gamma=\gamma(S) \]\[ \beta=\beta(S) \]

所以图像质量不同：

\[ S_A\neq S_B \]

导致 feature modulation 不同。

这就是：

\[ \boxed{\text{Adaptive Enhancement}} \]

作者的消融实验也显示，去掉 global score 或简单将其直接加入 latent，性能都会明显下降，说明 GPP-LN 的注入方式确实重要。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 十七、但这里还有 diffusion timestep \(t\)，它去哪了？

这是 Figure 3 很容易忽略的地方。

Diffusion 每一步本来就需要知道：

\[ t \]

因为：

```
t=1000：几乎全是噪声
t=500：中等噪声
t=20：已经接近清晰
```

所以 Transformer 同时需要：

\[ \text{diffusion condition }t \]

和：

\[ \text{perceptual condition }S \]

官方代码中，先得到 timestep embedding：

\[ e_t=\operatorname{Embed}(t) \]

然后把 global perceptual score conditioning 加入该 embedding，再用于生成后续 modulation 参数。 [GitHub](https://github.com/LowLevelAI/GPP-LLIE/blob/main/model_incontext_revise.py)

因此你可以把 GPP-LN 更完整地理解成：

\[ \boxed{ \text{Feature modulation} = f(t,S) } \]

即：

> “我现在去噪到第几步” + “这张图整体质量如何”

共同决定网络如何变换特征。

---

# 十八、Local Prior \(M\) 为什么不能也用 GPP-LN？

因为 \(S\) 是：

\[ \text{scalar/global} \]

而 \(M\) 是：

\[ H\times W \]

有空间位置。

比如：

```
       Quality Map M

dark        medium        good
↓↓↓↓           ↓             ↓

0.1  0.2  0.6  0.8
0.1  0.3  0.7  0.9
0.2  0.4  0.8  0.9
```

如果把它压成一个 scalar，再去 modulation：

\[ M\rightarrow S \]

那么：

> “左边很暗、右边很亮”

这个信息就没了。

因此作者把 \(M\) 放到了：

\[ \boxed{\text{Attention}} \]

里。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 十九、LPP-Attn 到底在做什么？

论文明确说：

- Query \(Q\) 来自当前 image feature；
- Key \(K\) 和 Value \(V\) 受到 local prior \(M\) 指导。 [arXiv](https://arxiv.org/html/2412.20916v1)

官方实现更加直观：

\[ Q=f_Q(x) \]

而：

\[ K=f_K(M) \]\[ V=f_V(M) \]

然后：

\[ A = \operatorname{Softmax} \left( \frac{QK^\top}{\tau} \right) \]

最后：

\[ Y=AV \]

。 [GitHub](https://github.com/LowLevelAI/GPP-LLIE/blob/main/model_incontext_revise.py)

所以它实际上非常接近一种：

\[ \boxed{\text{Cross Attention}} \]

其中：

```
Query
 ↓
当前 diffusion feature

Key / Value
 ↓
VLM local quality prior
```

---

# 二十、LPP-Attn 的直觉特别重要

假设：

```
                图像

┌───────────┬───────────┐
│           │   灯      │
│ 非常暗    │ 很亮      │
│ M=0.1     │ M=0.9     │
│           │           │
└───────────┴───────────┘
```

普通 Transformer：

> 主要根据 feature 自己决定 attention。

GPP-LLIE：

> VLM 额外告诉 attention：“左边视觉质量差，右边已经很好。”

于是网络能够学习：

\[ \text{不同区域采用不同恢复行为} \]

而不是：

\[ \text{整张图统一提亮} \]

这正是它减少局部过曝光的重要机制之一。论文的真实世界实验也强调了其对不均匀照明和过曝光抑制的优势。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 二十一、这里的 Attention 还有一个很特别的地方：不是传统 spatial attention

作者指出，如果在高分辨率 LLIE 上做标准 ViT spatial self-attention：

\[ N=HW \]

复杂度约为：

\[ O(N^2) = O(H^2W^2) \]

成本太大。

因此论文将 attention 计算移到**channel dimension**，降低高分辨率输入的计算负担。 [arXiv](https://arxiv.org/html/2412.20916v1)

官方实现也能看到：

\[ Q,K,V \]

被 reshape 成：

\[ B\times Head\times C\times HW \]

然后计算：

\[ QK^\top \]

得到的是 channel-to-channel attention，而不是传统的 \(HW\times HW\) attention。 [GitHub](https://github.com/LowLevelAI/GPP-LLIE/blob/main/model_incontext_revise.py)

这一设计其实明显借鉴了 Restormer 的 transposed attention 思想。

---

# 二十二、Concat-and-Remove 是干什么的？

这是 Figure 3 里另外一个很容易被忽视的小设计。

每个 GPP-LLIE Block 开始的时候：

\[ x \]

和 low-light condition：

\[ z_{ll} \]

进行 channel concat：

\[ \boxed{ x' = \operatorname{Concat}(x,z_{ll}) } \]

。 [arXiv](https://arxiv.org/html/2412.20916v1)

这样 Transformer 每一层都不断重新看到：

> 原始低照图的内容信息。

而不是第一层看一次之后就逐渐“忘掉”。

---

然后 block 结束时：

\[ x'\in\mathbb R^{2C\times H\times W} \]

作者把后一半 channel 删除：

\[ x_{out}=x'[:C,:,:] \]

恢复成：

\[ C\times H\times W \]

官方代码中正是：

\[ x=\text{cat}(x,y) \]

处理之后再取回前半通道。 [GitHub](https://github.com/LowLevelAI/GPP-LLIE/blob/main/model_incontext_revise.py)

所以名字就叫：

\[ \boxed{\text{Concat-and-Remove}} \]

---

# 二十三、为什么不 concat 一次就算了？

因为 diffusion 是一个：

\[ T\rightarrow0 \]

不断生成的过程。

如果只在网络最开始注入一次：

\[ z_{ll} \]

随着深层计算：

\[ x_1\rightarrow x_2\rightarrow\cdots\rightarrow x_N \]

条件信息可能越来越弱。

因此作者采取：

```
Block 1:
x + z_ll → process → remove z_ll

Block 2:
x + z_ll → process → remove z_ll

Block 3:
x + z_ll → process → remove z_ll
```

等价于：

\[ \boxed{\text{每个 Block 都重新注入输入图像约束}} \]

以保持 fidelity。论文对此设计的目的就是让 LL 信息持续进入 reverse diffusion。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 二十四、现在把一个完整 GPP-LLIE Block 串起来

你可以把一个 block 理解成：

```
                        Global Score S
                              │
                              ↓
                       condition modulation
                              │

current latent x ──┐
                   ├─ Concat ─→ [x ; z_ll]
low-light z_ll ────┘
                              │
                              ↓
                           GPP-LN
                              │
                              ↓
                       Channel Self-Attn
                              │
                           residual
                              │
             Local Map M ─────┤
                    │         ↓
                    └──→ LPP-Attn
                              │
                           residual
                              │
                              ↓
                           GPP-LN
                              │
                              ↓
                             FFN
                              │
                           residual
                              │
                              ↓
                      Remove z_ll channels
                              │
                              ↓
                          next block
```

这基本就是 Figure 3 右半部分的真正含义。论文的三项主要结构创新——Concat-and-Remove、GPP-LN、LPP-Attn——分别对应内容保真、全局适应和局部适应。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 二十五、所以三个条件实际上各司其职

这是我认为最值得记住的一张“逻辑表”：

|信息|来源|告诉网络什么|注入位置|
|---|---|---|---|
|\(z_{ll}\)|Encoder|**原图是什么**|每个 Block concat|
|\(S\)|VLM global assessment|**整张图质量如何**|GPP-LN|
|\(M\)|VLM patch assessment|**哪里质量差**|LPP-Attn|
|\(t\)|Diffusion|**现在去噪到哪一步**|diffusion conditioning|

因此：

\[ \boxed{ \epsilon_\theta = f( z_t, z_{ll}, S, M, t ) } \]

可以理解成五个问题：

> **现在生成到了哪里？**

\[ t \]

> **当前 latent 是什么？**

\[ z_t \]

> **原图内容是什么？**

\[ z_{ll} \]

> **整张图需要怎样的增强？**

\[ S \]

> **具体哪些区域更需要处理？**

\[ M \]

---

# 二十六、训练和测试时 Figure 3 不完全一样

这一点一定要分清。

## 训练阶段

有：

\[ (I_{ll},I_{nl}) \]

因此：

\[ I_{nl} \rightarrow \mathcal E \rightarrow z_{nl}^{0} \]

然后人为加噪：

\[ z_{nl}^{0} \rightarrow z_{nl}^{t} \]

让网络学习逆过程。

与此同时：

\[ I_{ll} \rightarrow z_{ll} \]

以及：

\[ I_{ll} \rightarrow VLM\rightarrow S,M \]

共同作为条件。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

## 测试阶段

没有：

\[ I_{nl} \]

只有：

\[ I_{ll} \]

因此 normal-light GT 那条支路消失。

直接：

\[ z_T\sim\mathcal N(0,I) \]

然后利用：

\[ z_{ll},S,M \]

逐步逆扩散：

\[ z_T\rightarrow\cdots\rightarrow z_0 \]

最后 decoder：

\[ z_0 \xrightarrow{\mathcal D} I_{out} \]

论文训练采用 1000 个 diffusion timesteps，而推理时用 25 步采样加速。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 二十七、为什么这篇论文叫“Generative Perceptual Priors”，现在就清楚了

这里其实有两种“生成”。

第一层：

## Generative VLM

LLaVA 根据 image + instruction：

\[ p(token\mid image,prompt) \]

生成语言 token。

作者不直接取文本，而是从：

\[ P(good) \]

和：

\[ P(poor) \]

中提取：

\[ S,M \]

。 [arXiv](https://arxiv.org/html/2412.20916v1)

第二层：

## Generative Diffusion Model

真正生成正常光 latent：

\[ z_T \rightarrow z_0 \]

再生成：

\[ I_{out} \]

。 [arXiv](https://arxiv.org/html/2412.20916v1)

所以这篇论文不是：

\[ \text{VLM生成图片} \]

而是：

\[ \boxed{ \text{VLM负责感知} + \text{Diffusion负责生成} } \]

这句话可以作为整篇论文的核心记忆点。

---

# 二十八、为什么作者不用 VLM semantic embedding？

这是论文和很多 VLM restoration 方法的主要区别。

传统方法可能：

\[ I \rightarrow CLIP/LLaVA \rightarrow \text{semantic embedding} \]

比如：

> person, tree, building

作者认为低照图像本来就严重退化，所以让 VLM准确识别语义内容并不可靠；而且传统语义先验常依赖预定义类别。相反，他们让 VLM判断：

\[ \boxed{\text{contrast / visibility / sharpness}} \]

这些和增强任务本身直接相关的 low-level attributes。 [arXiv](https://arxiv.org/html/2412.20916v1)

因此这篇论文其实是：

\[ \text{VLM} \]

从：

\[ \text{Semantic Understanding} \]

转向：

\[ \boxed{\text{Low-level Perceptual Assessment}} \]

这是它真正新颖的地方。

---

# 二十九、为什么还必须有 Diffusion？

你可能会问：

> 既然 \(S,M\) 已经知道哪里暗了，那普通 CNN 提亮不就行了吗？

作者的目标不只是：

\[ \text{brightness correction} \]

而是：

\[ \boxed{ \text{realistic + visually attractive enhancement} } \]

因此选择 diffusion model 学习 normal-light image distribution，让恢复不仅像素正确，还具有更好的 perceptual quality。论文实验中 GPP-LLIE 在 FID、LPIPS、DISTS 和 PSNR 上均表现出优势；尤其作者强调 FID/LPIPS 的改善说明了生成感知质量方面的提升。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 三十、消融实验反过来也验证了 Figure 3 的设计

论文的消融非常值得和网络图一起看。

完整模型：

\[ FID=36.73 \]

去掉 local prior / LPP-Attn 后：

\[ FID=49.83 \]

说明：

\[ \boxed{M+\text{LPP-Attn}} \]

非常重要。 [arXiv](https://arxiv.org/html/2412.20916v1)

如果把 LPP-Attn 替换为 StableSR 风格的 spatial feature transform，效果也不如作者提出的 attention 注入方案。 [arXiv](https://arxiv.org/html/2412.20916v1)

同时，去掉 global perceptual score，或者简单把 global score 直接加到 noisy latent 上，也明显变差。这说明：

\[ \boxed{\text{先验是什么}} \]

和：

\[ \boxed{\text{先验怎么注入}} \]

两者都重要。 [arXiv](https://arxiv.org/html/2412.20916v1)

---

# 三十一、最后把整个网络用一句“人话流程”串起来

给网络一张很暗的照片。

### 第一步：VLM先看

LLaVA说：

> 整体 visibility 很差，contrast 一般，sharpness 还可以。

得到：

\[ S \]

再逐块观察：

> 左下特别差，中央一般，右上灯附近已经很好。

得到：

\[ M \]

---

### 第二步：Encoder看原始内容

得到：

\[ z_{ll} \]

它告诉模型：

> 图里原本的结构和内容是这些，别乱生成。

---

### 第三步：Diffusion从噪声开始

\[ z_T\sim\mathcal N(0,I) \]

---

### 第四步：每个 Transformer Block 都问三个问题

**原图是什么？**

\[ z_{ll} \]

→ Concat-and-Remove。

**整体应该怎么调整？**

\[ S \]

→ GPP-LN。

**哪里应该重点调整？**

\[ M \]

→ LPP-Attn。

---

### 第五步

反复：

\[ z_T \rightarrow z_{T-1} \rightarrow \cdots \rightarrow z_0 \]

最后：

\[ z_0 \xrightarrow{\mathcal D} \hat I_{out} \]

---

# 三十二、你现在应该如何理解这篇文章

如果把所有复杂名字都去掉，它其实提出了一套很清晰的思想：

\[ \boxed{ \text{先“诊断”低照度图像，再根据诊断结果生成增强图像} } \]

其中：

\[ \text{VLM}=\text{诊断医生} \]

判断：

\[ \text{哪里有问题、问题多严重} \]

而：

\[ \text{Diffusion Transformer}=\text{治疗者} \]

负责：

\[ \text{真正恢复图像} \]

并且两种诊断结果分别作用于两个不同尺度：

\[ \boxed{ S \rightarrow \text{Global adaptation} } \]\[ \boxed{ M \rightarrow \text{Local adaptation} } \]

因此我认为 **Figure 2 + Figure 3 应该连在一起看，而不能把它们当成两张独立网络图**：

\[ \boxed{ I_{ll} \overset{\text{Fig.2}}{\longrightarrow} (S,M) \overset{\text{Fig.3}}{\longrightarrow} \text{GPP-Guided Diffusion} \longrightarrow I_{out} } \]

这就是 GPP-LLIE 的完整闭环。 [arXiv](https://arxiv.org/html/2412.20916v1)

如果我们继续往下精读，**下一步最值得单独拆的是 Figure 3 右侧的 GPP-LLIE Block**。尤其是 **GPP-LN 中 \(S,t,\gamma,\beta\) 到底怎样作用，以及 LPP-Attn 中 \(Q/K/V\) 的维度怎么变、为什么它算的是 channel attention 而不是 spatial attention**。这部分我可以直接按张量尺寸一步一步走一遍，会比只看论文图更容易彻底理解。