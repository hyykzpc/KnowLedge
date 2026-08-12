可以。这篇 **《Bilevel Layer-Positioning LoRA for Real Image Dehazing》** 是一篇 **CVPR 2026** 工作，作者为 Yan Zhang、Long Ma、Yuxin Feng 等。它非常适合你现在关注的“**VLM + 图像恢复/去雾**”方向：论文确实使用了 **CLIP 作为视觉语言模型提供无监督语义监督**，但它的去雾主体仍然是传统图像恢复网络；真正的方法创新集中在 **CLIP-based H2C Loss + 自动寻找 LoRA 注入层的 BiLaLoRA**。[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Bilevel_Layer-Positioning_LoRA_for_Real_Image_Dehazing_CVPR_2026_paper.html?utm_source=chatgpt.com)

---

# 一、先用一句话理解这篇论文

这篇论文解决的是：

> **一个在合成雾数据上训练好的去雾网络，怎样只利用没有 GT 的真实雾图像，低成本地适配到真实世界？**

作者给了两个答案：

\[ \boxed{ \text{BiLaLoRA} = \underbrace{\text{H2C Loss}}_{\text{告诉模型往哪里优化}} + \underbrace{\text{Bilevel Layer-Positioning LoRA}}_{\text{告诉模型应该改哪些层}} } \]

这两个模块其实分别对应真实图像去雾的两个核心问题。现实场景几乎拿不到同场景的清晰 GT，因此普通 \(L_1/L_2\) 监督很难用；与此同时，如果为了适配一个真实场景就把整个网络 Full Fine-Tuning 一遍，训练和存储成本又很大。论文正是围绕这两点展开。[arXiv](https://arxiv.org/html/2603.10872)

---

# 二、整篇论文的总体流程

先不要急着看双层优化公式，整体流程其实非常简单：

\[ \text{Synthetic Hazy/Clear} \]\[ \downarrow \]\[ \text{预训练普通去雾网络} \]\[ \downarrow \]\[ \boxed{\text{Freeze Backbone}} \]

然后到了真实域：

\[ I_{\text{real haze}} \rightarrow \text{Frozen Backbone + LoRA} \rightarrow I_{\text{out}} \]

但这里没有真实清晰图像 \(I_{\text{GT}}\)。

于是作者引入：

\[ \boxed{\text{CLIP H2C Loss}} \]

来判断输出到底有没有沿着：

\[ \text{hazy} \rightarrow \text{clear} \]

这个语义方向变化。

与此同时，作者不是把 LoRA 随便插到所有层，也不是人工规定“encoder 后几层”，而是给候选模块都加入一个可学习的：

\[ \alpha_i \]

让网络自己寻找：

> **真实域 domain gap 到底主要集中在哪些层？**

最后选择 Top-\(k\) 层，只保留这些 LoRA，再继续微调。论文实验中使用的是 **Top-3**。[arXiv](https://arxiv.org/html/2603.10872)

所以你可以把全文理解成两个阶段：

\[ \boxed{ \text{Stage 1: 找层} \quad\rightarrow\quad \text{Stage 2: 调这些层} } \]

---

# 三、第一个关键创新：H2C Loss

这一部分是论文和 **VLM** 关系最直接的地方。

## 3.1 为什么普通监督不能用？

传统监督去雾：

\[ I_{\text{hazy}} \xrightarrow{F_\theta} I_{\text{out}} \]

然后：

\[ L=\|I_{\text{out}}-I_{\text{GT}}\|_1 \]

问题是现实场景没有：

\[ I_{\text{GT}} \]

例如今天拍了一张有雾的城市：

> 你几乎不可能同时获得“完全同一个时间、同一个机位、同一个照明，但是空气中没有雾”的照片。

所以作者换了一个问题：

> 人为什么可以看出一张图“更清晰了”？

并不需要知道每一个像素的 GT，而是可以理解：

\[ \text{“这张图有雾”} \]

和

\[ \text{“这张图是清晰照片”} \]

之间存在一个**语义上的变化方向**。

CLIP 恰好提供了这个公共图文 latent space。[arXiv](https://arxiv.org/html/2603.10872)

---

# 四、H2C 最核心的思想不是“匹配 clear”，而是“匹配方向”

这是这篇论文第一个非常值得注意的地方。

作者准备两个文本 prompt：

负向：

\[ T_{\text{neg}} = \text{``a photo with haze''} \]

正向：

\[ T_{\text{pos}} = \text{``a clear photo''} \]

通过 CLIP Text Encoder 得到：

\[ E_T(T_{\text{neg}}),\qquad E_T(T_{\text{pos}}) \]

然后构造一个**文本方向向量**：

\[ \boxed{ \Delta T_{\text{text}} = T_{\text{pos}}-T_{\text{neg}} } \]

它代表：

\[ \boxed{\text{有雾}\rightarrow\text{清晰}} \]

的语义变化方向。

代码中也确实直接使用了这两个 prompt，并使用 CLIP ViT-B/32 编码文本。[GitHub](https://github.com/YanZhang-zy/BiLaLoRA/blob/main/BiLaLoRA_Train.py)

---

## 4.1 图像这边怎么做？

真实有雾输入：

\[ I_{\text{in}} \]

经过当前去雾模型：

\[ I_{\text{out}}=F(I_{\text{in}}) \]

分别送进 CLIP Image Encoder：

\[ V_{\text{in}}=E_I(I_{\text{in}}) \]\[ V_{\text{out}}=E_I(I_{\text{out}}) \]

于是可以得到模型实际造成的语义变化：

\[ \boxed{ \Delta V_{\text{img}} = V_{\text{out}}-V_{\text{in}} } \]

也就是说：

> 输入图片在 CLIP 空间里原本位于 \(V_{\text{in}}\)，经过你的去雾网络以后跑到了 \(V_{\text{out}}\)。

它移动的方向就是：

\[ V_{\text{out}}-V_{\text{in}} \]

作者希望这个方向与：

\[ T_{\text{pos}}-T_{\text{neg}} \]

尽可能一致。[arXiv](https://arxiv.org/html/2603.10872)

所以：

\[ \boxed{ L_{\mathrm{H2C}} = 1- \frac{ \Delta V_{\mathrm{img}}^\top \Delta T_{\mathrm{text}} }{ \|\Delta V_{\mathrm{img}}\|_2 \|\Delta T_{\mathrm{text}}\|_2 } } \]

本质就是：

\[ L_{\mathrm{H2C}} = 1-\cos (\Delta V_{\text{img}}, \Delta T_{\text{text}}) \]

当两个方向一致：

\[ \cos\theta\rightarrow1 \]

于是：

\[ L_{\mathrm{H2C}}\rightarrow0 \]

代码实现和这个公式完全一致：它分别计算输出/输入的 CLIP image embedding，再计算二者差值，与正负文本 embedding 的差值做 cosine similarity。[GitHub](https://github.com/YanZhang-zy/BiLaLoRA/blob/main/BiLaLoRA_Train.py)

---

# 五、这里为什么一定要“方向”，而不是直接让输出接近 “a clear photo”？

这是 H2C 最漂亮的一点。

假设直接做：

\[ L = 1- \cos (V_{\text{out}},T_{\text{clear}}) \]

那么所有图片都会被推动到：

\[ T_{\text{clear}} \]

附近。

这实际上只告诉模型：

> “你最后看起来要像一个 clear image。”

却没有强调：

> “你应该从**当前这张图**出发，只做 haze→clear 的变化。”

这样就可能改变原图的颜色、内容和风格。

而 H2C 约束的是：

\[ \boxed{ V_{\text{out}}-V_{\text{in}} } \]

也就是说它不要求：

\[ V_{\text{out}}=某一个固定位置 \]

而要求：

\[ \boxed{\text{你移动的方向正确}} \]

非常类似于：

> 我不规定你的终点在哪里，但规定你的移动方向必须朝“去雾”走。

论文消融也验证了这一点：只保留 positive guidance 会产生明显颜色失真；只做 negative haze suppression 又容易出现过度去雾。正负文本共同定义一个 direction 后效果最好。[arXiv](https://arxiv.org/html/2603.10872)

---

# 六、所以这篇论文到底是不是“VLM 去雾”？

**是，但要准确表述。**

它不是：

\[ \text{VLM} \rightarrow \text{直接输出清晰图} \]

也不是：

\[ \text{LLM/VLM} \rightarrow \text{控制恢复网络结构} \]

而是：

\[ \boxed{ \text{Frozen CLIP} \rightarrow \text{Cross-modal Semantic Supervisor} } \]

即：

\[ \text{CLIP/VLM} \]

负责判断：

\[ \text{当前恢复方向是否符合“haze → clear”} \]

真正生成图像的是 DEA 等去雾网络。

所以如果你以后汇报这篇论文，可以说：

> **The VLM is employed as a frozen cross-modal semantic prior rather than a restoration backbone.**

这一点和我们之前看的那些“利用 VLM 生成 perceptual prior 再注入 restoration network”的工作并不完全相同。[arXiv](https://arxiv.org/html/2603.10872)

---

# 七、第二个创新：为什么需要 Layer-Positioning LoRA？

接下来进入论文真正题目中的：

> **Bilevel Layer-Positioning LoRA**

先从普通 LoRA 开始。

对于网络某一层参数：

\[ W_0\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}} \]

Full Fine-Tuning 是直接学习：

\[ W_0\rightarrow W_0+\Delta W \]

而 LoRA 假设：

\[ \Delta W \]

可以写成一个低秩分解：

\[ \Delta W=BA \]

其中：

\[ A\in\mathbb R^{r\times d_{\text{in}}}, \]\[ B\in\mathbb R^{d_{\text{out}}\times r}, \]

并且：

\[ r\ll d_{\text{in}},d_{\text{out}}. \]

因此：

\[ W' = W_0+\gamma BA \]

训练时：

\[ W_0\quad\text{freeze} \]

只训练：

\[ A,B \]

所以参数量大幅减少。论文使用 \(r=8\)，最终实验中的 LoRA scaling factor 为 \(\gamma=2\)。[arXiv](https://arxiv.org/html/2603.10872)

---

# 八、但是论文认为普通 LoRA 还有一个问题

问题不是：

> LoRA 要不要用？

而是：

> **LoRA 应该放在哪里？**

例如一个 Encoder 有：

\[ L_1,L_2,L_3,L_4,L_5,L_6,\dots \]

你可以：

\[ L_1+\text{LoRA} \]

也可以：

\[ L_6+\text{LoRA} \]

甚至所有层都插。

问题是不同网络、不同 domain gap，需要改变的层并不一样。

---

# 九、Figure 3 是理解整个 BiLaLoRA 动机最重要的图

论文先做了一个非常关键的实验。

他们把 MSBDN 和 DEA 在真实域上 Full Fine-Tuning，然后把已经适配后的某些模块“移植回”原始网络，观察哪个模块能带来最大的 MUSIQ 提升。[arXiv](https://arxiv.org/pdf/2603.10872v1)

Figure 3 从三个粒度分析：

\[ \text{Network Level} \rightarrow \text{Block Level} \rightarrow \text{Layer Level} \]

首先：

\[ \text{Encoder contribution} \gg \text{Decoder contribution} \]

说明 synthetic→real 的 domain gap 主要需要 encoder 侧适配。

然后继续看 encoder：

\[ B_1,B_2,\dots,B_4 \]

发现最后一个 block：

\[ B_4 \]

贡献最大。

但再继续进入 \(B_4\) 内部：

\[ L_1,L_2,L_3,L_4 \]

事情变了。

对于 MSBDN：

\[ L_3 \]

可能更重要。

而 DEA：

\[ L_4 \]

更加关键。

也就是说：

\[ \boxed{ \text{重要 Block 可能具有共性，但具体 bottleneck layer 与模型结构有关} } \]

论文因此提出一个重要观点：

\[ \boxed{ \text{Domain Gap 的性能瓶颈层不是固定的} } \]

这就是 **Layer-Positioning** 出现的真正原因。[arXiv](https://arxiv.org/pdf/2603.10872v1)

---

# 十、Layer-Positioning 到底怎么实现？

作者为每一个候选 LoRA 位置加上一个可学习参数：

\[ \alpha_i \]

然后：

\[ \boxed{ W'_i = W_i + \sigma(\alpha_i)\gamma\Delta W_i } \]

其中：

\[ \sigma(\alpha_i)\in(0,1) \]

相当于一个 gate。

如果：

\[ \sigma(\alpha_i)\approx0 \]

说明：

> 这一层的 LoRA 基本没必要。

如果：

\[ \sigma(\alpha_i)\approx1 \]

说明：

> 这一层非常需要真实域适配。

于是原本离散的问题：

\[ \text{“这一层选还是不选？”} \]

被连续化成：

\[ \boxed{ \alpha_i\in(0,1) } \]

因此可以直接用梯度下降优化。论文把它视为一种 differentiable architecture search。[arXiv](https://arxiv.org/html/2603.10872)

---

# 十一、官方代码把这个过程做得更加具体

对于论文主要实验使用的 DEA，代码中的 LoRA 搜索空间包含：

\[ 4+4+8=16 \]

个候选 block：

\[ \text{down\_level1\_block1}\sim\text{block4} \]\[ \text{down\_level2\_block1}\sim\text{block4} \]

以及

\[ \text{level3\_block1}\sim\text{block8}. \]

每个候选 submodule 拥有一个共享的 \(\alpha\)。而该 submodule 内部的 Conv2d 会被替换为 LoRA Conv；同一 submodule 内的这些卷积共享相同的 gate：

\[ \sigma(\alpha_{\text{submodule}}) \]

因此严格来说，他们搜索的是：

\[ \boxed{\text{哪个网络 submodule/block 应该启用 LoRA}} \]

而不是给网络里的每一个单独卷积都独立学习一个 layer score。[GitHub](https://github.com/YanZhang-zy/BiLaLoRA/blob/main/BiLaLoRA_Train.py)

这点主论文写得比较抽象，结合代码会清楚很多。

---

# 十二、那为什么不能直接同时训练 \(\alpha\) 和 LoRA？

这就是标题里最重要的：

\[ \boxed{\text{Bilevel}} \]

假设普通 joint optimization：

\[ \min_{\alpha,\omega} L_{\text{train}}(\alpha,\omega) \]

其中：

\[ \omega=\{A,B\} \]

是 LoRA 参数。

那么：

\[ \alpha \]

和：

\[ \omega \]

同时在同一个 training set 上优化。

这里容易发生一个问题：

> \(\alpha\) 找到的是“最容易把训练集 loss 降低的层”，而不一定是“真实泛化最好的层”。

作者实验也发现 naïve joint learning 的性能逊于 bilevel optimization。[arXiv](https://arxiv.org/html/2603.10872)

---

# 十三、BiLaLoRA 的核心思想：Train LoRA，Validation 选层

作者把问题写成：

\[ \boxed{ \min_{\boldsymbol{\alpha}} \varphi \big( \boldsymbol{\omega}^{*}(\boldsymbol{\alpha}), \boldsymbol{\alpha} \big) } \]

subject to：

\[ \boxed{ \boldsymbol{\omega}^{*}(\boldsymbol{\alpha}) \in \arg\min_{\boldsymbol{\omega}} \psi(\boldsymbol{\omega},\boldsymbol{\alpha}) } \]

这就是一个标准的 bilevel optimization。

直观上：

### 内层

固定当前层选择：

\[ \alpha \]

训练 LoRA：

\[ \boxed{ \omega^*(\alpha) = \arg\min_{\omega} L_{\text{train}} } \]

问题是：

> **假如选择这些层，那么这些层里的 LoRA 权重怎样训练最好？**

### 外层

在 LoRA 已经尽量训练好的基础上，观察 validation：

\[ \boxed{ \min_\alpha L_{\text{val}} ( \omega^*(\alpha),\alpha ) } \]

问题变成：

> **在这些 LoRA 都训练好的情况下，到底选哪些层泛化最好？**

官方代码也明确将真实数据分成两半：一半用于 LoRA weight optimization，另一半用于 architecture parameter \(\alpha\) 的更新；外层 loss 和内层 loss 都使用 H2C，只是使用的数据不同。[GitHub](https://github.com/YanZhang-zy/BiLaLoRA/blob/main/BiLaLoRA_Train.py)

---

# 十四、这是理解 Bilevel 的最好类比

可以把：

\[ \omega \]

理解成：

> **学生**

把：

\[ \alpha \]

理解成：

> **课程选择器**

内层：

> 学生在训练集上学习。

外层：

> 看学生在验证集上的成绩，根据泛化表现决定哪些课程真正有用。

如果课程选择和学生训练全部根据训练集成绩优化：

\[ \alpha,\omega \rightarrow L_{\text{train}} \]

那么课程选择器很容易：

\[ \boxed{\text{过拟合训练集}} \]

而 bilevel 做的是：

\[ \omega \rightarrow \text{Train} \]\[ \alpha \rightarrow \text{Validation} \]

所以 Layer Positioning 的目标不是：

> 找“训练 loss 最敏感的层”。

而更接近：

> **找“真正有助于目标域泛化的层”。**

这正是论文在 §6.2 强调 bilevel modeling 优于 naïve joint learning 的原因。[arXiv](https://arxiv.org/html/2603.10872)

---

# 十五、公式 4–7 到底在算什么？

因为：

\[ \omega^*=\omega^*(\alpha) \]

所以外层：

\[ \varphi(\omega^*(\alpha),\alpha) \]

对 \(\alpha\) 求导时，不能只算：

\[ \frac{\partial\varphi}{\partial\alpha} \]

还要考虑：

\[ \boxed{ \alpha变化 \rightarrow \omega^*变化 \rightarrow Validation Loss变化 } \]

因此：

\[ \mathbf g_\alpha = \nabla_\alpha\varphi + \left( \nabla_\alpha \omega^* \right)^T \nabla_\omega\varphi. \]

第二项就是：

\[ \boxed{\text{implicit gradient}} \]

也就是所谓的 hypergradient。[arXiv](https://arxiv.org/html/2603.10872)

---

# 十六、为什么论文后面突然出现 Hessian？

因为要计算：

\[ \nabla_\alpha\omega^*(\alpha) \]

根据内层最优条件：

\[ \nabla_\omega f(\omega^*,\alpha)=0 \]

利用 implicit function theorem，可以推出：

\[ \nabla_\alpha\omega^* = - \left[ \nabla_{\omega\omega}^{2}f \right]^{-1} \nabla_{\omega\alpha}^{2}f. \]

这里出现了：

\[ H^{-1} = \left[ \nabla_{\omega\omega}^{2}f \right]^{-1} \]

也就是 Hessian inverse。

但是神经网络里：

\[ \omega \]

有成千上万个参数。

直接构造：

\[ H\in\mathbb R^{N\times N} \]

再求逆，成本太高。[arXiv](https://arxiv.org/html/2603.10872)

---

# 十七、BiLaLoRA 的技巧：把二阶问题近似成一阶问题

作者使用 rank-one outer-product approximation：

\[ \nabla_{\omega\omega}^2f \approx \nabla_\omega f \nabla_\omega f^T \]

以及：

\[ \nabla_{\omega\alpha}^2f \approx \nabla_\omega f \nabla_\alpha f^T. \]

最终得到：

\[ \boxed{ \mathbf g_\alpha \approx \nabla_\alpha\varphi - \frac{ \nabla_\omega\varphi^T \nabla_\omega f }{ \|\nabla_\omega f\|^2 } \nabla_\alpha f } \]

这样就不需要显式构造 Hessian，只需要：

\[ \boxed{\text{First-order Gradients}} \]

因此层搜索本身不会变成一个特别昂贵的二阶优化问题。[arXiv](https://arxiv.org/html/2603.10872)

---

# 十八、Algorithm 1：真正训练时发生什么？

这个算法可以拆得非常简单。

第一阶段叫：

\[ \boxed{\text{Bilevel Layer-Positioning Stage}} \]

此时所有候选位置都拥有：

\[ \text{LoRA}_i+\alpha_i. \]

反复执行：

\[ \text{Validation Data} \rightarrow \text{更新 }\alpha \]

然后：

\[ \text{Training Data} \rightarrow \text{更新 LoRA }\omega. \]

等搜索阶段结束后：

\[ \alpha_1,\alpha_2,\dots,\alpha_N \]

已经形成一个排名。[arXiv](https://arxiv.org/html/2603.10872)

然后：

\[ \boxed{ \alpha^* = \operatorname{TopK}(\alpha,k) } \]

论文选择：

\[ k=3. \]

进入第二阶段：

\[ \boxed{\text{LoRA Fine-Tuning Stage}} \]

其余候选 LoRA 全部丢掉，只留下 Top-3：

\[ \text{LoRA}_{i_1}, \text{LoRA}_{i_2}, \text{LoRA}_{i_3}. \]

然后继续：

\[ \omega \leftarrow \omega-\eta\nabla_\omega L_{\mathrm{H2C}}. \]

官方实现甚至会根据 Top-\(k\) gate 重新构建 backbone，只在被选中的 submodule 中重新插入 LoRA。[GitHub](https://github.com/YanZhang-zy/BiLaLoRA/blob/main/BiLaLoRA_Train.py)

所以 **BiLaLoRA 并不是推理时动态选择层**。

而是：

\[ \boxed{ \text{训练阶段搜索层} \rightarrow \text{固定 Top-3} \rightarrow \text{最终模型只保留 Top-3 LoRA} } \]

这一点很重要。

---

# 十九、完整训练流程现在可以串起来了

把论文全部模块放在一起：

\[ \boxed{ \text{Synthetic Data} } \]\[ \downarrow \]\[ \text{DEA Pre-training with }L_1 \]\[ \downarrow \]\[ \boxed{\text{Freeze DEA}} \]

真实域：

\[ I_{\mathrm{hazy}} \]

一方面经过：

\[ \text{DEA + Candidate LoRAs} \rightarrow I_{\mathrm{out}} \]

另一方面：

\[ I_{\mathrm{hazy}},I_{\mathrm{out}} \rightarrow \text{CLIP Image Encoder} \]

文本：

\[ \text{``a photo with haze''} \]\[ \text{``a clear photo''} \]

经过：

\[ \text{CLIP Text Encoder} \]

得到：

\[ L_{\mathrm{H2C}} \]

然后：

\[ \boxed{ \begin{aligned} \text{Train set}&\rightarrow\omega\\ \text{Validation set}&\rightarrow\alpha \end{aligned} } \]

搜索结束：

\[ \operatorname{Top3}(\alpha) \]

最后只微调：

\[ \boxed{\text{Top-3 LoRA}} \]

这就是整篇论文。

---

# 二十、实验结果说明了什么？

论文以 DEA 为主要 baseline。DEA 先在 THaze 上使用 \(L_1\) 训练；真实域适配时使用 500 张 daytime real hazy images 和 100 张 NHRW nighttime images，并分别构建 daytime 和 nighttime adapter。LoRA 设置为 \(r=8\)，最终选择 Top-3 layers。[arXiv](https://arxiv.org/html/2603.10872)

Full Fine-Tuning 与 BiLaLoRA 的比较尤其直观：

\[ 4.215\text{ h} \rightarrow 0.940\text{ h} \]

训练时间下降约：

\[ \boxed{77.7\%} \]

而 MUSIQ：

\[ 64.43 \rightarrow 64.40 \]

几乎没有下降；推理 FLOPs 和 runtime 仅有约 1% 左右的增加。[arXiv](https://arxiv.org/html/2603.10872)

在 RTTS、URHI 和 Fattal 三个真实去雾数据集的平均结果上，BiLaLoRA 达到：

\[ \text{FADE}=0.638, \]\[ \text{BIQME}=0.611, \]\[ \text{Entropy}=7.572, \]\[ \text{MUSIQ}=64.40. \]

论文的整体指标基本位于比较方法的第一或第二名。[arXiv](https://arxiv.org/html/2603.10872)

---

# 二十一、消融实验非常关键

把 Table 3 简化来看。

原始模型大约：

\[ \text{MUSIQ}=62.05 \]

加入完整 positive+negative H2C，但还没有 layer positioning：

\[ 63.31 \]

再加入 naïve joint layer search：

\[ 64.07 \]

最终使用：

\[ \boxed{ \text{H2C}+\text{Bilevel Layer-Positioning} } \]

达到：

\[ \boxed{64.40} \]

同时 FADE 从：

\[ 1.018 \]

下降到：

\[ 0.638. \]

也就是说性能提升不是只来自 LoRA，更不是只来自 CLIP，而是：

\[ \boxed{ \text{H2C 提供正确的目标域监督} + \text{Bi-level 搜索找到正确的适配位置} } \]

两者是互补的。[arXiv](https://arxiv.org/html/2603.10872)

---

# 二十二、Figure 13 为什么最后选择 3 层？

作者继续改变：

\[ k=1,2,3,4,5,6 \]

发现随着 LoRA 层数增加：

\[ 1\rightarrow2\rightarrow3 \]

性能明显增加。

达到：

\[ \boxed{k=3} \]

以后，继续：

\[ 3\rightarrow4\rightarrow5\rightarrow6 \]

收益基本趋于饱和，而参数量继续增加。Figure 13 中虚线圈出来的最优点因此就是：

\[ \boxed{3\text{ LoRA layers}} \]

这说明 real-domain adaptation 的 domain gap 很可能高度集中在少数 bottleneck layers 中，而不是需要把整个网络重新训练一遍。[arXiv](https://arxiv.org/pdf/2603.10872v1)

---

# 二十三、我认为这篇论文最值得你关注的三个思想

最重要的不是“LoRA 用到了去雾”，而是三个更一般化的思想。

第一，**把 Image Restoration 从绝对目标变成了语义方向学习**：

\[ \boxed{ I_{\mathrm{out}}\neq \text{某个固定 clear target} } \]

而是：

\[ \boxed{ I_{\mathrm{in}}\rightarrow I_{\mathrm{out}} \text{ 的变化方向应该等于} \text{haze}\rightarrow\text{clear} } \]

这对于真实无配对数据非常适合。

第二，**domain adaptation 不一定需要修改整个网络**。Figure 3 明确显示 synthetic-real gap 集中在特定 encoder block/layer，因此应该：

\[ \boxed{\text{Find bottleneck}\rightarrow\text{Adapt bottleneck}} \]

而不是：

\[ \text{Adapt Everything}. \]

第三，**“训练参数”和“选择结构”是两个不同问题**：

\[ \omega:\text{How to adapt} \]\[ \alpha:\text{Where to adapt} \]

因此作者才使用：

\[ \boxed{\text{Bilevel Optimization}} \]

把二者分离。

---

# 二十四、这篇论文也有几个需要批判性看的地方

首先，CLIP 提供的是比较粗粒度的高层语义监督。它擅长区分“hazy / clear”，但对于低层视觉里的局部 texture、halo、轻微 color shift 等问题并不一定敏感，所以 H2C 更像是 **perceptual direction prior**，而不是严格的 image fidelity supervision。论文自己也通过 positive-only / negative-only 的颜色偏移和过度去雾现象间接说明了这一点。[arXiv](https://arxiv.org/html/2603.10872)

其次，所谓 “model-agnostic layer positioning” 并不意味着完全无需设计 search space。以官方 DEA 代码为例，作者仍然人为给出了 16 个候选 submodules，只是 BiLaLoRA 自动决定这 16 个候选中的哪 3 个最重要。[GitHub](https://github.com/YanZhang-zy/BiLaLoRA/blob/main/BiLaLoRA_Train.py)

还有一点值得注意：它虽然避免了针对每个域 Full Fine-Tuning，但不同场景仍然可以训练不同 adapter。论文就分别训练了 daytime adapter 和 nighttime adapter。因此它更准确的定位是：

\[ \boxed{\text{cheap domain-specific adaptation}} \]

而不是：

[  
\text{one adapter universally handles every domain}.  
] [arXiv](https://arxiv.org/html/2603.10872)

---

# 二十五、最后给你一个“记住这篇论文”的版本

以后看到 **BiLaLoRA**，你可以直接在脑子里想成：

\[ \boxed{ \text{一个合成数据训练好的去雾模型} } \]

真实图没有 GT，所以：

\[ \boxed{ \text{CLIP H2C} : \text{告诉模型沿 haze}\rightarrow\text{clear 方向优化} } \]

但又不想 Full Fine-Tuning，所以：

\[ \boxed{ \text{LoRA} : \text{只改变少量低秩参数} } \]

又不知道 LoRA 应该插在哪里，所以：

\[ \boxed{ \alpha : \text{给不同 layer 打重要性分数} } \]

但直接和 LoRA 一起在训练集优化容易过拟合，因此：

\[ \boxed{ \text{Bilevel} : \text{Train 上学 LoRA，Validation 上选 Layer} } \]

最后：

\[ \boxed{ \text{Top-3 Layers} \rightarrow \text{LoRA Fine-Tuning} } \]

这就是整篇论文最核心的逻辑链。

**下一步精读我建议直接进入 Figure 3 + Algorithm 1。** Figure 3 回答的是“作者为什么相信只需要适配少数层”，Algorithm 1 回答的是“\(\alpha\) 和 LoRA 到底按照什么顺序更新”；把这两个彻底弄懂之后，公式 (3)–(7) 就会非常自然。