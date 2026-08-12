可以。这篇 **CVPR 2026 的 Multinex: Lightweight Low-light Image Enhancement via Multi-prior Retinex** 很值得精读，因为它的思路和传统 Retinex 网络有一个很关键的区别：**它不执着于把低照度图像真正分解成 illumination 和 reflectance，而是把 Retinex 当成“如何修改图像”的结构先验。** 最终做到完整版约 **44.7K 参数**，Nano 版本只有约 **0.7K 参数**。[arXiv](https://arxiv.org/html/2604.10359v1)

我建议先建立整篇论文的“地图”，然后下一步重点精读 **Figure 2**，因为 Figure 2 基本就是整篇文章。

---

# 1. 这篇论文到底想解决什么？

作者认为当前 LLIE 有两个问题。

第一个是**亮度和颜色没有真正解耦**。

最普通的 RGB 网络直接学习

\[ I_{\text{low}}\rightarrow I_{\text{normal}} \]

网络必须自己同时搞清楚：

- 哪里需要提亮；
- 提亮多少；
- RGB 三个通道怎么变；
- 什么是物体本来的颜色；
- 哪些颜色变化只是光照造成的。

这对几十 M 参数的网络也许还能学，但对一个几十 K 参数的小网络来说太难了。

传统 Retinex 尝试解决这个问题：

\[ I=L\odot R \]

其中：

\[ L=\text{illumination} \]\[ R=\text{reflectance} \]

理论上，\(L\) 管亮度，\(R\) 管物体本身属性。

但很多深度 Retinex 方法依然是在 **RGB 信息高度耦合的条件下，让网络自己估计 L 和 R**，所以问题并没有完全消失。作者还认为依赖单一 RGB、YUV、HSV 或学习得到的颜色空间，各自都会有一些耦合或颜色稳定性问题。[arXiv](https://arxiv.org/html/2604.10359v1)

第二个问题就是：

> **网络太大。**

于是 Multinex 的核心问题其实是：

> 如果我只有几十 K，甚至几百个参数，还能不能做高质量低照度增强？

作者的答案不是“设计一个更厉害的小 Transformer”，而是：

> **不要让网络学习那些我们已经知道怎么计算的东西。**

这句话实际上就是整篇文章的灵魂。

---

# 2. Multinex 的核心思想

我把整个 Multinex 压缩成：

\[ \boxed{ \text{人工解析先验} + \text{极小神经网络融合} + \text{Retinex式残差} } \]

它不是把 RGB 直接扔进大网络。

而是先从 RGB **免费计算**出两组特征：

\[ I \rightarrow \begin{cases} \mathcal S_L & \text{亮度先验}\\ \mathcal S_R & \text{颜色先验} \end{cases} \]

然后两个很小的网络分别得到：

\[ \mathcal S_L \xrightarrow{f_L} \Delta_L \]

和

\[ \mathcal S_R \xrightarrow{f_R} \Delta_R. \]

最后：

\[ \boxed{ \hat I=I+\Delta_L\odot\Delta_R } \]

这就是它所谓的 **Multi-prior Retinex**。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 3. 最关键的创新：它其实“不做传统 Retinex 分解”

这一点一定要区分清楚。

传统 Retinex 通常希望得到：

\[ I=L\odot R \]

或者根据低照度图像估计：

\[ L_{\text{low}}, R_{\text{low}} \]

然后调整 illumination，再重新生成结果。

但 Multinex 认为：

> 为什么非得准确恢复 \(L\) 和 \(R\)？

这是一个很难、而且本身欠定的问题。

所以作者把问题改成：

\[ \hat I = I+\Delta_I. \]

也就是说：

> 不重建整张正常图像，只预测“原图应该改多少”。

然后再借用 Retinex 的思想，把这个 correction 分成：

\[ \boxed{ \Delta_I=\Delta_L\odot\Delta_R } \]

因此：

\[ \boxed{ \hat I=I+\Delta_L\odot\Delta_R } \tag{1} \]

这里：

\[ \Delta_L\in\mathbb R^{H\times W\times1} \]

而：

\[ \Delta_R\in\mathbb R^{H\times W\times3}. \]

所以对 RGB 第 \(i\) 个通道：

\[ \hat I_i = I_i+ f_L(\cdot)\odot f_{R_i}(\cdot). \]

论文把前者称为 **Multinex Luminance**，后者称为 **Multinex Reflectance**。注意它们严格来说并不是经典 Retinex 意义上的真实 illumination / reflectance，而是**亮度 correction 和颜色 correction**。[arxiv.org](https://arxiv.org/html/2604.10359v1)

这个区别非常重要。

---

# 4. 为什么这个设计特别适合轻量网络？

因为原始图像：

\[ I \]

已经含有大量结构、纹理和边缘。

如果网络重新生成：

\[ \hat I=f(I), \]

那么一个 45K 参数模型实际上承担的是：

> “重新画一张正常曝光照片。”

这显然非常困难。

而现在：

\[ \hat I=I+\Delta, \]

网络只需要回答：

> **哪些地方应该改变？改变多少？**

原始纹理直接通过 skip/residual 保留下来。

作者明确把它描述成：

> enhancement 而不是 reconstruction。

这也是为什么它能够压到非常小。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 5. 但为什么还需要 \(\Delta_L\Delta_R\)？

如果只是：

\[ \hat I=I+f(I) \]

不就可以了吗？

关键就在这里。

普通 residual network 还是必须：

\[ RGB\rightarrow RGB\ residual \]

也就是说亮度 correction 和颜色 correction 依然混在一起。

Multinex 强制把它拆成两个任务：

\[ \boxed{ \text{亮度分支} \times \text{颜色分支} } \]

即：

\[ \Delta_I = \underbrace{\Delta_L}_{H\times W\times1} \odot \underbrace{\Delta_R}_{H\times W\times3}. \]

直觉上，可以把：

\[ \Delta_L \]

理解成：

> **这个像素需要多强的修改。**

而：

\[ \Delta_R \]

理解成：

> **这个修改应该怎样分配到 RGB 三个通道。**

例如一个很暗的红色物体。

亮度分支可能判断：

\[ \Delta_L(x,y)=0.7 \]

说明这里需要很明显地提亮。

颜色分支则可能输出：

\[ \Delta_R(x,y)=[0.8,0.3,0.2]. \]

于是：

\[ \Delta_I = 0.7[0.8,0.3,0.2] = [0.56,0.21,0.14]. \]

因此主要增加 R，同时适度增加 G/B。

这就是所谓 Retinex-like factorization 的意义。

---

# 6. 一个需要特别注意的理论细节

这里我认为读论文时一定不能被“Retinex”这个名字带偏。

数学上：

\[ \Delta_I=\Delta_L\odot\Delta_R \]

**并不能保证** \(\Delta_L\) 就是真实 illumination，\(\Delta_R\) 就是真实 reflectance。

因为这种分解本身并不唯一。例如：

\[ \Delta_L'=2\Delta_L, \qquad \Delta_R'=\frac12\Delta_R \]

最后得到的：

\[ \Delta_L'\Delta_R' = \Delta_L\Delta_R \]

完全一样。

所以 Multinex 的“物理解释性”实际上主要来自于：

\[ \boxed{ \text{两条分支接收到的先验不同} } \]

也就是：

\[ f_L \]

只能看到专门设计的 luminance descriptors，而：

\[ f_R \]

只能看到 chrominance descriptors。

因此，**真正让两条分支产生语义分工的，不只是式 (1)，更重要的是接下来的 Multi-prior Guidance Stack。** 这也是我认为本论文真正有价值的地方。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 7. Multi-prior 到底是什么？

这是论文名字里的 **Multi**。

作者认为，一个颜色空间提供的 brightness/chroma 表征不一定可靠。

所以干脆：

> **同时算很多种不同定义的亮度和颜色。**

但不是把 RGB、HSV、YUV 全部塞进去，而是从不同颜色理论中抽取最有用的“单个 descriptor”。

于是得到两个 stack。

---

# 8. Luminance Guidance Stack：4 个亮度先验

定义：

\[ \boxed{ \mathcal S_L= [ Y_{\text{Rec.709}}, Y_{\text{vmax}}, Y_{\text{lightness}}, Y_{L_2} ] } \]

所以输入 RGB：

\[ H\times W\times3 \]

被转换为：

[  
\mathcal S_L\in\mathbb R^{H\times W\times4}.  
] [arXiv](https://arxiv.org/html/2604.10359v1)

### \(Y_{\text{Rec.709}}\)

\[ Y_{\text{Rec.709}} = 0.2126R+0.7152G+0.0722B. \]

这是典型的 perceptual luminance。

G 权重最大，因为人眼对绿色更敏感。

它回答：

> **人眼感觉这个像素有多亮？**

---

### \(Y_{\text{vmax}}\)

\[ Y_{\text{vmax}} = \max(R,G,B). \]

这很像 HSV 中的 \(V\)。

它回答：

> **这个像素三个颜色通道里，最亮的那个到底有多亮？**

这对：

- 高光；
- 灯源；
- 强颜色区域

特别有意义。

---

### \(Y_{\text{lightness}}\)

\[ Y_{\text{lightness}} = \frac{\max(R,G,B)+\min(R,G,B)}{2}. \]

这就是 HSL 的 Lightness 风格定义。

它利用 maximum 和 minimum，因此和 Rec.709 的线性组合看到的东西不完全一样。

---

### \(Y_{L_2}\)

\[ Y_{L_2} = \sqrt{R^2+G^2+B^2+\epsilon}. \]

它把 RGB 看成一个三维向量：

\[ [R,G,B]. \]

然后求这个向量的长度。

所以它回答：

> **这个像素整体 RGB energy 有多大？**

论文后面的分析发现，\(Y_{L_2}\) 在梯度/结构信息方面尤其有用，而 \(Y_{\text{vmax}}\) 更擅长表达全局 illumination variation；因此这些 descriptor 并不只是重复计算同一件事。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 9. 为什么不只用 Rec.709？

这是 Multi-prior 最重要的动机。

假设：

\[ R=0.8,\quad G=0,\quad B=0. \]

那么：

\[ Y_{\text{Rec.709}} \approx0.17. \]

Rec.709 会认为这个像素“不太亮”。

但：

\[ Y_{\text{vmax}}=0.8. \]

它却告诉网络：

> 这里其实存在一个非常强的红色响应。

因此单独使用某一种 luminance 定义，会丢掉某些信息。

作者做了很完整的 ablation：

只用一个 luminance prior 时大约只有 **18–19 dB PSNR**；随着多个先验组合，性能持续增加，四个全部使用达到约 **23.19 dB / 0.843 SSIM**。[arXiv](https://arxiv.org/html/2604.10359v1)

所以：

\[ \boxed{\text{multi-prior}\neq简单冗余} \]

它本质是在用人工设计的 feature engineering 帮小网络减负。

---

# 10. Reflectance Guidance Stack：5 个颜色先验

颜色分支则使用：

# [  
\boxed{  
\mathcal S_R

[C_b,C_r,r,g,S]  
}  
] [arXiv](https://arxiv.org/html/2604.10359v1)

它可以进一步看成三组。

第一组：

\[ C_b,\ C_r \]

是 YCbCr 风格的 color difference：

\[ C_b = -0.168736R -0.331264G +0.5B \]\[ C_r = 0.5R -0.418688G -0.081312B. \]

它们的目的就是：

> 尽量把颜色差异和整体 brightness 分开。

---

第二组是 normalized chromaticity：

\[ r = \frac{R}{R+G+B+\epsilon} \]\[ g = \frac{G}{R+G+B+\epsilon}. \]

这个设计非常漂亮。

假设光照只是把整个 RGB 放大：

\[ [R,G,B]\rightarrow k[R,G,B]. \]

那么：

\[ r' = \frac{kR}{kR+kG+kB} = \frac{R}{R+G+B}. \]

所以：

\[ \boxed{r'=r} \]

同理：

\[ g'=g. \]

也就是说它对**整体亮度缩放近似不敏感**。

这是非常适合 reflectance/color branch 的性质：

> 亮暗变了，但颜色比例尽量保持。

论文的 ablation 中，单独使用 \([r,g]\) 也是三类 reflectance priors 中效果最好的，作者认为正与这种 illumination-invariant color ratio 有关。[arXiv](https://arxiv.org/html/2604.10359v1)

---

第三个：

\[ S = \frac{\max(R,G,B)-\min(R,G,B)} {\max(R,G,B)+\epsilon}. \]

就是 saturation 风格描述。

它回答：

> **这个像素离灰轴有多远？**

如果：

\[ R\approx G\approx B \]

那么：

\[ S\approx0. \]

说明接近灰色。

反之，一个纯红：

\[ R\gg G,B \]

则：

\[ S \]

很大。

因此颜色分支实际上知道：

\[ \boxed{ \text{色差} + \text{RGB颜色比例} + \text{饱和度} } \]

三个互补信息。

---

# 11. 到这里，你就可以理解 Figure 2 左半部分了

输入：

\[ I \]

不是直接进入网络，而是先走完全无参数的 analytical transform：

\[ I \rightarrow \mathcal S_L \]

和：

\[ I \rightarrow \mathcal S_R. \]

得到：

\[ 4\text{ channel luminance descriptors} \]

和：

\[ 5\text{ channel color descriptors}. \]

然后：

\[ \mathcal S_L\xrightarrow{f_L}\Delta_L \]\[ \mathcal S_R\xrightarrow{f_R}\Delta_R. \]

再：

\[ \Delta_I = \Delta_L\odot\Delta_R \]

最后：

\[ \boxed{ \hat I=I+\Delta_I. } \]

这就是整个 Multinex 的主干。[arxiv.org](https://arxiv.org/html/2604.10359v1)

---

# 12. 那两个 Fusion Module 到底干什么？

两个分支：

\[ f_L,\quad f_R \]

结构基本相同，只是参数彼此独立。

它们并不是复杂 Transformer，而主要由：

\[ \boxed{ FB + CWA } \]

组成。[arXiv](https://arxiv.org/html/2604.10359v1)

其中 FB 是 **Fusion Block**。

论文写成：

\[ \bar X= \mathrm{MSEF} \circ \mathrm{ReLU} \circ \mathrm{DSConv} \circ \mathrm{MSEF}(X). \]

即：

\[ X \rightarrow MSEF \rightarrow DSConv \rightarrow ReLU \rightarrow MSEF. \]

这里 DSConv 是 depthwise-separable convolution，因此计算量很低。

MSEF 则是一种轻量的 squeeze-excitation 风格模块，通过：

- LayerNorm；
- global average pooling；
- channel weighting；
- depthwise convolution；

同时获得一点全局 channel 信息和局部空间信息。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 13. CWA：我认为是 Figure 2 第二个需要重点理解的模块

CWA = **Component-Wise Attention**。

计算：

# [  
A

\sigma  
\left(  
Conv_{1\times1}  
(  
DWConv_{7\times7}(X)  
)  
\right).  
] [arXiv](https://arxiv.org/html/2604.10359v1)

为什么叫 component-wise？

因为输入不是普通 CNN feature。

输入通道本身就具有明确含义：

\[ [Y_{709},Y_{\max},Y_{\text{lightness}},Y_{L2}]. \]

所以 CWA 做的其实是：

> 对不同位置，判断**哪一种人工 descriptor 更值得相信**。

比如：

暗部区域可能：

\[ Y_{L2} \]

更重要；

灯源附近：

\[ Y_{\max} \]

可能更重要；

普通区域：

\[ Y_{709} \]

可能更加可靠。

因此 attention 的对象并不是普通 latent channel，而是：

\[ \boxed{\text{有物理意义的不同先验}} \]

这一点使 CWA 比普通 self-attention 更符合这篇论文的设计哲学。

---

# 14. 整个 Fusion Module 可以画成

论文公式为：

\[ \bar{\mathcal S} = FB^T \left( Conv_{1\times1}(\mathcal S) \right) \]

同时另一条路：

\[ A=CWA(\mathcal S). \]

然后：

\[ A\odot\bar{\mathcal S} \]

再继续：

\[ FB^T \]

最后：

\[ Conv_{1\times1} \]

输出 correction。

即：

[  
\boxed{  
\mathcal S  
\rightarrow Conv  
\rightarrow FB^T  
\rightarrow  
\times CWA(\mathcal S)  
\rightarrow FB^T  
\rightarrow Conv  
}  
] [arXiv](https://arxiv.org/html/2604.10359v1)

完整版：

\[ T=3. \]

Nano：

\[ T=1 \]

并进一步简化 fusion path 和 FB，因此才能做到约 **0.7K 参数**。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 15. 为什么 CWA 放在网络中间？

论文还专门做了 ablation。

把 attention 放：

- 前面；
- 中间；
- 后面。

PSNR/SSIM 分别大约是：

\[ 22.78/0.831 \]\[ \boxed{23.19/0.843} \]\[ 22.41/0.823. \]

也就是说，中间最好。[arXiv](https://arxiv.org/html/2604.10359v1)

直觉很好理解：

如果 attention 太晚：

> descriptors 已经被卷积混合得差不多了，再判断“哪个 prior 有用”意义就弱了。

如果太早：

> prior 还完全没有经过 feature refinement。

因此作者采用：

\[ \text{先稍微提取特征} \rightarrow \text{根据原始 descriptors 做 attention} \rightarrow \text{再进一步融合}. \]

---

# 16. Ablation 非常能说明论文到底哪里有效

最值得看的其实是 Table 4。

没有 Multinex priors，仅让网络从原始 RGB 学：

\[ \hat I_i=f_i(I_i) \]

只有：

\[ 14.15\text{ dB}. \]

只使用 luminance：

\[ 20.57\text{ dB}. \]

只使用 reflectance：

\[ 18.50\text{ dB}. \]

两者一起：

[  
\boxed{23.19\text{ dB}}.  
] [arXiv](https://arxiv.org/html/2604.10359v1)

这个实验实际上非常重要。

因为网络规模保持在相近量级，却从：

\[ 14.15\rightarrow23.19 \]

说明这篇论文性能提升的核心并不是：

> “发明了一个更好的卷积模块。”

而是：

\[ \boxed{\text{给微型网络一个更好的输入表示}} \]

这也是这篇文章最值得借鉴的思想。

---

# 17. Loss 反而非常普通

训练目标：

\[ \mathcal L = \lambda_{\text{MSE}}\mathcal L_{\text{MSE}} + \lambda_{\text{MS-SSIM}}\mathcal L_{\text{MS-SSIM}} + \lambda_{\text{Perc}}\mathcal L_{\text{Perc}} \]

权重：

\[ 1,\quad0.2,\quad0.01. \]

分别负责：

- pixel fidelity；
- multi-scale structure；
- perceptual feature consistency。 [arXiv](https://arxiv.org/html/2604.10359v1)

三者一起的 ablation 最好：

[  
23.19\text{ PSNR},  
\qquad  
0.843\text{ SSIM}.  
] [arXiv](https://arxiv.org/html/2604.10359v1)

所以这篇论文的创新重点**完全不在 loss**。

---

# 18. 实验结果应该怎样看？

完整版 Multinex 约：

\[ 44.7K \]

参数，Nano：

\[ 0.69K. \]

在 LOL-v1 / LOL-v2 等 benchmark 上，它在相应 lightweight/micro 参数规模中表现很强，同时论文还在 MEF、LIME、DICM、NPE 上进行了 no-reference 测试。[arXiv](https://arxiv.org/html/2604.10359v1)

比较有意思的是下游任务。

作者把增强器放到 YOLOv3 前，对 ExDark 做目标检测。Nano 只有约 **0.7K 参数**，论文报告的 mAP50 达到 **80.7%**，甚至高于完整版的 79.7%。这说明对于 machine vision，增强网络未必需要追求极复杂的像素恢复；有时稳定的 illumination/color normalization 已经足够有帮助。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 19. 这篇论文最大的优点

我认为不是“45K 参数”。

真正值得学习的是这个设计哲学：

\[ \boxed{ \text{如果网络容量有限，就不要让网络重新学习已有知识。} } \]

通常深度学习思路是：

\[ RGB \xrightarrow{\text{big network}} features. \]

Multinex 是：

\[ RGB \xrightarrow{\text{analytic transforms}} \text{high-value features} \xrightarrow{\text{tiny network}} output. \]

也就是把：

> **representation learning**

的一部分从神经网络里面拿出来，变成 deterministic feature construction。

这也是为什么：

\[ 0.7K \]

这样的极端参数规模还能工作。

---

# 20. 但它也有很明显的局限

第一，它主要解决：

\[ \text{illumination + color correction} \]

而不是很强的：

\[ \text{detail reconstruction}. \]

作者自己也指出，与大型模型相比，有些情况下 Multinex 的细节恢复较弱；在非常困难的 DICM / MEF 样本上，会出现 noise、light distortion，以及过曝导致的 color loss。作者把其中一部分原因归因于极小的参数规模。[arXiv](https://arxiv.org/html/2604.10359v1)

所以它更像：

\[ \boxed{\text{极高性价比增强器}} \]

而不是：

\[ \boxed{\text{极限 restoration 模型}}. \]

---

# 21. 还有一个很值得研究的潜在问题

尤其结合极暗场景来看：

\[ r=\frac{R}{R+G+B+\epsilon} \]

和：

\[ S= \frac{\max-\min}{\max+\epsilon}. \]

在极低照度区域，如果：

\[ R,G,B\approx0 \]

那么 denominator 很小。

此时 sensor noise 的微小变化就可能导致：

\[ r,g,S \]

出现很大的相对变化。

论文自己的 reflectance ablation 也提到，saturation 单独使用相对较弱，与其 noise sensitivity 有关。[arXiv](https://arxiv.org/html/2604.10359v1)

所以如果你研究的是**极暗图像恢复**，这里其实有很大的改进空间：

\[ \boxed{ \text{confidence-aware / noise-aware multi-prior} } \]

例如让模型根据：

\[ SNR \]

或者 local intensity 判断：

> 暗区不要太相信 \(r,g,S\)，亮区再提高 chromaticity prior 权重。

我认为这是从这篇论文非常自然能延伸出来的方向。

---

# 22. 用一句话总结整篇 Multinex

如果让我把这篇论文压缩成一句话，就是：

\[ \boxed{ \text{先用多个解析颜色/亮度公式，把RGB变成“容易学习”的表示， 再让极小网络预测亮度修正和颜色修正， 最后用Retinex式乘法形成残差，加回原图。} } \]

完整 pipeline 就是：

\[ \boxed{ I \rightarrow \begin{cases} [Y_{709},Y_{\max},Y_{\text{lightness}},Y_{L2}] \xrightarrow{f_L}\Delta_L\\[2mm] [C_b,C_r,r,g,S] \xrightarrow{f_R}\Delta_R \end{cases} } \]

然后：

\[ \boxed{ \hat I = I+\Delta_L\odot\Delta_R } \]

这是整篇文章真正需要记住的公式。[arXiv](https://arxiv.org/html/2604.10359v1)

下一步最适合直接**逐箭头精读 Figure 2**：我可以把图中的 `S_L → Conv1×1 → FB → CWA → FB → Δ_L`、`S_R → Δ_R`、中间每一个 tensor 的尺寸，以及 **FB / MSEF / CWA 到底在代码里做了什么**全部展开讲清楚。




可以。我们这次把 **Figure 2 当作一张完整的网络结构图** 来精读，而不是只解释某几个模块。最合理的阅读顺序是：

\[ \boxed{ I \rightarrow \text{Multi-view Representation} \rightarrow (S_L,S_R) \rightarrow (f_L,f_R) \rightarrow (\Delta_L,\Delta_R) \rightarrow \Delta_I \rightarrow \hat I } \]

先理解**顶层算法逻辑**，再下钻到 Fusion Module、FB、CWA，最后再回来看为什么它能做到只有约 45K 参数。Figure 2 的官方图和论文文字对这条数据流是一致的。[arXiv](https://arxiv.org/html/2604.10359v1/x1.png)

---

# 一、先把 Figure 2 分成 5 个区域

Figure 2 实际上同时画了“主网络”和三个模块细节，因此初看比较乱。可以把它拆成：

1. **最上方：Multinex 整体 pipeline**
2. **左下：Understanding Guidance Stacks**
3. **中下：Fusion Module \(f(S)\)**
4. **右中：Fusion Block，简称 FB**
5. **右下：Component-wise Attention，简称 CWA**

其中真正参与一次前向推理的主要是：

\[ \boxed{ \text{顶部主干} + \text{Fusion Module} + \text{FB} + \text{CWA} } \]

左下角的 **Understanding Guidance Stacks** 主要是作者为了分析和解释 \(S_L,S_R\) 为什么合理而设计的可视化分析，**不是推理时还要额外运行的网络模块**。这是读 Figure 2 时首先要区分清楚的一点。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 二、第一阶段：输入图像 \(I\)

输入为低照度 RGB 图像：

\[ I\in[0,1]^{H\times W\times3}. \]

也就是：

\[ I=[I_R,I_G,I_B]. \]

传统 CNN 往往直接做：

\[ I\xrightarrow{\text{CNN}}\hat I. \]

但 Multinex 不这么做。

作者认为，对于只有几十 K 参数的小网络，让它从 RGB 中自己学习：

- 什么代表亮度；
- 什么代表颜色；
- 什么代表饱和度；
- 什么信息对光照变化稳定；

负担太重。

所以 Figure 2 的第一个关键模块叫：

\[ \boxed{\text{Multi-view Representation Construction}} \]

它先利用**人工定义的解析公式**把 RGB 变成容易被网络利用的先验。也就是说，这里基本没有学习参数。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 三、第二阶段：从 \(I\) 构造两个 Guidance Stack

这是 Figure 2 左上角最重要的分叉：

\[ I \rightarrow \begin{cases} S_L & \text{亮度信息}\\ S_R & \text{颜色信息} \end{cases} \]

这里的 \(L\) 可以理解成 luminance / illumination，\(R\) 对应 Retinex 中 reflectance 的思想。

但要特别注意：

> **这里不是把图像严格分解成真实 illumination 和 reflectance。**

它只是人为构造两套更偏向“亮度”和“颜色”的描述量，让后面的两个微型网络分别学习。

这也是 Multinex 和经典 RetinexNet 一类方法的本质区别之一。论文明确说，它把 Retinex 当成 **structural prior**，而不是要求网络真正完成物理意义上的 Retinex decomposition。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 四、亮度 Guidance Stack：\(S_L\)

Figure 2 下方那一组灰度图就是：

\[ S_L= [ Y_{\mathrm{Rec.709}}, Y_{\mathrm{vmax}}, Y_{\mathrm{lightness}}, Y_{L_2} ]. \]

因此：

\[ \boxed{ S_L\in\mathbb R^{H\times W\times4} } \]

它不是一张 illumination map，而是 **4 张不同定义的亮度图堆叠起来**。[arXiv](https://arxiv.org/html/2604.10359v1)

这四个通道分别从四种角度看“亮不亮”。

### 1. Rec.709 luminance

\[ Y_{\mathrm{Rec.709}} = 0.2126R+0.7152G+0.0722B. \]

它考虑了人眼对不同颜色敏感度不同，因此绿色权重大。

可以理解成：

\[ \boxed{\text{人眼感知意义上的亮度}} \]

---

### 2. \(Y_{\mathrm{vmax}}\)

\[ Y_{\mathrm{vmax}}=\max(R,G,B). \]

这实际上就是问：

> RGB 三个通道里，最强的响应是多少？

所以高光、灯源、强彩色区域都会比较突出。

可以理解成：

\[ \boxed{\text{最强光响应}} \]

---

### 3. \(Y_{\mathrm{lightness}}\)

\[ Y_{\mathrm{lightness}} = \frac{\max(R,G,B)+\min(R,G,B)}2. \]

这是 HSL 风格的 Lightness。

与只看最大值不同，它同时利用最大和最小通道，所以对对比度和亮度的描述又有所不同。

---

### 4. \(Y_{L_2}\)

\[ Y_{L_2} = \sqrt{R^2+G^2+B^2+\varepsilon}. \]

相当于把：

\[ (R,G,B) \]

看成三维向量，然后计算：

\[ \Vert(R,G,B)\Vert_2. \]

所以它表达的是：

\[ \boxed{\text{RGB整体能量}} \]

论文后续分析还发现，不同 luminance prior 的作用确实不同。例如 \(Y_{L_2}\) 对高频结构/梯度尤其有贡献，而 \(Y_{\mathrm{vmax}}\) 更擅长表达某些全局 illumination variation。[arXiv](https://arxiv.org/html/2604.10359v1)

因此作者不是在重复计算 4 次“亮度”，而是：

\[ \boxed{ \text{让小网络从四个互补观察角度理解曝光} } \]

---

# 五、颜色 Guidance Stack：\(S_R\)

Figure 2 上方那组彩色 feature maps 是：

\[ S_R= [C_b,C_r,r,g,S]. \]

所以：

\[ \boxed{ S_R\in\mathbb R^{H\times W\times5} } \]

它们主要描述颜色，而尽可能弱化绝对亮度的影响。[arXiv](https://arxiv.org/html/2604.10359v1)

其中有三类信息。

第一类是：

\[ C_b,C_r \]

即蓝色色差和红色色差。它们类似 YCbCr 中的 chrominance component：

\[ C_b=-0.168736R-0.331264G+0.5B \]\[ C_r=0.5R-0.418688G-0.081312B. \]

它们试图表达：

\[ \boxed{\text{颜色偏蓝多少、偏红多少}} \]

而不是直接表达像素有多亮。

第二类：

\[ r=\frac{R}{R+G+B+\varepsilon}, \qquad g=\frac{G}{R+G+B+\varepsilon}. \]

这是 normalized chromaticity。

它非常重要，因为假设照明只是把 RGB 同时乘以 \(k\)：

\[ (R,G,B)\rightarrow(kR,kG,kB), \]

则：

\[ r' = \frac{kR}{kR+kG+kB} = r. \]

也就是说：

\[ \boxed{ r,g\text{ 对整体强度缩放具有近似不变性} } \]

所以它特别适合作为“reflectance/color”线索。

最后一个：

\[ S= \frac{\max(R,G,B)-\min(R,G,B)} {\max(R,G,B)+\varepsilon} \]

表示 saturation，告诉网络：

> 这个颜色到底有多鲜艳，离灰轴有多远？

论文因此把这五个 map 总结为 color-difference、chromaticity ratio 和 saturation 三组互补颜色线索。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 六、到这里，其实 Multinex 已经完成第一次“解耦”

现在输入已经由：

\[ I\in\mathbb R^{H\times W\times3} \]

变成：

\[ \boxed{ S_L:H\times W\times4 } \]

和：

\[ \boxed{ S_R:H\times W\times5 } \]

注意，这不是完全数学严格意义上的 illumination/color decomposition，而是：

\[ \boxed{ RGB \rightarrow \text{brightness-oriented representation} + \text{color-oriented representation} } \]

也就是先验驱动的**软解耦**。

这就是 Multinex 为什么叫：

> **Multi-prior Retinex**

Multi-prior 指多个解析先验；Retinex 指亮度和颜色分别建模的结构思想。

---

# 七、然后进入 Figure 2 的核心：两个 Fusion Module

接下来是：

\[ S_L\xrightarrow{f_L}\Delta_L \]

和：

\[ S_R\xrightarrow{f_R}\Delta_R. \]

两个 Fusion Module：

\[ f_L,\quad f_R \]

**结构相同，但参数不共享。**

其一般形式写成：

\[ f(S): \mathbb R^{H\times W\times K} \rightarrow \mathbb R^{H\times W\times D}. \]

其中亮度分支：

\[ K=4,\quad D=1 \]

所以：

\[ S_L:H\times W\times4 \rightarrow \Delta_L:H\times W\times1. \]

颜色分支：

\[ K=5,\quad D=3 \]

因此：

[  
S_R:H\times W\times5  
\rightarrow  
\Delta_R:H\times W\times3.  
] [arXiv](https://arxiv.org/html/2604.10359v1)

这里一定要注意：

\[ \boxed{ \Delta_L\neq L,\qquad \Delta_R\neq R } \]

它们不是传统 Retinex 的 illumination 和 reflectance 本身，而是：

\[ \Delta_L=\text{luminance correction} \]\[ \Delta_R=\text{color/reflectance correction}. \]

---

# 八、现在下钻 Figure 2 中间：Fusion Module \(f(S)\)

这是整张 Figure 2 最需要认真读的地方。

统一把输入写成：

\[ S\in\mathbb R^{H\times W\times K}. \]

无论它是 \(S_L\) 还是 \(S_R\)，Fusion Module 都走两条并行路径：

\[ \boxed{ \begin{array}{ccc} S &\rightarrow& 1\times1 Conv\rightarrow FB^T\rightarrow\bar S\\ \downarrow&&\\ CWA&&\rightarrow A \end{array} } \]

然后：

\[ A\odot\bar S \]

再继续：

\[ FB^T\rightarrow Conv_{1\times1}\rightarrow\Delta. \]

论文正式写成：

\[ \bar S = FB^T(Conv_{1\times1}(S)) \]

以及：

# [  
f(S)

Conv_{1\times1}  
\left[  
FB^T  
\left(  
CWA(S)\odot\bar S  
\right)  
\right].  
] [arXiv](https://arxiv.org/html/2604.10359v1)

下面逐个解释。

---

# 九、Fusion Module 第一步：\(1\times1\) Conv

输入：

\[ S:H\times W\times K \]

经过：

\[ Conv_{1\times1} \]

得到：

\[ X:H\times W\times C. \]

例如对于：

\[ S_L:H\times W\times4 \]

就相当于：

\[ 4\rightarrow C. \]

这里的 \(1\times1\) 卷积有一个非常重要的知识点：

> **它不进行空间邻域建模，主要进行通道混合。**

假设某个像素：

\[ S_L(x,y)= [Y_{709},Y_{\max},Y_{\text{lightness}},Y_{L2}]. \]

那么一个输出通道本质上就是：

\[ X_j(x,y) = \sum_{k=1}^{4}w_{jk}S_{L,k}(x,y)+b_j. \]

因此它实际上是在学习：

\[ \boxed{ \text{四种亮度 prior 应该怎样线性组合} } \]

比如某一个隐特征可能偏向：

\[ Y_{\max} \]

另一个可能偏向：

\[ Y_{L2}+Y_{709}. \]

所以第一层 \(1\times1\) Conv 可以理解成：

\[ \boxed{ \text{prior space}\rightarrow\text{learned feature space} } \]

这一步非常便宜。

---

# 十、然后进入 FB：Fusion Block

Figure 2 右上给出了 FB：

\[ X \rightarrow MSEF \rightarrow DSConv \rightarrow ReLU \rightarrow MSEF \rightarrow \bar X. \]

论文写成：

\[ \bar X = MSEF \circ ReLU \circ DSConv \circ MSEF(X). \]

其中 DSConv 使用 \(3\times3\) kernel。[arXiv](https://arxiv.org/html/2604.10359v1)

这里实际上在解决两个问题：

\[ \boxed{ \text{空间信息} + \text{通道重要性} } \]

---

# 十一、先理解 DSConv：为什么不用普通卷积？

普通 \(3\times3\) Conv 假设输入输出均为 \(C\) 个通道，参数量大致是：

\[ 9C^2. \]

而 Depthwise Separable Convolution 拆成两步。

先 Depthwise：

\[ 3\times3 \]

每个 channel 自己卷积：

\[ 9C \]

参数。

再 Pointwise：

\[ 1\times1 \]

进行通道混合：

\[ C^2. \]

所以总计：

\[ 9C+C^2 \]

相比：

\[ 9C^2 \]

少很多。

因此：

\[ \boxed{ DSConv = \text{便宜的空间建模} + \text{便宜的通道融合} } \]

这就是 Multinex 能做轻量模型的基础操作之一。论文也明确将 DWConv 和 DSConv 作为降低 learnable weights 的基础模块。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 十二、FB 中真正有意思的是 MSEF

MSEF 全称：

**Multi-Stage Squeeze & Excite Fusion**。

它来源于 SE（Squeeze-and-Excitation）思想。

要先理解普通 SE 在做什么。

一个 feature：

\[ X\in\mathbb R^{H\times W\times C}. \]

不同 channel 的重要性往往不同。

于是 SE 先做 Global Average Pooling：

\[ H\times W\times C \rightarrow 1\times1\times C. \]

得到每个 channel 的全局统计量。

然后用一个很小的 MLP 学：

\[ [w_1,w_2,\ldots,w_C]. \]

再：

\[ X_c\rightarrow w_cX_c. \]

所以 SE 本质上在回答：

\[ \boxed{ \text{“整张图来看，哪个 feature channel 更重要？”} } \]

---

# 十三、MSEF 比普通 SE 又多了一些局部信息

论文给出的 MSEF 中，channel weights 来自：

\[ w = \tanh \left[ W_2 ReLU \left( W_1 GAP(LN(X)) \right) \right]. \]

其中：

- LN：Layer Normalization；
- GAP：Global Average Pooling；
- \(W_1,W_2\)：压缩—扩张 bottleneck；
- \(w\)：每个 channel 的自适应权重。

然后：

\[ Z_i=w_iLN_i(X). \]

最终还有一个 depthwise-convolution-based local branch，并通过 residual 形式输出。论文因此把 MSEF 描述为同时利用**全局语义信息和局部细节信息**的轻量机制。[arXiv](https://arxiv.org/html/2604.10359v1)

你可以不纠结它的每一个细枝末节，先抓住核心：

\[ \boxed{ MSEF \approx \text{全局通道重标定} + \text{局部空间细化} } \]

---

# 十四、为什么一个 FB 要放两个 MSEF？

结构：

\[ X \rightarrow MSEF_1 \rightarrow DSConv \rightarrow ReLU \rightarrow MSEF_2. \]

它不是随便堆的。

第一个 MSEF：

\[ \boxed{\text{先判断当前各 channel 的价值}} \]

然后 DSConv：

\[ \boxed{\text{进行局部空间信息交换和特征融合}} \]

ReLU：

\[ \boxed{\text{引入非线性}} \]

第二个 MSEF：

\[ \boxed{\text{对融合之后的新 feature 再重新判断重要性}} \]

论文也是这样描述的：第一个 MSEF 利用全局上下文校准通道；DSConv + ReLU 做轻量空间 filtering；第二个 MSEF 再重新评估增强后的 activations。[arXiv](https://arxiv.org/html/2604.10359v1)

所以 FB 可以粗略概括成：

\[ \boxed{ \text{重标定} \rightarrow \text{局部融合} \rightarrow \text{再重标定} } \]

---

# 十五、为什么 Figure 2 里一个 FB 方块，公式却是 \(FB^T\)？

这是图中一个很容易误解的地方。

Figure 2 为了画得简洁，只画：

\[ FB \]

但完整版 Multinex 实际是：

\[ FB^T. \]

其中：

\[ T=3. \]

所以真实完整版相当于：

\[ Conv \rightarrow FB \rightarrow FB \rightarrow FB. \]

CWA 后面还有另外：

\[ FB \rightarrow FB \rightarrow FB. \]

也就是说完整版 Fusion Module 两个 stage 分别都有 \(T=3\) 个 FB。Nano 则使用 \(T=1\)，而且路径还进一步简化。[arXiv](https://arxiv.org/html/2604.10359v1)

因此千万不要从 Figure 2 误以为：

> Fusion Module 总共只有两个 FB。

图里的 FB 是对：

\[ FB^T \]

的抽象表示。

---

# 十六、现在看 Fusion Module 的第二条路径：CWA

这一条路径非常关键：

\[ S \rightarrow CWA \rightarrow A. \]

CWA：

**Component-Wise Attention**

公式是：

\[ A = \sigma \left[ Conv_{1\times1} \left( DWConv_{7\times7}(S) \right) \right]. \]

最终：

[  
A\in[0,1]^{H\times W\times C}.  
] [arXiv](https://arxiv.org/html/2604.10359v1)

它本质上是在生成：

\[ \boxed{\text{一张空间变化的 feature gating map}} \]

---

# 十七、CWA 为什么先使用 \(7\times7\) DWConv？

这里需要区分：

### DWConv

Depthwise Convolution：

\[ S_1\rightarrow Conv(S_1) \]\[ S_2\rightarrow Conv(S_2) \]

每一个通道**独立卷积**。

不同通道之间暂时不交流。

---

而普通 Conv 会：

\[ S_1,S_2,S_3,\cdots \]

一起混合。

作者特别使用：

\[ 7\times7\ DWConv \]

就是为了：

\[ \boxed{ \text{先保持每一种 descriptor 的独立性} } \]

论文明确说明，CWA 早期使用 DWConv，就是为了避免 early-stage inter-channel mixing。[arXiv](https://arxiv.org/html/2604.10359v1)

这一点和 Multi-prior 的设计高度一致。

例如：

\[ S_L= [Y_{709},Y_{\max},Y_{\text{lightness}},Y_{L2}] \]

作者不希望一上来就把：

\[ Y_{709}+Y_{\max}+Y_{L2} \]

全部混成一个 feature。

而是先让：

\[ Y_{709} \]

看看自己的 \(7\times7\) 局部区域；

\[ Y_{\max} \]

也看看自己的 \(7\times7\) 区域；

\[ Y_{L2} \]

也独立分析。

于是可以理解成：

\[ \boxed{ \text{先独立检查每一种 prior 在局部区域里的响应模式} } \]

---

# 十八、为什么 CWA 用 \(7\times7\)，FB 却只用 \(3\times3\)？

这是很有设计意味的。

FB 的任务主要是：

\[ \text{feature refinement} \]

所以：

\[ 3\times3 \]

足够做局部特征提取，而且计算量低。

但 CWA 的任务是：

> 判断某个 descriptor 在当前位置附近究竟可靠不可靠。

这个判断更需要稍大的 context。

所以作者使用：

\[ 7\times7. \]

而且因为是 DWConv，参数量仍然很小。

于是可以记成：

\[ \boxed{ FB:\quad3\times3\quad\text{精细加工} } \]\[ \boxed{ CWA:\quad7\times7\quad\text{较大范围判断prior可靠性} } \]

这是从结构上可以得到的一个很自然的理解。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 十九、CWA 接下来的 \(1\times1\) Conv 干什么？

DWConv 后仍然保留：

\[ K \]

个 descriptor。

然后：

\[ Conv_{1\times1} \]

执行：

\[ K\rightarrow C. \]

也就是说，把原始 prior 维度对齐到主分支的 feature channels：

\[ \bar S\in\mathbb R^{H\times W\times C}. \]

然后 sigmoid：

\[ \sigma(x)=\frac1{1+e^{-x}} \]

把数值映射到：

\[ 0\sim1. \]

得到：

\[ A\in[0,1]^{H\times W\times C}. \]

所以：

\[ A(x,y,c) \]

可以理解为：

> 在 \((x,y)\) 这个位置，第 \(c\) 个 feature 应该保留多少。

论文称它为 soft attention score。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 二十、注意：CWA 不完全等价于“给四个 prior 各一个权重”

这个地方需要比我前面回答得更严格一些。

很容易简单地说：

\[ A=[a_{709},a_{\max},a_{\text{lightness}},a_{L2}] \]

然后解释成“给四个 prior 加权”。

这种理解有助于直觉，但**严格来说并不完全准确**。

因为 CWA 中还有：

\[ Conv_{1\times1}:K\rightarrow C. \]

所以最终：

\[ A \]

是在**投影后的 feature channel 空间**里做 gating，而不一定保持“一张 attention map 精确对应一个原始 prior”。

更严格的说法应该是：

\[ \boxed{ CWA利用保持独立性的各个descriptor局部响应， 生成与融合特征对齐的空间—通道注意力。 } \]

因此它具有 prior-aware 的性质，但不能简单理解成四个固定 scalar 权重。

这一点非常值得记住。

---

# 二十一、CWA 和 Transformer Self-Attention 完全不是一回事

Transformer attention 通常有：

\[ Q,K,V \]

以及：

\[ Attention(Q,K,V) = Softmax \left( \frac{QK^T}{\sqrt d} \right)V. \]

它会建立 token-token 之间的关系。

而 Multinex CWA：

\[ S \rightarrow DWConv \rightarrow Conv_{1\times1} \rightarrow Sigmoid. \]

没有：

\[ QK^T. \]

也没有 token-to-token global correlation。

所以它更接近：

\[ \boxed{ \text{spatial-channel gating} } \]

而不是 Transformer attention。

这也解释了为什么它计算量这么低。

---

# 二十二、主路径和 CWA 路径终于汇合

主路径：

\[ S \rightarrow Conv_{1\times1} \rightarrow FB^T \rightarrow \bar S. \]

得到：

\[ \bar S\in\mathbb R^{H\times W\times C}. \]

CWA：

\[ S \rightarrow CWA \rightarrow A. \]

得到：

\[ A\in[0,1]^{H\times W\times C}. \]

然后进行 Hadamard product：

\[ \boxed{ F=A\odot\bar S } \]

即逐元素：

\[ F(x,y,c) = A(x,y,c)\bar S(x,y,c). \]

这就是 Figure 2 中圆圈带点的那个符号：

\[ \odot. \]

---

# 二十三、这个乘法的意义是什么？

假设某个 feature：

\[ \bar S(x,y,c)=1.2. \]

CWA 判断它当前比较重要：

\[ A(x,y,c)=0.9. \]

则：

\[ F=1.08. \]

基本保留下来。

如果：

\[ A=0.1, \]

则：

\[ F=0.12. \]

大幅压制。

所以：

\[ \boxed{ \bar S=\text{网络提取出的内容} } \]

而：

\[ \boxed{ A=\text{这些内容应该被使用多少} } \]

这两个东西不要混淆。

---

# 二十四、为什么 attention 不直接乘在原始 \(S\) 上？

Figure 2 的设计是：

\[ S \rightarrow Conv \rightarrow FB^T \rightarrow\bar S \]

然后：

\[ CWA(S)\odot\bar S. \]

也就是说：

> CWA 的权重来自原始 guidance stack，但作用对象是已经经过第一阶段提取的 feature。

这个设计很巧妙。

如果：

\[ A\odot S \]

太早做 attention，那么网络还没有完成足够的 feature extraction。

而现在：

\[ \bar S \]

已经变成学习得到的表示，再利用原始 prior：

\[ S \]

产生的 guidance 去筛选它。

所以逻辑是：

\[ \boxed{ \text{原始物理先验} \rightarrow \text{指导学习特征} } \]

而不是让深层特征自己完全决定 attention。

---

# 二十五、Attention 后为什么还需要第二组 \(FB^T\)？

因为：

\[ A\odot\bar S \]

只是进行了 gating。

不同 feature 之间还需要进一步：

- 空间融合；
- 非线性变换；
- channel recalibration。

于是：

\[ A\odot\bar S \rightarrow FB^T. \]

可以把两个 FB stage 的职责粗略理解成：

第一阶段：

\[ \boxed{ \text{从多个 prior 中提取 feature} } \]

CWA：

\[ \boxed{ \text{决定这些 feature 在何处更值得使用} } \]

第二阶段：

\[ \boxed{ \text{对筛选后的 feature 再融合和细化} } \]

最后才预测 correction。

---

# 二十六、最后一个 \(1\times1 Conv\)：变成 \(\Delta\)

最后：

\[ F' \in \mathbb R^{H\times W\times C} \]

经过：

\[ Conv_{1\times1}. \]

亮度分支输出：

\[ C\rightarrow1 \]

得到：

\[ \boxed{ \Delta_L\in\mathbb R^{H\times W\times1} } \]

颜色分支：

\[ C\rightarrow3 \]

得到：

\[ \boxed{ \Delta_R\in\mathbb R^{H\times W\times3} } \]

论文明确说明最后的卷积分别使用 1 和 3 个 filters。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 二十七、一个非常重要的细节：\(\Delta_L,\Delta_R\) 后面没有 Sigmoid

很多图像增强网络会：

\[ Sigmoid \]

限制输出：

\[ [0,1]. \]

但 Multinex 明确表示：

> \(f_L,f_R\) 的输出没有 restrictive activation。

也就是说：

\[ \Delta_L,\Delta_R \]

理论上可以：

[

> 0,\quad=0,\quad<0.  
> ] [arXiv](https://arxiv.org/html/2604.10359v1)

为什么？

因为它们不是普通 image。

它们是：

\[ \boxed{\text{correction field}} \]

网络可能需要增加某些像素，也可能需要抑制某些颜色或者亮度。

如果强行 Sigmoid：

\[ \Delta>0 \]

自由度就会受限制。

---

# 二十八、两个 Fusion Module 最后怎么结合？

现在得到：

\[ \Delta_L:H\times W\times1 \]

和：

\[ \Delta_R:H\times W\times3. \]

Figure 2 中间的圆点表示逐元素乘：

\[ \boxed{ \Delta_I = \Delta_L\odot\Delta_R } \]

这里 \(\Delta_L\) 会广播到三个 RGB 通道：

\[ \Delta_I^R = \Delta_L\Delta_R^R \]\[ \Delta_I^G = \Delta_L\Delta_R^G \]\[ \Delta_I^B = \Delta_L\Delta_R^B. \]

因此：

[  
\Delta_I  
\in  
\mathbb R^{H\times W\times3}.  
] [arXiv](https://arxiv.org/html/2604.10359v1)

---

# 二十九、这个乘法为什么叫 Retinex-like？

经典 Retinex：

\[ I=L\odot R. \]

其中：

\[ L=\text{illumination} \]\[ R=\text{reflectance}. \]

Multinex 不预测真正的：

\[ L,R, \]

而预测：

\[ \Delta_L,\Delta_R. \]

然后仍然采用：

\[ \Delta_I=\Delta_L\odot\Delta_R. \]

所以作者保留了：

\[ \boxed{ \text{luminance}\times\text{reflectance} } \]

这种结构约束。

因此称：

\[ \boxed{\text{Retinex-like fusion}} \]

而不是经典 Retinex decomposition。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 三十、\(\Delta_L\) 和 \(\Delta_R\) 可以怎样直观理解？

假设某个像素：

\[ \Delta_L(x,y)=0.8. \]

可以粗略理解成：

> 这个位置需要较强的 correction。

而颜色支路：

\[ \Delta_R(x,y) = [0.7,0.4,0.2]. \]

那么：

\[ \Delta_I = 0.8 [0.7,0.4,0.2] \]

得到：

\[ [0.56,0.32,0.16]. \]

于是 RGB 三个通道被不同程度地修正。

所以：

\[ \Delta_L \]

控制一种**共享的空间亮度调制**，

而：

\[ \Delta_R \]

提供**RGB-specific 的 correction pattern**。

这是一个非常好的直觉。

---

# 三十一、但不要把 \(\Delta_L\) 当成严格的 illumination map

这是读这篇论文时一个很重要的理论问题。

因为：

\[ \Delta_I = \Delta_L\Delta_R. \]

假设：

\[ \Delta_L'=2\Delta_L \]

以及：

\[ \Delta_R'=\frac12\Delta_R, \]

则：

\[ \Delta_L'\Delta_R' = \Delta_L\Delta_R. \]

也就是说这个 factorization **并不唯一**。

所以仅从数学上不能保证：

\[ \Delta_L=\text{真实illumination} \]

和：

\[ \Delta_R=\text{真实reflectance}. \]

它们之所以表现出不同功能，主要来自两种 inductive bias：

\[ S_L\rightarrow f_L \]

只能看到 brightness-oriented priors；

而：

\[ S_R\rightarrow f_R \]

只能看到 chromatic priors。

再加上：

\[ \Delta_L:1\ channel,\qquad \Delta_R:3\ channels. \]

所以网络被**结构性地鼓励**产生 luminance / color 的功能分工。

这比说“网络成功分解出了真实 Retinex 成分”更加严谨。

---

# 三十二、最后还有最重要的一条残差连接

Figure 2 最右侧有一条虚线，从原输入：

\[ I \]

直接走到最终输出。

最终：

\[ \boxed{ \hat I = I+\Delta_I } \]

所以：

\[ \boxed{ \hat I = I+\Delta_L\odot\Delta_R } \]

这是整篇论文最核心的公式。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 三十三、为什么这里要用 residual，而不是直接输出 \(\hat I\)？

这是 Multinex 能够极轻量的真正原因之一。

直接：

\[ S_L,S_R \rightarrow \hat I \]

意味着网络需要重新生成：

- 边缘；
- 纹理；
- 结构；
- 高频细节；
- 颜色；
- 光照。

但是：

\[ \hat I=I+\Delta_I \]

意味着：

\[ I \]

本身直接保留了：

\[ \boxed{ \text{纹理、边缘和原始结构} } \]

网络只学习：

\[ \boxed{ \text{“应该改什么？”} } \]

所以任务从：

\[ \text{reconstruction} \]

变成：

\[ \text{enhancement/correction}. \]

作者也明确把这作为降低网络负担的重要设计理由。[arXiv](https://arxiv.org/html/2604.10359v1)

---

# 三十四、现在解释 Figure 2 左下角：Understanding Guidance Stacks

这一小块非常容易被误解为网络结构。

其实不是。

它主要回答：

> “凭什么说 \(S_L,S_R\) 这些人工 prior 是合理的？”

---

## 对 \(S_L\)：Descriptor Importance Analysis

作者分析：

\[ S_L= [Y_{709},Y_{\max},Y_{\text{lightness}},Y_{L2}] \]

是不是包含互补信息。

其中一个分析是 PCA-guided orthogonal energy：

\[ E_{K_L}(S_L). \]

Figure 2 左下展示的彩色 energy map，就是为了观察不同 luminance priors 在哪些空间区域具有互补信息。

高响应区域意味着多个 prior 在这里提供了比较丰富、互补的表达。论文还通过 gradient information 衡量各 descriptor 对局部结构的贡献。[arXiv](https://arxiv.org/html/2604.10359v1)

这些分析后来发现：

\[ Y_{L2} \]

对 gradient/high-frequency structure 很重要，而：

\[ Y_{\max} \]

对某些全局 shading 特性更重要。[arXiv](https://arxiv.org/html/2604.10359v1)

---

## 对 \(S_R\)：Linear Reconstruction Analysis

由于：

\[ C_b,C_r,r,g,S \]

单独一张图很难直接说“它就是物体颜色”。

所以作者做：

\[ LRA(S_R,I) \]

即 Linear Reconstruction Analysis。

核心思想是：

> 如果这些颜色 descriptors 确实包含足够的颜色信息，那么从它们应该能够较好地线性重建输入颜色结构。

作者利用 PCA 和 ridge regression 做 reconstruction，Figure 2 展示 reconstructed color image，来说明：

\[ S_R \]

联合起来确实保留了较丰富的颜色结构。[arXiv](https://arxiv.org/html/2604.10359v1)

再次强调：

\[ \boxed{ DIA/LRA是分析工具，不属于实际增强网络前向过程。 } \]

---

# 三十五、现在把 Figure 2 的 tensor shape 全部串起来

这是读网络图时最推荐掌握的一条线。

输入：

\[ \boxed{ I:H\times W\times3 } \]

固定解析变换：

\[ I\rightarrow \boxed{ S_L:H\times W\times4 } \]\[ I\rightarrow \boxed{ S_R:H\times W\times5 } \]

然后亮度分支：

\[ S_L \rightarrow Conv_{1\times1} \rightarrow H\times W\times C \]\[ \rightarrow FB^T \rightarrow \bar S_L:H\times W\times C. \]

同时：

\[ S_L \rightarrow CWA \rightarrow A_L:H\times W\times C. \]

然后：

\[ A_L\odot\bar S_L \]\[ \rightarrow FB^T \rightarrow Conv_{1\times1} \rightarrow \boxed{ \Delta_L:H\times W\times1 }. \]

颜色分支：

\[ S_R \rightarrow Conv_{1\times1} \rightarrow H\times W\times C \]\[ \rightarrow FB^T \rightarrow\bar S_R. \]

同时：

\[ S_R \rightarrow CWA \rightarrow A_R. \]

再：

\[ A_R\odot\bar S_R \rightarrow FB^T \rightarrow Conv_{1\times1} \rightarrow \boxed{ \Delta_R:H\times W\times3 }. \]

最后：

\[ \Delta_L \odot \Delta_R \rightarrow \boxed{ \Delta_I:H\times W\times3 } \]

再：

\[ I+\Delta_I \rightarrow \boxed{ \hat I:H\times W\times3 }. \]

这就是整张 Figure 2。

---

# 三十六、从设计者视角重新看 Figure 2

如果只看模块名字，很容易觉得：

> “就是一些 Conv、attention、SE，没什么特别的。”

但真正的创新逻辑其实不在某一个卷积层，而是在**信息是如何一步一步被限制和引导的**。

可以概括成下面这条思想链：

\[ RGB \]

先不让网络自己理解，而是人工变成：

\[ \boxed{ \text{亮度先验} + \text{颜色先验} } \]

然后两个很小的网络分别学习：

\[ \boxed{ \text{brightness correction} + \text{color correction} } \]

每个分支内部又通过：

\[ \boxed{ CWA } \]

根据输入先验自适应选择有效信息。

最后借用 Retinex：

\[ \boxed{ \Delta_I=\Delta_L\Delta_R } \]

形成结构化 correction。

最后再通过：

\[ \boxed{ \hat I=I+\Delta_I } \]

强制保留输入中的大量结构信息。

---

# 三十七、所以 Figure 2 最核心的三个 inductive bias

理解这三个东西，基本就真正理解了 Multinex。

### 第一个：Representation prior

不是：

\[ RGB\rightarrow CNN \]

而是：

\[ RGB \rightarrow S_L,S_R \rightarrow CNN. \]

即：

\[ \boxed{ \text{先验替网络提前完成部分representation learning} } \]

---

### 第二个：Retinex structural prior

不是直接：

\[ CNN\rightarrow\Delta_I. \]

而是：

\[ CNN_L\rightarrow\Delta_L \]\[ CNN_R\rightarrow\Delta_R \]

然后：

\[ \boxed{ \Delta_I=\Delta_L\odot\Delta_R. } \]

即对 correction 做 brightness/color factorization。

---

### 第三个：Residual prior

不是：

\[ CNN\rightarrow\hat I \]

而是：

\[ \boxed{ \hat I=I+\Delta_I. } \]

即假设：

> 原始低照度图像已经包含绝大部分结构，只需要学习 correction。

这三个先验共同导致网络不需要很大。

---

# 三十八、最后，用一句“代码逻辑”记住 Figure 2

如果把 Figure 2 写成伪代码，其实非常清楚：

```
# fixed analytic representation
S_L = luminance_priors(I)      # H×W×4
S_R = chromatic_priors(I)      # H×W×5

# learned lightweight fusion
Delta_L = f_L(S_L)             # H×W×1
Delta_R = f_R(S_R)             # H×W×3

# Retinex-like residual factorization
Delta_I = Delta_L * Delta_R    # H×W×3

# preserve original structure
I_hat = I + Delta_I
```

而一个 Fusion Module 则近似：

```
F = Conv1x1(S)
F = FB_T(F)

A = sigmoid(Conv1x1(DWConv7x7(S)))

F = A * F
F = FB_T(F)

Delta = Conv1x1(F)
```

其中：

\[ T=3 \]

对应完整版 Multinex。[arXiv](https://arxiv.org/html/2604.10359v1)

---

## Figure 2 最值得你真正记住的不是模块名

最终建议你脑中留下这样一幅结构：

\[ \boxed{ \begin{array}{ccccc} & S_L & \rightarrow & \text{lightweight fusion} & \rightarrow\Delta_L\\ I & & & &\\ & S_R & \rightarrow & \text{lightweight fusion} & \rightarrow\Delta_R \end{array} } \]

其中：

\[ S_L=\text{4个互补亮度先验}, \qquad S_R=\text{5个互补颜色先验}, \]

Fusion Module 内部：

\[ \boxed{ 1\times1Conv \rightarrow FB^T \rightarrow \times CWA(S) \rightarrow FB^T \rightarrow 1\times1Conv } \]

而：

\[ FB= \boxed{ MSEF \rightarrow DSConv_{3\times3} \rightarrow ReLU \rightarrow MSEF } \]\[ CWA= \boxed{ DWConv_{7\times7} \rightarrow Conv_{1\times1} \rightarrow Sigmoid } \]

最终：

\[ \boxed{ \hat I = I+ \underbrace{\Delta_L}_{1\text{ channel}} \odot \underbrace{\Delta_R}_{3\text{ channels}} } \]

**整张 Figure 2 的本质就是：用固定的多先验降低 representation learning 难度，用极轻量网络完成自适应融合，再用 Retinex 式因子化约束 correction，最后以 residual 方式保留原图结构。** [arXiv](https://arxiv.org/html/2604.10359v1)

下一步继续精读这张图时，最值得单独拆开的就是 **MSEF**：它在 Figure 2 里只是一个小方块，但实际上内部还有 `LN → GAP → bottleneck → channel weight → DWConv → residual`，把 **MSEF Figure 11 和 Figure 2 对起来**之后，就能把 Multinex 的全部网络操作彻底走通。