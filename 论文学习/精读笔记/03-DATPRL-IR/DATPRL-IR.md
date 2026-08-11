# DATPRL-IR：Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration

> 论文：Guanglu Dong 等，ICLR 2026。论文将其方法称为 Domain-Aware Task Prompt Representation Learning（DATPRL），并据此构建 DATPRL-IR。
>
> 原文 PDF：`D:\Zotero\DATA\storage\8HW2Z9CV\Dong 等 - 2026 - LEARNING DOMAIN-AWARE TASK PROMPT REPRE- SENTATIONS FOR MULTI-DOMAIN ALL-IN-ONE IM- AGE RESTORATION.pdf`

## 1. 一句话理解

DATPRL-IR 试图让一个图像恢复模型同时处理自然图像、医学图像和遥感图像中的多个任务。它维护两个可学习的 prompt pool：task prompt pool 存“做什么恢复任务”的知识，domain prompt pool 存“图像来自什么领域”的知识；模型根据输入图像查询并组合 prompt，再通过 cross-attention 和 adaptive gated fusion 将二者注入 encoder-decoder 恢复骨干。

## 2. 写作思路：为什么要同时建模任务与领域

论文的写作路线是“单域 AiOIR 的成功 → 多域扩展的困难 → 共享/特定知识的分解 → 双 prompt pool → 实验验证可扩展性”：

1. 先回顾 AiOIR：一个模型处理去雨、去模糊、超分等多种任务，避免为每个退化训练单独网络。
2. 指出既有工作大多只在一个图像域内区分任务，遇到自然、医学、遥感混合时，模型还需要学会领域差异。
3. 提出“任务之间、领域之间都有共享知识，也都有特定知识”。例如超分和去模糊都需要结构恢复；“灰度 + 人体器官”更像医学，“俯视 + 建筑”更像遥感。
4. 借鉴 L2P 的 prompt pool，把知识分散存放在一组 key-value prompt 中，并对每个实例动态检索，而不是给每个任务硬编码一个固定 prompt。
5. 使用 MLLM 生成高质量图像的领域描述，再通过 CLIP 文本特征对 domain prompt 做跨模态约束；MLLM 与 CLIP 只用于训练，推理时不增加开销。
6. 用 6-task/3-domain、9-task/3-domain、prompt 数量、MLLM 替换、prompt 设计和泛化实验说明方法并非只适用于一个任务。

## 3. 传统方法与术语解释

### 3.1 单任务恢复

经典做法是为每种任务训练一个模型：SR 恢复空间分辨率，去雨估计雨纹，去模糊估计模糊核或直接学习逆映射，医学 CT 去噪则需要考虑成像噪声分布。优点是目标明确，缺点是部署多个模型、维护多个权重，并且模型之间不能共享知识。

### 3.2 All-in-One Image Restoration

AiOIR 用一个网络处理多种退化。常见路线包括：

- **对比学习**：让不同退化的特征可区分，同时共享清晰图像表征。
- **prompt learning**：让可学习 prompt 编码退化或任务条件。
- **显式指令**：用文字说明“去雨/去噪/超分”，但依赖指令表达能力。
- **MoE**：设置多个专家或复杂度专家，按输入分配计算。
- **退化分类**：先判断退化类别，再条件化恢复。

本文与它们的区别是：不仅查询 task prompt，还查询 domain prompt，并让二者在实例级组合。

### 3.3 PSNR、SSIM 与 Fourier loss

PSNR 偏像素保真，SSIM 偏结构相似；论文还把 L1 误差放到 Fourier 域，约束频域中的低频亮度与高频纹理变化。频域损失不是“自动更好”，它的价值在于给多任务恢复提供一种跨域共享的结构/纹理约束。

## 4. 结构图精读

![DATPRL-IR 框架（论文 Fig. 2，PDF 第 4 页）](./结构图/fig2-04.png)

图中颜色是重要线索：橙色对应 domain prompt，绿色对应 task prompt，蓝色区域是实际恢复 backbone；火焰表示训练，雪花表示冻结。

### 4.1 左侧：MLLM 不是推理期恢复器

高质量图像输入 LLaVA-1.5-7B，生成“内容、颜色、亮度、拍摄视角”等多角度文本描述。这些描述进入 CLIP Text Encoder，产生领域语义特征，进而通过 cross-modal alignment 约束 domain prompt。

这一步的教育性重点是：MLLM 负责把视觉领域知识蒸馏进 prompt，而不是在每次推理时阅读图片并生成恢复结果。论文明确说明 LLaVA 与 CLIP 在 inference 阶段不使用。

### 4.2 两个 prompt pool 如何查询

每个 prompt 是 key-value 对：key 用于匹配输入，value 携带真正注入网络的表示。

**Task Prompt Pool：**

1. encoder 的中间特征 `F_mid` 经过 projector 得到 query `Q_task`。
2. 与所有 task keys 做余弦相似度。
3. 选 top-k task values。
4. 用相似度 softmax 权重组合成实例级 task representation `PR_t`。

论文的组合公式可读成：

```text
α_j = softmax(sim(Q_task, K_j) / T_task)
PR_t = Σ_j α_j V_j
```

因此，prompt pool 不是“一个任务对应一个向量”的查表，而是多个 prompt 的软混合。这样既能保留任务特定信息，也能让相近任务共享部分 prompt。

**Domain Prompt Pool：**

1. 第一层浅层特征经过另一个 projector 得到 `Q_dom`。
2. 查询 domain keys，选择 top-k domain values。
3. 通过同样的 PCM 组合成 `PR_d`。
4. 用 LLaVA/CLIP 文本特征对 `PR_d` 加跨模态对齐损失，使它不只是任意可学习向量，而更可能编码领域先验。

浅层特征适合看颜色、灰度、视角等域属性；中间特征更适合判断具体恢复任务，这是一个结构上合理的分工。

### 4.3 中间：Domain-Aware Task Representation

`PR_t` 和 `PR_d` 通过 cross-attention 融合成 `PR_dt`。它不是简单拼接，因为 cross-attention 允许一类表示选择性读取另一类表示，形成“这个领域中的这个任务”的条件。

随后在骨干的不同层使用 Adaptive Gated Fusion：

```text
F_el = CrossAttn(α_l F_l, (1 - α_l) PR_dt)
```

`α_l` 是每层可学习门控系数。不要把它理解成全网络一个固定比例：不同层可能更依赖原始恢复特征，也可能更需要 prompt 条件。论文分析发现多数层仍主要依赖 backbone，prompt 更多是辅助指导；较早的块往往更依赖 prompt。

### 4.4 Prompt 正则化解决什么问题

双 prompt pool 可能出现三个退化：所有 prompt 学成相似内容、模型只依赖少数 prompt、不同输入选择趋同。论文因此加入：

- **Diversity regularization**：惩罚 prompt value 两两相似度过高，防止表示塌缩。
- **Balance/entropy regularization**：鼓励 prompt 被更均衡地使用。
- **Contrastive regularization**：增强实例级 prompt 选择的敏感性，细节在附录。

## 5. 论文创新点

### 创新 1：首次把 AiOIR 推向多域统一设置

论文声称这是首个 multi-domain all-in-one image restoration 探索。真正需要学习的不只是“去什么退化”，还有医学、自然、遥感的成像风格、颜色和结构差异。

### 创新 2：任务知识与领域知识解耦但可融合

只用 task prompt 容易忽略域差异，只用 domain prompt 又不足以决定恢复操作。双池把“做什么”和“在哪里做”拆开，再通过 cross-attention 组合。

### 创新 3：从固定条件改成实例级检索与组合

同一个 domain 内的图像并不完全相同，同一个 task 也可能跨域。top-k + softmax PCM 允许 prompt 表示随实例改变。

### 创新 4：用 MLLM 蒸馏领域先验且不增加推理成本

MLLM 只在训练期间提供文本语义约束，部署时依赖轻量 projector、prompt pool 和恢复 backbone。这是“训练期用大模型，推理期去掉大模型”的典型思路。

## 6. 实验解读

- 在 6-task/3-domain 设置中，DATPRL-IR-6T 平均 PSNR/SSIM 为 30.77/0.8653，高于表中的 MoCEIR 30.40/0.8627。
- 从 6-task 扩展到 9-task，原有任务没有明显退化，论文据此支持“任务之间存在可迁移共享知识”。这比只展示一个任务上的最好分数更能支撑方法动机。
- TP 与 DP 单独使用都能提高性能，二者同时使用最好。例如表 2 中 Rain100L 去雨的 PSNR 从无 prompt 的 38.34 提高到双池的 39.56。
- prompt 数量和 top-k 存在折中：太少表达能力不足，太多会引入冗余；论文设置 task/domain prompt 数均为 15，top-k 分别为 3/5。
- 更换 LLaVA-1.5-13B 或 Qwen3-VL-2B-Instruct 后结果变化很小，说明该方法主要需要粗粒度域描述，并非强依赖某个特定 MLLM。

## 7. 批判性阅读

1. **“领域”与“任务”并不总能严格分离。** 医学 MRI SR 和自然图像 SR 的恢复难点可能已经交织，双池是建模假设，不是物理定律。
2. **prompt pool 的语义可解释性有限。** 虽然 domain prompt 受 CLIP 对齐约束，但每个 prompt 未必能被人直接命名为“医学 prompt”。
3. **实验依赖多套数据集与合成退化。** 跨域性能提高可能部分来自数据配比、任务互补或训练设置，需进一步看 zero-shot 与真实混合退化实验。
4. **PSNR/SSIM 不能覆盖医学安全性。** 医学图像的视觉清晰不等价于诊断可靠，部署时还需任务特定验证。

## 8. 学习问题

- 如果把 domain prompt pool 改成一个显式域标签，会损失什么？答案是失去盲恢复和域内实例差异，也不能自然处理未知域。
- 如果把 task/domain representation 直接相加，为什么可能不如 cross-attention？因为加法没有显式建模两种知识之间的条件读取关系。
- 为什么 prompt pool 需要 diversity 与 balance？因为可学习的 key-value 检索很容易只使用少量 prompt，最终退化为一个小字典或单 prompt 模型。

## 9. 按全文顺序通读

### 摘要与引言：从 AiOIR 到 MD-AiOIR

论文首先区分三个层次：单任务恢复、单域 all-in-one 恢复、多域 all-in-one 恢复。早期研究为自然图像超分、去雨、去模糊，CT 去噪、MRI SR、PET synthesis，以及遥感 SR、云去除、去雾分别训练模型。这样做虽然目标清楚，却带来模型数量和数据训练成本的线性增长。

AiOIR 试图用一个网络覆盖多个任务，但已有方法主要解决“同一域里如何分辨退化”。当自然、医学、遥感数据放在一起时，难点不只增加了任务数量，还增加了成像域差异。论文的核心观察是：不同任务和域不是完全独立的，它们既有共享特征，也有独有特征。prompt pool 正好可以作为一个可检索的知识容器。

引言中的两个例子很关键：

- “grayscale + human organs” 组合提示医学域；
- “bird’s-eye view + buildings” 组合提示遥感域。

这说明 domain representation 不需要只存一个硬标签，而可以由多个视觉属性组合出来。随后论文提出 DATPRL，并把 task representation 与 domain representation 融合为 domain-aware task representation。

### 相关工作逐段理解

**Single-Task Restoration** 回顾 CNN、Transformer、Mamba 等骨干，说明恢复网络本身已经很成熟，论文不打算重新设计一个新的 encoder-decoder。

**All-in-One Restoration** 介绍 AirNet 的对比学习、IDR 的退化先验、PromptIR 的可学习 prompt、DA-CLIP 的 degradation/content 解耦、InstructIR 的人工指令、MoCEIR 的 mixture-of-experts 和 DCPT 的退化分类。它们共同证明“条件信息有助于多任务恢复”，但大多仍在单域内工作。

**Prompt Learning-based Restoration** 说明 prompt 已从 NLP 迁移到图像恢复。DATPRL 与已有 PromptIR 的主要差异不是“也用了 prompt”，而是采用双池、实例级 query-retrieval-composition 和 MLLM 蒸馏的域先验。

## 10. 方法的完整训练逻辑

### 10.1 Task Prompt Pool

任务池中有 `N_t` 个 key-value 对 `(K_j^task, V_j^task)`。输入图像先经过恢复 backbone，取得中间特征 `F_mid`。轻量 projector（3 层 CNN，包含 Conv2d、AdaptiveAvgPool2d、MLP）把它映射到 1024 维 `Q_task`。

对每个 key 计算余弦相似度，选 top-3 values。温度参数 `T_task` 控制 softmax 的尖锐程度：温度小，模型更像选择一个 prompt；温度大，多个 prompt 会更平均地混合。最终的 `PR_t` 是选中 values 的加权和。

这里的“共享知识”不是另设一个 shared prompt，而是通过不同实例反复选中相同 prompt、以及 prompt pool 的共同训练自然形成。若去雨和去模糊都选中某些 prompt，它们就可能承载结构恢复的共享部分。

### 10.2 Domain Prompt Pool 与 MLLM 蒸馏

domain pool 同样由 15 个 key-value 对构成，但 query 来自 backbone 第一层浅层特征 `F_sha`。浅层特征保留颜色、灰度、视角、纹理等域线索，适合做域检索；本文 top-5 domain prompt 的检索宽度比 task pool 更大，是为了覆盖同一域内部的多个视觉属性。

训练时，LLaVA-1.5-7B 对对应的高质量图像生成描述，例如：

- 医学：黑白脑部、腹部、低对比度和局部亮点；
- 遥感：俯视森林、河流、高速公路、建筑和泳池；
- 自然：人物、花园、海岸等。

描述进入 CLIP text encoder 得到 `F_text`。domain representation 与文本特征的 cosine alignment loss 鼓励 domain pool 捕获“内容 + 颜色 + 视角 + 亮度”等域信息。值得注意的是，图像输入是高质量对应图像的描述，而不是让 MLLM直接解释退化图像；这更稳定地提供域语义。

### 10.3 Domain-aware Task Representation

论文没有把 `PR_t` 与 `PR_d` 直接相加，而是通过 cross-attention 生成 `PR_dt`。这样任务 prompt 可以根据 domain prompt 有条件地读取信息，形成“遥感去雾”与“自然去雾”不同的组合表示。

之后采用 adaptive gated fusion。每个 layer 有独立 `α_l`，控制 backbone feature 与 prompt representation 的贡献。附录 Figure 6(c) 显示多数层主要依赖 backbone，说明 prompt 不是要替换主干；较早层更依赖 prompt，深层逐渐回到恢复特征。这是一个很有价值的设计分析：条件信息不必在所有层同等强度地注入。

### 10.4 三种正则化如何防止 prompt 失效

1. **Diversity loss**：计算 value 之间的 pairwise cosine similarity，超过阈值 `τ_div=0.1` 才惩罚，避免 prompt 学成一团。
2. **Balance loss**：对 query 到所有 keys 的 softmax 概率计算熵，用 `log P-H(p)` 鼓励各 prompt 有机会被使用。
3. **Contrastive loss**：把 query 拉近被选中的正 key，推远未选中的负 key，提升实例级检索敏感度。

总损失为：

```text
L = λ_pix L_pix + λ_fft L_fft + λ_align L_align
    + λ_div L_div + λ_bal L_bal + λ_con L_con
```

其中 `L_pix` 是 RGB 域 L1，`L_fft` 是 Fourier 域 L1；其余项分别约束跨模态对齐、prompt 多样性、使用均衡和 query-key 可分性。

## 11. 实验设置的完整阅读

论文用 3 个域、6 个任务做主实验：自然域 SR/去雨，医学域 MRI SR/CT 去噪，遥感域 RSI SR/云去除；再加入自然去模糊、PET synthesis、遥感去雾形成 9-task 设置。

数据集不是一个统一的大数据集，而是多来源任务数据：DF2K、Rain100L、GoPro、IXI MRI、AAPM-Mayo、PolarStar m660、UCMerced、CUHK CR1、RICE1。由于医学数据原始格式不同，论文把非 RGB 医学图像转灰度，再复制成 3 通道，以便输入统一网络。这是工程统一，不代表医学图像真的拥有 RGB 颜色语义。

训练使用 Adam，`β1=0.9`、`β2=0.99`、初始学习率 `4×10^-4`、cosine annealing、batch size 12、1000K iterations，RTX 5090。prompt key 为 1×1024，value 为 2×1024，两个 pool 均为 15 个 prompt，task/domain top-k 为 3/5。

### 11.1 6-task 与 9-task 结果

主表中，DATPRL-IR-6T 平均 PSNR/SSIM 为 30.77/0.8653，优于 MoCEIR 的 30.40/0.8627。9-task 表中，DATPRL-IR-9T 在自然 SR/去雨/去模糊为 29.05/0.8181、39.67/0.9867、29.57/0.8881；医学 CT 去噪 33.77/0.9273、PET synthesis 37.12/0.9502；遥感 SR、云去除、去雾分别为 28.31/0.7913、26.00/0.7592、26.94/0.9347。

作者重点观察到：任务从 6 增到 9 后，原任务没有显著下降，部分还提升。这支撑了“共享知识帮助多任务”的主张，但也要考虑训练数据总量、采样策略与更多任务带来的正则化效应。

### 11.2 双 pool 消融

只用 task pool、只用 domain pool、二者同时用都优于无 prompt；二者同时用最好。这个消融比只比较“有无整个模型”更能对应方法动机：任务先验和领域先验确实承担不同作用。

### 11.3 prompt 数和 top-k

10/15/20 个 prompt 与不同 top-k 的实验显示中等容量最好：pool 太小表达不足，太大可能引入冗余；top-k 太小不够灵活，太大又降低特异性。15 + task top-3 + domain top-5 是一个经验折中，不是理论最优。

### 11.4 zero-shot 与跨分布泛化

附录是全文中最能支撑“多域”价值的部分：

- **未见图像域**：AIGC 的 SDXL-Turbo、DeepFloyd、Midjourney，以及漫画 Manga109。DATPRL-IR 在这些域上无需微调仍优于多种基线。
- **完全未见恢复任务**：SHIQ 高光去除和 Snow100K-S 去雪。DATPRL-IR 在表 10 中分别达到 26.22/0.9336 与 27.20/0.9018 的 PSNR/SSIM，超过 MoCEIR。
- **未见分布**：真实去雨 SPA-Data 与遥感去模糊 UCMerced。遥感去模糊达到 28.91/0.8108，明显高于 MoCEIR 的 27.35/0.7956。

这些结果说明 prompt pool 可能学到了可迁移的恢复/域表示，但“未见任务零样本成功”仍需谨慎解释：不同恢复任务共享低层图像结构，且实验中的输入形式、损失和 backbone 对迁移很重要。

### 11.5 感知质量指标的适用边界

附录只在自然图像任务上报告 CLIPIQA、MANIQA、MUSIQ、NIQE、FID，因为这些指标主要在自然图像上预训练，直接用于医学和遥感可能没有可靠解释。这是论文中很值得学习的实验规范：不是所有指标都应该跨域套用。

### 11.6 融合策略消融

附录比较“domain → image → task”的顺序融合与本文的“先 task/domain 融合成 `PR_dt`，再与 image 自适应融合”。顺序融合在多个任务上有不错结果，但整体下降；作者解释为 task 和 domain 在 MD-AiOIR 中同等重要，先构成 domain-aware task representation 更能保持二者平衡。

## 12. 全文结论与局限

DATPRL-IR 的完整贡献不是简单增加两个 prompt 表，而是建立了一条知识流：

```text
图像浅层/中层特征
   → query 两个 prompt pool
   → top-k 检索 + PCM
   → task/domain cross-attention
   → AGF 注入恢复 backbone
   → RGB/Fourier 重建
```

局限也在附录中被承认：训练成本高于单域/单任务模型，扩展到更多域仍有学习难度，prompt 的共享/特定知识还缺少更强解释性。医疗任务把灰度复制成 3 通道也只是统一接口，不能替代 3D/物理成像模型。
