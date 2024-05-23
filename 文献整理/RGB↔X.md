https://arxiv.org/abs/2405.00666

https://doi.org/10.1145/3641519.3657445

![Overview of Model](https://github.com/hongsi466474/A4MD/blob/6f0668dc2f3bbf5b0970b892b07bab3c79bc2f92/%E5%9B%BE%E7%89%87/RGB%E2%86%94X/IMG_0278.jpeg?raw=true)

# RGB↔X: Image decomposition and synthesis using material- and lighting-aware diffusion models

[Zheng Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng,+Z), [Valentin Deschaintre](https://arxiv.org/search/cs?searchtype=author&query=Deschaintre,+V), [Iliyan Georgiev](https://arxiv.org/search/cs?searchtype=author&query=Georgiev,+I), [Yannick Hold-Geoffroy](https://arxiv.org/search/cs?searchtype=author&query=Hold-Geoffroy,+Y), [Yiwei Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu,+Y), [Fujun Luan](https://arxiv.org/search/cs?searchtype=author&query=Luan,+F), [Ling-Qi Yan](https://arxiv.org/search/cs?searchtype=author&query=Yan,+L), [Miloš Hašan](https://arxiv.org/search/cs?searchtype=author&query=Ha%C5%A1an,+M)

**Adobe** Research || University of California, **Santa Barbara** USA

**SIGGRAPH** Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA

CCS Concepts：计算方法 $\to$ 渲染

关键词：扩散模型 本征分解 逼真渲染

# 动机

- 传统渲染：精准但需要完整的场景描述
- 扩散模型：易用、真实但难精确控制

<!--- 探索 *扩散模型* 、 *渲染* 和 *固有通道估计* 之间的联系，重点关注材质/光线估计和以材质/光线为条件的图像合成，所有这些都在同一个扩散框架中进行。 --->

## 思想：用简单的东西来控制

- 探索一个中间域：
	- 仅指定某些外观属性
	- 给模型自由：对剩余部分能生成一个合理版本的假象
- **X**：诸如漫反射反照率、镜面粗糙度和金属度，以及各种空间变化的照明表示之类的 **信息缓冲区（information buffers）** 称为 **固有通道（intrinsic channels）**

- **RGB→X**问题：从RGB图片估计固有道的问题

- **X→RGB**问题：根据给定描述合成图像的问题

## 目标

- 探究扩散模型、渲染和固有通道估计之间的关系
- 关注两个问题：
	- RGB→X：固有通道估计
	- X→RGB：用固有通道控制图生成

# 贡献

提出一个基于扩散的统一架构，它可以实现逼真的图像分析（描述几何、材料和照明信息的内在通道估计）和合成（给定内在通道的逼真渲染），并在逼真的室内场景图像领域进行了演示，如图。

![图一](https://github.com/hongsi466474/A4MD/blob/6f0668dc2f3bbf5b0970b892b07bab3c79bc2f92/%E5%9B%BE%E7%89%87/RGB%E2%86%94X/IMG_0277.jpeg?raw=true)

> 我们的工作是实现图像分解和合成统一框架的第一步。
> 
> 我们相信，它能为各种下游编辑任务带来益处，包括材质编辑、重新照明以及根据简单/未充分指定的场景定义进行逼真渲染。

- 一个 RGB→X 模型，通过使用来自多个异构数据集的更多训练数据并增加对照明估计的支持，改进了之前的工作 [Kocsis等人，2023]；
- 一个 X→RGB 模型，能够从给定的固有通道 X 合成逼真的图像，支持部分信息和可选的文本提示。文章结合了现有的数据集，并添加了一个新的、高质量的室内场景数据集，以达到高度逼真的效果。

# 方法

## 固有通道和数据集

### 固有通道
- **法向量** $\mathbf{n}\in\mathbb{R}^{H\times W\times 3}$ ：在相机空间中指定几何信息;
- **反照率** $\mathbf{a}\in\mathbb{R}^{H\times W\times 3}$ ：通常也称为基色，规定了电介质不透明表面的漫反射反照率和金属表面的镜面反射反照率；
- **粗糙度** $\mathbf{r}\in\mathbb{R}^{H\times W}$ ：通常理解为 GGX 或贝克曼微切面分布中参数 $\alpha$ 的平方根。高粗糙度意味着材料更无光泽，而低粗糙度则意味着更闪亮；
- **金属性** $\mathbf{m}\in\mathbb{R}^{H\times W}$ ：通常定义为介于电介质表面和金属表面之间的线性混合权重；
- **漫射辐照度** $\mathbf{E}\in\mathbb{R}^{H\times W\times 3}$ ：用作光照表示。它表示到达表面点的光量在上余弦加权半球上的积分。

> 玻璃：零粗糙度和金属性

### 数据集

![表一](https://github.com/hongsi466474/A4MD/blob/6f0668dc2f3bbf5b0970b892b07bab3c79bc2f92/%E5%9B%BE%E7%89%87/RGB%E2%86%94X/IMG_0279.jpeg?raw=true)

## 模型

![图二](https://github.com/hongsi466474/A4MD/blob/6f0668dc2f3bbf5b0970b892b07bab3c79bc2f92/%E5%9B%BE%E7%89%87/RGB%E2%86%94X/IMG_0278.jpeg?raw=true)

### RGB→X 模型

- 对预先训练好的 stable diffusion 微调
- 核心思想：
	- 将输入文本提示符重新用作“开关”来控制输出，
	- 一次产生一个固有通道
- 两个好处：
	- 允许使用异构数据集的混合，这些数据集在可用通道上有所不同
		- 例如，仍然可以使用只有反照率通道的数据集来训练模型
		- 大规模扩大了可用的训练数据集
	- 避免处理多个输出通道
		- 事实证明，这让训练更具挑战性

![](https://github.com/hongsi466474/A4MD/blob/15dd5360cadbc8cbe47a4db2b963235cab57ad4b/%E5%9B%BE%E7%89%87/RGB%E2%86%94X/RGB%E2%80%94%3EX.jpeg?raw=true)

#### 损失函数：
```math
\begin{gather}
\mathbf{v}^{RGB\to X}_{t}=\sqrt{ \bar{\alpha}_{t} }\epsilon-\sqrt{ 1-\bar{\alpha}_{t} }\mathbf{z}^X_{0}  \tag{1} \\
L_{\theta}=\Vert\mathbf{v}^{RGB\to X}_{t}-\hat{\mathbf{v}}^{RGB\to X}_{\theta}(t,\mathbf{z}^X_{t},\mathcal{E}(\mathbf{I}),\tau(\mathbf{p}^X))\Vert^2_{2}. \tag{2}
\end{gather}$$
```

- $t$ 为时间步（在训练过程中均匀提取的噪声量）
- $\epsilon\in\mathcal{N}(0,1)$ ，$\bar{\alpha}_t$ 是 $t$ 的标量函数
- $\hat{\mathbf{v}}^{RGB\to X}_{\theta}$ 是带参数 $\theta$ 的RGB→X扩散模型
- $\mathbf{z}^X_0$ 为目标 latent ，$\mathbf{z}^X_t$ 为在第 $t$ 步对 $\mathbf{z}^X_0$ 噪声 $\epsilon$ 的噪声 latent
- $\mathcal{E}$ 为原始 stable diffusion 模型的编码器
- $\mathbf{p}^X$ 是为 $\mathbf{I}$ 计算的文本提示（prompt）
- $\tau$ 为 CLIP 文本编码器：将prompt编码为文本嵌入向量（textembedding vector）

### X→RGB 模型

- 对预先训练好的 stable diffusion 微调
- 核心思想：
	- 通道 drop-out 策略：在训练期间随机丢弃条件通道。
		- 例如，以0.3的概率丢弃反照率通道
	- 联合训练有条件和无条件的扩散模型
- 两个好处：
	- 允许使用异构数据集的混合，这些数据集在可用通道上有所不同
	- 让图像生成胜任任何条件的子集
		- 例如，不提供反照率或照明将导致模型产生似是而非的图像，使用其之前来补偿缺失的信息
 
![](https://github.com/hongsi466474/A4MD/blob/6f0668dc2f3bbf5b0970b892b07bab3c79bc2f92/%E5%9B%BE%E7%89%87/RGB%E2%86%94X/X-%3ERGB.jpeg?raw=true)

#### 损失函数：
```math
\begin{gather}
\mathbf{v}^{X\to RGB}_{t}=\sqrt{ \bar{\alpha}_{t} }\epsilon-\sqrt{ 1-\bar{\alpha}_{t} }\mathbf{z}^{RGB}_{0}  \tag{4} \\
L'_{\theta}=\Big\Vert\mathbf{v}^{X\to RGB}_{t}-\hat{\mathbf{v}}^{X\to RGB}_{\theta}\Big(t,\mathbf{z}^{RGB}_{t},\mathbf{z}^X_{t},\tau(\mathbf{p})\Big)\Big\Vert^2_{2}. \tag{5}
\end{gather}
```

- $\hat{\mathbf{v}}^{X\to RGB}_{\theta}$ 是带参数 $\theta$ 的X→RGB扩散模型
- $\mathbf{z}^{RGB}_0=\mathcal{E}(\mathbf{I})$ 为目标 latent，
- $\mathbf{z}^X_t=(\mathcal{P}(\mathbf{n}),\mathcal{P}(\mathbf{a}),\mathcal{P}(\mathbf{r}),\mathcal{P}(\mathbf{m}),\mathcal{P}(\mathbf{E}))，其中\mathcal{P}(x)\in\{\mathcal{E}(x),0\}$ 【通道drop-out】
- CLIP 文本嵌入 $\tau(\mathbf{p})$ 用作模型交叉注意力层的上下文

# 结果

## RGB→X

![](https://github.com/hongsi466474/A4MD/blob/6f0668dc2f3bbf5b0970b892b07bab3c79bc2f92/%E5%9B%BE%E7%89%87/RGB%E2%86%94X/IMG_0085.jpeg?raw=true)

![](https://github.com/hongsi466474/A4MD/blob/6f0668dc2f3bbf5b0970b892b07bab3c79bc2f92/%E5%9B%BE%E7%89%87/RGB%E2%86%94X/IMG_0282.jpeg?raw=true)

## X→RGB

![](https://github.com/hongsi466474/A4MD/blob/6f0668dc2f3bbf5b0970b892b07bab3c79bc2f92/%E5%9B%BE%E7%89%87/RGB%E2%86%94X/IMG_0283.jpeg?raw=ture)
