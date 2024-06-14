https://arxiv.org/abs/2406.02968

http://hse1032.github.io.gsgan.html

![Fig3](https://github.com/hongsi466474/A4MD/blob/e51399c3cc2eb9ce4e2295bca33bc6e06806ba70/3DGS/GS-GAN/%E6%88%AA%E5%9B%BE/Fig3.png?raw=true)

# Adversarial Generation of Hierarchical Gaussians  for 3D Generative Model

[Sangeek Hyun](https://arxiv.org/search/cs?searchtype=author&query=Hyun,+S), [Jae-Pil Heo](https://arxiv.org/search/cs?searchtype=author&query=Heo,+J)

Sungkyunkwan University（韩国首尔私立高校）

自添关键词： #3D高斯 #高斯喷溅 #GAN/3D-GANs #分层高斯 

# 动机

- 3D-GANs ：体积渲染使用的是光线投射，计算成本高，限制分辨率
- 3DGS ：渲染快，表示不可微，有场景依赖性，难应用于 3D-GANs
- 💡：将 3D 高斯表示的光栅化应用扩展到 3D-GANs，利用其高效渲染速度处理高分辨率数据
	- ×：在对抗学习框架中，简单合成一组高斯而不加约束的原始生成器架构存在训练不稳定和高斯缩放调整不精确的问题。
		- E.g. 训练早期，所有高斯在渲染图中消失；虽然收敛但长形高斯产生伪影
	- √：对高斯表示进行正则化，尤其侧重于位置和缩放参数
		- 分层高斯表示的生成器架构：将香菱层级的高斯建模为依赖关系，鼓励生成器以“粗到细”方式合成 3D 空间

![Fig1](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-GAN/%E6%88%AA%E5%9B%BE/Fig1.png?raw=true)

# 贡献

- 首先利用基于光栅化的 3D 高斯表示法实现高效的 3D-GANs
- 引入分层 3D 高斯表示法：在相邻层级的高斯之间建立依赖关系来正则化高斯参数，从而稳定 3DGS 的 3D-GANs 的训练
- 与目前最先进的 3D-GANs 相比，本文方法大大提高熏染速度，同时保持了相当的生成质量和 3D 一致性

# 方法

## 前置知识

### 3D 生成对抗网络（3D-GANs）

-  3D 生成器 $g(z,\theta)$ 和 2D 鉴别器 $d(I)$ 之间的对抗学习架构
	- latent code $z\in\mathbb{R}^{d_z}\sim p_z$ ，相机位置 $\theta\in\mathbb{R}^{d_\theta}\sim p_\theta$ 

### 3DGS
- $$G(x)=e^{-\frac{1}{2}(x-\mu)^\top\Sigma(x-\mu)},\Sigma=RSS^\top R^\top \tag{1}$$
- $$C=\sum^N _{i=1}c _{i}\alpha _{i}\prod^{i-1} _{j=1}(1-\alpha _{j}') \tag{2}$$

## 分层 3D 高斯表示

 ![Fig2](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-GAN/%E6%88%AA%E5%9B%BE/Fig2.png?raw=true)
 
- 从粗到细的层次结构级别 $l\in\{1,\dots,L\}$ ，其中每个级别包含一组高斯参数，不同层级的高斯负责不同级别的细节
- 相近层级的高斯参数间建立依赖关系： $G^l$ 起源于 $G^{l-1}$ 
	- 位置参数：
		-  局部约束，将更精细级高斯 $G^l$ 的位置 $\mu^l$ 与其对应的 $G^{l-1}$ 绑在一起
		- $$\mu^{l}= \mu^{l-1}+R^{l-1}S^{l-1}\hat{\mu}^l \tag{3}$$
			-  $R^{l-1}$ 和 $S^{l-1}$ 为由 $q^{l-1}$ 和 $s^{l-1}$ 获得的旋转和缩放矩阵
			-  $\hat{\mu}^l$ 为引局部坐标系定义的局部位置参数，它根据粗级高斯 $G^{l-1}$ 的位置  $\mu^{l-1}$ 定心，缩放比例 $s^{l-1}$ 缩放， $q^{l-1}$ 旋转
		- 这一操作确保了较细级别的高斯的位置取决于较粗级别高斯，同时位于较粗级别对应高斯的附近
	- 缩放参数：
		- 
		- $$s^l=s^{l-1}+\hat{s}^l+\Delta s,其中 \hat{s}^l，\Delta s<0 \tag{4}$$
			- 细层级缩放 $s^l$ 由相对于粗层级缩放 $s^{l-1}$ 的相对缩放差 $\hat{s}^l$ 定义
			- 限制粗层级缩放 $s^{l-1}$ 始终为负值向量
			- 常数 $\Delta s$ 能进一步降低细层级的缩放
	- 其他参数：
		- 在 $l$ 层额外定义了残差高斯参数 $\hat{\alpha}^l,\hat{q}^l,\hat{c}^l$ ，并将其添加到上一层高斯参数中：
		- $$\alpha^l=\alpha^{l-1}+\hat{\alpha}^l,q^l=q^{l-1}+\hat{q}^l,c^l=c^{l-1}+\hat{c}^l \tag{5}$$
		- 称 $G^{l-1}$ 与残差参数 $\{\hat{\mu}^l,\hat{s}^l,\hat{\alpha}^l,\hat{q}^l,\hat{c}^l\}\in\hat{G}^l$ 之间的层级关系为 $\texttt{densify}(G^{l-1},\hat{G}^l)$ ，即公式(3)(4)(5)提及的分层相邻的高斯中参数的组合过程
			- 使生成器能以从粗到细的方式对 3D 空间建模，其中细层级高斯描绘了粗层级高斯的细节部分
			- 通过大幅减少高斯可能的位置来稳定 GAN 的训练，鼓励生成器使用各种尺度的高斯，提高生成不同层级细节的能力

## 分层 3D 高斯的生成器架构

![Fig3](https://github.com/hongsi466474/A4MD/blob/e51399c3cc2eb9ce4e2295bca33bc6e06806ba70/3DGS/GS-GAN/%E6%88%AA%E5%9B%BE/Fig3.png?raw=true)

- 基于 Transformer 的架构
- 生成器 $g(z,\theta)$ 表示一系列生成器块，$\texttt{block}_{l}$ 表示层级 $l$ 的生成器块
- $\texttt{block}_{0}$ ：输入 $N$ 个可学习位置 $\texttt{const}\in\mathbb{R}^{N\times3}$ ，和 latent code $z$ ，输出高纬特征 $x^0$ 
	- $N$ 为初始高斯数
	- 通过输出层 $\texttt{toGaus}_0$ 处理高斯的第 $i$ 个特征 $x^0_i$ ，得到高斯参数 $\{\mu^0_i,s^0_i,q^0_i,\alpha^0_i,c^0_i\}\subset G^0_i$ 
- $\texttt{block}_{l}$ ：输入前一块的特征 $x^{l-1}_i$ 和 latent code $z$ ，输出特征 $x^l_j$ 
	- 输出块 $\texttt{toGaus}_l$ 不直接合成高斯参数 $G^l_j$ 
	- 中间输出 $\hat{G}^l_j\supset\{\hat{\mu}^l_{j},\hat{s}^l_{j},\hat{q}^0_{j},\hat{\alpha}^0_{j},\hat{c}^0_{j}\}$ 与上一层级相应的高斯 $G^{l-1}_i$ 结合，最终合成第 $l$ 级的高斯 $G^l_j$ 
- 这一过程建立了相邻层级 $G^{l-1}_i$ 和 $G^l_j$ 的高斯之间的层级依赖关系：
	- $$x^l_{j}=\texttt{block}_{l}(x^{l-1}_{i},z),\hat{G}^l_{j}=\texttt{toGaus}_{l}(x^l_{j}),G^l_{j}=\texttt{densify}(G^{l-1}_{i},\hat{G}^l_{j}) \tag{6}$$
- 细层级的尺寸更小，故数量应该随层级增大而增多，以合成更好的细节
	- 扩大高斯参数 $G^l_j$ 的数量，使每个参数共有 $r^lN$ 个向量
		- 即，让 $G^l_j$ 依赖 $G^{l-1}_i$ ，其中 $j=ri+k,k\in\{0,1,\dots,r-1\}$ ，$r$ 为上采样率
- 合成每一层级的高斯参数后，将它们全部用于生成图像
	- 渲染高斯个数： $N+rN+\dots+r^{L-1}N$ 
	- 同 3DGS 使用基于瓦片的光栅化

### 锚定高斯以分解高斯，用于正则化和渲染

- 上述架构中，高斯不仅用于表示 3D 场景，还用于正则化粗层级高斯
	- 必须对高斯训练，以精确地引导细层级高斯的 5 个参数，同时描绘真实世界中的清晰细节
	- ×：高斯的缩放在特定轴线上可能几乎为零，导致对细层级高斯的位置进行过强的正则化
	- √：引入一组辅助高斯，只用于帮助正则化，不参与渲染
- $$[\hat{G}^l_{j},\hat{A}^l_{j}]=\texttt{toGaus}_{l}(x^l_{j}),G^l_{j}=\texttt{densify}(A^{l-1}_{i},\hat{G}^l_{j}),A^l_{j}=\texttt{densify}(A^{l-1}_{i},\hat{A}^l_{j}) \tag{7}$$
	-  $\hat{G}^l_{j},\hat{A}^l_{j}$ 为给定特征 $x^l_j$ 下 $\texttt{toGaus}_{l}$ 合成的两组高斯
	- $A^l$ 为层级 $l$ 的锚点高斯，其参数化与 $G^l$ 相同
		- 只用于正则化，作为 $\texttt{densfy}$ 的输入，尤其是粗层级的高斯输入
		- 锚点高斯 $A^{l-1}_i$ 只学习指导其对应细层级高斯 $G^l_j$ 的参数
		- 可缓解零方差导致的强正则化效果
	- $l=0$ 时，令 $A^0_{i}=\hat{A}^0_{i},G^0_{i}=\texttt{densify}(A^0_{i},\hat{G}^0_{i})$ 

### 架构细节

- 利用映射网络将 latent code $z$ 修改为 style code $w$ ，它会影响 AdaIN 的合成过程
- 用 AdaIN 替换注意力层和 MLP 层的 layer norm 
- 粗层级的区块，用不含位置编码的一般自注意力机制；细层级区块，用局部注意力机制
	- $r^{l-1}N$ 个点之间的迭代对计算要求较高
- 对于在最粗层级之后扩展生成器中的特征，只需使用带跳过连接的子像素操作，重复高斯 $G^l$ $r$ 次
- 使用 layerscale ：一个可学习的向量，通过乘以残差块的输出来调整残差块的效果
	- 通常情况下，它有一个零值向量初始化，从而在训练早期消除层的影响
- 将相机方向作为 $\texttt{toGaus}$ 中颜色层的一个条件，以模拟视角相关特性
- 采用背景生成器
	- 与生成器架构相似，容量有所减小，使得高斯位于半径为 3 的球中，前景位于 $[-1,1]$ 的立方中

## 训练目标

- 带 $R1$ 正则的非饱和对抗损失：
	- $$\mathcal{L}_{adv}=\mathbb{E}_{z\sim P_{z},\theta\sim P_{\theta}}[f(d(g(z,\theta)))]+\mathbb{E}_{I_{r}\sim P_{I_{r}}}[f(-d(I_{r}))+\lambda\Vert\nabla d(I_{r})\Vert^2] \tag{8}$$
		- 软加函数 $f(t)=-\log(1+\exp(-t))$ ，$\lambda$ 为 $R1$ 正则化强度
- 通过在图中获得的位置嵌入和相机参数之间引入对比学习，为鉴别器和生成器提供 3D 信息：
	- 鉴别器的位置分支 $d_p$ 能估计输入图像的位置嵌入 $p_I$ 
	- 引入由 MLP 层组成的位置编码器，将相机参数 $\theta$ 编码到位置嵌入 $p_\theta$ 中
	- 利用对比目标来增强相应 $p_I$ 和 $p_\theta$ 之间的相似性
	- $$\mathcal{L}_{pose}=-\log\left( \frac{\exp\left( \frac{\text{sim}(p_{I},p^+_{\theta})}{\tau} \right)}{\sum^B_{b=1}\left( \exp\left( \frac{\text{sim}(p_{I},p^b_{\theta})}{\tau} \right) \right)} \right) \tag{9}$$
		- $\text{sim}(\cdot,\cdot)$ 为余弦相似度，$B$ 为 batch 大小
		- $p^+_{\theta}$ 为与图像中位置嵌入 $p_I$ 对应的正样本，$\tau$ 为温度比例参数
		- 用真实数计算鉴别器的 $\mathcal{L}_{pose}$ ，用假数据训练生成器
- 引入两种损失以规范锚点高斯在最粗层级 $\mu^0_A$ 中的的位置：
	- 将 $\mu^0_A$ 的平均位置正则化为零，鼓励高斯中心位于时间空间的原点附近
	- 减小 $K$ 个最近的锚点高斯之间的位置距离，防止锚点高斯和其他高斯分离
	- $$\mathcal{L}_{center}=\frac{1}{N}\Vert\sum^N_{j=1}\mu^0_{A,j}\Vert^2,\mathcal{L}_{knn}=\frac{1}{NK}\sum^N_{j=1}\Vert\sum^K_{k=1}(\mu^0_{A,j}-\text{KNN}(\mu^0_{A,j},k))\Vert^2 \tag{10}$$
		- $\text{KNN}(\mu^0_{A,j},k)$ 为 $\mu^0_{A}$ 的第 $j$ 个锚点高斯的第 $k$ 个临近位置
- 总损失：
- $$\mathcal{L}=\mathcal{L}_{adv}+\lambda_{pose}\mathcal{L}_{pose}+\lambda_{center}\mathcal{L}_{center}+\lambda_{knn}\mathcal{L}_{knn} \tag{11}$$
	- $\lambda_*$ 为对应目标函数的强度

# 实验

## 实验设置

- 数据集：分辨率为 $256\times256$ 的 FFHQ 和 $512\times512$ 的 AFHQCat
- 对数据集进行了水平翻转增强，并对大小有限的 AFHQ-Cat 数据集使用了自适应数据增强
- 相机姿态标签来自 EG3D 官方资料库，由现成的姿态估计器预测
- 在每个数据集上从头开始训练模型

## 实验结果

### 定量结果

![Table1](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-GAN/%E6%88%AA%E5%9B%BE/Table1.png?raw=true)

### 定性结果

![Fig4](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-GAN/%E6%88%AA%E5%9B%BE/Fig4.png?raw=true)

### 3D 一致性比较

![Table2](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-GAN/%E6%88%AA%E5%9B%BE/Table2.png?raw=true)

### 训练早期阶段的训练稳定性

![Fig5](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-GAN/%E6%88%AA%E5%9B%BE/Fig5.png?raw=true)

### 消融实验

![Table3+Fig6](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-GAN/%E6%88%AA%E5%9B%BE/Table3+Fig6.png?raw=true)


# 局限

- 本文方法合成高斯的数量固定，在不同场景的表现上会有局限
- 分层高斯表示法的规模在一定程度上取决于超参数，e.g. $\Delta s$ 
- 这些因素都需要调整超参数，并会影响生成器的性能
