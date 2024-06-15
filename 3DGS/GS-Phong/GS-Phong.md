https://arxiv.org/abs/2405.20791

![Fig3](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-Phong/%E6%88%AA%E5%9B%BE/Fig3.png?raw=true)

# GS-Phong: Meta-Learned 3D Gaussians for Relightable Novel View Synthesis

[Yumeng He](https://arxiv.org/search/cs?searchtype=author&query=He,+Y), [Yunbo Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+Y), [Xiaokang Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang,+X)

MoE Key Lab of Artificial Intelligence, AI Institute, **Shanghai Jiao Tong** University

自添关键词： #高斯喷溅 #重新光照 #Blinn-Phong光照模型 #新视角合成

# 动机

- 学习解耦的光照成分（环境光、漫反射、镜面等）以创建与三维物体交互的照明效果
- 体积渲染
	- 类 NeRF 隐式辐射表示：能合成逼真的阴影效果
		- 典型方法：分离式神经网络（NRHints），一个建模密度场，一个建模光线传播
			- ×：限制模型对更复杂、分布不均匀的光照条件进行推广
		- ×：此类方法费时，经常超过半天
	- 3DGS：训练和推理时间短但显式表示很难计算阴影
		- 从一致光照条件下的多视角图中学BRDF
			- √：可以处理给定新环境地图的重新照明任务
			- ×：需要大量不同光照条件下的图像，不适合“一次一光”（OLAT）设置，即场景由移动点光源照明，训练数据由单目相机采集，如图 1

 ![Fig1](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-Phong/%E6%88%AA%E5%9B%BE/Fig1.png?raw=true)

# 思想：根据对 $V,L,\hat{G}$ 不同的依赖程度解耦光照成分；将 OLAT 定义为多任务学习问题

- e.g. 漫反射颜色的强度随高斯点的法线方向和光线方向变化；镜面反射则取决于 $(V,L,\hat{G})$ 中所有分量
	-  $V$ ：视线方向；  $L$ ：光线位置； $\hat{G}$ ：优化的几何参数
- 将特定光照位置下的渲染任务视为一个单独的任务

# 目标

- 在 OLAT 设置下学习一个渲染函数 $C\sim F(V,L;\hat{G})$ ， $C$ 为输出颜色
	-  $V$ 和 $L$ 改变的同时，它增加 $\hat{G}$ 的模糊性，从而增加分辨真是光效的挑战

# 贡献

- GS-Phong
	- 结合 Phong 模型的物理先验对 3DGS 进行的修改，能在 3D 渲染中将不同的光照成分解耦
	- 能在未知光照条件下实现逼真的新视角合成
- 一种元学习程序
	- 主要思路：利用双层优化来获得统一的几何知识（不同视角和不同光照位置都通用）

# OLAT 的问题定义

- “一次一光”，One Light At A Time（OLAT），指三维重建领域的一种特殊设置
	- 包括单个点光源连续照射被摄体，在每个光源位置捕获一幅图像
- 本文的具体设置：
	- 在一组随机捕获的图像 $\{I\}_ N$ 上训练
		- 每张图包括相机参数和光照信息（光线位置和颜色，不含光照强度）
	- 只有一个单目相机
- OLAT
	- √：数据采集效率高
	- ×：三维重建不易
		- 训练集中的光照位置变化迅速， 3D 点的颜色会发生剧烈变化，增加模型估计真实颜色和几何形状的模糊性
		- 本文的俩贡献能解决这个挑战

# 前置知识

## 3DGS

- 一个3D高斯（3DG）点的定义：
  - $$G(\boldsymbol{x}) = \exp\left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^\top\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right) \tag{1}$$
	  - 每个 3DG 点用一个三维均值位置坐标 $\boldsymbol{\mu}$ 和一个协方差矩阵 $\Sigma$ 表示
		  - 确保 $\Sigma$ 有意义：用四元数 $\boldsymbol{q}$ 和三维缩放向量 $\boldsymbol{s}$ 参数化，定义为$\Sigma=\boldsymbol{RSS^\top R^\top}$
	- 每个高斯有属性：
		- 不透明度 $\boldsymbol{o}$ 
		- 依赖视角的颜色 $\boldsymbol{c}$ （由一组球谐（SH）表示）
- 从一个视点渲染图像，3DGs 被投影到图像平面，得到 2DGs 。二维协方差矩阵被近似为$$\Sigma'=\boldsymbol{JW}\Sigma\boldsymbol{W^\top J^\top} \tag{2}$$
	- $\boldsymbol{W}$ ：视角转换
	- $\boldsymbol{J}$ ：透视投影变换的仿射近似值的 Jacobian 矩阵
	- 二维均值由投影矩阵计算得到
- 像素颜色通过 $N$ 个有序 2DGs 的 $\alpha$ 混合合成：
  - $$\mathcal{C}=\sum_ {i\in N}T_ {i}\alpha\boldsymbol{c}_ {i}，其中 T_ {i}=\prod_ {j=1}^{i-1}(1-\alpha_ {i}) \tag{3}$$
	  - $\alpha$ ：不透明度 $\boldsymbol{o}$ 与根据图像空间上的 

## Blinn-Phong 光照模型

- 表面的反射光传播有三个组成部分：
	- 环境反射（ $L_ a$ ）：环境中存在的恒定照明，模拟光线如何从其他表面散射和反射，以确定亮度基线水平
	- 漫反射（ $L_ d$ ）：
		- 在 Lambertian 扩散定律作用下，光照射到粗糙表面时会向多个方向散射，从而产生哑光效果
		- 由光线方向（ $\boldsymbol{\mathrm{l}}$ ）和表面法向（ $\boldsymbol{\mathrm{n}}$ ）间的夹角计算得出，确保面向光源的表面看起来更亮
	- 镜面反射（ $L_ s$ ）：根据光线方向（ $\boldsymbol{\mathrm{l}}$ ）与角平分线（ $\boldsymbol{\mathrm{h}}$ ）的夹角计算得出
		-  $\boldsymbol{\mathrm{h}}$ ：视线方向（ $\boldsymbol{\mathrm{v}}$ ）光线方向（ $\boldsymbol{\mathrm{l}}$ ）间的角平分线
- 还包含一个光泽系数，用于控制镜面反射的扩散，从而该模型能代表表面的光泽度

## 经验性发现

- 在 OLAT 训练设置下，基于 3DGS 现有重照明技术的性能明显下降
- 基于 NeRF 模型的现有技术是 NRHints 在光照位置分布不均（OOD）的测试场景中表现不佳（图2）
	- 训练集的光照布置在半球的一侧，测试机的光照布置在另一侧
	- ×：基线模型无法长生合理的渲染结果；基于 NeRF 的模型对阴影和镜面反射隐式建模降低了性能
	- √：本文提出一种明确的光照信息建模方法，自然而然地实现在未知 OOD 场景下的重建和重新照明

![Fig2](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-Phong/%E6%88%AA%E5%9B%BE/Fig2.png?raw=true)

# 方法

![Fig3](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-Phong/%E6%88%AA%E5%9B%BE/Fig3.png?raw=true)
## 可学习的 Phong 模型

- 通过可学习的高斯点与来自观测者的光源的光线相互作用来分离光照成分
- Phong 模型：
	- $f:\mathbb{R}^3\to\mathbb{R}$ ，将 3D 点 $\boldsymbol{\mathrm{x}}:(x,y,z)$ 作为输入，返回相关的光照强度（一个实数）
- GS-Phong： 
	- 高斯点属性：
		- 基本属性：位置 $\boldsymbol{\mathrm{x}}$ ，旋转 $R$ ，放缩 $S$ ，不透明度 $\alpha$ ，球谐系数 $f$ 
		- 附加属性：法向 $\boldsymbol{\mathrm{n}}$ ，3 个通道的漫反射颜色 $k_ d$ ，1 个通道的镜面反射系数 $k_ s$ 
			- 更好理解光照效果，促进与光线无关的物体几何形状的学习过程
	- 高斯点的整体颜色： ^323fe4
		- $$L_ {p}=L_ {a}+L_ {d}+L_ {s}=k_ {a}I_ {a}+\sum_ {m\in lights}(k_ {d}I_ {d}+k_ {s}I_ {s} \tag{4})$$
			- $k_ {\{a,d,s\}}$ ：每个成分的颜色；
			- $I_ {\{a,d,s\}}$ ：每个成分的强度
	- 对每个训练视图，依次对每个高斯点的环境色、漫反射色和镜面反射色进行建模
		- 将高斯的 SH 系数 $f$ 限制为零阶，以反映环境色
			- 环境色增加了恒定的颜色，以考虑忽略的光照，并填充黑色阴影
			- 在图形和光照应用中，零阶球谐系数通常代表定义在球面上的函数的均匀分量，基本上是光照环境的平均或恒定部分
		- 漫反射色和镜面反射色：将各成分的颜色和强度相乘获得
			- 强度：
				- $$I_ {d}=\frac{I}{r^2}\max(0,\boldsymbol{\mathrm{n}\cdot\boldsymbol{\mathrm{l}}}), I_ {s}=\frac{I}{r^2}\max(0,\boldsymbol{\mathrm{n}\cdot\boldsymbol{\mathrm{h}}})^p \tag{5}$$
					- $I$ ：光线发射强度
					- $\boldsymbol{\mathrm{l}}$ ：点到光源的归一化向量
					- $\boldsymbol{\mathrm{h}}=\frac{\boldsymbol{\mathrm{v+l}}}{\Vert\boldsymbol{\mathrm{v+l}}\Vert}$ ：点到摄像机的归一化向量 $\boldsymbol{\mathrm{v}}$ 和 $\boldsymbol{\mathrm{l}}$ 的角平分线
			- 对系数/颜色 $k_ d$ 和 $k_ s$ 进行隐式建模，其中 $k_ s$ 乘以点光源的 RGB 颜色

## 与光无关的几何图形元学习框架

- OLAT 设置下最大的挑战：学习与光无关的几何图形
	- 不同帧中，相同点的外观波动大
	- GS-Phong 
		- √：能有效分离不同的光照系数
		- ×：大部份依赖光照成分（包括阴影、漫反射颜色和镜面反射颜色）都依赖高斯点的预测法向
			- ！：“蛋和鸡”问题
			- 光照系数和物体几何优化相互交织，阻碍模型正确学习场景属性，易出现局部最优
- √：双层优化的元学习来增强对光线无关的几何信息估计
	- 如图 1 ，本文在多任务学习框架中整合和来自多种光照条件的信息，并用二阶导数更新模型
	- 关键思路：鼓励学习到的几何在不同光照位置之间进行泛化

![Fig1](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-Phong/%E6%88%AA%E5%9B%BE/Fig1.png?raw=true)

- 阴影计算
	- 基于 BVH 的光线追踪方法
		- 对每个点，通过追踪从高斯平均位置到光源的射线确定光的可见度，然后计算射线的累计透射率，即 $T^{light}_ {i}=\prod^{i-1}_ {j=1}(1-\alpha_ {j})$ 
		- 只有漫反射和镜面反射颜色受入射光强度影响
		- 更新每个点的颜色： 
			- 在漫反射和镜面反射项中加入光能见度因子
			- $$L_ {p}=k_ {a}I_ {a}+T^{light}_ {i}\times\sum_ {m\in lights}(k_ {d}I_ {d}+k_ {s}I_ {s}) \tag{6}$$
	- 将这一模块集成到 GS-Phong 的管道中有挑战性
		- “蛋与鸡”问题使模型很难将阴影从场景分离，导致颜色预测不准
		- 要实现准确的场景属性，就必须掌握阴影计算

![Algm1](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-Phong/%E6%88%AA%E5%9B%BE/Algm1.png?raw=true)

- 内循环
	- 在训练管道中集成元学习方案，以解决 OLAT 中固有的阴影学习问题
	- 训练集根据不同的光照条件分为：支撑集、查询集
	- 过程：
		- 最初，所有任务都以相同的模型参数（记为 $\theta$ ）开始
		- 每 $n$ 次迭代中，算法从内部优化循环开始
			- 每个任务从支撑集中选择一个自己来更新模型，得到 $\theta^*$ 
				- 根据从无阴影的 Phong 模型（公式（4））中得出的梯度对 Phong 相关的科学系参数进行微调
				- 利用从有阴影的 Phong 模型（公式（6））中获得的损失更新阴影系数并利用更新后的高斯属性
			- 每个任务的参数都将用于外循环

- 外循环
	- 所有任务进行全局更新，模型参数（高斯属性和阴影参数）都要集体更新
	- 从查询集中选择另一个子集来评估更新后的模型 $\theta^\ast$ ，产生损失 $l_ {\theta^\ast}$ 
	- 将所有任务的损失汇总后，正式更新模型，通过 $\theta^*$ 向 $\theta$ 反向传播二阶梯度
	- 每个任务都有独特的损失函数
	- 训练中，模型会在这些任务之间不断转换
- 这种结构化的方法，通过特定任务的适应性迭代，进行全局更新，使模型有效学习不同光照条件下的通用特征

## 优化

### 三阶段训练方案
- 第一阶段：模型初始化阶段
	- 类似 3DGS ，专注于训练基本的高斯属性
- 第二阶段：引入法线调整
	- 精确的 Phong 学习在很大程度上取决于精确的法线向量
	- 用深度推断的伪法线 $\boldsymbol{\mathrm{n}}$ 监督预测的法线 $\boldsymbol{\mathrm{\hat{n}}}$ 
- 第三阶段：整合漫反射、镜面和阴影系数，通过双层元学习框架同时训练所有可学习参数
### 损失函数
$$\begin{aligned}
阶段1（高斯初始化）& ：\mathcal{L}_ {stage1}=\mathcal{L}_ {RGB}+\mathcal{L}_ {sparse}\\
阶段2（法向微调）& ：\mathcal{L}_ {stage1}=\mathcal{L}_ {RGB}+\mathcal{L}_ {sparse}+\mathcal{L}_ {normmal}+\mathcal{L}_ {smooth}\\
阶段3（元学习）& ：\mathcal{L}_ {stage1}=\mathcal{L}_ {RGB}+\mathcal{L}_ {sparse}+\mathcal{L}_ {normmal}+\mathcal{L}_ {smooth}+\mathcal{L}_ {diffuse}
\end{aligned} \tag{13}$$
- 重建损失 $\mathcal{L}_ {RGB}=(1-\lambda)\mathcal{L}_ {1}+\lambda\mathcal{L}_ {D-SSIM}$ 
- 法向：
- $$\boldsymbol{\mathrm{n}}=\begin{cases}
\boldsymbol{\mathrm{v}}+\Delta\boldsymbol{\mathrm{n}}_ {1} &\text{if}w_ {o}\cdot\boldsymbol{\mathrm{v}}>0\\\
-(\boldsymbol{\mathrm{v}}+\Delta\boldsymbol{\mathrm{n}}_ {2}) &\text{otherwise}
\end{cases} \tag{8}$$
- 法向相关损失： $\mathcal{L}_ {normmal}=\lambda_ {npred}\mathcal{L}_ {normmal\_ pred}+\lambda_ {nres}\mathcal{L}_ {normmal\_ res}+\lambda_ {s}\mathcal{L}_ {s}$
	- 根据 GaussianShader 用高斯最短轴方向 $\boldsymbol{\mathrm{v}}$ 和两个独立的法向残差（即向内或向外的法向）计算法向
	- $\mathcal{L}_ {normmal\_ pred}=\Vert\bar{\boldsymbol{\mathrm{n}}}-\hat{\boldsymbol{\mathrm{n}}}\Vert^2$ ：预测法向与深度推断法向的 L1 损失
	- $\mathcal{L}_ {normmal\_ res}=\Vert\Delta\boldsymbol{\mathrm{n}}\Vert^2$ ：残差正则化损失，防止正态残差偏离最短轴线过多
	- $\mathcal{L}_ {s}=\Vert\min(s_ {1},s_ {2},s_ {3})\Vert_ {1}$ ：限制高斯的最小尺度接近于零，有效地使高斯趋于平面形状，以确保最短轴线能代表法向
- 稀疏损失： $\mathcal{L}_ {sparse}=\lambda_ {opacity}\mathcal{L}_ {opacity}+\lambda_ {vis}\mathcal{L}_ {vis}$
  - $\mathcal{L}_ {opacity}$ ：让高斯球的不透明度 $\alpha$ 要么趋于 0 ，要么趋于 1：
    - $$\mathcal{L}_ {opacity}=\frac{1}{\vert\alpha\vert}\sum_ {i}[\log(\aleph_ {i})+\log(1-\alpha_ {i})] \tag{9}$$
	- $\mathcal{L}_ {vis}$ ：光可见度项 $T^{light}_ i$ 的损失
- 光滑损失：
  - $$\mathcal{L}_ {smooth}=\lambda_ {soomth}\Vert\boldsymbol{\mathrm{n}}-\text{sg}(g(\boldsymbol{\mathrm{n}},k))\Vert \tag{10}$$
	  - 利用渲染的法线贴图添加法线平滑损失，以防止物体表面出现高频伪影
	  - $\text{sg}(\cdot)$ ：停止梯度操作
	  - $g(\cdot)$ ：高斯模糊
	  - $k$ ：模糊核大小，设为 $9\times9$ 
- 漫反射先验损失：
  - $$\mathcal{L}_ {diffuse}=\lambda_ {diffuse} \frac{1}{N}\sum^N_ {i=1}\Vert\mathcal{I}^a_ {i}-\boldsymbol{s}\cdot\mathcal{I}^d_ {i}\Vert^2 \tag{11}$$
	  - 有效限制漫反射 RGB 变得过于混乱和陷入局部极小值的情况
	  - $N$ ：点的总数
	  - $\mathcal{I}^a_ {i},\mathcal{I}^d_ {i}$ ：表示第 $i$ 个点的 RGB 颜色的三通道向量
	  - 思想：限制漫反射 RGB 接近于环境色的 $\boldsymbol{s}$ 倍
		  - $\boldsymbol{s}$ 为三通道向量，通过最小化每个颜色通道的损失函数来分析获得
        - $$\boldsymbol{s}_ {k}=\frac{\sum^N_ {i=1}\text{sg}(\mathcal{I}^a_ {i,k})\cdot\mathcal{I}^d_ {i,k}}{\sum^{N}_ {i=1}(\mathcal{I}^d_ {i,k})^2},k\in\{0,1,2\} \tag{12}$$

## 执行细节

- 用 Adam 优化器
- 三个阶段的所有场景都采用同一套超参数
- 对漫反射颜色先验使用了标准指数衰减调度技术，促进训练的顺利进行
- 第三阶段（元学习阶段）的前 1 千次迭代中，将漫反射颜色损失从 0.02 指数衰减到 0.002，并在其余的训练中将其固定
- 第一阶段迭代 1 万次，第二阶段迭代 5 千次，第三阶段迭代次
- 单张 RTX3090 ，训练耗时约 40 分钟

# 实验

## 实验设置

- 在合成数据集中，选择了代表不同表面的层织球、塑料杯和橡胶杯三个合成场景
	- 层织球场景结合了复杂的可见度和强烈的相互反射
- 对于每个合成场景，用 500 张 OLAT 图进行训练，对于真实场景，用 600 张训练
- 对比方法：3DGS、Relightable 3DG、GaussianShader
	- 按原始设置分别训练 3 万 epoch，3 万 epoch，7 万 epoch
- 新视角渲染的评价指标：PSNR、SSIM、LPIPS

## 结果比较

![Fig4](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-Phong/%E6%88%AA%E5%9B%BE/Fig4.png?raw=true)

![Table1](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-Phong/%E6%88%AA%E5%9B%BE/Table1.png?raw=true)

![Fig5](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-Phong/%E6%88%AA%E5%9B%BE/fig5.png?raw=true)
## 消融实验

![Table2](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/3DGS/GS-Phong/%E6%88%AA%E5%9B%BE/Table2.png?raw=true)
# 局限

- 需要进一步研究以增强在处理极端光照条件或复杂场景几何形状时的稳健性
