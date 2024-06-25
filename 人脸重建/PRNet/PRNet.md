https://arxiv.org/abs/1803.07835

http://github.com/yfeng95.PRNet


![](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E4%BA%BA%E8%84%B8%E9%87%8D%E5%BB%BA/PRNet/%E6%88%AA%E5%9B%BE/Fig2.png?raw=true)

# Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network

[Yao Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng,+Y), [Fan Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu,+F), [Xiaohu Shao](https://arxiv.org/search/cs?searchtype=author&query=Shao,+X), [Yanfeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+Y), [Xi Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou,+X)

**Shanghai Jiao Tong** University || CloudWalk Technology （云从科技）|| Research Center for Intelligent Security Technology, CIGIT（**中科院**）

关键词： #3D重建/面部 #人脸对齐 #密集对应（Dense Correspondence）

自添关键词： 

# 动机

- 2D 对齐：在处理大型姿态或遮挡物问题存在困难
- 用 CNN 估计 3D 可变形模型（3DMM）系数或 3D 模型扭曲函数，从单一的 2D 面部图像还原相应三维信息
	- √：能提供密集人脸对齐和 3D 人脸重建结果
	- ×：性能受限于人脸模型基础或模板所定义的 3D 空间；包括透视投影或 3D 薄板样条（TPS）变换在内的必要操作也增加了整个流程的复杂性
		- √：训练一个复杂网络，根据单张图像的 2D 坐标对 68 个面部地标进行回归
			- ×：需要额外的网络估计深度值；不提供密集对齐
		- √：3D 人脸的体积表示法，利用网络从 2D 图像进行回归
			- ×：忽略点的语义，限制了恢复形状的分辨率，需要一个复杂网络来还原
- ⭐：这些限制在以前基于模型的方法中不存在
	- ➡：寻找方法，以无模型的方式同时获得 3D 重建和密集对齐

# 思想

- 用位置图表示人脸几何形状和对齐信息，然后通过编码器-解码器网络学习
- 特点
	- **端到端**：可绕过3DMM拟合，直接从单个图像回归3D面部结构和密集对齐
	- **多任务**：通过回归位置图，可以获得 3D 几何图形以及语义含义。从而轻易完成密集对齐、单目 3D 人脸重建、姿态估计等任务
	- **比实时更快**：该方法可以以超过 100fps（使用 GTX 1080）的速度运行以回归位置图
	- **鲁棒性**：在不受约束的条件下对面部图像进行测试。此方法对姿势、照明和遮挡是鲁棒的

# 贡献

![Fig1](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E4%BA%BA%E8%84%B8%E9%87%8D%E5%BB%BA/PRNet/%E6%88%AA%E5%9B%BE/Fig1.png?raw=true)

- 首次以端到端的方式同时解决人脸对齐和 3D 人脸重建问题，而不受低维求解空间的限制
- UV 位置图表示法：直接回归 3D 人脸结构和密集对齐；记录了 3D 人脸的位置信息，并与 UV 空间上每个点的语义进行密集对齐
- 权重掩码：为位置图上每个点分配不同权重，并计算加权损失；提高网络性能
- 轻量级框架：运行速度超过 100FPS，可直接从单张 2D 人脸图像中获得 3D 人脸重建和对齐结果
- 在 AFLW2000-3D 和 Florence 数据集上，三维人脸重建和墨迹人脸对齐任务都比其他最先进的方法相对提高了 25%

# 方法

## 3D 人脸重建

![Fig2](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E4%BA%BA%E8%84%B8%E9%87%8D%E5%BB%BA/PRNet/%E6%88%AA%E5%9B%BE/Fig2.png?raw=true)

- 用 UV 空间存储 3D 人脸模型中各点的 3D 坐标，如图2
	- 用左手系定义 3D 人脸点云，3D 空间的原点与输入图像的左上角重叠，正 x 轴指向图像的右侧
	- GT 3D 人脸点云投影到 x-y 平面时与 2D 图像中的人脸吻合
	- 位置图：将纹理图中的 rgb 值替换 为 xyz 坐标
- 根据 3DMM 创建 UV 坐标：保持位置图中点的语义
	- 为充分利用 300W-LP 数据集，让 UV 坐标与 BFM（Basel Face Model） 相对应
	- 用来自 [文献【35】](https://arxiv.org/abs/1708.07199) 的参数化 UV 坐标，该坐标通过保角 Laplace 权计算 Tutte 嵌入，然后将网格边界映射为正方形
	- BFM 中顶点数超过 5 万，选择位置图大小为 $256\times256$ ，可得到高精度点云的同时忽略重采样的误差
- 用 CNN 直接从无约束的 2D 图像中回归位置图
	- 同时获得 3D 人脸结构和密集对齐结果
	- 位置图记录了 3D 人脸的点集及其语义
	- 位置图也能推断面部不可兼得部分，从而此方法可预测完整的 3D 人脸

## 网络结构和损失函数

![Fig3](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E4%BA%BA%E8%84%B8%E9%87%8D%E5%BB%BA/PRNet/%E6%88%AA%E5%9B%BE/Fig3.png?raw=true)

### 网络结构

- 用编码器-解码器结构学习转换函数：将输入的 RGB 图像转换为位置图图像
	- 编码器：1 个卷积层+10 个残差块；将 $256\times256\times3$ 的输入图像缩小为 $8\times8\times512$ 的特征图
	- 解码器：17 个转置卷积层；生成预测的 $256\times256\times3$ 的位置图
	- 所有卷积层&转置卷积层的核大小为 4，用 RELU 层激活

### 损失函数

![Fig4](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E4%BA%BA%E8%84%B8%E9%87%8D%E5%BB%BA/PRNet/%E6%88%AA%E5%9B%BE/Fig4.png?raw=true)

- 用权重掩码：把辨别特征的重心放在人脸中心区域
	- 如图4，权重掩码是一个灰度图，记录了位置图上各点的权重
	- 大小和像素对应关系与位置图相同
	- 68 个面部关键点的位置权重最高
	- 颈部区域权重为 0 
- $$Loss=\sum\Vert P(x,y)-\tilde{P}(x,y)\Vert\cdot W(x,y) \tag{1}$$
	- $P(x,y)$ ：像素坐标 $x,y$ 对应的预测位置图值
	- $\tilde{P}(x,y)$ ：GT 位置图
	- $W(x,y)$ ：权重掩码
	- 实验中权重比例：子区域1(68个面部地标) : 子区域2(眼鼻嘴) : 子区域3(其他面部区域) : 子区域4(颈部) = 16 : 4 : 3 : 0

## 训练细节

- 训练集：300W-LP
	- 包含不同角度人脸图像，并标注了估计的 3DMM 系数
- 根据 GT 边界框裁剪图像，并调整为 $256\times256$ 
- 用标注的 3DMM 系数生成对应的 3D 位置，渲染到 UV 空间获得 GT 位置图
	- 训练中位置图的大小也为 $256\times256$ ，意味着要回归超过 4.5 万点云的精度

> 虽然训练数据是由 3DMM 生成，但网络输出（位置图）不局限于 3DMM 的任何人脸模板或线性空间

- 通过在 2D 图像平面上随机旋转和平移目标人脸来扰动训练集
	- 旋转角度：-45 度到 45 度不等
	- 平移变化：从输入大小的 10% 随机变化
	- 缩放范围：0.9 到 1.2 不等
- 通过缩放颜色通道来增加训练数据，缩放范围为 0.6 到 1.4 不等
- 在原始图像中添加噪点纹理来合成遮挡物
- 网络训练使用 Adam 优化器，学习率从 0.0001 开始，每 5 个 epoch 后衰减一半，批次大小为 16
- 用 TensorFlow 实现

# 实验

## 测试数据集

- AFLW2000-3D：评估本文方法在人脸重建和人脸对齐任务中的性能
- AFLW-LFPA：本文在 3D 人脸对齐任务中对该数据集进行评估，用 34 个可见地标作为 GT 来衡量结果的准确性
- Florence：对比本文方法和 VRN-Guided 与 3DDFA 在人脸重建任务中的性能

## 3D 人脸对齐

- 可见点和不可见点（包括 68 个关键点）的密集对齐
- 点的可见性（1 表示可见，0 表示不可见）

![](https://github.com/yfeng95/PRNet/raw/master/Docs/images/alignment.jpg)

![Fig5](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E4%BA%BA%E8%84%B8%E9%87%8D%E5%BB%BA/PRNet/%E6%88%AA%E5%9B%BE/Fig5.png?raw=true)

![Table1](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E4%BA%BA%E8%84%B8%E9%87%8D%E5%BB%BA/PRNet/%E6%88%AA%E5%9B%BE/Table1.png?raw=true)

![Fig6](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E4%BA%BA%E8%84%B8%E9%87%8D%E5%BB%BA/PRNet/%E6%88%AA%E5%9B%BE/Fig6.png?raw=true)

![Fig7](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E4%BA%BA%E8%84%B8%E9%87%8D%E5%BB%BA/PRNet/%E6%88%AA%E5%9B%BE/Fig7.png?raw=true)

## 3D 人脸重建

![Fig8](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E4%BA%BA%E8%84%B8%E9%87%8D%E5%BB%BA/PRNet/%E6%88%AA%E5%9B%BE/Fig8.png?raw=true)

![Fig9](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E4%BA%BA%E8%84%B8%E9%87%8D%E5%BB%BA/PRNet/%E6%88%AA%E5%9B%BE/Fig9.png?raw=true)

## 运行时间

![Table2](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E4%BA%BA%E8%84%B8%E9%87%8D%E5%BB%BA/PRNet/%E6%88%AA%E5%9B%BE/Table2.png?raw=true)

## 消融实验

![Table3](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E4%BA%BA%E8%84%B8%E9%87%8D%E5%BB%BA/PRNet/%E6%88%AA%E5%9B%BE/Table3.png?raw=true)

![Fig10](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E4%BA%BA%E8%84%B8%E9%87%8D%E5%BB%BA/PRNet/%E6%88%AA%E5%9B%BE/Fig10.png?raw=true)


