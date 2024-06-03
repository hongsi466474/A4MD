https://arxiv.org/abs/2405.17811

https://gaoxiangjun.github.io/mani_gs/


![Overview](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/Mani-GS/Fig2.jpg?raw=true)

# Mani-GS: Gaussian Splatting Manipulation with Triangle Mesh

[Xiangjun Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao,+X), [Xiaoyu Li](https://arxiv.org/search/cs?searchtype=author&query=Li,+X), [Yiyu Zhuang](https://arxiv.org/search/cs?searchtype=author&query=Zhuang,+Y), [Qi Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+Q), [Wenbo Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu,+W), [Chaopeng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+C), [Yao Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao,+Y), [Ying Shan](https://arxiv.org/search/cs?searchtype=author&query=Shan,+Y), [Long Quan](https://arxiv.org/search/cs?searchtype=author&query=Quan,+L)

The **Hong Kong** University of **Science and Technology** || **Tencent** AI Lab || **Nanjing** University

关键词： #高斯喷溅 #操纵 #渲染

自添关键词： #高斯编辑 #编辑 #三角网格 

# 动机

- 神经3D表示，如 NeRF ：能产生真实感渲染结果但操纵与编辑缺乏灵活性
	- 前人工作：在典型空间中变形 NeRF；基于显式网格操纵辐射场
	- ×：操纵 NeRF 可控性弱且训练和推理都很耗时
- 3DGS：用基于点的3D表示实现极为高保真的新视图合成，训练和渲染速度快
	- ×：缺乏在保持渲染质量的同时自由操纵 3DGS 的有效手段

## 目标：实现可操纵的逼真渲染

- 用三角网格通过自适应直接操纵 3DGS：减少为不同类型的高斯操纵设计各种算法的需要
- 三角形状感知高斯绑定和适配方法：达到 3DGS 操纵后也能高保真渲染

# 贡献

- 3DGS 操纵方法：有效传输三角网格操纵到 3DGS ，并高保真渲染
- 自适应三角形状感知高斯绑定策略：对网格准确度有高包容性支持多种类 3DGS 操纵
- 评估本方法并取得最先进的结果，展示各种 3DGS 操纵：大尺度变形、局部操纵、软体模拟，如图1：

![Fig1](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/Mani-GS/Fig1.jpg?raw=true)

# 方法

![Overview](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/Mani-GS/Fig2.jpg?raw=true)

## 前置知识

- 一个3D高斯（3DG）点的定义：
  - $$G(\boldsymbol{x}) = \exp\left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^\top\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right) \tag{1}$$
	- 每个 3DG 点用一个三维均值位置坐标 $\boldsymbol{\mu}$ 和一个协方差矩阵 $\Sigma$ 表示
		- 确保 $\Sigma$ 有意义：用四元数 $\boldsymbol{q}$ 和三维缩放向量 $\boldsymbol{s}$ 参数化，定义为 $\Sigma=\boldsymbol{RSS^\top R^\top}$
	- 每个高斯有属性：
		- 不透明度 $\boldsymbol{o}$ 
		- 依赖视角的颜色 $\boldsymbol{c}$ （由一组球谐（SH）表示）
- 从一个视点渲染图像，3DGs 被投影到图像平面，得到 2DGs 。二维协方差矩阵被近似为
  - $$\Sigma'=\boldsymbol{JW}\Sigma\boldsymbol{W^\top J^\top} \tag{2}$$
	  - $\boldsymbol{W}$ ：视角转换
	  - $\boldsymbol{J}$ ：透视投影变换的仿射近似值的 Jacobian 矩阵
	  - 二维均值由投影矩阵计算得到
- 像素颜色通过 $N$ 个有序 2DGs 的 $\alpha$ 混合合成：
  - $$\mathcal{C}=\sum_ {i\in N}T_ {i}\alpha\boldsymbol{c}_ {i}，其中 T_ {i}=\prod_ {j=1}^{i-1}(1-\alpha_ {i}) \tag{3}$$
	  - $\alpha$ ：不透明度 $\boldsymbol{o}$ 与根据图像空间上的 

## 网格提取

### 筛选泊松重建（Screenedo Poisson Reconstruction）

- 将 3DGS 看作点云
- ！：3DGs 没有用于重建的法向
  - √：给 3DGs 额外分配一个高斯属性，即法向 $\boldsymbol{n}$ ：
    - $$\\{\mathcal{D},\mathcal{N}\\}=\sum_ {i\in N}T_ {i}\alpha_ {i}\\{d_ {i},\boldsymbol{n}_ {i}\\} \tag{4}$$
      - $d_ {i}$ ：深度； $\boldsymbol{n}_ {i}$ ：每个高斯点的法向
	    - 优化法向：让渲染后的法向 $\mathcal{N}$ 与伪法向贴图 $\tilde{\mathcal{N}}$ 保持对齐
		  - $\tilde{\mathcal{N}}$ ：根据局部平面性假设从渲染深度 $\mathcal{D}$ 计算得出
- 训练完带法向的 3DGS 后，用筛选泊松表面重建（Screenedo Poisson Surface Reconstruction）算法提取网格

- 训练细节：
  - 第一阶段的损失：
    - $$\mathcal{L}_ {stage1}=\lambda_ {1}\mathcal{L}_ {1}+\lambda_ {2}\mathcal{L}_ {SSIM}+\lambda_ {3}\mathcal{L}_ {n}+\lambda_ {4}\mathcal{L}_ {mask} \tag{11}$$
      - $\lambda_ 1=1,\lambda_ 2=0.2,\lambda_ 3=0.01,\lambda_ 4=0.1$
      - 法向一致性：
        - $$\mathcal{L}_ {n}=\Vert\mathcal{N-\tilde{N}\Vert}_ {2} \tag{9}$$
          - 掩码交叉熵损失：解决背景区域中不必要的高斯问题
            - $$\mathcal{L}_ {mask}=-B^m\log B-(1-B^m)\log(1-B) \tag{10}$$
            - $B^m$ ：物体mask
            - 累计透射率 $B=\sum_ {i\in N}T_ {i}\alpha_ {i}$
	- 用自适应密度控制对这一阶段进行 3 万次训练
		- 在迭代 500 到 1 万的范围内，每 500 次迭代执行一次自适应密度控制
		- 训练阶段完成后，使用高斯位置与法向作为输入，继续进行筛选泊松表面重构
	- 网格提取过程不到 1 分钟

### Marching Cube for GS

- DreamGaussian：试图将相邻高斯点的 $\alpha$ 值汇总为 marching cube 采样点的综合密度值
	- ×：总会忽略薄而小的结构
- √：用 ball query Nearest Neighbor searching 方法，只搜索网格采样点的最近高斯点
	- 在预定距离内有相邻点的，密度值为 1 ，否则为 0 
	- 再用 Marching Cube 提取几何
	- 此方法提取的网格可能不准确，但本文的网格-高斯混合策略能在略不准确的网格下达到高保真渲染，且支持高斯操纵

- 训练细节
	- 采样分辨率为 $256\times256\times256$ 的网格
	- 对每个采样点，找到最近的高斯点
		- 采样点的最近高斯点在预设距离阈值 $\tau$ 内的，密度值为 1，否则为 0
		- 实操中 $\tau=0.01$ 
		- 行进立方的密度阈值设置为 1e-4

### 神经隐式表面（Neural Implicit Surface）

- NeuS：提取符号距离函数的零值面作作为表面
	- ×：可能包含过多三角形表面，可能约 150 万
- √：用网格清理程序消除 noisy floaters，利用网格删减技术将三角形数量减至约 30 万

### 网格质量

- NeuS > Poisson Reconstruction > Marching-Cube

## 对网格绑定 GS

- 目标：关联 3DGS 与网格三角形，编辑网格以操纵 3DGS及其渲染结果
	- 对于三角网格 $\boldsymbol{T}$ （有 $K$ 个顶点 $\\{\boldsymbol{v}_ {i}\\}^K_ {i=1}$ 和 $M$ 个三角形 $\\{\boldsymbol{f}_ {i}\\}^M_ {i=1}$），想构建一个与网格三角形绑定的 3DGS 模型并优化每个高斯属性 $\\{\boldsymbol{\mu}_ {i},\boldsymbol{q}_ {i},\boldsymbol{s}_ {i},o_ {i},\boldsymbol{c}_ {i}\\}$ 
- 对于网格 $\boldsymbol{T}$ 的每个三角形 $\boldsymbol{f}$ ，有三个顶点 $(\boldsymbol{v_ {1},v_ {2},v_ {3}})$ ，在此三角形上初始化 $N$ 个高斯
	- 初始化的高斯位置均值 $\boldsymbol{\mu}=(w_ {1}\boldsymbol{v_ {1}}+w_ {2}\boldsymbol{v_ {2}+}+w_ {3}\boldsymbol{v_ {3}})$ 
		- $\boldsymbol{w}=(w_ {1},w_ {2},w_ {3})$ 为三角形上每个高斯的预设重心坐标， $w_ {1}+w_ {2}+w_ {3}=1$

### 几种绑定策略的对比

-  **网格上的高斯**
	- 通过网格实现可控 3DGS 操纵：
		- 直观想法：SuGaR，将 GS 完美连接到三角形
			- 强依赖网格准确性
			- 在复杂对象渲染建模方面，缺乏 3DGS 的灵活性
			- 获得的网格质量远低于 GT 和最近的隐式表面重建的结果，增加编辑困难

- **网格上带偏移的高斯**
	- 弥补提取的网格误差，对高斯 3D 均值 $\boldsymbol{\mu}$ 引入偏差 $\Delta\boldsymbol{\mu}$ ，使高斯能移出所附三角形 $\boldsymbol{f}$ 
		- √：提高重建静态物体的渲染质量
		- ×：由于高斯之间的局部相对位置不匹配，会导致被操作对象出现噪声和一味的渲染失真

- **三角形状感知高斯绑定和自适应**
	- 在操纵后保持高保真渲染结果的关键点：保持局部刚性以及高斯之间的相对位置（均值&旋转）
	- 本文关键想法：在每个三角形空间中定义一个局部坐标系
		- 第一个轴的方向：第一条边的方向
		- 第二个轴的方向：法向
		- 第三个轴的方向：前俩轴方向的叉积
		- 三角坐标系的旋转：
      - $$\boldsymbol{R^t}=[\boldsymbol{r^t_ {1},r^t_ {2},r^t_ {3}}]=\boldsymbol{[\frac{v_ {2}-v_ {1}}{\Vert v_ {2}-v_ {1}\Vert},n^t,\frac{v_ {2}-v_ {1}}{\Vert v_ {2}-v_ {1}\Vert}\times n^t]} \tag{5}$$
        - $\boldsymbol{v_ {1},v_ {2}}$ 为第一第二顶点的位置
        - 法向：
          - $$\boldsymbol{n^t}=\frac{\boldsymbol{(v_ {2}-v_ {1})\times(v_ {3}-v_ {1})}}{\Vert\boldsymbol{(v_ {2}-v_ {1})\times(v_ {3}-v_ {1})\Vert}} \tag{6}$$
	- 优化三角空间中的高斯局部位置 $\boldsymbol{\mu^l}$ 和局部旋转 $\boldsymbol{R^l}$ 
	- 全局旋转、放缩和位置：
    - $$\boldsymbol{R=R^t R^l,s=s^l,\mu=R^t\mu^l+\mu^t} \tag{7}$$
      - $\boldsymbol{\mu^t}$ ：每个三角中心的全局坐标
	- **实践**：初始化 $N$ 个高斯点，绑定每个高斯点，其初始化位置均匀分布在三角形上
		- 初始化位置用重心坐标计算，预定义重心坐标集为 $\left[ \frac{1}{2},\frac{1}{4},\frac{1}{4}\right],\left[ \frac{1}{4}, \frac{1}{2}, \frac{1}{4}\right],\left[ \frac{1}{4}, \frac{1}{4}, \frac{1}{2} \right]$ 
		- 三角形扩大时，相应地局部缩放和位置也随之扩大：
      - $$\boldsymbol{R=R^t R^l,s=\beta e s^l,\mu=e R^t\mu^l+\mu^t} \tag{8}$$
        - $\beta$ ：超参数，本文大多数情况设为 10
        - 自适应向量 $\boldsymbol{e}=[e_ {1},e_ {2},e_ {3}]$ 确保全局缩放 $\boldsymbol{s}$ 与三角形状成正比
          - $e_ 1=l_ 1$ ：三角形第一边的长度
          - $e_ 2=\frac{e_ 1+e_ 3}{2}$ ： $e_ {1},e_ {3}$ 的均值
          - $e_ 3=\frac{l_ 2+l_ 3}{2}$ ：第二边和第三边的均值
		- 第二阶段不用自适应控制不影响最终结果的表现
      - 训练 3 万次迭代，损失函数为：
        - $$\mathcal{L}_ {stage2}=\lambda_ {1}\mathcal{L}_ {1}+\lambda_ {2}\mathcal{L}_ {SSIM}+\lambda_ {3}\mathcal{L}_ {mask} \tag{12}$$
          - $\lambda_ 1=1,\lambda_ 2=0.2,\lambda_ 3=0.1$ 


## 通过网格操纵 GS

- 本文策略能在模型训练和网格操纵一结束就立刻对 3DGS 操纵和调整
- 网格操纵期间，局部三角空间中的属性不变，三角形旋转、位置以及边长能立即计算
- 全局高斯位置、缩放以及旋转通过前述公式自适应调整
- 本文实验用 Blender 操纵网格

# 实验

## 训练细节

- 第一阶段：提取网格，其间本文仅从神经隐式表面场（NeuS）或 3DGS （筛选泊松或行进立方）中提取三角网格
	- 网格所含三角形数目过多，尝试减至 30 万
- 第二阶段：对网格执行三角形状感知高斯绑定和自适应策略
	- 对每个三角形，最开始绑定 $N=3$ 个高斯到表面
	- 高斯属性通过监督多视角渲染损失来优化
- 第一阶段训练迭代 3 万次，第二阶段迭代 2 万次
- 只用一张 NVIDIA A100 GPU

## 数据集、度量以及对比的方法

- 对比的方法：NeRF-Editing、SuGaR
- 度量：PSNR，SSIM，LPIPS
- 数据集：NeRF 合成数据集、DTU 数据集

## 评估

### 静态渲染

- 定量结果对比
![表1](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/Mani-GS/Table1.png?raw=true)

- 定性结果对比
![图3](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/Mani-GS/Fig3.png?raw=true)

### 操纵渲染

![图4](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/Mani-GS/Fig4.png?raw=true)

![图5](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/Mani-GS/Fig5.png?raw=true)

### 消融实验

![表2](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/Mani-GS/Table2.jpg?raw=true)

![图6](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/Mani-GS/Fig6.png?raw=true)

# 局限

- 还是有失真的结果：当操纵局部区域网格包含高度非钢变形时，结果会失真
- 在超过 3.5 万个三角形的网格上执行物理仿真耗时很长
- 提取的网格与 GT 有明显差异时，结果可能无法准确渲染边界，e.g. 出现不相连的区域
