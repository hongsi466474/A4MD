https://arxiv.org/abs/2404.09412

![](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/DeferredGS/overview.jpeg?raw=true)

# DefferedGS: Decoupled and Editable Gaussian Splatting with Diferred Shading

[Tong Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu,+T), [Jia-Mu Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun,+J), [Yu-Kun Lai](https://arxiv.org/search/cs?searchtype=author&query=Lai,+Y), [Yuewen Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma,+Y), [Leif Kobbelt](https://arxiv.org/search/cs?searchtype=author&query=Kobbelt,+L), [Lin Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao,+L)

关键词： #高斯喷溅 #反向渲染 #编辑

# 动机

- NeRF 能做到逼真重建和编辑但渲染低效
- 高斯喷溅（GS）渲染快，但球谐函数同时建模了纹理和光照，与其离散的几何表示，都限制了纹理与光照的解耦和独立编辑；
- 近来工作对 GS 表示的解耦纹理和光照的尝试，在反射场景无法产生可信的几何与解耦结果
- 高斯几何属性在原始光照条件下优化，可能不适合新的光照条件，采用前向照明技术会在重新光照时产生混合伪影

## 目标：从一组图像提取几何、纹理与光照信息

- 解耦：用可学习的环境贴图建模光照，并在高斯上定义纹理参数和法向等附加属性
- 减轻伪影：延迟着色

# 贡献

- 提出 DeferredGS，一个可编辑几何图形、纹理和照明的解耦拼接表示，其几何形状通过一个法向蒸馏模块得到增强
- 第一个应用延迟着色技术到 GS 的方法，减轻了以往方法的混合伪影
- 产生的分解和编辑结果比以往方法更好

# 方法

![](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/DeferredGS/overview.jpeg?raw=true)

## GS 和纹理定义

- 基于点的 $\alpha$ -混合渲染公式： ^b45b75
	- $$C_{*}=\sum^N_{i=1}*_{i}\alpha_{i}T_{i} \tag{1}$$ 
	- $N$ 为一条光线上采样点的数量
	- $T_i=\prod^{i-1}_{j=1}(1-\alpha_{j})$ 为累计透射率
	- $*$ 可为要混合的采样点的任何属性（eg. 深度、颜色和法向）

- 将 $SH$ 替换为可优化的纹理参数（漫反射反照率 $k_d$ ，粗糙度 $r$ ，镜面反照率 $k_s$ ），从而解耦纹理和光照

## 法向场蒸馏模型

- 将当前的多视角几何重建技术与 GS 技术集合
	- 具体而言就是：联合训练一个类NeRF网络 Instant-RefNeuS 和一个 GS 表示。

### Instant-RefNeus

- 用 SDF 作几何表示，记为 $f_s(x)$ ，$x$ 为3D空间的采样点
- 将 SDF值转换为体密度：$\sigma(t)=\max\Big(-\frac{\frac{\mathrm{d}\Phi_{s}}{\mathrm{d}t}(f(x(t)))}{\Phi_{s}(f(x(t)))},0\Big)$
	- $\Phi_s(x)=(1+e^{-sx})^{-1}$ 
	- $s$ 为可训练的偏差参数

-  颜色：将外观分为两个分支以避免几何伪影（eg. 反光场景上的凹面）
	- 视角独立分支：预测采样点 $x$ 处的视角独立（漫反射）颜色 $c_d$ 及镜面色调 $p$ 
	- 视角相关分支：以视角方向为输入，输出镜面反射光照 $c_l$ 
	- 最终颜色为 $c'=c_d+pc_l$ 

> 带 $'$ 的项指NeRF网络，不带的指GS表示

- 用公式 [[DeferredGS#^b45b75|(1)]] 对光线 $v$ 上的采样点的颜色作积分以渲染像素 $C'_c(v)$ 
- 点 $x_i$ 的不透明度 $\alpha_i=1-\exp\big(-\int^{t_{i+1}}_{t_{i}}\sigma(t)\mathrm{d}t\big)$ 

- 损失函数：
	- $$L_{nerf}=\sum_{c\in V}\big\Vert C'_{c}(v)-C^t(v)\big\Vert+\lambda\sum_{v\in V}\sum^M_{i=1}\big\Vert\Vert\nabla_{x_{v,i}}\Vert-1\big\Vert^2_{2} \tag{2}$$
	- $V$ 为一个batch的光线数，$M$ 为一条光线上的采样点数，$C^t(v)$ 为对应的GT颜色
	- 第二项为 Eikonal 损失，$\Vert\nabla_{x_{v,i}}\Vert$ 为光线 $v$ 在第 $i$ 个采样点 $x_{v,i}$ 处的SDF网络 $f_s(x)$ 梯度的 spatial norm
	- 本文取 $\lambda=0.1$ 

### 从 Instant-RefNeuS 蒸馏法向信息到GS

- 对每个高斯定义一个额外的法向 $n$ ，并用公式 [[DeferredGS#^b45b75|(1)]] 求像素 $C_n$ 的 GS 法向
- 通过最小化 $C_n$ 与 Instant-RefNeuS 中相应渲染的法线之间的差值来蒸馏 GS 的法向场：$$L_{nd}=\sum_{v\in V}\Vert 1-C_{n(v)}\cdot C'_{n}(v)\Vert=\sum_{v\in V}\big\Vert 1-C_{n}(v)\cdot C_{\nabla_{x_{v,i}}}\big\Vert \tag{3}$$

## GS 中的延迟着色

### 着色计算

- 渲染公式：$$L(\omega_{o})=\int_{\Omega}L_{i}(\omega_{i})f(\omega_{i},\omega_{o})(\omega_{i}\cdot n)\mathrm{d}\omega_{i} \tag{4}$$ ^7beb55
	- $L(\omega_o)$ 与 $L(\omega_i)$ 分别表示 $\omega_o$ 方向的出射光线和 $\omega_i$ 方向的入射光线
	- $f(\omega_{i},\omega_{o})$ 是点的 BRDF 特性，可用迪斯尼着色模型确定：$$f(\omega_i),\omega_o)=\frac{k_{d}}{\pi}+\frac{DFG}{4(\omega_{i}\cdot n)(\omega_{o}\cdot n)} \tag{5}$$
		- $D$ ：正态分布函数
		- $F$ ：菲涅尔项
		- $G$ ：几何项
		- 这三项的计算详见 [5.1](https://zhuanlan.zhihu.com/p/443873239)
- 用 $6\times512\times512$ High Dynamic Range (HDR) cube map 模拟场景的环境光照
- 用 SplitSum ([2.2.4.1](https://zhuanlan.zhihu.com/p/121719442)) 近似法分离光照和 BRDF 积分，以实现高效阴影计算
- 最终着色：$c_g=c_{dif}+c_{spec}$ 
	- 漫反射颜色 ：$$c_{dif}=\frac{k_{d}}{\pi}\int_{\Omega}L_{i}(\omega_{i})(\omega_{i}\cdot n)\mathrm{d}\omega_{i} \tag{6}$$
		- $L_{i}(\omega_{i})(\omega_{i}\cdot n)$ 只依赖于法向 $n$ ，并且可以被预计算并存储在 2D 纹理
	- 镜面反射颜色：$$\begin{align}
c_{spec} & \approx Int_{light}\cdot Int_{BRDF} \\
& =\int_{\Omega}L_{i}(\omega_{i})D(\omega_{i},\omega_{o})(\omega_{i}\cdot n)\mathrm{d}\omega_{i}\cdot \int_{\Omega}f(\omega_{i},\omega_{o})(\omega_{i}\cdot n)\mathrm{d}\omega_{i} \tag{7}
\end{align}$$
		- $Int_{light}$ 表示入射光与正态分布函数 $D$ 的积分
			- 对于给定的环境贴图， $Int_{light}$ 只与粗糙值 $r$ 有关
			- 故可以预计算并存在 mipmap 中（每一个 mip level 对应一个固定的粗糙值）
		- $Int_{BRDF}$ 是均匀白色环境光照下镜面 BRDF 的积分
			- 由 BRDF 中的粗糙值和入射光方向与法向之间的点积 $\omega_i\cdot n$ 决定
			- 也可预计算并存在 2D 纹理中，在渲染时进行有效查询

### 获得解耦的 GS 表示

- × 直截了当的方法：前向着色
	- 在新光照条件下渲染会产生显著的伪影
- √ 延迟渲染
	- 首先用公式 [[DeferredGS#^b45b75|(1)]] 将位置 $P$ 、法向 $n$ 、漫反射反照率 $k_d$ 、粗糙度 $r$ 镜面反照率 $k_s$ 和不透明度 $\alpha$ 等要素光栅化为二维像素 $C_P,C_n,C_{k_d},C_r,C_{k_s},C_{\alpha}$ 
	- 汇总所有像素，得到相应的 2D 贴图 $I_P,I_n,I_{k_d},I_r,I_{k_s},I_{\alpha}$ 
	- 用公式 [[DeferredGS#^7beb55|(4)]] 计算像素的阴影颜色 $C$ 
		- 只对 $C_\alpha>0.5$ 的像素点计算着色
	- 损失函数：
		- $$L=L_{nerf}+\lambda_{nd}L_{nd}+\lambda_{L1}L_{L1}+\lambda_{ssim}L_{ssim}+\lambda_{mask}L_{mask}+\lambda_{TV}L_{TV} \tag{8}$$
			- $L_{L1}$ 为 $L1$ 损失，$L_{ssim}$ 为 D-SSIM 损失；（光栅化的图与 GT 图之间的）
			- $L_{mask}=\Vert I_\alpha-I_m\Vert_1$ 约束高斯位于前景，$I_m$ 为 GT mask 图
			- $L_{TV}$ ：应用于图像 $I_{k_{d}},I_{k_{s}}$ 的总变化损失，促进光滑的纹理估计
			- 本文取 $\lambda_{nd}=0.5,\lambda_{L1}=0.8,\lambda_{ssim}=0.2,\lambda_{mask}=0.5,\lambda_{TV}=0.001$ 

## 编辑解耦的高斯

- 对高斯属性优化后，可以获得解耦的 GS 表示，并能在新的光照下渲染经编辑的纹理或几何图形的原始场景

## 重新光照

- 提取出原始场景的环境贴图，用新的替换
- 使用延迟着色以新的光照渲染场景

## 几何编辑

- 用 Instant-ReNeuS 提取网格作为几何代理引导高斯的变形
	- 用 As-Rigid-As-Possible (ARAP) 变形算法变形网格【顶点旋转+平移】
	- 在原始网格平面找到每个高斯最近的邻近点 ，通过重心插值得到该点的旋转和平移
	- 对高斯的位置、协方差矩阵、法向应用相应的邻近点，以变形高斯
	- 光栅化变形后的高斯，可渲染变形后的场景

## 纹理编辑

- 支持对所有纹理组件进行更改
- 步骤：
	- 确定学要调整的高斯
		- 是否位于编辑 mask 中
		- 高斯深度与用公式 [[DeferredGS#^b45b75|(1)]] 渲染的深度图中相应深度之间的差异
	- 优化高斯的纹理属性 $*$ ：
		- ×：只考虑输入编辑视点，从其他视角查看优化后的高斯时，可能产生混合伪影
			- $$\arg\min_{*}\Vert I^i_{*}-I^i_{e}\Vert,*\in\{k_{d},r,k_{s}\} \tag{9}$$
				- $I^i_*$ ：从输入编辑视点 $i$ 渲染的纹理贴图，e.g. 漫反射反照率贴图
				- $I^i_e$ ：通过将编辑的内容放到 $I^i_*$ 上，从视点 $i$ 获取编辑后的纹理贴图
		- √：用随机视角输入来增强优化效果，即从多个随机视点生成编辑后的渲染结果
			- 将编辑后的内容 (unproject) 到 Instant-RefNeuS 提取的网格上，形成纹理网格
			- 从随机视角渲染纹理网格
			- 编辑损失：$$\arg\min_{*}\sum_{j\in L}\Vert I^j_{*}-I^j_{e}\Vert,*\in\{k_{d},r,k_{s}\} \tag{10}$$
				- 集合 $L$ 包含输入视点和随机采样的视点

# 结果

## 数据集和指标

- 两个合成数据集：NeRF Synthetic 和 Shiny Blender；一个真实数据集： Stanford ORB
- 定量：
	- 图像之间的相似性：PSNR、SSIM、LPIPS
	- 法线质量：MSE

## 执行细节

- 3090 GPU with 24GB memory
- 3-4 小时
- 单个3090，$800\times800$ 分辨率，以~30FPS的速度合成新视图

## 新视角合成

![](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/DeferredGS/Figure3.jpeg?raw=true)

![](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/DeferredGS/Table1.jpeg?raw=true)

## 解耦

![](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/DeferredGS/Figure4.jpeg?raw=true)

![](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/DeferredGS/Table2.jpeg?raw=true)

##  编辑

![](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/DeferredGS/Figure5.jpeg?raw=true)

![](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/DeferredGS/Table3.jpeg?raw=true)

![](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/DeferredGS/Figure6.jpeg?raw=true)

## 消融

![](https://github.com/hongsi466474/A4MD/blob/%E6%96%87%E7%8C%AE%E7%9B%B8%E5%85%B3/%E5%9B%BE%E7%89%87/DeferredGS/Figure7-10.jpeg?raw=true)

# 局限

- 如图10，本方法可能在有阴影的场景中产生错误的解耦结果
- 纹理编辑后的优化高斯可能包含噪声，因为高斯是全局表示，一个像素会受到多个高斯影响
