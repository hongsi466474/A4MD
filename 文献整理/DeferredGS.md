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

- 基于点的 $\alpha$ -混合渲染公式：$$C_{*}=\sum^N_{i=1}*_{i}\alpha_{i}T_{i} \tag{1}$$ ^ae1d4f
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

- 用公式 [[DeferredGS#^ae1d4f|(1)]] 对光线 $v$ 上的采样点的颜色作积分以渲染像素 $C'_c(v)$ 
- 点 $x_i$ 的不透明度 $\alpha_i=1-\exp\big(-\int^{t_{i+1}}_{t_{i}}\sigma(t)\mathrm{d}t\big)$ 

- 顺势函数：$$L_{nerf}=\sum_{c\in V}\big\Vert C'_{c}(v)-C^t(v)\big\Vert+\lambda\sum_{v\in V}\sum^M_{i=1}\big\Vert\Vert\nabla_{x_{v,i}}\Vert-1\big\Vert^2_{2} \tag{2}$$
	- $V$ 为一个batch的光线数，$M$ 为一条光线上的采样点数，$C^t(v)$ 为对应的GT颜色
	- 第二项为 Eikonal 损失，$\Vert\nabla_{x_{v,i}}\Vert$ 为光线 $v$ 在第 $i$ 个采样点 $x_{v,i}$ 处的SDF网络 $f_s(x)$ 梯度的 spatial norm

### 从 Instant-RefNeuS 蒸馏法向信息到GS

- 对每个高斯定义一个额外的法向 $n$ ，并用公式 [[DeferredGS#^ae1d4f|(1)]] 求像素 $C_n$ 的 GS 法向
- 通过最小化 $C_n$ 与 Instant-RefNeuS 中相应渲染的法线之间的差值来蒸馏 GS 的法向场：$$L_{nd}=\sum_{v\in V}\Vert 1-C_{n(v)}\cdot C'_{n}(v)\Vert=\sum_{v\in V}\big\Vert 1-C_{n}(v)\cdot C_{\nabla_{x_{v,i}}}\big\Vert \tag{3}$$

## GS 中的延迟着色



# 结果
