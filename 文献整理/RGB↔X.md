https://arxiv.org/abs/2405.00666

https://doi.org/10.1145/3641519.3657445

# RGB↔X: Image decomposition and synthesis using material- and lighting-aware diffusion models

Zheng Zeng, Valentin Deschaintre, Iliyan Georgiev, Yannick Hold-Geoffroy, Yiwei Hu, Fujun Luan, Ling-Qi Yan, and Miloš Hašan.

**Adobe** Research || University of California, **Santa Barbara** USA

**SIGGRAPH** Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA

关键词：扩散模型 本征分解 逼真渲染

# 动机

- 传统渲染：精准但需要完成的场景描述
- 扩散模型：易用、真实但难精确控制

<!--- 探索 *扩散模型* 、 *渲染* 和 *固有通道估计* 之间的联系，重点关注材质/光线估计和以材质/光线为条件的图像合成，所有这些都在同一个扩散框架中进行。 --->

## 思想：用简单的东西来控制

- **X**：诸如漫反射反照率、镜面粗糙度和金属度，以及各种空间变化的照明表示之类的 **信息缓冲区（information buffers）** 称为 **固有通道（intrinsic channels）**

- **RGB→X**问题：从RGB图片估计以上那些通道的问题

- **X→RGB**问题：根据给定描述合成图像的问题

## 目标

- 探究扩散模型、渲染和固有通道估计之间的关系
- 关注两个问题：
  - RGB→X：固有通道估计
  - X→RGB：用固有通道控制图生成

# 贡献

提出一个基于扩散的统一架构，它可以实现逼真的图像分析（描述几何、材料和照明信息的内在通道估计）和合成（给定内在通道的逼真渲染），并在逼真的室内场景图像领域进行了演示，如图。



我们的工作是实现图像分解和合成统一框架的第一步。我们相信，它能为各种下游编辑任务带来益处，包括材质编辑、重新照明以及根据简单/未充分指定的场景定义进行逼真渲染。

- 一个 RGB→X 模型，通过使用来自多个异构数据集的更多训练数据并增加对照明估计的支持，改进了之前的工作 [Kocsis等人，2023]；
- 一个 X→RGB 模型，能够从给定的固有通道 X 合成逼真的图像，支持部分信息和可选的文本提示。文章结合了现有的数据集，并添加了一个新的、高质量的室内场景数据集，以达到高度逼真的效果。
