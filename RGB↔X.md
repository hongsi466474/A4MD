https://arxiv.org/abs/2405.00666

https://doi.org/10.1145/3641519.3657445

# RGB↔X: Image decomposition and synthesis using material- and lighting-aware diffusion models

Zheng Zeng, Valentin Deschaintre, Iliyan Georgiev, Yannick Hold-Geoffroy, Yiwei Hu, Fujun Luan, Ling-Qi Yan, and Miloš Hašan.

**Adobe** Research || University of California, **Santa Barbara** USA

**SIGGRAPH** Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA

关键词：扩散模型 本征分解 真实渲染

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
