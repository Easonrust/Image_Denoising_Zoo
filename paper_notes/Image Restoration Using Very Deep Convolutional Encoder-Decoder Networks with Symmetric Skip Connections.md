## Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections

16年的论文

### 思路

- 提出了一种非常深的网络体系结构，它由一系列对称卷积和反卷积层组成，用于图像恢复。

  - 卷积层充当特征提取器，其对图像内容的主要组件进行编码，同时消除损坏。
  - 解卷积层解码图像抽象以恢复图像内容细节。

- 在相应的卷积层和反卷积层之间添加跳过连接。这些跳过连接有助于将渐变反向传播到底层并将图像细节传递到顶层，从而使端到端映射的训练更容易，更有效，从而在网络更深入时实现性能提升。

  > 类似U-Net的方法

由于网络具有大容量且较深，所以可以使用单个模型处理不同级别的噪声损坏

![image-20200302201900454](/Users/leyang/Library/Application Support/typora-user-images/image-20200302201900454.png)

网络的整体结构如上

代码：https://github.com/yjn870/REDNet-pytorch 

pytorch版本



## MemNet

17年的论文

传统CNN基本都是单向传播, 在靠后的层, 接收到的信号十分微弱, 这种单向传播的网络, 比如VDSR/DRCN等, 称为短期记忆网络

而有些网络结构中, 网络中的神经元不仅受到直接前驱的影响, 另外还受到额外指定的前驱神经元的影响, 这种被称为限制的长期记忆网络,如RedNet

> MemNet引入了一个包含递归单元(recursive unit)和门控单元(gate unit)的内存块(Memory block), 以期通过自适应学习过程明确地挖掘持续记忆, 递归单元学习当前状态在不同接收域(receptive field)下的多层表征(multi-level representation, Fig 1(c)中哪些蓝色的圈), 这些表征就可以视为由当前Memory block产生的short-term memory, 而long-term memory是由之前的Memory block产生的表征(就是Fig 1(c)的绿色箭头, 它表示long-term memory从之前的Memory block来), 这些short-term memory和long-term memory被合并输送到gate unit, gate unit自适应控制应保留多少先前的状态, 并决定应存储多少当前状态, 如文中所说的,  is a non-linear function to maintain persistent memory

memory block的结构：

![image-20200302203250734](/Users/leyang/Library/Application Support/typora-user-images/image-20200302203250734.png)

MemNet 的结构

![image-20200302203643572](/Users/leyang/Library/Application Support/typora-user-images/image-20200302203643572.png)感觉也是使用了skip connections，跳连接使底层和浅层信息都起作用

Pytorch版本：https://github.com/Vandermode/pytorch-MemNet/blob/master/memnet.py

-----------------------------------------------------------------------------------------------------------------------------------------------------------

找到了一片深度学习在图像去噪中的综述

## Deep Learning onImage Denoising: An Overview

深度学习技术在图像去噪上应用包括

- **外加的白噪声图像去噪的深度学习技术**

- 真实噪声图像去噪的深度学习技术

- 盲去噪的深度学习技术

- 混合噪声图像去噪的深度学习技术。

### 外加的白噪声图像去噪的深度学习技术（感觉我们主要是这个）

1. CNN/NN for AWNI denoising

   这里提了设计网络结构的方式

   - 利用多视角来设计网络；
   - 改变Loss函数；
   - 增加CNN的宽度或者深度；
   - 在CNN中增加任意的插件；
   - 在CNN中使用跳跃连接 (Skip connection)或者级联操作（Cascaded operations）。
     - RedNet为此思路

   > 补充说明：第（1）种方式：包括三种类型：一幅噪声图像作为多个子网络的输入；一个样本的不同角度作为网络的输入；一个网络的不同通道作为输入。第（4）种方式：任意插件包括激活函数、空洞卷积、全连接层和池化层等。第（5）种方式：包括skip connection和cascaded operation。表 1 提供CNNs/NNs for AWNI denoising的总结。

2. CNN/NN再结合common feature extraction

3. CNN/NN再结合opitimization

   1）提高去噪速度，（2） 提高去噪的性能。对于提高去噪效率，把优化方法嵌入到CNN来寻找最优解决是不错工具。此外，把噪声映射和噪声图像块作为CNN的输入也能提高预测噪声的速度







**深度学习一般在图像去噪上的主要应用是提高去噪性能、去噪效率和复杂的噪声图像。**

- 提高去噪性能：
  - 增大网络的感受野（增加**网络宽度和深度**是增加感受野最常见的方式， *同时可能会使网络过于复杂*）能捕获更多上下文信息来提高去噪性能。
  - CNN和先验结合能提取出更鲁棒的特征image prior
  - 组合局部和全局的信息能提高网络的记忆能力
  - 把信号处理机制融合到CNN能更好遏制噪声
  - 数据增加能提高图像去噪性能
  - 迁移学习、图学习和网络搜索能很好处理噪声图像。
- 提高去噪效率：压缩网络能有效地提高去噪的速度。减少网络宽度和深度、利用小的卷积核、组卷积都能有效地提高去噪速度。
- 解决复杂噪声：利用分布机制是非常流行的。第一步利用CNN来估计噪声级别作为ground truth或者恢复高分辨率图像。第二步用来恢复潜在干净图像。

***

**如今面临的一些挑战**

（1）更深的网络需要占用更多内存。（2）更深的去噪网络不能稳定地训练真实噪声图像、没有类标的噪声图像的模型。（3）真实噪声图像不是容易获得的。（4）更深的网络是困难来解决无监督去噪任务。（5）寻找更精确的去噪衡量指标。

https://zhuanlan.zhihu.com/p/106518406

github上做降噪的一个人https://zhuanlan.zhihu.com/p/106518406

