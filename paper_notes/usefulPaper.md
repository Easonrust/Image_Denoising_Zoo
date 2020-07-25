## dilated

190 CNN with dilated convolutions, and BN for image denoising*

90 Improved Unet from iterative shrinkage idea for medical image restoration

11 CNN with fully connected layer, RL and dilated convolutions for image denoising *

218 Deep CNN with internal and external residual learning for image denoising、

151 CNN with dilated convolutions for image denoising *

## attention

181 Gaussian image denoising, blind denoising and real noisy image denoising

182 Gaussian image denoising and real noisy image denoising

## Multiscale

187 U-net with multi scales technique for image denoising



# 增大感受野的方法

### dilated convolution（空洞卷积）

是在标准的 convolution map 里注入空洞，以此来增加 reception field。相比原来的正常convolution，dilated convolution 多了一个 hyper-parameter 称之为 dilation rate 指的是kernel的间隔数量

![img](https://pic3.zhimg.com/50/v2-d552433faa8363df84c53b905443a556_hd.jpg)

![img](https://pic2.zhimg.com/50/v2-4959201e816888c6648f2e78cccfd253_hd.jpg)

空洞卷积起源于予以分割，不通过pooling也能有较大的感受野看到更多的信息

# 注意力集中机制

https://github.com/Jongchan/attention-module

Official PyTorch code for "BAM: Bottleneck Attention Module (BMVC2018)" and "CBAM: Convolutional Block Attention Module (ECCV2018)"

*GRDN:Grouped Residual Dense Network for Real Image Denoising and GAN-based Real-world Noise Modeling*

![image-20200307210116908](/Users/leyang/Library/Application Support/typora-user-images/image-20200307210116908.png)

RDN上加深了，然后加上了**CBAM**的attention module，然后同样地使用残差学习来进行图像降噪。

主要用于真实噪声降噪

## Attention-guided CNN for image denoising

利用**稀疏机制、特征增强机制和Attention机制**在小网络复杂度的情况下提取显著性特征进而移除复杂图像背景中噪声。

ADNet主要利用四个模块：一个稀疏块（SB），一个特征增强块（FEB）, 一个注意力机制（AB）和一个重构块(RB)来进行图像去噪。

![image-20200307210905934](/Users/leyang/Library/Application Support/typora-user-images/image-20200307210905934.png)

主要用于Gaussian image denoising and real noisy image denoising





## A Multiscale Image Denoising Algorithm Based On Dilated Residual Convolution Network

we propose a novel deep residual learning model that combines the dilated residual convolution and multi-scale convolution groups

> the multiscale convolution group is utilized to learn those patterns and enlarge the receptive field
>
> In order to decrease the gridding artifacts, we integrate the hybrid dilated convolution design into our model

### Dilated filter

直接使用dilated convolution会导致gridding artifacts（网络伪影）

> It occurs when a feature map has higher-frequency content than the sampling rate of the dilated convolution
>
> ![](https://tva1.sinaimg.cn/large/00831rSTly1gcrgcmpdehj311a0jcgo5.jpg)

为了解决这个问题可以采用hybrid dilated convolution(HDC)，

![image-20200312211517396](https://tva1.sinaimg.cn/large/00831rSTly1gcrgg2jdd6j31go0rktdz.jpg)

### Multiscale convolution group

使用不同大小的卷积核获得不同尺度的特征

> The addition of the multiscale structure increases the width of network , on the other hand, improves the generalization of network

![image-20200312211921054](https://tva1.sinaimg.cn/large/00831rSTly1gcrgjsdn9sj31p30u07be.jpg)

inspired by Inception Module

## Inception Module

inception结构的主要贡献有两个：一是使用1x1的卷积来进行升降维；二是在多个尺寸上同时进行卷积再聚合。

### Inception v1

将1x1，3x3，5x5的conv和3x3的pooling，堆叠在一起，一方面增加了网络的width，另一方面增加了网络对尺度的适应性；增加了网络的宽度，增加了网络对尺度的适应性，不同的支路的感受野是不同的，所以有多尺度的信息在里面。

![image-20200312214519683](https://tva1.sinaimg.cn/large/00831rSTly1gcrhazv5kdj310k0j4dnv.jpg)



### v2

一方面了加入了BN层，减少了Internal Covariate Shift（内部neuron的数据分布发生变化），使每一层的输出都规范化到一个N(0, 1)的高斯；



## Attention in CNN

深度学习与视觉注意力机制结合的研究工作，大多数是集中于使用掩码( mask )来形成注意力机制。掩码的原理在于通过另一层新的权重，将图片数据中关键的特征标识出来，通过学习训练，让深度神经网络学到每一张新图片中需要关注的区域，也就形成了注意力

### 几种常见注意力的模块

1. 空间域

   CBAM（ Convolution Block Attention Module ）

   空间注意力，表现在图像上就是对 feature map 上不同位置的关注程度不同。反映在数学上就是指：针对某个大小为 ![[公式]](https://www.zhihu.com/equation?tex=H%C3%97W%C3%97C) 的特征图，有效的一个空间注意力对应一个大小为 ![[公式]](https://www.zhihu.com/equation?tex=H%C3%97W) 的矩阵，每个位置对原 feature map 对应位置的像素来说就是一个权重，计算时做 pixel-wise multiply 。

   - 基于 channel 进行 global max pooling 和 global average pooling ;
   - 将上述的结果基于 channel 做 concat ;
   - 将 concat 后的结果经过一个 7x7 卷积操作， channel 降为1;
   - 将结果经过 sigmoid 生成 spatial attention feature ，可以与输入的特征图做乘法，为 feature map 增加空间注意力。

2. 通道域

   SE Block

   这种注意力主要分布在 channel 中，表现在图像上就是对不同的图像通道的关注程度不同。反映在数学上就是指：针对某个大小为 H×W×C 的 feature map ，有效的一个通道注意力对应一个大小为 1×1×C 的矩阵，每个位置对原特征图对应 channel 的全部像素是一个权重，计算时做 channel-wise multiply 。

3. 混合域

4. Residual attention learning

   







