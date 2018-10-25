# DenseNet  

### 代码功能   
使用slim框架复现论文《Densely Connected Convolutional Networks》中的DenseNet网络  
![](./pic/densenet1.png 'densenet1')   
![](./pic/densenet2.png 'densenet2')   

### DenseNet简介   
1）densenet实现分析  
- 输入图片为224x224x3，首先使用7x7卷积核，2步长进行卷积，将图片变为112x112大小，再使用3x3卷积核，2步长的最大池化，得到56x56的feature map，此时通道数为2*growth，即48    
- 进行4个dense block的变换（特征抽取过程），每个block都使用相同的1x1 的bottleneck layer和3x3的卷积核进行多次重复操作，每个block中的重复次数不一，每个小模块的输入都是该block内部前面所有输入的拼接（xl= H([x0,x1,...xl-1])），为了保证能正常拼接，同一个block中的feature map空间尺寸保持不变，但通道数增加。为了限制通道数的快速增加，每个小模块进行3x3卷积前增加了1x1的bottleneck layers，既能降维减少计算量又能融合各个通道。  
- 每个block之间通过transition layers进行连接，整个网络的feature map空间尺寸在这些transition layers中依次减小。每个block最终输出的feature map通道数为growth0 + growth * layers，虽然内部有bottleneck进行通道数的限制，但整个block最终输出的feature map通道数依旧很大，transition layers通过1x1的卷积对通道数进行降维（融合了每个block输出的各个通道），再通过2x2的平均池化对空间尺寸进行降维，使输入到下一个block的feature map达到一个合理的特征尺寸  
- 在最后一个block输出时，使用7x7的全局平均池化，将整个feature map的尺寸变成一个长条形的1x1xN的特征向量，并在全连接后用于softmax进行分类
- 论文中所涉及到的卷积，实际上是一个复合操作，包含了BN（批归一化） - RELU（激活） - conv（卷积） - dropout（随机失活）  
![](./pic/densenet3.png 'densenet3')   

2）growth rate  
- 论文中提出的参数growth rate（增长率K）关联与整个densenet的中每个阶段的feature map通道数的变化（K的值是个超参数，本代码中为24），通道数的变化以K为一个单位，进行整数倍的增加，应用于所有进行通道数变换的操作，如最开始的初始化卷积操作使用2growth，block内部的3x3卷积使用1growth，bottleneck layer使用4*growth rate限制通道数，而transition layer中的卷积操作使用reduction参数（取值0~1之前的比例）来控制通道数。 growth控制着整个网络的宽度（避免通道数增长太快），选择小的growth可以让网络变窄，计算的参数数量也会减少。  
  
3）densenet网络的优势  
- resnet：xl=H(xl-1)+(xl-1)  
- densenet：xl= H([x0,x1,...xl-1])  
相对于resnet来说，densenet使用前方更多的feature map，在前向传播的时候能更充分的抽取特征，加强了feature的传递，在反向传播的时候也能减缓梯度消失的问题，也正因为如此，对于densenet来说，网络的深度限制会放宽。
dense block的结构可以使feature得到更有效的利用，而且由于增加了bottleneck layers和transition layers（使用基本的growth rate）对通道数进行限制，可以使网络变窄，参数数量也降低，一定程度上减轻了过拟合的问题（当然也有dropout的功劳）  




框架参考  https://github.com/tensorflow/models/tree/master/research/slim   
论文参考 [《Densely Connected Convolutional Networks》](https://arxiv.org/abs/1608.06993)   