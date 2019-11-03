# Case Studies

## 1. Why look at case studies？

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/Why_look_at_case_studies.PNG)

## 2. Classic networks

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/LeNet-5.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/AlexNet.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/VGG-16.PNG)

## 3. Residual Networks(ResNets)

​		太深的神经网络训练起来会很难，因为有梯度消失和爆炸这类问题。

​		残差网络（ResNet）是使用了残差结构的网络。

​		从a^[l]到a^[l+1]需要经过所有这些步骤，称作这组层的主路径。在残差网络中，如图中紫线部分出现了short cut，这意味着最后一个等式消失了，取而代之为下方等式，使这成为了一个残差块（Residual block），把信息传到更深的神经网络中。

​		使用残差块让你可以训练更深的神经网络，而你建立一个ResNet的方法就是通过大量的这些残差块堆叠起来形成一个深层网络。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/ResBlock.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/ResNet.PNG)

## 4. Why ResNets work?

​		残差网络有效的主要原因是这些额外层学习起恒等函数非常简单，几乎总能保证它不会影响总体的表现，甚至许多时候可以提升网络的表现，神经网络中另一个值得讨论的细节是在图中红线部分，我们假定两项的维度是相同的，如果维度不同，我们做的就是增加一个额外的矩阵，它可以是一个学习到的矩阵或者是固定的矩阵。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/Why_ResNets_work1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/Why_ResNets_work2.PNG)

## 5. Network in Network and 1x1 convolutions

​		对1x1卷积的一种理解是它本质上是一个完全连接的神经网络，逐一作用于这36个不同的位置，这个完全连接的神经网络所做的就是它接受32个数的输入，然后输出过滤器的数个输出值，然后对这36个位置中的每一个都进行相同的操作。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/Network_in_Network1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/Network_in_Network2.PNG)

​		1x1卷积的应用：缩小nh和nw可以使用池化层，缩小nc可以使用1x1卷积，从而达到在网络中减少计算量。

## 6. Inception network motivation

​		inception网络或inception层是指与其在卷积神经网络中选择一个你想使用的卷积核尺寸乃至选择你是否需要一个卷积层或一个池化层。

​		这是inception网络的核心，并且这个网络最基础的一个特点就是，你不用去只挑选一个卷积核的大小或pooling，你可以所有可能都做，然后把的所有的输出结果都连接起来，然后让神经网络去学习它想要用到的参数，以及它想要用到的卷积核大小。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/Inception1.PNG)

​		inception网络的问题：计算成本问题。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/Inception2.PNG)

​		第二种方法中，我们先将这个较大的输入减小成一个较小的中间值，这个中间层叫瓶颈层，运算成本从1.2亿次运算减小到了大概1240万次运算。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/Inception3.PNG)

​		你只要合理的实现这个瓶颈层，你既可以缩小输入张量的维度又不会影响到整体性能，还能节省计算成本。

## 7. Inception Network

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/Inception_Network1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/Inception_Network2.PNG)

​		额外的旁枝作用是它把隐层作为输入来做预测。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/Inception_Network3.PNG)

# Practical advice for using ConvNets

## 1. Using open-source implementations

​		事实证明许多神经网络都很难复现，因为许多关于超参数调整的细节。在github上下载代码，并在下载好的代码上开发，这样的好处是这些网络可能需要很长时间来训练，也许有人已经用了多个GPU和非常大的数据集训练好了，这样你就可以直接对这些网络使用迁移学习。

## 2. Transfer learning

​		如果你想实现一个计算机视觉应用，而不想从零开始训练权重，比方说从随机初始化开始训练，实现更快的方式是下载已经训练好权重的网络结构，把这个作为预训练，迁移到你感兴趣的新任务上，计算机视觉的研究社区已经很擅长把许多数据库发布在网络上，如ImageNet、MSCOCO、PASCAL等，许多计算机视觉的研究者已经在上面训练了自己的算法，有时算法训练要耗费好几周时间，占据许多GPU。这意味着你可以下载这些开源的权重为你自己的神经网络做好初始化开端，并且可以用迁移学习来迁移知识，从这些大型数据库迁移知识到你自己的问题上。

​		你可以下载神经网络的开源应用不但下载源码，还要下载相应的权重，你可以下载在有1000类物体的ImageNet数据库上训练的神经网络，因此该网络有一个可以输出千分之一类别概率的softmax神经元，你能做的是去掉其softmax层然后创造自己的softmax层来输出Tigger/Misty/其他，从网络方面来看，建议你冻结前面层的参数，因此你可以只训练与你自己softmax层有关的参数。通过别人训练好的权重，即使在很小的数据库上也可能得到很小的性能。

​		如果你有更大的数据集，你可以冻结更少的层数，你可以用最后几层的权重，作为初始化开始做梯度下降，或者你也可以去掉最后几层，然后用自己的新神经元和最终softmax输出。但有个模式，即你的数据越多，所冻结的层数可以越少，其中的想法就是如果你选了个数据集，有足够的数据，不仅可以训练单层softmax，还可以训练由所采用网络的最后几层组成的中型网络，其中最后几层你可以使用。

​		最后如果你有许多数据，你可以用该开源网络和权重，用它们初始化整个网络然后训练。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/Transfer_learning.PNG)

## 3. Data augmentation

​		大多数的计算机视觉工作能用到很多数据，因此数据增强是其中一种常用技术用于来改善计算机视觉系统的性能。

​		最简单的数据增强方式应该是对图像做垂直镜像。另一种常用技术是随即裁剪，这样就获得了不同的示例来扩充你的训练集。

​		第二类常用的数据增强方式是色彩变换。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/Data_augmentation1.PNG)

​		一种常见的数据增强方法是，通过一个线程用来加载数据并作失真处理，然后传递给其他线程来做深度学习训练，通常这两部分可以并行。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/Data_augmentation2.PNG)

## 4. The state of computer vision

​		你可以认为大多数机器学习问题是从拥有相对较少数据到拥有大量数据之间的问题。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/The_state_of_computer_vision1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/The_state_of_computer_vision2.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week2/images/The_state_of_computer_vision3.PNG)

