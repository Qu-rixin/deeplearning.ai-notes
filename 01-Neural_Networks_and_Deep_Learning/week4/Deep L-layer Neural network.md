# Deep L-layer Neural network

有一些函数，只有非常深的神经网络才能学习，而比较浅的神经网络模型无法做到。

对于特定的问题，可能事先很难知道需要多深的网络，所以一般先尝试逻辑回归，再尝试一个或两个隐层，可以把隐层数量作为另一个超参数，不断尝试，然后通过交叉验证或开发集进行评估。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Deep_L-layer_Neural_network.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Deep_L-layer_Neural_network2.PNG)

## 1. Forward and backward propagation

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Forward_and_backward_propagation1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Forward_and_backward_propagation2.PNG)

注：*指的是逐个元素相乘。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Forward_and_backward_propagation3.PNG)

注：这里z[l]用到了前面前向传播缓存的z[l]。

## 2. Forward Propagation in a Deep Network

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Forward_Propagation_in_a_Deep_Network1.PNG)

在计算每一层参数的时候，没有比for更好的方法：

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Forward_Propagation_in_a_Deep_Network2.PNG)

在实现神经网络的过程中，想增加得到没有bug程序的概率，其中个方法就是非常仔细和系统化地思考矩阵的维数。

## 3. Getting your matrix dimensions right

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Getting_your_matrix_dimensions_right1.PNG)

向量化后，w、b、dw、db的维度始终是一样的，但Z、A以及X的维度会在向量化后发生改变，向量化：

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Getting_your_matrix_dimensions_right2.PNG)

注：m是训练集的大小。

## 4. Why deep representations?

如果你在建立一个人脸识别或者人脸探测系统，深度神经网络所做的事就是：当你输入一张脸部的照片，然后你可以把深度神经网络的第一层当成一个特征探测器或者是边缘探测器，在图中小方块就是一个隐层单元，它可能会去找照片里的边缘的方向，也可能找水平向的边缘在哪里。在之后的课程中会讲卷积神经网络，现在可以现把神经网络的第一层当作看图，然后去找这张照片的各个边缘。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Why_deep_representations.PNG)

我们可以把照片里组成边缘的像素放在一起看，然后它可以把被探测的边缘组合成面部的不同部分，比如说，可能有一个神经元会去找眼睛的部分，另外还有别的部分去找鼻子的部分，然后把这许多的边缘结合在一起，就可以检测人脸的不同部分，最后再把这些部分放在一起，就可以探测识别不同的人脸了。

可以直觉的把这种神经网络的前几层，当作探测简单的函数，比如边缘，之后把它们跟后几层结合在一起，那么总体上就可以学习到更复杂的函数了。

一个技术性的细节需要理解：边缘探测器其实相对来说都是针对照片中非常小块的面积，但主要的概念是，一般会从比较小的细节入手，比如边缘，然后一步步到更大更复杂的区域。

这种从简单到复杂的金字塔状的表示方法也可以应用在图像或人脸识别以外的其他数据上。

所以在深度神经网络的隐层中，较早的前几层能学习到一些低层次的简单特征，等到后几层就能把简单的特征结合起来去探测更复杂的东西，同时我们所计算的之前的几层，也就是相对简单的输入函数。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Why_deep_representations2.PNG)

## 5. Building Blocks of deep neural networks

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Building_Blocks_of_deep_neural_networks1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Building_Blocks_of_deep_neural_networks2.PNG)

注：把反向函数计算出来的z值缓存下来，这会使

## 6. Parameters vs Hyperparameters

超参数（控制实际参数的参数）：需要自己设置的参数，这些参数最终决定了w和b的值。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Parameters_vs_Hyperparameters1.PNG)

注：α会加快学习过程，并且收敛在更低的损失函数值上，就用这个α。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/Parameters_vs_Hyperparameters2.PNG)

## 7. What does this have to do with the brain?

![](https://github.com/Qu-rixin/deeplearning.ai-notes/tree/master/01-Neural_Networks_and_Deep_Learning/week4/images3/What_does_this_have_to_do_with_the_brain.PNG)

