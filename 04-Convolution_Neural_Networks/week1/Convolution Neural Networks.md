# Convolutional Neural Networks

## 1. Computer Vision

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Computer_Vision1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Computer_Vision2.PNG)

## 2. Edge detection example

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Edge_detection_example1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Edge_detection_example2.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Edge_detection_example3.PNG)

​		图中是一个6x6的灰度图像，因为是灰度图像，所以只是一个6x6x1的矩阵，为了边缘检测，比如这个图像中的垂直边缘，你能做的就是建立一个3x3的矩阵，在池化过程中，用卷积神经网络中的专业术语来说，这会被称为一个过滤器（filter或kernel），你需要做的就是获得6x6的图像并求卷积，用这个3x3的过滤器去求它的卷积，这个卷积运算的输出是一个4x4的矩阵，你可以将它理解为一个4x4的图像，计算过程如图。

为什么这是在进行垂直边缘检测？

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Edge_detection_example4.PNG)

​		在图中左下角的图像中，图像的正中间很清晰的有一个明显的垂直边缘，从白色到暗色的过渡，所以当你用这个3x3的过滤器做卷积运算，这个过滤器可以被可视化为图中中间正下方的图像，更明亮的像素点在左边，然后有中间调的颜色在中间，然后更暗的在右边，你得到的是在右边的矩阵，画成图像如图中右下角所示，这与检测出的垂直边缘相对应。这里的维数看起来不太对，检测出来的边缘看起来很厚，那只是因为我们在这个例子中用了个很小的图片。

## 3. More edge detection

​		将颜色翻转后变化如图，如果你不在乎这两个的区别，你可以取输出矩阵的绝对值，但是这个过滤器确实能够区分亮到暗的边界和暗到亮的边界。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/More_edge_detection1.PNG)

​	![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/More_edge_detection2.PNG)

​		如图右上角的过滤器可以检测水平边界。总而言之，不同的过滤器可以找到垂直和水平的边界。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/More_edge_detection3.PNG)

​		事实上，这些3x3的垂直边界检测器只是一个可能的选择，在计算机视觉的文献里，对于用哪些数字组合是最好的仍然存在相当大的争议，你也可以使用不同的组合如Sobel过滤器，这个过滤器的优点在于它给中间行赋予了更大的权重，从而可能使它更加稳定。

​		随着深度学习的发展，如果你想要检测一些复杂图片的边界，可能不需要计算机视觉的研究员挑选出这9个矩阵元素，你可以把矩阵里的这9个元素当做参数，通过反向传播来学习得到它们的数值，除了垂直和水平边界，同样能够学习去检测45度边界或无论什么角度。

## 4. Padding

​		如果你有一个nxn的图片并且想使用一个fxf的过滤器，这样输出的维度是(n-f+1)x(n-f+1)，这其中有两个缺陷：第一个是，如果每一次你使用一个卷积操作，你的图像都会缩小，你做不了几次卷积，你的图片就会变得非常小，每次你想检测边界或者其他特征时都缩小你的图片。第二个缺陷是，图片角落或者边际上的像素只会在输出中被使用一次，你丢失了许多图片上靠近边界的信息。

​		所以为了同时解决上述的两个问题，你能做的是在使用卷积操作前填充（pad）图片，通常当你填充时，你使用0来填充，如果p是填充的数量，所以在这种情况下，p=padding=1，因为我们使用了一个1像素的额外边缘填充了一圈，这样输出变成了(n+2p-f+1)x(n+2p-f+1)，这样的效果大大降低了之前那种边界和角落信息被忽略的严重程度。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Padding1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Padding2.PNG)

​		填充选择：Vaild卷积基本上没有填充，在这种情况下，当你用一个fxf的过滤器去卷积一个nxn的图片你会得到一个(n-f+1)x(n-f+1)维的输出。另一个最常用的填充选择叫做Same卷积，意思是你选择的填充将使输出大小等于输入大小。通常在CV领域，f基本上是使用奇数。

## 5. Stride convolutions

带步长的卷积是在卷积神经网络中，组成基础卷积模块的另一部分。

​		在例子中，我们这次用2为步长的方式进行卷积，这意味着，在左上角3x3的区域进行元素间相乘是不变的，但是，这次不把蓝色区域移动一个步长，而是移动两个步长。

输入和输出的维度间的关系可以用以下的方程进行表示，如果你有一个NxN的图像，用FxF的过滤器对这个图像进行卷积，对图像使用p层填充，并假设步长为S，输出维度为( (n+2p-f)/s+1 )x( (n+2p-f)/s+1 )。在分数中，如果分子不能被分母整除则使用向下取整，不能计算图像外的像素。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Stride_convolutions1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Stride_convolutions2.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Stride_convolutions3.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Stride_convolutions4.PNG)

​		在标准的数学课本中，对卷积的定义，其实在做元素间相乘并求和的之前，你需要将过滤器对水平和竖直轴进行镜像映射，之后将其放进矩阵中进行同样操作。之前将翻转操作省略了，从技术层面上来说，实际上，之前的操作叫交叉相关，不是卷积。

​		总结：在机器学习的约定中，我们通常忽略掉翻转的操作，技术上它应该叫做交叉相关，但是，大多数深度学习的文献都叫它卷积操作。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Stride_convolutions5.PNG)

## 6. Convolutions over volumes

​		假设你想在这图片中检测特征，不仅仅是对灰度图像而是对RGB图像，RGB图像可以看成三张6x6图像的叠加，为了检测这个图像中的图像边缘或一些图片中的其他特征，你使用3x3x3的过滤器，因此过滤器本身将有对应的三层。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Convolutions_over_volumes1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Convolutions_over_volumes2.PNG)

​		如图可以检测红色通道的垂直边缘，或者，你并不在意一个垂直边缘属于哪个颜色，你可以设置下面的过滤器，可以检测任何颜色的边缘。通过使用不同参数，你可以从这个3x3x3的过滤器中得到不同特征的检测器。通常在计算机视觉领域，当你的输入有一个固定的高度和宽度和固定数量的通道，那么你的过滤器可能会有不同的高度和不同的宽度，但有相同数量的通道。

​		如果我们不仅仅想要检测垂直边缘？如果我们想同时检测垂直边缘和水平边缘，又或是45度边缘，换句话来说，如果你想要同时应用多个过滤器呢？

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Convolutions_over_volumes3.PNG)

​		现在你可以检测两个特征，比如垂直、水平边缘或者10个，128个，甚至几百个不同的特征，那么这个输出会是通道的数目等于过滤器的数目。

## 7. One layer of a convolutional network

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/One_layer_of_a_convolutional_network1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/One_layer_of_a_convolutional_network2.PNG)

​		不管图片多大，所用的参数个数都是一样的，这个特征使得卷积神经网络不太容易过拟合。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/One_layer_of_a_convolutional_network3.PNG)

## 8. A simple convolution network example

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/A_simple_convolution_network_example1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/A_simple_convolution_network_example2.PNG)

​		虽然我们可能只是使用卷积层来设计一个相当好的神经网络，大多数神经网络架构也将有好几个池化层和几个全连接层。

## 9. Pooling layers

​		除了卷积层，ConvNets通常还使用池化层来减少展示量，以此来提高计算速度，并使一些特征的检测功能更强大。

​		如图所示，我们将给四个区域赋予不同的颜色，然后，在它的输出里，每个输出值将会是其对应颜色区域的最大值，这相当于你使用了一个大小为2的过滤器，并且使用的步长为2，这些实际上就是max pooling的超参数，因为我们是从这个过滤器大小开始的。

​		**max pooling**（最大值采样）背后的机制：如果你把这个4x4的区域看作某个特征的集合，即神经网络某个层中的激活状态，那么一个大的数字意味着它或许检测到了一个特定的特征，所以左侧上方的四分之一区域有这样的特征，它或许是一个垂直的边沿，亦或是一个更高或更弱，显然，左侧上方的四分之一区域有那个特征，然而这个特征，或许它不是猫眼检测，但是，右侧上方的四分之一区域没有这个特征，所以，max pooling做的是，检测到所有地方的特征，四个特征中的一个被保留在max pooling的输出里。所以，max pooling所做的其实是，如果在滤波器中任何地方检测到了这些特征就保留最大的数值，但是，如果这个特征没有被检测到，可能左侧上方的四分之一区域就没有这个特征，于是，那些数值的最大值仍然相当小。

​		但是不得不承认，大家使用max pooling的主要原因是因为在很多实验中发现它的效果很好。max pooling的一个有趣的特性是，它有一套超参数，但是它没有任何参数要学习。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Pooling_layers1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Pooling_layers2.PNG)

​		卷积层所推导出的公式对max pooling同样适用。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Pooling_layers3.PNG)

​		目前最大值采样的使用通常比均值采样多得多，唯一例外是有时候在深度非常大的神经网络，你也许可以使用均值采样来合并表示。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Pooling_layers4.PNG)

## 10. Convolutional neural network example

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Convolutional_neural_network_example1.PNG)	

​		神经网络中当人们说到网络层数的时候，通常是指那些有权重，有参数的网络层数量，因为池化层没有权重，没有参数，只有一些超参数。

​		一个常用的法则实际上是，不要试着创造你自己的超参数组，而是查看文献，看看其他人使用的超参数，从中选择一组适用于其他人的超参数，很可能它也适用于你的应用。

​		随着网络的深入，高度和宽度会减小，然而通道数会增加。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Convolutional_neural_network_example2.PNG)

​		这里需要指出几点：首先，注意最大池化层没有任何参数，其次，注意卷积层趋向于拥有越来越少的参数，实际上，多数参数在神经网络的全连接层上，同时，随着神经网络的深入，你会发现激活输入大小也逐渐变小，如果减少太快，通常也不利于网络性能。

## 11. Why convolutions？

​		卷积层和只用完全连接的神经层比起来有两个优势，**参数共享**和**连接的稀疏性**。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Why_convolutions1.PNG)

​		卷积神经网络参数少得多。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Why_convolutions2.PNG)

​		卷积神经网络参数很少的原因有两个：一个是参数共享，二是建立稀疏联系。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/04-Convolution_Neural_Networks/week1/images/Why_convolutions3.PNG)

