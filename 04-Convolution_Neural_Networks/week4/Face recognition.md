# Face recognition

## 1. What is face recognition？

​		人脸识别这个词，通常说的是人脸验证和人脸识别。

​		人脸识别比人脸验证问题复杂多了，假设你有一个验证系统它能达到99%的正确率，99%可能还不错，但如果假设在一个识别系统里k等于100，如果你应用这个系统来实现数据库里100个人的识别任务，你现在就有100倍的概率犯错，如果对每个人判断错误的概率是1%，如果你有一个100人的数据库，并且你希望有一个可接受的误差，你可能需要一个有99.9%甚至更高准确率的识别系统，才能将它应用到100人的数据库。事实上，如果你的数据库有100人，很可能需要比99%大得多才能效果不错。

​		接下来会关注建立一个人脸验证系统作为一个构件，而如果准确度足够高的话，那么你可能也会在识别系统中使用它。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\What_is_face_recognition.PNG)

## 2. One-shot learning

​		人脸识别的挑战之一是你需要解决单样本学习的问题，也就是说，对于大多数人脸识别的应用，你需要能够在只讲过一张照片的前提下认出一个人或者只见过一个人脸部的例子。

​		过去经验表面，如果只有一个训练样本，那么深度学习算法的效果很不好。

​		在单样本学习问题中，你必须从一个样本中学习，就可以做到能认出这个人，大多数人脸识别系统都需要做到这样，因为你的数据库中可能只有一张您员工的照片，你可以尝试的一种方法是输入这个人的照片到卷积神经网络，使用softmax单元来输出4种或者5种标签分别对应这4个人，或者4个都不是，所以在softmax里我们会有5种输出，但实际上这样效果并不好，因为如此小的训练集，不足以训练一个稳定的神经网络。而且如果有新人加入你的团队，你现在将会有5个组员需要识别，所以输出就变成了6种，这时你又要重新训练你的神经网络吗？这听起来实在不像一个好办法。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\One-shot_learning1.PNG)

​		所以要让人脸识别能够做到一次学习，为了能有更好的效果，你现在要做的应该是学习similarity函数。详细的说，你想要神经网络学习这样一个用d表示的函数，它以两张图片作为输入，然后输出这两张照片的差异值，如果你放进同一人的两张照片，你希望它能输出一个很小的值，如果放进两个长相差别很大的人的照片，它就输出一个很大的值。

​		所以在识别过程中，如果这两张照片的差异值小于某个阈值τ，它是一个超参数，那么这时就能预测这两张图片是同一人，如果差异值大于τ，就能预测这是不同的两个人，这就是解决人脸验证问题的一个可行办法。

​		要将它应用于人脸识别任务中，你要做的是拿这张新图片然后用d函数去比较这两张照片，这样可能会输出一个非常大的数字，之后，你再让它和数据库的第二张图片比较，因为这两张照片是同一人，所以我们希望会输出一个很小的数，然后你再用它与数据库中的其它图片进行比较，通过这样计算，最终你知道这个人是谁。对应的，如果某个人不在你的数据库中，你通过函数d将他们的照片两两进行比较，最后我们希望d会对所有的比较都输出很大的值，这就是证明这个人并不是数据库中4个人的其中一个，要注意在这个过程中，你是如何解决一次性学习问题的，只要你能学习到这个函数d通过输入一对图片，它将会告诉你，这两张图片是否是同一个人，如果之后有新人加入了你的团队，你只需将它的照片加入你的数据库，系统仍然能照常工作。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\One-shot_learning2.PNG)

## 3. Siamese network

​		实现d函数功能的一个方式就是使用Siamese网络。

​		关注最后一层的向量，假设它有128个数，它是由网络深层的全连接层计算出来的，我们给这128个数字命名为f(x^(1))，你可以把它看作是输入图像x^(1)的编码，取这个输入图像，然后表示成128维的向量，建立一个人脸识别系统的方法就是如果你要比较两个图片的话，你要做的就是把第二张图片喂给有着相同参数的同样的神经网络，然后得到一个不同的128维的向量，这个向量代表或编码第二个照片。最后如果你相信这些编码很好的地代表了这两个图片，你要做地就是将x^(1)和x^(2)之间的距离d定义为两幅图片的编码之差的范数，对于两个不同的输入运行相同的卷积神经网络，然后比较它们，这一般叫做Siamese网络架构。

​		怎么训练这个Siamese神经网络呢？不要忘了这两个网络有相同的参数，所以你实际要做的就是训练一个网络，它计算得到的编码可以用于计算函数d，它可以判断两张照片是否是同一人，更确切地说神经网络地参数定义了一个编码函数f(x^(i))，如果给定输入图像x^(i)，这个网络会输出x^(i)的一个128维的编码，你要做的就是学习参数，使得如果两个图片x^(i)和x^(j)是同一人，那么你得到的两个编码的距离就小，相反，如果x^(i)和x^(j)是不同的人，那么你会想让它们之间的编码距离大一点，如果你改变这个网络所有层的参数，你会得到不同的编码结果，你要做的就是用反向传播，来改变这些所有参数以确保满足这些所有条件。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\Siamese_network.PNG)

## 4. Triplet loss

​		要想通过学习神经网络的参数来得到优质的人脸图片编码，方法之一就是定义三元组损失函数然后应用梯度下降。

​		为了应用三元组损失函数，你需要比较成对的图像，为了学习网络的参数，你需要同时看几幅图片。用三元组损失的术语来说，你要做的通常是看一个anchor图片，你想让anchor图片和positive图片的距离很接近，然而当anchor图片于negative图片对比时，你会想让它们的距离离得更远一些，这就是为什么叫做三元组损失，它代表你通常会同时看三张图片。

​		把这些写成公式的话，你想要的是网络参数或者编码能满足如图中所示特性。为了确保网络对于所有编码不会总是输出0，也为了确保它不会把所有编码都设成相互相等的，我们需要修改这个目标函数，也就是说这个不能刚好小于等于0，所以这个应该小于一个负的α，这里α是另一个超参数，这个就可以阻止网络输出无用的结果，α也叫做间隔。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\Triplet_loss1.PNG)

​		三元组损失函数的定义基于三张图片，通过最小化这个损失函数，达到的效果就是使这部分成为0或者小于0。整个网络的代价函数应该是训练集中这些单个三元组损失的总和。

​		假设你有一个10000个图片的训练集，里面是1000个不同的人的照片，你要做的就是取这10000个图片，然后生成三元组，然后训练你的算法，对这种代价函数用梯度下降，这个代价函数就是定义在你数据集的这样的三元组图片上，注意为了定义三元组的数据集，你需要成对的A和P，即同一人的成对的图片，为了训练你的系统，你确实需要一个数据集，里面有同一个人的多张照片，当然训练完这个系统之后，你可以应用到你的一次性学习问题上，对于你的人脸识别系统可能你只有想要识别的某个人的一张照片，但对于训练集，你需要确保有同一个人的多张照片。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\Triplet_loss2.PNG)

​		现在我们来看，你如何选择这些三元组来形成训练集。一个问题是如果你从训练集中随机选择A、P、N，遵守A和P是同一个人，而A和N是不同的人的原则，有个问题就是，如果随机的选择它们，那么这个约束条件很容易达到，因为随机选择的图片A和N比A和P差别很大的概率很大，这样网络并不能从中学到什么。

​		所以为了构建一个数据集，你要做的就是，尽可能选择难训练的A、P、N，难训练的三元组就是你的A、P和N的选择使得d(A,P)很接近d(A,N)，这样你的学习算法会竭尽全力使d(A,N)变大，使d(A,P)变小，这样左右两边至少有一个α间隔，并且选择这样的三元组还可以增加你算法的计算效率。如果随机选择这些三元组，其中会有许多会很简单，梯度算法不会有什么效果，因为网络总是很轻松，就能得到正确的结果，只有选择难得三元组，梯度下降法才能发挥作用。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\Triplet_loss3.PNG)

## 5. Face verification and binary classification

​		三元组损失是一个学习人脸识别卷积神经网络参数的好方法，还有其他学习参数的方法，让我们看看如何将人脸识别当成一个二分类问题。

​		另一个训练神经网络的方法是选取一对神经网络，选取Siamese网络使其同时计算这些嵌入，然后将其输入到逻辑回归单元，然后进行预测，如果是相同的人，那么输出1，若是不同的人，输出是0，这就把人脸识别转换成二分类问题，训练这种系统时可以替换三元组损失方法，最后的逻辑回归单元是怎么处理的：

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\Face_verification_and_binary_classification1.PNG)

​		不需要每次都计算这些嵌入（图中紫色圈），你可以提前计算好，那么当一个新员工走近时，你可以使用上方的卷积神经网络，来计算这些编码，然后使用它和预先计算好的编码进行比较，然后输出预测值y^hat，因为不需要存储原始图像，如果你有一个很大的员工数据库，你不需要为每个员工每次都计算这些编码，这个预先计算的思想，可以节省大量计算这个预训练的工作，将人脸识别当作一个二分类的问题也可以用在三元组损失函数上。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\Face_verification_and_binary_classification2.PNG)

​		总结：把人脸验证当作一个监督学习，创建一个只有成对图片的训练集，不是三个一组，而是成对的图片，目标标签是1，表示一对图片是一个人，目标标签是0，表示图片中是不同的人，利用不同的成对图片，使用反向传播算法去训练神经网络，训练Siamese神经网络。

# Neural Style Transfer

## 1. What is neural style transfer？

​		为了实现神经风格迁移，你需要查找卷积网络提取的特征在不同的神经网络。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\What_is_neural_style_transfer.PNG)

## 2. What are deep ConvNets learning?

​		如图Alex网络，从第一层的隐藏单元开始，假设你遍历了训练集，然后发现一些图片最大化的激活了那个运算单元，换句话说，将你的训练集经过神经网络，然后弄明白哪一张图片最大限度地激活了特定单元，注意在第一层的隐层单元只能看到小部分卷积神经网络，如果要画出来哪些激活了激活单元，只有一小块图片块是有意义的，因为这就是特定单元所能看到的全部，你可能找到这样九个图片块，似乎是图片浅层区域，显示了隐层单元所看到的，找到了像这样的边缘或者线，激活了隐层单元的激活项的图片块，然后你可以选一个另一个第一层的隐层单元，对其他隐层单元也进行处理，会发现其他隐层单元趋向于激活类似于这样的图片。你可以这样理解，通常会找一些简单的特征。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\What_are_deep_ConvNets_learning1.PNG)

​		在更深的层中，一个隐层单元会看到一张更大的部分，在极端的情况下，靠后的隐层单元可以看到更大的图片块。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\What_are_deep_ConvNets_learning2.PNG)

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\What_are_deep_ConvNets_learning2.PNG)

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\What_are_deep_ConvNets_learning3.PNG)

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\What_are_deep_ConvNets_learning4.PNG)

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\What_are_deep_ConvNets_learning5.PNG)

​		所以我们有了一些进展，从检测简单的事物，比如说第一层的边缘、第二层的质地到深层的复杂事物。

## 3. Cost function

​		在风格迁移中，通过最小化代价函数，你可以生成你想要的任何图像。

​		为了实现神经风格迁移，你要做的就是定义一个关于G的代价函数J，用来评判某个生成图像的好坏，我们将使用梯度下降法去最小化J(G)以便生成这个图像，怎么判断生成图像的好坏呢？我们把这个代价函数定义为两个部分，第一部分被称作内容代价，这是一个关于内容图片和生成图片的函数，它是用来度量生成图片的内容与内容图片C的内容有多相似，然后我们会把结果加上一个风格代价函数，也就是关于S和G的函数，用来度量图片G的风格和图片S的风格的相似度，最后我们用α和β来确定内容代价和风格代价两者之间的权重。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\Cost_function1.PNG)

​		接下来要做的就是随机初始化生成图像G，然后用上面定义的代价函数J(G)，你现在可以做的是，使用梯度下降的方法将其最小化。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\Cost_function2.PNG)

## 4. Content cost function

​		假设你用隐层l来计算内容代价，如果l是个很小的数，比如用第一层，这个代价函数就会使你的生成图片像素上非常接近你的内容图片，然而如果你用很深的层，那么那就会问内容图片里是否有狗，然后它就会确保生成图片里面有一个狗，所以在实际中，这个层l在网络中既不会选的太浅也不会太深，通常l会选在网络的中间层，然后用一个预训练的卷积模型，可以是VGG模型，或者其他网络，现在你需要衡量假设有一个内容图片和一个生成图片，它们在内容上的相似度，如果图中两个激活值相似，那么就意味着两个图片内容相似。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\Content_cost_function.PNG)

## 5. Style cost function

​		图片的风格到底是什么意思？这么说，比如你有这样一张图片，你可能已经对这个计算很熟悉了，它能算出这里是否含有不同的隐层，现在你选择了某一层l，去为图片的风格定义一个深度测量，现在我们要做的就是将图片的风格定义为l层中各个通道之间激活项的相关系数。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\Style_cost_function1.PNG)

​		如何知道这些不同通道之间激活项的相关系数呢？现在注意这个激活块，把它的不同通道渲染成不同的颜色，为了能捕捉图片的风格，你需要进行下面这些操作。

​		如果我们在通道之间使用相关系数来描述通道的风格，你能做的就是测量你的生成图像中，第一个通道是否与第二个通道相关，通过测量，你能得知在生成的图像中垂直纹理和橙色同时出现或者不同时出现的频率，这样你将测量生成图像的风格与输入的风格图像的相似程度。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\Style_cost_function2.PNG)

​		对于这两个图像，也就是生成图像和风格图像，你需要计算一个风格矩阵，说得更具体一点，就是用l层来测量风格。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\Style_cost_function3.PNG)

​		所以你要做的就是计算出这张图像的风格矩阵，以便能够测量出刚才所说的这些相关系数。这就是计算图像风格的方法，你要同时对风格图像S和生成图像G都进行这个运算。最后，如果将得到这两个矩阵的误差。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\Style_cost_function4.PNG)

​		实际上，你对每一层都使用风格代价函数会让结果变得更好。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\Style_cost_function5.PNG)

# Convolutional Networks in1D or 3D

## 1. 1D and 3D generalizations of models

​		我们大部分讨论的图像数据，某种意义上而言都是2D数据。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\1D_and_3D1.PNG)

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\1D_and_3D2.PNG)

​		以CT图像为例，本质上，将这个数据是3维的，数据具备一定长度、宽度与高度，其中每一个切片，都与躯干的切片对应。

![](C:\Users\Think\Desktop\吴恩达笔记\04-Convolution Neural Networks\week4\images\1D_and_3D3.PNG)

​		所以，如果你想要在3D扫描或CT扫描中，建立特征识别，如图。