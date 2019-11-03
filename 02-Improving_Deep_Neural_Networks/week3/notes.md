# Hyperparameter tuning

## 1. Tuning process

​		调整神经网络的过程，包含了对很多不同超参数的设置。

​		有一种痛苦，是在深度神经网络训练时面对大量超参数，包括学习率α，如果使用动量算法时还包括动量超参数β，还有Adam算法里的β1、β2、ε，也许还包括网络层数，以及每层网络中隐层的数量，然后你需要选择mini-batch的大小。

​		根据实际经验，一部分超参数比其他的更重要，对于大多数学习算法的应用，学习率α时需要调优的超参数中最重要的一个，没有之一，除了α，接下来调整的一些超参数也许是动量项β，还可以调整mini-batch大小来保证最优化算法的运行效率，还可以调整隐层单元数量。接下来重要性排在第三的超参数，网络层数有时候对结果起到重要作用，学习率衰减有时也一样，当使用Adam算法时，几乎不调整β1、β2和ε。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Tuning_process1.PNG)

​		在深度学习中使用在网格中进行随机取样，这样做的原因是事先你很难知道，在你的问题中，哪个超参数是最重要的，有一些超参数实际上比其他的重要很多。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Tuning_process2.PNG)

​		当对超参数进行取样时，另一种常见的做法是，使用区域定位的抽样方案，即在产生不错结果的点所在区域进行限定，然后在这个区域内进行密度更高的抽样。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Tuning_process3.PNG)

## 2. Using an appropriate scale to pick hyperparameters

​		实际上，随机抽样并不意味着在有效值范围内的均匀随机抽样，相反更重要的是选取适当的尺度用以研究这些超参数。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Using_an_appropriate_scale_to_pick_hyperparameters1.PNG)

对数尺度上的取样方法的实现：

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Using_an_appropriate_scale_to_pick_hyperparameters2.PNG)

最后，另一个棘手的情况是超参数β的取样：

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Using_an_appropriate_scale_to_pick_hyperparameters3.PNG)

## 3. Hyperparameters tuning in practice:Pandas vs. Caviar

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Hyperparameters_tuning_in_practice1.PNG)

​		对于探寻超参数有两种主流思想，其中一种场景，是你精心设计一种模型，通常你需要处理一个庞大的数据集，但没有充足的计算资源，你需要照看你的模型，观察性能曲线，耐心地微调学习率。另一种情形，你有充足的计算资源，并行训练许多个模型，这种情况下你可能设置一些超参数，然后让模型运行一天或几天。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Hyperparameters_tuning_in_practice3.PNG)

# Batch Normalization

在深度学习不断兴起的过程中，最重要的创新之一是一种叫批量归一化的算法，它可以让你的超参数搜索变得更简单，让你的神经网络变得更加具有鲁棒性，可以让你的神经网络对于超参数的选择上不再那么敏感，而且可以让你更容易训练非常深的网络。

## 1. Normalizing activations in a network

​		对输入特征进行归一化可以加速学习过程，对于层数更多的模型，你不仅有输入特征值x，你还有各层激活函数的输出结果可以归一化，这就叫做批量归一化。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Normalizing_activations_in_a_network1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Normalizing_activations_in_a_network2.PNG)

​		通过设置γ和β你可以控制z(i)在你希望的范围内，或者说通过这两个参数来让你的隐层单元有可控的均值和方差。

## 2. Fitting batch norm into a neural network

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Fitting_batch_norm_into_a_neural_network1.PNG)

​		实际上，通常应用到训练集上的是mini-batch BN算法，所以实际使用BN算法时，我们取第一个mini-batch并计算z1，然后我们取一个合适的mini-batch，并利用它计算z1的均值和方差，接着BN算法就会减去均值，除以标准差，然后用β1和γ1来重新调整，从而得到z1（带波浪号），这些都是在第一个mini-batch上计算的，然后通过激活函数来计算a1，然后通过w2、b2来计算z2，以此类推，在第一个mini-batch上计算梯度下降，接着转到第二个mini-batch上，也就是x2。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Fitting_batch_norm_into_a_neural_network2.PNG)

​		不管b^[l]的值是多少，实际上都要被减去，因为经过BN这一步，我们将计算z^[l]的均值并将其减去，所以在mini-batch中对所有例子加上一个常量并不会改变什么，因为无论我们加上什么常量，它都会被减均值。所以如果你使用BN算法，可以忽略b或者认为它永远等于0。

​		因为BN算法使层级中各个z^[l]的均值为0，我们就没有理由保留参数b^[l]，相应的被β^[l]所代替。最后记住z^[l]的维度如图。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Fitting_batch_norm_into_a_neural_network3.PNG)

## 3. Why does batch norm work?

​		其中一个理由是我们看到经过归一化的输入特征，它们的均值为0方差为1，这将答大幅加速学习过程，所以与其含有某些在0到1范围内变动的特征，通过归一化所有的输入特征让它们都拥有相同的变化范围将加速学习，所以BN算法有效的另一个原因是它同样如此，只不过它应用于隐层的值，而不是这里的输入特征。

​		BN算法有效的第二个原因是，它产生权重，在深层次网络中，假设我们的数据集是在所有黑猫图片上训练得到的，现在我们将其应用到，所有的正确结果的彩色猫图片上，那么我们的分类结果将会出错，不要指望你的学习算法可以找到那个绿色的决策边界。

​		数据分布随着协变量在变化，想法是，如果我们学习到了某种x-y映射，如果x的分布变化了，那么我们就得重新训练学习算法。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Why_does_batch_norm_work1.PNG)

那么协变量问题如何影响神经网络呢？

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Why_does_batch_norm_work2.PNG)

​		BN算法所做的就是，它减少了这些隐层单元值的分布的不稳定性，BN算法确保的是无论它怎么变输出的均值和方差保持不变，它们的均值一直为0，方差一直为1，可能它们的值不是0或1，但它们的值由β和γ控制。所以BN算法减少输入值变化所产生的问题，它的确使这些值变得稳定。实际上尽管前几层继续学习，后面层适应前面层变化的力量被减弱，BN算法削弱了前面层参数和后层参数之间的耦合，它允许网络的每一层独立学习，所以这将有效提升整个网络的学习速度。

​		结论是，BN算法意味着，尤其是从神经网络某后一层角度看，前面层的影响不会很大，因为它们被同一均值和方差限制。BN算法还有第二个效果，它具有轻微的正则化效果。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Why_does_batch_norm_work3.PNG)

​		通过使用更大尺寸的mini-batch，不仅减少噪声的同时也减少了正则化效果，一定不要将BN算法看作是正则化方法，而是把它来作为归一化，隐层单元激活函数用以加速学习的方法。

## 4. Batch Norm at test time

​		BN算法每次处理一个mini-batch的数据，但是在测试时，你大概会需要一个个实例来处理。

​		为便于在测试时使用神经网络，我们需要一种单独的方式来估算μ和σ，在BN中我们通常用指数加权平均数来估算的，这个平均数是根据

mini-batch来计算的。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Batch_Norm_at_test_time.PNG)

# Multi-class classification

## 1. Softmax regression

​		有一种更普遍的逻辑回归方法叫softmax回归，这种方法能让你在一个多种分类中的类别时做出预测，而不是识别两类中的类别。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Softmax_regression1.PNG)

​		这种情况下，我们就需要构建一个新的神经网络，其输出层单元为4个，我们想要得到的是输出层的单元告诉我们每个类别的概率，所有概率相加为1，标准化做法就是使你的网络使用这里的softmax层，以及生成这些输出层，在网络的最后一层，你需要像往常一样计算每层的线性部分，现在你需要把这个softmax激活函数用起来，这个激活函数和softmax层有些不同，首先我们计算一个临界的值t，t这个公式就是对所有元素求幂，如图。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Softmax_regression2.PNG)

​		总结从z^[l]到a^[l]的计算过程，从计算幂再得出临时变量，再做归一化。我们可以把整个过程总结为一个softmax激活函数g，这个激活函数不同之处在于，这个函数g需要输入一个4\*1的向量也会输出一个4\*1的向量，尤以前我们的激活通常是接收单行输入如何输出一个实数的输出，softmax的不同之处在就是，由于它需要把输出归一化以及输入输出都是向量。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Softmax_regression3.PNG)

​		这些图展示了softmax分类器在没有隐层的时候可以做什么，使用更深的网络就可以发现更复杂的非线性决策平面来区分不同的类别。

## 2. Training a softmax classifier

​		softmax对应的名字是hardmax，hardmax将矢量z映射到另一个向量，即hardmax函数遍历z中的元素，将z中最大的元素对应的位置置1，其余置0，与之相对的softmax中z到这些概率值的映射要平和些。

​		softmax激活函数将logistic激活函数从二分类推广到多分类。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Training_a_softmax_classifier1.PNG)

​		损失函数的功能是查看训练集的真实分类值，并令其对应的概率值尽可能大。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Training_a_softmax_classifier2.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/Training_a_softmax_classifier3.PNG)

# Programming Frameworks

## 1. Deep Learning frameworks

## 2. TensorFlow

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/TensorFlow1.PNG)

```python
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

w = tf.Variable(0,dtype=tf.float32)
cost = tf.add(tf.add(w**2,tf.multiply(-10.,w)),25)
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
session.run(train)
print(session.run(w))
for i in range(1000):
    session.run(train)

print(session.run(w))
```

结果：

0.099999994
4.9999886

​		w是尝试优化的值，tensorflow会自动知道如何根据add和multiply等方法分别求导，这就是为什么你只需定义前向传播函数，它就会知道如何计算反向传播函数或是说梯度，因为它内置了加法、乘法以及平方等方法的梯度计算方法。

**如何将训练数据导入一个tensorflow程序？**

```python
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


coefficients = np.array([[1.], [-10.], [25.]])
w = tf.Variable(0,dtype=tf.float32)
x = tf.placeholder(tf.float32,[3,1])
# cost = tf.add(tf.add(w**2,tf.multiply(-10.,w)),25)
# cost = w**2 - 10*w + 25
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0] 
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
session.run(train,feed_dict={x:coefficients})
print(session.run(w))
for i in range(1000):
    session.run(train,feed_dict={x:coefficients})

print(session.run(w))
```

结果：

0.099999994
4.9999886

​		x控制着二项式的系数，tensorflow中的placeholder是你之后会赋值的变量，这是把训练数据导入代价函数很方便的方法。

​		tensorflow之所以强大是因为你只需指定如何计算代价函数。

​		tensorflow程序的核心是计算代价函数，之后tensorflow会自动求导，并计算出如何最小化代价函数。

​		这段代码实际上做的是，让tensorflow构建一个计算图，计算图执行如图上方的操作，tensorflow的好处是，正如向上面的图一样用前向传播来计算代价函数，tensorflow已经内置了所有必须的后向传播方程，这样的编程框架有助于提高你的效率。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week3/images/TensorFlow2.PNG)

