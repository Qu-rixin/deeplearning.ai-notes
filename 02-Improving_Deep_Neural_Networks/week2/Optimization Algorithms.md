# Optimization Algorithms

​		本周将学习优化算法，这可以使神经网络运行的更快。

​		机器学习的应用是一个高度依赖经验的过程，伴随着大量迭代的过程，你需要训练诸多的模型，才能找到合适的那一个，所以优化算法可以帮你快速训练模型。

## 1. Mini-batch gradient descent

​		向量化能让你有效地对所有m个例子进行计算，允许你处理整个训练集，而无需某个明确的公式。向量化能让你相对较快地处理所有m个样本，但m很大的话，处理速度仍然缓慢。

​		在对整个训练集执行梯度下降法时，你要做的是，必须处理整个训练集，然后才能进行一步梯度下降法，然后你需要再重新处理然后再进行下一步梯度下降法，整个500万个样本训练之前，先让梯度下降法处理一部分，你的算法速度会更快。

​		你可以把训练集分割为小一点的子训练集，这些子集被取名为Mini-batch，假设每一个子集中有1000个样本，那么将其中x^(1)到x^(1000)取出来，将其称为第一个子训练集，也叫做Mini-batch，然后你再取出接下来的1000个样本，从x^(1001)到x^(2000)，以此类推。

​		对y也要进行相同处理，你也要相应的拆分y的训练集。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Mini-batch_gradient_descent1.PNG)

符号：

x^(i)是第i个训练样本；

z^[l]表示神经网络中第l层的z值；

X^{t}、Y^{t}代表不同的Mini-batch；

​		batch梯度下降法指的是之前的梯度下降法，就是同时处理整个训练集，这个名字来源于，能够同时看到整个batch训练集的样本被处理。

​			Mini-batch梯度下降法指的是每次同时处理的是单个mini-batchX、Y，而不是同时处理全部的X和Y训练集。

**Mini-batch梯度下降法**：

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Mini-batch_gradient_descent2.PNG)

​		使用batch梯度下降法，一次遍历训练集只能让你做一个梯度下降，使用mini-batch梯度下降法一次遍历训练集，就是一代，能让你做5000个梯度下降。

​		当然正常来说，你想要多次遍历训练集，你还需要为另一个while循环设置另一个for循环。所以你可以一直处理遍历训练集，直到最后你能收敛到一个合适的精度。

​		几乎每一个研习深度学习的人，在训练巨大数据集时都会用到mini-batch梯度下降法。

## 2. Understanding mini-batch gradient descent

​		使用mini-batch梯度下降法，如果你做出成本函数在整个过程中的图，则并不是每次迭代都是下降的，特别是在每次迭代中，你要处理的是X^{t}、Y^{t}，如果要做出成本函数J^{t}的图，也就是每次迭代下你都在训练不同的样本集或者说训练不同的mini-batch，如下：

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Understanding_mini-batch_gradient_descent1.PNG)

​		噪声产生的原因在于每一个mini-batch所计算出的J不同。

​		你需要决定的变量之一是mini-batch的大小，m就是训练集的大小，极端情况下m=mini-batch的大小。另一种极端情况就是mini-batch的大小=1，就有了新的算法，叫做**随机梯度下降法**，每个样本都是独立的mini-batch。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Understanding_mini-batch_gradient_descent2.PNG)

​		图中显示了在两个极端下，成本函数的优化情况，第一种情况：相对噪声低一些，幅度也大一些；第二种情况：每次迭代只对一个样本进行梯度下降，大部分的时候你向着全局最小值接近，有时候你会远离全局最小值，因为那个样本恰好给你指的方向不对，因为随机梯度下降法是有很多噪声的，平均来看，它最终会靠近最小值，不过有时候也会方向错误，因为随机梯度下降法永远不会收敛，而会一直在最小值附近波动，但它并不会在到达最小值并停留在此。

​		实际上，你选的mini-batch的大小在二者之间，原因在于，mini-batch的大小为m，每个迭代需要处理大量训练样本，该算法的主要弊端在于单次迭代耗时太长。如果训练样本不大，batch梯度下降法运行地很好。相反，如果使用随机梯度下降法，如果你只要处理1个样本，这样做没有问题，通过减小学习率，噪声会被改善或有所减小，你会失去所有向量化带给你的加速，因为一次只处理一个训练样本。

​		所以实践中最好选择不大不小的mini-batch尺寸，实际上学习率达到最快，一方面，得到了大量向量化，另一方面，你不需要等待整个训练集被处理完，就可以开始进行后续工作。它比随机梯度下降法更持续地靠近最小值的方向。

## 3. Exponentially weighted averages

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Exponentially_weighted_averages1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Exponentially_weighted_averages2.PNG)

## 4. Understanding exponentially weighted averages

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Understanding_exponentially_weighted_averages1.PNG)

所有系数相加起来为1或者逼近1，称为偏差修正。

## 5. Bias correction in exponentially weighted average

开始学习阶段，修正偏差可以帮助你更好地预测温度。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Bias_correction_in_exponentially_weighted_average.PNG)

​		在机器学习中，在计算指数加权平均数的大部分时候，大家不在乎执行偏差修正，因为大部分人宁愿熬过初始时期，拿到具有偏差的估计，然后计算下去。如果你关心初始时期的偏差，在刚开始计算指数加权移动平均数的时候，偏差修正能帮助你在早期获得更好的估测。

## 6. Gradient descent with momentum

​		基本思想：计算梯度的指数加权平均数，并利用该梯度更新你的权重。

​		使用梯度下降法可能如蓝线，慢慢摆动到最小值，减慢了梯度下降法的速度，并且无法使用更大的学习率，因为结果可能偏离函数的范围。

​		在纵轴上，你希望学习的慢一点，不要摆动，在横轴上，你希望加快学习。例如，在上几个导数中，如果平均这些梯度，你会发现这些纵轴上的摆动，平均值接近于0，但是在横轴方向上，平均值仍然较大，你发现momentum梯度下降法纵轴方向的摆动变小了。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Gradient_descent_with_momentum1.PNG)

​		超参数为α和β，β控制指数加权平均数，β常用值是0.9（平均前十次迭代的梯度），不过0.9不是很棒的鲁棒数，那么关于偏差修正（除以1-β^t），实际上人们不这么做，因为十次迭代后，你的移动平均已经过了初始阶段，不再是一个具有偏差的预测。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Gradient_descent_with_momentum2.PNG)

​		最后说一点，如果你查阅了momentum梯度下降法的相关资料，通常会看到1-β删除了，实际上两者效果都不错，只是会影响到学习率的最佳值，最好用左边的公式。

## 7. RMSprop

全称root mean square prop算法。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/MSprop.PNG)

​		工作原理：在水平方向上，我们希望学习速率较快，而在垂直方向上，我们希望减少震荡，对于S_dw和S_db这两项，我们希望S_dw相对较小，而S_db相对较大，所以可以减缓垂直方向上的更新，实际上，如果你看一下导数，就会发现垂直方向上的导数比水平方向更大，所以在b方向上的斜率很大，对于这样的导数，db很大而dw很小。所以垂直方向更新量除以更大的数，这有助于减弱震荡，水平方向的更新量会除以一个较小的数。

​		使用RMSprop的效果就是使你的更新在垂直方向上震荡更小，而在水平方向可以一直走下去，另一个好处是，可以使用更大的学习率α，学习得更快，而不用在垂直方向上发散。实践上，你通常会面对非常高维的参数空间，直观理解是，在出现震荡的维度里，你会计算更大的和值，即导数平方的加权平均，最后抑制了这些出现震荡的方向。

## 8. Adam optimization algorithm

​		Adam（Adaptive Moment Estimation）优化算法本质上是将动量算法和RMSprop结合起来，Adam算法被广泛使用且已经被证明在很多不同种类的神经网络架构中都是十分有效的。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Adam_optimization_algorithm1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Adam_optimization_algorithm2.PNG)

## 9. Learning rate decay

还有一种方法使学习算法更快，那就是渐渐减小学习率。

​		当你使用适量的小样本进行mini-batch梯度下降法，也许一个批次只有64个或128个样本，当你迭代时，步长会有些浮动，它会逐步向最小值靠近，但不会完全收敛到这点，所以你的算法会在最小值周围浮动，但却永远不会真正收敛，因为你的学习率α取了固定值，且不同批次也可能产生些噪声，但如果你慢慢降低你的学习率α，那么在初始阶段，因为学习率α取值还比较大，学习速度仍然可以比较快，但随着学习率降低，步长也会渐渐变小，所以最终将围绕着离极小值点更近的区域摆动，即使继续训练下去也不会漂流远离，逐渐降低学习率α背后的思考是，在学习的初始步骤中，你可以采取大得多的步长，但随着学习开始收敛于一点时，较低的学习率可以允许你采取更小的步长。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Learning_rate_decay1.PNG)

衰减率（decay-rate）是超参数。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Learning_rate_decay2.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/Learning_rate_decay3.PNG)

学习率衰减通常通常位于尝试的事情中比较靠后的位置，设置一个固定数值的α，还要使它优化的良好对结果是有巨大影响的。

## 10. The problem of local optima

从经验上来说，对于一个高维空间的函数，如果梯度为零，则在每个方向上，可能是凸函数或者是凹函数，称为鞍点。在高维空间中，更有可能遇到鞍点而不是局部最优。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/The_problem_of_local_optima1.PNG)

真正会降低学习速率的是停滞区。停滞区是指导数长时间接近于0的一段区域。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/02-Improving_Deep_Neural_Networks/week2/images/The_problem_of_local_optima2.PNG)

