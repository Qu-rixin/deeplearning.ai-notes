# Introduction to ML strategy

## 1. Why ML Strategy？

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Why_ML_Strategy.PNG)

## 2. Orthogonalization

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Orthogonalization1.PNG)

​		正交化在第二个例子中指，把一个维度看作控制转向角，把另一个维度看作控制车速，那么你需要一个尽可能影响转向角的旋钮，另一个旋钮指的是刹车和油门来控制车速，但是，如果你有一个能同时影响这两者的操纵杆，那么汽车在转向和变速时会很困难，通过使用正交化，正交化是指（两个变量）呈90度角通过正交控制，能与你真正想要控制的事情保持一致，它使调整相应旋钮变得更简便，改变方向盘角度、油门和刹车。让汽车完成你的各项指令。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Orthogonalization2.PNG)

​		为保证有监督机器学习系统良好地运行，通常你需要调整系统旋钮，保证四件事准确无误。第一，通常确保至少训练组运行良好，因此训练组的运行情况需要进行一些可行性评估，对于一些应用，这个能需要与人类的一些性能进行比较，但取决于你的应用，如果训练组表现良好，你将希望这能使开发组运行良好，同时，你也希望测试组运行良好，最终，你希望能在成本函数的测试组里运行良好，因为结果会影响系统在实际情况的表现。如果你的算法不适合训练集的成本函数，你需要一个旋钮或一组可使用的旋钮以确保你可以优化算法使其适用于训练组，这些旋钮可能用来训练一个更大的网络，或者，你可能需要一个更好的优化算法相反，如果你发现算法对dev集的拟合结果不太好，那么就需要另一套独立的旋钮。

# Setting up your goal

## 1. Single number evaluation metric

​		无论你是调整超参数或者尝试不同的学习算法或者在搭建机器学习系统时尝试不同的手段，你会发现，如果你有一个单实数评估指标，你的进展会快很多，它可以快速告诉你，尝试新的手段比之前的手段好还是坏，所以当团队开始进行机器学习项目时，可以为你的问题设置一个单实数评估指标。

​		评估你的分类器的一个合理方式是观察它的查准率和查全率，简而言之，查准率的定义是在你的分类器标记为猫的例子中有多少真的是猫，所以如果分类器A有95%的查准率，这意味着你的分类器说这图有猫的时候，有95%的机会真的是猫，查全率就是，对于所有真猫的图片，你的分类器正确识别出了多少百分比，实际为猫的图片中，有多少被系统识别出来。如果分类器A查全率是90%，，这意味着对于所有图像，比如你的开发集中都是真的猫图，分类器A准确地分辨出了其中的90%。

​		事实证明，查准率和查全率之间往往需要折中，这两个指标都要顾及到，你希望得到的效果是，当你的分类器说，某个东西是猫的时候，有很大机会它真的是猫，但对于所有是猫的图片，你也希望系统能够将大部分分类为猫，所以用查准率与查全率来评估分类器，是比较合理的。但使用查准率和查全率作为评估指标的时候，有个问题，如果分类器A在查全率上表现更好，分类器B在查准率上表现更好，你就无法判断哪个分类器更好。

​		如果你尝试了很多不同的想法、很多不同的超参数，你希望能够快速实验不仅仅是两个分类器，也许是十几个分类器，快速选出“最好的”那个，这样你可以从那里出发再迭代，如果有两个评估指标，就很难去快速地选择，所以并不推荐使用两个评估指标查准率和查全率来选择一个分类器你只需找到一个新的评估指标，能够结合查准率和查全率，在机器学习文献中，结合查准率和查全率的标准方法是所谓的F1分数，你可以认为这是查准率P和查全率R的平均值，在数学上称为调和平均数，这个指标在权衡查准率和查全率时有一些优势。

​		很多机器学习开发团队就是这样，有一个定义明确的开发集，用来测量查准率和查全率，再加上这样一个单一数值评估指标（单实数评估指标），能让你快速判断分类器A或分类器B更好，它可以加速改进你的机器学习算法的迭代过程。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Single_number_evaluation_metric1.PNG)

​		假设你在开发一个猫应用，来服务四个地理大区的爱猫人士，我们假设你的两个分类器在来自四个地理大区的数据中得到了不同的错误率所以跟踪以下，你的分类器在不同市场和地理大区中的表现，应该是有用的，但是通过跟踪四个数字，很难扫一眼这些数值就快速判断算法A和算法B哪个更好，如果你测试了很多不同的分类器，那么选最优是很难的，所以在这个例子中，除了跟踪分类器在四个不同地理大区的表现，也要算平均值，假设平均值是一个合理的单实数评估指标，通过计算平均值，你就可以快速判断那个分类器的错误率最低，然后继续使用那个算法。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Single_number_evaluation_metric2.PNG)

## 2. Satisficing and optimizing metrics

​		要把你顾及到的所有事情组合成单实数评估指标有时并不容易，在那些情况里，我发现有时候设立满足和优化指标是很有用的。

​		假设你决定你很看重猫分类器的分类准确度，但除了准确度之外，我们还需要考虑运行时间，就是需要多长时间来分类一张图。你可以将准确度和运行时间组合成一个整体评估指标。你还可以选择一个分类器，能够最大限度提高准确度，但必须满足运行时间要求，所以这种情况下，我们就说准确度是一个优化指标，因为你想要准确度最大化，但运行时间，就是我们所说的满足指标只要到达一定范围即可，所以，这是一个相当合理的权衡方式。通过定义优化和满足指标就可以给你提供一个明确的方式去选择“最好的”分类器。

​		所以更一般地说，如果你要考虑N个指标，有时候选择其中一个指标作为优化指标是合理的，所以你尽可能优化那个指标，然后剩下N-1个指标都是满足指标，意味着只要它们达到一定阈值，你不在乎它超过那个门槛之后的表现。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Satisficing_and_optimizing_metrics.PNG)

## 3. Train/dev/test distributions

​		dev集也叫开发集，有时也称为保留交叉验证集。机器学习的工作流程是，用训练集训练不同的模型，然后使用开发集来评估不同的思路，然后选择一个，然后不断迭代去改善开发集的性能，直到最后得到一个令人满意的成本，然后再用测试集去评估。

​		现在假设你要开发一个猫的分类器，那么设立开发集和测试集的方法：其中一种做法，你可以选择其中4个区域来构成开发集，然后其他四个区域的数据构成测试集，事实证明，这个想法非常糟糕，因为这个例子中你的开发集和测试集来自不同的分布。现在设立你的开发集上加上一个单实数评估指标，一旦建立了这样的开发集和指标，团队就可以快速迭代，尝试不同的想法，跑实验，可以很快地使用开发集和指标去评估不同分类器，然后尝试选出最好的那个，所以机器学习团队一般都擅长使用不同方法去逼近目标然后不断迭代不断逼近靶心。

​		建议将所有数据随机洗牌，放入开发集和测试集，所以开发集和测试集都有来自八个地区的数据，并且开发集和测试集都来自同一分布，这分布就是你的所有数据混在一起。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Train_dev_test_distributions1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Train_dev_test_distributions2.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Train_dev_test_distributions3.PNG)

​		建议在设立开发集和测试集时，要选择这样的开发集和测试集，能够反映你未来会得到的数据，特别是，这里的开发集和测试集可能来自同一分布，所以不管你未来会得到什么样的数据，一旦你的算法效果不错，要尝试收集类似的数据，而且不管那些数据是什么都要随机分配到开发集和测试集上，因为这样，你才能将瞄准想要的目标，让团队迭代来逼近同一目标。

## 4. Size of dev and test sets

​		你可能听过一条经验法则，在机器学习中，把你取得的全部数据，用70/30比例分成训练集和测试集，在机器学习早期，这样分是合理的，以前的数据量小得多。但在现代机器学习中，我们习惯操作规模大得多的数据集，比如说你有一百万个训练的例子这样分才是合理的，98%作为训练集，1%开发集，1%测试集。因为深度学习算法对数据的胃口很大，我们可以看到那些有海量数据集的问题有更高比例的数据划分到训练集里。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Size_of_dev_and_test_sets1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Size_of_dev_and_test_sets2.PNG)

​		要记住测试集的目的是完成系统开发之后测试集可以帮你评估投产系统的性能，方针就是令你的测试集足够大，能够以高置信度评估系统整体性能，所以除非你需要对最终投产系统有一个很精准的指标，一般来说，测试集不需要上百万个例子。对某些应用，你也许不需要对系统性能有置信度很高的评估，也许你只需要训练集和开发集，事实上，有时在实践中，有些人只会分成训练集和测试集，他们实际上在测试集上迭代，所以这里没有测试集（不建议）。

## 5. When to change dev/test sets and metrics

​		假设你在构建一个猫分类器，试图找到很多猫的照片向你的爱猫人士用户展示，你决定使用的指标是分类错误率，所以算法A和B分别有3%和5%的错误率，所以算法A似乎做得更好，但实际试一下这些算法，算法A由于其他原因，把很多色情图片分类成了猫，所以如果你部署了算法A，那么用户就会看到更多猫图，因为它的错误率只有3%，但它同时也会给用户推送一些色情图片，这是你的公司完全不能接受的，你的用户y也完全b不能接受。

​		相比之下，算法B有5%的错误率，这样分类器就得到较少的图像，但它不会推送色情图像，所以从你们公司的角度来看，以及从用户接受的角度看，算法B实际上是个更好的算法，因为它不让任何色情图片通过。

​		这种情况下，评估指标加上开发集，它们都倾向于选择算法A，但你和你的用户更倾向于使用算法B，因为它不会将色情图片分类为猫，所以当你的评估指标无法正确衡量算法之间的优劣排序时，在这种情况下，原来的指标错误地预测算法A是更好的算法，这就发出了信号，你应该改变评估指标了，或者要改变开发集或测试集，这种情况下，你用的分类错误率指标可以写成如图形式，m是你的开发集样本数，I表示一个函数统计出里面这个表达式为真的样本数，所以这个公式就统计了分类错误的样本，这个评估指标的问题在于，它对色情图片和非色情图片一视同仁。其中一个修改评估指标的方法是，加一个权重项，我们称之为w(i)，其中如果图片x(i)不是色情图片w(i)=1，如果x(i)是色情图片，w(i)可能就是10，甚至100，这样你赋予了色情图片更大的权重，让算法将色情图片分类为猫图时，错误率这个项快速变大。如果你希望得到归一化常数，在技术上就是w(i)对所有i求和，这样错误率仍然在0/1之间，加权的细节并不重要，实际上要使用这种j加权，你必须自己过一遍开发集和测试集，在开发集和测试集中自己把色情图片标记出来，这样你才能使用这个加权函数。但粗略的结论是，如果你的评估指标，无法正确评估好算法的排名，那么就需要花时间定义一个新的评估指标，这是定义评估指标的其中一种可能方式评估指标的意义在于，准确告诉你已知两个分类器，哪一个更适合你的应用，我们不需要太注重新错误率指标是怎么定义的，关键在于，如果你对旧的错误率指标不满意，那就不要一直沿用你不满意的错误率指标，而应该尝试定义一个新的指标，能够更加符合你的偏好，定义出实际更适合的算法。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/When_to_change_dev_test_sets_and_metrics1.PNG)

​		到目前为止，我们只讨论了如何定义一个指标去评估分类器，更好的把分类器排序，能够区分出它们在识别色情图片的不同水平，这实际上是一个正交化的例子，在处理机器学习问题时，应该把它切分成独立的步骤，一步是弄清楚，如何定义一个指标来衡量你想做的事情的表现然后我们可以分开考虑如何改善系统在这个指标上的表现。

​		要把机器学习任务看成两个独立的步骤，用目标这个比喻，第一步就是设定目标，所以要定义你要瞄准的目标，这是完全独立的一步，第二步，如何精确瞄准，如何命中目标。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/When_to_change_dev_test_sets_and_metrics2.PNG)

## 6. Why human-level performance?

​		贝叶斯最优错误率，一般认为是理论上可能达到的最优错误率，永远不会被超越，当超越人类的表现时有时j进展会变慢，有两个原因：一是人类水平在很多任务中离贝叶斯最优错误率已经不远了，所以当你超越人类表现之后，也许没有太多的空间继续改善了，第二个原因是，只要你表现的比人类更差，那么实际上可以使用某些工具来提高性能，一旦你超越了人类的表现，这些工具就没那么好用了。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Why_human-level_performance1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Why_human-level_performance2.PNG)

# Comparing to human-level performance

## 1. Avoidable bias

​		你希望你的学习算法能在训练集上表现良好，但有时你实际上并不想做得太好，你得知道人类水平的表现是怎样的，可以确切告诉你，算法在训练集上的表现，到底应该有多好或者多不好。

​		举例，比如人类具有近乎完美的准确度，在这种情况下，如果您的学习算法达到8%的训练错误率，那么你也许想在训练集上得到更好的结果，所以事实上，你的算法在训练集上的表现和人类水平的表现有很大差距的话，说明你的算法对训练集的拟合并不好，所以从减少偏差和方差的工具这个角度看，这种情况下，应该把重点放在减少偏差上，你需要做的是，比如训练更大的神经网络，或者跑久一点梯度下降。

​		但现在我们看看同样的训练错误率和开发错误率，人类的准确度为7.5%，你可以知道你的系统表现得还不错，在第二个例子，你可能希望减少学习算法的方差，也许你可以试试正则化或者收集更多数据，让你的开发错误率更接近你的训练错误率。这个猫分类器，用人类水平的错误率估计或代替贝叶斯错误率或者贝叶斯最优错误率，对计算机视觉任务而言，这样替代相当合理，因为人类实际非常擅长计算机视觉任务，所以人类能做到的水平和贝叶斯错误率相差不远，根据定义，人类水平错误率比贝叶斯错误率高一点，因为贝叶斯错误率是理论上限，但人类水平错误率和贝叶斯错误率差不太远。

​		定义术语：贝叶斯错误率和训练错误率之间的差值称为可避免偏差，可避免偏差说明了，有一些别的偏差或错误率有个无法超越的最低水平。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Avoidable_bias.PNG)

## 2. Understanding human-level performance

​		如何界定人类水平错误率？思考人类水平错误率最有用的方式之一是把它当作贝叶斯错误率的替代或估计，在定义人类水平错误率时要弄清楚你的目标所在，如果要表明你可以超越单个人类，那么就有理由，在某些场合部署你的系统，也许这个定义是合适的，但如果你的目标是替代贝叶斯错误率，那么(d)中的定义才是合适的。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Understanding_human-level_performance1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Understanding_human-level_performance2.PNG)

​		总结：估计人类水平错误率，使用人类水平错误率来估计贝叶斯错误率，贝叶斯错误率估计值到训练错误率的差距可知道可避免偏差，而训练错误率和开发错误率之间的差距告诉你方差的问题有多大，你的算法能否将训练集泛化推广到开发集，和之前的课程中的区别是，以前比较的是训练错误率和0%，直接用这个值估计偏差。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Understanding_human-level_performance3.PNG)

​		对人类水平的估计可以让你做出的贝叶斯错误率的估计，这样可以让你更快地作出决定是否应该专注于减少算法的偏差或者减少算法的方差，这个决策技巧通常很有效，直到你的系统性能开始超越人类，那么你对贝叶斯错误率的估计就不再准确了，但这些技巧还是可以帮你做出明确的决定。

## 3. Surpassing human-level performance

​		机器学习的进展会在接近或者超越人类水平的时候变得越来越慢，一旦你超过这个0.5%的门槛，要进一步优化你的机器学习问题就没有明确的选项和前进的方向了，这并不意味着你不能取得进展，你仍然可以取得重大进展，但现有的一些工具就没那么好用了。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Surpassing_human-level_performance1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Surpassing_human-level_performance2.PNG)

​		这四个例子都是从结构化数据中学习得来的，人类在自然感知任务中往往表现得非常好，所以有可能在自然感知任务的表现要超越人类要更难一些。现在计算机可以检索大量数据，它可以比人类更敏锐地识别出数据中的统计规律。

## 4. Improving your model performance

​		要想让一个监督学习算法达到实用，基本上希望你可以完成两件事情，首先，你的算法对训练集的拟合很好，这可以看成是你可以做到可避免偏差很低，第二件事是你可以在训练集中做得好，然后推广到开发集和测试集也很好，这也就是说方差不是太大。在正交化的思想中，应该分别进行上面两步优化。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Improving_your_model_performance1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week1/images/Improving_your_model_performance2.PNG)

