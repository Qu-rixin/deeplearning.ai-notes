# Error Analysis

## 1. Carrying out error analysis

​		如果你希望让学习算法能够胜任人类能做的任务，但你的学习算法还没有达到人类的表现，那么人工检查一下你的算法犯的错误，也许可以让你了解接下来应该做什么，这个过程叫错误分析。

​		假设你的猫分类器的准确率90%，其中将狗错误分类成猫，你是不是应该去开始做一个项目专门处理狗？这个错误分析流程可以让你很快知道这个方向是否值得努力。假设事实上，你的100个错误标记例子中只有5%是狗，即使你完全解决了狗的问题，你也只能修正这100个错误中的5个，可以说明这样花时间是不值的。在机器学习中，性能上限意味着最好能到哪里完全解决狗的问题可以对你有多少帮助。现在，假设我们观察一下这100个错误标记的开发集例子，你发现实际有50张图都是狗，现在花时间去解决狗的问题可能效果就很好，这种情况下，如果你真的解决了狗的问题，那你的错误率可能就从10%下降到5%。错误分析可能节省大量时间，可能迅速决定什么是最重要的或最重要的方向。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Carrying_out_error_analysis1.PNG)

​		有时你在做错误分析时，也可以同时并行评估几个想法。要进行错误分析来评估图中三个想法，通常要做的是建立这样一个表格如图，所以在错误分析的过程中，你就看看算法识别错误的开发集例子，将错误信息记录在表格中。这个分析步骤的结果可以给出一个估计，是否值得去处理每个不同的错误类型，可以真正帮助你选出高优先级任务，并了解每种手段对性能有多大提升空间。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Carrying_out_error_analysis2.PNG)

## 2. Cleaning up incorrectly labeled data

​		监督学习问题的数据由输入X和输出标签Y构成，如果你观察一下数据，并发现有些输出标签Y是错的，是否值得花时间去修正这些标签呢？

​		在猫分类问题中，如果你发现你的数据有一些标记错误的例子，你该怎么办？首先参考训练集，事实证明深度学习算法对于训练集中的随机错误是相当健壮的，只要你标记出错的例子离随机错误不太远，如果错误足够随机，那么放着这些错误不管可能也没问题。深度学习对随机误差很健壮，对系统性错误就没那么健壮了。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Cleaning_up_incorrectly_labeled_data1.PNG)

​		那么如果是开发集和测试集中这些标记出错的例子呢？如果你担心开发集或测试集上标记出错的例子带来的影响，一般建议在错误分析时，添加一个额外的列，这样你也可以统计标签Y错误的例子数，所以比如说，也许你统计一下对100个标记出错的例子的影响，所以你会找到100个例子，其中你的分类器的输出和开发集的标签不一致，可能是标签错了，而不是分类器错了。

​		是否值得修正这6%标记出错的例子？建议是，如果这些标记错误严重影响了你在开发集上评估算法的能力，那么就应该去花时间修正错误的标签，但是，如果它们没有严重影响到，你用开发集评估成本偏差的能力，那可能就不应该花宝贵的时间去处理。

​		设立开发集的目的是你希望它来从两个分类器A和分类器B中选择一个。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Cleaning_up_incorrectly_labeled_data2.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Cleaning_up_incorrectly_labeled_data3.PNG)

​		首先，不管用什么修正手段都要同时作用到开发集和测试集上，之前讨论过开发集和测试集必须来自相同分布，开发集确立了你的目标，当你击中目标后，你希望算法能够推广到测试集上，这样你的团队能高效的在来自同一分布的开发集和测试集上迭代，如果你打算修正开发集上的部分数据，那么最好对测试集做同样的修正以确保它们继续来自同一分布。

## 3. Build your first system quickly，then iterate

​		一般来说，对几乎所有的机器学习程序，可能会有50个不同的方向可以前进，并且每个方向都是相对合理的，可以改善你的系统，但挑战在于，你如何选择一个方向集中精力处理。

​		初始系统的意义在于，有一个学习过的系统，让你确定偏差方差范围，就可以知道下一步优先做什么，让你能够进行错误分析。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Build_your_first_system_quickly_then_iterate1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Build_your_first_system_quickly_then_iterate2.PNG)

# Mismatched training and dev/test data

## 1. Training and testing on different distributions

​		在深度学习时代，越来越多的团队都用来自和开发集和测试集分布不同的数据来训练，这里有一些微妙并且不错的想法来处理训练集和测试集存在差异的情况。

​		现在你有一个相对小的数据集只有10000个样本来自那个分布，而你还有一个大得多的数据集来自另一个分布，图片的外观和你真正想要处理的并不一样，但你又不想直接用这10000张图片，因为这样你的训练集就太小了，使用这20万张图片似乎有帮助，但是麻烦在于这20万张图片并不完全来自你想要的分布。

​		你的做法有：其一（不建议），你可以将两组数据合并在一起，这样就有21万张图片，你可以把这些图片随机分配到训练集、开发集和测试集，这样做的好处在于你的训练集、开发集和测试集都来自同一分布，这样更好管理，但坏处在于，就是如果你观察开发集，很多图片都来自于网页下载的图片，那并不是你真正关系的数据，你真正要处理的是来自手机上的图片，记住你设立开发集的目的是瞄准你的目标，你的大部分精力都用在优化来自网页下载的图片。

​		其二，训练集可以是来自网页下载的200000张图片，然后如果需要的话，再加上5000张来自手机上传的图片，然后对于开发集和测试集都是手机图片，优点在于你的开发集为手机图片，这才是你关心的图片分布，缺点在于你的训练集分布和开发集、测试集分布并不一样，但事实证明，这样把数据分成训练、开发和测试集在长期能给你带来更好的系统性能。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Training_and_testing_on_different_distributions1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Training_and_testing_on_different_distributions2.PNG)

## 2. Bias and Variance with mismatched data distributions

​		如图，如果你的开发集来自和训练集一样的分布，你可能会说这里存在很大的方差问题，你的算法不能很好的从训练集出发泛化，它处理训练集很好，但处理开发集就突然间效果很差了，但如果你的训练数据和开发数据来自不同分布，你就不能放心这个结论了，特别是，也许算法在开发集上做得不错，可能因为训练集都是高分辨率图片，但开发集要难以识别得多，所以这个分析的问题在于，当你看训练数据再看开发错误，有两件事变了，首先算法只见过训练集数据，没见过开发集数据，第二，开发集数据来自不同分布，而且因为你同时改变了两件事，很难确定这增加的9%错误率有多少因为算法没看到开发集中的数据导致的，有多少因为开发集数据就是不一样。

​		为了分辨清楚两个因素的影响，定义一组新的数据是有意义的，我们称之为训练-开发集，这是一个新的数据子集，我们应该从训练集的分布里挖出来，但你不会用来训练你的网络。

​		如图，我们要做的就是随机打散训练集，然后分出一部分训练集作为训练-开发集，就像开发集和测试集来自同一分布，训练集和训练-开发集也来自同一分布，但不同的地方，现在你只在训练集训练你的神经网络，你不会让神经网络在训练-开发集上跑反向传播，为了进行错误分析，你要做的就是看看分类器在训练集上的错误、训练-开发集上的错误，还有开发集上的错误。

​		从图中可知，当你从训练数据变到训练-开发集数据时，错误率上升了很多，而训练数据和训练-开发数据的差异在于，你的神经网络能看到第一部分数据并直接在上面做了训练，但没有在训练-开发集上直接训练，这就告诉你，算法存在方差问题，因为训练-开发集的错误率是在和训练集来自同一分布的数据中测得的，所以你知道，尽管你的神经网络在训练集中表现良好，但无法泛化到来自相同分布的训练-开发集里，它无法很好地泛化推广到来自同一分布，但以前没见过的数据。

​		在图中第二个例子中，存在数据不匹配问题，因为你的学习算法没有直接在训练-开发集或者开发集上训练过，但是这两个数据集来自不同分布，不管算法在学习什么，它在训练-开发集上做得很好，但在开发集上不好，所以总之你的算法擅长处理和你关心的数据不同的分布，我们称之为**数据不匹配问题**。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Bias_and_Variance_with_mismatched_data_distributions1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Bias_and_Variance_with_mismatched_data_distributions2.PNG)

​		我们要看的关键数据有人类水平错误率、训练集错误率、训练-开发集错误率、开发集错误率，然后根据这些错误率之间的差值你可以知道可避免偏差、方差、数据不匹配问题各自有多大。你希望你的算法至少要在训练集上表现接近人类而这表明了方差的大小。右图表明，训练集数据比开发集数据要难识别的多。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Bias_and_Variance_with_mismatched_data_distributions3.PNG)

## 3. Addressing data mismatch

 ![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Addressing_data_mismatch1.PNG)

​		如果你的目标是让训练数据更接近你的开发集，其中一种技术是人工合成数据。

​		通过人工合成可以快速制造更多的训练数据，人工数据合成有一个潜在的问题，比如说你在安静的背景里录得10000小时音频数据，然后你只录制了一小时车辆背景噪声，那么你可以将汽车噪音回放10000次并叠加到音频数据中，如果你这么做，人听起来没什么问题，但有一个风险，有可能你的学习算法对这1小时汽车噪声过拟合，特别是，如果这组汽车里录的音频可能是你可以想象到所有汽车噪音背景的集合，如果你只录了一小时汽车噪音，那么你可能只模拟了全部数据空间的一小部分，你可能只从汽车噪音的很小的子集来合成数据，而对人耳来说，这个音频听起来没什么问题，因为一小时的车辆噪音听起来和其他任意车辆噪音是一样的，但你有可能从这整个空间很小的子集出发合成数据，神经网络最后可能对你这一小时的汽车噪音过拟合。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Addressing_data_mismatch2.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Addressing_data_mismatch3.PNG)

总而言之，如果你认为存在数据不匹配问题，建议做错误分析或者看看训练集或者看看开发集试图找出这两个数据分布有什么不同，然后看看是否有办法收集更多看起来像开发集的数据做训练，你也可以做人工合成，但避免过拟合。

# Learning from multiple tasks

## 1. Transfer learning

​		深度学习最强大的理念就是有时候神经网络可以从一个任务中习得知识，并将这些知识应用到另一个独立的任务中。也许你已经训练了一个神经网络能够识别猫这样的对象，然后使用那些知识或者部分习得的知识去帮助你阅读X射线扫描图，这就是迁移学习。

​		假设你已经训练好了一个图像识别神经网络，所以你首先用一个神经网络并在x、y对上训练，其中x是图像，y是某些对象。如果你把这个神经网络拿来，然后让它适应在不同任务中学到的知识，你可以做的是把神经网络最后的输出层拿走（删掉），还有进入到最后一层的权重删掉，然后为最后一层重新赋予随机权重，然后让它在放射诊断数据上训练。具体来说，在第一阶段训练过程中，当你进行图像识别任务训练时，你可以训练神经网络中的所有常用参数，所有权重、所有层，然后你就得到了一个能够做图像识别预测的网络，在训练了这个神经网络后，要实现迁移学习，你要做的是，把数据集换成新的x、y对，你要做的是初始化最后一层权重，现在，我们在这个新的数据集上重新训练网络，如果你的放射数据集很小，你可能只需要重新训练最后一层的权重，如果你有足够的数据，你可以重新训练神经网络中剩下的所有层，那么先进行预训练，再进行微调。

​		为什么这样做有效果？有很多低层次特征，比如边缘检测、曲线检测，从非常大的图像识别数据库中习得的这些能力可能有助于你的学习算法在放射科诊断中做得更好，算法学到了很多结构信息、图像形状信息，其中一些知识可能会很有用，所以学会了图像识别，它就可能学到足够多的信息可以了解不同图像的组成部分是怎样的，学到线条、点这些知识，这些知识可以帮助你在放射科诊断的学习更快一些或者需要更少的学习数据。

​		那么迁移学习什么时候是意义的呢？迁移学习起作用的场合是迁移来源问题你有很多数据，但迁移目标问题你没有那么多数据。所以你从图像识别训练中学到的很多知识可以迁移，并且真正帮你加强放射科诊断任务的性能，即使你的放射科数据很小。反过来的话，迁移学习就没有意义了。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Transfer_learning1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Transfer_learning2.PNG)

## 2. Multi-task learning

​		在多任务学习中，你是同时开始学习的，试图让单个神经网络同时做几件事，然后希望这里每个任务都能帮到其他所有任务。

​		无人驾驶汽车可能需要同时检测不同的物体，比如检测行人、车辆和停车标志。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Multi-task_learning1.PNG)

​		这与softmax回归主要区别在于，softmax将单个标签分配给单个样本，而这张图可以有很多不同的标签。如果你训练了一个神经网络，试图最小化这个成本函数，你要做的就是多任务学习，因为你现在做的是建立单个神经网络，观察每张图，然后解决四个问题，系统试图告诉你每张图里面有没有这四个物体，另外你也可以训练四个不同的神经网络而不是一个网络做四件事，因为神经网络一些早期特征在识别不同物体时都会用到，然后你会发现训练一个神经网络做四件事情会比训练四个完全独立的神经网络分别做四件事性能要更好，这就是多任务学习。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Multi-task_learning2.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Multi-task_learning3.PNG)

# End-to-end deep learning

## 1. What is end-to-end deep learning

​		简而言之，以前有一些数据处理系统或者学习系统，它们需要多个阶段的处理，那么端到端的深度学习就是忽略所有这些不同的阶段用单个神经网络代替它。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/What_is_end-to-end_deep_learning1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/What_is_end-to-end_deep_learning2.PNG)

​		比起一步到位，把问题分解成两个更简单的步骤，首先是弄清楚脸在哪里，第二步是看着脸，弄清楚这是谁，这种方法在整体上得到更好的表现。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/What_is_end-to-end_deep_learning3.PNG)

​		端到端的深度学习表现可以很好，也可以简化系统架构，让你不需要搭建那么多手工设计的组件，但它并不是每次都能成功。

## 2. Whether to use end-to-end learning

​		端到端深度学习的优点有，第一，如果你有足够多的x，y数据，那么不管从x到y最合适的函数映射是什么，如果你训练了一个足够大的网络，你希望网络自己能搞清楚，第二，如图。缺点有，如图。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Whether_to_use_end-to-end_learning1.PNG)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/03-Structured_Machine_Learning_Project/week2/images/Whether_to_use_end-to-end_learning2.PNG)

​		如果你想使用机器学习或者深度学习来学习某些单独的组件，那么当你应用监督学习时，你应该仔细选择要学习的x到y的映射类型，这取决于那些任务你可以收集数据，相比之下，空谈纯端到端深度学习方法是很激动人心的，但是就目前能收集到的数据而言，还有我们今天能够用神经网络学习的数据类型而言，这实际上不是最有希望的方法，其实前景不如这样复杂的多步方法。

