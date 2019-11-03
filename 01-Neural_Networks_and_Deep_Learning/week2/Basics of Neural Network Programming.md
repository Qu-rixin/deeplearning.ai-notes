# Basics of Neural Network Programming

深度学习一般指的是训练神经网络；

数据分为结构化数据、非结构化数据（图片、音频、文本）；

图片的特征是像素、文本的特征是独立的单词；

RNN（循环神经网络）适用于时序数据，CNN适用于图片；

## 1. Binary Classification

​		在神经网络中特征向量在矩阵中按列排列更容易处理，在python中X.shape=(n(x), m)表示矩阵的形状，X是特征矩阵，label向量Y同样按列排列，在python中同样用命令Y.shape输出(1, m)表示矩阵的形状。

在Andrew Ng课程中M代表训练样本。

## 2. Logistic Regression

yhat=P(y=1|x)告诉你这是一张猫图的概率。

给定x是n_x维的特征向量，逻辑回归的参数w也是n_x维的向量，b是一个实数。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/logistic%20regression.PNG?raw=true)

## 3. Logistic Regression cost function

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/logistic%20regression%20cost%20function.PNG?raw=true)

​		在逻辑回归时一般不用第一种损失函数（平方误差），因为优化问题将会变成非凸问题（优化问题会产生多个局部最优解，梯度下降法也就无法找到全局最优解）。

​		所以在逻辑回归问题中使用第二种损失函数（可以产生一个凸象最优问题，使优化变得容易）。

**损失函数应尽可能小**

损失函数被单一的优化示例所定义，它检测单一优化示例的运行情况。

代价函数检测优化组的整体运行情况，找到参数w和b。

## 4. Gradient Descent

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Gradient%20Descent.PNG?raw=true)

​		由于损失函数，w和b可以任意初始化，但通常初始化为0，在一次迭代中，梯度下降法从初始点开始，然后朝最陡的下坡方向走一步，最后收敛到全局最优或者接近全局最优值。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Gradient%20Descent2.PNG?raw=true)

α是学习率：学习率可以控制我们在每一次迭代或者梯度下降法中步长大小。

导数是函数在一点的斜率，w通过w自身减去学习率乘导数来更新。

无论你初始化的位置是在左边还是右边，梯度下降法会朝着全局最小值方向移动。

## 5. Derivatives

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/derivatives.PNG?raw=true)

## 6. more Derivatives examples

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/More%20Derivatives%20examples.PNG?raw=true)

## 7. compute Graph

正向传播

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/compute%20Graph.PNG?raw=true)

## 8. Derivatives with a Computation Graph

反向传播

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Derivatives%20with%20a%20Computation%20Graph.PNG?raw=true)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Derivatives%20with%20a%20Computation%20Graph2.PNG?raw=true)

## 9. Logistic Regression Gradient descent

在逻辑回归中 我们要做的就是修改参数w和b，来减少损失函数

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Logistic%20Regression%20Gradient%20descent.PNG?raw=true)

通过反向传播计算导数，最终用计算到的dw1、dw2和db来更新，进行梯度下降。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Logistic%20Regression%20Gradient%20descent2.PNG?raw=true)

## 10. Gradient Descent on m examples



![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Gradient%20Descent%20on%20m%20examples2.PNG?raw=true)

缺点：这样实现需要两个循环，第一个for循环（上面的绿线），第二个for循环（对每一个特征），当你有更多特征时会使算法效率降低，矢量化可以消去for循环提高效率。

## 11. Vectorization

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Vectorization.PNG?raw=true)

[python编辑器](https://hub.gke.mybinder.org/user/ipython-ipython-in-depth-fpfigfi9/notebooks/binder/Untitled.ipynb?kernel_name=python3)

```python
import time
a = np.random.rand(1000000)
b = np.random.rand(1000000)
tic = time.time()
c = np.dot(a,b)
toc = time.time()
print(c)
print("Vectorization version:"+str(1000*(toc-tic))+"ms")

c = 0
tic = time.time()
for i in range(1000000):
    c+=a[i]*b[i]
toc = time.time()
print(c)
print("for loop:"+str(1000*(toc-tic))+"ms")
```

结果：

```
250030.5837471652
Vectorization version:0.6399154663085938ms
250030.58374716912
for loop:649.7213840484619ms
```

## 12. More vectorization examples

在编写神经网络或逻辑回归时都要尽可能避免显示的使用显示的for循环。当无法避免使用for时，如果可以使用内置函数（built-in function）通常会比直接使用for更快。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/More%20vectorization%20examples.PNG?raw=true)

## 13. Broadcasting in Python

广播是一种能使python代码运行的更快的技术。

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Broadcasting%20in%20Python.PNG?raw=true)

使用广播技术，不适用for来计算各个食物中碳水、蛋白质、脂肪的热量百分比：

```python
import numpy as np

A = np.array([[56.0, 0.0, 4.4, 68.0],
             [1.2, 104.0, 52.0, 8.0],
             [1.8, 135.0, 99.0, 0.9]])
print(A)

cal = A.sum(axis=0)#沿垂直方向求和
print(cal)

percentage = 100*A/cal.reshape(1,4)
print(percentage)
```

广播原理：

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Broadcasting%20in%20Python2.PNG?raw=true)

python自动扩展矩阵。

python广播的通用规则：

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Broadcasting%20in%20Python3.PNG?raw=true)

## 14. A note on python/numpy vectors

python的一些技巧：

```python
import numpy as np

a = np.random.randn(5)
print(a)#a为秩为1的数组
```

```
[-0.97225365  1.80901505 -1.30144599 -0.9095583   1.08054655]
```

```python
print(a.shape)
```

```
(5,)
```

```python
print(a.T)
```

```
[-0.97225365  1.80901505 -1.30144599 -0.9095583   1.08054655]
```

```python
print(np.dot(a,a.T))
```

```
7.906451431001427
```

使用时应该如下表示一个向量：

```python
a = np.random.randn(5,1)
print(a)
```

```
[[ 1.22576571]
 [ 0.12748487]
 [-0.47530115]
 [ 1.37419341]
 [-0.23316721]]
```

```python
print(a.T)
```

```
[[ 1.22576571  0.12748487 -0.47530115  1.37419341 -0.23316721]]
```

```python
print(np.dot(a,a.T))
```

```
[[ 1.50250158  0.15626658 -0.58260785  1.68443917 -0.28580837]
 [ 0.15626658  0.01625239 -0.06059371  0.17518887 -0.02972529]
 [-0.58260785 -0.06059371  0.22591118 -0.65315571  0.11082464]
 [ 1.68443917  0.17518887 -0.65315571  1.88840753 -0.32041684]
 [-0.28580837 -0.02972529  0.11082464 -0.32041684  0.05436695]]
```

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/A%20note%20on%20python%20numpy%20vectors.PNG?raw=true)

## 15. Quick tour of jupyter/ipython notebooks

## 16. Explanation of logistic regression cost function（Optional）

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Explanation%20of%20logistic%20regression%20cost%20function（Optional）.PNG?raw=true)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Explanation%20of%20logistic%20regression%20cost%20function（Optional）2.PNG?raw=true)

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Explanation%20of%20logistic%20regression%20cost%20function（Optional）3.PNG?raw=true)

最大似然估计，选择使式子最大化的参数：

![](https://github.com/Qu-rixin/deeplearning.ai-notes/blob/master/01-Neural_Networks_and_Deep_Learning/week2/images/Explanation%20of%20logistic%20regression%20cost%20function（Optional）4.PNG?raw=true)

通过最小化代价函数J(w,b)实际上对逻辑回归模型进行了最大似然估计，这基于训练样本是独立同分布。

