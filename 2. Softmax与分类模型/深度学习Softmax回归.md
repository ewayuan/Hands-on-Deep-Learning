# 深度学习Softmax回归:

Softmax回归模型，该模型是logistic回归模型在多分类问题上的推广，在多分类问题中，类标签 $y$ 可以取两个以上的值。Softmax回归模型对于诸如MNIST手写数字分类等问题是很有用的，该问题的目的是辨识10个不同的单个数字。Softmax回归是有监督的，不过后面也会介绍它与深度学习/无监督学习方法的结合。

回想一下在 logistic 回归中，我们的训练集由 $m$ 个已标记的样本构成：$\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}$ ，其中输入特征$x^{(i)} \in \Re^{n+1}$。（我们对符号的约定如下：特征向量 $x$ 的维度为 $n+1$，其中 $x_0 = 1$ 对应截距项 。） 由于 logistic 回归是针对二分类问题的，因此类标记 $y^{(i)} \in \{0,1\}$。假设函数(hypothesis function) 如下：

$$\begin{align}h_\theta(x) = \frac{1}{1+\exp(-\theta^Tx)},\end{align}$$

我们将训练模型参数 $\textstyle \theta$，使其能够最小化代价函数：

$$\begin{align}J(\theta) = -\frac{1}{m} \left[ \sum_{i=1}^m y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) \right]\end{align}$$

在 softmax回归中，我们解决的是多分类问题（相对于 logistic 回归解决的二分类问题），类标 $y$ 可以取 $k$ 个不同的值（而不是 2 个）。因此，对于训练集 $\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}$，我们有 $y^{(i)} \in \{1, 2, \ldots, k\}$。（注意此处的类别下标从 1 开始，而不是 0）。例如，在 MNIST 数字识别任务中，我们有 $k=10$ 个不同的类别。
对于给定的测试输入 $x$，我们想用假设函数针对每一个类别 $j$ 估算出概率值 $p(y=j|x)$。也就是说，我们想估计 $x$ 的每一种分类结果出现的概率。因此，我们的假设函数将要输出一个 $k$ 维的向量（向量元素的和为1）来表示这 $k$ 个估计的概率值。 具体地说，我们的假设函数 $h_{\theta}(x)$ 形式如下：

$$\begin{align}h_\theta(x^{(i)}) =\begin{bmatrix}p(y^{(i)} = 1 | x^{(i)}; \theta) \\p(y^{(i)} = 2 | x^{(i)}; \theta) \\\vdots \\p(y^{(i)} = k | x^{(i)}; \theta)\end{bmatrix}=\frac{1}{ \sum_{j=1}^{k}{e^{ \theta_j^T x^{(i)} }} }\begin{bmatrix}e^{ \theta_1^T x^{(i)} } \\e^{ \theta_2^T x^{(i)} } \\\vdots \\e^{ \theta_k^T x^{(i)} } \\\end{bmatrix}\end{align}$$

其中 $\theta_1, \theta_2, \ldots, \theta_k \in \Re^{n+1}$ 是模型的参数。请注意 $\frac{1}{ \sum_{j=1}^{k}{e^{ \theta_j^T x^{(i)} }} }$ 这一项对概率分布进行归一化，使得所有概率之和为 1 。

为了方便起见，我们同样使用符号 $\theta$ 来表示全部的模型参数。在实现Softmax回归时，将 $\theta 用一个 \textstyle k \times(n+1)$ 的矩阵来表示会很方便，该矩阵是将 $\theta_1, \theta_2, \ldots, \theta_k$ 按行罗列起来得到的，如下所示：

$$\theta = \begin{bmatrix}\mbox{---} \theta_1^T \mbox{---} \\\mbox{---} \theta_2^T \mbox{---} \\\vdots \\\mbox{---} \theta_k^T \mbox{---} \\\end{bmatrix}$$

## 代价函数

现在我们来介绍softmax回归算法的代价函数。在下面的公式中，$1\lbrace{\cdot\rbrace}$，是示性函数，其取值规则为：$1\lbrace{值为真的表达式\rbrace}=1$，而$1\lbrace{值为假的表达式\rbrace}=0$。举例来说，表达式 $1\lbrace{2+2=4\rbrace}$的值为1 ，$1\lbrace{1+1=5\rbrace}$的值为0。我们的代价函数为:

$$\begin{align}J(\theta) = - \frac{1}{m} \left[ \sum_{i=1}^{m} \sum_{j=1}^{k}  1\left\{y^{(i)} = j\right\} \log \frac{e^{\theta_j^T x^{(i)}}}{\sum_{l=1}^k e^{ \theta_l^T x^{(i)} }}\right]\end{align}$$

值得注意的是，上述公式是logistic回归代价函数的推广。logistic回归代价函数可以改为：

$$\begin{align}J(\theta) &= -\frac{1}{m} \left[ \sum_{i=1}^m   (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) + y^{(i)} \log h_\theta(x^{(i)}) \right] \\&= - \frac{1}{m} \left[ \sum_{i=1}^{m} \sum_{j=0}^{1} 1\left\{y^{(i)} = j\right\} \log p(y^{(i)} = j | x^{(i)} ; \theta) \right]\end{align}$$

可以看到，Softmax代价函数与logistic 代价函数在形式上非常类似，只是在Softmax损失函数中对类标记的 $k$ 个可能值进行了累加。注意在Softmax回归中将 $x$ 分类为类别 $j$ 的概率为：

$$p(y^{(i)} = j | x^{(i)} ; \theta) = \frac{e^{\theta_j^T x^{(i)}}}{\sum_{l=1}^k e^{ \theta_l^T x^{(i)}} }.$$

对于  $J(\theta)$ 的最小化问题，目前还没有闭式解法。因此，我们使用迭代的优化算法（例如梯度下降法，或 L-BFGS）。经过求导，我们得到梯度公式如下：

$$\begin{align}\nabla_{\theta_j} J(\theta) = - \frac{1}{m} \sum_{i=1}^{m}{ \left[ x^{(i)} \left( 1\{ y^{(i)} = j\}  - p(y^{(i)} = j | x^{(i)}; \theta) \right) \right]  }\end{align}$$

让我们来回顾一下符号 "$\nabla_{\theta_j}$" 的含义。$\nabla_{\theta_j} J(\theta)$ 本身是一个向量，它的第 $l$ 个元素 $\frac{\partial J(\theta)}{\partial \theta_{jl}}$ 是 $J(\theta)$对 $\theta_j$ 的第 $l$ 个分量的偏导数。
有了上面的偏导数公式以后，我们就可以将它代入到梯度下降法等算法中，来最小化 $J(\theta)$。 例如，在梯度下降法的标准实现中，每一次迭代需要进行如下更新: $\theta_j := \theta_j - \alpha \nabla_{\theta_j} J(\theta)(\textstyle j=1,\ldots,k）$。
当实现 softmax 回归算法时， 我们通常会使用上述代价函数的一个改进版本。具体来说，就是和权重衰减(weight decay)一起使用。我们接下来介绍使用它的动机和细节。