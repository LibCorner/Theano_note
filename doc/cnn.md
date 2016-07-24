
## Tips and Tricks(一些小窍门)
--------------------------------

## 选择超参数
因为CNN比标准的MLP有更多的超参数，所以CNN很难训练。虽然根据经验学习速率和正则化约束的一般规则仍然使用，优化CNN时应该记住以下几点.

### 过滤器的个数

当选择的每一层的过滤器的数量，要记住，计算单个卷积滤波器的激活函数比用传统MLP昂贵得多。

假设第（l-1）层包含K<sup>l-1</sup>个特征映射(feature map)和M x N个像素，第l层有K<sup>l</sup>个m x n的过滤器。然后计算一个特征图（在所有(M-m)x(N-n)个像素上应用m x n的过滤器）需要花费(M-m)x(N-n) x m x n x K<sup>l-1</sup> 。共计算K<sup>l</sup>次。

对于标准的MLP,如果在第l层有K<sup>l<sup/>个不同的神经元，花费的时间只有K<sup>l</sup>xK<sup>l-1</sup>。因此CNN中使用的过滤器个数一般比MLP隐层神经元个数小的多，
一般由特征图的大小决定（它本身是输入image的size和filter shape的函数）

因为特征图的size随着深度而减小，接近输入层的层可以有较少的filter,而更高层次的层可以有更多的filter。实际上，为了平衡每一层的计算，通常用特征图数和像素数的乘积来大致衡量不同的层的计算。为了保存关于输入的信息，需要保持activations的总数（特征图数乘以像素数）不随着层数减少。特征图的个数直接控制capaity,因此由可用的样本和任务的复杂度决定。

### 过滤器的shape

在文献中常见的filter的shape差别很大，通常都是基于数据集的。在MNIST-sized images(28x28)最好的结果第一层通常为5x5,然而自然的image数据集(每一维常有成百的像素)通常使用更大的filter,如
12x12或15x15。

### Max Pooling Shape

一般使用2x2或者没有max-pooling。非常大的输入images可以在lower-layers有4x4的池化,但是这样就会用因子16削减信号，会使结果丢失太多信息。

## Tips

* Whitening the data (e.g. with PCA)
* Decay the learing rate in each epoch : 每个epoch都减小学习速率