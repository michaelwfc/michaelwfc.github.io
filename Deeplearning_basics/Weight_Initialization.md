- [Layer Weight Initialization](#layer-weight-initialization)
  - [Xavier Initialization](#xavier-initialization)
  - [He Initialization](#he-initialization)
  - [svd 初始化](#svd-初始化)


# Layer Weight Initialization

原理: 通过权重初始化让网络在训练之初保持激活层的输出（输入）为zero mean unit variance分布，以减轻梯度消失和梯度爆炸。


## Xavier Initialization

scale = np.sqrt(1/ num_of_dim[l-1])

推荐给tanh, sigmoid激活函数

## He Initialization

scale = np.sqrt(2/num_of_dim[l-1])
He Initialization是推荐针对使用ReLU激活函数的神经网络使用的，不过对其他的激活函数，效果也不错。

## svd 初始化

对RNN有比较好的效果。
参考论文：[https://arxiv.org/abs/1312.6120](http://link.zhihu.com/?target=https%3A//arxiv.org/abs/1312.6120)