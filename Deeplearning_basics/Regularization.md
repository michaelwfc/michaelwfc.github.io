- [Regularization](#regularization)
  - [L2 Regularization \& Weight decay](#l2-regularization--weight-decay)
    - [Weight decay](#weight-decay)
  - [L1 Regularization](#l1-regularization)
  - [Dropout (随机失活)](#dropout-随机失活)


# Regularization 

## L2 Regularization & Weight decay

[L2正则=Weight Decay？](https://zhuanlan.zhihu.com/p/40814046)

The idea behind L2 regularization or weight decay is that networks with smaller weights (all other things being equal) are observed to overfit less and generalize better. 


L2正则和Weight Decay并不等价：
- 在标准SGD的情况下，通过对衰减系数做变换，可以将L2正则和Weight Decay看做一样。
- Adam这种自适应学习率算法中两者并不等价。


### Weight decay

$W_t =(1- \lambda) W_t - \alpha \frac{\partial L}{\partial W_t}$

$(1- \lambda) W_t$:  that exponentially decays the weights x and thus forces the network to learn smaller weights.

## L1 Regularization

## Dropout (随机失活)
dropout是通过遍历神经网络每一层的节点，然后通过对该层的神经网络设置一个keep_prob(节点保留概率)，即该层的节点有keep_prob的概率被保留，keep_prob的取值范围在0到1之间。
Why does drop-out work？
  
通过设置神经网络该层节点的保留概率，使得神经网络不会去偏向于某一个节点(因为该节点有可能被删除)，dropout最终会产生收缩权重的平方范数的效果，来压缩权重，从而使得每一个节点的权重不会过大，有点类似于L2正则化，来减轻神经网络的过拟合。

dropout的工作原理主要可以分为3步：
1、遍历神经网络的每一层节点，设置节点保留概率keep_prob，假设keep_prob=0.5
2、删除神经网络的节点，并删除网络与移除节点之间的连接
3、输入样本，使用简化后的网络进行训练，每次输入样本的时候都要重复这三步。


Dropout可以被认为是集成了大量深层神经网络的 Bagging 集成近似方法

而 Dropout 神经元被丢弃的概率为 1 − keep_prob，且关闭的神经元不参与前向传播计算与参数更新。每当我们关闭一些神经元，我们实际上修改了原模型的结构，那么每次迭代都训练一个不同的架构，参数更新也更加关注激活的神经元。
Dropout类似于bagging ensemble减少variance。也就是投通过投票来减少可变性。
Bagging 是通过结合多个模型降低泛化误差的技术，主要的做法是分别训练几个不同的模型，然后让所有模型表决测试样例的输出。

通常我们在全连接层部分使用dropout，在卷积层则不使用。但「dropout」并不适合所有的情况，不要无脑上Dropout。
Dropout一般适合于全连接层部分，而卷积层由于其参数并不是很多，所以不需要dropout，加上的话对模型的泛化能力并没有太大的影响。

预测注意事项： * 1/p
dropout在测试阶段不需要使用，因为如果在测试阶段使用dropout可能会导致预测值产生随机变化(因为dropout使节点随机失活)。而且，在训练阶段已经将权重参数除以keep_prob来保证输出的期望值不变，所以在测试阶段没必要再使用dropout。

keep_prob：一般设为keep_prob = 0.5
神经网络的不同层在使用dropout的时候，keep_prob可以不同。
对于参数比较多layer，比较复杂，keep_prob可以小一些，防止过拟合。
对于结构比较简单的层,keep_prob的值可以大一些甚至为1。 
keep_prob等于1表示不使用dropout，即该层的所有节点都保留。

