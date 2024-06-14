- [Optimizer](#optimizer)
  - [SGD(Stochastic Gradient Descent)](#sgdstochastic-gradient-descent)
  - [Momentum](#momentum)
  - [RMSProp(Root Mean Squared Prop)](#rmsproproot-mean-squared-prop)
  - [Adam(Adptive Momentum)](#adamadptive-momentum)
  - [AdamW(Adam with Weight decay)](#adamwadam-with-weight-decay)
- [Regularization](#regularization)
  - [L2 Regularization \& Weight decay](#l2-regularization--weight-decay)
    - [Weight decay](#weight-decay)
  - [L1 Regularization](#l1-regularization)
  - [Dropout (随机失活)](#dropout-随机失活)
  - [Normalization](#normalization)
    - [Batch-Normalization](#batch-normalization)
    - [Layer Normalization (bert中使用)](#layer-normalization-bert中使用)
- [Activation](#activation)
  - [Sigmoid](#sigmoid)
  - [Tanh](#tanh)
  - [Relu(Rectified Linear Unit)](#relurectified-linear-unit)
  - [Leak Relu](#leak-relu)
  - [SELU](#selu)
  - [Gelu (Gaussian Linear Error Units)](#gelu-gaussian-linear-error-units)
- [Loss](#loss)
  - [Classification losses](#classification-losses)
    - [Cross entropy loss](#cross-entropy-loss)
    - [Binary Cross entropy loss](#binary-cross-entropy-loss)
    - [hinge loss](#hinge-loss)
  - [Regression Losses](#regression-losses)
    - [Mean Square Error Loss](#mean-square-error-loss)
    - [Huber loss](#huber-loss)
  - [Ranking losses](#ranking-losses)
    - [Triple （Rankding）loss](#triple-rankdingloss)
    - [KL散度](#kl散度)
    - [相对熵](#相对熵)
- [Layer Weight Initialization](#layer-weight-initialization)
  - [Xavier Initialization](#xavier-initialization)
  - [He Initialization](#he-initialization)
  - [svd 初始化](#svd-初始化)



# Optimizer

## SGD(Stochastic Gradient Descent)

mini batch size:

- m(all train data): Batch gradient descent, <span style="color:red"> Too big per iteration <span>
- 1: stochastic gradient descent(SGD), <span style="color:red">lose speed up from vectorization<span>
- n:  1<n<m

update the weights by gradient $\frac{\partial L}{\partial W_t}$  
$W_t = W_t - \alpha \frac{\partial L}{\partial W_t}$


缺点:
- 下降速度慢
- 可能会在沟壑的两边持续震荡，停留在一个局部最优点。
  
基本的mini-batch SGD优化算法在深度学习取得很多不错的成绩。然而也存在一些问题需解决：

1. 选择恰当的初始学习率很困难。
2. 学习率调整策略受限于预先指定的调整规则。
3. 相同的学习率被应用于各个参数。
4. 高度非凸的误差函数的优化过程，如何避免陷入大量的局部次优解或鞍点。

## Momentum

一阶动量
Intuition： compute an exponentially weighted average of gradients

Initial the Momentum :  
$V_{dw} =0$  

On iteration t:

- iterate to compute Momentum:  
$V_{dw} = \beta V_{dw} + (1-\beta) dw$  
$\beta = (0.8,0.999)$

- update the weights by$V_{dw}$  
$W_t = W_t - \alpha V_{dw}$

## RMSProp(Root Mean Squared Prop)

Intuition：extend an exponentially weighted average of gradients to squared gradients

Initial the squared Momentum :  
$S_{dw} =0$  

On iteration t:
 
- iterate to compute squared Momentum:  
$S_{dw} = \beta S_{dw} + (1-\beta) {dw}^2$  

- update the weights by$\frac{dw}{\sqrt(S_{dw}) + \epsilon}$  
$W_t = W_t - \alpha \frac{dw}{\sqrt(S_{dw}) + \epsilon}$

 This adaptively adjusts the learning rate for each parameter and enables the usage of larger learning rates.



## Adam(Adptive Momentum)

Initial Momentum and squared Momentum :  
$V_{dw} =0$   
$S_{dw} =0$  

On iteration t:
 
- iterate to compute Momentum and  squared Momentum:  
$V_{dw} = \beta_1 V_{dw} + (1-\beta_1) dw$   
$S_{dw} = \beta_2 S_{dw} + (1-\beta)_2 {dw}^2$  

$\beta_1 = 0.9$
$\beta_2 = 0.999$

- bias correction  
$V_{dw}^{corrected} =\frac{V_{dw}}{1-\beta_1^t}$  
$S_{dw}^{corrected} =\frac{S_{dw}}{1-\beta_2^t}$

- update the weights by $  
$W_t = W_t - \alpha \frac{V_{dw}^{corrected}}{\sqrt(S_{dw}^{corrected}) + \epsilon}$




## AdamW(Adam with Weight decay)

[Why AdamW matters]https://towardsdatascience.com/why-adamw-matters-736223f31b5d）
Fixing Weight Decay Regularization in Adam“ in which they demonstrate that L2 regularization is significantly less effective for adaptive algorithms than for SGD.



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


## Normalization
参考文献：
<Layer normalization>
详解深度学习中的Normalization，BN/LN/WN https://zhuanlan.zhihu.com/p/33173246

NLP中 batch normalization与 layer normalization
https://zhuanlan.zhihu.com/p/74516930
深度学习中的Normalization模型
https://zhuanlan.zhihu.com/p/43200897


### Batch-Normalization
原理：
1.	Normalization之后更接近Gauss 分布，有利于加快训练
2.	减弱 Covariance shift 现象：
the “covariate shift” problem can be reduced by fixing the mean and the variance of the summed inputs within each layer.
使得layer之间不那么相互依赖，更加具有独立性，从而有利于训练的稳定性，具有 regularizatioin 的作用
全连接层如何使用 BatchNorm?
全连接层输出的结果： [batch_size, input_size] ->  [batch_size, output_size]
每个hiddent_unit 处有1对γ和β参数，同一个 batch 同一个 hiddent_unit 共享同一对参数。
参数量： output_size*2
卷积层如何使用BatchNorm？
卷积之后的输出的 tensor : [Batch_size, height, width, channels]

1个卷积核产生1个feature map，1个feature map有1对γ和β参数，同一batch同channel的feature map共享同一对γ和β参数，若卷积层有n个卷积核，则有n对γ和β参数。
参数量： channels * 2



### Layer Normalization (bert中使用)
Layer Normalization的基本思想
为了能够在只有当前一个训练实例的情形下，也能找到一个合理的统计范围，一个最直接的想法是：
MLP的同一隐层自己包含了若干神经元；
CNN中同一个卷积层包含k个输出通道，每个通道包含m*n个神经元，整个通道包含了k*m*n个神经元；
RNN的每个时间步的隐层也包含了若干神经元。
那么我们完全可以直接用同层隐层神经元的响应值作为集合S的范围来求均值和方差。这就是。

因为NLP领域中，句子长度不同，batch normalization不使用，LN更为合适


全连接层后的LN
对全连接层的所有输出节点进行 LN, 只需要对应1对参数
参数量：2
CNN卷积后LN:
对卷积层后 k个通道，每个通道包含 m*n feature map ，所有通道都一起进行 LN
参数量：2
RNN的隐藏层后的LN
类似全连接层

Bert 中的所在位置： embedding + self_attention + ff 层之后都接一个 LN
Embedding层后的LN:
Embedding的输出：[max_length, embedding_size]
对每个维度上的所有的token 进行layer_normalization，每个维度对应2个参数
参数量： embedding_size*2
self_attention层后的LN:
self_attention的输出 [max_length, hidden_size]
对每个维度进行layer_normalization，每个维度对应2个参数
参数量： hidden_size*2

ff 层后LN
ff的输出 [max_length, hidden_size]
对每个维度进行layer_normalization，每个维度对应2个参数
参数量： hidden_size  *2

Instance Normalization
Layer Normalization在抛开对Mini-Batch的依赖目标下，为了能够统计均值方差，很自然地把同层内所有神经元的响应值作为统计范围，那么我们能否进一步将统计范围缩小？对于CNN明显是可以的，因为同一个卷积层内每个卷积核会产生一个输出通道，而每个输出通道是一个二维平面，也包含多个激活神经元，自然可以进一步把统计范围缩小到单个卷积核对应的输出通道内部。
CNN卷积后的 IN:
针对不同的通道的 m*n的feature map，设计不同 IN，故
参数量： channels * 2

Group normalization(2018)
从上面的Layer Normalization和Instance Normalization可以看出，这是两种极端情况，Layer Normalization是将同层所有神经元作为统计范围，而Instance Normalization则是CNN中将同一卷积层中每个卷积核对应的输出通道单独作为自己的统计范围。那么，有没有介于两者之间的统计范围呢？通道分组是CNN常用的模型优化技巧，所以自然而然会想到对CNN中某一层卷积层的输出或者输入通道进行分组，在分组范围内进行统计。这就是Group Normalization的核心思想，是Facebook何凯明研究组2017年提出的改进模型。
2018年Kaiming提出了GN(Group normalization)，成为了ECCV2018最佳论文提名。





# Activation

## Sigmoid

尽量不要用sigmoid: sigmoid函数在-4到4的区间里，才有较大的梯度。之外的区间，梯度接近0，很容易造成梯度消失问题

优点：
-sigmoid函数可以将实数映射到[0,1]区间内。平滑、易于求导。

缺点：

1. 激活函数含有幂运算和除法，计算量大；
2. 反向传播时，很容易就会出现梯度消失的情况，从而无法完成深层网络的训练；
3. sigmoid的输出不是0均值的,这会导致后一层的神经元将得到上一层输出的非0均值的信号作为输入。

## Tanh

tanh激活函数是0均值的，tanh激活函数相比sigmoid函数更'陡峭'了，对于有差异的特征区分得更开了，tanh也不能避免梯度消失问题。

## Relu(Rectified Linear Unit)

RELU 是多伦多大学 Vinod Nair 与图灵奖获得者 Geoffrey Hinton 等人的研究，其研究被 ICML 2010 大会接收

优点：
1.计算量小；采用sigmoid等函数，算激活函数时（指数运算），计算量大，反向传播求误差梯度时，求导涉及除法，计算量相对大，而采用Relu激活函数，整个过程的计算量节省很多。
2.激活函数导数维持在1，可以有效缓解梯度消失和梯度爆炸问题；对于深层网络，sigmoid函数反向传播时，很容易就会出现梯度消失的情况（在sigmoid接近饱和区时，变换太缓慢，导数趋于0，这种情况会造成信息丢失，从而无法完成深层网络的训练

3.使用Relu会使部分神经元为0，这样就造成了网络的稀疏性，并且减少了参数之间的相互依赖关系，缓解了过拟合问题的发生。

缺点：输入激活函数值为负数的时候，会使得输出为0，那么这个神经元在后面的训练迭代的梯度就永远是0了（由反向传播公式推导可得），参数w得不到更新，也就是这个神经元死掉了。这种情况在你将学习率设得较大时（网络训练刚开始时）很容易发生（波浪线一不小心就拐到负数区域了，然后就拐不回来了）。

解决办法：一些对Relu的改进，如ELU、PRelu、Leaky ReLU等，给负数区域一个很小的输出，不让其置0，从某种程度上避免了使部分神经元死掉的问题。

## Leak Relu

## SELU

## Gelu (Gaussian Linear Error Units)

where  the standard Gaussian cumulative distribution function. The GELU nonlinearity weights inputs by their percentile, rather than gates inputs by their sign as in ReLUs (). Consequently the GELU can be thought of as a smoother ReLU.

BERT、RoBERTa、ALBERT 等目前业内顶尖的 NLP 模型都使用了这种激活函数。另外，在 OpenAI 声名远播的无监督预训练模型 GPT-2 中，研究人员在所有编码器模块中都使用了 GELU 激活函数。


# Loss

##	Classification losses

### Cross entropy loss 

交叉熵loss函数及原理

### Binary Cross entropy loss 

### hinge loss

Also known as max-margin objective. It’s used for training SVMs for classification. It has a similar formulation in the sense that it optimizes until a margin. That’s why this name is sometimes used for Ranking Losses.


##	Regression Losses
### Mean Square Error Loss
### Huber loss


##	Ranking losses
https://gombru.github.io/2019/04/03/ranking_loss/

the objective of Ranking Losses is to predict relative distances between inputs. This task if often called metric learning

### Triple （Rankding）loss 

### KL散度

### 相对熵






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
