
- [Activation](#activation)
  - [Sigmoid](#sigmoid)
  - [Tanh](#tanh)
  - [Relu(Rectified Linear Unit)](#relurectified-linear-unit)
  - [Leak Relu](#leak-relu)
  - [SELU](#selu)
  - [Gelu (Gaussian Linear Error Units)](#gelu-gaussian-linear-error-units)


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

