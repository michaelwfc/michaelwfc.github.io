- [Optimizerv](#optimizerv)
  - [SGD(Stochastic Gradient Descent)](#sgdstochastic-gradient-descent)
  - [Momentum](#momentum)
  - [RMSProp(Root Mean Squared Prop)](#rmsproproot-mean-squared-prop)
  - [Adam(Adptive Momentum)](#adamadptive-momentum)
  - [AdamW(Adam with Weight decay)](#adamwadam-with-weight-decay)


# Optimizerv

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
