- [**Normalization**](#normalization)
  - [Batch-Normalization](#batch-normalization)
    - [原理：](#原理)
    - [全连接层如何使用 BatchNorm?](#全连接层如何使用-batchnorm)
    - [卷积层如何使用BatchNorm？](#卷积层如何使用batchnorm)
    - [How Batch Normalization Works in CNNs](#how-batch-normalization-works-in-cnns)
    - [Practical Implementation in PyTorch](#practical-implementation-in-pytorch)
    - [Benefits of Batch Normalization in CNNs](#benefits-of-batch-normalization-in-cnns)
    - [Summary](#summary)
    - [Reasons for (\\gamma) and (\\beta) in Batch Normalization](#reasons-for-gamma-and-beta-in-batch-normalization)
    - [Role of (\\gamma) and (\\beta) in Training](#role-of-gamma-and-beta-in-training)
    - [Practical Example in PyTorch](#practical-example-in-pytorch)
    - [Summary](#summary-1)
  - [Layer Normalization](#layer-normalization)
    - [How Layer Normalization Works](#how-layer-normalization-works)
    - [Advantages of Layer Normalization](#advantages-of-layer-normalization)
    - [why we use Layer Normalization in Bert](#why-we-use-layer-normalization-in-bert)
    - [Bert Layer Normalization Params](#bert-layer-normalization-params)
    - [Layer Normalization in BERT](#layer-normalization-in-bert)
    - [Mathematical Formulation](#mathematical-formulation)
    - [Practical Implementation in PyTorch](#practical-implementation-in-pytorch-1)
  - [Instance Normalization](#instance-normalization)
  - [Group normalization(2018)](#group-normalization2018)


# **Normalization**

## Batch-Normalization

Normalization is a technique used in deep learning to stabilize and accelerate the training of neural networks.

### 原理：
1.	Normalization之后更接近Gauss 分布，有利于加快训练
2.	减弱 Covariance shift 现象：
the “covariate shift” problem can be reduced by fixing the mean and the variance of the summed inputs within each layer.
使得layer之间不那么相互依赖，更加具有独立性，从而有利于训练的稳定性，具有 regularizatioin 的作用

### 全连接层如何使用 BatchNorm?
全连接层输出的结果： [batch_size, input_size] ->  [batch_size, output_size]
每个hiddent_unit 处有1对γ和β参数，同一个 batch 同一个 hiddent_unit 共享同一对参数。
参数量： output_size*2

### 卷积层如何使用BatchNorm？
卷积之后的输出的 tensor : [Batch_size, height, width, channels]

1个卷积核产生1个feature map，1个feature map有1对γ和β参数，同一batch同channel的feature map共享同一对γ和β参数，若卷积层有n个卷积核，则有n对γ和β参数。
参数量： channels * 2

Batch normalization in convolutional neural networks (CNNs) works similarly to batch normalization in fully connected networks but is adapted to handle the spatial dimensions (height and width) of image data. Here’s a detailed explanation of how batch normalization operates within CNNs:

### How Batch Normalization Works in CNNs

For CNNs, batch normalization is applied to the feature maps produced by convolutional layers. The main steps are similar to those in fully connected layers but take into account the spatial dimensions of the data.

1. **Input Shape:**
   - In CNNs, the input to a batch normalization layer typically has the shape \((N, C, H, W)\), where:
     - \(N\) is the batch size.
     - \(C\) is the number of channels (features) in the output of the convolutional layer.
     - \(H\) is the height of the feature map.
     - \(W\) is the width of the feature map.

2. **Normalization:**
   - Batch normalization normalizes each feature map channel across the mini-batch and spatial dimensions. For each channel \(c\), the mean and variance are computed over the mini-batch and the spatial dimensions:
     \[
     \mu_c = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{nchw}
     \]
     \[
     \sigma_c^2 = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} (x_{nchw} - \mu_c)^2
     \]

3. **Standardization:**
   - Normalize the input feature map for each channel:
     \[
     \hat{x}_{nchw} = \frac{x_{nchw} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}
     \]
     where \(\epsilon\) is a small constant added for numerical stability.

4. **Scaling and Shifting:**
   - Apply learned affine transformation to the normalized feature map:
     \[
     y_{nchw} = \gamma_c \hat{x}_{nchw} + \beta_c
     \]
     where \(\gamma_c\) (scale) and \(\beta_c\) (shift) are learnable parameters specific to each channel.

### Practical Implementation in PyTorch

Here’s an example of implementing batch normalization in a CNN using PyTorch:

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# Example usage
model = SimpleCNN()

# Assume input tensor with batch size of 8, 3 channels, height and width of 32
input_tensor = torch.randn(8, 3, 32, 32)
output_tensor = model(input_tensor)
print(output_tensor.shape)  # Output: torch.Size([8, 32, 8, 8])
```

### Benefits of Batch Normalization in CNNs

1. **Training Stability:**
   - By normalizing the feature maps, batch normalization reduces the internal covariate shift, leading to more stable and faster training.

2. **Higher Learning Rates:**
   - Normalized inputs allow for the use of higher learning rates, which can speed up convergence.

3. **Regularization Effect:**
   - Batch normalization provides a regularization effect, reducing the need for other regularization techniques like dropout in some cases.

4. **Improved Gradient Flow:**
   - Normalized activations lead to better gradient flow through the network, which helps in training deep CNNs.

5. **Reduced Sensitivity to Initialization:**
   - Networks with batch normalization are less sensitive to weight initialization, making the training process more robust.

### Summary

Batch normalization in CNNs normalizes the activations across the mini-batch and spatial dimensions for each feature map channel, stabilizing and accelerating the training process. It allows for higher learning rates, acts as a regularizer, and improves gradient flow, making it an essential component in modern CNN architectures.


In batch normalization, the two learnable parameters, \(\gamma\) (scale) and \(\beta\) (shift), are crucial for several reasons. These parameters ensure that the normalization process does not limit the representational power of the network and allow the model to learn the optimal scaling and shifting of the normalized outputs. Here's a detailed explanation:

### Reasons for \(\gamma\) and \(\beta\) in Batch Normalization

1. **Restoring Representational Power:**
   - After normalization, the mean of the features is zero and the variance is one. While this normalization helps in stabilizing the training, it can also alter the original distribution of the data, potentially limiting the network's ability to represent complex functions.
   - \(\gamma\) and \(\beta\) are introduced to restore the network's ability to represent the identity function if necessary. This means the network can learn to undo the normalization if the original scale and shift are beneficial for the task.

2. **Flexibility in Representations:**
   - The learnable parameters allow the network to adaptively scale and shift the normalized activations. This flexibility helps the network to better model the underlying data distribution and improves its ability to learn complex patterns.
   - Without these parameters, the normalized activations might be forced into a distribution that is not optimal for the task, potentially reducing the model's performance.

3. **Learning Optimal Distributions:**
   - By learning the optimal values for \(\gamma\) and \(\beta\), the network can find the best trade-off between normalized activations and the ability to learn complex mappings.
   - During training, \(\gamma\) and \(\beta\) are adjusted through backpropagation to minimize the loss function, allowing the network to learn the most effective scaling and shifting for the specific task.


### Role of \(\gamma\) and \(\beta\) in Training

1. **Gradient Flow:**
   - The parameters \(\gamma\) and \(\beta\) ensure that the gradient flow is not disrupted by the normalization process. By scaling and shifting the normalized activations, they help maintain a smooth gradient flow, which is essential for efficient training of deep networks.

2. **Adaptability:**
   - \(\gamma\) and \(\beta\) provide adaptability to the network, allowing it to learn the best representation of the data. This adaptability is crucial in deep learning, where the goal is to learn hierarchical representations of the input data.

3. **Preservation of Information:**
   - By allowing the network to learn how to scale and shift the normalized activations, \(\gamma\) and \(\beta\) help preserve the information contained in the original features. This preservation is important for maintaining the network's performance on the given task.

### Practical Example in PyTorch

Here’s an example to illustrate how \(\gamma\) and \(\beta\) are used in PyTorch's batch normalization layer:

```python
import torch
import torch.nn as nn

# Example model with batch normalization
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization layer
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch normalization layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # Apply batch normalization
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)  # Apply batch normalization
        x = self.relu(x)
        return x

# Example usage
model = SimpleCNN()

# Assume input tensor with batch size of 8, 3 channels, height and width of 32
input_tensor = torch.randn(8, 3, 32, 32)
output_tensor = model(input_tensor)
print(output_tensor.shape)  # Output: torch.Size([8, 32, 32, 32])

# Access the learnable parameters
print(model.bn1.weight)  # This is gamma
print(model.bn1.bias)    # This is beta
```

In this example, `nn.BatchNorm2d(16)` and `nn.BatchNorm2d(32)` create batch normalization layers for the convolutional outputs with 16 and 32 channels, respectively. The learnable parameters \(\gamma\) and \(\beta\) are accessed through `model.bn1.weight` and `model.bn1.bias`.

### Summary

The learnable parameters \(\gamma\) and \(\beta\) in batch normalization are essential for:
- Restoring the network's representational power.
- Providing flexibility in scaling and shifting the normalized outputs.
- Ensuring optimal gradient flow and adaptability during training.
- Preserving the information in the original features while benefiting from the stabilization provided by normalization.

## Layer Normalization

- by normalizing the inputs across the features for each training example, which differs from other normalization techniques like batch normalization, which normalizes across the batch.

Here’s a detailed explanation of how layer normalization works:

### How Layer Normalization Works

1. **Input to a Layer:**
   - Consider an input \(\mathbf{x}\) to a layer in a neural network. This input can be a vector or a higher-dimensional tensor.

2. **Compute Mean and Variance:**
   - For a given input \(\mathbf{x}\), calculate the mean \(\mu\) and variance \(\sigma^2\) across all the features in that layer for each individual training example. Specifically, for a given input vector \(\mathbf{x} = [x_1, x_2, \ldots, x_d]\):
     \[
     \mu = \frac{1}{d} \sum_{i=1}^d x_i
     \]
     \[
     \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2
     \]

3. **Normalize:**
   - Normalize the input \(\mathbf{x}\) using the computed mean and variance. This process centers the data around zero mean and scales it to have unit variance:
     \[
     \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
     \]
     where \(\epsilon\) is a small constant added for numerical stability.

4. **Scale and Shift:**
   - Finally, apply a learned affine transformation to the normalized input. This transformation involves scaling and shifting the normalized values using parameters \(\gamma\) and \(\beta\):
     \[
     y_i = \gamma \hat{x}_i + \beta
     \]
     where \(\gamma\) and \(\beta\) are learnable parameters that allow the network to restore the representational power of the original activations.

### Advantages of Layer Normalization

1. **Independence from Batch Size:**
   - Unlike batch normalization, layer normalization does not depend on the size of the batch and is applied at the level of individual training examples. This makes it suitable for RNNs and other models where the batch size may be variable or small.
  
2. **Handling Variable Sequence Lengths:**
   - Layer normalization is applied to each token in the sequence independently of other tokens. This makes it naturally suited to handle variable-length sequences, which are common in NLP tasks.

3. **Sequential Data:**
   - Layer normalization is better suited for the sequential nature of the data processed by BERT. It ensures that the normalization process respects the structure of the sequence data, leading to better performance and training stability.


### why we use Layer Normalization in Bert

However, batch normalization has certain limitations when applied to transformer architectures like BERT:

1. Sequence Length Variability:

In natural language processing (NLP) tasks, the sequence length can vary significantly between different inputs. Batch normalization would require consistent sequence lengths across the batch, making it challenging to handle variable-length sequences efficiently.

2. Dependency on Batch Size:

Batch normalization depends on the statistics of the entire batch. This can cause issues when the batch size is small or varies during training, leading to noisy estimates of the mean and variance, which can destabilize training.

3. Sequential Data Nature:

In transformer models like BERT, the input data is sequential and often processed token-by-token or in smaller chunks. The dependencies across tokens within a sequence are crucial, and batch normalization might not be well-suited to capture these dependencies effectively.

### Bert Layer Normalization Params

- Bert 中的所在位置： embedding + self_attention + ff 层之后都接一个 LN
- Embedding层后的LN:  
  Embedding的输出：[max_length, embedding_size]   
  对每个维度上的所有的token 进行layer_normalization，每个维度对应2个参数  
  参数量： embedding_size*2  

- self_attention层后的LN:  
  self_attention的输出 [max_length, hidden_size]  
  对每个维度进行layer_normalization，每个维度对应2个参数  
  参数量： hidden_size*2

- ff 层后LN  
  ff的输出 [max_length, hidden_size]  
  对每个维度进行layer_normalization，每个维度对应2个参数  
  参数量： hidden_size  *2 


In BERT, the layer normalization parameters are not dependent on the sequence length. Instead, they are dependent on the feature dimensions of the inputs to the layers they normalize. Here’s a detailed explanation of how layer normalization parameters are handled in BERT:

### Layer Normalization in BERT

1. **Layer Normalization Parameters:**
   - The layer normalization in BERT involves two key learnable parameters:
     - \(\gamma\) (scale): A scaling factor.
     - \(\beta\) (shift): A shifting factor.
   - These parameters are vectors with the same size as the hidden dimension of the model (e.g., 768 for BERT base).

2. **Normalization Process:**
   - Layer normalization standardizes the inputs across the feature dimension for each individual example, not across the sequence length. This means that for an input tensor \(\mathbf{x} \in \mathbb{R}^{\text{seq\_len} \times \text{hidden\_dim}}\), the mean and variance are computed over the hidden dimension.

3. **Sequence Length Independence:**
   - The sequence length (\(\text{seq\_len}\)) can vary, but this does not affect the layer normalization parameters. The \(\gamma\) and \(\beta\) parameters are applied to each position in the sequence independently. The normalization is done per position per feature, ensuring that each token in the sequence is normalized independently of the other tokens.

### Mathematical Formulation

Given an input tensor \(\mathbf{x} \in \mathbb{R}^{\text{seq\_len} \times \text{hidden\_dim}}\):

1. **Compute Mean and Variance:**
   - For each position \(i\) in the sequence:
     \[
     \mu_i = \frac{1}{\text{hidden\_dim}} \sum_{j=1}^{\text{hidden\_dim}} x_{ij}
     \]
     \[
     \sigma_i^2 = \frac{1}{\text{hidden\_dim}} \sum_{j=1}^{\text{hidden\_dim}} (x_{ij} - \mu_i)^2
     \]

2. **Normalize:**
   - Normalize the inputs at each position:
     \[
     \hat{x}_{ij} = \frac{x_{ij} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}
     \]

3. **Scale and Shift:**
   - Apply the learned \(\gamma\) and \(\beta\) parameters:
     \[
     y_{ij} = \gamma_j \hat{x}_{ij} + \beta_j
     \]
   - Note that \(\gamma_j\) and \(\beta_j\) are shared across all positions \(i\) but are specific to each feature \(j\).

### Practical Implementation in PyTorch

Here’s how you can implement layer normalization in PyTorch, demonstrating the sequence length independence:

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Example usage
hidden_dim = 768
layer_norm = LayerNorm(hidden_dim)

# Assume input tensor with sequence length of 10, batch size of 32, and hidden dimension of 768
input_tensor = torch.randn(32, 10, hidden_dim)
output_tensor = layer_norm(input_tensor)

print(output_tensor.shape)  # Output: torch.Size([32, 10, 768])
```

In this implementation, `LayerNorm` computes the mean and standard deviation along the last dimension (hidden dimension) for each sequence position independently, ensuring that the normalization parameters \(\gamma\) and \(\beta\) are applied correctly across varying sequence lengths.

## Instance Normalization

Layer Normalization在抛开对Mini-Batch的依赖目标下，为了能够统计均值方差，很自然地把同层内所有神经元的响应值作为统计范围，那么我们能否进一步将统计范围缩小？对于CNN明显是可以的，因为同一个卷积层内每个卷积核会产生一个输出通道，而每个输出通道是一个二维平面，也包含多个激活神经元，自然可以进一步把统计范围缩小到单个卷积核对应的输出通道内部。
CNN卷积后的 IN:
针对不同的通道的 m*n的feature map，设计不同 IN，故
参数量： channels * 2

## Group normalization(2018)

从上面的Layer Normalization和Instance Normalization可以看出，这是两种极端情况，Layer Normalization是将同层所有神经元作为统计范围，而Instance Normalization则是CNN中将同一卷积层中每个卷积核对应的输出通道单独作为自己的统计范围。那么，有没有介于两者之间的统计范围呢？通道分组是CNN常用的模型优化技巧，所以自然而然会想到对CNN中某一层卷积层的输出或者输入通道进行分组，在分组范围内进行统计。这就是Group Normalization的核心思想，是Facebook何凯明研究组2017年提出的改进模型。
2018年Kaiming提出了GN(Group normalization)，成为了ECCV2018最佳论文提名。