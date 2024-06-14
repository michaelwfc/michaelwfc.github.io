# cuda install methods

1. 系统层面直接安装某个版本的CUDA（不推荐）
   - GPU不具有向前的兼容性，可能新的GPU就需要完全更新系统上安装的 cuda
   - 某个版本的包对应cuda版本没有
2. 系统层面安装nvidia 最新的driver，通过 conda安装 环境级的 cudatoolkit 和 pytorch。  
   通过 Anaconda 安装的应用程序包位于安装目录下的 /pkg 文件夹中，如 /home/xxx/anaconda3/pkgs/ ，用户可以在其中查看 conda 安装的 cudatoolkit 的内容，如下图所示。可以看到 conda 安装的 cudatoolkit 中主要包含的是支持已经编译好的 CUDA 程序运行的相关的动态链接库。
   - 磁盘占用
   - conda 安装的cudatoolkit 只是运行环境，缺少了很多cuda的包，包括cuda的编译器nvcc

  

3. Nvidia driver, nvidia docker

[Managing CUDA dependencies with Conda](https://towardsdatascience.com/managing-cuda-dependencies-with-conda-89c5d817e7e1)

|Nvidia driver|cuda       |cudnn      |pytorch    |tensorflow |
|-------------|-----------|-----------|-----------|-----------|
|             | 10.0      |           |           |           |
|             |10.1       |           | 1.7.1     |2.3.0|
|             |11.2       |           |           |     |



# cuda on windows10

- https://www.tensorflow.org/install/source#gpu
- https://pytorch.org/get-started/previous-versions/

## cuda on windows10 with conda

conda env on windows10:  
python3.7.13  
NVIDIA-SMI 472.39        
Driver Version: 472.98        
CUDA Version: 11.4  
cudnn==8.2.1  
torch==1.12.1  
tensorflow==2.6.0


```bash
# create env with conda
conda create -n *** python=3.7

# 查看 cuda 版本 
nvidia-smi -l 
# 搜索 cudatoolkit 版本 
conda search cudatoolkit 
conda install -c anaconda cudatoolkit==11.3.1 
conda list cudatoolkit 
# 搜索 cudnn版本 
conda search cudnn 
conda install -c anaconda cudnn==8.2.1 
conda list cudnn 
# install tensorflow or torch : 
conda install pytorch==1.12.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
# pip install will fialed because of net error
# pip install torch==1.12.1  --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorflow==2.6.0



# Python program to move a tensor from CPU to GPU 
# import torch library 
import torch
print(torch.__version__) 
print(torch.cuda.is_available())
# create a tensor 
x = torch.tensor([1.0,2.0,3.0,4.0]) 
print("Tensor:", x) 
# check tensor device (cpu/cuda) 
print("Tensor device:", x.device) 
# Move tensor from CPU to GPU 
# check CUDA GPU is available or not 
print("CUDA GPU:", torch.cuda.is_available()) 
if torch.cuda.is_available(): 
   x = x.to("cuda:0") 
   # orp x=x.to("cuda") 
print(x) 
# now check the tensor device 
print("Tensor device:", x.device)


import tensorflow as tf 
tf.__version__ 
tf.config.list_physical_devices() 
tf.debugging.set_log_device_placement(True)  
with tf.device('/device:GPU:0'):   
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) 
# print device failed,but nvidia-smi is runing
a.device
try:  
  # Specify an invalid GPU device  
  with tf.device('/device:GPU:2'):  
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  
    c = tf.matmul(a, b)  
except RuntimeError as e:  
  print(e)

```


# GPU
| architecture| GPUS                       |                   |
|-------------|----------------------------|-------------------|
|             | V100 | |
|             | A5000|
| Ampere      |NVIDIA A10G or RTX 4090/3090| Flash attention   |


