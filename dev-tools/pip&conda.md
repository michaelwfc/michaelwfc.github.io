
# Pip 
##  pip commands
| command|Description|Note|
|-------------|------------|----|
|pip config list| 查看当前使用config info|
|pip config -v debug|查看当前使用config files | ~/.pip/pip.conf[linux] or ~/pip/pip.ini or C:\Users\AppData\Roaming\pip\pip.ini|
|pip freeze > requirements.txt|||
|pip  download  -r requirements.txt -d dependency_packages|下载已安装的包||
|pip install --no-index --find-links=./dependency_packages -r requirements.txt| .离线安装已有requirements.txt的项目依赖||
|conda env create -f  conda_yk.yaml -n env_test| create env from yaml||
|conda config --show-sources|  C:\Users\username\.condarc||

## pip.ini
```json
[global]
timeout=10000
sslverify = false

http_proxy=http://proxy.***.com:80
https_proxy=http://proxy.***.com:80

index-url=http://mirrors.aliyun.com/pypi/simple/
index-url=https://pypi.tuna.tsinghua.edu.cn/simple/
index-url=http://pypi.mirrors.ustc.edu.cn/simple/


[install]
trusted-host=mirrors.aliyun.com
trusted-host=pypi.tuna.tsinghua.edu.cn
trusted-host=pypi.mirrors.ustc.edu.cn

trusted-host = pypi.python.org pypi.org files.pythonhosted.org

```

# Conda

## conda command

conda config command    |function        |   |
------------------------|----------------|---|
conda info       ||
conda update conda  ||
conda config --show-sources | show config files and configs|
conda config --remove channels |删除指定源
conda config --add channels | 添加指定源(国内源：建议使用清华的)| conda config --add channels ***
conda config --set show_channel_urls yes  ||
conda config --remove-key channels|换回默认源||


## conda env command
 conda env command    |function|     |
------------------------|----------------|---
conda env export > environment.yml| Create environment.yml file via conda|
conda env create -f environment.yml -n env_test| create env from yaml
conda info --envs | list all the conda environment available
conda create --name envname| Create new environment named as `envname`
conda remove --name envname --all| Remove environment and its dependencies
conda create --name clone_envname --clone envname| Clone an existing environment

## .condarc

<!-- Do not use tabs, there must be space between http: and http://... -->
```json
ssl_verify: false 
proxy_servers: 
  http: http://proxy.***.com:80 
  https: http://proxy.***.com:80 
channels: 
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main 
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free 
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ 
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/ 
  - https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/ 
  - https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/ 
  - https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/ 
  - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/ 
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/free/ 
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/ 
  - defaults
show_channel_urls: true 
allow_other_channels: true
```