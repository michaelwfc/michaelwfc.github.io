
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

 conda is a very powerful package manager that excels at managing dependencies and offers an easy way to create and use virtual environments for your projects. 


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


## .condarc

- Linux: /home/user/.condarc
- windows： C:\Users\user\.condarc

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

## conda env command

| conda env command                |Function                               |  Note   |
|----------------------------------|---------------------------------------|------------|
| conda env list                   |                                       |            |
|conda env export > environment.yml| Create environment.yml file via conda |
|conda env create -f environment.yml -n env_test| create env from yaml | 
|conda info --envs | list all the conda environment available
|conda create --name envname| Create new environment named as `envname`
|conda remove --name envname --all| Remove environment and its dependencies
|conda create --name clone_envname --clone envname| Clone an existing environment

### conda build env examples

```bash
conda create -n python3710 python=3.7.10
# 克隆环境
conda create --clone base -n paddle03  --no-deps -v --no-default-packages
# 离线安装
conda create  --clone root -n pytorch-1.7.1-cu10.1-env
conda install --use-local pytorch-1.7.1-py3.6_cuda9.2.148_cudnn7.6.3_0.tar.bz2

# windows
conda activate yourenvname
conda deactivate

# Linux
source activate
source deactivate

conda env remove -n paddle02 -all

```

## Using Anaconda3 On Linux

- download  Anaconda3-***-Linux-x86_64.sh
- install Anaconda3

```bash
chmod +x Anaconda3-5.1.0-Linux-x86_64.sh
sh Anaconda3-5.1.0-Linux-x86_64.sh
```

- activate .bashrc
  
```bash
source ~/.bashrc
```


# CMD ternimal

## Issue: "conda: command not found"

it usually means that Conda hasn't been added to your system’s PATH. Here’s how you can fix it

If you're getting a "no conda" message when trying to run the `conda` command on the command prompt (cmd) in Windows, it usually means that Conda hasn't been added to your system’s PATH. Here’s how you can fix it:

### 1. **Ensure Conda is Installed**

   If you used the Anaconda or Miniconda installer, open Anaconda Prompt (which automatically sets the PATH for Conda) and check the Conda version:
     ```bash
    conda --version
    conda list python #  This should list the installed Python version. If Python is installed, proceed to the next steps.
     ```
     If this command works in Anaconda Prompt, then Conda is installed correctly.

### 2. **Add Conda to PATH During Installation (Reinstall if Needed)**

   During installation, there’s an option to add Conda to the system PATH. If you skipped this, Conda won’t be recognized in other terminals like cmd.

   **Reinstall Anaconda/Miniconda** and make sure to check the box that says:
   - "Add Anaconda/Miniconda to my PATH environment variable"
   - "Register Anaconda/Miniconda as my default Python"

### 3. **Manually Add Conda to the System PATH**
   If you don’t want to reinstall, you can manually add Conda to the system PATH. Here’s how:

   #### a. **Find the Conda Installation Path**
   Typically, Conda is installed in:
   - **Anaconda:** `C:\Users\<YourUsername>\Anaconda3\`
   - **Miniconda:** `C:\Users\<YourUsername>\Miniconda3\`

   Make note of this path, as you'll need it to add Conda to the system PATH.

   #### b. **Edit the System PATH**
   1. Press `Windows + R`, type `sysdm.cpl`, and press Enter to open the System Properties window.
   2. Go to the **Advanced** tab and click on **Environment Variables**.
   3. Under "System variables" or "User variables," find the `Path` variable and select it. Click **Edit**.
   4. Add the following entries (adjust the path based on your installation):
      - For **Anaconda**:
        ```
        C:\Users\<YourUsername>\Anaconda3\
        C:\Users\<YourUsername>\Anaconda3\Scripts\
        C:\Users\<YourUsername>\Anaconda3\Library\bin\
        ```
      - For **Miniconda**:
        ```
        C:\Users\<YourUsername>\Miniconda3\
        C:\Users\<YourUsername>\Miniconda3\Scripts\
        C:\Users\<YourUsername>\Miniconda3\Library\bin\
        ```

      Replace `<YourUsername>` with your actual Windows username.

   5. Click **OK** to close all windows.

   #### c. **Restart the Command Prompt**
   After adding Conda to the PATH, close and reopen the command prompt. Now try running:
   ```bash
   conda --version
   ```
   This should print the Conda version if the PATH was correctly configured.

### 4. **Use Anaconda Prompt Instead of cmd (Optional)**
   If you don’t want to deal with the PATH issues, you can always use the Anaconda Prompt. It automatically sets up Conda’s PATH and is specifically designed for Conda operations.

### 5. **Check if Conda Initialization is Needed**
   If the `conda` command still doesn’t work, try initializing Conda for cmd:
   ```bash
   conda init cmd.exe
   ```
   Then restart your terminal.

### Summary

1. Reinstall Conda and check the "Add to PATH" option.
2. Or manually add Conda's installation path to the system PATH.
3. Open a new terminal and check if Conda is recognized by running `conda --version`. 

Let me know if you run into any issues!



#  Git Bash






## .bashrc

### git bashrc 的环境变量

```bash
# .bashrc
. /c/Users/user/anaconda3/etc/profile.d/conda.sh  # echo ". ${PWD}/conda.sh" >> ~/.bashrc
export PATH="$HOME/anaconda3/bin:$PATH"
export PATH="$PATH:/c/Users/****/poppler-0.68.0/bin"
export GRPC_JAVA_ADDRESS="/c/Program Files (x86)/Java/jre1.8.0_231/bin"
export GRPC_PYTHON_PORT=50051
export GRPC_SFTP_PATH=/c/Users/****/AppData/Local/Temp/GRPC_SFTP
export PYTHONPATH=/d/Projects/nlp/src
export PYTHON_CACHED_FILES=/d/Projects/nlp/data/cached_files
export https_proxy=http://proxy.****.com:80
export http_proxy=http://proxy.****.com:80

# ativate the environment
source ~/.bashrc
```



## Issue: "conda: command not found in git bash":

[Setting Up Conda in Git Bash](https://discuss.codecademy.com/t/setting-up-conda-in-git-bash/534473)

In order to make the conda command available in Git Bash, you need to add conda’s shell script to your .bashrc file.
The shell script we need is located inside of the folder your Anaconda distribution added to your computer. 

1. Verify Python is installed correctly in Conda with Anaconda Prompt  or cmd
2. add conda’s shell script to your .bashrc file
   navigate Anaconda distribution  etc -> profile.d , ex: /c/Users/user/anaconda3/etc/profile.d
   ```bash
   echo ". ${PWD}/conda.sh" >> ~/.bashrc
   # If the path does contain spaces (such as in your username), you will need to add single quotes in the command like so:
   echo ". '${PWD}'/conda.sh" >> ~/.bashrc 
   ```
   



