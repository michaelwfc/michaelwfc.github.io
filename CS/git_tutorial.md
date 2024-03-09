# git command

|git command  |Desription| Note|
|------------|----------|------|
|git clone https://***.git||
|git branch -a|查看所有分支||
|git branch -r|查看远程分支||
|git branch -vv|查看本地分支所关联的远程分支||
|git checkout dev| 切换到dev分支(后面创建新分支的时候，默认是会使用当下分支的指针||
git checkout -b feature|新建一个本地分支||
|git branch --set-upstream-to=remote-branch| 将本地分支关联到远程分支   ||
|git push --set-upstream origin remote-branch| ||
|git push origin remote-branch| ||
|git branch -D feature|delete local feature branch||
|git push origin :feature|delete remote feature branch||
|git tag|查看本地tag|
|git checkout tag_name |git切换到tag| 此时git可能会提示你当前处于“detached HEAD” 状态。因为tag相当于一个快照，不能修改它的代码。需要在tag代码基础上做修改，并创建一个分支|
|git checkout -b branch_name tag_name|切换到指定的tag（tag_name)，并创建一个分支 ||
|git stash list  |查看stash了哪些存储||
|git stash show |显示做了哪些改动，默认show第一个存储,如果要显示其他存贮，后面加stash@{$num}，比如第二个 git stash show stash@{1}||
|git stash pop stash@{$num} |应用某个存储||
|git log --pretty=oneline|||
|git restore --staged file| to discard changes in working directory|file not 没有进行 add . 和 commit 操作|  
|git add|The git add command is used to add changes to the staging index.||
|Git reset HEAD file |is primarily used to undo the staging index changes.| A --mixed reset will move any pending changes from the staging index back into the working directory.|
|git revert HEAD|undo a public commit  |Git will create a new commit with the inverse of the last commit. This adds a new commit to the current branch histor|
|git reset --hard commit-id |本地代码库回滚：回滚到commit-id，commit-id之后提交的commit都去除|Doing a reset is great for local changes however it adds complications when working with a shared remote repository|
|git reset --hard HEAD^|本地代码库回滚删一个commit||

# Initial a project

## Clone a remote repository

git clone https://github.com/RasaHQ/rasa.git

<!-- 克隆下载指定版本 -->
git clone -b 1.10.12 https://github.com/RasaHQ/rasa.git

git clone is basically a combination of:  

- git init (create the local repository)  
- git remote add (add the URL to that repository)  
- git fetch (fetch all branches from that URL to your local repository)  
- git checkout (create all the files of the main branch in your working tree)  
Therefore, no, you don't have to do a git init, because it is already done by git clone  

## Start a new project and push to remote repository

- git init # 初始化一个本地仓库
- git add README.md # 将README.md文件加入到仓库中
- git commit -m "first commit" # 将文件commit到本地仓库
- git branch -M main  
- git remote add origin git@github.com:******/faq.git  
- git push -u origin main

# 协同开发&分支管理

|Branch      | Description|
|------------|-----------|
| master/main| 应该是非常稳定的，也就是仅用来发布新版本，平时不能在上面干活；|
| dev        | 干活都在dev分支上，也就是说，dev分支是不稳定的，到某个时候，比如1.0版本发布时，再把dev分支合并到master上，在master分支发布1.0版本；|
| Feature branch |开发一个新feature，最好新建一个分支|
|bug|修复bug时，我们会通过创建新的bug分支进行修复，然后合并，最后删除

## PR Merge vs Rebase

## PR时有冲突

# Git Config

.gitconfig : gitconfig 配置文件

- system级别  
- global（用户级别）  
- local（当前仓库）

## git config comand

|git command| func|
|-----------|-----|
|git config --global --list  | 查看当前用户（global）设置|
|git config --global user.name "****"| config the username  |
|git config --global user.email "****"| config the email address|
|git config --list --show-origin| Retrieve the locations (and name value pairs) of all git configuration files|  

## Configure proxy in git

git config --global http.proxy http://proxy.******.com:80
git config --global https.proxy http://proxy.******.com:80 
git config --global http.sslVerify false
git config --global https.sslVerify false

 <!-- 取消代理  -->
git config --global --unset http.proxy 
git config --global --unset https.proxy

# Git认证方式

## ssh 认证方式

- git remote set-url origin  git@github.com:******/faq.git
- git remote add origin git@github.com:******/faq.git

## ssh config

- 使用秘钥生成工具(ssh-keygen、puttygen等)生成rsa秘钥  
ssh-keygen -t rsa -C "******": 生成后的公钥会存放在 C:/Users/******/.ssh/id_rsa.pub 或者 /home/wangfc/.ssh/id_rsa, shell在用户主目录里找到.ssh目录

- 将rsa公钥添加到代码托管平台
添加生成的id_rsa.pub文件中的公钥（用记事本打开全部复制）到github的setting / SSH AND GPG KEY / SSH keys  

- 测试是否关联成功,git bash输入
ssh git@github.com 如果提示successfully authenticated则成功


## http认证方式

- git remote -v  
origin  https://github.com/******/adversarial_text_classification.git (fetch)  
origin  https://github.com/******/adversarial_text_classification.git (push)  
如果是以上的结果那么说明此项目是使用 https 协议进行访问的（如果地址是 git 开头则表示是 ssh 协议）

- 设置认证方式为 https 协议
git remote set-url origin   https://github.com/******/faq.git


- git push -u origin master #将本地仓库push远程仓库，并将origin设为默认远程仓库



# GIT config location

HOME directory:  

- When using the Windows command shell, batch scripts or Windows programs, HOME is %USERPROFILE% . The global config file will be read from %USERPROFILE%\.gitconfig  
- However, when you're using a (Bash) shell under MSYS2 or Cygwin, HOME under that shell is %HOME% . The global config file will be read from $HOME/.gitconfig  
- The global configuration file can be found on yet another location, for Windows programs that use their own HOME environment. Take Emacs (which uses magit) for example: When Emacs is started from a Windows command shell, it sets the HOME variable to %APPDATA% , which means that .gitconfig will reside in %APPDATA%\.gitconfig

SYSTEM DIRECTORY:  

- For msysGit: %ProgramFiles(x86)%\Git\etc\gitconfig
- For Git for Windows (64 bit): %ProgramFiles%\Git\mingw64\etc\gitconfig
- For MSYS2-native git: [MSYS2-install-path]\etc\gitconfig