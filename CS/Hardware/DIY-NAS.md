---
title: "DIY-NAS"
source: "https://app.notion.com/p/DIY-NAS-1e7198c00c32809fa3fbf3a4a952ac62"
author:
published:
created: 2026-07-18
description: "A collaborative AI workspace, built on your company context. Build and orchestrate agents right alongside your team's projects, meetings, and connected apps."
tags:
  - "clippings"
---


## DIY NAS

[post.smzdm.com](https://post.smzdm.com/p/a3d2ex8r/)

[zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/480452743)

B站司波图，翻他早期的视频，慢慢看；然后百度直接搜关键词，

什么值得买上面很多硬核玩家，例如阿文菌算是入门里折腾教程比较好的，虽然他也退坑了。

NGA论坛玩家，恩山论坛玩家等都比较靠谱

## 需求

6/8盘位即可，不用太多，需要做冗余备份

支持 Time Machine，用于备份我的 MacBook Pro/iphone/android/windows

能够在家中或外出时随时随地访问到上面的文件

能有一个完善的家庭影音系统，或者叫媒体管理器，且支持硬件解码

万兆网卡，支持局域网内高速访问

有成熟的文件权限管理，隔离不同组别用户的访问

有虚拟机功能，运行 Linux 系统

有 Docker 功能，跑一些轻量化的容器

## 原理

NAS的本质是7×24小时运行的存储服务器，硬件选择要遵循三个原则：

1\. 低功耗：电费比硬件更贵，TDP 15W以下最佳

2\. 扩展性：至少4个SATA接口，支持PCI-E扩展

3\. 兼容性： [黑群晖](https://zhida.zhihu.com/search?content_id=255675755&content_type=Article&match_order=1&q=%E9%BB%91%E7%BE%A4%E6%99%96&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDYzNTY2MTMsInEiOiLpu5HnvqTmmZYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTU2NzU3NTUsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.EFJgVOoY6JJoK2tTSDKVINOvzHEgeWVUOTN5EspG5dQ&zhida_source=entity) / [TrueNAS](https://zhida.zhihu.com/search?content_id=255675755&content_type=Article&match_order=1&q=TrueNAS&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDYzNTY2MTMsInEiOiJUcnVlTkFTIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjU1Njc1NzU1LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.wfpmmZOkVDsl7z4_7u9vAEC8i8v10uD8YHqPZq0dFnE&zhida_source=entity) 等系统对硬件有要求

### zfs

zfs牛逼的点在于：

磁盘读写少，更加均匀，表现为同样的硬盘放在truenas上声音小，也就有效延长了硬盘寿命

压缩功能，变相硬盘扩容，8T硬盘压缩1.2相当于扩容了1.6T空间（但对视频图片没什么用）

自带去重，猜测就是指针改一下，维护个目录就行了这种形式（实际使用我坚决不开启）

写时复制，不会因为系统卡死、意外中断导致数据消失

无须花额外的钱买阵列卡，采用raidz就能达到冗余安全的效果（或直接it固件的直通卡，直接将所有硬盘交给NAS进行阵列的组建，才能利用上这些他们的文件系统），而且性能、稳定性不输阵列卡，也比群晖、威联通、winserver那种基于mdadm的软raid要好

几乎完备的企业级数据保护方案：如热备盘，系统mirror，包括前面提到过的raidz（热备盘机制不是很稳定，不建议用，系统mirror也没必要，因为写入真的很少，更不建议用U盘当系统盘，玩花的一般死的都早，建议老老实实企业级mlc固态128G当系统盘就行了，或者三星的固态，一块足够，两块万全）

性能方面可以加入ssd缓存提升存储池内的并发访问速度，组2.5G局域网甚至万兆网体验很美好（不建议花冤枉钱给freenas加缓存盘，加内存条的收益来的多的多的多）

拷贝文件速度稳定而快速

而且都是zfs，ubuntu、freenas、omv等表现也各不相同，因为freeBSD开发的东西原生更牛批，不是linux能比的，不过truenas scale出来，应该差距不大了，往后差距会越来越小。

### raid

组raid当然可以，而raid5不可靠（重建不可靠），raid6较为靠谱，4盘位NAS是足够组raid6的

## NAS软件系统架构

[Techno TimTrueNAS vs Unraid - Which one is the BEST NAS OS for my Home…](https://www.youtube.com/watch?v=4p-INidMqxY)

黑群晖

truenas

UNRAID

openmediavault

## PVE+TrueNas

软件：宿主机 [PVE](https://zhida.zhihu.com/search?content_id=194899123&content_type=Article&match_order=1&q=PVE&zhida_source=entity) ；虚拟机 [freenas](https://zhida.zhihu.com/search?content_id=194899123&content_type=Article&match_order=1&q=freenas&zhida_source=entity) ；硬盘直通给freenas；

freenas只管一件事：存储；其它虚拟机不存储数据，只负责计算相关工作，即所谓“物理机上ALL IN ONE，逻辑上各司其职”，以降低耦合性，提高灵活性、扩展性，不会把docker和虚拟机等应用都装在freenas上

现在叫 [truenas](https://zhida.zhihu.com/search?content_id=194899123&content_type=Article&match_order=1&q=truenas&zhida_source=entity) ，开源免费，truenas scale版本也加入debian怀抱，功能性更强

无论群晖还是威联通都是软raid，可靠性一般，相比freenas 的 [zfs](https://zhida.zhihu.com/search?content_id=194899123&content_type=Article&match_order=1&q=zfs&zhida_source=entity) 还是差了些。而且freenas，准确说是zfs文件系统带scrub内容校验功能，可以定期查看数据是否健康，能保证哪怕一个bit都不会丢失或错误，这样的数据保障正是我要的 数 据 安 全！

系统上我没有直接选择安装truenas scale，而是选择了PVE，实现“物理上的ALL IN ONE，逻辑上的低耦合”，具体实现方式：

底层是PVE，虽然PVE也可以有docker，可以zfs，但我不用，重要的是逻辑简单，各司其职

PVE只负责一件事：管理好虚拟机。PVE作为底层宿主机，在安装系统的时候可以raidz1、raidz2的，如果你不放心的话，可以拿几块SSD实现。

PVE基础上，虚拟一个freenas负责数据存储。这里虽然我用了truenas scale，但我不会用truenas 直接安装docker应用，还是那句话，逻辑简单，各司其职，要干各自擅长的事情，某个事情，而不是什么都干，充分降低耦合，提高灵活性。这里提醒一点，truenas scale系统仅仅占用5G附近，为了后续备份系统更快，给它16G才行，8G虽然够用然而升级的时候会提示空间不足，因此最低要给16G，多了也用不到白搭。

然后，truenas在我这里只负责存储，smb共享，nfs共享，iscsi等，做好定期的scrub和smart检测，及时发邮件提醒我换硬盘，或者再加rsync服务增量备份。

PVE基础上虚拟一个openwrt，用法自定，可以直接软路由，我个人直接旁软路由灵活的满足局域网内走外网的需求

PVE基础上虚拟一个windows，安装emby和qbittorrent，PVE直通显卡给windows可以实现硬解，甚至玩个游戏挂个游戏都行。

PVE可以直接直通硬盘，也就是说可以把装着系统盘的磁盘直接挂上去就能运行了。有点类似双系统，但其实不是，还是归PVE管。

我个人是PT爱好者，比较重度的那种，需要qbittorrent强有力的界面管理更新上千个种子，不是nas那种简单的webui能满足的，甚至我还经常面临迁移的情况，上千个种子迁移在windows版本下更方便。

PVE安装ubuntu、centos、debian、omv、爱快、黑威联通等，主要是做实验，出教程用。

其它的，我会用一个centos安装nginx跑代理服务，跑防火墙，一个debian安装docker只运行docker服务，一个ubuntu安装数据库服务，只运行各类数据库，甚至虚拟几个玩集群，pve也支持ceph等等等等

这就是我说的“硬件上的ALL IN ONE，逻辑上各司其职”不会真的boom！

### 同步app： Resilio Sync /syncthing

使用免费的 Resilio Sync /syncthing 用最快的速度将手机相册同步到 NAS 服务器

P2P类的同步软件，tracker一访问不了全部同步停止，

Rsync只要保证IP可访问即可，按需选择。 网盘同步的有容量限制，我现在同步的文件都有10t以上，Rsync不受容量限制，按需选择。 其他同步软件跟Rsync只是各有所长，不存在互秒的关系，如果Rsync是一个废物，它不会成为这么多NAS的标配同步软件。

### 相册app： immich

### 影音app： Emby/plex

Docker中暂时安装了一个容器，那就是家庭影音中心 Emby

docker run -d \\ --name emby \\ --device=/dev/dri:/dev/dri \\ -e UID=0 \\ -e GID=0 \\ -e GIDLIST=2 \\ -p 1900:1900 \\ -p 7360:7359 \\ -p 8097:8096 \\ -p 8921:8920 \\ -v /volume1/docker/emby/config:/config \\ -v /volume1/电影:/media/电影 \\ -v /volume1/剧集:/media/剧集 \\ --restart=always \\ amilys/embyserver:latest