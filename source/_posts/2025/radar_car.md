---
title: 毫米波雷达试验车
cover: /img/
date: 2025-11-4 15:22:24
categories: 
- [文档&笔记]
tags:
- 笔记
description: 
---
## 简介

## 开发环境
### 概述
由于毫米波雷达的驱动程序必须在x86架构的cpu上运行，因此无法使用原计划的NX或orangepi5，在电脑本地进行开发。如果无需使用显卡CUDA加速，使用VMware即可

win11系统下使用docker安装ros2的foxy-desktop镜像进行开发

### docker使用基础
+ 获取ROS2官方镜像
```
docker pull osrf/ros:foxy-desktop
```
+ 创建并启动ROS2容器
```
docker run -td --name=ros2 --net=host osrf/ros:foxy-desktop

-i: 交互式操作。启动容器立即进入容器，输入exit后容器退出
-t: 终端
-d: 后台运行。启动容器返回容器ID不会进入容器，容器在后台运行

--name: 自定义容器名称。在实际操作时也可以用容器id的前几位

--net：网络模式
桥接模式（Bridge）：默认模式，容器通过虚拟网桥 docker0 进行通信
主机模式（Host）：容器直接使用主机的网络接口和 IP 地址，性能较好，但隔离性较差
无网络模式（None）：容器没有网络能力，仅有回环接口，适用于独立应用或网络调试
容器模式（Container）：新容器与已存在的容器共享网络命名空间，共享 IP 地址和端口
```
如果需要显卡，也可以加参数`--gpus=all`

+ 正在运行的容器执行新的命令(/bin/bash打开终端)
```
docker exec -it ros2 /bin/bash
```
+ 启动已有的容器(`exit`退出)
```
docker start ros2
```
+ 删除容器
```
docker rm ros2
```
+ 查看容器(正在运行的和包括已停止的)
```
docker ps
docker ps -a
```

+ 将容器状态保存为新镜像
```
docker commit -a "Triority" -m "ubuntu22下构建基于ros2 humble的底盘和毫米波雷达驱动程序" ubuntu22 car_radar_driver:v1.0

docker commit [OPTIONS] <CONTAINER_ID> <new_image_name>:<tag>
-a: --author
-m: --message
```
+ 查看镜像
```
PS C:\Users\Triority> docker images
REPOSITORY         TAG            IMAGE ID       CREATED         SIZE
car_radar_driver   v1.0           c975f5d25b69   2 minutes ago   5.54GB
ubuntu             22.04          09506232a800   5 weeks ago     117MB
```
+ Push to Docker Hub
推送到Docker Hub的镜像必须遵循特定的命名格式：`<Your_Docker_Hub_Username>/<Repository_Name>:<Tag>`

下面这个命令不会复制镜像，它只是给现有的镜像创建了一个新的别名
```
docker tag car_radar_driver:v1.0 triority/car_radar_driver:v1.0

docker push triority/car_radar_driver:v1.0

docker rmi car_radar_driver:v1.0
```


## 主机显示docker内窗口：终端转发工具mobaxterm
在主机安装之后，选择`start local terminal`：
```
     ┌────────────────────────────────────────────────────────────────────┐
     │                • MobaXterm Personal Edition v25.3 •                │
     │              (X server, SSH client and network tools)              │
     │                                                                    │
     │ ⮞ Your computer drives are accessible through the /drives path     │
     │ ⮞ Your DISPLAY is set to 192.168.1.106:0.0                         │
     │ ⮞ When using SSH, your remote DISPLAY is automatically forwarded   │
     │ ⮞ Each command status is specified by a special symbol (✓ or ✗)    │
     │                                                                    │
     │ • Important:                                                       │
     │ This is MobaXterm Personal Edition. The Professional edition       │
     │ allows you to customize MobaXterm for your company: you can add    │
     │ your own logo, your parameters, your welcome message and generate  │
     │ either an MSI installation package or a portable executable.       │
     │ We can also modify MobaXterm or develop the plugins you need.      │
     │ For more information: https://mobaxterm.mobatek.net/download.html  │
     └────────────────────────────────────────────────────────────────────┘

  2025-11-06   15:37.33   /home/mobaxterm  
```
在容器中配置显示的转发地址：
```
export DISPLAY=192.168.1.106:0.0
``
此时内部的窗口就可以在主机内显示了（下面的命令会启动小海龟界面作为测试）
```
ros2 run turtlesim turtlesim_node
```
