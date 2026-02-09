---
title: 香橙派5（RK3588）AI模型推理
cover: 
date: 2026-1-11 20:12:14
categories: 
- [文档&笔记]
tags:
- 笔记
description: 使用RK3588的NPU进行图像恢复模型的运行部署
---
# RKNN安装
注意事项：
+ 如果使用VMWare创建ubuntu20.04系统进行以下步骤，默认的4G内存和20G存储空间是不够的，我用的是8+60G
+ 安装`requirements`不要使用清华源，没有`tensorflow==2.8.0`，这里换成华为源
```
sudo apt-get install python3 python3-dev python3-pip
sudo apt-get update
sudo apt-get install libxslt1-dev zlib1g-dev libglib2.0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc
git clone https://github.com/airockchip/rknn-toolkit2 -b v1.5.2
cd rknn-toolkit2-1.5.2
pip3 install -r doc/requirements_cp38-1.5.2.txt -i https://repo.huaweicloud.com/repository/pypi/simple/
pip3 install packages/rknn_toolkit2-1.5.2+b642f30c-cp38-cp38-linux_x86_64.whl
```

