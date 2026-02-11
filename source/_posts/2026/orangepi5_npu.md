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
# 前言


# 部署
## PC端安装RKNN：模型转换、NPU运行仿真
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


## 简单u-net图像恢复模型训练到部署全流程实践
我之前曾经使用U-net网络制作了一个视频去水印工具，但是模型结构中为了使模型能够利用时间维度信息使用了nn.Conv3d三维卷积，而这一算子不受NPU原生支持，因此将时间维度折叠到颜色通道维度，从而加速板载推理。下面给出全部代码供参考

### 模型等类定义：post_model.py

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
import pathlib
import torchvision.transforms.functional as TF


class VideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length=5, size=(640, 352)):
        self.root_dir = pathlib.Path(root_dir)
        self.watermarked_dir = self.root_dir / 'watermarked_videos'
        self.mask_dir = self.root_dir / 'mask_videos'
        self.original_dir = self.root_dir / 'original_clips'

        self.watermarked_files = sorted([p for p in self.watermarked_dir.glob('*.mp4')])
        self.mask_files = sorted([p for p in self.mask_dir.glob('*.mp4')])
        self.original_files = sorted([p for p in self.original_dir.glob('*.mp4')])

        self.sequence_length = sequence_length
        self.size = size  # (Width, Height)

    def __len__(self):
        return len(self.watermarked_files)

    def _read_frames(self, path, is_mask=False):
        cap = cv2.VideoCapture(str(path))
        frames = []
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)

            if is_mask:
                if len(frame.shape) == 3: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
                frame = TF.to_tensor(frame)  # [1, H, W]
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = TF.to_tensor(frame) * 2.0 - 1.0  # 归一化到 [-1, 1]
            frames.append(frame)
        cap.release()

        # 补齐帧数
        while len(frames) < self.sequence_length:
            frames.append(frames[-1])

        return torch.stack(frames)  # [T, C, H, W]

    def __getitem__(self, idx):
        watermarked = self._read_frames(self.watermarked_files[idx])
        mask = self._read_frames(self.mask_files[idx], is_mask=True)
        original = self._read_frames(self.original_files[idx])

        # 拼接输入: [T, 4, H, W] (RGB + Mask)
        masked_input = torch.cat([watermarked, mask], dim=1)
        return masked_input, original, mask


class UNet2D_Temporal(nn.Module):
    def __init__(self, num_frames=5, in_channels_per_frame=4, out_channels_per_frame=3):
        super(UNet2D_Temporal, self).__init__()

        # 输入通道 = 帧数 * 4
        self.in_total = num_frames * in_channels_per_frame
        self.out_total = out_channels_per_frame  # 输出单帧

        self.enc1 = self._block(self.in_total, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = self._block(256, 512)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self._block(512 + 256, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self._block(256 + 128, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self._block(128 + 64, 64)

        self.final_conv = nn.Conv2d(64, self.out_total, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, T*C, H, W]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([e3, self.up3(b)], dim=1))
        d2 = self.dec2(torch.cat([e2, self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([e1, self.up1(d2)], dim=1))

        return torch.tanh(self.final_conv(d1))

```

### 显卡训练：post_train.py
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from post_model import VideoDataset, UNet2D_Temporal


def train():
    # 配置
    SEQ_LEN = 5
    BATCH_SIZE = 8
    LR = 1e-4
    INPUT_SIZE = (640, 352)  # 宽, 高
    DEVICE = torch.device('cuda')
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = VideoDataset(root_dir=r"D:\Dataset", sequence_length=SEQ_LEN, size=INPUT_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = UNet2D_Temporal(num_frames=SEQ_LEN).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()  # 混合精度

    print("开始训练...")
    for epoch in range(50):
        model.train()
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for masked_input, gt_seq, mask_seq in loop:
            # 维度转换: [B, T, C, H, W] -> [B, T*C, H, W]
            b, t, c, h, w = masked_input.shape
            masked_input = masked_input.view(b, t * c, h, w).to(DEVICE).float()

            # 目标只取中间帧
            mid = SEQ_LEN // 2
            target_frame = gt_seq[:, mid, :, :, :].to(DEVICE).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(masked_input)
                loss = criterion(preds, target_frame)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), f"{SAVE_DIR}/epoch_{epoch + 1}.pth")


if __name__ == '__main__':
    train()

```

### PC上的推理测试：post_infer_PC.py
```python
import torch
import cv2
import numpy as np
from collections import deque
import torchvision.transforms.functional as TF
from post_model import UNet2D_Temporal


def preprocess(frame, mask, size):
    # 预处理保持与训练一致
    f = cv2.resize(frame, size)
    m = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    if len(m.shape) == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    f_t = TF.to_tensor(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) * 2.0 - 1.0
    m_t = TF.to_tensor(m)
    m_t = (m_t > 0.5).float()
    return torch.cat([f_t, m_t], dim=0)


def main():
    SEQ_LEN = 5
    SIZE = (640, 352)
    DEVICE = 'cuda'

    model = UNet2D_Temporal(num_frames=SEQ_LEN).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/epoch_2.pth"))
    model.eval()

    cap = cv2.VideoCapture(r"test_data\input.mp4")
    cap_m = cv2.VideoCapture(r"test_data\mask.mp4")
    writer = cv2.VideoWriter(r"test_data\output_pc.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, SIZE)

    buffer = deque(maxlen=SEQ_LEN)

    while True:
        ret, frame = cap.read()
        ret_m, mask = cap_m.read()
        if not ret: break

        feat = preprocess(frame, mask, SIZE)
        buffer.append(feat)
        if len(buffer) < SEQ_LEN:
            continue  # 简单处理，等填满再推

        # 拼接: [1, T*C, H, W]
        input_tensor = torch.cat(list(buffer), dim=0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            res = model(input_tensor)

        # 后处理
        res = (res[0].permute(1, 2, 0).cpu().numpy() + 1) / 2.0
        res = np.clip(res * 255, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

    writer.release()


if __name__ == "__main__":
    main()

```
### 模型导出 PyTorch -> ONNX：post_pth2onnx.py
```python
import torch
from post_model import UNet2D_Temporal


def export():
    SEQ_LEN = 5
    SIZE = (640, 352)  # W, H
    MODEL_PATH = "checkpoints/epoch_2.pth"
    ONNX_PATH = "model_fp16.onnx"

    model = UNet2D_Temporal(num_frames=SEQ_LEN)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # 输入形状: [1, 帧数*4, 高, 宽]
    # PyTorch 是 NCHW，这里 SIZE[1] 是高
    dummy_input = torch.randn(1, SEQ_LEN * 4, SIZE[1], SIZE[0])

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        opset_version=12,
        input_names=['input'],
        output_names=['output']
    )
    print(f"导出完成: {ONNX_PATH}")


if __name__ == "__main__":
    export()

```
### PC端 RKNN 仿真：post_simulate_rknn.py
```python
import cv2
import numpy as np
from collections import deque
from rknn.api import RKNN
import os

ONNX_MODEL = 'model_fp16.onnx'  
VIDEO_PATH = 'input.mp4'
MASK_PATH = 'mask.mp4'
OUTPUT_PATH = 'sim_output.mp4'

SEQ_LEN = 5
INPUT_W, INPUT_H = 640, 352  

def preprocess_uint8(frame, mask):
    f = cv2.resize(frame, (INPUT_W, INPUT_H))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    
    m = cv2.resize(mask, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)
    if len(m.shape) == 3: 
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    m = m[:, :, np.newaxis]
    return np.concatenate([f, m], axis=-1)

def finalize_input(buffer):
    data = np.concatenate(buffer, axis=-1).astype(np.float32)
    data = (data / 255.0) 
    for i in range(SEQ_LEN):
        base = i * 4
        data[:, :, base:base+3] = data[:, :, base:base+3] * 2.0 - 1.0
    # [H, W, C] -> [1, C, H, W]
    return data.transpose(2, 0, 1)[np.newaxis, ...]

def postprocess(output):
    res = output[0].transpose(1, 2, 0)
    res = (res + 1.0) / 2.0
    res = np.clip(res * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)


def main():
    if not os.path.exists(ONNX_MODEL):
        print(f"找不到 ONNX 模型: {ONNX_MODEL}"); return

    rknn = RKNN(verbose=False)

    # 必须先 config 再 load
    print("--> Config RKNN")
    rknn.config(
        target_platform='rk3588',
        # 训练时没有使用特定的 mean/std，这里保持默认
        mean_values=None,
        std_values=None
    )

    print("--> Loading ONNX model")
    if rknn.load_onnx(model=ONNX_MODEL) != 0:
        print("Load ONNX failed!"); return

    print("--> Building model for Simulator")
    if rknn.build(do_quantization=False) != 0:
        print("Build failed!"); return

    print("--> Init Runtime (Simulator)")
    if rknn.init_runtime() != 0:
        print("Init runtime failed!"); return

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap_m = cv2.VideoCapture(MASK_PATH)
    
    if not cap.isOpened() or not cap_m.isOpened():
        print("错误: 无法打开输入视频或掩码视频。"); return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (INPUT_W, INPUT_H))
    
    buffer = deque(maxlen=SEQ_LEN)
    frame_idx = 0

    print(f"--> 开始仿真推理。输出将保存至: {OUTPUT_PATH}")
    
    while True:
        ret, frame = cap.read()
        ret_m, mask = cap_m.read()
        if not ret or not ret_m: break
        
        feat = preprocess_uint8(frame, mask)
        buffer.append(feat)
        
        # 首帧填充
        if frame_idx == 0:
            for _ in range(SEQ_LEN - 1): buffer.append(feat)
            
        if len(buffer) == SEQ_LEN:
            input_data = finalize_input(list(buffer))
            
            # 推理，确保 data_format='nchw' 与上面的 transpose 对应
            outputs = rknn.inference(inputs=[input_data], data_format='nchw')
            
            final_frame = postprocess(outputs[0])
            writer.write(final_frame)
            
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"已仿真 {frame_idx} 帧...")

    print("仿真完成！")
    cap.release()
    cap_m.release()
    writer.release()
    rknn.release()

if __name__ == "__main__":
    main()
```
### 模型转换ONNX -> RKNN：post_onnx2rknn.py
```python
from rknn.api import RKNN

def convert():
    ONNX_MODEL = 'model_fp16.onnx'
    RKNN_MODEL = 'model_fp16.rknn'

    rknn = RKNN(verbose=True)
    
    # 1. 配置目标平台
    rknn.config(target_platform='rk3588')
    
    # 2. 加载 ONNX
    print('--> Loading model')
    if rknn.load_onnx(model=ONNX_MODEL) != 0:
        print('Load model failed!')
        exit()

    # 3. 构建模型 (关闭量化以使用 FP16)
    print('--> Building model')
    if rknn.build(do_quantization=False) != 0:
        print('Build model failed!')
        exit()

    # 4. 导出 RKNN
    print('--> Exporting RKNN')
    if rknn.export_rknn(RKNN_MODEL) != 0:
        print('Export failed!')
        exit()
        
    print('Done')

if __name__ == '__main__':
    convert()
```
### RK3588 板载部署：post_deploy_rk3588.py
首先要安装RKNNLite，其安装包在RKNN工具链中已经提供了多个预编译版本，直接pip安装就好

```python
import cv2
import numpy as np
from collections import deque
from rknnlite.api import RKNNLite
import time
import os


RKNN_MODEL = "model_fp16.rknn"
VIDEO_PATH = "input.mp4"
MASK_PATH = "mask.mp4"
OUTPUT_PATH = "output_rk3588.avi"
SEQ_LEN = 5
INPUT_W, INPUT_H = 640, 352


def preprocess_uint8(frame, mask):
    """保持 uint8 格式进行 Resize 和拼接 [H, W, 4]"""
    f = cv2.resize(frame, (INPUT_W, INPUT_H))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    
    m = cv2.resize(mask, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)
    if len(m.shape) == 3: 
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    m = m[:, :, np.newaxis] # [H, W, 1]
    
    # 返回 [H, W, 4]
    return np.concatenate([f, m], axis=-1)

def finalize_input_nhwc(buffer):
    # 拼接 5 帧 -> [H, W, 20]
    data = np.concatenate(buffer, axis=-1).astype(np.float32)
    
    # 归一化逻辑
    data = (data / 255.0) 
    for i in range(SEQ_LEN):
        base = i * 4
        data[:, :, base:base+3] = data[:, :, base:base+3] * 2.0 - 1.0
        
    # 返回 [1, H, W, 20]
    return data[np.newaxis, ...]

def postprocess(output):
    """
    注意：尽管输入是 NHWC，但 RKNN 的输出通常维持 NCHW [1, 3, H, W]
    这是由 ONNX 导出时的结构决定的。
    """
    # [1, 3, H, W] -> [H, W, 3]
    res = output[0].transpose(1, 2, 0)
    res = (res + 1.0) / 2.0
    res = np.clip(res * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)


def main():
    if not os.path.exists(RKNN_MODEL):
        print("Model not found!"); return

    rknn = RKNNLite()
    print("--> Loading RKNN")
    if rknn.load_rknn(RKNN_MODEL) != 0:
        print("Load failed"); return
        
    print("--> Init Runtime (3 Cores)")
    # 3核并行
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        print("Init failed"); return

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap_m = cv2.VideoCapture(MASK_PATH)
    
    if not cap.isOpened() or not cap_m.isOpened():
        print("Could not open input videos!"); return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (INPUT_W, INPUT_H))
    
    if not writer.isOpened():
        print("VideoWriter failed to open!"); return

    buffer = deque(maxlen=SEQ_LEN)
    frame_idx = 0
    start_time = time.time()

    print("--> Start Inference")
    while True:
        ret, frame = cap.read()
        ret_m, mask = cap_m.read()
        if not ret or not ret_m: break
        
        # 1. 预处理 (uint8)
        feat = preprocess_uint8(frame, mask)
        buffer.append(feat)
        
        # 首帧填充
        if frame_idx == 0:
            for _ in range(SEQ_LEN - 1): buffer.append(feat)
            
        if len(buffer) == SEQ_LEN:
            # 2. 准备最终输入 (NHWC)
            input_data = finalize_input_nhwc(list(buffer))
            
            # 3. NPU 推理
            outputs = rknn.inference(inputs=[input_data], data_format='nhwc')
            
            if outputs is None:
                print("Inference failed!"); break

            # 4. 后处理
            final_frame = postprocess(outputs[0])
            writer.write(final_frame)
            
        frame_idx += 1
        if frame_idx % 50 == 0:
            elap = time.time() - start_time
            print(f"Processed {frame_idx} frames, Current FPS: {frame_idx / elap:.2f}")

    print("Inference Done!")
    cap.release()
    cap_m.release()
    writer.release()
    rknn.release()

if __name__ == "__main__":
    main()

```
### 效果演示和示例资源下载

| 无水印原视频  |  {% dplayer "url=original.mp4" %} |
| :------------: | :------------: |
| 带水印的输入  |  {% dplayer "url=input.mp4" %} |
|  掩膜输入 |  {% dplayer "url=mask.mp4" %} |
|  PC推理输出 |  {% dplayer "url=output_pc.mp4" %} |
|  RKNN仿真输出 |  {% dplayer "url=sim_output.mp4" %} |
|  RK3588推理输出 |  {% dplayer "url=output_rk3588.mp4" %} |

~~仅仅经过两轮训练的~~[.pth模型](epoch_2.pth)、[ONNX模型](model_fp16.onnx)、[RKNN模型](model_fp16.rknn)

### 计算性能优化
#### 计算性能
根据终端输出，当前帧率仅有不足2fps：
```bash
orangepi@orangepi5:~/rknn_unet$ python3 post_deploy_rk3588.py 
--> Loading RKNN
--> Init Runtime (3 Cores)
I RKNN: [20:31:55.914] RKNN Runtime Information, librknnrt version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
I RKNN: [20:31:55.914] RKNN Driver Information, version: 0.9.8
I RKNN: [20:31:55.915] RKNN Model Information, version: 6, toolkit version: 1.5.2+b642f30c(compiler version: 1.5.2 (c6b7b351a@2023-08-23T15:34:44)), target: RKNPU v2, target platform: rk3588, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
--> Start Inference
rga_api version 1.9.3_[2]
Processed 50 frames, Current FPS: 1.30
Processed 100 frames, Current FPS: 1.31
Processed 150 frames, Current FPS: 1.30
Processed 200 frames, Current FPS: 1.29
Inference Done!
```
```bash
orangepi@orangepi5:~$ sudo cat /sys/kernel/debug/rknpu/load
NPU load:  Core0: 44%, Core1: 35%, Core2: 36%,
```
```bash
orangepi@orangepi5:~$ top
top - 20:32:25 up 33 min,  3 users,  load average: 0.36, 0.23, 0.30
Tasks: 258 total,   2 running, 256 sleeping,   0 stopped,   0 zombie
%Cpu(s):  6.2 us,  1.3 sy,  0.0 ni, 92.5 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
MiB Mem :   3735.4 total,    751.6 free,   1665.4 used,   1318.5 buff/cache
MiB Swap:   1867.7 total,   1867.2 free,      0.5 used.   1713.6 avail Mem 

    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND                                         
   7070 orangepi   1 -19 1995580 976256 372092 R  50.2  25.5   0:16.95 python3                                         
                                        
```
#### 多线程异步处理
显然主要问题在于npu没有完全处于工作状态，大量时间浪费在等待CPU预处理数据上了，因此改进加入异步处理：
```python
import cv2
import numpy as np
from collections import deque
from rknnlite.api import RKNNLite
import time
import os
import threading
from queue import Queue


RKNN_MODEL = "model_fp16.rknn"
VIDEO_PATH = "input.mp4"
MASK_PATH = "mask.mp4"
OUTPUT_PATH = "output_rk3588_pipeline.avi"
SEQ_LEN = 5
INPUT_W, INPUT_H = 640, 352

IN_QUEUE_SIZE = 4
OUT_QUEUE_SIZE = 4


def preprocess_uint8(frame, mask):
    f = cv2.resize(frame, (INPUT_W, INPUT_H))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    m = cv2.resize(mask, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)
    if len(m.shape) == 3: 
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    m = m[:, :, np.newaxis]
    return np.concatenate([f, m], axis=-1)

def finalize_input_nhwc(buffer):
    data = np.concatenate(buffer, axis=-1).astype(np.float32)
    data = (data / 255.0) 
    for i in range(SEQ_LEN):
        base = i * 4
        data[:, :, base:base+3] = data[:, :, base:base+3] * 2.0 - 1.0
    return data[np.newaxis, ...]

def postprocess(output):
    res = output[0].transpose(1, 2, 0)
    res = (res + 1.0) / 2.0
    res = np.clip(res * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)


# 生产者：读取视频并预处理
def capture_worker(in_q):
    cap_v = cv2.VideoCapture(VIDEO_PATH)
    cap_m = cv2.VideoCapture(MASK_PATH)
    
    if not cap_v.isOpened():
        print("Error: Could not open videos")
        in_q.put(None)
        return

    buffer = deque(maxlen=SEQ_LEN)
    frame_idx = 0

    while True:
        ret_v, frame = cap_v.read()
        ret_m, mask = cap_m.read()
        if not ret_v or not ret_m:
            break

        # CPU 预处理
        feat_uint8 = preprocess_uint8(frame, mask)
        buffer.append(feat_uint8)
        
        if frame_idx == 0:
            for _ in range(SEQ_LEN - 1):
                buffer.append(feat_uint8)

        if len(buffer) == SEQ_LEN:
            # 准备 NPU 输入数据
            input_data = finalize_input_nhwc(list(buffer))
            in_q.put(input_data) # 放入推理队列

        frame_idx += 1
    
    in_q.put(None) # 结束信号
    cap_v.release()
    cap_m.release()
    print("Capture thread finished.")


def inference_worker(in_q, out_q):
    rknn = RKNNLite()
    if rknn.load_rknn(RKNN_MODEL) != 0:
        print("Load RKNN failed")
        out_q.put(None)
        return
        
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        print("Init runtime failed")
        out_q.put(None)
        return

    while True:
        input_data = in_q.get()
        if input_data is None:
            break
        
        outputs = rknn.inference(inputs=[input_data], data_format="nhwc")
        
        if outputs is not None:
            out_q.put(outputs[0])
    
    out_q.put(None) # 结束信号
    rknn.release()
    print("Inference thread finished.")

# 消费者：后处理并写入视频
def writer_worker(out_q):
    cap_info = cv2.VideoCapture(VIDEO_PATH)
    fps = cap_info.get(cv2.CAP_PROP_FPS)
    width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0: fps = 25
    cap_info.release()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()

    while True:
        npu_output = out_q.get()
        if npu_output is None:
            break
        
        # CPU 后处理
        final_frame = postprocess(npu_output)
        # 恢复原尺寸写入
        final_frame = cv2.resize(final_frame, (width, height))
        writer.write(final_frame)
        
        frame_count += 1
        if frame_count % 50 == 0:
            avg_fps = frame_count / (time.time() - start_time)
            print(f"FPS: {avg_fps:.2f} | Processed: {frame_count}")

    writer.release()
    print(f"Writer thread finished. Saved to {OUTPUT_PATH}")


def main():
    if not os.path.exists(RKNN_MODEL):
        print(f"Model {RKNN_MODEL} not found!")
        return

    q_in = Queue(maxsize=IN_QUEUE_SIZE)
    q_out = Queue(maxsize=OUT_QUEUE_SIZE)

    t_cap = threading.Thread(target=capture_worker, args=(q_in,))
    t_inf = threading.Thread(target=inference_worker, args=(q_in, q_out))
    t_wri = threading.Thread(target=writer_worker, args=(q_out,))

    print("--> Pipeline Starting")
    start_all = time.time()

    t_cap.start()
    t_inf.start()
    t_wri.start()

    # 等待所有线程完成
    t_cap.join()
    t_inf.join()
    t_wri.join()

    total_time = time.time() - start_all
    print(f"Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
```
修改之后帧率从1.3提升至1.6，npu占用从不到40%提升到超过50%。同时CPU占用也大幅提高。
```bash
orangepi@orangepi5:~/rknn_unet$ python3 post_deploy_rk3588_queue.py 
--> Pipeline Starting
rga_api version 1.9.3_[2]
I RKNN: [21:12:16.441] RKNN Runtime Information, librknnrt version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
I RKNN: [21:12:16.441] RKNN Driver Information, version: 0.9.8
I RKNN: [21:12:16.442] RKNN Model Information, version: 6, toolkit version: 1.5.2+b642f30c(compiler version: 1.5.2 (c6b7b351a@2023-08-23T15:34:44)), target: RKNPU v2, target platform: rk3588, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
FPS: 1.64 | Processed: 50
FPS: 1.65 | Processed: 100
FPS: 1.64 | Processed: 150
FPS: 1.65 | Processed: 200
Capture thread finished.
Writer thread finished. Saved to output_rk3588_pipeline.avi
Inference thread finished.
Total processing time: 150.56 seconds
```
```bash
orangepi@orangepi5:~$ sudo cat /sys/kernel/debug/rknpu/load
NPU load:  Core0: 61%, Core1: 50%, Core2: 50%,
```
```bash
orangepi@orangepi5:~$ top
top - 21:15:58 up  1:17,  3 users,  load average: 1.49, 0.88, 0.40
Tasks: 266 total,   1 running, 265 sleeping,   0 stopped,   0 zombie
%Cpu(s):  9.9 us,  1.2 sy,  0.0 ni, 88.9 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
MiB Mem :   3735.4 total,    330.6 free,   2247.0 used,   1157.8 buff/cache
MiB Swap:   1867.7 total,   1859.0 free,      8.8 used.   1126.9 avail Mem 

    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND                                         
   8092 orangepi  20   0 2920008   1.5g 377856 S  78.9  41.2   0:54.46 python3
```
#### 多进程异步
到多线程的提升并没有我想象中的明显，因此可能是多线程受限于 Python 的 GIL 锁无法充分利用多核 CPU？因此尝试了多进程。但是测试表明性能不升反降，说明现在的瓶颈已经不在于CPU对数据的处理，而在于NPU受限于内存带宽依然在等待数据。此时想要减少内存IO只能修改模型规模或者INT8量化了
```bash
FPS: 1.49 | Count: 50
FPS: 1.49 | Count: 100
FPS: 1.50 | Count: 150
FPS: 1.49 | Count: 200
```
```bash
orangepi@orangepi5:~$ top
top - 21:24:43 up  1:25,  3 users,  load average: 0.53, 0.39, 0.36
Tasks: 259 total,   2 running, 257 sleeping,   0 stopped,   0 zombie
%Cpu(s):  9.2 us,  1.2 sy,  0.0 ni, 89.4 id,  0.0 wa,  0.0 hi,  0.1 si,  0.0 st
MiB Mem :   3735.4 total,    979.8 free,   1584.8 used,   1170.8 buff/cache
MiB Swap:   1867.7 total,   1859.0 free,      8.8 used.   1796.5 avail Mem 

    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND                                         
   8513 orangepi   1 -19  925252 615056 324344 R  46.8  16.1   0:09.10 python3                                         
   8514 orangepi  20   0  894128 142512  70432 S  18.3   3.7   0:04.35 python3                                         
   8512 orangepi  20   0 1306848 222452 106436 S  10.0   5.8   0:02.90 python3
```
```bash
orangepi@orangepi5:~$ sudo cat /sys/kernel/debug/rknpu/load
NPU load:  Core0: 46%, Core1: 37%, Core2: 37%,
```
#### INT8量化
警告：下面的代码有问题，输出偏色，依然在修复中，请勿使用，等后面项目做完再回来搞量化的事。这一节后面的实验证明瓶颈依然在CPU，暂时量化不是重点

首先需要生成npy格式的校准数据集：
```python
import cv2
import numpy as np
import os
import random

# --- 配置 ---
WATERMARK_DIR = r"D:\Dataset\watermarked_videos"
MASK_DIR = r"D:\Dataset\mask_videos"
CALIB_SAVE_DIR = "./calibration_data"
DATASET_TEXT = "dataset.txt"
SEQ_LEN = 5
INPUT_W, INPUT_H = 640, 352  # (W, H)
SAMPLE_COUNT = 100

os.makedirs(CALIB_SAVE_DIR, exist_ok=True)


def preprocess_uint8(frame, mask):
    f = cv2.resize(frame, (INPUT_W, INPUT_H))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    m = cv2.resize(mask, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)
    if len(m.shape) == 3: m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    m = m[:, :, np.newaxis]
    return np.concatenate([f, m], axis=-1)  # [H, W, 4] uint8


def generate_calibration():
    video_files = [f for f in os.listdir(WATERMARK_DIR) if f.endswith('.mp4')]
    random.shuffle(video_files)

    count = 0
    with open(DATASET_TEXT, 'w') as f_txt:
        for v_name in video_files:
            if count >= SAMPLE_COUNT:
                break

            v_path = os.path.join(WATERMARK_DIR, v_name)
            m_path = os.path.join(MASK_DIR, v_name)

            cap_v = cv2.VideoCapture(v_path)
            cap_m = cv2.VideoCapture(m_path)

            # 随机跳过前面的帧
            for _ in range(random.randint(5, 20)):
                cap_v.grab()
                cap_m.grab()

            buffer = []
            for _ in range(SEQ_LEN):
                ret_v, frame = cap_v.read()
                ret_m, mask = cap_m.read()
                if not ret_v or not ret_m: break
                buffer.append(preprocess_uint8(frame, mask))

            cap_v.release()
            cap_m.release()

            if len(buffer) == SEQ_LEN:
                # 1. 拼接为 [H, W, 20]
                data = np.concatenate(buffer, axis=-1)

                # NHWC -> NCHW
                # 将通道移到前面: [20, H, W]
                data = data.transpose(2, 0, 1)
                # 增加 Batch 维度[1, 20, H, W]
                data = data[np.newaxis, ...]

                npy_name = f"sample_{count:03d}.npy"
                npy_path = os.path.abspath(os.path.join(CALIB_SAVE_DIR, npy_name))
                np.save(npy_path, data)

                f_txt.write(npy_path + '\n')
                count += 1
                print(f"Generated {npy_name} with shape {data.shape}")

    print(f"已生成 {count} 个校准样本。")


if __name__ == "__main__":
    generate_calibration()

```
然后使用rknn的量化工具进行量化得到INT8模型：
```python
from rknn.api import RKNN
import os

ONNX_MODEL = 'model_fp16.onnx'
RKNN_MODEL = 'model_int8.rknn'
DATASET_TEXT = './dataset.txt'

def convert():
    rknn = RKNN(verbose=False)

    print('--> Config RKNN')
    # 这里的 means 和 stds 会按照通道顺序应用,因为输入是 [R1,G1,B1,M1, R2,G2,B2,M2...],统一设置 127.5
    channel_count = 20
    means = [[127.5] * channel_count]
    stds = [[127.5] * channel_count]

    rknn.config(
        target_platform='rk3588',
        mean_values=means,
        std_values=stds,
        quantized_dtype='asymmetric_quantized-8',
        quantized_algorithm='normal',
        # 开启优化，减少 NPU 内部的数据搬运
        optimization_level=3 
    )

    print('--> Loading ONNX model')
    if rknn.load_onnx(model=ONNX_MODEL) != 0:
        print("Load failed!"); return

    print('--> Building model (INT8)')
    # 现在 dataset 里的 npy 是 (1, 20, 352, 640)，符合工具要求
    if rknn.build(do_quantization=True, dataset=DATASET_TEXT) != 0:
        print("Build failed!"); return

    print('--> Exporting RKNN')
    rknn.export_rknn(RKNN_MODEL)
    print('Done!')

if __name__ == '__main__':
    convert()
```
编写新的推理代码，和之前的改动不大，主要是输入输出数据类型的变化，现在是 -128 到 127 的原始整数（INT8）

十分尴尬的是，这段代码执行的帧率并没有变化，依然是1.6，且NPU占用降低到了25%左右，CPU占用保持80%左右不变。然而当我把视频输出resize的代码注释掉之后帧率提升到了1.8，显然其实瓶颈还是在CPU对数据的预处理
```python
import cv2
import numpy as np
from collections import deque
from rknnlite.api import RKNNLite
import time
import os


RKNN_MODEL = "model_int8.rknn" 
VIDEO_PATH = "input.mp4"
MASK_PATH = "mask.mp4"
OUTPUT_PATH = "output_rk3588_int8.avi"

SEQ_LEN = 5
INPUT_W, INPUT_H = 640, 352
TARGET_FRAME_IDX = 2


def preprocess_int8(frame, mask):
    """
    返回 [H, W, 4] uint8
    """
    f = cv2.resize(frame, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    
    # Resize Mask 并严格二值化
    m = cv2.resize(mask, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)
    if len(m.shape) == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    m = m[:, :, np.newaxis] 
    
    # 拼接 RGB(3) + Mask(1) = 4通道 uint8
    return np.concatenate([f, m], axis=-1)

def postprocess_int8(output_tensor, original_size):
    """
    output_tensor: NPU输出的 int8 类型数组 (范围 -128 到 127)
    """
    # 1. 将 int8 转换为 float32 并执行反量化 (还原到 -1.0 ~ 1.0)
    # 因为模型最后是 Tanh，量化后的 127 对应 1.0
    res = output_tensor[0].astype(np.float32) / 127.0
    
    # 2. 维度转换 [3, H, W] -> [H, W, 3]
    res = res.transpose(1, 2, 0)
    
    # 3. 反归一化：将 [-1, 1] 映射回 [0, 255] 像素空间
    res = (res + 1.0) / 2.0
    res = np.clip(res * 255, 0, 255).astype(np.uint8)
    
    # 4. 颜色空间转换 BGR 并拉伸回原视频尺寸
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    return cv2.resize(res, original_size, interpolation=cv2.INTER_LINEAR)


def main():
    if not os.path.exists(RKNN_MODEL):
        print(f"Error: 找不到模型文件 {RKNN_MODEL}")
        return

    rknn = RKNNLite()
    if rknn.load_rknn(RKNN_MODEL) != 0:
        print("加载模型失败"); return

    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        print("初始化失败"); return

    cap_v = cv2.VideoCapture(VIDEO_PATH)
    cap_m = cv2.VideoCapture(MASK_PATH)
    
    if not cap_v.isOpened():
        print("错误: 无法打开视频文件"); return

    fps = cap_v.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0: fps = 25

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_w, orig_h))

    buffer = deque(maxlen=SEQ_LEN)
    frame_idx = 0
    start_time = time.time()

    print("--> 开始 NPU 推理...")
    
    while True:
        ret_v, frame = cap_v.read()
        ret_m, mask = cap_m.read()
        if not ret_v or not ret_m:
            break

        feat_uint8 = preprocess_int8(frame, mask)
        buffer.append(feat_uint8)
        
        if frame_idx == 0:
            for _ in range(SEQ_LEN - 1):
                buffer.append(feat_uint8)

        if len(buffer) == SEQ_LEN:
            # NPU 输入：[1, H, W, 20] uint8
            input_data = np.concatenate(list(buffer), axis=-1)
            input_data = input_data[np.newaxis, ...]

            # NPU 推理inputs 传入 uint8，RKNN 内部节点自动执行归一化减法
            outputs = rknn.inference(inputs=[input_data], data_format='nhwc')
            
            if outputs is None:
                print("推理出错!"); break

            # 将 int8 转换回像素
            final_frame = postprocess_int8(outputs[0], (orig_w, orig_h))
            writer.write(final_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            print(f"帧数: {frame_idx} | 平均 FPS: {frame_idx / elapsed:.2f}")

    cap_v.release()
    cap_m.release()
    writer.release()
    rknn.release()
    
    total_time = time.time() - start_time
    print(f"\n任务完成！")
    print(f"总处理帧数: {frame_idx}")
    print(f"最终平均帧率: {frame_idx / total_time:.2f} FPS")

if __name__ == "__main__":
    main()
    
```

## 项目实践：实时水下图像恢复的模型板载部署

