---
title: 香橙派5（RK3588）AI模型推理
cover: /img/pi5-01.png
date: 2026-1-11 20:12:14
categories: 
- [文档&笔记]
tags:
- 笔记
- 香橙派
- python
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
由于项目代码等内容不宜公开，且模型转换和部署方式和上面相同，因此不再描述。现在模型已经部署完成，但是单帧推理256*256输入板载单个npu用时900ms，三个npu并行耗时700ms，必须对推理速度进行优化，目标是达到1080p 30FPS。
### 在onnx2rknn查看算子推理设备
由于模型很多算子并不支持直接由npu执行，这部分操作被退回至CPU执行，因此消耗了大量时间。要优化速度首先要知道算子在哪个设备上执行的，以及具体时间消耗是多少。

在使用rknn-toolkit2进行转换时，就已经会打印出算子计算设备：
```python
rknn = RKNN(verbose=True) # 这里设为 True 会打印最底层的转换细节
# 一些代码...
rknn.build() # 控制台会输出一个关于模型的表格
```
我这里表格内容如下：
```
D RKNN: [04:33:42.488] RKNNModelBuildPass: [Statistics]
D RKNN: [04:33:42.488] total_regcfg_size     :    742928
D RKNN: [04:33:42.488] total_diff_regcfg_size:    181328
D RKNN: [04:33:42.488] ID   OpType           DataType Target InputShape                                   OutputShape            DDR Cycles     NPU Cycles     Total Cycles   Time(us)       MacUsage(%)    Task Number    Lut Number     RW(KB)         FullName        
D RKNN: [04:33:42.488] 0    InputOperator    FLOAT16  CPU    \                                            (1,3,256,256)          0              0              0              0              \              0              0              384.00         InputOperator:input
D RKNN: [04:33:42.488] 1    Conv             FLOAT16  NPU    (1,3,256,256),(3,3,1,1)                      (1,3,256,256)          0              0              0              0              \              0              0              1408.05                        
D RKNN: [04:33:42.488] 2    Pad              FLOAT16  CPU    (1,3,256,256),(8)                            (1,3,258,258)          0              0              0              0              \              0              0              2064.12        Pad:/pr_encoder/conv1/conv/conv.0/Pad
D RKNN: [04:33:42.488] 3    ConvLeakyRelu    FLOAT16  NPU    (1,3,258,258),(64,3,3,3),(64)                (1,64,256,256)         0              0              0              0              \              0              0              9241.31        Conv:/pr_encoder/conv1/conv/conv.1/Conv
D RKNN: [04:33:42.488] 4    Pad              FLOAT16  CPU    (1,64,256,256),(8)                           (1,64,258,258)         0              0              0              0              \              0              0              16512.56       Pad:/pr_encoder/conv1/conv/conv.4/Pad
D RKNN: [04:33:42.488] 5    ConvLeakyRelu    FLOAT16  NPU    (1,64,258,258),(64,64,3,3),(64)              (1,64,256,256)         0              0              0              0              \              0              0              16584.75       Conv:/pr_encoder/conv1/conv/conv.5/Conv
D RKNN: [04:33:42.488] 6    MaxPool          FLOAT16  NPU    (1,64,256,256)                               (1,64,128,128)         0              0              0              0              \              0              0              10240.00       MaxPool:/pr_encoder/pool1/MaxPool
D RKNN: [04:33:42.488] 7    Pad              FLOAT16  CPU    (1,64,128,128),(8)                           (1,64,130,130)         0              0              0              0              \              0              0              4160.56        Pad:/pr_encoder/conv2/conv/conv.0/Pad
D RKNN: [04:33:42.488] 8    ConvLeakyRelu    FLOAT16  NPU    (1,64,130,130),(64,64,3,3),(64)              (1,64,128,128)         0              0              0              0              \              0              0              4232.75        Conv:/pr_encoder/conv2/conv/conv.1/Conv
D RKNN: [04:33:42.488] 9    Pad              FLOAT16  CPU    (1,64,128,128),(8)                           (1,64,130,130)         0              0              0              0              \              0              0              4160.56        Pad:/pr_encoder/conv2/conv/conv.4/Pad
D RKNN: [04:33:42.488] 10   ConvLeakyRelu    FLOAT16  NPU    (1,64,130,130),(64,64,3,3),(64)              (1,64,128,128)         0              0              0              0              \              0              0              4232.75        Conv:/pr_encoder/conv2/conv/conv.5/Conv
D RKNN: [04:33:42.488] 11   MaxPool          FLOAT16  NPU    (1,64,128,128)                               (1,64,64,64)           0              0              0              0              \              0              0              2560.00        MaxPool:/pr_encoder/pool2/MaxPool
D RKNN: [04:33:42.488] 12   Pad              FLOAT16  CPU    (1,64,64,64),(8)                             (1,64,66,66)           0              0              0              0              \              0              0              1056.56        Pad:/pr_encoder/conv3/conv/conv.0/Pad
D RKNN: [04:33:42.488] 13   ConvLeakyRelu    FLOAT16  NPU    (1,64,66,66),(64,64,3,3),(64)                (1,64,64,64)           0              0              0              0              \              0              0              1128.75        Conv:/pr_encoder/conv3/conv/conv.1/Conv
D RKNN: [04:33:42.488] 14   Pad              FLOAT16  CPU    (1,64,64,64),(8)                             (1,64,66,66)           0              0              0              0              \              0              0              1056.56        Pad:/pr_encoder/conv3/conv/conv.4/Pad
D RKNN: [04:33:42.488] 15   ConvLeakyRelu    FLOAT16  NPU    (1,64,66,66),(64,64,3,3),(64)                (1,64,64,64)           0              0              0              0              \              0              0              1128.75        Conv:/pr_encoder/conv3/conv/conv.5/Conv
D RKNN: [04:33:42.488] 16   MaxPool          FLOAT16  NPU    (1,64,64,64)                                 (1,64,32,32)           0              0              0              0              \              0              0              640.00         MaxPool:/pr_encoder/pool3/MaxPool
D RKNN: [04:33:42.488] 17   Pad              FLOAT16  CPU    (1,64,32,32),(8)                             (1,64,34,34)           0              0              0              0              \              0              0              272.56         Pad:/pr_encoder/conv4/body/body.0/Pad
D RKNN: [04:33:42.488] 18   ConvLeakyRelu    FLOAT16  NPU    (1,64,34,34),(64,64,3,3),(64)                (1,64,32,32)           0              0              0              0              \              0              0              344.75         Conv:/pr_encoder/conv4/body/body.1/Conv
D RKNN: [04:33:42.488] 19   Pad              FLOAT16  CPU    (1,64,32,32),(8)                             (1,64,34,34)           0              0              0              0              \              0              0              272.56         Pad:/pr_encoder/conv4/body/body.3/Pad
D RKNN: [04:33:42.488] 20   Conv             FLOAT16  NPU    (1,64,34,34),(64,64,3,3),(64)                (1,64,32,32)           0              0              0              0              \              0              0              344.75         Conv:/pr_encoder/conv4/body/body.4/Conv
D RKNN: [04:33:42.488] 21   Conv             FLOAT16  NPU    (1,64,32,32),(1,64,7,7),(64)                 (1,64,5,5)             0              0              0              0              \              0              0              137.50         Conv:/pr_encoder/conv4/body/body.5/global_pool/GlobalAveragePool_2conv_0
D RKNN: [04:33:42.488] 22   Conv             FLOAT16  NPU    (1,64,5,5),(1,64,5,5),(64)                   (1,64,1,1)             0              0              0              0              \              0              0              6.62           Conv:/pr_encoder/conv4/body/body.5/global_pool/GlobalAveragePool_output_0
D RKNN: [04:33:42.488] 23   ConvLeakyRelu    FLOAT16  NPU    (1,64,1,1),(4,64,1,1),(4)                    (1,4,1,1)              0              0              0              0              \              0              0              0.70           Conv:/pr_encoder/conv4/body/body.5/conv_double/conv_double.0/Conv
D RKNN: [04:33:42.488] 24   Conv             FLOAT16  NPU    (1,4,1,1),(64,4,1,1),(64)                    (1,64,1,1)             0              0              0              0              \              0              0              1.39           Conv:/pr_encoder/conv4/body/body.5/conv_double/conv_double.2/Conv
D RKNN: [04:33:42.488] 25   Sigmoid          FLOAT16  NPU    (1,64,1,1)                                   (1,64,1,1)             0              0              0              0              \              0              0              0.25           Sigmoid:/pr_encoder/conv4/body/body.5/conv_double/conv_double.3/Sigmoid
D RKNN: [04:33:42.488] 26   Mul              FLOAT16  NPU    (1,64,32,32),(1,64,1,1)                      (1,64,32,32)           0              0              0              0              \              0              0              256.12         Mul:/pr_encoder/conv4/body/body.5/Mul
D RKNN: [04:33:42.488] 27   Add              FLOAT16  NPU    (1,64,32,32),(1,64,32,32)                    (1,64,32,32)           0              0              0              0              \              0              0              384.00         Add:/pr_encoder/conv4/Add
D RKNN: [04:33:42.488] 28   Pad              FLOAT16  CPU    (1,64,32,32),(8)                             (1,64,34,34)           0              0              0              0              \              0              0              272.56         Pad:/pr_conv/body/body.0/Pad
D RKNN: [04:33:42.488] 29   ConvLeakyRelu    FLOAT16  NPU    (1,64,34,34),(64,64,3,3),(64)                (1,64,32,32)           0              0              0              0              \              0              0              344.75         Conv:/pr_conv/body/body.1/Conv
D RKNN: [04:33:42.488] 30   Pad              FLOAT16  CPU    (1,64,32,32),(8)                             (1,64,34,34)           0              0              0              0              \              0              0              272.56         Pad:/pr_conv/body/body.3/Pad
D RKNN: [04:33:42.488] 31   Conv             FLOAT16  NPU    (1,64,34,34),(64,64,3,3),(64)                (1,64,32,32)           0              0              0              0              \              0              0              344.75         Conv:/pr_conv/body/body.4/Conv
D RKNN: [04:33:42.488] 32   Conv             FLOAT16  NPU    (1,64,32,32),(1,64,7,7),(64)                 (1,64,5,5)             0              0              0              0              \              0              0              137.50         Conv:/pr_conv/body/body.5/global_pool/GlobalAveragePool_2conv_0
D RKNN: [04:33:42.488] 33   Conv             FLOAT16  NPU    (1,64,5,5),(1,64,5,5),(64)                   (1,64,1,1)             0              0              0              0              \              0              0              6.62           Conv:/pr_conv/body/body.5/global_pool/GlobalAveragePool_output_0
D RKNN: [04:33:42.488] 34   ConvLeakyRelu    FLOAT16  NPU    (1,64,1,1),(4,64,1,1),(4)                    (1,4,1,1)              0              0              0              0              \              0              0              0.70           Conv:/pr_conv/body/body.5/conv_double/conv_double.0/Conv
D RKNN: [04:33:42.488] 35   Conv             FLOAT16  NPU    (1,4,1,1),(64,4,1,1),(64)                    (1,64,1,1)             0              0              0              0              \              0              0              1.39           Conv:/pr_conv/body/body.5/conv_double/conv_double.2/Conv
D RKNN: [04:33:42.488] 36   Sigmoid          FLOAT16  NPU    (1,64,1,1)                                   (1,64,1,1)             0              0              0              0              \              0              0              0.25           Sigmoid:/pr_conv/body/body.5/conv_double/conv_double.3/Sigmoid
D RKNN: [04:33:42.488] 37   Mul              FLOAT16  NPU    (1,64,32,32),(1,64,1,1)                      (1,64,32,32)           0              0              0              0              \              0              0              256.12         Mul:/pr_conv/body/body.5/Mul
D RKNN: [04:33:42.488] 38   Add              FLOAT16  NPU    (1,64,32,32),(1,64,32,32)                    (1,64,32,32)           0              0              0              0              \              0              0              384.00         Add:/pr_conv/Add
D RKNN: [04:33:42.488] 39   Resize           FLOAT16  CPU    (1,64,32,32),(0),(4)                         (1,64,64,64)           0              0              0              0              \              0              0              640.02         Resize:/pr_Up3/up/up.0/Resize
D RKNN: [04:33:42.488] 40   Concat           FLOAT16  NPU    (1,64,64,64),(1,64,64,64)                    (1,128,64,64)          0              0              0              0              \              0              0              2048.00        Concat:/Concat  
D RKNN: [04:33:42.488] 41   Pad              FLOAT16  CPU    (1,128,64,64),(8)                            (1,128,66,66)          0              0              0              0              \              0              0              2113.06        Pad:/pr_UpConv3/conv/conv.0/Pad
D RKNN: [04:33:42.488] 42   ConvLeakyRelu    FLOAT16  NPU    (1,128,66,66),(64,128,3,3),(64)              (1,64,64,64)           0              0              0              0              \              0              0              1745.25        Conv:/pr_UpConv3/conv/conv.1/Conv
D RKNN: [04:33:42.488] 43   Pad              FLOAT16  CPU    (1,64,64,64),(8)                             (1,64,66,66)           0              0              0              0              \              0              0              1056.56        Pad:/pr_UpConv3/conv/conv.4/Pad
D RKNN: [04:33:42.488] 44   ConvLeakyRelu    FLOAT16  NPU    (1,64,66,66),(64,64,3,3),(64)                (1,64,64,64)           0              0              0              0              \              0              0              1128.75        Conv:/pr_UpConv3/conv/conv.5/Conv
D RKNN: [04:33:42.488] 45   Resize           FLOAT16  CPU    (1,64,64,64),(0),(4)                         (1,64,128,128)         0              0              0              0              \              0              0              2560.02        Resize:/pr_Up2/up/up.0/Resize
D RKNN: [04:33:42.488] 46   Concat           FLOAT16  NPU    (1,64,128,128),(1,64,128,128)                (1,128,128,128)        0              0              0              0              \              0              0              8192.00        Concat:/Concat_1
D RKNN: [04:33:42.488] 47   Pad              FLOAT16  CPU    (1,128,128,128),(8)                          (1,128,130,130)        0              0              0              0              \              0              0              8321.06        Pad:/pr_UpConv2/conv/conv.0/Pad
D RKNN: [04:33:42.488] 48   ConvLeakyRelu    FLOAT16  NPU    (1,128,130,130),(64,128,3,3),(64)            (1,64,128,128)         0              0              0              0              \              0              0              6417.25        Conv:/pr_UpConv2/conv/conv.1/Conv
D RKNN: [04:33:42.488] 49   Pad              FLOAT16  CPU    (1,64,128,128),(8)                           (1,64,130,130)         0              0              0              0              \              0              0              4160.56        Pad:/pr_UpConv2/conv/conv.4/Pad
D RKNN: [04:33:42.488] 50   ConvLeakyRelu    FLOAT16  NPU    (1,64,130,130),(64,64,3,3),(64)              (1,64,128,128)         0              0              0              0              \              0              0              4232.75        Conv:/pr_UpConv2/conv/conv.5/Conv
D RKNN: [04:33:42.488] 51   Resize           FLOAT16  CPU    (1,64,128,128),(0),(4)                       (1,64,256,256)         0              0              0              0              \              0              0              10240.02       Resize:/pr_Up1/up/up.0/Resize
D RKNN: [04:33:42.488] 52   Concat           FLOAT16  NPU    (1,64,256,256),(1,64,256,256)                (1,128,256,256)        0              0              0              0              \              0              0              32768.00       Concat:/Concat_2
D RKNN: [04:33:42.488] 53   Conv             FLOAT16  NPU    (1,128,256,256),(1,128,7,7),(128)            (1,128,37,37)          0              0              0              0              \              0              0              16739.00       Conv:/gap/GlobalAveragePool_2conv_0
D RKNN: [04:33:42.488] 54   Conv             FLOAT16  NPU    (1,128,37,37),(1,128,7,7),(128)              (1,128,6,6)            0              0              0              0              \              0              0              364.00         Conv:/gap/GlobalAveragePool_2conv_1
D RKNN: [04:33:42.488] 55   Conv             FLOAT16  NPU    (1,128,6,6),(1,128,6,6),(128)                (1,128,1,1)            0              0              0              0              \              0              0              18.75          Conv:/gap/GlobalAveragePool_output_0
D RKNN: [04:33:42.488] 56   Conv             FLOAT16  NPU    (1,128,1,1),(40,128,1,1),(40)                (1,40,1,1)             0              0              0              0              \              0              0              10.52          Conv:/u_conv_layer/Conv
D RKNN: [04:33:42.488] 57   Slice            FLOAT16  NPU    (1,40,1,1),(1),(1),(1),(1)                   (1,20,1,1)             0              0              0              0              \              0              0              0.16           Slice:/Slice    
D RKNN: [04:33:42.488] 58   Conv             FLOAT16  NPU    (1,20,1,1),(128,20,1,1),(128)                (1,128,1,1)            0              0              0              0              \              0              0              8.80           Conv:/conv_u/Conv
D RKNN: [04:33:42.488] 59   Conv             FLOAT16  NPU    (1,128,1,1),(40,128,1,1),(40)                (1,40,1,1)             0              0              0              0              \              0              0              10.52          Conv:/s_conv_layer/Conv
D RKNN: [04:33:42.488] 60   Slice            FLOAT16  NPU    (1,40,1,1),(1),(1),(1),(1)                   (1,20,1,1)             0              0              0              0              \              0              0              0.16           Slice:/Slice_1  
D RKNN: [04:33:42.488] 61   ConvLeakyRelu    FLOAT16  NPU    (1,20,1,1),(128,20,1,1),(128)                (1,128,1,1)            0              0              0              0              \              0              0              8.80           Conv:/conv_s/Conv
D RKNN: [04:33:42.488] 62   Reshape          FLOAT16  CPU    (1,128,256,256),(4)                          (128,65536,1,1)        0              0              0              0              \              0              0              32768.03       Reshape:/insnorm/InstanceNormalization_2ln_reshape1
D RKNN: [04:33:42.488] 63   exLayerNorm      FLOAT16  NPU    (128,65536,1,1),(1,65536,1,1)                (128,65536,1,1)        0              0              0              0              \              0              0              32896.00       exLayerNorm:/insnorm/InstanceNormalization_2ln
D RKNN: [04:33:42.488] 64   Reshape          FLOAT16  CPU    (128,65536,1,1),(4)                          (1,128,256,256)        0              0              0              0              \              0              0              32768.03       Reshape:/insnorm/InstanceNormalization_2ln_reshape2
D RKNN: [04:33:42.488] 65   Mul              FLOAT16  NPU    (1,128,256,256),(1,128,1,1)                  (1,128,256,256)        0              0              0              0              \              0              0              32768.25       Mul:/Mul        
D RKNN: [04:33:42.488] 66   Add              FLOAT16  NPU    (1,128,256,256),(1,128,1,1)                  (1,128,256,256)        0              0              0              0              \              0              0              32768.25       Add:/Add        
D RKNN: [04:33:42.488] 67   Pad              FLOAT16  CPU    (1,128,256,256),(8)                          (1,128,258,258)        0              0              0              0              \              0              0              33025.06       Pad:/pr_UpConv1/conv/conv.0/Pad
D RKNN: [04:33:42.488] 68   ConvLeakyRelu    FLOAT16  NPU    (1,128,258,258),(64,128,3,3),(64)            (1,64,256,256)         0              0              0              0              \              0              0              24977.25       Conv:/pr_UpConv1/conv/conv.1/Conv
D RKNN: [04:33:42.488] 69   Pad              FLOAT16  CPU    (1,64,256,256),(8)                           (1,64,258,258)         0              0              0              0              \              0              0              16512.56       Pad:/pr_UpConv1/conv/conv.4/Pad
D RKNN: [04:33:42.488] 70   ConvLeakyRelu    FLOAT16  NPU    (1,64,258,258),(64,64,3,3),(64)              (1,64,256,256)         0              0              0              0              \              0              0              16584.75       Conv:/pr_UpConv1/conv/conv.5/Conv
D RKNN: [04:33:42.488] 71   Pad              FLOAT16  CPU    (1,64,256,256),(8)                           (1,64,258,258)         0              0              0              0              \              0              0              16512.56       Pad:/out_conv/out_conv.0/body/body.0/Pad
D RKNN: [04:33:42.488] 72   ConvLeakyRelu    FLOAT16  NPU    (1,64,258,258),(64,64,3,3),(64)              (1,64,256,256)         0              0              0              0              \              0              0              16584.75       Conv:/out_conv/out_conv.0/body/body.1/Conv
D RKNN: [04:33:42.488] 73   Pad              FLOAT16  CPU    (1,64,256,256),(8)                           (1,64,258,258)         0              0              0              0              \              0              0              16512.56       Pad:/out_conv/out_conv.0/body/body.3/Pad
D RKNN: [04:33:42.488] 74   Conv             FLOAT16  NPU    (1,64,258,258),(64,64,3,3),(64)              (1,64,256,256)         0              0              0              0              \              0              0              16584.75       Conv:/out_conv/out_conv.0/body/body.4/Conv
D RKNN: [04:33:42.488] 75   Conv             FLOAT16  NPU    (1,64,256,256),(1,64,7,7),(64)               (1,64,37,37)           0              0              0              0              \              0              0              8369.50        Conv:/out_conv/out_conv.0/body/body.5/global_pool/GlobalAveragePool_2conv_0
D RKNN: [04:33:42.488] 76   Conv             FLOAT16  NPU    (1,64,37,37),(1,64,7,7),(64)                 (1,64,6,6)             0              0              0              0              \              0              0              182.00         Conv:/out_conv/out_conv.0/body/body.5/global_pool/GlobalAveragePool_2conv_1
D RKNN: [04:33:42.488] 77   Conv             FLOAT16  NPU    (1,64,6,6),(1,64,6,6),(64)                   (1,64,1,1)             0              0              0              0              \              0              0              9.38           Conv:/out_conv/out_conv.0/body/body.5/global_pool/GlobalAveragePool_output_0
D RKNN: [04:33:42.488] 78   ConvLeakyRelu    FLOAT16  NPU    (1,64,1,1),(4,64,1,1),(4)                    (1,4,1,1)              0              0              0              0              \              0              0              0.70           Conv:/out_conv/out_conv.0/body/body.5/conv_double/conv_double.0/Conv
D RKNN: [04:33:42.488] 79   Conv             FLOAT16  NPU    (1,4,1,1),(64,4,1,1),(64)                    (1,64,1,1)             0              0              0              0              \              0              0              1.39           Conv:/out_conv/out_conv.0/body/body.5/conv_double/conv_double.2/Conv
D RKNN: [04:33:42.488] 80   Sigmoid          FLOAT16  NPU    (1,64,1,1)                                   (1,64,1,1)             0              0              0              0              \              0              0              0.25           Sigmoid:/out_conv/out_conv.0/body/body.5/conv_double/conv_double.3/Sigmoid
D RKNN: [04:33:42.488] 81   Mul              FLOAT16  NPU    (1,64,256,256),(1,64,1,1)                    (1,64,256,256)         0              0              0              0              \              0              0              16384.12       Mul:/out_conv/out_conv.0/body/body.5/Mul
D RKNN: [04:33:42.488] 82   Add              FLOAT16  NPU    (1,64,256,256),(1,64,256,256)                (1,64,256,256)         0              0              0              0              \              0              0              24576.00       Add:/out_conv/out_conv.0/Add
D RKNN: [04:33:42.488] 83   Pad              FLOAT16  CPU    (1,64,256,256),(8)                           (1,64,258,258)         0              0              0              0              \              0              0              16512.56       Pad:/out_conv/out_conv.1/body/body.0/Pad
D RKNN: [04:33:42.488] 84   ConvLeakyRelu    FLOAT16  NPU    (1,64,258,258),(64,64,3,3),(64)              (1,64,256,256)         0              0              0              0              \              0              0              16584.75       Conv:/out_conv/out_conv.1/body/body.1/Conv
D RKNN: [04:33:42.488] 85   Pad              FLOAT16  CPU    (1,64,256,256),(8)                           (1,64,258,258)         0              0              0              0              \              0              0              16512.56       Pad:/out_conv/out_conv.1/body/body.3/Pad
D RKNN: [04:33:42.488] 86   Conv             FLOAT16  NPU    (1,64,258,258),(64,64,3,3),(64)              (1,64,256,256)         0              0              0              0              \              0              0              16584.75       Conv:/out_conv/out_conv.1/body/body.4/Conv
D RKNN: [04:33:42.488] 87   Conv             FLOAT16  NPU    (1,64,256,256),(1,64,7,7),(64)               (1,64,37,37)           0              0              0              0              \              0              0              8369.50        Conv:/out_conv/out_conv.1/body/body.5/global_pool/GlobalAveragePool_2conv_0
D RKNN: [04:33:42.488] 88   Conv             FLOAT16  NPU    (1,64,37,37),(1,64,7,7),(64)                 (1,64,6,6)             0              0              0              0              \              0              0              182.00         Conv:/out_conv/out_conv.1/body/body.5/global_pool/GlobalAveragePool_2conv_1
D RKNN: [04:33:42.488] 89   Conv             FLOAT16  NPU    (1,64,6,6),(1,64,6,6),(64)                   (1,64,1,1)             0              0              0              0              \              0              0              9.38           Conv:/out_conv/out_conv.1/body/body.5/global_pool/GlobalAveragePool_output_0
D RKNN: [04:33:42.488] 90   ConvLeakyRelu    FLOAT16  NPU    (1,64,1,1),(4,64,1,1),(4)                    (1,4,1,1)              0              0              0              0              \              0              0              0.70           Conv:/out_conv/out_conv.1/body/body.5/conv_double/conv_double.0/Conv
D RKNN: [04:33:42.488] 91   Conv             FLOAT16  NPU    (1,4,1,1),(64,4,1,1),(64)                    (1,64,1,1)             0              0              0              0              \              0              0              1.39           Conv:/out_conv/out_conv.1/body/body.5/conv_double/conv_double.2/Conv
D RKNN: [04:33:42.488] 92   Sigmoid          FLOAT16  NPU    (1,64,1,1)                                   (1,64,1,1)             0              0              0              0              \              0              0              0.25           Sigmoid:/out_conv/out_conv.1/body/body.5/conv_double/conv_double.3/Sigmoid
D RKNN: [04:33:42.488] 93   Mul              FLOAT16  NPU    (1,64,256,256),(1,64,1,1)                    (1,64,256,256)         0              0              0              0              \              0              0              16384.12       Mul:/out_conv/out_conv.1/body/body.5/Mul
D RKNN: [04:33:42.488] 94   Add              FLOAT16  NPU    (1,64,256,256),(1,64,256,256)                (1,64,256,256)         0              0              0              0              \              0              0              24576.00       Add:/out_conv/out_conv.1/Add
D RKNN: [04:33:42.488] 95   Conv             FLOAT16  NPU    (1,64,256,256),(3,64,1,1),(3)                (1,3,256,256)          0              0              0              0              \              0              0              10240.44       Conv:/out_conv/out_conv.2/Conv
D RKNN: [04:33:42.488] 96   OutputOperator   FLOAT16  CPU    (1,3,256,256)                                \                      0              0              0              0              \              0              0              2048.00        OutputOperator:output
D RKNN: [04:33:42.489] <<<<<<<< end: N4rknn18RKNNModelBuildPassE
```
可以看到所有的 Pad 算子（对应代码里的 nn.ReflectionPad2d(1)）、Resize (对应 nn.Upsample) 目标都是 CPU，且InstanceNormalization 虽然核心 exLayerNorm (ID 63) 在 NPU 上，但前后的 Reshape 全在 CPU

但是这些操作具体需要多少时间就需要通过adb连接到开发板使用rknn_server进行分析了

### adb连接rknn_server：测量算子推理耗时
1.首先确认adb已经开启：
```bash
ps -ef | grep adbd
```
2.启动 rknn_server：
```bash
# 检查是否安装了rknn_server
orangepi@orangepi5:~$ strings /usr/bin/rknn_server | grep -i "rknn_server version"
rknn_server version: 2.3.0 (e80ac5c build@2024-11-07T12:52:53)
# 启动
orangepi@orangepi5:~$ ulimit -n 65535
orangepi@orangepi5:~$ /usr/bin/rknn_server &
[1] 3496
orangepi@orangepi5:~$ start rknn server, version:2.3.0 (e80ac5c build@2024-11-07T12:52:53)
I NPUTransfer(3496): Starting NPU Transfer Server, Transfer version 2.2.2 (@2024-06-18T03:50:51)
```
3.在PC上连接：

```bash
triority@ubuntu:~/Desktop/rknn_test/UIE$ adb connect 192.168.10.43:5555
* daemon not running; starting now at tcp:5037
* daemon started successfully
connected to 192.168.10.43:5555

triority@ubuntu:~/Desktop/rknn_test/UIE$ adb devices
List of devices attached
192.168.10.43:5555	device

```
现在对转换程序做一些修改：
```python
from rknn.api import RKNN

# 网络连接填入IP:端口:'192.168.3.15:5555'
# USB 填入 adb devices 看到的序列号，或者仅连接一个设备时留空
DEVICE_ID = '192.168.10.43:5555' 

rknn = RKNN(verbose=True)

rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]], target_platform='rk3588')

print('--> Loading model')
ret = rknn.load_onnx(model='puienet_deploy.onnx')
if ret != 0:
    print('Load model failed!')
    exit(ret)

print('--> Building model')
ret = rknn.build(do_quantization=False) # 先fp16
if ret != 0:
    print('Build model failed!')
    exit(ret)


print('--> 初始化板载运行环境...')
ret = rknn.init_runtime(target='rk3588', device_id=DEVICE_ID, perf_debug=True)
if ret != 0:
    print('初始化 runtime 失败！')
    # 即使失败也导出模型，所以这里不直接 exit
else:
    print('--> 开始评估算子级性能 (eval_perf)...')
    # is_print=True 会在终端打印详细的每一层耗时表格
    rknn.eval_perf(is_print=True)

print('--> Export rknn model')
ret = rknn.export_rknn('./puienet_rk3588.rknn')
if ret != 0:
    print('Export rknn model failed!')
    exit(ret)

print('Done')

rknn.release()

```
输出的统计结果如下：
```bash
---------------------------------------------------------------------------------------------------
                                 Operator Time Consuming Ranking Table            
---------------------------------------------------------------------------------------------------
OpType             CallNumber   CPUTime(us)  GPUTime(us)  NPUTime(us)  TotalTime(us)  TimeRatio(%)  
---------------------------------------------------------------------------------------------------
ConvLeakyRelu      21           0            0            49852        49852          27.24%        
Transpose          2            0            0            27731        27731          15.15%        
Resize             3            23583        0            0            23583          12.89%        
exNorm             1            0            0            20555        20555          11.23%        
Pad                20           0            0            20167        20167          11.02%        
Conv               22           0            0            17892        17892          9.78%         
Add                5            0            0            8039         8039           4.39%         
Mul                5            0            0            7468         7468           4.08%         
Concat             3            0            0            4261         4261           2.33%         
MaxPool            3            0            0            3153         3153           1.72%         
ConvSigmoid        4            0            0            149          149            0.08%         
OutputOperator     1            103          0            0            103            0.06%         
Slice              2            0            0            42           42             0.02%         
InputOperator      1            4            0            0            4              0.00%         
Reshape            2            0            0            4            4              0.00%         
---------------------------------------------------------------------------------------------------
Total                           23690        0            159313       183003        
---------------------------------------------------------------------------------------------------
```
还有详细的每一层的具体统计
```bash
--> 开始评估算子级性能 (eval_perf)...
CPU Current Frequency List:
    - 1800000
    - 2400000
    - 2400000
NPU Current Frequency List:
    - 1000000000
DDR Current Frequency List:
    - Unknown

Warning: The performance result is just for debugging, may worse than actual performance!
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                          Network Layer Information Table                                                                                
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ID   OpType             DataType Target InputShape                               OutputShape            Cycles(DDR/NPU/Total)    Time(us)     MacUsage(%)          WorkLoad(0/1/2)      RW(KB)       FullName        
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1    InputOperator      FLOAT16  CPU    \                                        (1,3,256,256)          0/0/0                    4                                 0.0%/0.0%/0.0%       0            InputOperator:input
2    Conv               FLOAT16  NPU    (1,3,256,256),(3,3,1,1)                  (1,3,256,256)          60956/131072/131072      390          0.30/0.00/0.00       100.0%/0.0%/0.0%     384                          
3    Pad                FLOAT16  NPU    (1,3,256,256),(8)                        (1,3,258,258)          0/0/0                    307                               100.0%/0.0%/0.0%     1024         Pad:/pr_encoder/conv1/conv/conv.0/Pad
4    ConvLeakyRelu      FLOAT16  NPU    (1,3,258,258),(64,3,3,3),(64)            (1,64,256,256)         400064/2359296/2359296   2514         8.80/0.00/0.00       100.0%/0.0%/0.0%     1049         Conv:/pr_encoder/conv1/conv/conv.1/Conv
5    Pad                FLOAT16  NPU    (1,64,256,256),(8)                       (1,64,258,258)         0/0/0                    2003                              100.0%/0.0%/0.0%     8192         Pad:/pr_encoder/conv1/conv/conv.4/Pad
6    ConvLeakyRelu      FLOAT16  NPU    (1,64,258,258),(64,64,3,3),(64)          (1,64,256,256)         717967/4718592/4718592   5873         80.34/0.00/0.00      100.0%/0.0%/0.0%     8392         Conv:/pr_encoder/conv1/conv/conv.5/Conv
7    MaxPool            FLOAT16  NPU    (1,64,256,256)                           (1,64,128,128)         0/0/0                    2371                              100.0%/0.0%/0.0%     8192         MaxPool:/pr_encoder/pool1/MaxPool
8    Pad                FLOAT16  NPU    (1,64,128,128),(8)                       (1,64,130,130)         0/0/0                    531                               100.0%/0.0%/0.0%     2048         Pad:/pr_encoder/conv2/conv/conv.0/Pad
9    ConvLeakyRelu      FLOAT16  NPU    (1,64,130,130),(64,64,3,3),(64)          (1,64,128,128)         183240/1179648/1179648   1461         80.74/0.00/0.00      100.0%/0.0%/0.0%     2184         Conv:/pr_encoder/conv2/conv/conv.1/Conv
10   Pad                FLOAT16  NPU    (1,64,128,128),(8)                       (1,64,130,130)         0/0/0                    578                               100.0%/0.0%/0.0%     2048         Pad:/pr_encoder/conv2/conv/conv.4/Pad
11   ConvLeakyRelu      FLOAT16  NPU    (1,64,130,130),(64,64,3,3),(64)          (1,64,128,128)         183240/1179648/1179648   1461         80.74/0.00/0.00      100.0%/0.0%/0.0%     2184         Conv:/pr_encoder/conv2/conv/conv.5/Conv
12   MaxPool            FLOAT16  NPU    (1,64,128,128)                           (1,64,64,64)           0/0/0                    605                               100.0%/0.0%/0.0%     2048         MaxPool:/pr_encoder/pool2/MaxPool
13   Pad                FLOAT16  NPU    (1,64,64,64),(8)                         (1,64,66,66)           0/0/0                    163                               100.0%/0.0%/0.0%     512          Pad:/pr_encoder/conv3/conv/conv.0/Pad
14   ConvLeakyRelu      FLOAT16  NPU    (1,64,66,66),(64,64,3,3),(64)            (1,64,64,64)           48865/294912/294912      354          83.31/0.00/0.00      100.0%/0.0%/0.0%     616          Conv:/pr_encoder/conv3/conv/conv.1/Conv
15   Pad                FLOAT16  NPU    (1,64,64,64),(8)                         (1,64,66,66)           0/0/0                    162                               100.0%/0.0%/0.0%     512          Pad:/pr_encoder/conv3/conv/conv.4/Pad
16   ConvLeakyRelu      FLOAT16  NPU    (1,64,66,66),(64,64,3,3),(64)            (1,64,64,64)           48865/294912/294912      357          82.61/0.00/0.00      100.0%/0.0%/0.0%     616          Conv:/pr_encoder/conv3/conv/conv.5/Conv
17   MaxPool            FLOAT16  NPU    (1,64,64,64)                             (1,64,32,32)           0/0/0                    177                               100.0%/0.0%/0.0%     512          MaxPool:/pr_encoder/pool3/MaxPool
18   Pad                FLOAT16  NPU    (1,64,32,32),(8)                         (1,64,34,34)           0/0/0                    74                                100.0%/0.0%/0.0%     128          Pad:/pr_encoder/conv4/body/body.0/Pad
19   ConvLeakyRelu      FLOAT16  NPU    (1,64,34,34),(64,64,3,3),(64)            (1,64,32,32)           14925/73728/73728        114          64.67/0.00/0.00      100.0%/0.0%/0.0%     216          Conv:/pr_encoder/conv4/body/body.1/Conv
20   Pad                FLOAT16  NPU    (1,64,32,32),(8)                         (1,64,34,34)           0/0/0                    69                                100.0%/0.0%/0.0%     128          Pad:/pr_encoder/conv4/body/body.3/Pad
21   Conv               FLOAT16  NPU    (1,64,34,34),(64,64,3,3),(64)            (1,64,32,32)           14925/73728/73728        115          64.11/0.00/0.00      100.0%/0.0%/0.0%     216          Conv:/pr_encoder/conv4/body/body.4/Conv
22   Conv               FLOAT16  NPU    (1,64,32,32),(1,64,7,7),(64)             (1,64,5,5)             5953/1568/5953           47           0.33/0.00/0.00       100.0%/0.0%/0.0%     134          Conv:/pr_encoder/conv4/body/body.5/global_pool/GlobalAveragePool_2conv0
23   Conv               FLOAT16  NPU    (1,64,5,5),(1,64,5,5),(64)               (1,64,1,1)             287/400/400              38           0.01/0.00/0.00       100.0%/0.0%/0.0%     6            Conv:/pr_encoder/conv4/body/body.5/global_pool/GlobalAveragePool_2conv1
24   ConvLeakyRelu      FLOAT16  NPU    (1,64,1,1),(4,64,1,1),(4)                (1,4,1,1)              31/32/32                 21           0.00/0.00/0.00       100.0%/0.0%/0.0%     0            Conv:/pr_encoder/conv4/body/body.5/conv_double/conv_double.0/Conv
25   ConvSigmoid        FLOAT16  NPU    (1,4,1,1),(64,4,1,1),(64)                (1,64,1,1)             61/128/128               37           0.00/0.00/0.00       100.0%/0.0%/0.0%     1            Conv:/pr_encoder/conv4/body/body.5/conv_double/conv_double.2/Conv
26   Mul                FLOAT16  NPU    (1,64,32,32),(1,64,1,1)                  (1,64,32,32)           0/0/0                    60                                100.0%/0.0%/0.0%     128          Mul:/pr_encoder/conv4/body/body.5/Mul
27   Add                FLOAT16  NPU    (1,64,32,32),(1,64,32,32)                (1,64,32,32)           0/0/0                    71                                100.0%/0.0%/0.0%     256          Add:/pr_encoder/conv4/Add
28   Pad                FLOAT16  NPU    (1,64,32,32),(8)                         (1,64,34,34)           0/0/0                    68                                100.0%/0.0%/0.0%     128          Pad:/pr_conv/body/body.0/Pad
29   ConvLeakyRelu      FLOAT16  NPU    (1,64,34,34),(64,64,3,3),(64)            (1,64,32,32)           14925/73728/73728        114          64.67/0.00/0.00      100.0%/0.0%/0.0%     216          Conv:/pr_conv/body/body.1/Conv
30   Pad                FLOAT16  NPU    (1,64,32,32),(8)                         (1,64,34,34)           0/0/0                    69                                100.0%/0.0%/0.0%     128          Pad:/pr_conv/body/body.3/Pad
31   Conv               FLOAT16  NPU    (1,64,34,34),(64,64,3,3),(64)            (1,64,32,32)           14925/73728/73728        115          64.11/0.00/0.00      100.0%/0.0%/0.0%     216          Conv:/pr_conv/body/body.4/Conv
32   Conv               FLOAT16  NPU    (1,64,32,32),(1,64,7,7),(64)             (1,64,5,5)             5953/1568/5953           47           0.33/0.00/0.00       100.0%/0.0%/0.0%     134          Conv:/pr_conv/body/body.5/global_pool/GlobalAveragePool_2conv0
33   Conv               FLOAT16  NPU    (1,64,5,5),(1,64,5,5),(64)               (1,64,1,1)             287/400/400              37           0.01/0.00/0.00       100.0%/0.0%/0.0%     6            Conv:/pr_conv/body/body.5/global_pool/GlobalAveragePool_2conv1
34   ConvLeakyRelu      FLOAT16  NPU    (1,64,1,1),(4,64,1,1),(4)                (1,4,1,1)              31/32/32                 22           0.00/0.00/0.00       100.0%/0.0%/0.0%     0            Conv:/pr_conv/body/body.5/conv_double/conv_double.0/Conv
35   ConvSigmoid        FLOAT16  NPU    (1,4,1,1),(64,4,1,1),(64)                (1,64,1,1)             61/128/128               38           0.00/0.00/0.00       100.0%/0.0%/0.0%     1            Conv:/pr_conv/body/body.5/conv_double/conv_double.2/Conv
36   Mul                FLOAT16  NPU    (1,64,32,32),(1,64,1,1)                  (1,64,32,32)           0/0/0                    58                                100.0%/0.0%/0.0%     128          Mul:/pr_conv/body/body.5/Mul
37   Add                FLOAT16  NPU    (1,64,32,32),(1,64,32,32)                (1,64,32,32)           0/0/0                    64                                100.0%/0.0%/0.0%     256          Add:/pr_conv/Add
38   Resize             FLOAT16  CPU    (1,64,32,32),(0),(4)                     (1,64,64,64)           0/0/0                    1190                              0.0%/0.0%/0.0%       128          Resize:/pr_Up3/up/up.0/Resize
39   Concat             FLOAT16  NPU    (1,64,64,64),(1,64,64,64)                (1,128,64,64)          0/0/0                    241                               100.0%/0.0%/0.0%     1024         Concat:/Concat  
40   Pad                FLOAT16  NPU    (1,128,64,64),(8)                        (1,128,66,66)          0/0/0                    299                               100.0%/0.0%/0.0%     1024         Pad:/pr_UpConv3/conv/conv.0/Pad
41   ConvLeakyRelu      FLOAT16  NPU    (1,128,66,66),(64,128,3,3),(64)          (1,64,64,64)           75554/589824/589824      739          79.81/0.00/0.00      100.0%/0.0%/0.0%     1233         Conv:/pr_UpConv3/conv/conv.1/Conv
42   Pad                FLOAT16  NPU    (1,64,64,64),(8)                         (1,64,66,66)           0/0/0                    162                               100.0%/0.0%/0.0%     512          Pad:/pr_UpConv3/conv/conv.4/Pad
43   ConvLeakyRelu      FLOAT16  NPU    (1,64,66,66),(64,64,3,3),(64)            (1,64,64,64)           48865/294912/294912      357          82.61/0.00/0.00      100.0%/0.0%/0.0%     616          Conv:/pr_UpConv3/conv/conv.5/Conv
44   Resize             FLOAT16  CPU    (1,64,64,64),(0),(4)                     (1,64,128,128)         0/0/0                    4510                              0.0%/0.0%/0.0%       512          Resize:/pr_Up2/up/up.0/Resize
45   Concat             FLOAT16  NPU    (1,64,128,128),(1,64,128,128)            (1,128,128,128)        0/0/0                    811                               100.0%/0.0%/0.0%     4096         Concat:/Concat_1
46   Pad                FLOAT16  NPU    (1,128,128,128),(8)                      (1,128,130,130)        0/0/0                    1025                              100.0%/0.0%/0.0%     4096         Pad:/pr_UpConv2/conv/conv.0/Pad
47   ConvLeakyRelu      FLOAT16  NPU    (1,128,130,130),(64,128,3,3),(64)        (1,64,128,128)         277808/2359296/2359296   3009         78.41/0.00/0.00      100.0%/0.0%/0.0%     4369         Conv:/pr_UpConv2/conv/conv.1/Conv
48   Pad                FLOAT16  NPU    (1,64,128,128),(8)                       (1,64,130,130)         0/0/0                    576                               100.0%/0.0%/0.0%     2048         Pad:/pr_UpConv2/conv/conv.4/Pad
49   ConvLeakyRelu      FLOAT16  NPU    (1,64,130,130),(64,64,3,3),(64)          (1,64,128,128)         183240/1179648/1179648   1463         80.63/0.00/0.00      100.0%/0.0%/0.0%     2184         Conv:/pr_UpConv2/conv/conv.5/Conv
50   Resize             FLOAT16  CPU    (1,64,128,128),(0),(4)                   (1,64,256,256)         0/0/0                    17883                             0.0%/0.0%/0.0%       2048         Resize:/pr_Up1/up/up.0/Resize
51   Concat             FLOAT16  NPU    (1,64,256,256),(1,64,256,256)            (1,128,256,256)        0/0/0                    3209                              100.0%/0.0%/0.0%     16384        Concat:/Concat_2
52   Conv               FLOAT16  NPU    (1,128,256,256),(1,128,7,7),(128)        (1,128,37,37)          724645/134848/724645     1864         0.90/0.00/0.00       100.0%/0.0%/0.0%     16396        Conv:/gap/GlobalAveragePool_2conv0
53   Conv               FLOAT16  NPU    (1,128,37,37),(1,128,7,7),(128)          (1,128,6,6)            15758/4704/15758         123          0.36/0.00/0.00       100.0%/0.0%/0.0%     355          Conv:/gap/GlobalAveragePool_2conv1
54   Conv               FLOAT16  NPU    (1,128,6,6),(1,128,6,6),(128)            (1,128,1,1)            812/1152/1152            39           0.02/0.00/0.00       100.0%/0.0%/0.0%     18           Conv:/gap/GlobalAveragePool_2conv2
55   Conv               FLOAT16  NPU    (1,128,1,1),(40,128,1,1),(40)            (1,40,1,1)             456/192/456              36           0.03/0.00/0.00       100.0%/0.0%/0.0%     10           Conv:/u_conv_layer/Conv
56   Slice              FLOAT16  NPU    (1,40,1,1),(1),(1),(1),(1)               (1,20,1,1)             0/0/0                    21                                100.0%/0.0%/0.0%     0            Slice:/Slice    
57   Conv               FLOAT16  NPU    (1,20,1,1),(128,20,1,1),(128)            (1,128,1,1)            381/256/381              36           0.01/0.00/0.00       100.0%/0.0%/0.0%     8            Conv:/conv_u/Conv
58   Conv               FLOAT16  NPU    (1,128,1,1),(40,128,1,1),(40)            (1,40,1,1)             456/192/456              37           0.03/0.00/0.00       100.0%/0.0%/0.0%     10           Conv:/s_conv_layer/Conv
59   Slice              FLOAT16  NPU    (1,40,1,1),(1),(1),(1),(1)               (1,20,1,1)             0/0/0                    21                                100.0%/0.0%/0.0%     0            Slice:/Slice_1  
60   ConvLeakyRelu      FLOAT16  NPU    (1,20,1,1),(128,20,1,1),(128)            (1,128,1,1)            381/256/381              36           0.01/0.00/0.00       100.0%/0.0%/0.0%     8            Conv:/conv_s/Conv
61   Transpose          FLOAT16  NPU    (1,128,256,256)                          (256,256,1,128)        0/0/0                    15921                             100.0%/0.0%/0.0%     16384        Transpose:/Concat_2_output_0_tp
62   Reshape            FLOAT16  NPU    (256,256,1,128),(4)                      (1,65536,1,128)        0/0/0                    2                                 0.0%/0.0%/0.0%       16384        Reshape:/Concat_2_output_0_tp_rs
63   exNorm             FLOAT16  NPU    (1,65536,1,128),(1,1,1,128),...          (1,65536,1,128)        0/0/0                    20555                             100.0%/0.0%/0.0%     16448        exNorm:/insnorm/InstanceNormalization
64   Reshape            FLOAT16  NPU    (1,65536,1,128),(4)                      (256,256,1,128)        0/0/0                    2                                 0.0%/0.0%/0.0%       16384        Reshape:/insnorm/InstanceNormalization_rs
65   Transpose          FLOAT16  NPU    (256,256,1,128)                          (1,128,256,256)        0/0/0                    11810                             100.0%/0.0%/0.0%     16384        Transpose:/insnorm/InstanceNormalization_rs_tp
66   Mul                FLOAT16  NPU    (1,128,256,256),(1,128,1,1)              (1,128,256,256)        0/0/0                    3653                              100.0%/0.0%/0.0%     16384        Mul:/Mul        
67   Add                FLOAT16  NPU    (1,128,256,256),(1,128,1,1)              (1,128,256,256)        0/0/0                    3646                              100.0%/0.0%/0.0%     16384        Add:/Add        
68   Pad                FLOAT16  NPU    (1,128,256,256),(8)                      (1,128,258,258)        0/0/0                    3987                              100.0%/0.0%/0.0%     16384        Pad:/pr_UpConv1/conv/conv.0/Pad
69   ConvLeakyRelu      FLOAT16  NPU    (1,128,258,258),(64,128,3,3),(64)        (1,64,256,256)         1081285/9437184/9437184  14314        65.93/0.00/0.00      100.0%/0.0%/0.0%     16785        Conv:/pr_UpConv1/conv/conv.1/Conv
70   Pad                FLOAT16  NPU    (1,64,256,256),(8)                       (1,64,258,258)         0/0/0                    1974                              100.0%/0.0%/0.0%     8192         Pad:/pr_UpConv1/conv/conv.4/Pad
71   ConvLeakyRelu      FLOAT16  NPU    (1,64,258,258),(64,64,3,3),(64)          (1,64,256,256)         717967/4718592/4718592   5874         80.33/0.00/0.00      100.0%/0.0%/0.0%     8392         Conv:/pr_UpConv1/conv/conv.5/Conv
72   Pad                FLOAT16  NPU    (1,64,256,256),(8)                       (1,64,258,258)         0/0/0                    2013                              100.0%/0.0%/0.0%     8192         Pad:/out_conv/out_conv.0/body/body.0/Pad
73   ConvLeakyRelu      FLOAT16  NPU    (1,64,258,258),(64,64,3,3),(64)          (1,64,256,256)         717967/4718592/4718592   5895         80.04/0.00/0.00      100.0%/0.0%/0.0%     8392         Conv:/out_conv/out_conv.0/body/body.1/Conv
74   Pad                FLOAT16  NPU    (1,64,256,256),(8)                       (1,64,258,258)         0/0/0                    2018                              100.0%/0.0%/0.0%     8192         Pad:/out_conv/out_conv.0/body/body.3/Pad
75   Conv               FLOAT16  NPU    (1,64,258,258),(64,64,3,3),(64)          (1,64,256,256)         717967/4718592/4718592   5877         80.29/0.00/0.00      100.0%/0.0%/0.0%     8392         Conv:/out_conv/out_conv.0/body/body.4/Conv
76   Conv               FLOAT16  NPU    (1,64,256,256),(1,64,7,7),(64)           (1,64,37,37)           362323/67424/362323      981          0.85/0.00/0.00       100.0%/0.0%/0.0%     8198         Conv:/out_conv/out_conv.0/body/body.5/global_pool/GlobalAveragePool_2conv0
77   Conv               FLOAT16  NPU    (1,64,37,37),(1,64,7,7),(64)             (1,64,6,6)             7879/2352/7879           101          0.22/0.00/0.00       100.0%/0.0%/0.0%     177          Conv:/out_conv/out_conv.0/body/body.5/global_pool/GlobalAveragePool_2conv1
78   Conv               FLOAT16  NPU    (1,64,6,6),(1,64,6,6),(64)               (1,64,1,1)             406/576/576              40           0.01/0.00/0.00       100.0%/0.0%/0.0%     9            Conv:/out_conv/out_conv.0/body/body.5/global_pool/GlobalAveragePool_2conv2
79   ConvLeakyRelu      FLOAT16  NPU    (1,64,1,1),(4,64,1,1),(4)                (1,4,1,1)              31/32/32                 20           0.00/0.00/0.00       100.0%/0.0%/0.0%     0            Conv:/out_conv/out_conv.0/body/body.5/conv_double/conv_double.0/Conv
80   ConvSigmoid        FLOAT16  NPU    (1,4,1,1),(64,4,1,1),(64)                (1,64,1,1)             61/128/128               37           0.00/0.00/0.00       100.0%/0.0%/0.0%     1            Conv:/out_conv/out_conv.0/body/body.5/conv_double/conv_double.2/Conv
81   Mul                FLOAT16  NPU    (1,64,256,256),(1,64,1,1)                (1,64,256,256)         0/0/0                    1835                              100.0%/0.0%/0.0%     8192         Mul:/out_conv/out_conv.0/body/body.5/Mul
82   Add                FLOAT16  NPU    (1,64,256,256),(1,64,256,256)            (1,64,256,256)         0/0/0                    2096                              100.0%/0.0%/0.0%     16384        Add:/out_conv/out_conv.0/Add
83   Pad                FLOAT16  NPU    (1,64,256,256),(8)                       (1,64,258,258)         0/0/0                    2041                              100.0%/0.0%/0.0%     8192         Pad:/out_conv/out_conv.1/body/body.0/Pad
84   ConvLeakyRelu      FLOAT16  NPU    (1,64,258,258),(64,64,3,3),(64)          (1,64,256,256)         717967/4718592/4718592   5833         80.89/0.00/0.00      100.0%/0.0%/0.0%     8392         Conv:/out_conv/out_conv.1/body/body.1/Conv
85   Pad                FLOAT16  NPU    (1,64,256,256),(8)                       (1,64,258,258)         0/0/0                    2048                              100.0%/0.0%/0.0%     8192         Pad:/out_conv/out_conv.1/body/body.3/Pad
86   Conv               FLOAT16  NPU    (1,64,258,258),(64,64,3,3),(64)          (1,64,256,256)         717967/4718592/4718592   5875         80.32/0.00/0.00      100.0%/0.0%/0.0%     8392         Conv:/out_conv/out_conv.1/body/body.4/Conv
87   Conv               FLOAT16  NPU    (1,64,256,256),(1,64,7,7),(64)           (1,64,37,37)           362323/67424/362323      982          0.85/0.00/0.00       100.0%/0.0%/0.0%     8198         Conv:/out_conv/out_conv.1/body/body.5/global_pool/GlobalAveragePool_2conv0
88   Conv               FLOAT16  NPU    (1,64,37,37),(1,64,7,7),(64)             (1,64,6,6)             7879/2352/7879           54           0.41/0.00/0.00       100.0%/0.0%/0.0%     177          Conv:/out_conv/out_conv.1/body/body.5/global_pool/GlobalAveragePool_2conv1
89   Conv               FLOAT16  NPU    (1,64,6,6),(1,64,6,6),(64)               (1,64,1,1)             406/576/576              37           0.01/0.00/0.00       100.0%/0.0%/0.0%     9            Conv:/out_conv/out_conv.1/body/body.5/global_pool/GlobalAveragePool_2conv2
90   ConvLeakyRelu      FLOAT16  NPU    (1,64,1,1),(4,64,1,1),(4)                (1,4,1,1)              31/32/32                 21           0.00/0.00/0.00       100.0%/0.0%/0.0%     0            Conv:/out_conv/out_conv.1/body/body.5/conv_double/conv_double.0/Conv
91   ConvSigmoid        FLOAT16  NPU    (1,4,1,1),(64,4,1,1),(64)                (1,64,1,1)             61/128/128               37           0.00/0.00/0.00       100.0%/0.0%/0.0%     1            Conv:/out_conv/out_conv.1/body/body.5/conv_double/conv_double.2/Conv
92   Mul                FLOAT16  NPU    (1,64,256,256),(1,64,1,1)                (1,64,256,256)         0/0/0                    1862                              100.0%/0.0%/0.0%     8192         Mul:/out_conv/out_conv.1/body/body.5/Mul
93   Add                FLOAT16  NPU    (1,64,256,256),(1,64,256,256)            (1,64,256,256)         0/0/0                    2162                              100.0%/0.0%/0.0%     16384        Add:/out_conv/out_conv.1/Add
94   Conv               FLOAT16  NPU    (1,64,256,256),(3,64,1,1),(3)            (1,3,256,256)          443317/131072/443317     1021         2.41/0.00/0.00       100.0%/0.0%/0.0%     8192         Conv:/out_conv/out_conv.2/Conv
95   OutputOperator     FLOAT16  CPU    (1,3,256,256)                            \                      0/0/0                    103                               0.0%/0.0%/0.0%       2048         OutputOperator:output
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Total Operator Elapsed Per Frame Time(us): 183003
Total Memory Read/Write Per Frame Size(KB): 407048.59
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
### 推理速度优化
#### 需求
需要实现720p分辨率下24FPS的推理速度，当前256*256分辨率单帧时间180ms，帧率差距巨大。分辨率方面可以通过分块处理解决
#### 推理速度时间优化
主要修改点：
+ 绝大部分算子只能使用单个npu核心参与计算，因此可以独立操作三个核心，使用队列分配任务，分别完成不同块的计算从而单帧时间提升3倍
+ 上采样的插值用双线性插值计算回退至CPU，时间占23.5ms，换成最近邻插值可以给npu处理
+ 镜像填充ReflectionPad2d回退CPU用时20ms，换成Conv2d(padding=1)零填充就可以交给npu处理
+ InstanceNorm现在时间消耗48ms，如果能GroupNorm组归一化也能快得多，当然使用BatchNorm能更快

完成上述工作后，推理时间缩短到96ms，时间减少了一半不过依然达不到要求，为了保证一定的效果，可以接受720p下的24fps，也就是三个核心分15块总时间40ms以内，也就是单核单块8ms。

修改模型结构，进一步简化运算：
+ 减少卷积层通道数（通道数的增加会导致计算量以二次方速度增加，因此128通道改为16通道）和层数，时间缩短到17ms
+ 删除了SE层，时间缩短到13ms

```
---------------------------------------------------------------------------------------------------
                                 Operator Time Consuming Ranking Table            
---------------------------------------------------------------------------------------------------
OpType             CallNumber   CPUTime(us)  GPUTime(us)  NPUTime(us)  TotalTime(us)  TimeRatio(%)  
---------------------------------------------------------------------------------------------------
ConvLeakyRelu      14           0            0            5929         5929           43.67%        
ConvAdd            3            0            0            1651         1651           12.16%        
Concat             2            0            0            1076         1076           7.92%         
Conv               7            0            0            1064         1064           7.84%         
Add                1            0            0            922          922            6.79%         
Mul                1            0            0            883          883            6.50%         
MaxPool            2            0            0            781          781            5.75%         
BatchNormalization 1            0            0            761          761            5.60%         
Resize             2            0            0            372          372            2.74%         
OutputOperator     1            91           0            0            91             0.67%         
Slice              2            0            0            40           40             0.29%         
InputOperator      1            8            0            0            8              0.06%         
---------------------------------------------------------------------------------------------------
Total                           99           0            13479        13578         
---------------------------------------------------------------------------------------------------
```

可以看到现在的13ms中，43.67%的时间依然是在执行卷积操作，而我还需要提速38%，因此下一步首先就是在高分辨率卷积层增加卷积步长以减少分辨率，后续特征图处理128*128，修改之后时间达到5.3ms

```
---------------------------------------------------------------------------------------------------
                                 Operator Time Consuming Ranking Table            
---------------------------------------------------------------------------------------------------
OpType             CallNumber   CPUTime(us)  GPUTime(us)  NPUTime(us)  TotalTime(us)  TimeRatio(%)  
---------------------------------------------------------------------------------------------------
ConvLeakyRelu      13           0            0            2574         2574           48.09%        
ConvAdd            3            0            0            555          555            10.37%        
Concat             2            0            0            501          501            9.36%         
Conv               7            0            0            429          429            8.01%         
Resize             2            0            0            331          331            6.18%         
Add                1            0            0            229          229            4.28%         
Mul                1            0            0            224          224            4.18%         
BatchNormalization 1            0            0            213          213            3.98%         
MaxPool            1            0            0            176          176            3.29%         
Slice              2            0            0            61           61             1.14%         
OutputOperator     1            51           0            0            51             0.95%         
InputOperator      1            9            0            0            9              0.17%         
---------------------------------------------------------------------------------------------------
Total                           60           0            5293         5353          
---------------------------------------------------------------------------------------------------

```
现在已经实现了时间的指标，下一步就是效果上的指标了。

全部npu参数如下：
```
CPU Current Frequency List:
    - 1800000
    - 2400000
    - 2400000
NPU Current Frequency List:
    - 1000000000
DDR Current Frequency List:
    - Unknown

Warning: The performance result is just for debugging, may worse than actual performance!
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                          Network Layer Information Table                                                                                
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ID   OpType             DataType Target InputShape                               OutputShape            Cycles(DDR/NPU/Total)    Time(us)     MacUsage(%)          WorkLoad(0/1/2)      RW(KB)       FullName        
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1    InputOperator      FLOAT16  CPU    \                                        (1,3,256,256)          0/0/0                    9                                 0.0%/0.0%/0.0%       0            InputOperator:input
2    ConvLeakyRelu      FLOAT16  NPU    (1,3,256,256),(16,3,3,3),(16)            (1,16,128,128)         38889/147456/147456      347          3.98/0.00/0.00       100.0%/0.0%/0.0%     386          Conv:/pr_encoder/conv1/conv1.0/Conv
3    ConvLeakyRelu      FLOAT16  NPU    (1,16,128,128),(16,16,3,3),(16)          (1,16,128,128)         44528/147456/147456      216          34.13/0.00/0.00      100.0%/0.0%/0.0%     516          Conv:/pr_encoder/conv2/conv/conv.0/Conv
4    ConvLeakyRelu      FLOAT16  NPU    (1,16,128,128),(16,16,3,3),(16)          (1,16,128,128)         44528/147456/147456      213          34.61/0.00/0.00      100.0%/0.0%/0.0%     516          Conv:/pr_encoder/conv2/conv/conv.2/Conv
5    MaxPool            FLOAT16  NPU    (1,16,128,128)                           (1,16,64,64)           0/0/0                    176                               100.0%/0.0%/0.0%     512          MaxPool:/pr_encoder/pool2/MaxPool
6    ConvLeakyRelu      FLOAT16  NPU    (1,16,64,64),(16,16,3,3),(16)            (1,16,64,64)           11280/36864/36864        78           23.63/0.00/0.00      100.0%/0.0%/0.0%     132          Conv:/pr_encoder/conv3/conv/conv.0/Conv
7    ConvLeakyRelu      FLOAT16  NPU    (1,16,64,64),(16,16,3,3),(16)            (1,16,64,64)           11280/36864/36864        121          15.23/0.00/0.00      100.0%/0.0%/0.0%     132          Conv:/pr_encoder/conv3/conv/conv.2/Conv
8    ConvLeakyRelu      FLOAT16  NPU    (1,16,64,64),(16,16,3,3),(16)            (1,16,64,64)           11280/36864/36864        121          15.23/0.00/0.00      100.0%/0.0%/0.0%     132          Conv:/pr_conv/body/body.0/Conv
9    ConvAdd            FLOAT16  NPU    (1,16,64,64),(16,16,3,3),(16),...        (1,16,64,64)           16822/36864/36864        122          15.11/0.00/0.00      100.0%/0.0%/0.0%     260          Conv:/pr_conv/body/body.2/Conv_ConvAdd
10   Resize             FLOAT16  NPU    (1,16,64,64),(0),(4)                     (1,16,128,128)         0/0/0                    177                               100.0%/0.0%/0.0%     128          Resize:/pr_Up2/up/up.0/Resize
11   Concat             FLOAT16  NPU    (1,16,128,128),(1,16,128,128)            (1,32,128,128)         0/0/0                    293                               100.0%/0.0%/0.0%     1024         Concat:/Concat  
12   ConvLeakyRelu      FLOAT16  NPU    (1,32,128,128),(16,32,3,3),(16)          (1,16,128,128)         66887/147456/147456      298          49.48/0.00/0.00      100.0%/0.0%/0.0%     1033         Conv:/pr_UpConv2/conv/conv.0/Conv
13   ConvLeakyRelu      FLOAT16  NPU    (1,16,128,128),(16,16,3,3),(16)          (1,16,128,128)         44528/147456/147456      257          28.69/0.00/0.00      100.0%/0.0%/0.0%     516          Conv:/pr_UpConv2/conv/conv.2/Conv
14   Concat             FLOAT16  NPU    (1,16,128,128),(1,16,128,128)            (1,32,128,128)         0/0/0                    208                               100.0%/0.0%/0.0%     1024         Concat:/Concat_1
15   Conv               FLOAT16  NPU    (1,32,128,128),(1,32,7,7),(32)           (1,32,19,19)           45445/18032/45445        140          0.79/0.00/0.00       100.0%/0.0%/0.0%     1027         Conv:/gap/GlobalAveragePool_2conv0
16   Conv               FLOAT16  NPU    (1,32,19,19),(1,32,7,7),(32)             (1,32,3,3)             1140/784/1140            34           0.08/0.00/0.00       100.0%/0.0%/0.0%     25           Conv:/gap/GlobalAveragePool_2conv1
17   Conv               FLOAT16  NPU    (1,32,3,3),(1,32,3,3),(32)               (1,32,1,1)             57/144/144               31           0.00/0.00/0.00       100.0%/0.0%/0.0%     1            Conv:/gap/GlobalAveragePool_2conv2
18   Conv               FLOAT16  NPU    (1,32,1,1),(40,32,1,1),(40)              (1,40,1,1)             123/96/123               32           0.01/0.00/0.00       100.0%/0.0%/0.0%     2            Conv:/u_conv_layer/Conv
19   Slice              FLOAT16  NPU    (1,40,1,1),(1),(1),(1),(1)               (1,20,1,1)             0/0/0                    31                                100.0%/0.0%/0.0%     0            Slice:/Slice    
20   Conv               FLOAT16  NPU    (1,20,1,1),(32,20,1,1),(32)              (1,32,1,1)             97/64/97                 30           0.00/0.00/0.00       100.0%/0.0%/0.0%     2            Conv:/conv_u/Conv
21   Conv               FLOAT16  NPU    (1,32,1,1),(40,32,1,1),(40)              (1,40,1,1)             123/96/123               31           0.01/0.00/0.00       100.0%/0.0%/0.0%     2            Conv:/s_conv_layer/Conv
22   Slice              FLOAT16  NPU    (1,40,1,1),(1),(1),(1),(1)               (1,20,1,1)             0/0/0                    30                                100.0%/0.0%/0.0%     0            Slice:/Slice_1  
23   BatchNormalization FLOAT16  NPU    (1,32,128,128),(32),(32),(32),(32)       (1,32,128,128)         0/0/0                    213                               100.0%/0.0%/0.0%     1024         BatchNormalization:/insnorm/BatchNormalization
24   ConvLeakyRelu      FLOAT16  NPU    (1,20,1,1),(32,20,1,1),(32)              (1,32,1,1)             97/64/97                 31           0.00/0.00/0.00       100.0%/0.0%/0.0%     2            Conv:/conv_s/Conv
25   Mul                FLOAT16  NPU    (1,32,128,128),(1,32,1,1)                (1,32,128,128)         0/0/0                    224                               100.0%/0.0%/0.0%     1024         Mul:/Mul        
26   Add                FLOAT16  NPU    (1,32,128,128),(1,32,1,1)                (1,32,128,128)         0/0/0                    229                               100.0%/0.0%/0.0%     1024         Add:/Add        
27   ConvLeakyRelu      FLOAT16  NPU    (1,32,128,128),(16,32,3,3),(16)          (1,16,128,128)         66887/147456/147456      253          58.28/0.00/0.00      100.0%/0.0%/0.0%     1033         Conv:/pr_UpConv1/conv/conv.0/Conv
28   ConvLeakyRelu      FLOAT16  NPU    (1,16,128,128),(16,16,3,3),(16)          (1,16,128,128)         44528/147456/147456      213          34.61/0.00/0.00      100.0%/0.0%/0.0%     516          Conv:/pr_UpConv1/conv/conv.2/Conv
29   ConvLeakyRelu      FLOAT16  NPU    (1,16,128,128),(16,16,3,3),(16)          (1,16,128,128)         44528/147456/147456      213          34.61/0.00/0.00      100.0%/0.0%/0.0%     516          Conv:/out_conv/out_conv.0/body/body.0/Conv
30   ConvAdd            FLOAT16  NPU    (1,16,128,128),(16,16,3,3),(16),...      (1,16,128,128)         66693/147456/147456      216          34.13/0.00/0.00      100.0%/0.0%/0.0%     1028         Conv:/out_conv/out_conv.0/body/body.2/Conv_ConvAdd
31   ConvLeakyRelu      FLOAT16  NPU    (1,16,128,128),(16,16,3,3),(16)          (1,16,128,128)         44528/147456/147456      213          34.61/0.00/0.00      100.0%/0.0%/0.0%     516          Conv:/out_conv/out_conv.1/body/body.0/Conv
32   ConvAdd            FLOAT16  NPU    (1,16,128,128),(16,16,3,3),(16),...      (1,16,128,128)         66693/147456/147456      217          33.98/0.00/0.00      100.0%/0.0%/0.0%     1028         Conv:/out_conv/out_conv.1/body/body.2/Conv_ConvAdd
33   Conv               FLOAT16  NPU    (1,16,128,128),(3,16,1,1),(3)            (1,3,128,128)          33255/32768/33255        131          1.17/0.00/0.00       100.0%/0.0%/0.0%     512          Conv:/out_conv/out_conv.2/Conv
34   Resize             FLOAT16  NPU    (1,3,128,128),(0),(4)                    (1,3,256,256)          0/0/0                    154                               100.0%/0.0%/0.0%     256          Resize:/final_upsample/Resize
35   OutputOperator     FLOAT16  CPU    (1,3,256,256)                            \                      0/0/0                    51                                0.0%/0.0%/0.0%       1024         OutputOperator:output
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Total Operator Elapsed Per Frame Time(us): 5353
Total Memory Read/Write Per Frame Size(KB): 16884.12
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
                                 Operator Time Consuming Ranking Table            
---------------------------------------------------------------------------------------------------
OpType             CallNumber   CPUTime(us)  GPUTime(us)  NPUTime(us)  TotalTime(us)  TimeRatio(%)  
---------------------------------------------------------------------------------------------------
ConvLeakyRelu      13           0            0            2574         2574           48.09%        
ConvAdd            3            0            0            555          555            10.37%        
Concat             2            0            0            501          501            9.36%         
Conv               7            0            0            429          429            8.01%         
Resize             2            0            0            331          331            6.18%         
Add                1            0            0            229          229            4.28%         
Mul                1            0            0            224          224            4.18%         
BatchNormalization 1            0            0            213          213            3.98%         
MaxPool            1            0            0            176          176            3.29%         
Slice              2            0            0            61           61             1.14%         
OutputOperator     1            51           0            0            51             0.95%         
InputOperator      1            9            0            0            9              0.17%         
---------------------------------------------------------------------------------------------------
Total                           60           0            5293         5353          
---------------------------------------------------------------------------------------------------

```
可以进行一些简要的分析：
+  ID 9, 30, 32：原本的卷积和残差加法被 RKNN 自动优化成了 ConvAdd，加法操作被合并到了卷积计算过程中，耗时几乎计为 0
+  ID 2 的 ConvLeakyRelu 处理 256x256 输入，耗时仅 347us，后续大部分计算都在 128x128 尺度，这使得卷积总耗时大幅收缩
+  之前 SE 模块产生的Add、Mul、ConvSigmoid 合计占了 4.1ms，删除之后也带来相对较大的提升
+  总读写（RW）从之前的 300MB+ 降到了目前的 65MB，缓解了带宽压力

#### 推理质量效果优化



### 可能遇到的问题
#### adb版本不对应
这里要注意一件事，rknn要求adb版本必须是40，更高或更低都不行。其中platform-tools 28.0.1 对应 ADB 1.0.40。

如果下面的代码提示版本不匹配，例如终端输出这样表示系统adb版本为39，与要求的40不匹配：
```bash
connected to 192.168.10.43:5555
--> 尝试连接板子 [192.168.10.43:5555] 进行初始化...
I target set by user is: rk3588
I Check RK3588 board npu runtime version
I Starting ntp or adb, target is RK3588
I Device [192.168.10.43:5555] not found in ntb device list.
I Start adb...
adb server version (39) doesn't match this client (40); killing...
* daemon started successfully
error: device '192.168.10.43:5555' not found
E init_runtime: Connect to Device Failure (-1), Please make sure the USB connection is normal!
E init_runtime: Catch exception when init runtime!
E init_runtime: Traceback (most recent call last):
E init_runtime:   File "rknn/api/rknn_base.py", line 2469, in rknn.api.rknn_base.RKNNBase.init_runtime
E init_runtime:   File "rknn/api/rknn_runtime.py", line 211, in rknn.api.rknn_runtime.RKNNRuntime.__init__
E init_runtime:   File "rknn/api/rknn_platform.py", line 344, in rknn.api.rknn_platform.start_ntp_or_adb
E init_runtime: Exception: Init runtime environment failed!
W If you can't handle this error, please try updating to the latest version of the toolkit2 and runtime from:
  https://eyun.baidu.com/s/3eTDMk6Y (Pwd: rknn)  Path: RK_NPU_SDK / RK_NPU_SDK_1.X.0 / develop /
  If the error still exists in the latest version, please collect the corresponding error logs and the model,
  convert script, and input data that can reproduce the problem, and then submit an issue on:
  https://redmine.rock-chips.com (Please consult our sales or FAE for the redmine account)
初始化 runtime 失败！
请检查：1.板子 ./rknn_server 是否正在运行？ 2.PC和板子是否能 ping 通？
--> Export rknn model
```
目前直接从Google下载最新版已经是41了，所以必须手动指定下载旧版本：
```bash
wget https://dl.google.com/android/repository/platform-tools_r28.0.1-linux.zip
```
解压后进入目录验证版本:
```bash
triority@ubuntu:~/platform-tools$ ./adb version
Android Debug Bridge version 1.0.40
Version 4986621
Installed as /home/triority/platform-tools/adb
```
现在用这个adb覆盖系统已有那个：
```bash
# 替换 SDK 路径下的
sudo cp adb /usr/lib/android-sdk/platform-tools/adb
# 替换系统指令路径下的
sudo cp adb /usr/bin/adb
```

#### onnx版本误识别
以及我遇到了onnx库版本的bug导致识别onnx文件的opset_version为19导致版本过高不支持报错，需要卸载重装其他版本
```bash
triority@ubuntu:~/Desktop/rknn_test/UIE$ python3 onnx2rknn_adb.py 
I rknn-toolkit2 version: 2.3.0
--> Loading model
I Loading : 100%|███████████████████████████████████████████████| 66/66 [00:00<00:00, 131695.56it/s]
--> Building model
D base_optimize ...
D base_optimize done.
D 
D fold_constant ...
E build: Traceback (most recent call last):
  File "rknn/api/rknn_log.py", line 344, in rknn.api.rknn_log.error_catch_decorator.error_catch_wrapper
  File "rknn/api/rknn_base.py", line 1945, in rknn.api.rknn_base.RKNNBase.build
  File "rknn/api/graph_optimizer.py", line 938, in rknn.api.graph_optimizer.GraphOptimizer.fold_constant
  File "rknn/api/session.py", line 34, in rknn.api.session.Session.__init__
  File "rknn/api/session.py", line 131, in rknn.api.session.Session.sess_build
  File "/home/triority/.local/lib/python3.8/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 360, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "/home/triority/.local/lib/python3.8/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 399, in _create_inference_session
    sess = C.InferenceSession(session_options, self._model_bytes, False, self._read_config_from_model)
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Failed to load model with error: /onnxruntime_src/onnxruntime/core/graph/model_load_utils.h:46 void onnxruntime::model_load_utils::ValidateOpsetForDomain(const std::unordered_map<std::basic_string<char>, int>&, const onnxruntime::logging::Logger&, bool, const string&, int) ONNX Runtime only *guarantees* support for models stamped with official released onnx opset versions. Opset 19 is under development and support for this is limited. The operator schemas and or other functionality may change before next ONNX release and in this case ONNX Runtime will not guarantee backward compatibility. Current official support for domain ai.onnx is till opset 18.


I ===================== WARN(0) =====================
E rknn-toolkit2 version: 2.3.0
Traceback (most recent call last):
  File "rknn/api/rknn_log.py", line 344, in rknn.api.rknn_log.error_catch_decorator.error_catch_wrapper
  File "rknn/api/rknn_base.py", line 1945, in rknn.api.rknn_base.RKNNBase.build
  File "rknn/api/graph_optimizer.py", line 938, in rknn.api.graph_optimizer.GraphOptimizer.fold_constant
  File "rknn/api/session.py", line 34, in rknn.api.session.Session.__init__
  File "rknn/api/session.py", line 131, in rknn.api.session.Session.sess_build
  File "/home/triority/.local/lib/python3.8/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 360, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "/home/triority/.local/lib/python3.8/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 399, in _create_inference_session
    sess = C.InferenceSession(session_options, self._model_bytes, False, self._read_config_from_model)
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Failed to load model with error: /onnxruntime_src/onnxruntime/core/graph/model_load_utils.h:46 void onnxruntime::model_load_utils::ValidateOpsetForDomain(const std::unordered_map<std::basic_string<char>, int>&, const onnxruntime::logging::Logger&, bool, const string&, int) ONNX Runtime only *guarantees* support for models stamped with official released onnx opset versions. Opset 19 is under development and support for this is limited. The operator schemas and or other functionality may change before next ONNX release and in this case ONNX Runtime will not guarantee backward compatibility. Current official support for domain ai.onnx is till opset 18.
```

```bash
pip3 uninstall onnx onnxruntime
pip3 install onnx==1.16.0 onnxruntime
```
