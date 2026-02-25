---
title: 视频水印消除：U-Net结构入门深度学习（一）
cover: /img/0628.png
date: 2025-08-12 15:44:04
categories: 
- [文档&笔记]
tags:
- 笔记
- 神经网络
- python
description: 用于消除去西藏的无人机视频素材的osd数据。以及最近的情况（好几个月没有更新文章啦）
---
# 近期情况
已经几个月没有在网站上更新文章了，已经到了被朋友催更的地步。一部分原因是前段时间在忙活毕业的事情，包括毕业论文和期末考试给我折磨的死去活来（主要是期末考试）；另一部分原因是一直在填坑，之前的好多项目做了一半没做完的，都在接着做，所有虽然没有新建文章但是之前的文章是一直在更新的啦

七月初毕业回家，全家出去玩了几天，回来之后开始谋划我都毕业旅行，打算去西藏和深圳。可惜去西藏的半个月花光了存款，深圳只好暂时放弃，不过考虑到温度，深圳香港寒假再去也是明智的选择。去西藏的经历的文章稍后推出（x

虽然原计划八月份要回学校干活了，但是宿舍没有解决，要在家等到9.2开学再去学校了。计划学点东西，包括深度学习，模电，信号处理（显然后两个是为自制短波台准备的，九月份开学去考b类操作证）以及背点单词准备六级。

去西藏无人机拍摄的素材有osd数据的叠加，虽然已经有现成的软件去除水印，但是吧神经网络学过那么久了从来没正经用过（pytorch学习笔记是已经有两年之久的2023年的文章了），于是就有了这篇文章

# 水印数据
![](微信截图_20250812155953.png)
上图即为穿越机拍摄的视频的截图。为了训练这个神经网络，首先需要大量对应的无水印视频和水印视频。电脑的机械硬盘里有大概700G的电影，都是没有任何水印的蓝光原盘高码率视频，除了超高码率的缺点以外非常适合用来生成数据集。最开始想直接做视频截取图片叠加水印，已经写好了视频抽帧的程序，但是后来想到应该让网络学习连续的帧片段而不是单独处理每一张图片，否则视频的连贯性肯定要出问题。反正两个造数据的程序都放在下面

`video2img.py`:视频截取图片。由于视频码率很高处理特别慢还写了多进程~~写了我好几个小时还没用上~~
```py
import cv2
import os
from tqdm import tqdm
import multiprocessing


num_processes = multiprocessing.cpu_count()
img_time_interval = 5
img_size = (1920, 1080)
img_Dir = 'data\img'
video_Dir = 'D:\movie\data'
video_Type = '.mkv'
start_frame_num = 0

def video2img(video_path, img_path, interval, size, progress_queue, shared_total_counter):
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open file {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval)
    if frame_interval == 0:
        print("Error: Frame interval is calculated as 0.")
        cap.release()
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, size)
            output_image_path = os.path.join(img_path, f"Frame_{shared_total_counter.value}.jpg")
            cv2.imwrite(output_image_path, frame)
            shared_total_counter.value += 1
        frame_count += 1
        progress_queue.put(1)
    cap.release()


def find_video_files(directory, file_type=".mkv"):
    mkv_files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_type):
                full_path = os.path.join(root, file)
                mkv_files_list.append(full_path)
    return mkv_files_list



if __name__ == '__main__':
    video_path_list = find_video_files(video_Dir, video_Type)
    frame_total = 0
    for i in video_path_list:
        cap = cv2.VideoCapture(i)
        frame_total = frame_total + int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    print(f"The total number of frames of the video is {frame_total}")

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    frame_completed = 0
    shared_total_counter = manager.Value('i', start_frame_num)
    pool = multiprocessing.Pool(processes=num_processes)
    pbar = tqdm(total=frame_total, desc="Video frame processing")
    for i in video_path_list:
        pool.apply_async(video2img, args=(i, img_Dir, img_time_interval, img_size, progress_queue, shared_total_counter))

    while frame_completed < frame_total:
        _ = progress_queue.get()
        frame_completed += 1
        pbar.update(1)

    pbar.close()
    pool.close()
    pool.join()
    print("\nCompleted!")

```

`video_mask.py`:视频截取指定帧长度的视频，并随机生成文字掩膜，保存视频片段、带水印的视频片段和掩膜图片。这个程序大部分是用Google AI Studio写的
```py
import cv2
import os
from tqdm import tqdm
import multiprocessing


num_processes = multiprocessing.cpu_count()
img_time_interval = 5
img_size = (1920, 1080)
img_Dir = 'data\img'
video_Dir = 'D:\movie\data'
video_Type = '.mkv'
start_frame_num = 0

def video2img(video_path, img_path, interval, size, progress_queue, shared_total_counter):
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open file {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval)
    if frame_interval == 0:
        print("Error: Frame interval is calculated as 0.")
        cap.release()
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, size)
            output_image_path = os.path.join(img_path, f"Frame_{shared_total_counter.value}.jpg")
            cv2.imwrite(output_image_path, frame)
            shared_total_counter.value += 1
        frame_count += 1
        progress_queue.put(1)
    cap.release()


def find_video_files(directory, file_type=".mkv"):
    mkv_files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_type):
                full_path = os.path.join(root, file)
                mkv_files_list.append(full_path)
    return mkv_files_list



if __name__ == '__main__':
    video_path_list = find_video_files(video_Dir, video_Type)
    frame_total = 0
    for i in video_path_list:
        cap = cv2.VideoCapture(i)
        frame_total = frame_total + int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    print(f"The total number of frames of the video is {frame_total}")

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    frame_completed = 0
    shared_total_counter = manager.Value('i', start_frame_num)
    pool = multiprocessing.Pool(processes=num_processes)
    pbar = tqdm(total=frame_total, desc="Video frame processing")
    for i in video_path_list:
        pool.apply_async(video2img, args=(i, img_Dir, img_time_interval, img_size, progress_queue, shared_total_counter))

    while frame_completed < frame_total:
        _ = progress_queue.get()
        frame_completed += 1
        pbar.update(1)

    pbar.close()
    pool.close()
    pool.join()
    print("\nCompleted!")

```

# 网络结构
由于要处理的数据是运动性极强的FPV视频，需要使用循环神经网络来让网络考虑上一帧内容，以及用卷积神经网络来恢复图像。因此使用带有ConvLSTM的U-Net结构。

输入图像经过4组卷积，维度扩展到512维，然后在这里通过ConvLSTM将当前信息和过去记忆拼接考虑，并通过四个门控更新记忆和生成输出和新的隐藏状态，再经过4组和前面卷积组有跳跃连接的转置卷积恢复图像尺寸，最终得到三维度的RGB输出

```py
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # 将输入门、遗忘门、输出门和细胞门的卷积操作合并计算
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,  # 4 for i, f, o, g gates
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        # 计算4*门
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class RecurrentUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(RecurrentUNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # ConvLSTM瓶颈
        self.bottleneck_dim = features[-1]
        self.conv_lstm = ConvLSTMCell(input_dim=self.bottleneck_dim,
                                      hidden_dim=self.bottleneck_dim,
                                      kernel_size=(3, 3), bias=True)

        # 解码器
        in_channels = features[-1]
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels, feature, kernel_size=2, stride=2))
            self.ups.append(ConvBlock(feature * 2, feature))
            in_channels = feature

        # 输出
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, hidden_state=None):
        # 视频片段x的期望形状:[batch_size, sequence_length, Channels, H, W]
        batch_size, seq_len, _, H, W = x.shape
        if hidden_state is None:
            bottleneck_h, bottleneck_w = H // (2 ** (len(self.downs) - 1)), W // (2 ** (len(self.downs) - 1))
            hidden_state = self.conv_lstm.init_hidden(batch_size, (bottleneck_h, bottleneck_w))
        outputs = []

        # 序列帧循环
        for t in range(seq_len):
            current_frame = x[:, t, :, :, :]
            skip_connections_t = []

            # 编码器
            for i, down in enumerate(self.downs):
                current_frame = down(current_frame)
                skip_connections_t.append(current_frame)
                if i < len(self.downs) - 1:
                    current_frame = self.pool(current_frame)
            # ConvLSTM
            h, c = self.conv_lstm(input_tensor=current_frame, cur_state=hidden_state)
            hidden_state = (h, c)
            current_frame = h
            # 反转跳跃连接列表
            skip_connections_t = skip_connections_t[::-1]

            # 解码器
            for i in range(0, len(self.ups), 2):
                current_frame = self.ups[i](current_frame)
                skip_connection = skip_connections_t[i // 2]
                # 如果池化导致奇数尺寸，上采样后的尺寸与跳跃连接不匹配，则强制修改尺寸
                if current_frame.shape != skip_connection.shape:
                    current_frame = nn.functional.interpolate(current_frame, size=skip_connection.shape[2:])
                concat_skip = torch.cat((skip_connection, current_frame), dim=1)
                current_frame = self.ups[i + 1](concat_skip)

            # 生成帧
            frame_output = self.final_conv(current_frame)
            outputs.append(frame_output)

        return torch.stack(outputs, dim=1), hidden_state
```

# 改进和效果记录
## RGB三维输入
让网络直接学习加了水印的视频，训练了五轮的效果如下（由于显存限制，图像降低分辨率到480*270之后给网络计算）

| {% dplayer "url=b34be09f1bee268c4fc728776988c605.mp4" %} | {% dplayer "url=9a9d89aef96d5c03e600bdaa5e65f646.mp4" %} |
|:---:|:---:|
| 水印视频 | 去水印视频 |
| <img width=2000/> | <img width=2000/> |

看起来网络已经学会了什么样的东西是文字水印并进行了一定处理。于是，为什么要让模型自己学习文字水印长啥样啊我明明有掩膜图片的，无人机视频也能用白色过滤选中水印区域啊，于是稍作修改，改成叠加了掩膜的四个通道输入

## RGB+Mask四维输入
`train.py`: 训练程序
```py
import torch
import torch.nn as nn
import cv2
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pathlib
import torchvision
import datetime
from tqdm import tqdm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # 将输入门、遗忘门、输出门和细胞门的卷积操作合并计算
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,  # 4 for i, f, o, g gates
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        # 计算4*门
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class RecurrentUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(RecurrentUNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # ConvLSTM瓶颈
        self.bottleneck_dim = features[-1]
        self.conv_lstm = ConvLSTMCell(input_dim=self.bottleneck_dim,
                                      hidden_dim=self.bottleneck_dim,
                                      kernel_size=(3, 3), bias=True)

        # 解码器
        in_channels = features[-1]
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels, feature, kernel_size=2, stride=2))
            self.ups.append(ConvBlock(feature * 2, feature))
            in_channels = feature

        # 输出
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, hidden_state=None):
        # 视频片段x的期望形状:[batch_size, sequence_length, Channels, H, W]
        batch_size, seq_len, _, H, W = x.shape
        if hidden_state is None:
            bottleneck_h, bottleneck_w = H // (2 ** (len(self.downs) - 1)), W // (2 ** (len(self.downs) - 1))
            hidden_state = self.conv_lstm.init_hidden(batch_size, (bottleneck_h, bottleneck_w))
        outputs = []

        # 序列帧循环
        for t in range(seq_len):
            current_frame = x[:, t, :, :, :]
            skip_connections_t = []

            # 编码器
            for i, down in enumerate(self.downs):
                current_frame = down(current_frame)
                skip_connections_t.append(current_frame)
                if i < len(self.downs) - 1:
                    current_frame = self.pool(current_frame)
            # ConvLSTM
            h, c = self.conv_lstm(input_tensor=current_frame, cur_state=hidden_state)
            hidden_state = (h, c)
            current_frame = h
            # 反转跳跃连接列表
            skip_connections_t = skip_connections_t[::-1]

            # 解码器
            for i in range(0, len(self.ups), 2):
                current_frame = self.ups[i](current_frame)
                skip_connection = skip_connections_t[i // 2]
                # 如果池化导致奇数尺寸，上采样后的尺寸与跳跃连接不匹配，则强制修改尺寸
                if current_frame.shape != skip_connection.shape:
                    current_frame = nn.functional.interpolate(current_frame, size=skip_connection.shape[2:])
                concat_skip = torch.cat((skip_connection, current_frame), dim=1)
                current_frame = self.ups[i + 1](concat_skip)

            # 生成帧
            frame_output = self.final_conv(current_frame)
            outputs.append(frame_output)

        return torch.stack(outputs, dim=1), hidden_state


class VideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, transform=None, size=(480, 270)):
        self.root_dir = pathlib.Path(root_dir)
        self.clips_dir = self.root_dir / 'clips'
        self.mask_clips_dir = self.root_dir / 'mask_clips'
        self.mask_dir = self.root_dir / 'masks'

        self.clips_files = sorted([p for p in self.clips_dir.glob('*.mp4')])
        self.mask_clips_files = sorted([p for p in self.mask_clips_dir.glob('*.mp4')])
        self.mask_files = sorted([p for p in self.mask_dir.glob('*.png')])
        assert len(self.clips_files) == len(self.mask_clips_files) == len(self.mask_files), "The number of dataset files does not match!"

        self.sequence_length = sequence_length
        self.transform = transform
        # 输入格式(width, height)，PyTorch(height, width)
        self.target_size = size
        self.target_size_torch = (size[1], size[0])

    def __len__(self):
        return len(self.clips_files)

    def __getitem__(self, idx):
        clips_path = str(self.clips_files[idx])
        mask_clips_path = str(self.mask_clips_files[idx])
        mask_path = str(self.mask_files[idx])

        def read_and_resize_frames(video_path, num_frames, size):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < num_frames:
                cap.release()
                raise ValueError(f"Video {video_path} : total_frames ({total_frames}) < num_frames ({num_frames})。")

            frames = []
            start_frame_index = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(TF.to_tensor(frame_rgb))
            cap.release()

            if len(frames) != num_frames:
                raise ValueError(f"Read frame failed: {video_path}")

            return torch.stack(frames)

        clips_seq = read_and_resize_frames(clips_path, self.sequence_length, self.target_size)
        masked_seq = read_and_resize_frames(mask_clips_path, self.sequence_length, self.target_size)
        mask_image = torchvision.io.read_image(str(mask_path))
        mask_image_resized = TF.resize(mask_image, self.target_size_torch, antialias=True)

        # 归一化
        clips_seq = clips_seq * 2.0 - 1.0
        masked_seq = masked_seq * 2.0 - 1.0

        mask_seq = mask_image_resized.float() / 255.0
        mask_seq[mask_seq > 0.5] = 1.0
        mask_seq[mask_seq <= 0.5] = 0.0
        mask_seq = mask_seq.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)
        mask_seq = mask_seq[:, 0:1, :, :]
        masked_seq = torch.cat((masked_seq, mask_seq), dim=1)

        if self.transform:
            pass

        return masked_seq, clips_seq, mask_seq


if __name__ == '__main__':
    lr = 1e-4
    batch_size = 2
    epochs = 50
    sequence_len = 4
    size = (480, 270)
    dataset_loader_workers = 4

    dataset_path = "D:\Dataset"
    # 继续训练时加载模型路径和已完成轮次，路径为空字符串则从零开始训练且设置的轮次无效
    load_model_path = "model\epoch_7.pth"
    load_model_epoch = 7




    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = RecurrentUNet(in_channels=4, out_channels=3).to(device)
    if load_model_path == "":
        load_model_epoch = 0
    else:
        model.load_state_dict(torch.load(load_model_path, map_location=device))

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters.")

    print("Preparing dataset...")
    train_dataset = VideoDataset(root_dir=dataset_path, sequence_length=sequence_len, size=size)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataset_loader_workers,
        pin_memory=False)

    print("Start training...")
    for epoch in range(load_model_epoch, epochs):
        model.train()
        total_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for batch_idx, (masked_seq, clips_seq, mask_seq) in enumerate(train_loader):
                masked_seq = masked_seq.to(device)
                clips_seq = clips_seq.to(device)
                mask_seq = mask_seq.to(device)

                optimizer.zero_grad()
                restored_seq, h_last = model(masked_seq)

                loss = criterion(restored_seq, clips_seq)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                pbar.set_postfix(loss=f'{loss.item():.4f}')
                pbar.update(1)

        avg_loss = total_loss / len(train_loader)
        print(f"--- {datetime.datetime.now():%H:%M:%S}: Epoch {epoch + 1} avg_loss: {avg_loss:.4f} ---")

        torch.save(model.state_dict(), f"model\epoch_{epoch + 1}.pth")

    print("Completed!")

```

`infer.py`:有了训练的代码，这段程序就很容易了（懒得写了），以下内容由Google AI Studio生成。不要问为什么注释序号从2开始，因为第一部分被我整个删掉换成了`from train import RecurrentUNet`
```py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import pathlib
from tqdm import tqdm

from train import RecurrentUNet


# ==============================================================================
# 2. 推理主函数
# ==============================================================================

def postprocess_and_write(output_tensor, writer):
    """辅助函数：后处理并写入文件 (无需修改)"""
    output_tensor = output_tensor.squeeze(0).cpu()
    for i in range(output_tensor.shape[0]):
        frame = (output_tensor[i] + 1.0) / 2.0
        frame = (frame.clamp(0, 1) * 255).byte()
        frame_np = frame.permute(1, 2, 0).numpy()
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)


def infer_video_4channel(config):
    """
    使用4通道输入(RGB+Mask)对视频进行分段推理。
    """
    device = torch.device(config["device"])
    target_size = config["input_size"]
    target_size_torch = (target_size[1], target_size[0])

    print("正在加载模型...")
    # *** 关键改动: in_channels=4 ***
    # 模型输出仍然是修复后的RGB图像，所以 out_channels=3
    model = RecurrentUNet(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval()
    print(f"模型已加载到设备: {device}")

    # --- 新增: 加载并预处理掩膜 ---
    print(f"正在加载并处理掩膜: {config['mask_path']}")
    mask_image = torchvision.io.read_image(config['mask_path'])
    mask_resized = TF.resize(mask_image, target_size_torch, antialias=True)
    mask_tensor = mask_resized.float() / 255.0
    mask_tensor[mask_tensor > 0.5] = 1.0
    mask_tensor[mask_tensor <= 0.5] = 0.0
    # 确保是单通道 [1, H, W]，并放在CPU上以便和每帧拼接
    mask_tensor_cpu = mask_tensor[0:1, :, :]

    print(f"正在处理输入视频: {config['input_video_path']}")
    cap = cv2.VideoCapture(config["input_video_path"])
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {config['input_video_path']}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'VP09')
    out_writer = cv2.VideoWriter(config["output_video_path"], fourcc, original_fps, target_size)

    hidden_state = None
    chunk_frames = []

    with torch.no_grad():
        with tqdm(total=total_frames, desc="正在推理") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. 预处理RGB帧
                frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frame_tensor_3ch = TF.to_tensor(frame_rgb) * 2.0 - 1.0

                # 2. *** 关键改动: 拼接成4通道输入 ***
                # torch.cat 沿着第0维(通道维)拼接 [3,H,W] 和 [1,H,W] -> [4,H,W]
                four_channel_tensor = torch.cat([frame_tensor_3ch, mask_tensor_cpu], dim=0)
                chunk_frames.append(four_channel_tensor)

                if len(chunk_frames) == config["chunk_size"]:
                    input_chunk = torch.stack(chunk_frames).unsqueeze(0).to(device)
                    restored_chunk, hidden_state = model(input_chunk, hidden_state)
                    postprocess_and_write(restored_chunk, out_writer)
                    chunk_frames = []

                pbar.update(1)

            if chunk_frames:
                pbar.set_description("处理最后一段")
                input_chunk = torch.stack(chunk_frames).unsqueeze(0).to(device)
                restored_chunk, hidden_state = model(input_chunk, hidden_state)
                postprocess_and_write(restored_chunk, out_writer)

    cap.release()
    out_writer.release()
    print(f"\n视频推理完成并保存到: {config['output_video_path']}")


# ==============================================================================
# 3. 配置和执行
# ==============================================================================

if __name__ == '__main__':
    inference_config = {
        "model_path": "model/epoch_2.pth",
        # 这是需要修复的视频，例如视频中某些区域被涂黑或有水印
        "input_video_path": "D:\Dataset\mask_clips\\0628.mp4",
        # 这是对应的单张二值化掩膜图片，白色区域代表需要修复的地方
        "mask_path": "D:\Dataset\masks\\0628.png",
        "output_video_path": "restored_video.mp4",
        "input_size": (480, 270),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "chunk_size": 10,
    }

    if not pathlib.Path(inference_config["model_path"]).exists():
        print(f"错误: 模型文件未找到 -> {inference_config['model_path']}")
    elif not pathlib.Path(inference_config["input_video_path"]).exists():
        print(f"错误: 输入视频未找到 -> {inference_config['input_video_path']}")
    # 新增对掩膜文件路径的检查
    elif not pathlib.Path(inference_config["mask_path"]).exists():
        print(f"错误: 掩膜文件未找到 -> {inference_config['mask_path']}")
    else:
        infer_video_4channel(inference_config)

```

| {% dplayer "url=video_0628.mp4" %} | ![](mask0628.png) |
|:---:|:---:|
| 水印视频 | 掩膜图片 |
| {% dplayer "url=restored_epoch2_video.mp4" %} | {% dplayer "url=restored_epoch8_video.mp4" %} |
| 训练2轮去水印效果 | 训练8轮去水印效果 |
| <img width=2000/> | <img width=2000/> |

## 最后卷积层溢出

此时去水印已经有一定效果，但是在纯色区域和动态区域出现了疑似像素值溢出的彩色条带状区域。

{% dplayer "url=明显的白色溢出.mp4" %}

考虑原因可能是网络最后一层是卷积层，输出的值没有范围限制，因此在后面追加了一个tanh层，现在RecurrentUNet类变成这样（只是在__init__和生成帧的两个位置改了两行）

```py
class RecurrentUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(RecurrentUNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # ConvLSTM瓶颈
        self.bottleneck_dim = features[-1]
        self.conv_lstm = ConvLSTMCell(input_dim=self.bottleneck_dim,
                                      hidden_dim=self.bottleneck_dim,
                                      kernel_size=(3, 3), bias=True)

        # 解码器
        in_channels = features[-1]
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels, feature, kernel_size=2, stride=2))
            self.ups.append(ConvBlock(feature * 2, feature))
            in_channels = feature

        # 输出
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden_state=None):
        # 视频片段x的期望形状:[batch_size, sequence_length, Channels, H, W]
        batch_size, seq_len, _, H, W = x.shape
        if hidden_state is None:
            bottleneck_h, bottleneck_w = H // (2 ** (len(self.downs) - 1)), W // (2 ** (len(self.downs) - 1))
            hidden_state = self.conv_lstm.init_hidden(batch_size, (bottleneck_h, bottleneck_w))
        outputs = []

        # 序列帧循环
        for t in range(seq_len):
            current_frame = x[:, t, :, :, :]
            skip_connections_t = []

            # 编码器
            for i, down in enumerate(self.downs):
                current_frame = down(current_frame)
                skip_connections_t.append(current_frame)
                if i < len(self.downs) - 1:
                    current_frame = self.pool(current_frame)
            # ConvLSTM
            h, c = self.conv_lstm(input_tensor=current_frame, cur_state=hidden_state)
            hidden_state = (h, c)
            current_frame = h
            # 反转跳跃连接列表
            skip_connections_t = skip_connections_t[::-1]

            # 解码器
            for i in range(0, len(self.ups), 2):
                current_frame = self.ups[i](current_frame)
                skip_connection = skip_connections_t[i // 2]
                # 如果池化导致奇数尺寸，上采样后的尺寸与跳跃连接不匹配，则强制修改尺寸
                if current_frame.shape != skip_connection.shape:
                    current_frame = nn.functional.interpolate(current_frame, size=skip_connection.shape[2:])
                concat_skip = torch.cat((skip_connection, current_frame), dim=1)
                current_frame = self.ups[i + 1](concat_skip)

            # 生成帧
            frame_output = self.tanh(self.final_conv(current_frame))
            outputs.append(frame_output)

        return torch.stack(outputs, dim=1), hidden_state
```

经过10轮的训练和测试，溢出问题得到解决：

| {% dplayer "url=epoch5_14938_restored_video.mp4" %} | {% dplayer "url=epoch10_14938_restored_video.mp4" %} | {% dplayer "url=epoch10_0628_restored_video.mp4" %} |
|:---:|:---:|:---:|
| 训练5轮去水印效果 | 训练10轮去水印效果 | 与前面的测试作比较 |
| <img width=2000/> | <img width=2000/> | <img width=2000/> |

看起来训练轮次的增加效果不是那么明显，在继续训练的同时排查一下是不是激活函数导致的梯度消失问题。以及考虑是否是损失函数的缺陷，考虑增加对抗损失？

## 梯度消失和可视化

对于梯度消失问题，用TensorBoard进行参数可视化，检查梯度范数。

```py
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs')

for epoch in range(50):

    # 记录梯度范数到TensorBoard，在backward()和step()之间
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 使用writer.add_scalar来记录，标签格式 'grads/层名' 可以在 TensorBoard 中分组
            writer.add_scalar(f'grads/{name}_norm', param.grad.norm(2), epoch)
    # 记录总的梯度范数，以监控梯度爆炸
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    writer.add_scalar('grads/total_norm', total_norm, epoch)

# 结束，关闭writer
writer.close()
```
此时在终端输入启动命令即可在web查看具体图表
```
tensorboard --logdir=runs
```

得到的数据如下，发生梯度消失的可能性比较大，特别是ConvLSTM梯度已经到-7次的数量级，很有可能与此处使用的多个sigmoid和tanh激活函数有关。

| ![](微信截图_20250816140036.png) | ![](微信截图_20250816140051.png) |
|:---:|:---:|
| <img width=2000/> | <img width=2000/> |

考虑增加应该跳跃连接来让梯度传播跳过ConvLSTM层？

```py
            # ConvLSTM
            h, c = self.conv_lstm(input_tensor=current_frame, cur_state=hidden_state)
            hidden_state = (h, c)
            #下一行为改动内容，原来是current_frame = h
            current_frame = h + current_frame
```

反正先试一下，在10轮的基础上继续训练一轮


| {% dplayer "url=epoch10_14938_restored_video.mp4" %} | {% dplayer "url=epoch11_14938_restored_video.mp4" %} |
|:---:|:---:|
| 原来10轮模型效果 | 改动后训练1轮也就是第11轮效果 |
| <img width=2000/> | <img width=2000/> |

效果不太明显，同时发生了图片整体的偏色，当然也不排除训练次数不够的原因，有待继续训练测试，下面是梯度图像

| ![](信截图_20250816151838.png) | ![](信截图_20250816151901.png) |
|:---:|:---:|
| <img width=2000/> | <img width=2000/> |



## 对抗损失函数
使用多层3D卷积网络作为判别器的对抗网络的推理作为损失函数
train_gan.py:
```py
import torch
import torch.nn as nn
import cv2
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pathlib
import torchvision
import datetime
from tqdm import tqdm

from train import RecurrentUNet, VideoDataset

class VideoDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        layers = []
        # 3D卷积输入视频片段[Batch, Channels, Time, Height, Width]，卷积核在时间维度上覆盖了3帧，在空间高度上覆盖4个像素
        # stride滑动步长，在时间维度上每次只移动1帧，在空间上每次移动2个像素，起到下采样的作用
        # padding输入视频数据块的三个维度的两侧填充0
        layers.append(nn.Conv3d(in_channels, features[0], kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)))
        # inplace=True会直接在存储输入数据的内存上进行计算并覆盖，节省一些GPU显存
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(len(features) - 1):
            layers.append(nn.Conv3d(features[i], features[i + 1],kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False))
            layers.append(nn.InstanceNorm3d(features[i + 1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv3d(features[-1], 1, kernel_size=(3, 4, 4), stride=(1, 1, 1), padding=(1, 1, 1)))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    lr_gen = 2e-4
    lr_disc = 2e-4
    L1_weigth = 100
    batch_size = 2
    epochs = 50
    sequence_len = 4
    size = (480, 270)
    dataset_loader_workers = 6

    # 数据集路径
    dataset_path = "D:\Dataset"
    # 继续训练时加载模型路径和已完成轮次，输入0则从零开始训练
    load_model_epoch = 0
    load_model_path_gen = ""
    load_model_path_disc = ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    gen = RecurrentUNet(in_channels=4, out_channels=3).to(device)
    disc = VideoDiscriminator(in_channels=3).to(device)

    if load_model_epoch != 0:
        print(f"Loading Generator from {load_model_path_gen}")
        gen.load_state_dict(torch.load(load_model_path_gen, map_location=device))
        print(f"Loading Discriminator from {load_model_path_disc}")
        disc.load_state_dict(torch.load(load_model_path_disc, map_location=device))

    # Adam优化器，学习率lr，beta1默认值0.9的动量大约是过去10个时间步梯度的平均，降低到0.5降低动量惯性
    # beta2默认值0.999的二阶矩估计大约是过去1000个时间步梯度平方的平均，保持较高的值有助于保持自适应学习率的稳定性，防止因为单次梯度爆炸而导致学习率剧烈变化
    opt_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.5, 0.999))
    # 二元交叉熵和L1损失函数
    adversarial_loss_fn = nn.BCEWithLogitsLoss()
    l1_loss_fn = nn.L1Loss()

    num_params_gen = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    num_params_disc = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"Generator has {num_params_gen:,} trainable parameters.")
    print(f"Discriminator has {num_params_disc:,} trainable parameters.")
    print("Preparing dataset...")
    train_dataset = VideoDataset(root_dir=dataset_path, sequence_length=sequence_len, size=size)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=dataset_loader_workers, pin_memory=False)

    print("Start training...")
    for epoch in range(load_model_epoch, epochs):
        total_loss_g = 0.0
        total_loss_d = 0.0
        gen.train()
        disc.train()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for batch_idx, (masked_seq, clips_seq, mask_seq) in enumerate(train_loader):
                # masked_seq: [B, T, 4, H, W], clips_seq: [B, T, 3, H, W]
                masked_seq = masked_seq.to(device)
                clips_seq = clips_seq.to(device)
                mask_seq = mask_seq.to(device)

                # disc训练
                clips_fake, _ = gen(masked_seq)
                opt_disc.zero_grad()
                # 将视频维度从[B, T, C, H, W]转换到[B, C, T, H, W]以匹配Conv3d
                real_clip_for_disc = clips_seq.permute(0, 2, 1, 3, 4)
                fake_clip_for_disc = clips_fake.permute(0, 2, 1, 3, 4)
                # 判别器分别推理真实视频与全1张量、虚假视频与全0张量，计算二元交叉熵损失
                disc_real = disc(real_clip_for_disc)
                loss_disc_real = adversarial_loss_fn(disc_real, torch.ones_like(disc_real))
                # 用 .detach() 阻止梯度传回生成器
                disc_fake = disc(fake_clip_for_disc.detach())
                loss_disc_fake = adversarial_loss_fn(disc_fake, torch.zeros_like(disc_fake))
                # 判别器总损失
                loss_disc = (loss_disc_real + loss_disc_fake) / 2

                loss_disc.backward()
                opt_disc.step()
                total_loss_d += loss_disc.item()

                # 训练生成器
                opt_gen.zero_grad()
                disc_fake_for_gen = disc(fake_clip_for_disc)
                loss_g_adv = adversarial_loss_fn(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))
                loss_g_l1 = l1_loss_fn(clips_fake, clips_seq) * L1_weigth
                loss_g = loss_g_adv + loss_g_l1
                loss_g.backward()
                opt_gen.step()
                total_loss_g += loss_g.item()

                pbar.set_postfix(
                    Loss_D=f'{loss_disc.item():.4f}',
                    Loss_G=f'{loss_g.item():.4f}',
                    G_adv=f'{loss_g_adv.item():.4f}',
                    G_L1=f'{loss_g_l1.item():.4f}')
                pbar.update(1)

            avg_loss_g = total_loss_g / len(train_loader)
            avg_loss_d = total_loss_d / len(train_loader)
            print(f"--- {datetime.datetime.now():%H:%M:%S}: Epoch {epoch + 1} avg_loss_G: {avg_loss_g:.4f}, avg_loss_D: {avg_loss_d:.4f} ---")

        pathlib.Path("model_gan").mkdir(parents=True, exist_ok=True)
        torch.save(gen.state_dict(), f"model_gan/gen_epoch_{epoch + 1}.pth")
        torch.save(disc.state_dict(), f"model_gan/disc_epoch_{epoch + 1}.pth")

    print("Completed!")

```

使用对抗损失后出现了一些进步和问题。14轮的模型在天空纯色区域明显好于13轮，但是出现少量溢出。再看15轮的结果就只剩溢出了......

| {% dplayer "url=GAN_epoch13_restored_video.mp4" %} | {% dplayer "url=GAN_epoch14_restored_video.mp4" %} | {% dplayer "url=GAN_epoch15_restored_video.mp4" %} |
|:---:|:---:|:---:|
| 13轮 | 14轮 | 15轮 |
| <img width=2000/> | <img width=2000/> | <img width=2000/> |

这部分训练是和前一节的梯度消失解决同时在两台电脑上测试的，因此并没有在convlstm引入跳跃连接，考虑到有可能由此引发这一问题，因此下面是加入了这一结构之后的训练效果。很明显溢出问题再次消失。

## 残差网络

虽然在convLSTM层加入了跳跃连接，但是在u-net部分的卷积层仍然出现了一定的梯度消失。不仅在梯度数据上，实际效果上从十几轮开始几乎就没有进步了。因此将残差网络的思想也加入这里，在卷积块加入了一个1*1的卷积层直连输入输出。卷积层的激活函数也从ReLU改为LeakyReLU，避免出现死亡节点。此外初始化遗忘门值改为1让信息全部通过，在初期避免进入饱和区（后来看这一点收效甚微，循环网络部分依然有严重的梯度消失）。判别器的第一层也加入了归一化层避免梯度消失或爆炸。

此外还增加了tensorboard对训练数据进行检测，包括梯度权重和损失的可视化。

由于额外加入了多个卷积层，对显存的占用更高，因此将输入图像序列长度sequence_len从4改为3，以能在我的8G显存3060ti上运行。由于这样减少了计算量，在我的电脑上单个epoch训练时间从70分钟缩短到55分钟。

由于改动巨大，github上开启了一个新的分支`dev`来和之前的代码`main`相区别。如果后续再有大规模修改这部分内容将以`GAN_GRAD`分支保存并在`dev`继续改进。

tarin.py:
```py
import torch
import torch.nn as nn
import cv2
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pathlib
import torchvision
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

        self.final_activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.convblock(x) + self.shortcut(x)
        return self.final_activation(out)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # 将输入门、遗忘门、输出门和细胞门的卷积操作合并计算
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,  # 4 for i, f, o, g gates
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        if bias:
            # 初始化遗忘门的偏置为1.0，尽可能地保持打开
            forget_gate_bias_start = hidden_dim
            forget_gate_bias_end = 2 * hidden_dim
            self.conv.bias.data[forget_gate_bias_start:forget_gate_bias_end].fill_(1.0)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        # 计算4*门
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class RecurrentUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(RecurrentUNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # ConvLSTM瓶颈
        self.bottleneck_dim = features[-1]
        self.conv_lstm = ConvLSTMCell(input_dim=self.bottleneck_dim,
                                      hidden_dim=self.bottleneck_dim,
                                      kernel_size=(3, 3), bias=True)

        # 解码器
        in_channels = features[-1]
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels, feature, kernel_size=2, stride=2))
            self.ups.append(ConvBlock(feature * 2, feature))
            in_channels = feature

        # 输出
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden_state=None):
        # 视频片段x的期望形状:[batch_size, sequence_length, Channels, H, W]
        batch_size, seq_len, _, H, W = x.shape
        if hidden_state is None:
            bottleneck_h, bottleneck_w = H // (2 ** (len(self.downs) - 1)), W // (2 ** (len(self.downs) - 1))
            hidden_state = self.conv_lstm.init_hidden(batch_size, (bottleneck_h, bottleneck_w))
        outputs = []

        # 序列帧循环
        for t in range(seq_len):
            current_frame = x[:, t, :, :, :]
            skip_connections_t = []

            # 编码器
            for i, down in enumerate(self.downs):
                current_frame = down(current_frame)
                skip_connections_t.append(current_frame)
                if i < len(self.downs) - 1:
                    current_frame = self.pool(current_frame)
            # ConvLSTM
            h, c = self.conv_lstm(input_tensor=current_frame, cur_state=hidden_state)
            hidden_state = (h, c)
            current_frame = h + current_frame
            # 反转跳跃连接列表
            skip_connections_t = skip_connections_t[::-1]

            # 解码器
            for i in range(0, len(self.ups), 2):
                current_frame = self.ups[i](current_frame)
                skip_connection = skip_connections_t[i // 2]
                # 如果池化导致奇数尺寸，上采样后的尺寸与跳跃连接不匹配，则强制修改尺寸
                if current_frame.shape != skip_connection.shape:
                    current_frame = nn.functional.interpolate(current_frame, size=skip_connection.shape[2:])
                concat_skip = torch.cat((skip_connection, current_frame), dim=1)
                current_frame = self.ups[i + 1](concat_skip)

            # 生成帧
            frame_output = self.tanh(self.final_conv(current_frame))
            outputs.append(frame_output)

        return torch.stack(outputs, dim=1), hidden_state


class VideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, transform=None, size=(480, 270)):
        self.root_dir = pathlib.Path(root_dir)
        self.clips_dir = self.root_dir / 'clips'
        self.mask_clips_dir = self.root_dir / 'mask_clips'
        self.mask_dir = self.root_dir / 'masks'

        self.clips_files = sorted([p for p in self.clips_dir.glob('*.mp4')])
        self.mask_clips_files = sorted([p for p in self.mask_clips_dir.glob('*.mp4')])
        self.mask_files = sorted([p for p in self.mask_dir.glob('*.png')])
        assert len(self.clips_files) == len(self.mask_clips_files) == len(self.mask_files), "The number of dataset files does not match!"

        self.sequence_length = sequence_length
        self.transform = transform
        # 输入格式(width, height)，PyTorch(height, width)
        self.target_size = size
        self.target_size_torch = (size[1], size[0])

    def __len__(self):
        return len(self.clips_files)

    def __getitem__(self, idx):
        clips_path = str(self.clips_files[idx])
        mask_clips_path = str(self.mask_clips_files[idx])
        mask_path = str(self.mask_files[idx])

        def read_and_resize_frames(video_path, num_frames, size):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < num_frames:
                cap.release()
                raise ValueError(f"Video {video_path} : total_frames ({total_frames}) < num_frames ({num_frames})。")

            frames = []
            start_frame_index = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(TF.to_tensor(frame_rgb))
            cap.release()

            if len(frames) != num_frames:
                raise ValueError(f"Read frame failed: {video_path}")

            return torch.stack(frames)

        clips_seq = read_and_resize_frames(clips_path, self.sequence_length, self.target_size)
        masked_seq = read_and_resize_frames(mask_clips_path, self.sequence_length, self.target_size)
        mask_image = torchvision.io.read_image(str(mask_path))
        mask_image_resized = TF.resize(mask_image, self.target_size_torch, antialias=True)

        # 归一化
        clips_seq = clips_seq * 2.0 - 1.0
        masked_seq = masked_seq * 2.0 - 1.0

        mask_seq = mask_image_resized.float() / 255.0
        mask_seq[mask_seq > 0.5] = 1.0
        mask_seq[mask_seq <= 0.5] = 0.0
        mask_seq = mask_seq.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)
        mask_seq = mask_seq[:, 0:1, :, :]
        masked_seq = torch.cat((masked_seq, mask_seq), dim=1)

        if self.transform:
            pass

        return masked_seq, clips_seq, mask_seq


if __name__ == '__main__':
    lr = 1e-4
    batch_size = 2
    epochs = 50
    sequence_len = 4
    size = (480, 270)
    dataset_loader_workers = 6

    dataset_path = r"D:/Dataset"
    # 继续训练时加载模型路径和已完成轮次，路径为空字符串则从零开始训练且设置的轮次无效
    load_model_path = r"model/epoch_10.pth"
    load_model_epoch = 10


    writer = SummaryWriter(r'runs/gradient_monitoring')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = RecurrentUNet(in_channels=4, out_channels=3).to(device)
    if load_model_path == "":
        load_model_epoch = 0
    else:
        model.load_state_dict(torch.load(load_model_path, map_location=device))

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters.")

    print("Preparing dataset...")
    train_dataset = VideoDataset(root_dir=dataset_path, sequence_length=sequence_len, size=size)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataset_loader_workers,
        pin_memory=False)

    print("Start training...")
    for epoch in range(load_model_epoch, epochs):
        model.train()
        total_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for batch_idx, (masked_seq, clips_seq, mask_seq) in enumerate(train_loader):
                masked_seq = masked_seq.to(device)
                clips_seq = clips_seq.to(device)
                mask_seq = mask_seq.to(device)

                optimizer.zero_grad()
                restored_seq, h_last = model(masked_seq)

                loss = criterion(restored_seq, clips_seq)
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # 使用writer.add_scalar来记录，标签格式 'grads/层名' 可以在 TensorBoard 中分组
                        writer.add_scalar(f'grads/{name}_norm', param.grad.norm(2), epoch)
                # 记录总的梯度范数，以监控梯度爆炸
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                writer.add_scalar('grads/total_norm', total_norm, epoch)

                optimizer.step()
                total_loss += loss.item()

                pbar.set_postfix(loss=f'{loss.item():.4f}')
                pbar.update(1)

        avg_loss = total_loss / len(train_loader)
        print(f"--- {datetime.datetime.now():%H:%M:%S}: Epoch {epoch + 1} avg_loss: {avg_loss:.4f} ---")

        torch.save(model.state_dict(), f"model/epoch_{epoch + 1}.pth")

    writer.close()
    print("Completed!")

```

tarin_GAN.py:
```py
import torch
import torch.nn as nn
import cv2
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pathlib
import torchvision
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from train import RecurrentUNet, VideoDataset

class VideoDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        layers = []
        # 3D卷积输入视频片段[Batch, Channels, Time, Height, Width]，卷积核在时间维度上覆盖了3帧，在空间高度上覆盖4个像素
        # stride滑动步长，在时间维度上每次只移动1帧，在空间上每次移动2个像素，起到下采样的作用
        # padding输入视频数据块的三个维度的两侧填充0
        layers.append(nn.Conv3d(in_channels, features[0], kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)))
        layers.append(nn.InstanceNorm3d(features[0]))
        # inplace=True会直接在存储输入数据的内存上进行计算并覆盖，节省一些GPU显存
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(len(features) - 1):
            layers.append(nn.Conv3d(features[i], features[i + 1],kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False))
            layers.append(nn.InstanceNorm3d(features[i + 1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv3d(features[-1], 1, kernel_size=(3, 4, 4), stride=(1, 1, 1), padding=(1, 1, 1)))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    lr_gen = 2e-4
    lr_disc = 2e-4
    L1_weigth = 100
    batch_size = 2
    epochs = 100
    sequence_len = 3
    size = (480, 270)
    dataset_loader_workers = 6
    Gradient_intervals = 50

    # 数据集和模型保存路径
    dataset_path = r"D:/Dataset"
    model_save_dir = r"model_gan_2"
    # 继续训练时加载模型路径和已完成轮次，输入0则从零开始训练
    load_model_epoch = 9
    load_model_path_gen = r"model_gan_2/gen_epoch_9.pth"
    load_model_path_disc = r"model_gan_2/disc_epoch_9.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    writer = SummaryWriter('runs/GAN_2')

    gen = RecurrentUNet(in_channels=4, out_channels=3).to(device)
    disc = VideoDiscriminator(in_channels=3).to(device)

    if load_model_epoch != 0:
        print(f"Loading Generator from {load_model_path_gen}")
        gen.load_state_dict(torch.load(load_model_path_gen, map_location=device))
        print(f"Loading Discriminator from {load_model_path_disc}")
        disc.load_state_dict(torch.load(load_model_path_disc, map_location=device))

    # Adam优化器，学习率lr，beta1默认值0.9的动量大约是过去10个时间步梯度的平均，降低到0.5降低动量惯性
    # beta2默认值0.999的二阶矩估计大约是过去1000个时间步梯度平方的平均，保持较高的值有助于保持自适应学习率的稳定性，防止因为单次梯度爆炸而导致学习率剧烈变化
    opt_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.5, 0.999))
    # 二元交叉熵和L1损失函数
    adversarial_loss_fn = nn.BCEWithLogitsLoss()
    l1_loss_fn = nn.L1Loss()

    num_params_gen = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    num_params_disc = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"Generator has {num_params_gen:,} trainable parameters.")
    print(f"Discriminator has {num_params_disc:,} trainable parameters.")
    print("Preparing dataset...")
    train_dataset = VideoDataset(root_dir=dataset_path, sequence_length=sequence_len, size=size)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=dataset_loader_workers, pin_memory=False)

    writer.add_graph(gen, next(iter(train_loader))[0].to(device))
    writer.add_graph(disc, next(iter(train_loader))[1].permute(0, 2, 1, 3, 4).to(device))

    print("Start training...")
    for epoch in range(load_model_epoch, epochs):
        total_loss_g = 0.0
        total_loss_d = 0.0
        total_loss_g_L1 = 0.0
        total_loss_g_adv = 0.0
        gen.train()
        disc.train()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for batch_idx, (masked_seq, clips_seq, mask_seq) in enumerate(train_loader):
                # masked_seq: [B, T, 4, H, W], clips_seq: [B, T, 3, H, W]
                masked_seq = masked_seq.to(device)
                clips_seq = clips_seq.to(device)
                mask_seq = mask_seq.to(device)

                # disc训练
                clips_fake, _ = gen(masked_seq)
                opt_disc.zero_grad()
                # 将视频维度从[B, T, C, H, W]转换到[B, C, T, H, W]以匹配Conv3d
                real_clip_for_disc = clips_seq.permute(0, 2, 1, 3, 4)
                fake_clip_for_disc = clips_fake.permute(0, 2, 1, 3, 4)
                # 判别器分别推理真实视频与全1张量、虚假视频与全0张量，计算二元交叉熵损失
                disc_real = disc(real_clip_for_disc)
                loss_disc_real = adversarial_loss_fn(disc_real, torch.ones_like(disc_real))
                # 用 .detach() 阻止梯度传回生成器
                disc_fake = disc(fake_clip_for_disc.detach())
                loss_disc_fake = adversarial_loss_fn(disc_fake, torch.zeros_like(disc_fake))
                # 判别器总损失
                loss_disc = (loss_disc_real + loss_disc_fake) / 2
                loss_disc.backward()
                # 记录梯度权重
                if batch_idx % Gradient_intervals == 0:
                    for name, param in disc.named_parameters():
                        if param.grad is not None:
                            # 使用 f-string 为每个梯度直方图创建唯一的、有组织的标签
                            # 'Gradients/' 会在 TensorBoard 中创建一个名为 Gradients 的分组
                            writer.add_histogram(
                                tag=f'Grad_disc/{name}',values=param.grad,global_step=epoch * len(train_loader) + batch_idx)
                opt_disc.step()

                # 训练生成器
                opt_gen.zero_grad()
                disc_fake_for_gen = disc(fake_clip_for_disc)
                loss_g_adv = adversarial_loss_fn(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))
                loss_g_l1 = l1_loss_fn(clips_fake, clips_seq) * L1_weigth
                loss_g = loss_g_adv + loss_g_l1
                loss_g.backward()
                if batch_idx % Gradient_intervals == 0:
                    for name, param in gen.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(tag=f'Grad_gan/{name}',values=param.grad,global_step=epoch * len(train_loader) + batch_idx)
                opt_gen.step()

                # 统计记录
                total_loss_g += loss_g.item()
                total_loss_d += loss_disc.item()
                total_loss_g_L1 += loss_g_l1.item()
                total_loss_g_adv += loss_g_adv.item()
                writer.add_scalar('Loss/loss_g_l1', loss_g_l1.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/loss_g_adv', loss_g_adv.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/loss_g', loss_g.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/loss_disc', loss_disc.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/D_real', loss_disc_real.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/D_fake', loss_disc_fake.item(), epoch * len(train_loader) + batch_idx)

                pbar.set_postfix(
                    D_real=f'{loss_disc_real.item():.4f}',
                    D_fake=f'{loss_disc_fake.item():.4f}',
                    Loss_D=f'{loss_disc.item():.4f}',
                    Loss_G=f'{loss_g.item():.4f}',
                    G_adv=f'{loss_g_adv.item():.4f}',
                    G_L1=f'{loss_g_l1.item():.4f}')
                pbar.update(1)

            avg_loss_g = total_loss_g / len(train_loader)
            avg_loss_d = total_loss_d / len(train_loader)
            avg_loss_g_L1 = total_loss_g_L1 / len(train_loader)
            avg_loss_d_adv = total_loss_g_adv / len(train_loader)
            print(f"--- {datetime.datetime.now():%H:%M:%S}: Epoch {epoch + 1} avg_loss_G: {avg_loss_g:.4f}, avg_loss_D: {avg_loss_d:.4f}, avg_loss_g_L1: {avg_loss_g_L1:.4f}, avg_loss_d_adv: {avg_loss_d_adv:.4f} ---")

        pathlib.Path("model_gan").mkdir(parents=True, exist_ok=True)
        torch.save(gen.state_dict(), f"{model_save_dir}/gen_epoch_{epoch + 1}.pth")
        torch.save(disc.state_dict(), f"{model_save_dir}/disc_epoch_{epoch + 1}.pth")

    writer.close()
    print("Completed!")

```

经过测试，效果提升极其显著。这里将前一两节内容结合得到的模型和多处改进的当前模型的推理结果进行对比，两者在5、10、20、30轮的效果分别如何：

| --- | 仅在循环网络部分加入一处跳跃连接 | 卷积层也加入残差网络等修改之后 |
|:---:|:---:|:---:|
| 5轮 | {% dplayer "url=GAN_1_5_restored_video.mp4" %} | {% dplayer "url=GAN_2_5_restored_video.mp4" %} |
| 10轮 | {% dplayer "url=GAN_1_10_restored_video.mp4" %} | {% dplayer "url=GAN_2_10_restored_video.mp4" %} |
| 20轮 | {% dplayer "url=GAN_1_20_restored_video.mp4" %} | {% dplayer "url=GAN_2_20_restored_video.mp4" %} |
| 30轮 | {% dplayer "url=GAN_1_30_restored_video.mp4" %} | {% dplayer "url=GAN_2_30_restored_video.mp4" %} |
| <img width=2000/> | <img width=2000/> | <img width=2000/> |

