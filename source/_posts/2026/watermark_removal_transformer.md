---
title: 视频水印消除：Transformer大法好（二）
cover: /img/0628.png
date: 2026-02-23 15:44:04
categories: 
- [文档&笔记]
tags:
- 笔记
- 神经网络
- python
description: 改进之前的模型，加入多尺度的transformer，效果极佳
---
# 前言
半年前的暑假做这个让自己学会了如何设计深度学习网络，但是使用纯卷积在复杂纹理区域依然难以学习到精确到像素级的纹理的重建和运动，因此效果并不能到完美的地步。考虑到我打算做使用transformer的雷达点云预处理，这里也把transformer加入曾经的纯U-Net结构实践一下transformer的使用。同时由于之前的文章加入了特别多的模块，不知道每一部分的实际效果到底如何，也打算做一次消融实验(来自未来自己的补充：消融实验有点懒得做了，做完transformer之后发现纯卷积效果怎么做都不如现在)。

先上推理效果：

| 输入视频  | {% dplayer "url=post_input.mp4" %}  |
| :------------: | :------------: |
| 水印掩膜  | {% dplayer "url=post_mask.mp4" %}  |
| 修复效果  | {% dplayer "url=SeqTransUNet_epoch_299.mp4" %}  |
| 掩膜叠加  | {% dplayer "url=overlay_show.mp4" %}  |

可以看到在运动的具有复杂纹理的山体岩石等区域，模型输出的画面依然没有破绽
# 训练数据
首先是对数据集的改进。之前是在视频上叠加了随机的文字水印，为了提高随机性，现在改成了随机颜色和内容的水印且随时间随机变化。上一次的模型在处理不同分辨率的视频是效果不好，因此这次的数据集类指定输出分辨率的处理方法不再是直接调整而是从输入视频随机裁剪然后再resize，完美解决了将来分块推理无法适应的问题。

数据集类video_dataset.py：
```python
import torch
import cv2
import pathlib
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class VideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, size=(640, 352)):
        self.root_dir = pathlib.Path(root_dir)
        self.watermarked_dir = self.root_dir / 'watermarked_videos'
        self.mask_dir = self.root_dir / 'mask_videos'
        self.original_dir = self.root_dir / 'original_clips'

        self.watermarked_files = sorted([p for p in self.watermarked_dir.glob('*.mp4')])
        self.mask_files = sorted([p for p in self.mask_dir.glob('*.mp4')])
        self.original_files = sorted([p for p in self.original_dir.glob('*.mp4')])

        self.sequence_length = sequence_length
        self.target_size = size  # (Width, Height), 默认 (640, 352)
        self.aspect_ratio = size[0] / size[1]

    def __len__(self):
        return len(self.watermarked_files)

    def _get_random_crop_params(self, video_path):
        """计算保持比例的随机裁剪坐标"""
        cap = cv2.VideoCapture(str(video_path))
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if w_orig / h_orig > self.aspect_ratio:
            max_h = h_orig
            max_w = int(max_h * self.aspect_ratio)
        else:
            max_w = w_orig
            max_h = int(max_w / self.aspect_ratio)

        crop_w = random.randint(self.target_size[0], max_w)
        crop_h = int(crop_w / self.aspect_ratio)

        x = random.randint(0, w_orig - crop_w)
        y = random.randint(0, h_orig - crop_h)

        return (x, y, crop_w, crop_h)

    def _read_frames(self, path, crop_params, is_mask=False):
        x, y, w, h = crop_params
        cap = cv2.VideoCapture(str(path))
        frames = []
        
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret: break
            
            # 裁剪 [y:y+h, x:x+w]
            frame = frame[y:y+h, x:x+w]

            # 缩放
            interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
            frame = cv2.resize(frame, self.target_size, interpolation=interp)

            if is_mask:
                if len(frame.shape) == 3: 
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
                frame = TF.to_tensor(frame)  # [1, H, W]
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 归一化 [-1, 1]
                frame = TF.to_tensor(frame) * 2.0 - 1.0
            
            frames.append(frame)
        
        cap.release()

        # 补齐帧数
        while len(frames) < self.sequence_length:
            frames.append(frames[-1] if frames else torch.zeros((1 if is_mask else 3, *self.target_size[::-1])))

        return torch.stack(frames)  # [T, C, H, W]

    def __getitem__(self, idx):
        crop_params = self._get_random_crop_params(self.watermarked_files[idx])

        watermarked = self._read_frames(self.watermarked_files[idx], crop_params)
        mask = self._read_frames(self.mask_files[idx], crop_params, is_mask=True)
        original = self._read_frames(self.original_files[idx], crop_params)

        # 拼接RGB + Mask [T, 4, H, W]
        masked_input = torch.cat([watermarked, mask], dim=1)
        
        return masked_input, original, mask
    
```
# 模型结构和损失函数
依然以之前的U-net为骨架，但是加入了时空分治全局注意力、多层次的局部窗口注意力、图像频率和光流深层特征的约束。

首先是U-net网络对于复杂纹理的效果总是倾向于模糊处理，虽然这可能是由L1损失函数导致的，但是即便后面加入了GAN等依然肉眼可见的模糊。在U-Net网络的底部，分辨率最低且维度最高的位置加入了能联系整个视频序列的transformer层，虽然前几轮初期收敛较慢，但是十几轮以后效果明显进步，能够融合整体特征了。因此继续在U-net的上采样之后继续加入空间上局部窗口时间上整体序列的两层transformer，模型已经可以学会根据周围运动和纹理补全水印区域了。

损失函数上，除了基础的L1损失保证画面整体，在水印处针对性的增加了3倍的额外L1损失，使模型能够更专注于水印区域的修补。然后加入了针对水印区域的FFT损失，比较修复之后和原视频在频域上的差别，让模型在复杂纹理区域倾向于产生复杂纹理而非平均化的模糊。此时模型已经可以学会在复杂纹理区域进行填补，但是一旦视频连续播放，复杂纹理区域依然有明显的运动不连贯性。因此最后加入了光流特征损失，使用预训练的RAFT光流模型计算模型输出的序列帧之间的位移，根据当前帧的位移向量从上一帧寻找数据填补，将填补后的画面和模型输出画面的VGG特征进行比较，保证时间上复杂纹理的连贯性。

训练代码：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import os
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
from torchvision import models
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights 
from video_dataset import VideoDataset 


def flow_warp(x, flow):
    """
    利用光流对图像进行重投影
    x: (B, C, H, W)
    flow: (B, 2, H, W)
    """
    B, C, H, W = x.size()
    # 生成网格
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float().to(x.device).unsqueeze(0).expand(B, -1, -1, -1)
    
    # 加上光流并归一化到 [-1, 1]
    target_grid = grid + flow.permute(0, 2, 3, 1)
    target_grid[:,:,:,0] = 2.0 * target_grid[:,:,:,0] / max(W-1, 1) - 1.0
    target_grid[:,:,:,1] = 2.0 * target_grid[:,:,:,1] / max(H-1, 1) - 1.0
    
    return F.grid_sample(x, target_grid, mode='bilinear', padding_mode='reflection', align_corners=True)


class VGGPerceptualLoss(nn.Module):
    """
    掩码引导的前两层 VGG 感知损失
    """
    def __init__(self, device):
        super().__init__()
        # 加载 VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.slice1 = nn.Sequential(*vgg[:4])  # conv1_2
        self.slice2 = nn.Sequential(*vgg[4:9]) # conv2_2
        for param in self.parameters():
            param.requires_grad = False
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, pred, target, mask):
        # 归一化
        pred = ((pred + 1) / 2 - self.mean) / self.std
        target = ((target + 1) / 2 - self.mean) / self.std
        
        # 提取特征
        h_pred1 = self.slice1(pred)
        h_target1 = self.slice1(target)
        h_pred2 = self.slice2(h_pred1)
        h_target2 = self.slice2(h_target1)
        
        # 掩码缩放
        m1 = F.interpolate(mask, size=h_pred1.shape[-2:], mode='nearest')
        m2 = F.interpolate(mask, size=h_pred2.shape[-2:], mode='nearest')
        
        # 计算损失
        loss = F.l1_loss(h_pred1 * m1, h_target1 * m1) + \
               F.l1_loss(h_pred2 * m2, h_target2 * m2)
        return loss

def mask_guided_fft_loss(pred, target, mask):
    pred_masked = pred * mask
    target_masked = target * mask
    pred_fft = torch.fft.rfft2(pred_masked, dim=(-2, -1), norm='ortho')
    target_fft = torch.fft.rfft2(target_masked, dim=(-2, -1), norm='ortho')
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

def warping_loss(vgg_loss_fn, pred_curr, pred_prev, flow_gt, mask):
    """
    光流重投影感知损失：
    1. 用 GT 光流将 pred_prev 变换到当前时刻 -> warped_prev
    2. 比较 warped_prev 和 pred_curr 在 VGG 空间的特征差异
    """
    warped_prev = flow_warp(pred_prev, flow_gt)
    return vgg_loss_fn(pred_curr, warped_prev, mask)


class WindowVideoAttention(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()
        self.w = window_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, T, H, W):
        B, T, N, C = x.shape
        x = x.view(B, T, H, W, C).view(B, T, H // self.w, self.w, W // self.w, self.w, C)
        x = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, T * self.w * self.w, C)
        qkv = self.qkv(x).reshape(-1, T * self.w * self.w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        x = attn.transpose(1, 2).reshape(-1, T * self.w * self.w, C)
        x = self.proj(x).view(B, H // self.w, W // self.w, T, self.w, self.w, C)
        x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, T, H * W, C)
        return x

class RefineBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim); self.attn = WindowVideoAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim); self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
    def forward(self, x, T, H, W):
        x = x + self.attn(self.norm1(x), T, H, W); x = x + self.mlp(self.norm2(x))
        return x

class FlashDividedAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim)
    def forward(self, x, T, S):
        B, N, C = x.shape; H, D = self.num_heads, C // self.num_heads
        xt = x.view(B, T, S, C).transpose(1, 2).reshape(B * S, T, C)
        qkv_t = self.qkv(xt).reshape(B * S, T, 3, H, D).permute(2, 0, 3, 1, 4)
        xt = F.scaled_dot_product_attention(qkv_t[0], qkv_t[1], qkv_t[2])
        x = xt.transpose(1, 2).reshape(B * S, T, C).view(B, S, T, C).transpose(1, 2).reshape(B, T * S, C)
        xs = x.view(B * T, S, C); qkv_s = self.qkv(xs).reshape(B * T, S, 3, H, D).permute(2, 0, 3, 1, 4)
        xs = F.scaled_dot_product_attention(qkv_s[0], qkv_s[1], qkv_s[2])
        return self.proj(xs.transpose(1, 2).reshape(B, T * S, C))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim); self.attn = FlashDividedAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim); self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
    def forward(self, x, T, S):
        x = x + self.attn(self.norm1(x), T, S); x = x + self.mlp(self.norm2(x))
        return x


class VideoTransUNet(nn.Module):
    def __init__(self, embed_dim=256, depth=6, num_heads=8, seq_len=10):
        super().__init__()
        self.mid_idx = seq_len // 2
        # 定义输出的 5 个帧索引：[3, 4, 5, 6, 7]
        self.out_indices = [self.mid_idx - 2, self.mid_idx - 1, self.mid_idx, self.mid_idx + 1, self.mid_idx + 2]
        
        self.enc1 = nn.Sequential(nn.Conv2d(4, 64, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(128, embed_dim, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len * 44 * 80, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.refine_1_4 = RefineBlock(dim=128, num_heads=4)
        self.refine_1_2 = RefineBlock(dim=64, num_heads=4)
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.final = nn.Sequential(nn.Conv2d(32, 3, 3, padding=1), nn.Tanh())

    def get_gaussian_kernel(self, kernel_size=15, sigma=4, device='cuda'):
        x = torch.arange(kernel_size).to(device) - kernel_size // 2
        gaussian = torch.exp(-x.pow(2) / (2 * sigma**2))
        kernel = (gaussian / gaussian.sum()).view(1, 1, kernel_size, 1) * (gaussian / gaussian.sum()).view(1, 1, 1, kernel_size)
        return kernel, kernel_size // 2

    def forward(self, rgb_seq, mask_seq):
        B, T, _, H, W = rgb_seq.shape
        m_flat = mask_seq.view(B * T, 1, H, W)
        g_kernel, pad = self.get_gaussian_kernel(15, 4, m_flat.device)
        soft_m = F.conv2d(m_flat, g_kernel, padding=pad)
        
        inp = torch.cat([rgb_seq, mask_seq], dim=2).view(B * T, 4, H, W)
        f1 = self.enc1(inp); f2 = self.enc2(f1); f3 = self.enc3(f2)
        m_s8 = F.interpolate(soft_m, scale_factor=0.125, mode='bilinear')
        m_s4 = F.interpolate(soft_m, scale_factor=0.25, mode='bilinear')
        m_s2 = F.interpolate(soft_m, scale_factor=0.5, mode='bilinear')
        
        h3, w3 = H // 8, W // 8
        f3_input = f3 * (1 - m_s8) 
        t_in = f3_input.view(B, T, -1, h3*w3).permute(0, 1, 3, 2).reshape(B, T * h3*w3, -1) + self.pos_embed
        for block in self.blocks: t_in = block(t_in, T, h3*w3)
        t_out = t_in.view(B, T, h3*w3, -1).transpose(2, 3).view(B*T, -1, h3, w3)
        
        # 1/4 细化
        d1 = self.dec1(t_out) 
        h4, w4 = H // 4, W // 4
        d1_seq = self.refine_1_4(d1.view(B, T, 128, h4*w4).permute(0, 1, 3, 2), T, h4, w4)
        
        # 1/2 细化
        d1_all = d1_seq.transpose(2, 3).reshape(B*T, 128, h4, w4)
        f2_all = f2 * (1 - m_s4)
        d2 = self.dec2(torch.cat([d1_all, f2_all], dim=1))
        h2, w2 = H // 2, W // 2
        d2_seq = self.refine_1_2(d2.view(B, T, 64, h2*w2).permute(0, 1, 3, 2), T, h2, w2)
        
        # 输出中间 5 帧
        preds = []
        for idx in self.out_indices:
            d2_curr = d2_seq[:, idx].transpose(1, 2).view(B, 64, h2, w2)
            m_s2_curr = m_s2.view(B, T, 1, h2, w2)[:, idx]
            f1_curr = f1.view(B, T, 64, h2, w2)[:, idx] * (1 - m_s2_curr)
            d3 = self.dec3(torch.cat([d2_curr, f1_curr], dim=1))
            preds.append(self.final(d3))
            
        return torch.stack(preds, dim=1) # (B, 5, 3, H, W)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'; os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_worker(rank, world_size):
    root_dir = r"/media/B/Triority/Dataset"
    save_dir = "model_save/transunet_seq"
    start_epoch = 190
    num_epochs = 300
    seq_len = 10
    batch_size = 3
    
    # 损失权重配置
    weights = {
        'l1_global': 1.0,
        'l1_watermark': 3.0,
        'fft': 0.2,
        'vgg': 0.1,
        'warp': 0.2
    }

    setup(rank, world_size)
    if rank == 0: os.makedirs(save_dir, exist_ok=True)

    dataset = VideoDataset(root_dir=root_dir, sequence_length=seq_len, size=(640, 352))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)

    model = VideoTransUNet(seq_len=seq_len).to(rank)
    if start_epoch > 0:
        ckpt = os.path.join(save_dir, f"epoch_{start_epoch-1}.pth")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location={'cuda:0':f'cuda:{rank}'}), strict=False)

    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    criterion = nn.L1Loss()
    scaler = GradScaler('cuda')
    
    vgg_loss_fn = VGGPerceptualLoss(rank)
    
    flow_model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(rank).eval()
    for p in flow_model.parameters(): p.requires_grad = False

    for epoch in range(start_epoch, num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_epoch_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=(rank != 0), desc=f"Epoch {epoch}")
        
        for i, (input_data, original_seq, mask_seq) in pbar:
            rgb_seq = input_data[:, :, :3, :, :].to(rank)
            mask_seq = mask_seq.to(rank)
            target_seq = original_seq.to(rank)

            # 掩码膨胀半径2px
            mask_for_pool = mask_seq.transpose(1, 2)
            mask_seq_dilated = F.max_pool3d(mask_for_pool, (1, 5, 5), stride=1, padding=(0, 2, 2)).transpose(1, 2)
            
            # 提取 5 个输出帧对应的 GT 和 Mask
            out_indices = model.module.out_indices
            target_out = target_seq[:, out_indices]
            mask_out = mask_seq_dilated[:, out_indices]

            optimizer.zero_grad()
            with autocast('cuda'):
                # (B, 5, 3, H, W)
                preds = model(rgb_seq, mask_seq_dilated)
                
                loss_spatial = 0
                loss_warp = 0
                
                # 空间损失计算 (L1, FFT, VGG)
                for t in range(5):
                    m = mask_out[:, t]
                    l1 = criterion(preds[:, t], target_out[:, t]) + weights['l1_watermark'] * criterion(preds[:, t]*m, target_out[:, t]*m)
                    fft = weights['fft'] * mask_guided_fft_loss(preds[:, t], target_out[:, t], m)
                    vgg = weights['vgg'] * vgg_loss_fn(preds[:, t], target_out[:, t], m)
                    loss_spatial += l1 + fft + vgg
                
                loss_spatial /= 5
                
                # 时间损失计算Warping Perceptual
                for t in range(1, 5):
                    # GT 光流 (target[t-1] -> target[t])需要输入 [-1, 1] 区间使用原始 GT 序列的对应帧
                    idx_curr = out_indices[t]
                    idx_prev = out_indices[t-1]
                    
                    with torch.no_grad():
                        # RAFT 推理
                        flow_gt = flow_model(target_seq[:, idx_curr], target_seq[:, idx_prev])[-1]
                    
                    # 将生成的上一帧warp 到当前帧的位置和生成的当前帧在 VGG 空间对比计算 Warping Loss
                    loss_warp += warping_loss(vgg_loss_fn, preds[:, t], preds[:, t-1], flow_gt, mask_out[:, t])

                loss_warp = (loss_warp / 4) * weights['warp']
                
                total_loss = loss_spatial + loss_warp

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_epoch_loss += total_loss.item()
            if rank == 0:
                pbar.set_postfix({'spat': f"{loss_spatial.item():.3f}", 'warp': f"{loss_warp.item():.3f}"})

        if rank == 0:
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pth"))
    dist.destroy_process_group()

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    world_size = 4
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)

```
# 推理
## 效果验证
实际训练过程中多层transformer是先后添加的。在1/8处的transformer层训练了70轮之后才加入1/4位置的，然后在150轮才加入1/2位置的，从250轮左右之后效果就已经很好了，最后继续训练到300轮。由于模型定义不一致，前面的模型已经无法使用，因此这里直接给出[290轮的模型权重](epoch_290.pth)和经过低学习率微调的[300轮模型权重](epoch_290.pth)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from tqdm import tqdm


class WindowVideoAttention(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()
        self.w = window_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, T, H, W):
        B, T, N, C = x.shape
        x = x.view(B, T, H, W, C)
        x = x.view(B, T, H // self.w, self.w, W // self.w, self.w, C)
        x = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        curr_shape = x.shape
        x = x.view(-1, T * self.w * self.w, C)
        qkv = self.qkv(x).reshape(-1, T * self.w * self.w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                           4)
        attn = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        x = attn.transpose(1, 2).reshape(-1, T * self.w * self.w, C)
        x = self.proj(x).view(*curr_shape)
        x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous()
        return x.view(B, T, H * W, C)


class RefineBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowVideoAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x, T, H, W):
        x = x + self.attn(self.norm1(x), T, H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class FlashDividedAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, T, S):
        B, N, C = x.shape
        H, D = self.num_heads, C // self.num_heads
        xt = x.view(B, T, S, C).transpose(1, 2).reshape(B * S, T, C)
        qkv_t = self.qkv(xt).reshape(B * S, T, 3, H, D).permute(2, 0, 3, 1, 4)
        xt = F.scaled_dot_product_attention(qkv_t[0], qkv_t[1], qkv_t[2])
        xt = xt.transpose(1, 2).reshape(B * S, T, C)
        x = xt.view(B, S, T, C).transpose(1, 2).reshape(B, T * S, C)
        xs = x.view(B * T, S, C)
        qkv_s = self.qkv(xs).reshape(B * T, S, 3, H, D).permute(2, 0, 3, 1, 4)
        xs = F.scaled_dot_product_attention(qkv_s[0], qkv_s[1], qkv_s[2])
        x = xs.transpose(1, 2).reshape(B, T * S, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashDividedAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, T, S):
        x = x + self.attn(self.norm1(x), T, S);
        x = x + self.mlp(self.norm2(x))
        return x


class VideoTransUNet(nn.Module):
    def __init__(self, embed_dim=256, depth=6, num_heads=8, seq_len=10):
        super().__init__()
        self.mid_idx = seq_len // 2
        # Many-to-Many 输出索引 [3, 4, 5, 6, 7]
        self.out_indices = [self.mid_idx - 2, self.mid_idx - 1, self.mid_idx, self.mid_idx + 1, self.mid_idx + 2]

        self.enc1 = nn.Sequential(nn.Conv2d(4, 64, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(128, embed_dim, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len * 44 * 80, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.refine_1_4 = RefineBlock(dim=128, num_heads=4)
        self.refine_1_2 = RefineBlock(dim=64, num_heads=4)
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.final = nn.Sequential(nn.Conv2d(32, 3, 3, padding=1), nn.Tanh())

    def get_gaussian_kernel(self, kernel_size=15, sigma=4, device='cuda'):
        x = torch.arange(kernel_size).to(device) - kernel_size // 2
        gaussian = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = (gaussian / gaussian.sum()).view(1, 1, kernel_size, 1) * (gaussian / gaussian.sum()).view(1, 1, 1, kernel_size)
        return kernel, kernel_size // 2

    def forward(self, rgb_seq, mask_seq):
        B, T, _, H, W = rgb_seq.shape
        m_flat = mask_seq.view(B * T, 1, H, W)
        g_kernel, pad = self.get_gaussian_kernel(15, 4, m_flat.device)
        soft_m = F.conv2d(m_flat, g_kernel, padding=pad)

        inp = torch.cat([rgb_seq, mask_seq], dim=2).view(B * T, 4, H, W)
        f1 = self.enc1(inp)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        m_s8 = F.interpolate(soft_m, scale_factor=0.125, mode='bilinear')
        m_s4 = F.interpolate(soft_m, scale_factor=0.25, mode='bilinear')
        m_s2 = F.interpolate(soft_m, scale_factor=0.5, mode='bilinear')

        h3, w3 = H // 8, W // 8
        f3_input = f3 * (1 - m_s8)
        t_in = f3_input.view(B, T, -1, h3 * w3).permute(0, 1, 3, 2).reshape(B, T * h3 * w3, -1) + self.pos_embed
        for block in self.blocks: t_in = block(t_in, T, h3 * w3)
        t_out = t_in.view(B, T, h3 * w3, -1).transpose(2, 3).view(B * T, -1, h3, w3)

        d1 = self.dec1(t_out)
        h4, w4 = H // 4, W // 4
        d1_seq = self.refine_1_4(d1.view(B, T, 128, h4 * w4).permute(0, 1, 3, 2), T, h4, w4)

        d1_all = d1_seq.transpose(2, 3).reshape(B * T, 128, h4, w4)
        f2_all = f2 * (1 - m_s4)
        d2 = self.dec2(torch.cat([d1_all, f2_all], dim=1))

        h2, w2 = H // 2, W // 2
        d2_seq = self.refine_1_2(d2.view(B, T, 64, h2 * w2).permute(0, 1, 3, 2), T, h2, w2)

        preds = []
        for idx in self.out_indices:
            d2_curr = d2_seq[:, idx].transpose(1, 2).view(B, 64, h2, w2)
            m_s2_curr = m_s2.view(B, T, 1, h2, w2)[:, idx]
            f1_curr = f1.view(B, T, 64, h2, w2)[:, idx] * (1 - m_s2_curr)
            d3 = self.dec3(torch.cat([d2_curr, f1_curr], dim=1))
            preds.append(self.final(d3))

        return torch.stack(preds, dim=1)  # (B, 5, 3, H, W)


def process_video(input_path, mask_path, output_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 10
    img_size = (640, 352)  # (W, H)

    model = VideoTransUNet(seq_len=seq_len).to(device)
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    cap_in = cv2.VideoCapture(input_path)
    cap_mask = cv2.VideoCapture(mask_path)
    fps = cap_in.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'VP09')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, img_size)

    print(f"Total Frames: {total_frames} | Mode: Sliding Window (Center Frame)")
    all_rgb, all_mask = [], []
    for _ in tqdm(range(total_frames), desc="Loading"):
        ret_i, frame_i = cap_in.read()
        ret_m, frame_m = cap_mask.read()
        if not ret_i or not ret_m:
            break

        frame_i = cv2.cvtColor(cv2.resize(frame_i, img_size), cv2.COLOR_BGR2RGB)
        img_t = (torch.from_numpy(frame_i).permute(2, 0, 1).float() / 127.5) - 1.0

        frame_m = cv2.resize(frame_m, img_size, interpolation=cv2.INTER_NEAREST)
        if len(frame_m.shape) == 3: frame_m = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
        _, frame_m = cv2.threshold(frame_m, 127, 1, cv2.THRESH_BINARY)
        mask_t = torch.from_numpy(frame_m).unsqueeze(0).float()

        all_rgb.append(img_t)
        all_mask.append(mask_t)

    cap_in.release()
    cap_mask.release()

    print("Inference starting...")
    with torch.no_grad():
        for i in tqdm(range(total_frames), desc="Processing"):
            # 滑动窗口: [i-5, ... i, ... i+4] (共10帧)
            indices = [max(0, min(total_frames - 1, t)) for t in range(i - seq_len // 2, i + seq_len // 2)]
            window_rgb = torch.stack([all_rgb[idx] for idx in indices]).unsqueeze(0).to(device)
            window_mask = torch.stack([all_mask[idx] for idx in indices]).unsqueeze(0).to(device)

            mask_for_pool = window_mask.transpose(1, 2)
            window_mask_dilated = F.max_pool3d(mask_for_pool, (1, 5, 5), stride=1, padding=(0, 2, 2)).transpose(1, 2)

            # 模型输出 5 帧 (B, 5, 3, H, W)
            output_stack = model(window_rgb, window_mask_dilated)

            # 只取最中间的那一帧，输出 [mid-2, mid-1, mid, mid+1, mid+2]索引 2
            output = output_stack[:, 2]

            output = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
            output = output.clip(0, 255).astype(np.uint8)
            out_writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    out_writer.release()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    INPUT = r"../dataset/test_data/input.mp4"
    MASK = r"../dataset/test_data/mask.mp4"
    OUTPUT = r"SeqTransUNet_epoch_299.mp4"
    MODEL = "../model_save/SeqTransUNet/epoch_299.pth"

    process_video(INPUT, MASK, OUTPUT, MODEL)

```
## 分块处理高分辨率视频
```python

```
