# dataset_cmyw_rgb.py
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class RandomFlipBoth:
    """对输入和真值同时进行相同的随机水平/垂直翻转（用于训练增强）"""
    def __call__(self, img, target):
        # img, target: (C, H, W) torch.Tensor
        if torch.rand(1) < 0.5:
            img = torch.flip(img, dims=[2])   # 水平翻转 (H 方向)
            target = torch.flip(target, dims=[2])
        if torch.rand(1) < 0.5:
            img = torch.flip(img, dims=[1])   # 垂直翻转 (W 方向)
            target = torch.flip(target, dims=[1])
        return img, target

class CMYW_RGB_PatchDataset(Dataset):
    """
    从预裁剪的 .npy 文件加载 CMYW 输入和 RGB 真值（输出 4 通道，含 W）。
    输入: cmyw_path (train_patches_CMYW.npy), gt_path (train_patches_GT.npy)
    输出: (cmyw, target) 其中 target 为 (RGB + W) 四通道，值域 [0,1]
    """
    def __init__(self, cmyw_path, gt_path, transform=None):
        self.cmyw = np.load(cmyw_path)   # shape (N, H, W, 4)
        self.gt = np.load(gt_path)       # shape (N, H, W, 3)
        assert self.cmyw.shape[0] == self.gt.shape[0], "样本数量不匹配"
        self.transform = transform

    def __len__(self):
        return len(self.cmyw)

    def __getitem__(self, idx):
        cmyw = self.cmyw[idx]   # (H, W, 4) uint8
        gt = self.gt[idx]       # (H, W, 3) uint8

        # 转为 float 并归一化到 [0,1]
        cmyw_t = torch.from_numpy(cmyw).float() / 255.0
        gt_t = torch.from_numpy(gt).float() / 255.0

        # 调整为 (C, H, W) 格式
        cmyw_t = cmyw_t.permute(2, 0, 1)   # (4, H, W)
        gt_t = gt_t.permute(2, 0, 1)       # (3, H, W)

        # 构造四通道真值（RGB + W），W = (R+G+B)/3，保持与原始脚本一致
        W_t = gt_t.mean(dim=0, keepdim=True)   # (1, H, W)
        target = torch.cat([gt_t, W_t], dim=0) # (4, H, W)
        target = gt_t

        if self.transform is not None:
            cmyw_t, target = self.transform(cmyw_t, target)

        return cmyw_t, target