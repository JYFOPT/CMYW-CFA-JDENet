# SIED_CMYW_GT_PatchDataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import glob
import re
import cv2

class RandomFlipBoth:
    """对输入和真值同时进行相同的随机水平/垂直翻转（用于训练增强）"""
    def __call__(self, img, target):
        if torch.rand(1) < 0.5:
            img = torch.flip(img, dims=[2])   # 水平翻转
            target = torch.flip(target, dims=[2])
        if torch.rand(1) < 0.5:
            img = torch.flip(img, dims=[1])   # 垂直翻转
            target = torch.flip(target, dims=[1])
        return img, target

class CMYW_RGB_PatchDataset(Dataset):
    """
    支持两种数据组织形式：

    1. 训练集（patch 模式）：
       - CMYW 文件：{scene_id}_{frame}_ratio{ratio}_patch_{y}_{x}.npy
       - GT 文件：{scene_id}_patch_{y}_{x}.jpg（scene_id 无前导零）
       - 配对依据：(scene_id_int, y, x)

    2. 验证集（整图模式）：
       - CMYW 文件：{scene_id}_{frame}_ratio{ratio}.npy
       - GT 文件：{scene_id}.jpg（scene_id 无前导零）
       - 配对依据：scene_id_int

    参数：
        cmyw_path: 目录路径（含 .npy 文件）
        gt_path:   目录路径（含 .jpg 或 .png 等图像文件）
        transform: 数据增强
        crop_size: 若指定则中心裁剪（仅对整图模式有效）
        mode: 'auto' 自动检测，或 'train' / 'val' 强制指定
    """
    def __init__(self, cmyw_path, gt_path, transform=None, crop_size=None, mode='auto'):
        self.transform = transform
        self.crop_size = crop_size

        # 判断是否为 mmap 大文件模式（保留，但一般不使用）
        if os.path.isfile(cmyw_path) and cmyw_path.endswith('.npy'):
            self.mode = 'mmap'
            self.cmyw = np.load(cmyw_path, mmap_mode='r')
            self.gt = np.load(gt_path, mmap_mode='r')
            assert self.cmyw.shape[0] == self.gt.shape[0], "样本数量不匹配"
            self.len = len(self.cmyw)
            self.scales = [1.0] * self.len
            self.pairs = None
            return

        # 目录模式
        self.mode = 'dir'
        # 自动检测模式：查看 cmyw 目录中第一个 .npy 文件名是否包含 '_patch_'
        if mode == 'auto':
            sample_files = glob.glob(os.path.join(cmyw_path, '*.npy'))
            if sample_files and '_patch_' in os.path.basename(sample_files[0]):
                pair_mode = 'train'
            else:
                pair_mode = 'val'
        else:
            pair_mode = mode

        # 构建字典
        if pair_mode == 'train':
            cmyw_dict = self._build_train_dict(cmyw_path)
            gt_dict = self._build_train_dict(gt_path)   # GT 同样使用训练集规则
        else:  # 'val'
            cmyw_dict = self._build_val_dict(cmyw_path)
            gt_dict = self._build_val_dict(gt_path)

        # 取键交集
        common_keys = set(cmyw_dict.keys()) & set(gt_dict.keys())
        if not common_keys:
            raise RuntimeError(
                f"CMYW 和 GT 目录没有匹配的键！\n"
                f"CMYW 键示例: {list(cmyw_dict.keys())[:3]}\n"
                f"GT 键示例: {list(gt_dict.keys())[:3]}"
            )
        self.pairs = [(cmyw_dict[k], gt_dict[k]) for k in sorted(common_keys)]
        self.len = len(self.pairs)

        # 从 CMYW 文件名中提取 ratio 作为缩放倍数（修正正则，避免匹配末尾点）
        self.scales = []
        for cmw_path, _ in self.pairs:
            match = re.search(r'_ratio(\d+(?:\.\d+)?)', os.path.basename(cmw_path))
            scale = float(match.group(1)) if match else 1.0
            self.scales.append(scale)

    def _build_train_dict(self, directory):
        """
        训练集：键为 (scene_id_int, y, x)
        提取规则：
            - scene_id: 文件名开头的数字串转为 int（去除前导零）
            - y, x: 从 'patch_<y>_<x>' 提取
        """
        exts = ['*.npy', '*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(directory, ext)))
        d = {}
        for f in files:
            basename = os.path.basename(f)
            # 提取场景 ID（第一个数字串）
            match = re.search(r'^(\d+)', basename)
            if not match:
                continue
            scene_id = int(match.group(1))  # 去除前导零
            # 提取坐标
            coord_match = re.search(r'patch_(\d+)_(\d+)', basename)
            if coord_match:
                y, x = int(coord_match.group(1)), int(coord_match.group(2))
                d[(scene_id, y, x)] = f
            else:
                # 训练集应该包含坐标，跳过无坐标的文件
                continue
        return d

    def _build_val_dict(self, directory):
        """
        验证集：键为 scene_id_int
        提取规则：文件名开头的数字串转为 int
        """
        exts = ['*.npy', '*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(directory, ext)))
        d = {}
        for f in files:
            basename = os.path.basename(f)
            match = re.search(r'^(\d+)', basename)
            if match:
                scene_id = int(match.group(1))
                d[scene_id] = f
        return d

    def __len__(self):
        return self.len

    def _load_image(self, path):
        """加载图像/数组，返回 (H, W, C) float32 范围 [0,1]"""
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            data = np.load(path).astype(np.float32)
            if data.dtype == np.uint8:
                data = data / 255.0
            return data
        else:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"无法读取图像: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            return img

    def __getitem__(self, idx):
        if self.mode == 'mmap':
            cmyw = self.cmyw[idx]
            gt = self.gt[idx]
            scale = self.scales[idx]
        else:
            cmyw_path, gt_path = self.pairs[idx]
            cmyw = self._load_image(cmyw_path)
            gt = self._load_image(gt_path)
            scale = self.scales[idx]

            # 可选的中心裁剪（主要用于验证集大图）
            if self.crop_size is not None:
                h, w = cmyw.shape[0], cmyw.shape[1]
                if isinstance(self.crop_size, int):
                    crop_h = crop_w = self.crop_size
                else:
                    crop_h, crop_w = self.crop_size
                if h < crop_h or w < crop_w:
                    raise ValueError(f"图像尺寸({h},{w})小于裁剪尺寸({crop_h},{crop_w})")
                start_h = (h - crop_h) // 2
                start_w = (w - crop_w) // 2
                cmyw = cmyw[start_h:start_h+crop_h, start_w:start_w+crop_w, :]
                gt = gt[start_h:start_h+crop_h, start_w:start_w+crop_w, :]

        # 应用放大倍数 r（对 CMYW 各通道）
        cmyw = cmyw * scale

        # 转为 tensor (C, H, W)
        cmyw_t = torch.from_numpy(cmyw).permute(2, 0, 1).float()
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).float()

        if self.transform is not None:
            cmyw_t, gt_t = self.transform(cmyw_t, gt_t)

        return cmyw_t, gt_t