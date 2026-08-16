import torch
from torch import nn
import torch.nn.functional as F

class DeformConv2d(nn.Module):
    """
    可变形卷积模块（DCNv2改进版）：通过学习动态偏移与调制标量，适配不规则目标/畸变图像
    核心创新：支持调制标量（modulation），增强偏移区域的特征权重，提升畸变场景鲁棒性
    输入：特征图 [B, C, H, W]（B=批次，C=通道，H/W=高/宽）
    输出：可变形卷积增强特征 [B, outc, H/stride, W/stride]（步长stride控制输出尺寸）
    """

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        参数说明：
            inc: 输入通道数
            outc: 输出通道数
            kernel_size: 卷积核大小（默认3×3）
            padding: 零填充大小（默认1，确保卷积后尺寸不变）
            stride: 卷积步长（默认1，步长>1时输出尺寸缩小）
            bias: 是否使用偏置（默认None）
            modulation: 是否启用调制标量（DCNv2特性，默认False）
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)  # 零填充层（适配padding参数）
        # 主卷积层：对偏移后的特征进行卷积，提取变形鲁棒特征
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # 偏移预测卷积：预测2N个偏移量（N=kernel_size²，x/y方向各N个）
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)  # 偏移初始化为0（默认固定卷积，训练中学习偏移）
        self.p_conv.register_backward_hook(self._set_lr)  # 反向传播钩子：降低偏移学习率（稳定训练）

        self.modulation = modulation  # 调制标量开关
        if modulation:
            # 调制标量预测卷积：预测N个调制系数（控制偏移区域特征权重）
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)  # 调制系数初始化为0（默认无调制）
            self.m_conv.register_backward_hook(self._set_lr)  # 调制系数学习率降低

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        """反向传播钩子：降低偏移/调制参数的学习率（0.1倍），避免训练震荡"""
        grad_input = (grad_input[i] * 0.1for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1for i in range(len(grad_output)))

    def forward(self, x):
        """
        前向传播流程：偏移预测→采样位置计算→双线性插值采样→调制（可选）→卷积输出
        """
        # 步骤1：预测偏移量（x/y方向各N个，N=kernel_size²）
        offset = self.p_conv(x)  # [B, 2N, H', W']（H'=H/stride，W'=W/stride）
        # 步骤2：预测调制标量（可选）
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))  # [B, N, H', W']（调制系数∈[0,1]）

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2# N=kernel_size²（x/y方向各N个偏移）

        # 步骤3：零填充（确保采样不越界）
        if self.padding:
            x = self.zero_padding(x)

        # 步骤4：计算采样位置（p = 基准位置 + 固定偏移 + 学习偏移）
        p = self._get_p(offset, dtype)  # [B, 2N, H', W']
        # 维度重排：[B, 2N, H', W']→[B, H', W', 2N]（适配后续采样）
        p = p.contiguous().permute(0, 2, 3, 1)

        # 步骤5：双线性插值的四个邻域位置计算（q_lt=左上，q_rb=右下，q_lb=左下，q_rt=右上）
        q_lt = p.detach().floor()  # 向下取整得到整数坐标
        q_rb = q_lt + 1# 右上坐标
        # 坐标裁剪（确保不超出特征图边界）
        q_lt = torch.cat([
            torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),  # x方向坐标裁剪
            torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)  # y方向坐标裁剪
        ], dim=-1).long()
        q_rb = torch.cat([
            torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
            torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)
        ], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)  # 左下坐标
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)  # 右上坐标

        # 步骤6：采样位置裁剪（避免插值越界）
        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:], 0, x.size(3) - 1)
        ], dim=-1)

        # 步骤7：双线性插值权重计算（基于采样位置与邻域坐标的距离）
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # 步骤8：基于邻域坐标采样特征
        x_q_lt = self._get_x_q(x, q_lt, N)  # [B, C, H', W', N]（左上邻域特征）
        x_q_rb = self._get_x_q(x, q_rb, N)  # 右下邻域特征
        x_q_lb = self._get_x_q(x, q_lb, N)  # 左下邻域特征
        x_q_rt = self._get_x_q(x, q_rt, N)  # 右上邻域特征

        # 步骤9：双线性插值融合邻域特征
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt  # [B, C, H', W', N]

        # 步骤10：调制标量应用（可选）
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)  # [B, H', W', N]
            m = m.unsqueeze(dim=1)  # [B, 1, H', W', N]（扩展通道维度）
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)  # [B, C, H', W', N]
            x_offset *= m  # 调制特征权重（突出有效偏移区域）

        # 步骤11：维度重排（适配主卷积输入）
        x_offset = self._reshape_x_offset(x_offset, ks)  # [B, C, H'×ks, W'×ks]
        # 步骤12：主卷积提取变形鲁棒特征
        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):
        """生成卷积核的固定偏移（如3×3核的9个位置偏移）"""
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1)
        )
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)  # [2N]
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)  # [1, 2N, 1, 1]（广播适配批次）
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        """生成采样基准位置（基于步长的网格坐标）"""
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride)
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)  # [1, N, h, w]
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)  # [1, N, h, w]
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)  # [1, 2N, h, w]
        return p_0

    def _get_p(self, offset, dtype):
        """计算最终采样位置：基准位置 + 固定偏移 + 学习偏移"""
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)  # 固定偏移
        p_0 = self._get_p_0(h, w, N, dtype)  # 基准位置
        p = p_0 + p_n + offset  # 最终采样位置
        return p

    def _get_x_q(self, x, q, N):
        """基于邻域坐标q采样特征x"""
        b, h, w, _ = q.size()
        padded_w = x.size(3)  # 填充后的宽度
        c = x.size(1)  # 通道数
        # 维度重排：[B, C, H, W]→[B, C, H×W]（便于索引）
        x = x.contiguous().view(b, c, -1)
        # 计算索引：offset_x * W + offset_y（将2D坐标转为1D索引）
        index = q[..., :N] * padded_w + q[..., N:]
        # 扩展索引维度：[B, H, W, N]→[B, C, H×W×N]（适配通道维度）
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        # 采样特征并恢复维度：[B, C, H×W×N]→[B, C, H, W, N]
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        """将采样特征重排为卷积输入格式：[B, C, H', W', N]→[B, C, H'×ks, W'×ks]"""
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([
            x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks)
            for s in range(0, N, ks)
        ], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)
        return x_offset


class SpatialAttention(nn.Module):
    """
    空间注意力模块：基于通道均值与最大值的空间权重学习
    功能：突出关键空间区域（如目标位置），抑制背景噪声
    输入：特征图 [B, C, H, W]
    输出：空间加权特征 [B, C, H, W]
    """

    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 7×7卷积：大感受野捕捉全局空间关联
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()  # 权重归一化（∈[0,1]）

    def forward(self, x):
        # 计算通道均值与最大值（压缩通道维度）
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]（全局均值）
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]（全局最大值）
        # 特征拼接：均值+最大值→2通道特征
        out = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        # 学习空间权重并加权特征
        out = self.conv(out)  # [B, 1, H, W]（空间权重图）
        out = self.sigmoid(out)
        out = x * out  # 空间加权（突出高权重区域）
        return out


class ChannelAttention(nn.Module):
    """
    通道注意力模块：基于全局平均池化的通道权重学习
    功能：筛选关键通道特征（如目标纹理通道），抑制冗余通道
    输入：特征图 [B, C, H, W]
    输出：通道加权特征 [B, C, H, W]
    """

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化→[B, C, 1, 1]
        # 通道压缩-恢复瓶颈结构（减少参数）
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)  # 非线性激活
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()  # 通道权重归一化

    def forward(self, x):
        # 全局平均池化+通道压缩+激活+通道恢复+权重归一化
        out = self.avg_pool(x)  # [B, C, 1, 1]
        out = self.fc1(out)  # [B, C/reduction, 1, 1]
        out = self.relu(out)  # 非线性增强
        out = self.fc2(out)  # [B, C, 1, 1]（通道权重）
        out = self.sigmoid(out)
        out = x * out  # 通道加权（突出高权重通道）
        return out


class DeformableAttention(nn.Module):
    """
    可变形注意力模块（DIA核心子模块1）：结合下采样-上采样与失真调制
    核心创新：通过下采样压缩空间维度，学习失真感知权重，适配畸变图像（如鱼眼图像）
    输入：特征图 [B, C, H, W]
    输出：可变形注意力增强特征 [B, C, H, W]（与输入维度一致）
    """

    def __init__(self, stride=1, distortionmode=False):
        super(DeformableAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)  # 2→1通道卷积
        self.sigmoid = nn.Sigmoid()  # 权重归一化
        self.distortionmode = distortionmode  # 失真调制开关
        self.upsample = nn.Upsample(scale_factor=2)  # 上采样（恢复尺寸）
        # 下采样卷积（均值/最大值特征各1个，步长2→尺寸减半）
        self.downavg = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.downmax = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        # 失真调制卷积（可选，针对畸变区域学习权重）
        if distortionmode:
            self.d_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv.weight, 0)  # 初始无调制
            self.d_conv.register_full_backward_hook(self._set_lra)  # 调制A学习率0.4
            self.d_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv1.weight, 0)
            self.d_conv1.register_full_backward_hook(self._set_lrm)  # 调制M学习率0.1

    @staticmethod
    def _set_lra(module, grad_input, grad_output):
        """失真调制A的反向传播钩子：学习率×0.4"""
        grad_input = [g * 0.4if g is not None else None for g in grad_input]
        grad_output = [g * 0.4if g is not None else None for g in grad_output]
        return tuple(grad_input)

    @staticmethod
    def _set_lrm(module, grad_input, grad_output):
        """失真调制M的反向传播钩子：学习率×0.1"""
        grad_input = [g * 0.1 if g is not None else None for g in grad_input]
        grad_output = [g * 0.1 if g is not None else None for g in grad_output]
        return tuple(grad_input)

    def forward(self, x):
        # 步骤1：计算通道均值与最大值（压缩通道维度）
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # 步骤2：下采样（尺寸减半，减少计算量）
        avg_out = self.downavg(avg_out)  # [B, 1, H/2, W/2]
        max_out = self.downmax(max_out)  # [B, 1, H/2, W/2]

        # 步骤3：失真调制（可选，针对畸变区域增强权重）
        if self.distortionmode:
            # 学习均值/最大值特征的调制系数
            d_avg_out = torch.sigmoid(self.d_conv(avg_out))  # [B, 1, H/2, W/2]
            d_max_out = torch.sigmoid(self.d_conv1(max_out))  # [B, 1, H/2, W/2]
            # 调制后拼接：均值调制×最大值特征 + 最大值调制×均值特征
            out = torch.cat([d_avg_out * max_out, d_max_out * avg_out], dim=1)  # [B, 2, H/2, W/2]
        else:
            # 无调制：直接拼接均值与最大值特征
            out = torch.cat([max_out, avg_out], dim=1)  # [B, 2, H/2, W/2]

        # 步骤4：学习注意力权重+上采样恢复尺寸
        out = self.conv(out)  # [B, 1, H/2, W/2]（注意力权重）
        mask = self.sigmoid(self.upsample(out))  # [B, 1, H, W]（上采样至原尺寸）

        # 步骤5：注意力加权+ReLU激活
        att_out = x * mask
        return F.relu(att_out)


class DeformableAttention2(nn.Module):
    """
    可变形注意力模块（DIA核心子模块2）：与DeformableAttention结构一致，仅调制学习率不同
    功能：适配不同畸变程度场景（如轻微畸变用DIA1，严重畸变用DIA2）
    输入/输出：与DeformableAttention一致
    """

    def __init__(self, stride=1, distortionmode=False):
        super(DeformableAttention2, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.distortionmode = distortionmode
        self.upsample = nn.Upsample(scale_factor=2)
        self.downavg = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.downmax = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        if distortionmode:
            self.d_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv.weight, 0)
            self.d_conv.register_full_backward_hook(self._set_lrb)  # 调制B学习率0.1
            self.d_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv1.weight, 0)
            self.d_conv1.register_full_backward_hook(self._set_lrn)  # 调制N学习率0.4

    @staticmethod
    def _set_lrb(module, grad_input, grad_output):
        """失真调制B的反向传播钩子：学习率×0.1"""
        grad_input = [g * 0.1 if g is not None else None for g in grad_input]
        grad_output = [g * 0.1 if g is not None else None for g in grad_output]
        return tuple(grad_input)

    @staticmethod
    def _set_lrn(module, grad_input, grad_output):
        """失真调制N的反向传播钩子：学习率×0.4"""
        grad_input = [g * 0.4if g is not None else None for g in grad_input]
        grad_output = [g * 0.4if g is not None else None for g in grad_output]
        return tuple(grad_input)

    def forward(self, x):
        # 流程与DeformableAttention完全一致，仅调制学习率不同
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = self.downavg(avg_out)
        max_out = self.downmax(max_out)

        if self.distortionmode:
            d_avg_out = torch.sigmoid(self.d_conv(avg_out))
            d_max_out = torch.sigmoid(self.d_conv1(max_out))
            out = torch.cat([d_avg_out * max_out, d_max_out * avg_out], dim=1)
        else:
            out = torch.cat([max_out, avg_out], dim=1)

        out = self.conv(out)
        mask = self.sigmoid(self.upsample(out))
        att_out = x * mask
        return F.relu(att_out)


if __name__ == "__main__":
    device = torch.device('cuda:0'if torch.cuda.is_available() else'cpu')
    x = torch.randn(1, 64, 32, 32).to(device)

    model = DeformableAttention2(distortionmode=True)
    model.to(device)

    y = model(x)

    print("微信公众号：十小大的底层视觉工坊")
    print("知乎、CSDN：十小大")
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)
