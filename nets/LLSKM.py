
import torch
import torch.nn as nn
import torch.nn.functional as F


class Avg_ChannelAttention(nn.Module):
    def __init__(self, channels, r=4):
        super(Avg_ChannelAttention, self).__init__()
        self.avg_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # bz,C_out,h,w -> bz,C_out,1,1
            nn.Conv2d(channels, channels // r, 1, 1, 0, bias=False),  # bz,C_out,1,1 -> bz,C_out/r,1,1
            nn.BatchNorm2d(channels // r),
            nn.ReLU(True),
            nn.Conv2d(channels // r, channels, 1, 1, 0, bias=False),  # bz,C_out/r,1,1 -> bz,C_out,1,1
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.avg_channel(x)


class LLSKM(nn.Module):
    """
    可学习局部显著性核模块（基础版）：重构3×3卷积核贡献，强化局部显著性
    核心设计：
        - 传统卷积特征：标准3×3卷积提取基础特征
        - 核权重重构：拆分卷积核为“邻域求和权重”与“中心权重”，分别生成特征
        - 通道注意力调制：自适应强化中心权重特征的显著性通道
    Args:
        channels: 输入/输出通道数
        kernel_size: 卷积核大小（默认3）
        stride: 步长（默认1）
        padding: 填充量（默认1）
        dilation: 膨胀率（默认1）
        groups: 分组数（默认1）
        bias: 是否带偏置（默认False）
    """
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(LLSKM, self).__init__()
        # General CNN
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        # Channel Attention for $\theta$
        self.attn = Avg_ChannelAttention(channels)
        self.kernel_size = kernel_size

    def forward(self, x):
        # Feature result from a $k\times k$ General CNN
        out_normal = self.conv(x)
        # Channel Attention for $\theta_n$
        theta = self.attn(x)

        # Sum up for each $k\times k$ CNN filter
        kernel_w1 = self.conv.weight.sum(2).sum(2)
        # Extend the $1\times 1$ to $k\times k$
        kernel_w2 = kernel_w1[:, :, None, None]
        # Filter the feature with $\textbf{W}_{sum}$
        out_center = F.conv2d(input=x, weight=kernel_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)
        # Filter the feature with $\textbf{W}_{c}$
        center_w1 = self.conv.weight[:, :, self.kernel_size // 2, self.kernel_size // 2]
        center_w2 = center_w1[:, :, None, None]
        out_offset = F.conv2d(input=x, weight=center_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)

        # The output feature of our Diff LSFM block
        # $\textbf{Y} = {{\mathcal{W}}_s (\textbf{X})} = \mathcal{W}_{sum}(\textbf{X}) - {\mathcal{W}}(\textbf{X}) + \theta_c (\textbf{X})\circ {\mathcal{W}_{c}}{(\textbf{X})}$
        return out_center - out_normal + theta * out_offset


class LLSKM_d(nn.Module):
    """
    可学习局部显著性核模块（膨胀版）：增大感受野，捕捉长距离局部依赖
    核心差异：卷积层采用dilation=2、padding=2，其他设计与基础版一致
    Args:
        同LLSKM模块，默认dilation=2、padding=2
    """
    def __init__(self, channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=1, bias=False):
        super(LLSKM_d, self).__init__()
        # General CNN
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        # Channel Attention for $\theta$
        self.attn = Avg_ChannelAttention(channels)
        self.kernel_size = kernel_size

    def forward(self, x):
        # Feature result from a $k\times k$ General CNN
        out_normal = self.conv(x)
        # Channel Attention for $\theta_n$
        theta = self.attn(x)

        # Sum up for each $k\times k$ CNN filter
        kernel_w1 = self.conv.weight.sum(2).sum(2)
        # Extend the $1\times 1$ to $k\times k$
        kernel_w2 = kernel_w1[:, :, None, None]
        # Filter the feature with $\textbf{W}_{sum}$
        out_center = F.conv2d(input=x, weight=kernel_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)
        # Filter the feature with $\textbf{W}_{c}$
        center_w1 = self.conv.weight[:, :, self.kernel_size // 2, self.kernel_size // 2]
        center_w2 = center_w1[:, :, None, None]
        out_offset = F.conv2d(input=x, weight=center_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)

        # The output feature of our Diff LSFM block
        # $\textbf{Y} = {{\mathcal{W}}_s (\textbf{X})} = \mathcal{W}_{sum}(\textbf{X}) - {\mathcal{W}}(\textbf{X}) + \theta_c (\textbf{X})\circ {\mathcal{W}_{c}}{(\textbf{X})}$
        return out_center - out_normal + theta * out_offset


if __name__ == "__main__":
    device = torch.device('cuda:0'if torch.cuda.is_available() else'cpu')

    # batch_size must > 1
    x = torch.randn(2, 64, 32, 32).to(device)
    model = LLSKM_d(64).to(device)

    y = model(x)

    print("微信公众号：十小大的底层视觉工坊")
    print("VX: shixiaodayyds, 备注【即插即用】添加交流群")
    print("知乎、CSDN、小红书：十小大")

    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)


