import torch.utils.data
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

class LLSKM_1D(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(LLSKM_1D, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.conv_1xn = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), stride=(stride, stride),
                                  padding=(0, padding))
        self.conv_nx1 = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), stride=(stride, stride),
                                  padding=(padding, 0))
        self.attn = Avg_ChannelAttention(channels)
        self.kernel_size = kernel_size

    def forward(self, x):
        m_batchsize, C, height, width = self.conv_1xn.weight.size()
        theta = self.attn(x)

        out_1xn_normal = self.conv_1xn(x)
        kernel_w1 = self.conv_1xn.weight

        out_nx1_normal = self.conv_nx1(out_1xn_normal)
        kernel_w2 = self.conv_nx1.weight

        nxn_kernel = (torch.bmm(kernel_w1.contiguous().view(-1, width, height),
                                kernel_w2.contiguous().view(-1, height, width))).view(m_batchsize, C, width, width)

        nxn_kernel_w1 = nxn_kernel.sum(2).sum(2)
        nxn_kernel_w2 = nxn_kernel_w1[:, :, None, None]
        out_center = F.conv2d(input=x, weight=nxn_kernel_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)

        nxn_center_w1 = nxn_kernel[:, :, self.kernel_size // 2, self.kernel_size // 2]
        nxn_center_w2 = nxn_center_w1[:, :, None, None]
        out_offset = F.conv2d(input=x, weight=nxn_center_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)

        out = out_center - out_nx1_normal + theta * out_offset

        return out