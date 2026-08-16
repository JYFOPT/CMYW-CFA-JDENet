'''
Taken from https://github.com/cszn/KAIR/blob/master/models/basicblock.py
'''

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

from torchvision.models import ConvNeXt

from nets.rstcablock import RSTCABlock, RSTCAGroup
'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


'''
# --------------------------------------------
# Useful blocks
# https://github.com/xinntao/BasicSR
# --------------------------------
# conv + normaliation + relu (conv)
# (PixelUnShuffle)
# (ConditionalBatchNorm2d)
# concat (ConcatBlock)
# sum (ShortcutBlock)
# resblock (ResBlock)
# Channel Attention (CA) Layer (CALayer)
# Residual Channel Attention Block (RCABlock)
# Residual Channel Attention Group (RCAGroup)
# Residual Dense Block (ResidualDenseBlock_5C)
# Residual in Residual Dense Block (RRDB)
# --------------------------------------------
'''


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR',
         negative_slope=0.2,n_prelu=1): #原版negative_slope=0.2 0.05
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias))
        elif t == 'D':
            L.append(nn.Dropout(p=0.2))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        # ------------------- 新增 LayerNorm 分支 -------------------
        elif t == 'N':  # LayerNorm (channels_first，适配4D张量 [B,C,H,W])
            L.append(LayerNorm3(normalized_shape=out_channels, eps=1e-6, data_format="channels_first"))
        elif t == 'n':  # LayerNorm (channels_last，适配4D张量 [B,H,W,C])
            L.append(LayerNorm3(normalized_shape=out_channels, eps=1e-6, data_format="channels_last"))
        # ----------------------------------------------------------
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t=='P':
            L.append(nn.PReLU(num_parameters=n_prelu, init=negative_slope))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'S':
            L.append(nn.Sigmoid())
        elif t == 'G':
            L.append(nn.GELU())
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


# --------------------------------------------
# inverse of pixel_shuffle
# --------------------------------------------
def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.
    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet
    Date:
        01/Jan/2019
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.
    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet
    Date:
        01/Jan/2019
    """

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


# --------------------------------------------
# conditional batch norm
# https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
# --------------------------------------------
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


# --------------------------------------------
# Concat the output of a submodule to its input
# --------------------------------------------
class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        return self.sub.__repr__() + 'concat'


# --------------------------------------------
# sum the output of a submodule to its input
# --------------------------------------------
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res



# contrast-aware channel attention module
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

# --------------------------------------------
# simplified information multi-distillation block (IMDB)
# x + conv1(concat(split(relu(conv(x)))x3))
# --------------------------------------------
class IMDBlock(nn.Module):
    """
    @inproceedings{hui2019lightweight,
      title={Lightweight Image Super-Resolution with Information Multi-distillation Network},
      author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},
      booktitle={Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)},
      pages={2024--2032},
      year={2019}
    }
    @inproceedings{zhang2019aim,
      title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
      author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
      booktitle={IEEE International Conference on Computer Vision Workshops},
      year={2019}
    }
    """
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CL', d_rate=0.25,
                 negative_slope=0.05):
        super(IMDBlock, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        assert mode[0] == 'C', 'convolutional layer first'

        self.conv1 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(self.r_nc, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0],
                            negative_slope=negative_slope)
        # self.conv5 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)#新增
        # self.conv6 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)#新增
        # self.cca=CCALayer(self.d_nc*4)#新增
        # self.act = activation(act_type='prelu', inplace=False,neg_slope=0.05)#新增

    def forward(self, x):
        d1, r1 = torch.split(self.conv1(x), (self.d_nc, self.r_nc), dim=1)    
        d2, r2 = torch.split(self.conv2(r1), (self.d_nc, self.r_nc), dim=1)  
        d3, r3 = torch.split(self.conv3(r2), (self.d_nc, self.r_nc), dim=1)
        
        # d1, r1 = torch.split(self.act(self.conv1(x)), (self.d_nc, self.r_nc), dim=1)
        # d2, r2 = torch.split(self.act(self.conv2(r1)), (self.d_nc, self.r_nc), dim=1)
        # d3, r3 = torch.split(self.act(self.conv3(r2)), (self.d_nc, self.r_nc), dim=1)
        # d5, r5 = torch.split(self.conv5(r3), (self.d_nc, self.r_nc), dim=1) #新加
        # d6, r6 = torch.split(self.conv6(r5), (self.d_nc, self.r_nc), dim=1) #新加
        # d4 =self.conv4(r6) #新加
        # res = self.conv1x1(torch.cat((d1, d2, d3,d5,d6, d4), dim=1)) #新加
        # res=self.conv1x1(self.cca(torch.cat((d1, d2, d3,d5,d6, d4), dim=1))) #新增
        d4 = self.conv4(r3) #原版
        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1)) #原版
        # res = self.conv1x1(self.cca(torch.cat((d1, d2, d3, d4), dim=1))) #新增
        return x + res


class Res3Block(nn.Module):
    """
    @inproceedings{hui2019lightweight,
      title={Lightweight Image Super-Resolution with Information Multi-distillation Network},
      author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},
      booktitle={Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)},
      pages={2024--2032},
      year={2019}
    }
    @inproceedings{zhang2019aim,
      title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
      author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
      booktitle={IEEE International Conference on Computer Vision Workshops},
      year={2019}
    }
    """

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBG',
                 d_rate=0.25,
                 negative_slope=0.05):
        super(Res3Block, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc1 = int(in_channels - self.d_nc)
        self.r_nc2 = int(self.r_nc1 - self.d_nc)
        self.r_nc3 = int(self.r_nc2 - self.d_nc)

        assert mode[0] == 'C', 'convolutional layer first'

        self.conv0 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv1 = conv(self.d_nc, self.d_nc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(self.d_nc, self.d_nc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(self.d_nc, self.d_nc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(self.d_nc, self.d_nc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv1x1 = conv(self.d_nc * 4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0],
                            negative_slope=negative_slope)
        # self.conv5 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)#新增
        # self.conv6 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)#新增
        # self.cca=CCALayer(self.d_nc*4)#新增
        # self.act = activation(act_type='prelu', inplace=False,neg_slope=0.05)#新增

    def forward(self, x):
        x0 = self.conv0(x)
        d1, r1 = torch.split(x0, (self.d_nc, self.r_nc1), dim=1)
        d2, r2 = torch.split(r1, (self.d_nc, self.r_nc2), dim=1)
        d3, r3 = torch.split(r2, (self.d_nc, self.r_nc3), dim=1)
        d4 = r3

        d1_1 = self.conv1(d1)

        d2_1 = self.conv2(d2)
        d2_2 = self.conv2(d2_1+d1_1)

        d3_1 = self.conv3(d3)
        d3_2 = self.conv3(d3_1+d2_1)
        d3_3 = self.conv3(d3_2+d2_2)

        d4_1 = self.conv4(d4)
        d4_2 = self.conv4(d4_1+d3_1)
        d4_3 = self.conv4(d4_2+d3_2)
        d4_4 = self.conv4(d4_3+d3_3)


        res = self.conv1x1(torch.cat((d1_1, d2_2, d3_3, d4_4), dim=1))  # 原版
        # res = self.conv1x1(self.cca(torch.cat((d1, d2, d3, d4), dim=1))) #新增
        return x + res


# ==================== 空间注意力模块 ====================
class SpatialAttention(nn.Module):
    """
    空间注意力模块：关注特征图的空间位置（哪里重要）
    参考 CBAM 的空间注意力部分
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 输入: 2通道 (maxpool, avgpool), 输出: 1通道的空间注意力图
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算通道维度的最大值和均值
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # 拼接后卷积生成空间注意力图
        y = torch.cat([max_pool, avg_pool], dim=1)  # [B, 2, H, W]
        y = self.conv(y)  # [B, 1, H, W]
        y = self.sigmoid(y)

        return x * y  # 广播乘法，关注重要空间位置


# ==================== 通道注意力模块 ====================
class ChannelAttention(nn.Module):
    """
    通道注意力模块：关注什么特征通道重要（SE-Net 风格）
    """

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的MLP
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化分支
        avg_out = self.fc(self.avg_pool(x))
        # 全局最大池化分支
        max_out = self.fc(self.max_pool(x))

        # 融合两个分支
        out = avg_out + max_out
        out = self.sigmoid(out)

        return x * out  # 通道加权
class SCERes2Block(nn.Module):
    """
    @inproceedings{hui2019lightweight,
      title={Lightweight Image Super-Resolution with Information Multi-distillation Network},
      author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},
      booktitle={Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)},
      pages={2024--2032},
      year={2019}
    }
    @inproceedings{zhang2019aim,
      title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
      author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
      booktitle={IEEE International Conference on Computer Vision Workshops},
      year={2019}
    }
    """

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBG',
                 d_rate=0.25,
                 negative_slope=0.05):
        super(SCERes2Block, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc1 = int(in_channels - self.d_nc)
        self.r_nc2 = int(self.r_nc1 - self.d_nc)
        self.r_nc3 = int(self.r_nc2 - self.d_nc)

        assert mode[0] == 'C', 'convolutional layer first'

        self.conv1x1_0 = conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode,
                            negative_slope=negative_slope)

        self.conv1 = nn.Conv2d(self.d_nc, self.d_nc, kernel_size=3, stride=1, padding=1,groups=self.d_nc)
        self.conv2 = nn.Conv2d(self.d_nc, self.d_nc, kernel_size=3, stride=1, padding=1,groups=self.d_nc)
        self.conv3 = nn.Conv2d(self.d_nc, self.d_nc, kernel_size=3, stride=1, padding=1,groups=self.d_nc)
        self.conv4 = nn.Conv2d(self.d_nc, self.d_nc, kernel_size=3, stride=1, padding=1,groups=self.d_nc)

        self.conv1x1_1 = conv(self.d_nc * 4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode,
                            negative_slope=negative_slope)

        # 空间注意力：关注"哪里"重要（HW维度）
        self.spatial_attention = SpatialAttention(kernel_size=3)

        # 通道注意力：关注"什么"特征重要（C维度）
        self.channel_attention = ChannelAttention(out_channels, reduction=16)


    def forward(self, x):

        x0 = self.conv1x1_0(x)

        d1_0, r1 = torch.split(x0, (self.d_nc, self.r_nc1), dim=1)
        d2_0, r2 = torch.split(r1, (self.d_nc, self.r_nc2), dim=1)
        d3_0, r3 = torch.split(r2, (self.d_nc, self.r_nc3), dim=1)
        d4_0 = r3

        d1_0 = d1_0
        d1 = self.spatial_attention(d1_0)

        d2_0 = self.conv2(d2_0)
        d2 = self.spatial_attention(d2_0)

        d3_0 = self.conv3(d3_0+d2_0)
        d3 = self.spatial_attention(d3_0)

        d4_0 = self.conv4(d4_0+d3_0)
        d4 = self.spatial_attention(d4_0)

        res = self.conv1x1_1(torch.cat((d1, d2, d3, d4), dim=1))

        res = self.channel_attention(res)


        return x + res

class MCC(nn.Module):
    def __init__(self, f_number, num_heads, padding_mode, bias=False) -> None:
        super().__init__()
        self.norm = LayerNorm3(f_number, eps=1e-6, data_format='channels_first')

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.pwconv = nn.Conv2d(f_number, f_number * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(f_number * 3, f_number * 3, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=f_number * 3)
        self.project_out = nn.Conv2d(f_number, f_number, kernel_size=1, bias=bias)
        self.feedforward = nn.Sequential(
            nn.Conv2d(f_number, f_number, 1, 1, 0, bias=bias),
            nn.GELU(),
            nn.Conv2d(f_number, f_number, 3, 1, 1, bias=bias, groups=f_number, padding_mode=padding_mode),
            nn.GELU()
        )

    def forward(self, x):
        attn = self.norm(x)
        _, _, h, w = attn.shape

        qkv = self.dwconv(self.pwconv(attn))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.feedforward(out + x)
        return out


class SimdFormerBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, num_heads=8, kernel_size=3, stride=1, padding=1, bias=True,
                 mode='CL',
                 d_rate=0.25, negative_slope=0.05):
        super(SimdFormerBlock, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        assert mode[0] == 'C', 'convolutional layer first'

        self.conv1 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(self.r_nc, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv1x1 = conv(self.d_nc * 4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0],
                            negative_slope=negative_slope)


        self.mcc1 = MCC(f_number=self.d_nc, num_heads=num_heads, padding_mode='reflect')
        self.mcc2 = MCC(f_number=self.d_nc, num_heads=num_heads, padding_mode='reflect')
        self.mcc3 = MCC(f_number=self.d_nc, num_heads=num_heads, padding_mode='reflect')
        self.mcc4 = MCC(f_number=self.d_nc, num_heads=num_heads, padding_mode='reflect')

    def forward(self, x):
        d1, r1 = torch.split(self.conv1(x), (self.d_nc, self.r_nc), dim=1)
        d1 = self.mcc1(d1)  # num_heads=8

        d2, r2 = torch.split(self.conv2(r1), (self.d_nc, self.r_nc), dim=1)
        d2 = self.mcc2(d2)  # num_heads=16

        d3, r3 = torch.split(self.conv3(r2), (self.d_nc, self.r_nc), dim=1)
        d3 = self.mcc3(d3)  # num_heads=32

        d4 = self.conv4(r3)
        d4 = self.mcc4(d4)  # num_heads=64

        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1))
        return x + res

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class LayerNorm3(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, dropout=0.):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 处理 4D 图像输入 (B, C, H, W)
        if x.dim() == 4:
            B, C, H, W = x.shape
            # 转换为 3D: (B, H*W, C)
            x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
            x = self.net(x)
            # 转回 4D: (B, C, H, W)
            x = x.permute(0, 2, 1).view(B, C, H, W)
            return x
        else:
            # 已经是 3D 或 2D，直接处理
            return self.net(x)



##########################################################################
## Convolution and Attention Fusion Module  (CAFM)
class SAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)


        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-2, -1) @ k) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (v @ attn)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out

        return output

class CAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out

        return output
##########################################################################
## CAMixing Block
class STransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(STransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = SAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class CTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(CTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Conv2DBlock(nn.Module):
    """3D 卷积块: Conv3D + BN + Activation"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, act='relu'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True) if act == 'relu' else nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=False) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=False))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class BN_Conv2d_Leaky(nn.Module):
    """
    BN_CONV_LeakyRELU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d_Leaky, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.leaky_relu(self.seq(x))

class Inception_A_res(nn.Module):
    """
    Inception-A block for Inception-ResNet-v1\
    and Inception-ResNet-v2 net
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n3, b3_n1, b3_n3_1, b3_n3_2, n1_linear):
        super(Inception_A_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n3, 3, 1, 1, bias=False),
        )
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n3_1, 3, 1, 1, bias=False),
            BN_Conv2d(b3_n3_1, b3_n3_2, 3, 1, 1, bias=False)
        )
        self.conv_linear = nn.Conv2d(b1 + b2_n3 + b3_n3_2, n1_linear, 1, 1, 0, bias=True)

        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n1_linear)
            )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat((out1, out2, out3), 1)
        out = self.conv_linear(out)
        out += x
        return F.relu(out)


class FB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=False, act=nn.ReLU(True)):
        super(FB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0:
                modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=True, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out


class ConvNeXtBlock(nn.Module):
    def __init__(self, num_channels):  # (1)
        super().__init__()
        hidden_channels = num_channels * 4  # (2)

        self.conv0 = nn.Conv2d(in_channels=num_channels,  # (3)
                               out_channels=num_channels,  # (4)
                               kernel_size=7,  # (5)
                               stride=1,
                               padding=3,  # (6)
                               groups=num_channels)  # (7)

        self.norm = nn.LayerNorm(normalized_shape=num_channels)  # (8)

        self.conv1 = nn.Conv2d(in_channels=num_channels,  # (9)
                               out_channels=hidden_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        self.gelu = nn.GELU()  # (10)

        self.conv2 = nn.Conv2d(in_channels=hidden_channels,  # (11)
                               out_channels=num_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)

    def forward(self, x):
        residual = x  # (1)
        # print(f'x & residual\t: {x.size()}')

        x = self.conv0(x)
        # print(f'after conv0\t: {x.size()}')

        x = x.permute(0, 2, 3, 1)  # (2)
        # print(f'after permute\t: {x.size()}')

        x = self.norm(x)
        # print(f'after norm\t: {x.size()}')

        x = x.permute(0, 3, 1, 2)  # (3)
        # print(f'after permute\t: {x.size()}')

        x = self.conv1(x)
        # print(f'after conv1\t: {x.size()}')

        x = self.gelu(x)
        # print(f'after gelu\t: {x.size()}')

        x = self.conv2(x)
        # print(f'after conv2\t: {x.size()}')

        x = x + residual  # (4)
        # print(f'after summation\t: {x.size()}')

        return x

class GLMIFBlock(nn.Module):
    """
    @inproceedings{hui2019lightweight,
      title={Lightweight Image Super-Resolution with Information Multi-distillation Network},
      author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},
      booktitle={Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)},
      pages={2024--2032},
      year={2019}
    }
    @inproceedings{zhang2019aim,
      title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
      author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
      booktitle={IEEE International Conference on Computer Vision Workshops},
      year={2019}
    }
    """

    def __init__(self, in_channels=64, out_channels=64, kernel_size=[1, 3, 5, 7], stride=1, padding=[0,1,2,3], bias=True,
                 mode='CL',
                 d_rate=[1/4,1/3,1/2], negative_slope=0.05,
                 ffn_expansion_factor = 4,dim= 64,LayerNorm_type = 'WithBias',img_size=[64, 64],
                 num_heads=4,
                 depths=[2],
                 norm_layer=nn.LayerNorm,
                 drop_rate=0):
        super(GLMIFBlock, self).__init__()

        self.ts = STransformerBlock(dim=dim, num_heads=num_heads,
                                    ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias,
                                    LayerNorm_type=LayerNorm_type)
        self.tc = CTransformerBlock(dim=dim, num_heads=num_heads,
                                    ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias,
                                    LayerNorm_type=LayerNorm_type)

        patches_resolution = [img_size[0] // 2, img_size[1] // 2]
        # self.rstcab = RSTCABlock(patches_resolution,  depths=2, num_heads=num_heads, nc=in_channels, window_size= 8)
        self.rg = nn.ModuleList([
            RSTCAGroup(patches_resolution, stage_id, depths, num_heads=num_heads, nc=in_channels, window_size= 8)
            for stage_id in range(len(depths))])

        self.conv = conv(in_channels, in_channels, bias=True, mode='C')

        self.d_nc1 = int(in_channels * d_rate[0])
        self.r_nc1 = int(in_channels *(1- d_rate[0]))
        self.d_nc2 = int(self.r_nc1 * d_rate[1])
        self.r_nc2 = int(self.r_nc1 *(1- d_rate[1]))
        self.d_nc3 = int(self.r_nc2 * d_rate[2])
        self.r_nc3 = int(self.r_nc2 *(1- d_rate[2]))

        # assert mode[0] == 'C', 'convolutional layer first'

        self.conv2d_1 = Conv2DBlock(in_ch=in_channels, out_ch=in_channels, act='relu')
        self.conv2d_2 = Conv2DBlock(in_ch=in_channels, out_ch=in_channels // 2, act='relu')
        self.conv2d_3 = Conv2DBlock(in_ch=in_channels, kernel_size=3,out_ch=in_channels, act='relu')
        self.conv1x1 = conv(self.d_nc1, self.d_nc1, kernel_size[0], stride, padding[0], bias, mode, negative_slope)
        self.conv3x3 = conv(self.d_nc2, self.d_nc2, kernel_size[1], stride, padding[1], bias, mode, negative_slope)
        self.conv5x5 = conv(self.d_nc3, self.d_nc3, kernel_size[2], stride, padding[2], bias, mode, negative_slope)
        self.conv7x7 = conv(self.d_nc3, self.d_nc3, kernel_size[3], stride, padding[3], bias, mode[0], negative_slope)

        self.inception_res = Inception_A_res(in_channels=in_channels,b1 = 32, b2_n1 =32, b2_n3=32,
                                             b3_n1 =32, b3_n3_1 =32, b3_n3_2=32, n1_linear=in_channels)

        conv_ = default_conv
        self.default_conv=default_conv(in_channels=in_channels*2,out_channels=in_channels,kernel_size=3)

        self.fusion_block = nn.Sequential(
                FB(conv_, in_channels*2, kernel_size=1, act=nn.ReLU(True)),
                FB(conv_, in_channels*2, kernel_size=1, act=nn.ReLU(True)),
                FB(conv_, in_channels*2, kernel_size=1, act=nn.ReLU(True)),
                FB(conv_, in_channels*2, kernel_size=1, act=nn.ReLU(True)),
            )

        self.norm = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm3 = LayerNorm3(in_channels*2, eps=1e-6, data_format="channels_first")
        self.BNnorm = nn.BatchNorm2d(in_channels * 2)
        # self.conv5 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)#新增
        # self.conv6 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)#新增
        # self.cca=CCALayer(self.d_nc*4)#新增
        # self.act = activation(act_type='prelu', inplace=False,neg_slope=0.05)#新增
        self.residual = IRMLP(in_channels*2, in_channels)

        self.Convnext = ConvNeXtBlock(num_channels=in_channels)

        self.MCC = MCC(f_number=512, num_heads=num_heads, padding_mode='reflect')


    def forward(self, x):

        xg = x
        xl = x

######## ####全局特征
        # xg1 = self.tc(xg)
        # xg2 = self.tc(xg)
        # xg = xg1+xg2

        x_size = (xg.shape[2], xg.shape[3])
        xg_1 = xg.flatten(2).transpose(1, 2)
        xg_1 = self.norm(xg_1)
        xg_1 = self.pos_drop(xg_1)
        for rg in self.rg:
            xg_1 = rg(xg_1, x_size)
        H, W = x_size
        B, L, C = xg_1.shape
        assert L == H * W, "input feature has wrong size"
        xg_1 = xg_1.view(B, H, W, C).permute(0, 3, 1, 2)

        xg_2 = self.MCC(x)
        xg = xg_1 + xg_2


###############局部特征
        # xl = xl.unsqueeze(1)  # 添加深度维度
        # xl_ = self.conv2d_1(xl)
        # xl = xl.squeeze(1)
        # d1, r1 = torch.split(xl_, (self.d_nc1, self.r_nc1), dim=1)
        # d2, r2 = torch.split(r1, (self.d_nc2, self.r_nc2), dim=1)
        # d3, r3 = torch.split(r2, (self.d_nc3, self.r_nc3), dim=1)
        # d4 = r3
        # d1 = self.conv1x1(d1)
        # d2 = self.conv3x3(d2)
        # d3 = self.conv5x5(d3)
        # d4 = self.conv7x7(d4)
        # xl_ = torch.cat([d1, d2, d3, d4], dim=1)  # [B, 64, H, W]
        # xl = xl.unsqueeze(1)  # 添加深度维度
        # xl = xl + self.conv2d_2(xl_)
        # xl = xl.squeeze(1)

        # xl = self.inception_res(xl)
        # xl = self.inception_res(xl)
        xl = self.Convnext(xl)
        xl = self.Convnext(xl)



        xgl=torch.cat([xg,xl],dim=1)

        # xgl = xgl + self.fusion_block(xgl)#
        # xgl = self.default_conv(xgl)#

        xgl = self.norm3(xgl)
        xgl = self.residual(xgl)

        out=x+xgl

        return out
# --------------------------------------------
# Enhanced Spatial Attention (ESA)
# --------------------------------------------
class ESA(nn.Module):
    def __init__(self, channel=64, reduction=4, bias=True):
        super(ESA, self).__init__()
        #               -->conv3x3(conv21)-----------------------------------------------------------------------------------------+
        # conv1x1(conv1)-->conv3x3-2(conv2)-->maxpool7-3-->conv3x3(conv3)(relu)-->conv3x3(conv4)(relu)-->conv3x3(conv5)-->bilinear--->conv1x1(conv6)-->sigmoid
        self.r_nc = channel // reduction
        self.conv1 = nn.Conv2d(channel, self.r_nc, kernel_size=1)
        self.conv21 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=1)
        self.conv2 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(self.r_nc, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = F.max_pool2d(self.conv2(x1), kernel_size=7, stride=3)  # 1/6
        x2 = self.relu(self.conv3(x2))
        x2 = self.relu(self.conv4(x2))
        x2 = F.interpolate(self.conv5(x2), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x2 = self.conv6(x2 + self.conv21(x1))
        return x.mul(self.sigmoid(x2))
        # return x.mul_(self.sigmoid(x2))


class CFRB(nn.Module):
    def __init__(self, in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1, bias=True, mode='CL', d_rate=0.5, negative_slope=0.05):
        super(CFRB, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = in_channels  # int(in_channels - self.d_nc)

        assert mode[0] == 'C', 'convolutional layer first'

        self.conv1_d = conv(in_channels, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.conv1_r = conv(in_channels, self.r_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv2_d = conv(self.r_nc, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.conv2_r = conv(self.r_nc, self.r_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv3_d = conv(self.r_nc, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.conv3_r = conv(self.r_nc, self.r_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv4_d = conv(self.r_nc, self.d_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.act = conv(mode=mode[-1], negative_slope=negative_slope)
        self.esa = ESA(in_channels, reduction=4, bias=True)

    def forward(self, x):
        d1 = self.conv1_d(x)
        x = self.act(self.conv1_r(x)+x)
        d2 = self.conv2_d(x)
        x = self.act(self.conv2_r(x)+x)
        d3 = self.conv3_d(x)
        x = self.act(self.conv3_r(x)+x)
        x = self.conv4_d(x)
        x = self.act(torch.cat([d1, d2, d3, x], dim=1))
        x = self.esa(self.conv1x1(x))
        return x


# --------------------------------------------
# Channel Attention (CA) Layer
# --------------------------------------------
class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_fc(y)
        return x * y


# --------------------------------------------
# Residual Channel Attention Block (RCAB)
# --------------------------------------------
class RCABlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', reduction=16, negative_slope=0.2):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, x):
        res = self.res(x)
        res = self.ca(res)
        return res + x


# --------------------------------------------
# Residual Channel Attention Group (RG)
# --------------------------------------------
class RCAGroup(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', reduction=16, nb=12, negative_slope=0.2):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding, bias, mode, reduction, negative_slope)  for _ in range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))
        self.rg = nn.Sequential(*RG)  # self.rg = ShortcutBlock(nn.Sequential(*RG))

    def forward(self, x):
        res = self.rg(x)
        return res + x


# --------------------------------------------
# Residual Dense Block
# style: 5 convs
# --------------------------------------------
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR', negative_slope=0.2):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel
        self.conv1 = conv(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(nc+gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(nc+2*gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(nc+3*gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv5 = conv(nc+4*gc, nc, kernel_size, stride, padding, bias, mode[:-1], negative_slope)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul_(0.2) + x


# --------------------------------------------
# Residual in Residual Dense Block
# 3x5c
# --------------------------------------------
class RRDB(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR', negative_slope=0.2):
        super(RRDB, self).__init__()

        self.RDB1 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.RDB2 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.RDB3 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul_(0.2) + x


"""
# --------------------------------------------
# Upsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# upsample_pixelshuffle
# upsample_upconv
# upsample_convtranspose
# --------------------------------------------
"""


# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    elif mode[0] == '4':
        uc = 'vC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1


'''
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
'''


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0,
                          bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        dim = dim//2
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x
'''
# --------------------------------------------
# NonLocalBlock2D:
# embedded_gaussian
# +W(softmax(thetaXphi)Xg)
# --------------------------------------------
'''


# --------------------------------------------
# non-local block with embedded_gaussian
# https://github.com/AlexHex7/Non-local_pytorch
# --------------------------------------------
class NonLocalBlock2D(nn.Module):
    def __init__(self, nc=64, kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='maxpool', negative_slope=0.2):

        super(NonLocalBlock2D, self).__init__()

        inter_nc = nc // 2
        self.inter_nc = inter_nc
        self.W = conv(inter_nc, nc, kernel_size, stride, padding, bias, mode='C'+act_mode)
        self.theta = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

        if downsample:
            if downsample_mode == 'avgpool':
                downsample_block = downsample_avgpool
            elif downsample_mode == 'maxpool':
                downsample_block = downsample_maxpool
            elif downsample_mode == 'strideconv':
                downsample_block = downsample_strideconv
            else:
                raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
            self.phi = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
            self.g = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
        else:
            self.phi = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')
            self.g = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_nc, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_nc, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_nc, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_nc, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z
