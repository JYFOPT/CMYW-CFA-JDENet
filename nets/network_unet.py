from tkinter.commondialog import Dialog

import torch
import torch.nn as nn
from torch.nn.functional import batch_norm

import nets.basicblock as B
from nets import rlfn_block
import numpy as np
from nets.DIA import DeformableAttention
from nets.DIA import DeformableAttention2
from nets.CAFM import CAFM
from nets.L2SKNet.L2SKNet import L2SKNet_UNet
'''
# ====================
# unet
# ====================
'''


class UNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], mode='C'+act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        self.m_down2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        self.m_down3 = B.sequential(*[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

        self.m_body  = B.sequential(*[B.conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=True, mode='C')

    def forward(self, x0):

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1) + x0

        
        return x


class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetRes, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        if(x.shape[2]%2 !=0 or x.shape[3]%2 !=0):
            # """
            #这里的作用是把宽高为奇数的变为偶数,防止上采样然后cat或者残差连接的时候尺寸不一致
            h, w = x.size()[-2:]
            paddingBottom = int(np.ceil(h/8)*8-h)
            paddingRight = int(np.ceil(w/8)*8-w)
            # print('paddingBottom',paddingBottom) #0  7
            # print('paddingRight',paddingRight) #0 1
            x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
            # exit()
            # """
        
            x1 = self.m_head(x)
            x2 = self.m_down1(x1)
            x3 = self.m_down2(x2)
            x4 = self.m_down3(x3)
            x = self.m_body(x4)
            x = self.m_up3(x+x4)
            x = self.m_up2(x+x3)
            x = self.m_up1(x+x2)
            x = self.m_tail(x+x1)
            x = x[..., :h, :w]
        else:
            x1 = self.m_head(x)
            x2 = self.m_down1(x1)
            x3 = self.m_down2(x2)
            x4 = self.m_down3(x3)
            x = self.m_body(x4)
            x = self.m_up3(x+x4)
            x = self.m_up2(x+x3)
            x = self.m_up1(x+x2)
            x = self.m_tail(x+x1)
        return x

class SIMDUNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='L',
                 downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(SIMDUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.IMDBlock(nc[0], nc[0], bias=False, mode='C'+act_mode) for _ in range(nb)],
                                    downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.IMDBlock(nc[1], nc[1], bias=False, mode='C'+act_mode) for _ in range(nb)],
                                    downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.IMDBlock(nc[2], nc[2], bias=False, mode='C'+act_mode) for _ in range(nb)],
                                    downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.IMDBlock(nc[3], nc[3], bias=False, mode='C'+act_mode) for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.IMDBlock(nc[2], nc[2], bias=False, mode='C'+act_mode) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.IMDBlock(nc[1], nc[1], bias=False, mode='C'+act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.IMDBlock(nc[0], nc[0], bias=False, mode='C'+act_mode) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        
        if(x.shape[2]%2 !=0 or x.shape[3]%2 !=0):
            # """
            #这里的作用是把宽高为奇数的变为偶数,防止上采样然后cat或者残差连接的时候尺寸不一致
            # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])    torch.Size([1, 4, 2041, 1359])
            h, w = x.size()[-2:]
            # print('h',h) #128  2041
            # print('w',w) #128  1359
            paddingBottom = int(np.ceil(h/8)*8-h)
            paddingRight = int(np.ceil(w/8)*8-w)
            # print('paddingBottom',paddingBottom) #0  7
            # print('paddingRight',paddingRight) #0 1
            x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
            # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])  torch.Size([1, 4, 2048, 1360])
            # exit()
            # """
        
            # print('x.shape',x.shape) #torch.Size([1, 4, 2041, 1359])
            x1 = self.m_head(x)
            # print('x1.shape',x1.shape) #torch.Size([1, 64, 2041, 1359]) 
            x2 = self.m_down1(x1)
            # print('x2.shape',x2.shape) #torch.Size([1, 128, 1020, 679])
            x3 = self.m_down2(x2)
            # print('x3.shape',x3.shape) #torch.Size([1, 256, 510, 339])
            x4 = self.m_down3(x3)
            # print('x4.shape',x4.shape) #torch.Size([1, 512, 255, 169])
            x = self.m_body(x4)
            # print('x.shape',x.shape) #torch.Size([1, 512, 255, 169])
            x = self.m_up3(x+x4)
            # print('x.shape',x.shape) #torch.Size([1, 256, 510, 338])
            x = self.m_up2(x+x3)
            # print('x.shape',x.shape)
            x = self.m_up1(x+x2)
            # print('x.shape',x.shape)
            x = self.m_tail(x+x1)
            # print('x.shape',x.shape)
            x = x[..., :h, :w]
        else:
            x1 = self.m_head(x)
            x2 = self.m_down1(x1)
            x3 = self.m_down2(x2)
            x4 = self.m_down3(x3)
            x = self.m_body(x4)
            x = self.m_up3(x+x4)
            x = self.m_up2(x+x3)
            x = self.m_up1(x+x2)
            x = self.m_tail(x+x1)
        return x


class SpatialAttention(nn.Module):
   def __init__(self, kernel_size=7):
       super(SpatialAttention, self).__init__()
       assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
       padding = 3 if kernel_size == 7 else 1
       self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
       self.sigmoid = nn.Sigmoid()
   def forward(self, x):
       avg_out = torch.mean(x, dim=1, keepdim=True)
       max_out, _ = torch.max(x, dim=1, keepdim=True)
       x = torch.cat([avg_out, max_out], dim=1)
       x = self.conv1(x)
       return self.sigmoid(x)


class ChannelAttention(nn.Module):
    """
    通道注意力模块（SE Module）
    Args:
        channels: 输入特征图的通道数
        reduction: 压缩率（通常取16，平衡计算量和性能）
    """

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 1. Squeeze：全局平均池化（压缩空间维度 H*W -> 1）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 2. Excitation：全连接层学习通道权重
        self.fc = nn.Sequential(
            # 第一层：通道数压缩为 channels//reduction
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            # 第二层：恢复原通道数，输出每个通道的权重
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()  # 归一化到0-1之间，作为权重
        )

    def forward(self, x):
        # x: 输入特征图，shape = [batch_size, channels, height, width]
        b, c, h, w = x.size()

        # Squeeze：全局平均池化，shape从 [b,c,h,w] -> [b,c,1,1]
        avg_out = self.avg_pool(x).view(b, c)  # 展平为 [b,c]

        # Excitation：学习通道权重，shape [b,c]
        fc_out = self.fc(avg_out).view(b, c, 1, 1)  # 恢复形状 [b,c,1,1]



        return fc_out


class DGASIMDUNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='L',
                 downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(DGASIMDUNet, self).__init__()

        # 头部：双分支
        self.m_head_3ch = B.conv(3, 32, bias=False, mode='C')  # 0,1,2通道 → 48
        self.m_head_1ch = B.conv(1, 32, bias=False, mode='C')  # 第3通道  → 16
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(32)

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.IMDBlock(nc[0], nc[0], bias=False, mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.IMDBlock(nc[1], nc[1], bias=False, mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.IMDBlock(nc[2], nc[2], bias=False, mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = B.sequential(*[B.IMDBlock(nc[3], nc[3], bias=False, mode='C' + act_mode) for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'),
                                  *[B.IMDBlock(nc[2], nc[2], bias=False, mode='C' + act_mode) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                  *[B.IMDBlock(nc[1], nc[1], bias=False, mode='C' + act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'),
                                  *[B.IMDBlock(nc[0], nc[0], bias=False, mode='C' + act_mode) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):

        if (x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0):
            # """
            # 这里的作用是把宽高为奇数的变为偶数,防止上采样然后cat或者残差连接的时候尺寸不一致
            # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])    torch.Size([1, 4, 2041, 1359])
            h, w = x.size()[-2:]
            # print('h',h) #128  2041
            # print('w',w) #128  1359
            paddingBottom = int(np.ceil(h / 8) * 8 - h)
            paddingRight = int(np.ceil(w / 8) * 8 - w)
            # print('paddingBottom',paddingBottom) #0  7
            # print('paddingRight',paddingRight) #0 1
            x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
            # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])  torch.Size([1, 4, 2048, 1360])
            # exit()
            # """

            # print('x.shape',x.shape) #torch.Size([1, 4, 2041, 1359])
            x1 = self.m_head(x)
            # print('x1.shape',x1.shape) #torch.Size([1, 64, 2041, 1359])
            x2 = self.m_down1(x1)
            # print('x2.shape',x2.shape) #torch.Size([1, 128, 1020, 679])
            x3 = self.m_down2(x2)
            # print('x3.shape',x3.shape) #torch.Size([1, 256, 510, 339])
            x4 = self.m_down3(x3)
            # print('x4.shape',x4.shape) #torch.Size([1, 512, 255, 169])
            x = self.m_body(x4)
            # print('x.shape',x.shape) #torch.Size([1, 512, 255, 169])
            x = self.m_up3(x + x4)
            # print('x.shape',x.shape) #torch.Size([1, 256, 510, 338])
            x = self.m_up2(x + x3)
            # print('x.shape',x.shape)
            x = self.m_up1(x + x2)
            # print('x.shape',x.shape)
            x = self.m_tail(x + x1)
            # print('x.shape',x.shape)
            x = x[..., :h, :w]
        else:
            # x: (B, 4, H, W)
            x_3ch = x[:, :3, :, :]  # 0,1,2 通道
            x_1ch = x[:, 3:4, :, :]  # 第3通道，保持 4-dim

            f_3ch = self.m_head_3ch(x_3ch)  # (B,32,H,W)
            f_1ch = self.m_head_1ch(x_1ch)  # (B,32,H,W)
            s_attention = self.spatial_attention(f_1ch)
            c_attention =  self.channel_attention(f_3ch)
            f_3ch = f_3ch * s_attention
            f_1ch = f_1ch * c_attention
            x1 = torch.cat([f_3ch, f_1ch], dim=1)




            x2 = self.m_down1(x1)
            x3 = self.m_down2(x2)
            x4 = self.m_down3(x3)
            x = self.m_body(x4)
            x = self.m_up3(x + x4)
            x = self.m_up2(x + x3)
            x = self.m_up1(x + x2)
            x = self.m_tail(x + x1)
        return x

class DGAGLMIFUNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='L',
                 downsample_mode='strideconv',
                 upsample_mode='convtranspose',
                 heads = [4,8,16,32]):
        super(DGAGLMIFUNet, self).__init__()

        # 头部：双分支
        self.m_head_3ch = B.conv(3, 32, bias=False, mode='CBL')  # 0,1,2通道 → 48
        self.m_head_1ch = B.conv(1, 32, bias=False, mode='CBL')  # 第3通道  → 16
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(32)

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        dim = [64,128,256,512]
        self.m_down1 = B.sequential(*[B.GLMIFBlock(nc[0], nc[0], dim=dim[0], bias=False, mode='CB' + act_mode,
                                                   num_heads=heads[0]) for _ in range(nb)],
                                    downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.GLMIFBlock(nc[1], nc[1], dim=dim[1], bias=False, mode='CB' + act_mode,
                                                   num_heads=heads[1]) for _ in range(nb)],
                                    downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.GLMIFBlock(nc[2], nc[2], dim=dim[2], bias=False, mode='CB' + act_mode,
                                                   num_heads=heads[2]) for _ in range(nb)],
                                    downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = B.sequential(*[B.GLMIFBlock(nc[3], nc[3], dim=dim[3], bias=False, mode='CB' + act_mode,
                                                  num_heads=heads[3]) for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2],  bias=False, mode='2'),
                                  *[B.GLMIFBlock(nc[2], nc[2],dim=dim[2], bias=False, mode='CB' + act_mode,
                                                 num_heads=heads[2]) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                  *[B.GLMIFBlock(nc[1], nc[1], dim=dim[1], bias=False, mode='CB' + act_mode,
                                                 num_heads=heads[1]) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0],  bias=False, mode='2'),
                                  *[B.GLMIFBlock(nc[0], nc[0],dim=dim[0], bias=False, mode='CB' + act_mode,
                                                 num_heads=heads[0]) for _ in range(nb)])

        self.m_tail = B.sequential(B.conv(nc[0], nc[0], bias=False, mode='CR'),
                                   B.conv(nc[0], out_nc, bias=False, mode='C'))
        self.conv = B.conv(nc[0], nc[0], bias=False, mode='CNR')

    def forward(self, x):

        if (x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0):
            # """
            # 这里的作用是把宽高为奇数的变为偶数,防止上采样然后cat或者残差连接的时候尺寸不一致
            # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])    torch.Size([1, 4, 2041, 1359])
            h, w = x.size()[-2:]
            # print('h',h) #128  2041
            # print('w',w) #128  1359
            paddingBottom = int(np.ceil(h / 8) * 8 - h)
            paddingRight = int(np.ceil(w / 8) * 8 - w)
            # print('paddingBottom',paddingBottom) #0  7
            # print('paddingRight',paddingRight) #0 1
            x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
            # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])  torch.Size([1, 4, 2048, 1360])
            # exit()
            # """

            # print('x.shape',x.shape) #torch.Size([1, 4, 2041, 1359])



            x_3ch = x[:, :3, :, :]  # 0,1,2 通道
            x_1ch = x[:, 3:4, :, :]  # 第3通道，保持 4-dim

            f_3ch = self.m_head_3ch(x_3ch)  # (B,32,H,W)
            f_1ch = self.m_head_1ch(x_1ch)  # (B,32,H,W)
            s_attention = self.spatial_attention(f_1ch)
            c_attention =  self.channel_attention(f_3ch)
            f_3ch = f_3ch * c_attention
            f_1ch = f_1ch * s_attention
            x1 = torch.cat([f_3ch, f_1ch], dim=1)
            x1 = self.conv(x1)



            # print('x1.shape',x1.shape) #torch.Size([1, 64, 2041, 1359])
            x2 = self.m_down1(x1)
            # print('x2.shape',x2.shape) #torch.Size([1, 128, 1020, 679])
            x3 = self.m_down2(x2)
            # print('x3.shape',x3.shape) #torch.Size([1, 256, 510, 339])
            x4 = self.m_down3(x3)
            # print('x4.shape',x4.shape) #torch.Size([1, 512, 255, 169])
            x = self.m_body(x4)
            # print('x.shape',x.shape) #torch.Size([1, 512, 255, 169])
            x = self.m_up3(x + x4)
            # print('x.shape',x.shape) #torch.Size([1, 256, 510, 338])
            x = self.m_up2(x + x3)
            # print('x.shape',x.shape)
            x = self.m_up1(x + x2)
            # print('x.shape',x.shape)
            x = self.m_tail(x + x1)
            # print('x.shape',x.shape)
            x = x[..., :h, :w]
        else:
            # x: (B, 4, H, W)
            x_3ch = x[:, :3, :, :]  # 0,1,2 通道
            x_1ch = x[:, 3:4, :, :]  # 第3通道，保持 4-dim

            f_3ch = self.m_head_3ch(x_3ch)  # (B,32,H,W)
            f_1ch = self.m_head_1ch(x_1ch)  # (B,32,H,W)
            s_attention = self.spatial_attention(f_1ch)
            c_attention =  self.channel_attention(f_3ch)
            f_3ch = f_3ch * c_attention
            f_1ch = f_1ch * s_attention
            x1 = torch.cat([f_3ch, f_1ch], dim=1)
            x1 = self.conv(x1)   #一个卷积融合特征不够用
            x1L = self.conv(x1)






            x2L = self.m_down1(x1L)
            x3L = self.m_down2(x2L)
            x4L = self.m_down3(x3L)
            x4R = self.m_body(x4L)
            x3R = self.m_up3(x4R + x4L)
            x2R = self.m_up2(x3R + x3L)
            x1R = self.m_up1(x2R + x2L)
            x = self.m_tail(x1R + x1L)
        return x


class Res3UNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='L',
                 downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(Res3UNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.Res3Block(nc[0], nc[0], bias=False, mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.Res3Block(nc[1], nc[1], bias=False, mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.Res3Block(nc[2], nc[2], bias=False, mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = B.sequential(*[B.Res3Block(nc[3], nc[3], bias=False, mode='C' + act_mode) for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'),
                                  *[B.Res3Block(nc[2], nc[2], bias=False, mode='C' + act_mode) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                  *[B.Res3Block(nc[1], nc[1], bias=False, mode='C' + act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'),
                                  *[B.Res3Block(nc[0], nc[0], bias=False, mode='C' + act_mode) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):

        if (x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0):
            # """
            # 这里的作用是把宽高为奇数的变为偶数,防止上采样然后cat或者残差连接的时候尺寸不一致
            # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])    torch.Size([1, 4, 2041, 1359])
            h, w = x.size()[-2:]
            # print('h',h) #128  2041
            # print('w',w) #128  1359
            paddingBottom = int(np.ceil(h / 8) * 8 - h)
            paddingRight = int(np.ceil(w / 8) * 8 - w)
            # print('paddingBottom',paddingBottom) #0  7
            # print('paddingRight',paddingRight) #0 1
            x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
            # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])  torch.Size([1, 4, 2048, 1360])
            # exit()
            # """

            # print('x.shape',x.shape) #torch.Size([1, 4, 2041, 1359])
            x1 = self.m_head(x)
            # print('x1.shape',x1.shape) #torch.Size([1, 64, 2041, 1359])
            x2 = self.m_down1(x1)
            # print('x2.shape',x2.shape) #torch.Size([1, 128, 1020, 679])
            x3 = self.m_down2(x2)
            # print('x3.shape',x3.shape) #torch.Size([1, 256, 510, 339])
            x4 = self.m_down3(x3)
            # print('x4.shape',x4.shape) #torch.Size([1, 512, 255, 169])
            x = self.m_body(x4)
            # print('x.shape',x.shape) #torch.Size([1, 512, 255, 169])
            x = self.m_up3(x + x4)
            # print('x.shape',x.shape) #torch.Size([1, 256, 510, 338])
            x = self.m_up2(x + x3)
            # print('x.shape',x.shape)
            x = self.m_up1(x + x2)
            # print('x.shape',x.shape)
            x = self.m_tail(x + x1)
            # print('x.shape',x.shape)
            x = x[..., :h, :w]
        else:
            x1 = self.m_head(x)
            x2 = self.m_down1(x1)
            x3 = self.m_down2(x2)
            x4 = self.m_down3(x3)
            x = self.m_body(x4)
            x = self.m_up3(x + x4)
            x = self.m_up2(x + x3)
            x = self.m_up1(x + x2)
            x = self.m_tail(x + x1)
        return x

class sSE(nn.Module):  # 空间(Space)注意力
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)  # 定义一个卷积层，用于将输入通道转换为单通道
        self.norm = nn.Sigmoid()  # 应用Sigmoid激活函数进行归一化

    def forward(self, x):
        q = self.conv(x)  # 使用卷积层减少通道数至1：b c h w -> b 1 h w
        q = self.norm(q)  # 对卷积后的结果应用Sigmoid激活函数：b 1 h w
        return x * q  # 通过广播机制将注意力权重应用到每个通道上


class cSE(nn.Module):  # 通道(channel)注意力
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 使用自适应平均池化，输出大小为1x1
        self.relu = nn.ReLU()  # ReLU激活函数
        self.Conv_Squeeze = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)  # 通道压缩卷积层
        self.norm = nn.Sigmoid()  # Sigmoid激活函数进行归一化
        self.Conv_Excitation = nn.Conv2d(in_ch // 2, in_ch, kernel_size=1, bias=False)  # 通道激励卷积层

    def forward(self, x):
        z = self.avgpool(x)  # 对输入特征进行全局平均池化：b c 1 1
        z = self.Conv_Squeeze(z)  # 通过通道压缩卷积减少通道数：b c//2 1 1
        z = self.relu(z)  # 应用ReLU激活函数
        z = self.Conv_Excitation(z)  # 通过通道激励卷积恢复通道数：b c 1 1
        z = self.norm(z)  # 对激励结果应用Sigmoid激活函数进行归一化
        return x * z.expand_as(x)  # 将归一化权重乘以原始特征，使用expand_as扩展维度与原始特征相匹配


class scSE(nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.cSE = cSE(in_ch)  # 通道注意力模块
        self.sSE = sSE(in_ch)  # 空间注意力模块

    def forward(self, x):
        c_out = self.cSE(x)  # 应用通道注意力
        s_out = self.sSE(x)  # 应用空间注意力
        return c_out + s_out  # 合并通道和空间注意力的输出

class JDENet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='L',
                 downsample_mode='strideconv',
                 upsample_mode='pixelshuffle'):
        super(JDENet, self).__init__()

        # 头部：双分支
        self.m_head_3ch = B.conv(3, 32, bias=False, mode='CBL')  # 0,1,2通道 → 48

        # 定义 3D 卷积：输入通道为 1，输出通道为 1，核大小 (3,3,3)
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3,1,1), padding=(1,0,0))  # 保持深度尺寸不变
        self.bn = nn.BatchNorm2d(64)  # 这里的 64 应与 x_3ch_64 的通道数一致
        self.gelu = nn.GELU()

        self.m_head_1ch = B.conv(1, 32, bias=False, mode='CBL')  # 第3通道  → 16
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(64)

        # self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')
        self.m_head_1 = B.conv(3, nc[0], bias=False, mode='CBG')
        self.m_head_2 = B.conv(1, nc[0], bias=False, mode='CBG')
        self.m_head_3 = B.conv(nc[0] * 2, nc[0], bias=False, mode='CBG')

        self.SCSE = scSE(nc[0])


        self.Res3 = B.Res3Block(nc[0], nc[0], bias=False, mode='CB' + act_mode)
        self.Res2 = B.SCERes2Block(nc[0], nc[0])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.SCERes2Block(nc[0], nc[0]) for _ in range(nb)],
                                    downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.SCERes2Block(nc[1], nc[1]) for _ in range(nb)],
                                    downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.SCERes2Block(nc[2], nc[2]) for _ in range(nb)],
                                    downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = B.sequential(*[B.GLMIFBlock(nc[3], nc[3], dim=512, bias=False, mode='CB' + act_mode,
                                                  num_heads=64) for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'),
                                  *[B.SCERes2Block(nc[2], nc[2]) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                  *[B.SCERes2Block(nc[1], nc[1]) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'),
                                  *[B.SCERes2Block(nc[0], nc[0]) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='CBR')


    def forward(self, x):

        if (x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0):
            # """
            # 这里的作用是把宽高为奇数的变为偶数,防止上采样然后cat或者残差连接的时候尺寸不一致
            # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])    torch.Size([1, 4, 2041, 1359])
            h, w = x.size()[-2:]
            # print('h',h) #128  2041
            # print('w',w) #128  1359
            paddingBottom = int(np.ceil(h / 8) * 8 - h)
            paddingRight = int(np.ceil(w / 8) * 8 - w)
            # print('paddingBottom',paddingBottom) #0  7
            # print('paddingRight',paddingRight) #0 1
            x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
            # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])  torch.Size([1, 4, 2048, 1360])
            # exit()
            # """

            # print('x.shape',x.shape) #torch.Size([1, 4, 2041, 1359])
            x1 = self.m_head(x)
            # print('x1.shape',x1.shape) #torch.Size([1, 64, 2041, 1359])
            x2 = self.m_down1(x1)
            # print('x2.shape',x2.shape) #torch.Size([1, 128, 1020, 679])
            x3 = self.m_down2(x2)
            # print('x3.shape',x3.shape) #torch.Size([1, 256, 510, 339])
            x4 = self.m_down3(x3)
            # print('x4.shape',x4.shape) #torch.Size([1, 512, 255, 169])
            x = self.m_body(x4)
            # print('x.shape',x.shape) #torch.Size([1, 512, 255, 169])
            x = self.m_up3(x + x4)
            # print('x.shape',x.shape) #torch.Size([1, 256, 510, 338])
            x = self.m_up2(x + x3)
            # print('x.shape',x.shape)
            x = self.m_up1(x + x2)
            # print('x.shape',x.shape)
            x = self.m_tail(x + x1)
            # print('x.shape',x.shape)
            x = x[..., :h, :w]
        else:
            # input
            x_3ch = x[:, :3, :, :]  # 0,1,2 通道
            x_3ch_64 = self.m_head_1(x_3ch)#             #64
            x_1ch = x[:, 3:4, :, :]  # 3通道
            x_1ch_64 = self.m_head_2(x_1ch)  #64

            s_attention = self.spatial_attention(x_1ch_64)
            c_attention = self.channel_attention(x_3ch_64)
            x_3ch_64 = x_3ch_64 * c_attention  # 64
            x_3ch_64 = x_3ch_64 * s_attention  # 64

            x0=torch.cat([x_3ch_64, x_1ch_64], dim=1)      #128
            #head
            x00 = self.m_head_3(x0)  #128-64

            x00 = self.SCSE(x00)

            # # demosaicking
            x00 = self.Res2(x00)      #64-64
            x00 = self.Res2(x00)  # 64-64
            x00 = self.Res2(x00)  # 64-64
            x1 = x00              # 64-64



            # enhance
            x2 = self.m_down1(x1)     #64-128
            x3 = self.m_down2(x2)     #128-256
            x4 = self.m_down3(x3)     #256-512
            x = self.m_body(x4)       #512-512
            x = self.m_up3(x + x4)    #512-256
            x = self.m_up2(x + x3)    #256-128
            x = self.m_up1(x + x2)    #128-64

            #output
            x = self.m_tail(x)
        return x

class DUJRENet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4,layers=4, act_mode='L',
                 downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(DUJRENet, self).__init__()

        # 头部：双分支
        self.m_head_3ch = B.conv(3, 32, bias=False, mode='CBL')  # 0,1,2通道 → 48
        self.m_head_1ch = B.conv(1, 32, bias=False, mode='CBL')  # 第3通道  → 16
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(3)

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')
        self.Res3 = B.Res3Block(nc[0], nc[0], bias=False, mode='C' + act_mode)

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))


        ####SCERES2B

        self.D_down1 = B.sequential(*[B.SCERes2Block(nc[0], nc[0]) for _ in range(nb)],
                                    downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.D_down2 = B.sequential(*[B.SCERes2Block(nc[1], nc[1]) for _ in range(nb)],
                                    downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.D_down3 = B.sequential(*[B.SCERes2Block(nc[2], nc[2]) for _ in range(nb)],
                                    downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.D_body = B.sequential(*[B.SCERes2Block(nc[3], nc[3]) for _ in range(nb)])
        self.D_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'),
                                  *[B.SCERes2Block(nc[2], nc[2]) for _ in range(nb)])
        self.D_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                  *[B.SCERes2Block(nc[1], nc[1]) for _ in range(nb)])
        self.D_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'),
                                  *[B.SCERes2Block(nc[0], nc[0]) for _ in range(nb)])

        # 为4个分支分别定义不同 num_heads 的 MCC 模块
        num_heads_list = [8 * (2 ** layer) for layer in range(layers)]  # [8, 16, 32, 64]


        #SIMDFORMER
        self.E_down1 = B.sequential(*[B.SimdFormerBlock(nc[0], nc[0],num_heads_list[0]) for _ in range(nb)],
                                    downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.E_down2 = B.sequential(*[B.SimdFormerBlock(nc[1], nc[1],num_heads_list[1]) for _ in range(nb)],
                                    downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.E_down3 = B.sequential(*[B.SimdFormerBlock(nc[2], nc[2],num_heads_list[2]) for _ in range(nb)],
                                    downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.E_body = B.sequential(*[B.SimdFormerBlock(nc[3], nc[3],num_heads_list[3]) for _ in range(nb)])

        self.E_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'),
                                  *[B.SimdFormerBlock(nc[2], nc[2],num_heads_list[2]) for _ in range(nb)])
        self.E_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                  *[B.SimdFormerBlock(nc[1], nc[1],num_heads_list[1]) for _ in range(nb)])
        self.E_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'),
                                  *[B.SimdFormerBlock(nc[0], nc[0],num_heads_list[0]) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):

        if (x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0):
            # """
            # 这里的作用是把宽高为奇数的变为偶数,防止上采样然后cat或者残差连接的时候尺寸不一致
            # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])    torch.Size([1, 4, 2041, 1359])
            h, w = x.size()[-2:]
            # print('h',h) #128  2041
            # print('w',w) #128  1359
            paddingBottom = int(np.ceil(h / 8) * 8 - h)
            paddingRight = int(np.ceil(w / 8) * 8 - w)
            # print('paddingBottom',paddingBottom) #0  7
            # print('paddingRight',paddingRight) #0 1
            x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
            # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])  torch.Size([1, 4, 2048, 1360])
            # exit()
            # """

            # print('x.shape',x.shape) #torch.Size([1, 4, 2041, 1359])
            x1 = self.m_head(x)
            # print('x1.shape',x1.shape) #torch.Size([1, 64, 2041, 1359])
            x2 = self.m_down1(x1)
            # print('x2.shape',x2.shape) #torch.Size([1, 128, 1020, 679])
            x3 = self.m_down2(x2)
            # print('x3.shape',x3.shape) #torch.Size([1, 256, 510, 339])
            x4 = self.m_down3(x3)
            # print('x4.shape',x4.shape) #torch.Size([1, 512, 255, 169])
            x = self.m_body(x4)
            # print('x.shape',x.shape) #torch.Size([1, 512, 255, 169])
            x = self.m_up3(x + x4)
            # print('x.shape',x.shape) #torch.Size([1, 256, 510, 338])
            x = self.m_up2(x + x3)
            # print('x.shape',x.shape)
            x = self.m_up1(x + x2)
            # print('x.shape',x.shape)
            x = self.m_tail(x + x1)
            # print('x.shape',x.shape)
            x = x[..., :h, :w]
        else:
            # # input
            # x_3ch = x[:, :3, :, :]  # 0,1,2 通道             #3
            # x_1ch = x[:, 3:4, :, :]  # 第3通道，保持 4-dim     #1
            # s_attention = self.spatial_attention(x_1ch)
            # c_attention =  self.channel_attention(x_3ch)
            # x_3ch = x_3ch * c_attention                     #3
            # x_3ch = x_3ch * s_attention                     #3
            # x0=torch.cat([x_3ch, x_1ch], dim=1)      #4
            #head
            x0 = self.m_head(x)  #4-64

            #  RAW IMAGE DENOISE
            xR1_L = self.D_down1(x0)     #64-128
            xR2_L = self.D_down2(xR1_L)     #128-256
            xR3_L = self.D_down3(xR2_L)     #256-512
            xR3_R = self.D_body(xR3_L)       #512-512
            xR2_R = self.D_up3(xR3_R+xR3_L)    #512-256
            xR1_R = self.D_up2(xR2_R + xR2_L)    #256-128
            xR = self.D_up1(xR1_R + xR1_L)    #128-64


            # RESTORATION & ENHANCE
            x2 = self.E_down1(xR)     #64-128
            x3 = self.E_down2(x2)     #128-256
            x4 = self.E_down3(x3)     #256-512
            x = self.E_body(x4)       #512-512
            x = self.E_up3(x + x4)    #512-256
            x = self.E_up2(x + x3)    #256-128
            xE = self.E_up1(x + x2)    #128-64

            #output
            x = self.m_tail(xE)
        return x

class RASTNet(nn.Module):
    def __init__(self, nc=[64, 128, 256, 512]):
        super(RASTNet, self).__init__()


        # 头部：双分支
        self.m_head_3ch = B.conv(3, 48, bias=False, mode='C')  # 0,1,2通道 → 48
        self.m_head_1ch = B.conv(1, 16, bias=False, mode='C')  # 第3通道  → 16
        self.L2SKNet_UNet=L2SKNet_UNet(in_ch=nc[0], out_ch=3)
        self.dia=DeformableAttention(distortionmode=True)
        self.dia2 = DeformableAttention2(distortionmode=True)


    def forward(self, x):
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if (x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0):
            # """
            # 这里的作用是把宽高为奇数的变为偶数,防止上采样然后cat或者残差连接的时候尺寸不一致
            # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])    torch.Size([1, 4, 2041, 1359])
            h, w = x.size()[-2:]
            # print('h',h) #128  2041
            # print('w',w) #128  1359
            paddingBottom = int(np.ceil(h / 8) * 8 - h)
            paddingRight = int(np.ceil(w / 8) * 8 - w)
            # print('paddingBottom',paddingBottom) #0  7
            # print('paddingRight',paddingRight) #0 1
            x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
            # print('x.shape',x.shape) #torch.Size([8, 4, 128, 128])  torch.Size([1, 4, 2048, 1360])
            # exit()
            # """

            # print('x.shape',x.shape) #torch.Size([1, 4, 2041, 1359])
            x1 = self.m_head(x)
            # print('x1.shape',x1.shape) #torch.Size([1, 64, 2041, 1359])
            x2 = self.m_down1(x1)
            # print('x2.shape',x2.shape) #torch.Size([1, 128, 1020, 679])
            x3 = self.m_down2(x2)
            # print('x3.shape',x3.shape) #torch.Size([1, 256, 510, 339])
            x4 = self.m_down3(x3)
            # print('x4.shape',x4.shape) #torch.Size([1, 512, 255, 169])
            x = self.m_body(x4)
            # print('x.shape',x.shape) #torch.Size([1, 512, 255, 169])
            x = self.m_up3(x + x4)
            # print('x.shape',x.shape) #torch.Size([1, 256, 510, 338])
            x = self.m_up2(x + x3)
            # print('x.shape',x.shape)
            x = self.m_up1(x + x2)
            # print('x.shape',x.shape)
            x = self.m_tail(x + x1)
            # print('x.shape',x.shape)
            x = x[..., :h, :w]
        else:

            # x: (B, 4, H, W)
            x_3ch = x[:, :3, :, :]  # 0,1,2 通道
            x_1ch = x[:, 3:4, :, :]  # 第3通道，保持 4-dim

            f_3ch = self.m_head_3ch(x_3ch)  # (B,48,H,W)

            f_3ch = self.dia(f_3ch)
            f_1ch = self.m_head_1ch(x_1ch)  # (B,16,H,W)

            f_1ch = self.dia2(f_1ch)
            x1 = torch.cat([f_3ch, f_1ch], dim=1)  # (B,64,H,W)

            x = self.L2SKNet_UNet(x1)
        return x

class RLFBUNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(RLFBUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[rlfn_block.RLFB(in_channels=nc[0],out_channels=nc[0]) for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[rlfn_block.RLFB(in_channels=nc[1],out_channels=nc[1]) for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[rlfn_block.RLFB(in_channels=nc[2],out_channels=nc[2]) for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[rlfn_block.RLFB(in_channels=nc[3],out_channels=nc[3]) for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[rlfn_block.RLFB(in_channels=nc[2],out_channels=nc[2]) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[rlfn_block.RLFB(in_channels=nc[1],out_channels=nc[1]) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[rlfn_block.RLFB(in_channels=nc[0],out_channels=nc[0]) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        if(x.shape[2]%2 !=0 or x.shape[3]%2 !=0):
            # """
            #这里的作用是把宽高为奇数的变为偶数,防止上采样然后cat或者残差连接的时候尺寸不一致
            h, w = x.size()[-2:]
            paddingBottom = int(np.ceil(h/8)*8-h)
            paddingRight = int(np.ceil(w/8)*8-w)
            # print('paddingBottom',paddingBottom) #0  7
            # print('paddingRight',paddingRight) #0 1
            x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
            # exit()
            # """
        
            x1 = self.m_head(x)
            x2 = self.m_down1(x1)
            x3 = self.m_down2(x2)
            x4 = self.m_down3(x3)
            x = self.m_body(x4)
            x = self.m_up3(x+x4)
            x = self.m_up2(x+x3)
            x = self.m_up1(x+x2)
            x = self.m_tail(x+x1)
            x = x[..., :h, :w]
        else:
            x1 = self.m_head(x)
            x2 = self.m_down1(x1)
            x3 = self.m_down2(x2)
            x4 = self.m_down3(x3)
            x = self.m_body(x4)
            x = self.m_up3(x+x4)
            x = self.m_up2(x+x3)
            x = self.m_up1(x+x2)
            x = self.m_tail(x+x1)
        return x

class SRLFBUNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(SRLFBUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[rlfn_block.SRLFB(in_channels=nc[0],out_channels=nc[0]) for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[rlfn_block.SRLFB(in_channels=nc[1],out_channels=nc[1]) for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[rlfn_block.SRLFB(in_channels=nc[2],out_channels=nc[2]) for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[rlfn_block.SRLFB(in_channels=nc[3],out_channels=nc[3]) for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[rlfn_block.SRLFB(in_channels=nc[2],out_channels=nc[2]) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[rlfn_block.SRLFB(in_channels=nc[1],out_channels=nc[1]) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[rlfn_block.SRLFB(in_channels=nc[0],out_channels=nc[0]) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        if(x.shape[2]%2 !=0 or x.shape[3]%2 !=0):
            # """
            #这里的作用是把宽高为奇数的变为偶数,防止上采样然后cat或者残差连接的时候尺寸不一致
            h, w = x.size()[-2:]
            paddingBottom = int(np.ceil(h/8)*8-h)
            paddingRight = int(np.ceil(w/8)*8-w)
            # print('paddingBottom',paddingBottom) #0  7
            # print('paddingRight',paddingRight) #0 1
            x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
            # exit()
            # """
        
            x1 = self.m_head(x)
            x2 = self.m_down1(x1)
            x3 = self.m_down2(x2)
            x4 = self.m_down3(x3)
            x = self.m_body(x4)
            x = self.m_up3(x+x4)
            x = self.m_up2(x+x3)
            x = self.m_up1(x+x2)
            x = self.m_tail(x+x1)
            x = x[..., :h, :w]
        else:
            x1 = self.m_head(x)
            x2 = self.m_down1(x1)
            x3 = self.m_down2(x2)
            x4 = self.m_down3(x3)
            x = self.m_body(x4)
            x = self.m_up3(x+x4)
            x = self.m_up2(x+x3)
            x = self.m_up1(x+x2)
            x = self.m_tail(x+x1)
        return x














class UNetResSubP(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetResSubP, self).__init__()
        sf = 2
        self.m_ps_down = B.PixelUnShuffle(sf)
        self.m_ps_up = nn.PixelShuffle(sf)
        self.m_head = B.conv(in_nc*sf*sf, nc[0], mode='C'+act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], mode='C'+act_mode+'C') for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[B.ResBlock(nc[2], nc[2], mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[B.ResBlock(nc[1], nc[1], mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[B.ResBlock(nc[0], nc[0], mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc*sf*sf, bias=False, mode='C')

    def forward(self, x0):
        x0_d = self.m_ps_down(x0)
        x1 = self.m_head(x0_d)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)
        x = self.m_ps_up(x) + x0

        return x


class UNetPlus(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=1, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetPlus, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode[1]))
        self.m_down2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode[1]))
        self.m_down3 = B.sequential(*[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode[1]))

        self.m_body  = B.sequential(*[B.conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb-1)], B.conv(nc[2], nc[2], mode='C'+act_mode[1]))
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb-1)], B.conv(nc[1], nc[1], mode='C'+act_mode[1]))
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb-1)], B.conv(nc[0], nc[0], mode='C'+act_mode[1]))

        self.m_tail = B.conv(nc[0], out_nc, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1) + x0
        return x

'''
# ====================
# nonlocalunet
# ====================
'''

class NonLocalUNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64,128,256,512], nb=1, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(NonLocalUNet, self).__init__()

        down_nonlocal = B.NonLocalBlock2D(nc[2], kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='strideconv')
        up_nonlocal = B.NonLocalBlock2D(nc[2], kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='strideconv')

        self.m_head = B.conv(in_nc, nc[0], mode='C'+act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))


        self.m_down1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        self.m_down2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        self.m_down3 = B.sequential(down_nonlocal, *[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

        self.m_body  = B.sequential(*[B.conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))


        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], up_nonlocal)
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1) + x0
        return x

'''
if __name__ == '__main__':
    x = torch.rand(1,3,256,256)
#    net = UNet(act_mode='BR')
    net = NonLocalUNet()
    net.eval()
    with torch.no_grad():
        y = net(x)
    y.size()
'''
