import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from .fusion import AddFuseLayer
from .res_block import ResidualBlock
from .LLSKMs import LLSKM, LLSKM_d, LLSKM_1D



class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)


def _make_layer(block, block_num, in_channels, out_channels, stride):
    layer = [block(in_channels, out_channels, stride)]
    for _ in range(block_num - 1):
        layer.append(block(out_channels, out_channels, 1))
    return nn.Sequential(*layer)


def _fuse_layer(in_high_channels, in_low_channels, out_channels):
    fuse_layer = AddFuseLayer(in_high_channels, in_low_channels, out_channels)
    return fuse_layer


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    用转置卷积实现 2× 上采样
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            # 4×4, stride=2, pad=1  →  输出尺寸严格 2×
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                               stride=2, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class L2SKNet_FPN(nn.Module):
    """
        MyNetBasedFPN_4_all_d - Basic Implementation
        _4:layers_number=4
    """

    def __init__(self, layers=(2, 2, 2, 2), channels=(16, 32, 64, 128)):
        super(L2SKNet_FPN, self).__init__()

        self.layer0 = _make_layer(block=ResidualBlock, block_num=layers[0],
                                  in_channels=1, out_channels=channels[0], stride=1)
        self.layer1 = _make_layer(block=ResidualBlock, block_num=layers[1],
                                  in_channels=channels[0], out_channels=channels[1], stride=2)
        self.layer2 = _make_layer(block=ResidualBlock, block_num=layers[2],
                                  in_channels=channels[1], out_channels=channels[2], stride=2)
        self.layer3 = _make_layer(block=ResidualBlock, block_num=layers[3],
                                  in_channels=channels[2], out_channels=channels[3], stride=2)

        self.contrast0_0 = nn.Sequential(
            LLSKM_d(channels[0], kernel_size=9, padding=8, dilation=2),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast0_1 = nn.Sequential(
            LLSKM_d(channels[0], kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast0_2 = nn.Sequential(
            LLSKM_d(channels[0], kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast0_3 = nn.Sequential(
            LLSKM(channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast1_0 = nn.Sequential(
            LLSKM_d(channels[1], kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast1_1 = nn.Sequential(
            LLSKM_d(channels[1], kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast1_2 = nn.Sequential(
            LLSKM(channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast2_0 = nn.Sequential(
            LLSKM_d(channels[2], kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast2_1 = nn.Sequential(
            LLSKM(channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast3 = nn.Sequential(
            LLSKM(channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrConV_0 = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )
        self.contrConV_1 = nn.Sequential(
            nn.Conv2d(channels[1], channels[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU()
        )
        self.contrConV_2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU()
        )

        self.fuse32 = _fuse_layer(channels[3], channels[2], channels[2])
        self.fuse21 = _fuse_layer(channels[2], channels[1], channels[1])
        self.fuse10 = _fuse_layer(channels[1], channels[0], channels[0])

        self.head = _FCNHead(channels[0], 1)

    def forward(self, x):
        _, _, hei, wid = x.shape

        c0 = self.layer0(x)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)

        _, _, c0_hei, c0_wid = c0.shape
        _, _, c1_hei, c1_wid = c1.shape
        _, _, c2_hei, c2_wid = c2.shape

        c0_0 = self.contrast0_0(c0)
        c0_1 = self.contrast0_1(c0)
        c0_2 = self.contrast0_2(c0)
        c0_3 = self.contrast0_3(c0)
        c0_all = c0_0 + c0_1 + c0_2 + c0_3
        c0 = self.contrConV_0(c0_all)

        c1_0 = self.contrast1_0(c1)
        c1_1 = self.contrast1_1(c1)
        c1_2 = self.contrast1_2(c1)
        c1_all = c1_0 + c1_1 + c1_2
        c1 = self.contrConV_1(c1_all)

        c2_0 = self.contrast2_0(c2)
        c2_1 = self.contrast2_1(c2)
        c2_all = c2_0 + c2_1
        c2 = self.contrConV_2(c2_all)

        c3 = self.contrast3(c3)

        out = F.interpolate(c3, size=[c2_hei, c2_wid], mode='bilinear')
        out = self.fuse32(out, c2)
        out = F.interpolate(out, size=[c1_hei, c1_wid], mode='bilinear')
        out = self.fuse21(out, c1)
        out = F.interpolate(out, size=[c0_hei, c0_wid], mode='bilinear')
        out = self.fuse10(out, c0)

        out = self.head(out)

        return out.sigmoid()


class L2SKNet_1D_FPN(nn.Module):
    """
        MyNetBasedFPN_4_all_lite - Basic Implementation
        _4:layers_number=4
    """

    def __init__(self, layers=(2, 2, 2, 2), channels=(16, 32, 64, 128)):
        super(L2SKNet_1D_FPN, self).__init__()

        self.layer0 = _make_layer(block=ResidualBlock, block_num=layers[0],
                                  in_channels=1, out_channels=channels[0], stride=1)
        self.layer1 = _make_layer(block=ResidualBlock, block_num=layers[1],
                                  in_channels=channels[0], out_channels=channels[1], stride=2)
        self.layer2 = _make_layer(block=ResidualBlock, block_num=layers[2],
                                  in_channels=channels[1], out_channels=channels[2], stride=2)
        self.layer3 = _make_layer(block=ResidualBlock, block_num=layers[3],
                                  in_channels=channels[2], out_channels=channels[3], stride=2)

        self.contrast0_0 = nn.Sequential(
            LLSKM_1D(channels[0], kernel_size=17, padding=8),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast0_1 = nn.Sequential(
            LLSKM_1D(channels[0], kernel_size=9, padding=4),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast0_2 = nn.Sequential(
            LLSKM_1D(channels[0], kernel_size=5, padding=2),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast0_3 = nn.Sequential(
            LLSKM_1D(channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast1_0 = nn.Sequential(
            LLSKM_1D(channels[1], kernel_size=9, padding=4),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast1_1 = nn.Sequential(
            LLSKM_1D(channels[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast1_2 = nn.Sequential(
            LLSKM_1D(channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast2_0 = nn.Sequential(
            LLSKM_1D(channels[2], kernel_size=5, padding=2),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast2_1 = nn.Sequential(
            LLSKM_1D(channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast3 = nn.Sequential(
            LLSKM_1D(channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrConV_0 = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )
        self.contrConV_1 = nn.Sequential(
            nn.Conv2d(channels[1], channels[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU()
        )
        self.contrConV_2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU()
        )

        self.fuse32 = _fuse_layer(channels[3], channels[2], channels[2])
        self.fuse21 = _fuse_layer(channels[2], channels[1], channels[1])
        self.fuse10 = _fuse_layer(channels[1], channels[0], channels[0])

        self.head = _FCNHead(channels[0], 1)

    def forward(self, x):
        _, _, hei, wid = x.shape

        c0 = self.layer0(x)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)

        _, _, c0_hei, c0_wid = c0.shape
        _, _, c1_hei, c1_wid = c1.shape
        _, _, c2_hei, c2_wid = c2.shape

        c0_0 = self.contrast0_0(c0)
        c0_1 = self.contrast0_1(c0)
        c0_2 = self.contrast0_2(c0)
        c0_3 = self.contrast0_3(c0)
        c0_all = c0_0 + c0_1 + c0_2 + c0_3
        c0 = self.contrConV_0(c0_all)

        c1_0 = self.contrast1_0(c1)
        c1_1 = self.contrast1_1(c1)
        c1_2 = self.contrast1_2(c1)
        c1_all = c1_0 + c1_1 + c1_2
        c1 = self.contrConV_1(c1_all)

        c2_0 = self.contrast2_0(c2)
        c2_1 = self.contrast2_1(c2)
        c2_all = c2_0 + c2_1
        c2 = self.contrConV_2(c2_all)

        c3 = self.contrast3(c3)

        out = F.interpolate(c3, size=[c2_hei, c2_wid], mode='bilinear')
        out = self.fuse32(out, c2)
        out = F.interpolate(out, size=[c1_hei, c1_wid], mode='bilinear')
        out = self.fuse21(out, c1)
        out = F.interpolate(out, size=[c0_hei, c0_wid], mode='bilinear')
        out = self.fuse10(out, c0)

        out = self.head(out)

        return out.sigmoid()


class L2SKNet_UNet(nn.Module):
    """
        MyNetBasedUnet_4 - Basic Implementation
        _4:layers_number=4
    """

    def __init__(self, in_ch=1, out_ch=1):
        super(L2SKNet_UNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.contrast0_0 = nn.Sequential(
            LLSKM_d(filters[0], kernel_size=9, padding=8, dilation=2),
            nn.BatchNorm2d(filters[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast0_1 = nn.Sequential(
            LLSKM_d(filters[0], kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm2d(filters[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast0_2 = nn.Sequential(
            LLSKM_d(filters[0], kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(filters[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast0_3 = nn.Sequential(
            LLSKM(filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast1_0 = nn.Sequential(
            LLSKM_d(filters[1], kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm2d(filters[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast1_1 = nn.Sequential(
            LLSKM_d(filters[1], kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(filters[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast1_2 = nn.Sequential(
            LLSKM(filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast2_0 = nn.Sequential(
            LLSKM_d(filters[2], kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(filters[2]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast2_1 = nn.Sequential(
            LLSKM(filters[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast3 = nn.Sequential(
            LLSKM(filters[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrConV_0 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU()
        )
        self.contrConV_1 = nn.Sequential(
            nn.Conv2d(filters[1], filters[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU()
        )
        self.contrConV_2 = nn.Sequential(
            nn.Conv2d(filters[2], filters[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU()
        )

        # self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 =  nn.Conv2d(in_ch, filters[0], kernel_size=3, stride=2, padding=1, bias=True)

        self.Conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=2, padding=1, bias=True)
        self.Conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=2, padding=1, bias=True)
        self.Conv4 = nn.Conv2d(filters[2], filters[3], kernel_size=3, stride=2, padding=1, bias=True)

        self.Up4 = up_conv(filters[3], filters[2])
        # self.Up_conv4 = conv_block(filters[3], filters[2])
        self.Up3 = up_conv(filters[2]*2, filters[1])
        # self.Up_conv3 = conv_block(filters[2], filters[1])
        self.Up2 = up_conv(filters[1]*2, filters[0])
        # self.Up_conv2 = conv_block(filters[1], filters[0])
        self.Up1 = up_conv(filters[0] * 2, filters[0])
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1)

        self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Conv2(e1)
        e3 = self.Conv3(e2)
        e4 = self.Conv4(e3)

        c0_0 = self.contrast0_3(e1)
        c0_1 = self.contrast0_3(e1)
        c0_2 = self.contrast0_3(e1)
        c0_3 = self.contrast0_3(e1)
        c0_all = c0_0 + c0_1 + c0_2 + c0_3
        e1 = self.contrConV_0(c0_all)

        c1_0 = self.contrast1_2(e2)
        c1_1 = self.contrast1_2(e2)
        c1_2 = self.contrast1_2(e2)
        c1_all = c1_0 + c1_1 + c1_2
        e2 = self.contrConV_1(c1_all)

        c2_0 = self.contrast2_1(e3)
        c2_1 = self.contrast2_1(e3)
        c2_all = c2_0 + c2_1
        e3 = self.contrConV_2(c2_all)

        e4 = self.contrast3(e4)

        d4 = self.Up4(e4)
        d4 = torch.cat((e3, d4), dim=1)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d1 = self.Up1(d2)
        out = self.Conv(d1)
        out = self.active(out)

        return out


class L2SKNet_1D_UNet(nn.Module):
    """
        MyNetBasedUnet_4_all_lite - Basic Implementation
        _4:layers_number=4
    """

    def __init__(self, in_ch=1, out_ch=1):
        super(L2SKNet_1D_UNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.contrast0_0 = nn.Sequential(
            LLSKM_1D(filters[0], kernel_size=17, padding=8),
            nn.BatchNorm2d(filters[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast0_1 = nn.Sequential(
            LLSKM_1D(filters[0], kernel_size=9, padding=4),
            nn.BatchNorm2d(filters[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast0_2 = nn.Sequential(
            LLSKM_1D(filters[0], kernel_size=5, padding=2),
            nn.BatchNorm2d(filters[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast0_3 = nn.Sequential(
            LLSKM_1D(filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast1_0 = nn.Sequential(
            LLSKM_1D(filters[1], kernel_size=9, padding=4),
            nn.BatchNorm2d(filters[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast1_1 = nn.Sequential(
            LLSKM_1D(filters[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(filters[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast1_2 = nn.Sequential(
            LLSKM_1D(filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast2_0 = nn.Sequential(
            LLSKM_1D(filters[2], kernel_size=5, padding=2),
            nn.BatchNorm2d(filters[2]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast2_1 = nn.Sequential(
            LLSKM_1D(filters[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrast3 = nn.Sequential(
            LLSKM_1D(filters[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.contrConV_0 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU()
        )
        self.contrConV_1 = nn.Sequential(
            nn.Conv2d(filters[1], filters[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU()
        )
        self.contrConV_2 = nn.Sequential(
            nn.Conv2d(filters[2], filters[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU()
        )

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])
        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        c0_0 = self.contrast0_0(e1)
        c0_1 = self.contrast0_1(e1)
        c0_2 = self.contrast0_2(e1)
        c0_3 = self.contrast0_3(e1)
        c0_all = c0_0 + c0_1 + c0_2 + c0_3
        e1 = self.contrConV_0(c0_all)

        c1_0 = self.contrast1_0(e2)
        c1_1 = self.contrast1_1(e2)
        c1_2 = self.contrast1_2(e2)
        c1_all = c1_0 + c1_1 + c1_2
        e2 = self.contrConV_1(c1_all)

        c2_0 = self.contrast2_0(e3)
        c2_1 = self.contrast2_1(e3)
        c2_all = c2_0 + c2_1
        e3 = self.contrConV_2(c2_all)

        e4 = self.contrast3(e4)

        d4 = self.Up4(e4)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        out = self.Conv(d2)

        out = self.active(out)  # 1,256,256

        return out
