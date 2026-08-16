import torch.nn as nn


class AddFuseLayer(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels, r=4):
        super(AddFuseLayer, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),  # bz,ch,h,w -> bz,cl,h,w
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)  # bz,ch,h,w -> bz,cl,h,w

        out = xh + xl

        return out
