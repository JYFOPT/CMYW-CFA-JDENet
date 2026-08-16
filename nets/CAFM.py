import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange


# Cross attention fusion module
class CAFM(nn.Module):
    def __init__(self, channel=112):
        super(CAFM, self).__init__()

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.avg1 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.channel = channel

        self.fusion = nn.Conv2d(channel * 2, channel, 1, 1, 0)

    def forward(self, f1, f2):
        b, c, h, w = f1.size()

        f1 = f1.reshape([b, c, -1])
        f2 = f2.reshape([b, c, -1])

        avg_1 = torch.mean(f1, dim=-1, keepdim=True).unsqueeze(-1)
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1

        avg_2 = torch.mean(f2, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2

        cross = torch.matmul(a1, a2.transpose(1, 2))

        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        a1 = a1.reshape([b, c, h, w])
        avg_out = torch.mean(a1, dim=1, keepdim=True)
        max_out, _ = torch.max(a1, dim=1, keepdim=True)
        a1 = torch.cat([avg_out, max_out], dim=1)
        a1 = F.relu(self.conv1_spatial(a1))
        a1 = self.conv2_spatial(a1)
        a1 = a1.reshape([b, 1, -1])
        a1 = F.softmax(a1, dim=-1)

        a2 = a2.reshape([b, c, h, w])
        avg_out = torch.mean(a2, dim=1, keepdim=True)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)
        a2 = torch.cat([avg_out, max_out], dim=1)
        a2 = F.relu(self.conv1_spatial(a2))
        a2 = self.conv2_spatial(a2)
        a2 = a2.reshape([b, 1, -1])
        a2 = F.softmax(a2, dim=-1)

        f1 = f1 * a1 + f1
        f2 = f2 * a2 + f2

        # 输入通道112对应256 * 28，实际应用自行修改
        f1 = rearrange(f1, 'b n (h d) -> b n h d', h=256 * 28 // self.channel, d=256 * 28 // self.channel)
        f2 = rearrange(f2, 'b n (h d) -> b n h d', h=256 * 28 // self.channel, d=256 * 28 // self.channel)

        out = self.fusion(torch.cat((f1, f2), dim=1))
        return out


if __name__ == "__main__":
    device = torch.device('cuda:0'if torch.cuda.is_available() else'cpu')

    x1 = torch.randn(1, 112, 64, 64).to(device)
    x2 = torch.randn(1, 112, 64, 64).to(device)
    model = CAFM(112).to(device)

    y = model(x1, x2)

    print("微信公众号：十小大的底层视觉工坊")
    print("VX: shixiaodayyds, 备注【即插即用】添加交流群")
    print("知乎、CSDN、小红书：十小大")

    print("输入分支1特征维度：", x1.shape)
    print("输入分支2特征维度：", x2.shape)
    print("输出特征维度：", y.shape)