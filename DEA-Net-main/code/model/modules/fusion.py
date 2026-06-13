from torch import nn

from .cga import SpatialAttention, ChannelAttention, PixelAttention


class CGAFusion(nn.Module):
    #内容引导注意力融合模块（CGA Fusion）- 用于融合不同层级的特征
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()  # 空间注意力
        self.ca = ChannelAttention(dim, reduction)  # 通道注意力
        self.pa = PixelAttention(dim)  # 像素注意力
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)  # 1x1卷积用于特征融合
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x, y):
        initial = x + y  # 初始特征相加
        cattn = self.ca(initial)  # 计算通道注意力
        sattn = self.sa(initial)  # 计算空间注意力
        pattn1 = sattn + cattn  # 融合空间和通道注意力
        pattn2 = self.sigmoid(self.pa(initial, pattn1))  # 计算像素级注意力权重
        result = initial + pattn2 * x + (1 - pattn2) * y  # 根据注意力权重融合特征
        result = self.conv(result)  # 1x1卷积进行特征整合
        return result