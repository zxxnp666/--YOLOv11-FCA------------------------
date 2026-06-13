from torch import nn

from .deconv import DEConv
from .cga import SpatialAttention, ChannelAttention, PixelAttention


class DEABlockTrain(nn.Module):
    #细节增强注意力模块（DEA Block）
    def __init__(self, conv, dim, kernel_size, reduction=8):
        super(DEABlockTrain, self).__init__()
        self.conv1 = DEConv(dim)  # 细节增强卷积层
        self.act1 = nn.ReLU(inplace=True)  # 激活函数
        self.conv2 = conv(dim, dim, kernel_size, bias=True)  # 标准卷积层
        self.sa = SpatialAttention()  # 空间注意力
        self.ca = ChannelAttention(dim, reduction)  # 通道注意力
        self.pa = PixelAttention(dim)  # 像素注意力
    def forward(self, x):
        res = self.conv1(x)  # 细节增强卷积
        res = self.act1(res)  # 激活
        res = res + x  # 残差连接
        res = self.conv2(res)  # 标准卷积
        cattn = self.ca(res)  # 计算通道注意力
        sattn = self.sa(res)  # 计算空间注意力
        pattn1 = sattn + cattn  # 融合空间和通道注意力
        pattn2 = self.pa(res, pattn1)  # 计算像素级注意力
        res = res * pattn2  # 应用注意力权重
        res = res + x  # 残差连接
        return res
class DEBlockTrain(nn.Module):
    #细节增强模块（DE Block）不含注意力机制的简化版本
    def __init__(self, conv, dim, kernel_size):
        super(DEBlockTrain, self).__init__()
        self.conv1 = DEConv(dim)  # 细节增强卷积层
        self.act1 = nn.ReLU(inplace=True)  # 激活函数
        self.conv2 = conv(dim, dim, kernel_size, bias=True)  # 标准卷积层
    def forward(self, x):
        res = self.conv1(x)  # 细节增强卷积
        res = self.act1(res)  # 激活
        res = res + x  # 残差连接
        res = self.conv2(res)  # 标准卷积
        res = res + x  # 残差连接
        return res