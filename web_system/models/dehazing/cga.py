"""
CGA: Content-Guided Attention
为每个通道分配独特的空间重要性图（SIM）
"""

import torch
import torch.nn as nn


class ContentGuidedAttention(nn.Module):
    """
    Content-Guided Attention Module
    
    为每个特征通道生成独特的空间注意力图，
    使网络能够关注不同通道的不同空间位置
    
    Args:
        channels: 输入通道数
        reduction: 通道降维比例
    """
    
    def __init__(self, channels, reduction=16):
        super(ContentGuidedAttention, self).__init__()
        
        self.channels = channels
        self.reduction = reduction
        
        # 通道降维
        self.conv_squeeze = nn.Conv2d(channels, channels // reduction, 
                                     kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        
        # 生成空间重要性图（SIM）
        self.conv_spatial = nn.Conv2d(channels // reduction, channels,
                                     kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图, shape [B, C, H, W]
            
        Returns:
            out: 注意力加权后的特征图, shape [B, C, H, W]
        """
        batch_size, channels, height, width = x.size()
        
        # 通道降维
        squeeze = self.conv_squeeze(x)  # [B, C//r, H, W]
        squeeze = self.bn1(squeeze)
        squeeze = self.relu(squeeze)
        
        # 生成空间重要性图（SIM）
        # 每个通道都有自己独特的空间注意力分布
        spatial_attention = self.conv_spatial(squeeze)  # [B, C, H, W]
        spatial_attention = self.bn2(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)
        
        # 应用空间注意力
        out = x * spatial_attention
        
        return out
    
    def get_attention_map(self, x):
        """
        获取注意力图（用于可视化）
        
        Args:
            x: 输入特征图, shape [B, C, H, W]
            
        Returns:
            attention_map: 注意力图, shape [B, C, H, W]
        """
        with torch.no_grad():
            squeeze = self.conv_squeeze(x)
            squeeze = self.bn1(squeeze)
            squeeze = self.relu(squeeze)
            
            spatial_attention = self.conv_spatial(squeeze)
            spatial_attention = self.bn2(spatial_attention)
            spatial_attention = self.sigmoid(spatial_attention)
            
        return spatial_attention


class CGAv2(nn.Module):
    """
    CGA的增强版本
    使用深度可分离卷积减少参数量
    """
    
    def __init__(self, channels, reduction=16):
        super(CGAv2, self).__init__()
        
        mid_channels = channels // reduction
        
        # 逐点卷积降维
        self.pointwise1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # 深度卷积提取空间特征
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, 3, 
                                   padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 逐点卷积升维
        self.pointwise2 = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 降维
        out = self.pointwise1(x)
        out = self.bn1(out)
        
        # 空间特征提取
        out = self.depthwise(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # 升维生成注意力
        out = self.pointwise2(out)
        out = self.bn3(out)
        attention = self.sigmoid(out)
        
        # 应用注意力
        return x * attention


if __name__ == '__main__':
    # 测试代码
    print("Testing CGA...")
    
    # 创建CGA模块
    cga = ContentGuidedAttention(channels=64, reduction=16)
    x = torch.randn(2, 64, 32, 32)
    
    # 前向传播
    out = cga(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # 获取注意力图
    attention_map = cga.get_attention_map(x)
    print(f"Attention map shape: {attention_map.shape}")
    print(f"Attention map range: [{attention_map.min():.3f}, {attention_map.max():.3f}]")
    
    # 测试CGAv2
    print("\nTesting CGAv2...")
    cga_v2 = CGAv2(channels=64, reduction=16)
    out_v2 = cga_v2(x)
    print(f"CGAv2 output shape: {out_v2.shape}")
    
    # 参数量对比
    params_cga = sum(p.numel() for p in cga.parameters())
    params_cga_v2 = sum(p.numel() for p in cga_v2.parameters())
    print(f"\nCGA parameters: {params_cga}")
    print(f"CGAv2 parameters: {params_cga_v2}")
