"""
FCA: Fast Channel Attention
快速通道注意力机制
"""

import torch
import torch.nn as nn


class FCAModule(nn.Module):
    """
    Fast Channel Attention Module
    
    通过全局平均池化和最大池化捕获通道间的关系，
    为每个通道分配权重，增强关键特征
    
    Args:
        channels: 输入通道数
        reduction: 通道降维比例，默认16
    """
    
    def __init__(self, channels, reduction=16):
        super(FCAModule, self).__init__()
        
        self.channels = channels
        self.reduction = reduction
        
        # 全局平均池化和最大池化
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
        """
        前向传播
        
        Args:
            x: 输入特征图, shape [B, C, H, W]
            
        Returns:
            out: 注意力加权后的特征图, shape [B, C, H, W]
        """
        # 全局平均池化分支
        avg_out = self.fc(self.avg_pool(x))
        
        # 全局最大池化分支
        max_out = self.fc(self.max_pool(x))
        
        # 融合两个分支
        attention = self.sigmoid(avg_out + max_out)
        
        # 应用通道注意力
        out = x * attention
        
        return out


class FCAv2(nn.Module):
    """
    FCA的改进版本
    加入空间注意力，形成通道-空间混合注意力
    """
    
    def __init__(self, channels, reduction=16):
        super(FCAv2, self).__init__()
        
        # 通道注意力
        self.channel_attention = FCAModule(channels, reduction)
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        out = self.channel_attention(x)
        
        # 空间注意力
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.spatial_attention(spatial_input)
        
        out = out * spatial_attention
        
        return out


class ECAModule(nn.Module):
    """
    ECA: Efficient Channel Attention
    更高效的通道注意力实现
    """
    
    def __init__(self, channels, k_size=3):
        super(ECAModule, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 全局平均池化
        y = self.avg_pool(x)  # [B, C, 1, 1]
        
        # 1D卷积
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 激活
        attention = self.sigmoid(y)
        
        return x * attention.expand_as(x)


if __name__ == '__main__':
    # 测试代码
    print("Testing FCA Module...")
    
    # 测试FCA
    fca = FCAModule(channels=256, reduction=16)
    x = torch.randn(2, 256, 32, 32)
    out = fca(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    params = sum(p.numel() for p in fca.parameters())
    print(f"FCA parameters: {params}")
    
    # 测试FCAv2
    print("\nTesting FCAv2...")
    fca_v2 = FCAv2(channels=256, reduction=16)
    out_v2 = fca_v2(x)
    print(f"FCAv2 output shape: {out_v2.shape}")
    
    params_v2 = sum(p.numel() for p in fca_v2.parameters())
    print(f"FCAv2 parameters: {params_v2}")
    
    # 测试ECA
    print("\nTesting ECA...")
    eca = ECAModule(channels=256, k_size=3)
    out_eca = eca(x)
    print(f"ECA output shape: {out_eca.shape}")
    
    params_eca = sum(p.numel() for p in eca.parameters())
    print(f"ECA parameters: {params_eca}")
