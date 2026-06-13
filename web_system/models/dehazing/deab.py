"""
DEAB: Detail-Enhanced Attention Block
DEConv + CGA 的组合模块
"""

import torch
import torch.nn as nn
from .deconv import DEConv
from .cga import ContentGuidedAttention


class DEAB(nn.Module):
    """
    Detail-Enhanced Attention Block
    
    组合DEConv和CGA，构成DEA-Net的基本构建块
    
    Args:
        channels: 特征通道数
        reduction: CGA的通道降维比例
        use_deconv: 是否使用DEConv（False则使用标准卷积）
        deploy: 是否为部署模式
    """
    
    def __init__(self, channels, reduction=16, use_deconv=True, deploy=False):
        super(DEAB, self).__init__()
        
        self.channels = channels
        self.use_deconv = use_deconv
        
        # 第一个卷积层（DEConv或标准卷积）
        if use_deconv:
            self.conv1 = DEConv(channels, channels, kernel_size=3, 
                               stride=1, padding=1, deploy=deploy)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels)
            )
        
        self.relu1 = nn.ReLU(inplace=True)
        
        # Content-Guided Attention
        self.cga = ContentGuidedAttention(channels, reduction)
        
        # 第二个卷积层
        if use_deconv:
            self.conv2 = DEConv(channels, channels, kernel_size=3,
                               stride=1, padding=1, deploy=deploy)
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels)
            )
        
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图, shape [B, C, H, W]
            
        Returns:
            out: 输出特征图, shape [B, C, H, W]
        """
        identity = x
        
        # 第一个DEConv + ReLU
        out = self.conv1(x)
        out = self.relu1(out)
        
        # Content-Guided Attention
        out = self.cga(out)
        
        # 第二个DEConv + ReLU
        out = self.conv2(out)
        out = self.relu2(out)
        
        # 残差连接
        out = out + identity
        
        return out
    
    def switch_to_deploy(self):
        """切换到部署模式"""
        if self.use_deconv:
            self.conv1.switch_to_deploy()
            self.conv2.switch_to_deploy()


class DEABv2(nn.Module):
    """
    DEAB的轻量级版本
    使用更少的参数，适合资源受限的场景
    """
    
    def __init__(self, channels, reduction=16):
        super(DEABv2, self).__init__()
        
        mid_channels = channels // 2
        
        # 瓶颈结构
        self.conv1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, 
                              groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.cga = ContentGuidedAttention(mid_channels, reduction)
        
        self.conv3 = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.cga(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


if __name__ == '__main__':
    # 测试代码
    print("Testing DEAB...")
    
    # 创建DEAB模块
    deab = DEAB(channels=64, reduction=16, use_deconv=True, deploy=False)
    x = torch.randn(2, 64, 32, 32)
    
    # 前向传播
    out = deab(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # 计算参数量
    params = sum(p.numel() for p in deab.parameters())
    print(f"Parameters: {params / 1e3:.2f}K")
    
    # 测试部署模式
    print("\nSwitching to deploy mode...")
    deab.switch_to_deploy()
    out_deploy = deab(x)
    print(f"Deploy mode output shape: {out_deploy.shape}")
    
    # 测试轻量级版本
    print("\nTesting DEABv2...")
    deab_v2 = DEABv2(channels=64, reduction=16)
    out_v2 = deab_v2(x)
    print(f"DEABv2 output shape: {out_v2.shape}")
    
    params_v2 = sum(p.numel() for p in deab_v2.parameters())
    print(f"DEABv2 parameters: {params_v2 / 1e3:.2f}K")
    print(f"Parameter reduction: {(1 - params_v2/params) * 100:.1f}%")
