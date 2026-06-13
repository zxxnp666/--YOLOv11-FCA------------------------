"""
DEA-Net: Single image dehazing based on detail-enhanced convolution 
and content-guided attention (IEEE TIP 2024)

论文: https://arxiv.org/abs/2301.04805
官方代码: https://github.com/cecret3350/DEA-Net
"""

import torch
import torch.nn as nn
from .deab import DEAB


class DEANet(nn.Module):
    """
    DEA-Net主模型
    
    Args:
        in_channels: 输入通道数，默认3（RGB）
        out_channels: 输出通道数，默认3（RGB）
        num_features: 特征通道数，默认64
        num_blocks: DEAB块数量，默认6
    """
    
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=6):
        super(DEANet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.num_blocks = num_blocks
        
        # Shallow feature extraction
        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True)
        
        # Detail-Enhanced Attention Blocks (DEAB)
        self.deab_blocks = nn.ModuleList([
            DEAB(num_features) for _ in range(num_blocks)
        ])
        
        # Reconstruction
        self.conv_out = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入雾天图像, shape [B, 3, H, W]
            
        Returns:
            dehazed: 去雾后的图像, shape [B, 3, H, W]
        """
        # Shallow feature extraction
        feat = self.conv_in(x)
        
        # Deep feature extraction with DEAB blocks
        for deab in self.deab_blocks:
            feat = deab(feat)
        
        # Reconstruction with residual learning
        out = self.conv_out(feat)
        dehazed = out + x  # Residual connection
        
        return dehazed
    
    def load_pretrained(self, checkpoint_path, device='cuda'):
        """
        加载预训练权重
        
        Args:
            checkpoint_path: 权重文件路径
            device: 设备（'cuda' 或 'cpu'）
        """
        print(f"Loading pretrained weights from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 处理不同的checkpoint格式
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # 加载权重
        self.load_state_dict(state_dict, strict=True)
        print("Pretrained weights loaded successfully!")
        
    def get_num_params(self):
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试代码
    model = DEANet()
    print(f"Model parameters: {model.get_num_params() / 1e6:.3f}M")
    
    # 测试前向传播
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
