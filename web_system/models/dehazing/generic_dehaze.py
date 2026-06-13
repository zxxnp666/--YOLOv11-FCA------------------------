"""
通用去雾模型包装器
支持多种去雾模型架构
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """基础卷积块"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
    
    def forward(self, x):
        return self.sa(x)


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(channels // reduction, 1)
        self.ca = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=True)
        )
    
    def forward(self, x):
        return self.ca(x)


class PixelAttention(nn.Module):
    """像素注意力模块"""
    def __init__(self, channels):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2, channels, kernel_size=7, padding=3, bias=True)
    
    def forward(self, x):
        return self.pa2(x)


class AttentionBlock(nn.Module):
    """带注意力机制的卷积块"""
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(channels, reduction=8)
        self.pa = PixelAttention(channels)
    
    def forward(self, x):
        identity = x
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x + identity


class MixModule(nn.Module):
    """Mix融合模块"""
    def __init__(self, channels, reduction=8):
        super(MixModule, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.pa = PixelAttention(channels)
        self.conv = nn.Conv2d(channels, channels, 1, bias=True)
    
    def forward(self, x):
        return self.conv(x)


class MPRNetLike(nn.Module):
    """
    类MPRNet架构的去雾模型
    兼容你的权重文件
    """
    def __init__(self):
        super(MPRNetLike, self).__init__()
        
        # Encoder - 修正通道数为32/64/128
        self.down1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1))
        self.down3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1))
        
        # Level 1 - 32通道
        self.down_level1_block1 = ConvBlock(32, 32)
        self.down_level1_block2 = ConvBlock(32, 32)
        self.down_level1_block3 = ConvBlock(32, 32)
        self.down_level1_block4 = ConvBlock(32, 32)
        
        self.up_level1_block1 = ConvBlock(32, 32)
        self.up_level1_block2 = ConvBlock(32, 32)
        self.up_level1_block3 = ConvBlock(32, 32)
        self.up_level1_block4 = ConvBlock(32, 32)
        
        # Level 2 - 64通道
        self.fe_level_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.down_level2_block1 = ConvBlock(64, 64)
        self.down_level2_block2 = ConvBlock(64, 64)
        self.down_level2_block3 = ConvBlock(64, 64)
        self.down_level2_block4 = ConvBlock(64, 64)
        
        self.up_level2_block1 = ConvBlock(64, 64)
        self.up_level2_block2 = ConvBlock(64, 64)
        self.up_level2_block3 = ConvBlock(64, 64)
        self.up_level2_block4 = ConvBlock(64, 64)
        
        # Level 3 - 128通道
        self.fe_level_3 = nn.Conv2d(128, 128, 3, padding=1)
        
        # Level 3 blocks with attention - 使用正确的命名
        self.level3_block1 = AttentionBlock(128)
        self.level3_block2 = AttentionBlock(128)
        self.level3_block3 = AttentionBlock(128)
        self.level3_block4 = AttentionBlock(128)
        self.level3_block5 = AttentionBlock(128)
        self.level3_block6 = AttentionBlock(128)
        self.level3_block7 = AttentionBlock(128)
        self.level3_block8 = AttentionBlock(128)
        
        # Decoder - 根据bias维度确定输出通道
        self.up1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1))
        self.up2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1))
        self.up3 = nn.Sequential(nn.Conv2d(32, 3, 3, padding=1))
        
        # Mix modules - mix2使用reduction=4
        self.mix1 = MixModule(128, reduction=8)
        self.mix2 = MixModule(64, reduction=4)
    
    def forward(self, x):
        """前向传播"""
        input_img = x  # 保存输入用于残差连接
        
        # Level 1 - 32通道
        x1 = self.down1(x)  # 3 -> 32
        x1 = self.down_level1_block1(x1)
        x1 = self.down_level1_block2(x1)
        
        # Level 2 - 64通道
        x2 = self.down2(x1)  # 32 -> 64
        x2_fe = self.fe_level_2(x2)  # 64 -> 64
        x2 = x2 + x2_fe
        x2 = self.down_level2_block1(x2)
        x2 = self.down_level2_block2(x2)
        
        # Level 3 - 128通道
        x3 = self.down3(x2)  # 64 -> 128
        x3_fe = self.fe_level_3(x3)  # 128 -> 128
        x3 = x3 + x3_fe
        
        # Level 3 blocks with attention
        x3 = self.level3_block1(x3)
        x3 = self.level3_block2(x3)
        x3 = self.level3_block3(x3)
        x3 = self.level3_block4(x3)
        x3 = self.level3_block5(x3)
        x3 = self.level3_block6(x3)
        x3 = self.level3_block7(x3)
        x3 = self.level3_block8(x3)
        
        # Decoder
        x = self.up1(x3)  # 128 -> 64
        x = x + x2  # skip connection
        x = self.up2(x)  # 64 -> 32
        x = x + x1  # skip connection
        out = self.up3(x)  # 32 -> 3
        
        # 残差学习：输出 = 输入 + 残差
        out = out + input_img
        
        return out
    
    def load_pretrained(self, weights_path, device='cpu'):
        """加载预训练权重 - 处理转置的权重"""
        print(f"Loading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 修复转置的权重 - up1和up2的权重需要转置
        if 'up1.0.weight' in state_dict:
            # up1: weight是[128,64,3,3]但bias是[64]，说明真实是Conv2d(128,64)
            # 需要转置为[64,128,3,3]
            w = state_dict['up1.0.weight']  # [128, 64, 3, 3]
            state_dict['up1.0.weight'] = w.permute(1, 0, 2, 3)  # [64, 128, 3, 3]
            print("  ✓ 转置 up1.0.weight: [128,64,3,3] -> [64,128,3,3]")
        
        if 'up2.0.weight' in state_dict:
            # up2: weight是[64,32,3,3]但bias是[32]，说明真实是Conv2d(64,32)
            # 需要转置为[32,64,3,3]
            w = state_dict['up2.0.weight']  # [64, 32, 3, 3]
            state_dict['up2.0.weight'] = w.permute(1, 0, 2, 3)  # [32, 64, 3, 3]
            print("  ✓ 转置 up2.0.weight: [64,32,3,3] -> [32,64,3,3]")
        
        # 手动加载权重
        model_dict = self.state_dict()
        loaded_keys = []
        skipped_keys = []
        
        for k, v in state_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    loaded_keys.append(k)
                else:
                    skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")
            else:
                skipped_keys.append(f"{k} (not in model)")
        
        self.load_state_dict(model_dict)

        loaded = len(loaded_keys)
        total = len(state_dict)
        
        print(f"✅ Loaded {loaded}/{total} keys")
        if skipped_keys:
            print(f"⚠️  Skipped {len(skipped_keys)} keys:")
            for k in skipped_keys[:5]:
                print(f"    {k}")
            if len(skipped_keys) > 5:
                print(f"    ... and {len(skipped_keys)-5} more")

        # 关键诊断信息：如果0个权重成功加载，明确给出严重警告
        if loaded == 0:
            print("❌ 严重警告：未能成功加载任何权重，请检查：")
            print("   - 权重文件是否为当前去雾网络的结构所训练得到；")
            print("   - 是否传入了错误的 checkpoint（例如检测模型权重当成去雾权重）；")
            print("   - 如为 DEA-Net 官方权重，请考虑使用专门的 DEA-Net 封装类加载。")
        elif loaded < total * 0.5:
            print("⚠️ 提示：仅加载了少部分权重，去雾效果可能明显弱于论文结果。")
        
        print("✅ Weights loaded完成，模型已切换到 eval 模式。")
        self.eval()

        # 返回加载统计，便于上层（如 Web 界面）进一步检查或记录
        return loaded, total
