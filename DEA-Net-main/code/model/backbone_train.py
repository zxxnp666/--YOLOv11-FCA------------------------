# DEA-Net去雾网络模型（训练版本）

import torch.nn as nn
import torch.nn.functional as F

from .modules import DEABlockTrain, DEBlockTrain, CGAFusion


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    # 默认卷积层，padding自动设置为kernel_size//2
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class DEANet(nn.Module):
    def __init__(self, base_dim=32):
        super(DEANet, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, base_dim, kernel_size=3, stride=1, padding=1)  # 3→32通道
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1),  # 32→64通道，分辨率减半
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),  # 64→128通道，分辨率再减半
            nn.ReLU(True)
        )
        # ==================== Level 1（原始分辨率）====================
        self.down_level1_block1 = DEBlockTrain(default_conv, base_dim, 3)  # 下采样路径DE Block 1
        self.down_level1_block2 = DEBlockTrain(default_conv, base_dim, 3)  # 下采样路径DE Block 2
        self.down_level1_block3 = DEBlockTrain(default_conv, base_dim, 3)  # 下采样路径DE Block 3
        self.down_level1_block4 = DEBlockTrain(default_conv, base_dim, 3)  # 下采样路径DE Block 4
        self.up_level1_block1 = DEBlockTrain(default_conv, base_dim, 3)    # 上采样路径DE Block 1
        self.up_level1_block2 = DEBlockTrain(default_conv, base_dim, 3)    # 上采样路径DE Block 2
        self.up_level1_block3 = DEBlockTrain(default_conv, base_dim, 3)    # 上采样路径DE Block 3
        self.up_level1_block4 = DEBlockTrain(default_conv, base_dim, 3)    # 上采样路径DE Block 4
        
        # ==================== Level 2（1/2分辨率）====================
        self.fe_level_2 = nn.Conv2d(
            in_channels=base_dim * 2, 
            out_channels=base_dim * 2, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )  # 特征提取卷积
        self.down_level2_block1 = DEBlockTrain(default_conv, base_dim * 2, 3)  # 下采样路径DE Block 1
        self.down_level2_block2 = DEBlockTrain(default_conv, base_dim * 2, 3)  # 下采样路径DE Block 2
        self.down_level2_block3 = DEBlockTrain(default_conv, base_dim * 2, 3)  # 下采样路径DE Block 3
        self.down_level2_block4 = DEBlockTrain(default_conv, base_dim * 2, 3)  # 下采样路径DE Block 4
        self.up_level2_block1 = DEBlockTrain(default_conv, base_dim * 2, 3)    # 上采样路径DE Block 1
        self.up_level2_block2 = DEBlockTrain(default_conv, base_dim * 2, 3)    # 上采样路径DE Block 2
        self.up_level2_block3 = DEBlockTrain(default_conv, base_dim * 2, 3)    # 上采样路径DE Block 3
        self.up_level2_block4 = DEBlockTrain(default_conv, base_dim * 2, 3)    # 上采样路径DE Block 4
        # ==================== Level 3（1/4分辨率，瓶颈层）====================
        self.fe_level_3 = nn.Conv2d(
            in_channels=base_dim * 4, 
            out_channels=base_dim * 4, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )  # 特征提取卷积
        self.level3_block1 = DEABlockTrain(default_conv, base_dim * 4, 3)  # DEA Block 1
        self.level3_block2 = DEABlockTrain(default_conv, base_dim * 4, 3)  # DEA Block 2
        self.level3_block3 = DEABlockTrain(default_conv, base_dim * 4, 3)  # DEA Block 3
        self.level3_block4 = DEABlockTrain(default_conv, base_dim * 4, 3)  # DEA Block 4
        self.level3_block5 = DEABlockTrain(default_conv, base_dim * 4, 3)  # DEA Block 5
        self.level3_block6 = DEABlockTrain(default_conv, base_dim * 4, 3)  # DEA Block 6
        self.level3_block7 = DEABlockTrain(default_conv, base_dim * 4, 3)  # DEA Block 7
        self.level3_block8 = DEABlockTrain(default_conv, base_dim * 4, 3)  # DEA Block 8
        
        # ==================== 解码器（上采样路径）====================
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, 
                             stride=2, padding=1, output_padding=1),  # 128→64通道，分辨率翻倍
            nn.ReLU(True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, 
                             stride=2, padding=1, output_padding=1),  # 64→32通道，分辨率再翻倍
            nn.ReLU(True)
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1)  # 32→3通道，输出RGB图像
        )
        
        # ==================== 特征融合模块 ====================
        self.mix1 = CGAFusion(base_dim * 4, reduction=8)  # Level 3特征融合（128通道）
        self.mix2 = CGAFusion(base_dim * 2, reduction=4)  # Level 2特征融合（64通道）

    def forward(self, x):
        # ==================== 编码器路径 ====================
        x_down1 = self.down1(x)  # Level 1: [B,3,H,W]→[B,32,H,W]
        x_down1 = self.down_level1_block1(x_down1)  # 细节增强
        x_down1 = self.down_level1_block2(x_down1)
        x_down1 = self.down_level1_block3(x_down1)
        x_down1 = self.down_level1_block4(x_down1)
        
        x_down2 = self.down2(x_down1)  # Level 2: [B,32,H,W]→[B,64,H/2,W/2]
        x_down2_init = self.fe_level_2(x_down2)  # 特征提取
        x_down2_init = self.down_level2_block1(x_down2_init)
        x_down2_init = self.down_level2_block2(x_down2_init)
        x_down2_init = self.down_level2_block3(x_down2_init)
        x_down2_init = self.down_level2_block4(x_down2_init)
        
        x_down3 = self.down3(x_down2_init)  # Level 3: [B,64,H/2,W/2]→[B,128,H/4,W/4]
        x_down3_init = self.fe_level_3(x_down3)  # 特征提取
        x1 = self.level3_block1(x_down3_init)  # DEA Block深度特征提取
        x2 = self.level3_block2(x1)
        x3 = self.level3_block3(x2)
        x4 = self.level3_block4(x3)
        x5 = self.level3_block5(x4)
        x6 = self.level3_block6(x5)
        x7 = self.level3_block7(x6)
        x8 = self.level3_block8(x7)
        x_level3_mix = self.mix1(x_down3, x8)  # CGA融合编码器特征和DEA输出
        
        # ==================== 解码器路径 ====================
        x_up1 = self.up1(x_level3_mix)  # Level 2: [B,128,H/4,W/4]→[B,64,H/2,W/2]
        x_up1 = self.up_level2_block1(x_up1)  # 特征重建
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)
        x_level2_mix = self.mix2(x_down2, x_up1)  # CGA融合编码器和解码器特征
        
        x_up2 = self.up2(x_level2_mix)  # Level 1: [B,64,H/2,W/2]→[B,32,H,W]
        x_up2 = self.up_level1_block1(x_up2)  # 特征重建
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)
        
        out = self.up3(x_up2)  # 输出层: [B,32,H,W]→[B,3,H,W]

        return out