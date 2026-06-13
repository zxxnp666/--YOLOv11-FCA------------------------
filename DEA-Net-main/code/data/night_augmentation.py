"""
夜晚场景数据增强
模拟夜晚特有的光照条件和干扰
"""

import torch
import torch.nn.functional as F
import numpy as np
import random


class NightAugmentation:
    """
    夜晚场景数据增强
    """
    def __init__(self, 
                 brightness_range=(0.7, 1.3),
                 add_light_source=True,
                 add_noise=True,
                 color_shift=True):
        self.brightness_range = brightness_range
        self.add_light_source = add_light_source
        self.add_noise = add_noise
        self.color_shift = color_shift
    
    def __call__(self, hazy, clear):
        """
        Args:
            hazy: 雾图 [C, H, W] tensor
            clear: 清晰图 [C, H, W] tensor
        Returns:
            增强后的 (hazy, clear)
        """
        # 1. 亮度调整（模拟不同光照条件）
        if random.random() > 0.5:
            hazy, clear = self.adjust_brightness(hazy, clear)
        
        # 2. 添加光源（模拟车灯、路灯）
        if self.add_light_source and random.random() > 0.7:
            hazy = self.add_light_sources(hazy)
        
        # 3. 添加噪声（模拟低光噪声）
        if self.add_noise and random.random() > 0.5:
            hazy = self.add_low_light_noise(hazy)
        
        # 4. 颜色偏移（模拟不同色温光源）
        if self.color_shift and random.random() > 0.6:
            hazy = self.apply_color_shift(hazy)
        
        return hazy, clear
    
    def adjust_brightness(self, hazy, clear):
        """调整整体亮度"""
        factor = random.uniform(*self.brightness_range)
        hazy = torch.clamp(hazy * factor, 0, 1)
        clear = torch.clamp(clear * factor, 0, 1)
        return hazy, clear
    
    def add_light_sources(self, img):
        """
        添加光源效果（模拟车灯、路灯）
        在随机位置添加高斯光晕
        """
        C, H, W = img.shape
        
        # 随机1-3个光源
        num_lights = random.randint(1, 3)
        
        for _ in range(num_lights):
            # 随机位置
            cx = random.randint(int(W*0.2), int(W*0.8))
            cy = random.randint(int(H*0.2), int(H*0.8))
            
            # 随机大小和强度
            sigma = random.uniform(20, 50)
            intensity = random.uniform(0.3, 0.6)
            
            # 创建高斯光晕
            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            gaussian = torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
            gaussian = gaussian.unsqueeze(0).repeat(C, 1, 1)
            
            # 添加到图像
            img = torch.clamp(img + intensity * gaussian, 0, 1)
        
        return img
    
    def add_low_light_noise(self, img):
        """
        添加低光噪声
        低光条件下噪声更明显
        """
        # 计算亮度
        brightness = img.mean(dim=0, keepdim=True)
        
        # 噪声强度与亮度成反比
        noise_level = 0.02 * (1.0 - brightness)
        
        # 添加高斯噪声
        noise = torch.randn_like(img) * noise_level
        img = torch.clamp(img + noise, 0, 1)
        
        return img
    
    def apply_color_shift(self, img):
        """
        应用颜色偏移
        模拟不同色温的光源（偏黄/偏蓝）
        """
        # 随机选择色温
        if random.random() > 0.5:
            # 暖色调（偏黄）- 钠灯
            img[0] *= random.uniform(1.0, 1.15)  # R
            img[1] *= random.uniform(1.0, 1.1)   # G
            img[2] *= random.uniform(0.9, 1.0)   # B
        else:
            # 冷色调（偏蓝）- LED灯
            img[0] *= random.uniform(0.95, 1.0)  # R
            img[1] *= random.uniform(0.95, 1.0)  # G
            img[2] *= random.uniform(1.0, 1.1)   # B
        
        img = torch.clamp(img, 0, 1)
        return img


class NightFogSynthesis:
    """
    夜晚雾效果合成
    考虑光源散射效果
    """
    def __init__(self, beta_range=(0.8, 1.5)):
        self.beta_range = beta_range
    
    def __call__(self, clear_img):
        """
        为清晰图像添加夜晚雾效果
        
        Args:
            clear_img: [C, H, W] tensor
        Returns:
            hazy_img: [C, H, W] tensor
        """
        C, H, W = clear_img.shape
        
        # 随机雾浓度
        beta = random.uniform(*self.beta_range)
        
        # 大气光（夜晚通常较暗）
        A = torch.tensor([0.6, 0.6, 0.65])  # 略偏蓝
        A = A.view(3, 1, 1)
        
        # 深度图（简单线性）
        depth = torch.linspace(0, 1, H).view(1, H, 1).repeat(1, 1, W)
        
        # 添加随机扰动
        depth = depth + torch.randn(1, H, W) * 0.1
        depth = torch.clamp(depth, 0, 1)
        
        # 透射率
        t = torch.exp(-beta * depth)
        t = torch.clamp(t, 0.1, 1.0)
        t = t.repeat(C, 1, 1)
        
        # 大气散射模型
        hazy = clear_img * t + A * (1 - t)
        hazy = torch.clamp(hazy, 0, 1)
        
        # 添加光源散射（雾中光晕更明显）
        if random.random() > 0.5:
            hazy = self.add_light_scattering(hazy, t)
        
        return hazy
    
    def add_light_scattering(self, hazy, transmission):
        """
        添加光源散射效果
        雾越浓，光晕越明显
        """
        C, H, W = hazy.shape
        
        # 检测亮区域（光源）
        brightness = hazy.mean(dim=0)
        light_mask = (brightness > 0.7).float()
        
        if light_mask.sum() > 0:
            # 对亮区域进行高斯模糊（模拟散射）
            kernel_size = 15
            sigma = 5.0
            
            # 创建高斯核
            kernel = self.gaussian_kernel(kernel_size, sigma)
            kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)
            
            # 应用模糊
            padding = kernel_size // 2
            scattered = F.conv2d(
                hazy.unsqueeze(0), 
                kernel, 
                padding=padding, 
                groups=C
            ).squeeze(0)
            
            # 根据透射率混合（雾越浓，散射越明显）
            scatter_weight = (1 - transmission) * 0.3
            hazy = hazy * (1 - scatter_weight) + scattered * scatter_weight
            hazy = torch.clamp(hazy, 0, 1)
        
        return hazy
    
    @staticmethod
    def gaussian_kernel(kernel_size, sigma):
        """生成高斯核"""
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel / kernel.sum()
        return kernel


# 便捷函数
def get_night_augmentation(mode='full'):
    """
    获取夜晚数据增强
    
    Args:
        mode: 'full', 'light', 'none'
    """
    if mode == 'full':
        return NightAugmentation(
            add_light_source=True,
            add_noise=True,
            color_shift=True
        )
    elif mode == 'light':
        return NightAugmentation(
            add_light_source=False,
            add_noise=True,
            color_shift=True
        )
    elif mode == 'none':
        return None
    else:
        raise ValueError(f"Unknown mode: {mode}")
