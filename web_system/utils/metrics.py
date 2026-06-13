"""
评估指标
包括去雾指标（PSNR、SSIM）和检测指标（mAP）
"""

import torch
import numpy as np
from math import log10
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1, img2, max_val=1.0):
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    
    Args:
        img1, img2: 图像tensor或numpy array
        max_val: 最大像素值
        
    Returns:
        psnr: PSNR值 (dB)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
        
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * log10(max_val / np.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2):
    """
    计算SSIM (Structural Similarity Index)
    
    Args:
        img1, img2: 图像numpy array [H, W, C]
        
    Returns:
        ssim: SSIM值 (0-1)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    # 确保数据范围在[0, 1]
    if img1.max() > 1:
        img1 = img1 / 255.0
    if img2.max() > 1:
        img2 = img2 / 255.0
    
    # 如果是单通道，添加通道维度
    if len(img1.shape) == 2:
        img1 = np.expand_dims(img1, axis=2)
    if len(img2.shape) == 2:
        img2 = np.expand_dims(img2, axis=2)
    
    ssim_val = ssim(img1, img2, channel_axis=2, data_range=1.0)
    return ssim_val


def calculate_mse(img1, img2):
    """计算MSE (Mean Squared Error)"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
        
    mse = np.mean((img1 - img2) ** 2)
    return mse


def calculate_mae(img1, img2):
    """计算MAE (Mean Absolute Error)"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
        
    mae = np.mean(np.abs(img1 - img2))
    return mae


def batch_psnr_ssim(dehazed_images, gt_images):
    """
    批量计算PSNR和SSIM
    
    Args:
        dehazed_images: 去雾图像列表
        gt_images: 真值图像列表
        
    Returns:
        avg_psnr, avg_ssim: 平均PSNR和SSIM
    """
    psnr_list = []
    ssim_list = []
    
    for dehazed, gt in zip(dehazed_images, gt_images):
        psnr = calculate_psnr(dehazed, gt)
        ssim_val = calculate_ssim(dehazed, gt)
        
        psnr_list.append(psnr)
        ssim_list.append(ssim_val)
    
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    
    return avg_psnr, avg_ssim


if __name__ == '__main__':
    # 测试代码
    print("Testing metrics...")
    
    # 创建测试图像
    img1 = np.random.rand(256, 256, 3)
    img2 = img1 + np.random.rand(256, 256, 3) * 0.1
    
    psnr = calculate_psnr(img1, img2)
    ssim_val = calculate_ssim(img1, img2)
    mse = calculate_mse(img1, img2)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"MSE: {mse:.6f}")
