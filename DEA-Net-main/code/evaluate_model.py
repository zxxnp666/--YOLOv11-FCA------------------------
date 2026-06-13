"""
简单的模型评估脚本
用于评估训练好的去雾模型
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 导入项目的模块
import sys
sys.path.append('code')
from model import DEANet
from data.data_loader import TestDataset
from metric import psnr, ssim


def pad_img(x, patch_size=4):
    """填充图像使其能被patch_size整除"""
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def evaluate_model(model_path, dataset_path, device='cuda', max_images=0):
    """
    评估模型
    
    Args:
        model_path: 模型文件路径 (如 'epoch_10.pk' 或 'best.pk')
        dataset_path: 数据集根目录 (如 'dataset/ITS')
        device: 'cuda' 或 'cpu'
        max_images: 最多评估多少张图片，0表示全部
    """
    print("="*80)
    print("Model Evaluation".center(80))
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    # 检查设备
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU instead")
        device = 'cpu'
    
    # 加载模型
    print("📦 Loading model...")
    net = DEANet(base_dim=32)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"✅ Model loaded from checkpoint (Epoch: {checkpoint.get('epoch', 'unknown')})")
        else:
            state_dict = checkpoint
            print(f"✅ Model loaded")
        
        # 处理DataParallel保存的模型（去掉'module.'前缀）
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            print("   Removed 'module.' prefix from state_dict")
        
        net.load_state_dict(state_dict)
    else:
        print(f"❌ Model file not found: {model_path}")
        return None, None
    
    net = net.to(device)
    net.eval()
    
    # 加载测试数据
    print("\n📂 Loading test dataset...")
    # 尝试test目录，如果不存在则尝试val目录（OTS数据集）
    test_hazy_dir = os.path.join(dataset_path, 'test', 'hazy')
    test_clear_dir = os.path.join(dataset_path, 'test', 'clear')
    
    if not os.path.exists(test_hazy_dir):
        # 尝试val目录
        test_hazy_dir = os.path.join(dataset_path, 'val', 'hazy')
        test_clear_dir = os.path.join(dataset_path, 'val', 'clear')
    
    if not os.path.exists(test_hazy_dir) or not os.path.exists(test_clear_dir):
        print(f"❌ Test dataset not found at: {dataset_path}/test/ or {dataset_path}/val/")
        return None, None
    
    test_set = TestDataset(test_hazy_dir, test_clear_dir)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
    
    total_images = len(test_loader)
    if max_images > 0:
        eval_images = min(max_images, total_images)
    else:
        eval_images = total_images
    
    print(f"✅ Test dataset loaded: {eval_images} images (total: {total_images})")
    
    # 开始评估
    print("\n🔍 Starting evaluation...")
    ssims = []
    psnrs = []
    
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for i, (inputs, targets, hazy_name) in enumerate(tqdm(test_loader, desc="Evaluating", total=eval_images)):
            if i >= eval_images:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 预测
            H, W = inputs.shape[2:]
            inputs_padded = pad_img(inputs, 4)
            pred = net(inputs_padded).clamp(0, 1)
            pred = pred[:, :, :H, :W]
            
            # 计算指标
            ssim_val = ssim(pred, targets).item()
            psnr_val = psnr(pred, targets)
            
            ssims.append(ssim_val)
            psnrs.append(psnr_val)
    
    # 计算平均值
    avg_psnr = np.mean(psnrs)
    avg_ssim = np.mean(ssims)
    
    # 打印结果
    print("\n" + "="*80)
    print("Evaluation Results".center(80))
    print("="*80)
    print(f"Images Evaluated: {eval_images}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"PSNR Range: [{min(psnrs):.4f}, {max(psnrs):.4f}]")
    print(f"SSIM Range: [{min(ssims):.4f}, {max(ssims):.4f}]")
    print("="*80 + "\n")
    
    # 保存评估结果
    save_dir = os.path.dirname(model_path)
    if save_dir:
        # 保存详细数据
        np.save(os.path.join(save_dir, 'eval_psnrs.npy'), psnrs)
        np.save(os.path.join(save_dir, 'eval_ssims.npy'), ssims)
        
        # 保存文本报告
        report_path = os.path.join(save_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Model Evaluation Report\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model File: {model_path}\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Evaluation Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("="*80 + "\n")
            f.write("Results\n")
            f.write("="*80 + "\n")
            f.write(f"Images Evaluated: {eval_images} / {total_images}\n")
            f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"PSNR Range: [{min(psnrs):.4f}, {max(psnrs):.4f}]\n")
            f.write(f"SSIM Range: [{min(ssims):.4f}, {max(ssims):.4f}]\n")
            f.write(f"PSNR Std: {np.std(psnrs):.4f}\n")
            f.write(f"SSIM Std: {np.std(ssims):.4f}\n")
            f.write("="*80 + "\n")
        
        print(f"💾 Results saved to:")
        print(f"   - {os.path.join(save_dir, 'eval_psnrs.npy')}")
        print(f"   - {os.path.join(save_dir, 'eval_ssims.npy')}")
        print(f"   - {report_path}")
        print()
    
    return avg_psnr, avg_ssim


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate DEA-Net model')
    parser.add_argument('--model', '--model_path', type=str, required=True, help='Path to model file (.pk)')
    parser.add_argument('--dataset', '--data_dir', type=str, default='dataset/ITS', help='Path to dataset directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--max_images', type=int, default=0, help='Maximum number of images to evaluate (0=all)')
    parser.add_argument('--base_dim', type=int, default=32, help='Base dimension of the model (not used, for compatibility)')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.dataset, args.device, args.max_images)
