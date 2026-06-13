"""
夜晚场景专用训练脚本
使用夜晚感知损失和数据增强
"""

import os, time, math
import numpy as np
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.backends import cudnn
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from logger import plot_loss_log, plot_psnr_log, plot_ssim_log, plot_metrics_combined
from metric import psnr, ssim
from model import DEANet
from loss.night_loss import NightDehazeLoss
from option_train import opt
from data.data_loader import TrainDataset, TestDataset


start_time = time.time()


def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr


def format_time(seconds):
    """格式化时间显示"""
    return str(timedelta(seconds=int(seconds)))


def train_night(net, loader_train, loader_test, optim, criterion):
    """
    夜晚场景训练
    使用夜晚感知损失
    """
    steps = len(loader_train) * opt.epochs
    T = steps
    
    loss_log = {
        'l1': [], 'night_aware': [], 'illumination': [], 
        'perceptual': [], 'cr': [], 'total': []
    }
    loss_log_tmp = {
        'l1': [], 'night_aware': [], 'illumination': [], 
        'perceptual': [], 'cr': [], 'total': []
    }
    psnr_log = []
    ssim_log = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    loader_train_iter = iter(loader_train)
    
    print("\n" + "="*80)
    print("Night Scene Training Started".center(80))
    print("="*80)
    print(f"Model: base_dim={opt.base_dim}")
    print(f"Loss: Night-Aware Loss (brightness + light_suppress + color)")
    print(f"Total Steps: {steps}")
    print(f"Total Epochs: {opt.epochs}")
    print(f"Steps per Epoch: {len(loader_train)}")
    print(f"Batch Size: {opt.bs}")
    print("="*80 + "\n")

    for step in range(start_step + 1, steps + 1):
        net.train()
        
        # 学习率调度
        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        # 获取数据
        try:
            x, y = next(loader_train_iter)
        except StopIteration:
            loader_train_iter = iter(loader_train)
            x, y = next(loader_train_iter)
        
        x = x.to(opt.device)
        y = y.to(opt.device)

        # 前向传播
        out = net(x)
        
        # 计算夜晚感知损失
        loss_total, losses = criterion(out, y)

        # 反向传播
        loss_total.backward()
        optim.step()
        optim.zero_grad()

        # 记录损失
        for key in losses:
            loss_log_tmp[key].append(losses[key].item())

        # 每个epoch结束时评估
        if step % len(loader_train) == 0:
            epoch = step // len(loader_train)
            
            # 计算平均损失
            for key in loss_log_tmp:
                if len(loss_log_tmp[key]) > 0:
                    loss_log[key].append(np.mean(loss_log_tmp[key]))
                    loss_log_tmp[key] = []
                else:
                    loss_log[key].append(0)
            
            # 评估
            if opt.eval_freq > 0 and epoch % opt.eval_freq == 0:
                net.eval()
                with torch.no_grad():
                    ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)
                
                psnr_log.append(psnr_eval)
                ssim_log.append(ssim_eval)
                
                # 保存最佳模型
                if ssim_eval > max_ssim:
                    max_ssim = ssim_eval
                    torch.save({
                        'step': step,
                        'epoch': epoch,
                        'max_psnr': max_psnr,
                        'max_ssim': max_ssim,
                        'ssims': ssims,
                        'psnrs': psnrs,
                        'state_dict': net.state_dict(),
                        'base_dim': opt.base_dim
                    }, os.path.join(opt.saved_model_dir, 'best.pk'))
                
                if psnr_eval > max_psnr:
                    max_psnr = psnr_eval
            
            # 打印进度
            elapsed = time.time() - start_time
            eta = elapsed / step * (steps - step)
            
            print(f"Epoch {epoch}/{opt.epochs} | Step {step}/{steps}")
            print(f"  Loss: {loss_log['total'][-1]:.4f} "
                  f"(L1:{loss_log['l1'][-1]:.4f} "
                  f"Night:{loss_log['night_aware'][-1]:.4f} "
                  f"Illum:{loss_log['illumination'][-1]:.4f})")
            print(f"  PSNR: {psnr_eval:.2f} dB | SSIM: {ssim_eval:.4f}")
            print(f"  LR: {lr:.6f} | ETA: {format_time(eta)}")
            print()
            
            # 保存checkpoint
            if epoch % 10 == 0:
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'state_dict': net.state_dict(),
                    'base_dim': opt.base_dim
                }, os.path.join(opt.saved_model_dir, f'epoch_{epoch}.pk'))
    
    # 保存最终模型
    torch.save({
        'step': steps,
        'epoch': opt.epochs,
        'max_psnr': max_psnr,
        'max_ssim': max_ssim,
        'state_dict': net.state_dict(),
        'base_dim': opt.base_dim
    }, os.path.join(opt.saved_model_dir, 'final.pk'))
    
    # 绘制训练曲线
    plot_loss_log(loss_log, opt.saved_plot_dir)
    plot_psnr_log(psnr_log, opt.saved_plot_dir)
    plot_ssim_log(ssim_log, opt.saved_plot_dir)
    plot_metrics_combined(psnr_log, ssim_log, opt.saved_plot_dir)
    
    print("\n" + "="*80)
    print("Training Completed!".center(80))
    print("="*80)
    print(f"Best PSNR: {max_psnr:.2f} dB")
    print(f"Best SSIM: {max_ssim:.4f}")
    print(f"Total Time: {format_time(time.time() - start_time)}")
    print("="*80 + "\n")


def test(net, loader_test, max_psnr, max_ssim, step):
    """评估模型"""
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    
    max_images = opt.max_eval_images if opt.max_eval_images > 0 else len(loader_test)
    
    for i, (inputs, targets, _) in enumerate(loader_test):  # 添加第三个返回值
        if i >= max_images:
            break
        
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        
        with torch.no_grad():
            pred = net(inputs)
        
        ssim_val = ssim(pred, targets).item()
        psnr_val = psnr(pred, targets)
        
        ssims.append(ssim_val)
        psnrs.append(psnr_val)
    
    return np.mean(ssims), np.mean(psnrs)
    
    return np.mean(ssims), np.mean(psnrs)


if __name__ == "__main__":
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print("Night Scene Dehazing Training".center(80))
    print(f"{'='*80}\n")
    
    # 创建模型
    print(f"Creating Model (base_dim={opt.base_dim})...")
    net = DEANet(base_dim=opt.base_dim).to(opt.device)
    
    # 加载预训练模型（如果有）
    if opt.resume and opt.pre_trained_model != 'null':
        print(f"Loading pretrained model from {opt.pre_trained_model}...")
        checkpoint = torch.load(opt.pre_trained_model, map_location=opt.device)
        net.load_state_dict(checkpoint['state_dict'])
        print(f"Pretrained model loaded (PSNR: {checkpoint.get('max_psnr', 0):.2f} dB)")
    
    print(f"Model parameters: {sum(p.numel() for p in net.parameters())/1e6:.2f}M")
    
    # 数据加载
    print(f"\nLoading Dataset: {opt.dataset}")
    
    # 根据数据集名称构建路径
    if opt.dataset == 'ITS':
        # ITS使用OTS数据（因为实际文件夹名是OTS）
        dataset_path = os.path.join(opt.data_dir, 'RESIDE', 'OTS')
    elif opt.dataset == 'OTS':
        dataset_path = os.path.join(opt.data_dir, 'RESIDE', 'OTS')
    elif opt.dataset == 'HAZE4K':
        dataset_path = os.path.join(opt.data_dir, 'Haze4K')
    elif opt.dataset == 'Night':
        dataset_path = os.path.join(opt.data_dir, 'Night')
    else:
        dataset_path = os.path.join(opt.data_dir, opt.dataset)
    
    train_dir = os.path.join(dataset_path, 'train')
    
    # OTS数据集使用val而不是test
    if opt.dataset in ['ITS', 'OTS']:
        test_dir = os.path.join(dataset_path, 'val')
    else:
        test_dir = os.path.join(dataset_path, 'test')
    
    train_hazy_dir = os.path.join(train_dir, 'hazy')
    train_clear_dir = os.path.join(train_dir, 'clear')
    test_hazy_dir = os.path.join(test_dir, 'hazy')
    test_clear_dir = os.path.join(test_dir, 'clear')
    
    loader_train = DataLoader(
        dataset=TrainDataset(train_hazy_dir, train_clear_dir),
        batch_size=opt.bs,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    loader_test = DataLoader(
        dataset=TestDataset(test_hazy_dir, test_clear_dir),
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers
    )
    print(f"Train samples: {len(loader_train.dataset)}")
    print(f"Test samples: {len(loader_test.dataset)}")
    
    # 优化器和损失函数
    optimizer = optim.Adam(net.parameters(), lr=opt.start_lr)
    
    # 使用夜晚感知损失
    criterion = NightDehazeLoss(
        w_l1=1.0,
        w_night_aware=0.5,
        w_illumination=0.3,
        w_perceptual=0.1,
        w_cr=0.1
    ).to(opt.device)
    
    print("\nLoss Function: Night-Aware Loss")
    print("  - L1 Loss: 1.0")
    print("  - Night-Aware Loss: 0.5 (brightness + light_suppress + color)")
    print("  - Illumination Loss: 0.3")
    print("  - Perceptual Loss: 0.1")
    print("  - CR Loss: 0.1")
    
    # 开始训练
    train_night(net, loader_train, loader_test, optimizer, criterion)
