"""
知识蒸馏训练脚本
用Teacher模型(base_dim=32)指导Student模型(base_dim=24)训练
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
from loss import ContrastLoss
from option_train import opt
from data.data_loader import TrainDataset, TestDataset


def kd_loss(student_output, teacher_output, temperature=4.0):
    """
    知识蒸馏损失
    对于图像去雾任务，使用MSE损失
    """
    # 确保尺寸匹配（如果有细微差异，调整到相同尺寸）
    if student_output.shape != teacher_output.shape:
        # 使用插值调整student输出到teacher的尺寸
        student_output = F.interpolate(
            student_output, 
            size=teacher_output.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
    
    # 对于图像任务，直接使用MSE
    # temperature参数可以用来调整损失的尺度
    loss = F.mse_loss(student_output, teacher_output) * (temperature / 4.0)
    
    return loss


def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr


def format_time(seconds):
    """格式化时间显示"""
    return str(timedelta(seconds=int(seconds)))


def train_with_kd(student_net, teacher_net, loader_train, loader_test, optim, criterion):
    """
    使用知识蒸馏训练student模型
    """
    steps = len(loader_train) * opt.epochs
    T = steps
    
    losses = []
    loss_log = {'L1': [], 'CR': [], 'KD': [], 'total': []}
    loss_log_tmp = {'L1': [], 'CR': [], 'KD': [], 'total': []}
    psnr_log = []
    ssim_log = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    loader_train_iter = iter(loader_train)
    
    # Teacher模型设置为eval模式，不更新参数
    teacher_net.eval()
    for param in teacher_net.parameters():
        param.requires_grad = False
    
    print("\n" + "="*80)
    print("Knowledge Distillation Training Started".center(80))
    print("="*80)
    print(f"Teacher Model: base_dim=32, params={sum(p.numel() for p in teacher_net.parameters())/1e6:.2f}M")
    print(f"Student Model: base_dim={opt.base_dim}, params={sum(p.numel() for p in student_net.parameters())/1e6:.2f}M")
    print(f"Total Steps: {steps}")
    print(f"Total Epochs: {opt.epochs}")
    print(f"Steps per Epoch: {len(loader_train)}")
    print(f"Batch Size: {opt.bs}")
    print(f"KD Temperature: {opt.kd_temperature}")
    print(f"KD Loss Weight: {opt.w_loss_KD}")
    print("="*80 + "\n")

    for step in range(start_step + 1, steps + 1):
        student_net.train()
        
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

        # Student前向传播
        student_out = student_net(x)
        
        # Teacher前向传播（不计算梯度）
        with torch.no_grad():
            teacher_out = teacher_net(x)
        
        # 调试：打印尺寸（只在第一步）
        if step == 1:
            print(f"\nDebug - Tensor shapes:")
            print(f"  Input: {x.shape}")
            print(f"  Student output: {student_out.shape}")
            print(f"  Teacher output: {teacher_out.shape}")
            print(f"  Target: {y.shape}\n")
        
        # 计算损失
        loss_total = 0
        loss_L1_val = 0
        loss_CR_val = 0
        loss_KD_val = 0
        
        # L1损失（与GT比较）
        if opt.w_loss_L1 > 0:
            loss_L1_val = criterion[0](student_out, y)
            loss_total += opt.w_loss_L1 * loss_L1_val
        
        # CR损失
        if opt.w_loss_CR > 0:
            loss_CR_val = criterion[1](student_out, y, x)
            loss_total += opt.w_loss_CR * loss_CR_val
        
        # KD损失（与Teacher比较）
        if opt.w_loss_KD > 0:
            loss_KD_val = kd_loss(student_out, teacher_out, opt.kd_temperature)
            loss_total += opt.w_loss_KD * loss_KD_val

        # 反向传播
        loss_total.backward()
        optim.step()
        optim.zero_grad()

        # 记录损失
        loss_log_tmp['L1'].append(loss_L1_val.item() if opt.w_loss_L1 > 0 else 0)
        loss_log_tmp['CR'].append(loss_CR_val.item() if opt.w_loss_CR > 0 else 0)
        loss_log_tmp['KD'].append(loss_KD_val.item() if opt.w_loss_KD > 0 else 0)
        loss_log_tmp['total'].append(loss_total.item())

        # 每个epoch结束时评估
        if step % len(loader_train) == 0:
            epoch = step // len(loader_train)
            
            # 计算平均损失
            for key in loss_log_tmp:
                loss_log[key].append(np.mean(loss_log_tmp[key]))
                loss_log_tmp[key] = []
            
            # 评估
            if opt.eval_freq > 0 and epoch % opt.eval_freq == 0:
                student_net.eval()
                with torch.no_grad():
                    ssim_eval, psnr_eval = test(student_net, loader_test, max_psnr, max_ssim, step)
                
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
                        'losses': losses,
                        'state_dict': student_net.state_dict(),
                        'base_dim': opt.base_dim
                    }, os.path.join(opt.saved_model_dir, 'best.pk'))
                
                if psnr_eval > max_psnr:
                    max_psnr = psnr_eval
            
            # 打印进度
            elapsed = time.time() - start_time
            eta = elapsed / step * (steps - step)
            print(f"Epoch {epoch}/{opt.epochs} | Step {step}/{steps} | "
                  f"Loss: {loss_log['total'][-1]:.4f} (L1:{loss_log['L1'][-1]:.4f} "
                  f"CR:{loss_log['CR'][-1]:.4f} KD:{loss_log['KD'][-1]:.4f}) | "
                  f"PSNR: {psnr_eval:.2f} | SSIM: {ssim_eval:.4f} | "
                  f"LR: {lr:.6f} | ETA: {format_time(eta)}")
            
            # 保存checkpoint
            if epoch % 10 == 0:
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'state_dict': student_net.state_dict(),
                    'base_dim': opt.base_dim
                }, os.path.join(opt.saved_model_dir, f'epoch_{epoch}.pk'))
    
    # 保存最终模型
    torch.save({
        'step': steps,
        'epoch': opt.epochs,
        'max_psnr': max_psnr,
        'max_ssim': max_ssim,
        'state_dict': student_net.state_dict(),
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


if __name__ == "__main__":
    start_time = time.time()
    
    # 检查参数
    if not opt.use_kd:
        print("Error: Please use --use_kd flag for knowledge distillation training")
        exit(1)
    
    if opt.teacher_model is None:
        print("Error: Please specify --teacher_model path")
        exit(1)
    
    if opt.base_dim >= 32:
        print("Warning: Student base_dim should be smaller than 32 for lightweight model")
    
    # 加载Teacher模型
    print(f"\nLoading Teacher Model from {opt.teacher_model}...")
    teacher_net = DEANet(base_dim=32).to(opt.device)
    checkpoint = torch.load(opt.teacher_model, map_location=opt.device)
    
    # 处理不同的checkpoint格式
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    # 如果是DataParallel保存的，需要去掉'module.'前缀
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    teacher_net.load_state_dict(state_dict)
    print(f"Teacher Model loaded (PSNR: {checkpoint.get('max_psnr', 0):.2f} dB)")
    
    # 创建Student模型
    print(f"\nCreating Student Model (base_dim={opt.base_dim})...")
    student_net = DEANet(base_dim=opt.base_dim).to(opt.device)
    print(f"Student Model created")
    
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
    optimizer = optim.Adam(student_net.parameters(), lr=opt.start_lr)
    criterion = [nn.L1Loss().to(opt.device), ContrastLoss().to(opt.device)]
    
    # 开始训练
    train_with_kd(student_net, teacher_net, loader_train, loader_test, optimizer, criterion)
