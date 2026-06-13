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

# 尝试导入夜晚优化（如果可用）
try:
    from loss.night_loss import NightDehazeLoss
    NIGHT_LOSS_AVAILABLE = True
except ImportError:
    NIGHT_LOSS_AVAILABLE = False
    print("Warning: Night loss not available, using standard loss")


start_time = time.time()
# 注意：这里先用iters_per_epoch计算，后面会用实际的len(loader_train)更新
steps = opt.iters_per_epoch * opt.epochs
T = steps


def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr


def format_time(seconds):
    """格式化时间显示"""
    return str(timedelta(seconds=int(seconds)))


def train(net, loader_train, loader_test, optim, criterion):
    global steps, T
    # 使用实际的dataloader长度重新计算总步数
    actual_steps_per_epoch = len(loader_train)
    steps = actual_steps_per_epoch * opt.epochs
    T = steps
    losses = []
    loss_log = {'L1': [], 'CR': [], 'total': []}
    loss_log_tmp = {'L1': [], 'CR': [], 'total': []}
    psnr_log = []
    ssim_log = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    loader_train_iter = iter(loader_train)
    
    print("\n" + "="*80)
    print("Training Started".center(80))
    print("="*80)
    print(f"Total Steps: {steps}")
    print(f"Total Epochs: {opt.epochs}")
    print(f"Steps per Epoch: {len(loader_train)}")
    print(f"Batch Size: {opt.bs}")
    print(f"Start Learning Rate: {opt.start_lr}")
    print(f"End Learning Rate: {opt.end_lr}")
    print("="*80 + "\n")

    for step in range(start_step + 1, steps + 1):
        net.train()
        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        x, y = next(loader_train_iter)
        x = x.to(opt.device)
        y = y.to(opt.device)

        out = net(x)
        
        # 检查是否使用夜晚损失
        if criterion[1] is None:  # 使用夜晚损失
            loss, loss_dict = criterion[0](out, y, x)  # 传入input_img
            loss_L1 = loss_dict.get('l1', torch.tensor(0.0))
            loss_CR = loss_dict.get('cr', torch.tensor(0.0))
        else:  # 使用标准损失
            if opt.w_loss_L1 > 0:
                loss_L1 = criterion[0](out, y)
            if opt.w_loss_CR > 0:
                loss_CR = criterion[1](out, y, x)
            loss = opt.w_loss_L1 * loss_L1 + opt.w_loss_CR * loss_CR
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        loss_log_tmp['L1'].append(loss_L1.item() if isinstance(loss_L1, torch.Tensor) else 0)
        loss_log_tmp['CR'].append(loss_CR.item() if isinstance(loss_CR, torch.Tensor) else 0)
        loss_log_tmp['total'].append(loss.item())

        # 计算当前epoch和进度
        current_epoch = step // len(loader_train) + 1
        step_in_epoch = step % len(loader_train)
        epoch_progress = (step_in_epoch / len(loader_train)) * 100
        total_progress = (step / steps) * 100
        
        # 计算时间信息
        elapsed_time = time.time() - start_time
        steps_per_sec = step / elapsed_time if elapsed_time > 0 else 0
        eta_seconds = (steps - step) / steps_per_sec if steps_per_sec > 0 else 0
        
        # 打印详细的训练信息
        print(
            f'\r[Epoch {current_epoch}/{opt.epochs}] '
            f'[Step {step_in_epoch}/{len(loader_train)} ({epoch_progress:.1f}%)] '
            f'[Total: {step}/{steps} ({total_progress:.1f}%)] '
            f'Loss: {loss.item():.5f} (L1:{loss_L1.item():.5f} CR:{opt.w_loss_CR * loss_CR.item():.5f}) '
            f'LR: {lr:.7f} | '
            f'Time: {format_time(elapsed_time)} | '
            f'ETA: {format_time(eta_seconds)}',
            end='', flush=True)

        if step % len(loader_train) == 0:
            loader_train_iter = iter(loader_train)
            current_epoch = int(step / len(loader_train))
            
            # 计算并记录平均loss
            for key in loss_log.keys():
                avg_loss = np.average(np.array(loss_log_tmp[key]))
                loss_log[key].append(avg_loss)
                loss_log_tmp[key] = []
            
            # 打印epoch总结
            print(f"\n{'='*80}")
            print(f"Epoch {current_epoch} Summary:")
            print(f"  Avg Total Loss: {loss_log['total'][-1]:.5f}")
            print(f"  Avg L1 Loss: {loss_log['L1'][-1]:.5f}")
            print(f"  Avg CR Loss: {loss_log['CR'][-1]:.5f}")
            print(f"{'='*80}\n")
            
            plot_loss_log(loss_log, current_epoch, opt.saved_plot_dir)
            np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)
        
        # 评估逻辑：根据eval_freq参数控制
        # eval_freq=0: 只在最后一个epoch评估
        # eval_freq=N: 每N个epoch评估一次
        should_eval = False
        if opt.eval_freq == 0:
            # 只在训练结束时评估
            should_eval = (step == steps)
        else:
            # 使用实际的dataloader长度而不是iters_per_epoch
            if step % len(loader_train) == 0:
                current_epoch = int(step / len(loader_train))
                should_eval = (current_epoch % opt.eval_freq == 0)
        
        if should_eval:
            # 使用实际的steps_per_epoch计算epoch
            actual_steps_per_epoch = len(loader_train)
            epoch = int(step / actual_steps_per_epoch)
            
            print(f"\n{'='*80}")
            print(f"Training Completed at Epoch {epoch}".center(80))
            print(f"{'='*80}")
            
            # ========== 清理显存，为评估做准备 ==========
            torch.cuda.empty_cache()
            
            # ========== 第一步：先保存模型（确保训练结果不丢失） ==========
            print(f"\n💾 Saving model checkpoint...")
            saved_checkpoint_path = os.path.join(opt.saved_model_dir, f'epoch_{epoch}.pk')
            torch.save({
                'epoch': epoch,
                'step': step,
                'max_psnr': max_psnr,
                'max_ssim': max_ssim,
                'ssims': ssims,
                'psnrs': psnrs,
                'losses': losses,
                'model': net.state_dict(),
                'optimizer': optim.state_dict()
            }, saved_checkpoint_path)
            print(f" Model saved to: {saved_checkpoint_path}")
            
            # ========== 第二步：再进行评估 ==========
            print(f"\n{'='*80}")
            print(f"Starting Evaluation at Epoch {epoch}...".center(80))
            print(f"{'='*80}")
            
            try:
                # 再次清理显存
                torch.cuda.empty_cache()
                
                with torch.no_grad():
                    ssim_eval, psnr_eval = test(net, loader_test)

                # 计算改进
                psnr_improve = psnr_eval - max_psnr if max_psnr > 0 else 0
                ssim_improve = ssim_eval - max_ssim if max_ssim > 0 else 0
                
                log = (f'\n[Evaluation Results]\n'
                       f'  Epoch: {epoch} | Step: {step}\n'
                       f'  PSNR: {psnr_eval:.4f} (Best: {max_psnr:.4f}, Δ: {psnr_improve:+.4f})\n'
                       f'  SSIM: {ssim_eval:.4f} (Best: {max_ssim:.4f}, Δ: {ssim_improve:+.4f})\n')
                
                print(log)
                with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                    f.write(log + '\n')

                ssims.append(ssim_eval)
                psnrs.append(psnr_eval)
                psnr_log.append(psnr_eval)
                ssim_log.append(ssim_eval)
                
                # 尝试绘图（如果失败也不影响）
                try:
                    plot_psnr_log(psnr_log, epoch, opt.saved_plot_dir)
                    plot_ssim_log(ssim_log, epoch, opt.saved_plot_dir)
                    plot_metrics_combined(psnr_log, ssim_log, epoch, opt.saved_plot_dir)
                    print(" Plots saved successfully")
                except Exception as e:
                    print(f"  Warning: Failed to save plots: {e}")

                # 如果是最佳模型，保存为best.pk
                if psnr_eval > max_psnr:
                    max_ssim = max(max_ssim, ssim_eval)
                    max_psnr = max(max_psnr, psnr_eval)
                    print(f'\n{"🎉 NEW BEST MODEL! 🎉".center(80)}')
                    print(f'  Saved at Epoch {epoch} | Step {step}')
                    print(f'  Best PSNR: {max_psnr:.4f} | Best SSIM: {max_ssim:.4f}')
                    print(f"{'='*80}\n")
                    
                    saved_best_model_path = os.path.join(opt.saved_model_dir, 'best.pk')
                    torch.save({
                        'epoch': epoch,
                        'step': step,
                        'max_psnr': max_psnr,
                        'max_ssim': max_ssim,
                        'ssims': ssims,
                        'psnrs': psnrs,
                        'losses': losses,
                        'model': net.state_dict(),
                        'optimizer': optim.state_dict()
                    }, saved_best_model_path)
                    print(f" Best model saved to: {saved_best_model_path}")
                else:
                    print(f"{'='*80}\n")
                
                # 保存评估数据
                np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
                np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)
                
            except Exception as e:
                print(f"\n❌ Evaluation failed: {e}")
                print(f"⚠️  But don't worry, model has been saved at: {saved_checkpoint_path}")
                print(f"{'='*80}\n")
            
            # 重置dataloader迭代器
            loader_train_iter = iter(loader_train)

def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def test(net, loader_test):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    # 限制测试图片数量
    max_images = opt.max_eval_images if opt.max_eval_images > 0 else len(loader_test)
    test_count = min(max_images, len(loader_test))
    print(f"  Testing on {test_count} images (total: {len(loader_test)})...")
    for i, (inputs, targets, hazy_name) in enumerate(loader_test):
        if i >= test_count:
            break
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        with torch.no_grad():
            H, W = inputs.shape[2:]
            inputs = pad_img(inputs, 4)
            pred = net(inputs).clamp(0, 1)
            pred = pred[:, :, :H, :W]
            # save_path = os.path.join(opt.saved_infer_dir, hazy_name[0])
            # save_image(pred, save_path)
        ssim_tmp = ssim(pred, targets).item()
        psnr_tmp = psnr(pred, targets)
        ssims.append(ssim_tmp)
        psnrs.append(psnr_tmp)
        # 显示测试进度
        if (i + 1) % 100 == 0 or (i + 1) == test_count:
            print(f"  Progress: {i+1}/{test_count} | "
                  f"Avg PSNR: {np.mean(psnrs):.4f} | "
                  f"Avg SSIM: {np.mean(ssims):.4f}", end='\r')
    print()  # 换行
    return np.mean(ssims), np.mean(psnrs)


def set_seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    set_seed_torch(666)

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
    
    # 检查数据集路径是否存在
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training dataset not found at: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test dataset not found at: {test_dir}")
    
    train_hazy_dir = os.path.join(train_dir, 'hazy')
    train_clear_dir = os.path.join(train_dir, 'clear')
    test_hazy_dir = os.path.join(test_dir, 'hazy')
    test_clear_dir = os.path.join(test_dir, 'clear')
    
    # 检查子目录是否存在
    for dir_path, dir_name in [(train_hazy_dir, 'train/hazy'), (train_clear_dir, 'train/clear'),
                                 (test_hazy_dir, 'test/hazy'), (test_clear_dir, 'test/clear')]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    print(f"Loading dataset from: {dataset_path}")
    
    # 检查是否使用夜晚数据增强
    use_night_aug = hasattr(opt, 'use_night_augmentation') and opt.use_night_augmentation
    if use_night_aug:
        print("Using night data augmentation for training")
    
    train_set = TrainDataset(train_hazy_dir, train_clear_dir, use_night_augmentation=use_night_aug)
    test_set = TestDataset(test_hazy_dir, test_clear_dir)
    
    print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")
    
    loader_train = DataLoader(dataset=train_set, batch_size=opt.bs, shuffle=True, num_workers=opt.num_workers)
    loader_test = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=min(4, opt.num_workers))

    # 检查GPU可用性
    if opt.device == 'cuda':
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available, falling back to CPU")
            opt.device = 'cpu'
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("Using CPU for training")
    
    # 创建模型，支持base_dim参数
    print(f"\nCreating model with base_dim={opt.base_dim}...")
    net = DEANet(base_dim=opt.base_dim)
    net = net.to(opt.device)

    epoch_size = len(loader_train)
    print("epoch_size: ", epoch_size)
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))

    # 选择损失函数
    criterion = []
    
    # 检查是否使用夜晚优化
    use_night_loss = hasattr(opt, 'use_night_loss') and opt.use_night_loss and NIGHT_LOSS_AVAILABLE
    
    if use_night_loss:
        print("\nUsing Night-Aware Loss for training")
        night_criterion = NightDehazeLoss(
            w_l1=opt.w_loss_L1,
            w_night_aware=0.5,
            w_illumination=0.3,
            w_perceptual=0.1,
            w_cr=opt.w_loss_CR
        ).to(opt.device)
        criterion.append(night_criterion)
        criterion.append(None)  # 占位，保持兼容性
    else:
        print("\nUsing Standard Loss (L1 + CR)")
        criterion.append(nn.L1Loss().to(opt.device))
        criterion.append(ContrastLoss(ablation=False))

    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.start_lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()
    
    # 加载预训练模型（如果指定）
    if opt.resume and opt.pre_trained_model != 'null':
        print(f"\nLoading pretrained model from: {opt.pre_trained_model}")
        try:
            checkpoint = torch.load(opt.pre_trained_model, map_location=opt.device)
            # 处理DataParallel保存的模型
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            
            # 处理DataParallel前缀不匹配问题
            # 情况1: 保存时没有module.前缀，但当前模型有DataParallel包装
            # 情况2: 保存时有module.前缀，但当前模型没有DataParallel包装
            model_keys = list(net.state_dict().keys())
            state_keys = list(state_dict.keys())
            
            # 检查是否需要添加或删除'module.'前缀
            if model_keys[0].startswith('module.') and not state_keys[0].startswith('module.'):
                # 当前模型有DataParallel，但保存的没有 -> 添加前缀
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
                print("  Added 'module.' prefix to state_dict")
            elif not model_keys[0].startswith('module.') and state_keys[0].startswith('module.'):
                # 当前模型没有DataParallel，但保存的有 -> 删除前缀
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                print("  Removed 'module.' prefix from state_dict")
            
            net.load_state_dict(state_dict)
            print(f"✓ Pretrained model loaded successfully")
            if 'max_psnr' in checkpoint:
                print(f"  Previous best PSNR: {checkpoint['max_psnr']:.2f} dB")
            if 'max_ssim' in checkpoint:
                print(f"  Previous best SSIM: {checkpoint['max_ssim']:.4f}")
        except Exception as e:
            print(f"❌ Warning: Failed to load pretrained model: {e}")
            print("Starting training from scratch...")
    
    train(net, loader_train, loader_test, optimizer, criterion)