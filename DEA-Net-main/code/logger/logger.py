from matplotlib import pyplot as plt
import numpy as np
import os


def plot_loss_log(loss_log, epoch, loss_dir):
    axis = np.linspace(1, epoch, epoch)
    for key in loss_log.keys():
        label = '{} Loss'.format(key)
        fig = plt.figure()
        plt.title(label)
        plt.plot(axis, np.array(loss_log[key]))
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(loss_dir, 'loss_{}.pdf'.format(key)))
        plt.close(fig)


def plot_psnr_log(psnr_log, epoch, psnr_dir):
    # 使用实际的数据点数量
    num_points = len(psnr_log)
    if num_points == 0:
        return
    axis = np.linspace(1, num_points, num_points)
    label = 'PSNR'
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, np.array(psnr_log), label='PSNR')
    plt.legend()
    plt.xlabel('Evaluation Count')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(os.path.join(psnr_dir, 'psnr.pdf'))
    plt.close(fig)


def plot_ssim_log(ssim_log, epoch, ssim_dir):
    """绘制SSIM曲线"""
    num_points = len(ssim_log)
    if num_points == 0:
        return
    axis = np.linspace(1, num_points, num_points)
    label = 'SSIM'
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, np.array(ssim_log), label='SSIM')
    plt.legend()
    plt.xlabel('Evaluation Count')
    plt.ylabel('SSIM')
    plt.grid(True)
    plt.savefig(os.path.join(ssim_dir, 'ssim.pdf'))
    plt.close(fig)


def plot_metrics_combined(psnr_log, ssim_log, epoch, save_dir):
    """绘制PSNR和SSIM的组合图"""
    num_points = len(psnr_log)
    if num_points == 0:
        return
    axis = np.linspace(1, num_points, num_points)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # PSNR
    ax1.plot(axis, np.array(psnr_log), 'b-', linewidth=2, label='PSNR')
    ax1.set_title('PSNR over Evaluations')
    ax1.set_xlabel('Evaluation Count')
    ax1.set_ylabel('PSNR (dB)')
    ax1.legend()
    ax1.grid(True)
    
    # SSIM
    ax2.plot(axis, np.array(ssim_log), 'r-', linewidth=2, label='SSIM')
    ax2.set_title('SSIM over Evaluations')
    ax2.set_xlabel('Evaluation Count')
    ax2.set_ylabel('SSIM')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_combined.pdf'))
    plt.close(fig)