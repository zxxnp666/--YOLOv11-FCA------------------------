"""
检查和管理 checkpoint 的工具脚本
"""

import os
import torch
from datetime import datetime
from pathlib import Path


def check_checkpoint_info(checkpoint_path):
    """
    查看checkpoint的详细信息
    
    Args:
        checkpoint_path: checkpoint文件路径
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
        return
    
    print("="*70)
    print(f"📦 Checkpoint信息: {checkpoint_path}")
    print("="*70)
    
    try:
        # 加载checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # 文件大小
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print(f"📁 文件大小: {file_size:.2f} MB")
        
        # 修改时间
        mtime = os.path.getmtime(checkpoint_path)
        mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"🕐 修改时间: {mod_time}")
        
        print("\n📊 Checkpoint内容:")
        print("-"*70)
        
        # 显示所有键
        if isinstance(ckpt, dict):
            for key in ckpt.keys():
                if key == 'model' or key == 'state_dict':
                    print(f"  ✓ {key}: <模型权重>")
                elif key == 'optimizer':
                    print(f"  ✓ {key}: <优化器状态>")
                elif isinstance(ckpt[key], (int, float, str)):
                    print(f"  ✓ {key}: {ckpt[key]}")
                else:
                    print(f"  ✓ {key}: {type(ckpt[key]).__name__}")
        
        # 特殊信息
        print("\n📈 训练信息:")
        print("-"*70)
        
        if 'epoch' in ckpt:
            print(f"  Epoch: {ckpt['epoch']}")
        
        if 'best_fitness' in ckpt:
            print(f"  Best mAP: {ckpt['best_fitness']:.4f}")
        
        if 'PSNR' in ckpt:
            print(f"  PSNR: {ckpt['PSNR']:.2f} dB")
        
        if 'SSIM' in ckpt:
            print(f"  SSIM: {ckpt['SSIM']:.4f}")
        
        if 'date' in ckpt:
            print(f"  保存时间: {ckpt['date']}")
        
        print("="*70)
        print("✓ Checkpoint检查完成")
        
    except Exception as e:
        print(f"❌ 加载checkpoint失败: {e}")


def list_checkpoints(checkpoint_dir='checkpoints'):
    """
    列出所有可用的checkpoint
    
    Args:
        checkpoint_dir: checkpoint目录
    """
    print("="*70)
    print("📦 可用的Checkpoints")
    print("="*70)
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ 目录不存在: {checkpoint_dir}")
        return
    
    # 查找所有.pt和.pth文件
    checkpoint_files = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith(('.pt', '.pth')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, checkpoint_dir)
                file_size = os.path.getsize(full_path) / (1024 * 1024)
                mtime = os.path.getmtime(full_path)
                mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                checkpoint_files.append({
                    'path': rel_path,
                    'size': file_size,
                    'time': mod_time
                })
    
    if not checkpoint_files:
        print("❌ 未找到任何checkpoint文件")
        print("\n💡 提示:")
        print("  1. 下载DEA-Net预训练权重")
        print("  2. 或开始训练生成checkpoint")
        return
    
    # 按类型分组显示
    print("\n🔹 DEA-Net权重:")
    deanet_found = False
    for ckpt in checkpoint_files:
        if 'deanet' in ckpt['path'].lower():
            print(f"  ✓ {ckpt['path']}")
            print(f"    大小: {ckpt['size']:.2f} MB | 时间: {ckpt['time']}")
            deanet_found = True
    if not deanet_found:
        print("  ❌ 未找到DEA-Net权重")
    
    print("\n🔹 YOLOv11-FCA权重:")
    yolo_found = False
    for ckpt in checkpoint_files:
        if 'yolov11' in ckpt['path'].lower():
            print(f"  ✓ {ckpt['path']}")
            print(f"    大小: {ckpt['size']:.2f} MB | 时间: {ckpt['time']}")
            yolo_found = True
    if not yolo_found:
        print("  ❌ 未找到YOLOv11-FCA权重（需要训练生成）")
    
    print("\n🔹 端到端模型权重:")
    e2e_found = False
    for ckpt in checkpoint_files:
        if 'end_to_end' in ckpt['path'].lower():
            print(f"  ✓ {ckpt['path']}")
            print(f"    大小: {ckpt['size']:.2f} MB | 时间: {ckpt['time']}")
            e2e_found = True
    if not e2e_found:
        print("  ❌ 未找到端到端模型权重")
    
    print("\n" + "="*70)
    print(f"✓ 共找到 {len(checkpoint_files)} 个checkpoint文件")


def backup_checkpoint(checkpoint_path, backup_dir='checkpoints/backup'):
    """
    备份checkpoint
    
    Args:
        checkpoint_path: 要备份的checkpoint路径
        backup_dir: 备份目录
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
        return
    
    # 创建备份目录
    os.makedirs(backup_dir, exist_ok=True)
    
    # 生成备份文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.basename(checkpoint_path)
    name, ext = os.path.splitext(filename)
    backup_filename = f"{name}_{timestamp}{ext}"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    # 复制文件
    import shutil
    shutil.copy2(checkpoint_path, backup_path)
    
    print("="*70)
    print("💾 Checkpoint备份完成")
    print("="*70)
    print(f"原文件: {checkpoint_path}")
    print(f"备份到: {backup_path}")
    
    file_size = os.path.getsize(backup_path) / (1024 * 1024)
    print(f"大小: {file_size:.2f} MB")
    print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Checkpoint管理工具')
    parser.add_argument('--list', action='store_true',
                       help='列出所有checkpoint')
    parser.add_argument('--info', type=str,
                       help='查看指定checkpoint的信息')
    parser.add_argument('--backup', type=str,
                       help='备份指定的checkpoint')
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints()
    elif args.info:
        check_checkpoint_info(args.info)
    elif args.backup:
        backup_checkpoint(args.backup)
    else:
        # 默认显示所有checkpoint
        list_checkpoints()
        
        print("\n💡 使用提示:")
        print("  查看checkpoint信息:")
        print("    python scripts/check_checkpoint.py --info checkpoints/yolov11_fca/train/weights/best.pt")
        print("\n  备份checkpoint:")
        print("    python scripts/check_checkpoint.py --backup checkpoints/yolov11_fca/train/weights/best.pt")


if __name__ == '__main__':
    main()
