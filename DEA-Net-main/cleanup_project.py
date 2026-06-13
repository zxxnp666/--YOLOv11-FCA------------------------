#!/usr/bin/env python3
"""
清理项目：删除数据准备工具和临时文件
保留训练必需的核心代码
"""

import os
import shutil
from pathlib import Path


# 要删除的文件
FILES_TO_DELETE = [
    # 数据准备工具
    "check_dataset.py",
    "create_dataset_structure.py",
    "create_ots_val.py",
    "filter_bdd100k_night.py",
    "filter_traffic_scenes.py",
    "filter_traffic_simple.py",
    "fix_ots_hazy.py",
    "random_select_ots.py",
    "show_dataset_tree.py",
    "smart_filter_ots.py",
    "synthesize_fog.py",
    "verify_and_fix_dataset.py",
    "test_yolo.py",
    "yolov8n.pt",
    "rename_new_data.py",
    
    # Web应用
    "web_app.py",
    "test_web_app.py",
    "web_requirements.txt",
    
    # 文档（可选，如果不需要可以删除）
    "数据集准备完整指南.md",
    "BDD100K夜晚数据集构建指南.md",
    "OTS筛选使用说明.md",
    
    # 备份文件
    "code/data/data_loader_backup.py",
    "code/data/data_loader_mixed.py",
]

# 要删除的目录
DIRS_TO_DELETE = [
    "templates",
    "uploads",
    "test_output",
]


def cleanup():
    """清理项目"""
    
    print("\n" + "="*80)
    print("项目清理工具".center(80))
    print("="*80 + "\n")
    
    deleted_files = []
    deleted_dirs = []
    failed = []
    
    # 删除文件
    print("📄 删除文件...")
    for file_path in FILES_TO_DELETE:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(file_path)
                print(f"  ✓ {file_path}")
            else:
                print(f"  - {file_path} (不存在)")
        except Exception as e:
            failed.append((file_path, str(e)))
            print(f"  ✗ {file_path}: {e}")
    
    # 删除目录
    print("\n📁 删除目录...")
    for dir_path in DIRS_TO_DELETE:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                deleted_dirs.append(dir_path)
                print(f"  ✓ {dir_path}/")
            else:
                print(f"  - {dir_path}/ (不存在)")
        except Exception as e:
            failed.append((dir_path, str(e)))
            print(f"  ✗ {dir_path}/: {e}")
    
    # 统计
    print("\n" + "="*80)
    print("清理完成".center(80))
    print("="*80)
    print(f"\n✓ 删除文件: {len(deleted_files)} 个")
    print(f"✓ 删除目录: {len(deleted_dirs)} 个")
    
    if failed:
        print(f"\n⚠️  失败: {len(failed)} 个")
        for item, error in failed:
            print(f"  - {item}: {error}")
    
    print("\n保留的核心文件:")
    print("  ✓ code/ - 训练代码")
    print("  ✓ dataset/ - 数据集")
    print("  ✓ trained_models/ - 模型权重")
    print("  ✓ results/ - 训练结果")
    print("  ✓ fig/ - 论文图片")
    print("  ✓ README.md - 项目说明")
    print("  ✓ requirements.txt - 依赖")
    print("  ✓ TRAINING_OUTPUT.md - 训练日志")
    
    print("\n" + "="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='清理项目文件')
    parser.add_argument('--dry-run', action='store_true',
                        help='预览要删除的文件，不实际删除')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("\n" + "="*80)
        print("预览模式 (不会实际删除)".center(80))
        print("="*80 + "\n")
        
        print("将要删除的文件:")
        for f in FILES_TO_DELETE:
            exists = "✓" if os.path.exists(f) else "✗"
            print(f"  {exists} {f}")
        
        print("\n将要删除的目录:")
        for d in DIRS_TO_DELETE:
            exists = "✓" if os.path.exists(d) else "✗"
            print(f"  {exists} {d}/")
        
        print("\n运行 'python cleanup_project.py' 执行删除")
        print("="*80 + "\n")
    else:
        confirm = input("确认删除以上文件？(y/n): ")
        if confirm.lower() == 'y':
            cleanup()
        else:
            print("取消操作")


if __name__ == '__main__':
    main()
