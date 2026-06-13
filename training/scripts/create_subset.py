"""
创建COCO数据集子集
用于毕业设计，减少数据量和训练时间
"""

import os
import shutil
import json
import random
from pathlib import Path
import argparse


def create_coco_subset(
    source_dir,
    target_dir,
    train_ratio=0.1,
    val_ratio=0.1,
    seed=42
):
    """
    创建COCO数据集子集
    
    Args:
        source_dir: 完整COCO数据集路径 (例如: D:/coco)
        target_dir: 输出子集路径 (例如: D:/coco_subset)
        train_ratio: 训练集比例 (0.1 = 10%)
        val_ratio: 验证集比例 (0.1 = 10%)
        seed: 随机种子
    """
    
    random.seed(seed)
    
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    print("="*60)
    print("创建COCO数据集子集")
    print("="*60)
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"训练集比例: {train_ratio*100}%")
    print(f"验证集比例: {val_ratio*100}%")
    print("="*60)
    
    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "train2017").mkdir(exist_ok=True)
    (target_dir / "val2017").mkdir(exist_ok=True)
    (target_dir / "annotations").mkdir(exist_ok=True)
    
    # 处理训练集
    print("\n处理训练集...")
    train_images = list((source_dir / "train2017").glob("*.jpg"))
    train_subset = random.sample(train_images, int(len(train_images) * train_ratio))
    
    print(f"  原始图像数: {len(train_images)}")
    print(f"  选取图像数: {len(train_subset)}")
    print(f"  复制中...")
    
    for i, img in enumerate(train_subset, 1):
        shutil.copy2(img, target_dir / "train2017" / img.name)
        if i % 1000 == 0:
            print(f"    已复制 {i}/{len(train_subset)} 张")
    
    print(f"  ✓ 训练集完成")
    
    # 处理验证集
    print("\n处理验证集...")
    val_images = list((source_dir / "val2017").glob("*.jpg"))
    val_subset = random.sample(val_images, int(len(val_images) * val_ratio))
    
    print(f"  原始图像数: {len(val_images)}")
    print(f"  选取图像数: {len(val_subset)}")
    print(f"  复制中...")
    
    for i, img in enumerate(val_subset, 1):
        shutil.copy2(img, target_dir / "val2017" / img.name)
        if i % 100 == 0:
            print(f"    已复制 {i}/{len(val_subset)} 张")
    
    print(f"  ✓ 验证集完成")
    
    # 处理标注文件
    print("\n处理标注文件...")
    
    # 获取选中的图像ID
    train_ids = {int(img.stem) for img in train_subset}
    val_ids = {int(img.stem) for img in val_subset}
    
    # 处理训练集标注
    train_ann_file = source_dir / "annotations" / "instances_train2017.json"
    if train_ann_file.exists():
        print("  处理 instances_train2017.json...")
        with open(train_ann_file, 'r') as f:
            train_ann = json.load(f)
        
        # 筛选图像和标注
        train_ann['images'] = [img for img in train_ann['images'] if img['id'] in train_ids]
        train_ann['annotations'] = [ann for ann in train_ann['annotations'] if ann['image_id'] in train_ids]
        
        # 保存
        with open(target_dir / "annotations" / "instances_train2017.json", 'w') as f:
            json.dump(train_ann, f)
        
        print(f"    图像: {len(train_ann['images'])}")
        print(f"    标注: {len(train_ann['annotations'])}")
    
    # 处理验证集标注
    val_ann_file = source_dir / "annotations" / "instances_val2017.json"
    if val_ann_file.exists():
        print("  处理 instances_val2017.json...")
        with open(val_ann_file, 'r') as f:
            val_ann = json.load(f)
        
        # 筛选图像和标注
        val_ann['images'] = [img for img in val_ann['images'] if img['id'] in val_ids]
        val_ann['annotations'] = [ann for ann in val_ann['annotations'] if ann['image_id'] in val_ids]
        
        # 保存
        with open(target_dir / "annotations" / "instances_val2017.json", 'w') as f:
            json.dump(val_ann, f)
        
        print(f"    图像: {len(val_ann['images'])}")
        print(f"    标注: {len(val_ann['annotations'])}")
    
    print("\n" + "="*60)
    print("✓ 子集创建完成！")
    print("="*60)
    
    # 统计大小
    total_size = sum(f.stat().st_size for f in target_dir.rglob('*') if f.is_file())
    print(f"\n总大小: {total_size / (1024**3):.2f} GB")
    print(f"\n目标目录: {target_dir}")
    print("\n下一步:")
    print(f"  1. 压缩目录: tar -czf coco_subset.tar.gz {target_dir.name}")
    print(f"  2. 上传到服务器")
    print(f"  3. 解压: tar -xzf coco_subset.tar.gz")


def main():
    parser = argparse.ArgumentParser(description='创建COCO数据集子集')
    parser.add_argument('--source', type=str, required=True,
                       help='完整COCO数据集路径 (例如: D:/coco)')
    parser.add_argument('--target', type=str, required=True,
                       help='输出子集路径 (例如: D:/coco_subset)')
    parser.add_argument('--train_ratio', type=float, default=0.1,
                       help='训练集比例 (默认: 0.1 = 10%%)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例 (默认: 0.1 = 10%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    create_coco_subset(
        source_dir=args.source,
        target_dir=args.target,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
