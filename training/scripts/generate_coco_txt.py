"""
生成COCO数据集的train2017.txt和val2017.txt文件
"""

import os
from pathlib import Path

def generate_image_list(dataset_path, split='train2017'):
    """
    生成图像路径列表文件
    
    Args:
        dataset_path: COCO数据集根目录
        split: 'train2017' 或 'val2017'
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / 'images' / split
    
    if not images_dir.exists():
        # 尝试另一种结构（直接在根目录）
        images_dir = dataset_path / split
    
    if not images_dir.exists():
        print(f"❌ 图像目录不存在: {images_dir}")
        return
    
    # 获取所有jpg图像
    image_files = sorted(images_dir.glob('*.jpg'))
    
    if not image_files:
        print(f"❌ 未找到图像文件: {images_dir}")
        return
    
    # 生成txt文件
    txt_file = dataset_path / f'{split}.txt'
    
    with open(txt_file, 'w') as f:
        for img in image_files:
            # 写入相对路径
            rel_path = f'./images/{split}/{img.name}'
            f.write(rel_path + '\n')
    
    print(f"✓ 生成 {txt_file}")
    print(f"  图像数量: {len(image_files)}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/datasets/coco',
                       help='COCO数据集根目录')
    args = parser.parse_args()
    
    print("="*60)
    print("生成COCO图像路径列表")
    print("="*60)
    print(f"数据集路径: {args.path}\n")
    
    # 生成train和val的txt文件
    generate_image_list(args.path, 'train2017')
    generate_image_list(args.path, 'val2017')
    
    print("\n" + "="*60)
    print("✓ 完成")
    print("="*60)
