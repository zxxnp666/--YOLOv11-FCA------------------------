"""
将 COCO JSON 格式转换为 YOLO txt 格式
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

def convert_coco_json(json_file, img_dir, output_dir):
    """
    转换 COCO JSON 到 YOLO 格式
    
    Args:
        json_file: COCO JSON 文件路径
        img_dir: 图像目录
        output_dir: 输出目录
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取 JSON
    print(f"读取 {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 创建图像 ID 到文件名的映射
    images = {img['id']: img for img in data['images']}
    
    # 创建类别 ID 映射（COCO 91类 -> 80类）
    categories = {cat['id']: i for i, cat in enumerate(data['categories'])}
    
    print(f"转换 {len(data['annotations'])} 个标注...")
    
    # 按图像分组标注
    img_annotations = {}
    for ann in tqdm(data['annotations']):
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    # 转换每张图像的标注
    print(f"生成 YOLO 格式文件...")
    for img_id, anns in tqdm(img_annotations.items()):
        if img_id not in images:
            continue
            
        img_info = images[img_id]
        img_w = img_info['width']
        img_h = img_info['height']
        img_name = Path(img_info['file_name']).stem
        
        # 创建 YOLO 格式标注
        yolo_lines = []
        for ann in anns:
            # 跳过无效标注
            if 'bbox' not in ann or ann.get('iscrowd', 0) == 1:
                continue
            
            # COCO bbox: [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # 转换为 YOLO 格式（归一化的中心坐标和宽高）
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width = w / img_w
            height = h / img_h
            
            # 获取类别 ID
            cat_id = ann['category_id']
            if cat_id not in categories:
                continue
            class_id = categories[cat_id]
            
            # YOLO 格式：class_id x_center y_center width height
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # 保存到文件
        if yolo_lines:
            output_file = output_dir / f"{img_name}.txt"
            with open(output_file, 'w') as f:
                f.writelines(yolo_lines)
    
    print(f"✓ 转换完成！生成了 {len(img_annotations)} 个标注文件")
    print(f"✓ 保存位置: {output_dir}")


if __name__ == '__main__':
    # 数据集根目录
    coco_dir = Path('/root/sj-tmp/YOLOv11-FCA/training/data/datasets/coco')
    
    # 转换训练集
    print("="*70)
    print("转换训练集...")
    print("="*70)
    convert_coco_json(
        json_file=coco_dir / 'annotations/instances_train2017.json',
        img_dir=coco_dir / 'train2017',
        output_dir=coco_dir / 'labels/train2017'
    )
    
    # 转换验证集
    print("\n" + "="*70)
    print("转换验证集...")
    print("="*70)
    convert_coco_json(
        json_file=coco_dir / 'annotations/instances_val2017.json',
        img_dir=coco_dir / 'val2017',
        output_dir=coco_dir / 'labels/val2017'
    )
    
    print("\n" + "="*70)
    print("🎉 全部转换完成！")
    print("="*70)
    print(f"labels 文件夹位置: {coco_dir / 'labels'}")
