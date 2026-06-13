"""
检查 COCO 数据集完整性
"""

from pathlib import Path
import json

def check_dataset(dataset_dir):
    """检查数据集完整性"""
    
    dataset_dir = Path(dataset_dir)
    
    print("="*70)
    print("🔍 检查 COCO 数据集完整性")
    print("="*70)
    
    # 1. 检查目录结构
    print("\n📁 检查目录结构...")
    required_dirs = [
        'train2017',
        'val2017',
        'annotations'
    ]
    
    for dir_name in required_dirs:
        dir_path = dataset_dir / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (缺失)")
            return False
    
    # 2. 检查标注文件
    print("\n📄 检查标注文件...")
    annotation_files = [
        'annotations/instances_train2017.json',
        'annotations/instances_val2017.json'
    ]
    
    for ann_file in annotation_files:
        ann_path = dataset_dir / ann_file
        if ann_path.exists():
            size_mb = ann_path.stat().st_size / 1024 / 1024
            print(f"  ✓ {ann_file} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {ann_file} (缺失)")
            return False
    
    # 3. 检查图像数量
    print("\n🖼️  检查图像数量...")
    
    train_images = list((dataset_dir / 'train2017').glob('*.jpg'))
    val_images = list((dataset_dir / 'val2017').glob('*.jpg'))
    
    print(f"  训练集图像: {len(train_images):,} 张")
    print(f"  验证集图像: {len(val_images):,} 张")
    
    if len(train_images) == 0:
        print("  ✗ 训练集图像为空！")
        return False
    
    if len(val_images) == 0:
        print("  ✗ 验证集图像为空！")
        return False
    
    # 4. 检查标注数量
    print("\n📝 检查标注文件...")
    
    labels_dir = dataset_dir / 'labels'
    if labels_dir.exists():
        train_labels = list((labels_dir / 'train2017').glob('*.txt')) if (labels_dir / 'train2017').exists() else []
        val_labels = list((labels_dir / 'val2017').glob('*.txt')) if (labels_dir / 'val2017').exists() else []
        
        print(f"  训练集标注: {len(train_labels):,} 个")
        print(f"  验证集标注: {len(val_labels):,} 个")
        
        # 计算覆盖率
        train_coverage = len(train_labels) / len(train_images) * 100 if train_images else 0
        val_coverage = len(val_labels) / len(val_images) * 100 if val_images else 0
        
        print(f"  训练集覆盖率: {train_coverage:.1f}%")
        print(f"  验证集覆盖率: {val_coverage:.1f}%")
        
        if len(train_labels) == 0:
            print("  ⚠️  训练集标注为空！需要运行转换脚本")
            print("     运行: python convert_coco_to_yolo.py")
    else:
        print("  ✗ labels/ 文件夹不存在")
        print("  ⚠️  需要运行转换脚本生成 YOLO 格式标注")
        print("     运行: python convert_coco_to_yolo.py")
    
    # 5. 检查 JSON 标注内容
    print("\n📊 检查标注内容...")
    
    try:
        with open(dataset_dir / 'annotations/instances_train2017.json', 'r') as f:
            train_data = json.load(f)
        
        print(f"  训练集:")
        print(f"    - 图像: {len(train_data['images']):,} 张")
        print(f"    - 标注: {len(train_data['annotations']):,} 个")
        print(f"    - 类别: {len(train_data['categories'])} 类")
        
        with open(dataset_dir / 'annotations/instances_val2017.json', 'r') as f:
            val_data = json.load(f)
        
        print(f"  验证集:")
        print(f"    - 图像: {len(val_data['images']):,} 张")
        print(f"    - 标注: {len(val_data['annotations']):,} 个")
        print(f"    - 类别: {len(val_data['categories'])} 类")
        
    except Exception as e:
        print(f"  ✗ 读取 JSON 失败: {e}")
        return False
    
    # 6. 总结
    print("\n" + "="*70)
    
    if labels_dir.exists() and len(train_labels) > 0 and len(val_labels) > 0:
        print("✅ 数据集完整，可以开始训练！")
        print("\n运行训练命令:")
        print("python train/train_yolov11_fca_custom.py --data coco.yaml --epochs 100 --batch 32 --device 0")
    else:
        print("⚠️  数据集不完整，需要先转换标注格式")
        print("\n运行转换命令:")
        print("python convert_coco_to_yolo.py")
    
    print("="*70)
    
    return True


if __name__ == '__main__':
    dataset_dir = '/root/sj-tmp/YOLOv11-FCA/training/data/datasets/coco'
    check_dataset(dataset_dir)
