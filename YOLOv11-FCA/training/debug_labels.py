"""
调试 labels 文件夹检测问题
"""

from pathlib import Path
import random

# 数据集路径
dataset_dir = Path('/root/sj-tmp/YOLOv11-FCA/training/data/datasets/coco')
train_img_dir = dataset_dir / 'train2017'
train_label_dir = dataset_dir / 'labels' / 'train2017'

print("="*70)
print("🔍 调试 labels 检测问题")
print("="*70)

# 1. 检查目录是否存在
print("\n📁 检查目录...")
print(f"图像目录: {train_img_dir}")
print(f"  存在: {train_img_dir.exists()}")

print(f"标注目录: {train_label_dir}")
print(f"  存在: {train_label_dir.exists()}")

if not train_img_dir.exists() or not train_label_dir.exists():
    print("\n❌ 目录不存在！")
    exit(1)

# 2. 统计文件数量
img_files = list(train_img_dir.glob('*.jpg'))
label_files = list(train_label_dir.glob('*.txt'))

print(f"\n📊 文件统计...")
print(f"图像文件: {len(img_files):,} 个")
print(f"标注文件: {len(label_files):,} 个")

if len(label_files) == 0:
    print("\n❌ 标注文件为空！需要运行 convert_coco_to_yolo.py")
    exit(1)

# 3. 检查文件名匹配
print(f"\n🔍 检查文件名匹配...")

# 随机抽取 10 个图像检查
sample_imgs = random.sample(img_files, min(10, len(img_files)))

matched = 0
unmatched = 0

for img_file in sample_imgs:
    # 对应的标注文件
    label_file = train_label_dir / f"{img_file.stem}.txt"
    
    if label_file.exists():
        matched += 1
        print(f"  ✓ {img_file.name} → {label_file.name}")
    else:
        unmatched += 1
        print(f"  ✗ {img_file.name} → {label_file.name} (不存在)")

print(f"\n匹配: {matched}/{len(sample_imgs)}")
print(f"不匹配: {unmatched}/{len(sample_imgs)}")

# 4. 检查标注文件内容
print(f"\n📝 检查标注文件内容...")

sample_label = random.choice(label_files)
print(f"随机标注文件: {sample_label.name}")

with open(sample_label, 'r') as f:
    lines = f.readlines()
    print(f"  行数: {len(lines)}")
    if lines:
        print(f"  第一行: {lines[0].strip()}")
        # 检查格式：class_id x_center y_center width height
        parts = lines[0].strip().split()
        if len(parts) == 5:
            print(f"  格式: ✓ 正确 (class_id x y w h)")
        else:
            print(f"  格式: ✗ 错误 (应该是 5 个值)")

# 5. 检查 ultralytics 的查找逻辑
print(f"\n🔍 模拟 ultralytics 查找逻辑...")

# ultralytics 会将 train2017 替换为 labels/train2017
img_path = train_img_dir / img_files[0].name
label_path_expected = Path(str(img_path).replace('train2017', 'labels/train2017').replace('.jpg', '.txt'))

print(f"图像路径: {img_path}")
print(f"期望标注路径: {label_path_expected}")
print(f"标注存在: {label_path_expected.exists()}")

# 6. 总结
print("\n" + "="*70)

if matched == len(sample_imgs) and len(label_files) > 0:
    print("✅ labels 文件夹结构正确！")
    print("\n可能的问题:")
    print("1. 缓存文件损坏，删除缓存重试:")
    print("   rm -f /root/sj-tmp/YOLOv11-FCA/training/data/datasets/coco/*.cache")
    print("\n2. coco.yaml 路径配置问题，检查:")
    print("   cat coco.yaml")
else:
    print("❌ labels 文件夹有问题！")
    print("\n建议:")
    print("1. 重新运行转换脚本:")
    print("   python convert_coco_to_yolo.py")

print("="*70)
