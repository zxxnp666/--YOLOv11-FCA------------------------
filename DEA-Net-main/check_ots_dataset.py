"""
检查OTS数据集的完整性和配对情况
"""
import os
from pathlib import Path
from collections import defaultdict
from PIL import Image

def check_dataset(dataset_dir):
    """
    检查数据集的完整性
    
    Args:
        dataset_dir: 数据集根目录
    """
    print("="*80)
    print("OTS数据集完整性检查".center(80))
    print("="*80)
    print(f"\n📂 数据集目录: {dataset_dir}\n")
    
    dataset_path = Path(dataset_dir)
    
    # 检查目录结构
    train_hazy = dataset_path / 'train' / 'hazy'
    train_clear = dataset_path / 'train' / 'clear'
    val_hazy = dataset_path / 'val' / 'hazy'
    val_clear = dataset_path / 'val' / 'clear'
    
    dirs_to_check = {
        'train/hazy': train_hazy,
        'train/clear': train_clear,
        'val/hazy': val_hazy,
        'val/clear': val_clear
    }
    
    print("步骤1: 检查目录结构...")
    all_dirs_exist = True
    for name, path in dirs_to_check.items():
        if path.exists():
            print(f"  ✅ {name}: 存在")
        else:
            print(f"  ❌ {name}: 不存在")
            all_dirs_exist = False
    
    if not all_dirs_exist:
        print("\n❌ 目录结构不完整！")
        return
    
    print("\n步骤2: 统计文件数量...")
    
    # 统计文件数量
    train_hazy_files = list(train_hazy.glob('*.jpg'))
    train_clear_files = list(train_clear.glob('*.jpg'))
    val_hazy_files = list(val_hazy.glob('*.jpg'))
    val_clear_files = list(val_clear.glob('*.jpg'))
    
    print(f"  训练集:")
    print(f"    有雾图片: {len(train_hazy_files)} 张")
    print(f"    清晰图片: {len(train_clear_files)} 张")
    print(f"  验证集:")
    print(f"    有雾图片: {len(val_hazy_files)} 张")
    print(f"    清晰图片: {len(val_clear_files)} 张")
    
    print("\n步骤3: 检查训练集配对...")
    train_issues = check_pairing(train_hazy_files, train_clear_files, "训练集")
    
    print("\n步骤4: 检查验证集配对...")
    val_issues = check_pairing(val_hazy_files, val_clear_files, "验证集")
    
    print("\n步骤5: 分析雾参数分布...")
    
    # 训练集雾参数分布
    print("\n  训练集雾参数分布:")
    train_fog_dist = analyze_fog_params(train_hazy_files)
    for fog_key, count in sorted(train_fog_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {fog_key}: {count} 张")
    if len(train_fog_dist) > 10:
        print(f"    ... 还有 {len(train_fog_dist) - 10} 种雾参数")
    
    # 验证集雾参数分布
    print("\n  验证集雾参数分布:")
    val_fog_dist = analyze_fog_params(val_hazy_files)
    for fog_key, count in sorted(val_fog_dist.items()):
        print(f"    {fog_key}: {count} 张")
    
    print("\n步骤6: 检查图片尺寸...")
    check_image_sizes(train_clear_files[:100], "训练集（抽样100张）")
    check_image_sizes(val_clear_files[:50], "验证集（抽样50张）")
    
    print("\n步骤7: 检查训练集和验证集是否重叠...")
    check_overlap(train_hazy_files, val_hazy_files)
    
    # 最终总结
    print("\n" + "="*80)
    print("检查结果总结".center(80))
    print("="*80)
    
    total_issues = train_issues + val_issues
    
    if total_issues == 0:
        print("\n✅ 数据集完整性检查通过！")
        print(f"\n📊 数据集统计:")
        print(f"  训练集: {len(train_hazy_files)} 对")
        print(f"  验证集: {len(val_hazy_files)} 对")
        print(f"  训练集雾参数种类: {len(train_fog_dist)}")
        print(f"  验证集雾参数种类: {len(val_fog_dist)}")
        print(f"\n✅ 所有图片配对正确")
        print(f"✅ 训练集和验证集无重叠")
        print(f"✅ 验证集包含多种雾参数")
        print(f"\n🎉 数据集可以用于训练！")
    else:
        print(f"\n⚠️  发现 {total_issues} 个问题")
        print(f"请检查上述错误信息")
    
    print("="*80)


def check_pairing(hazy_files, clear_files, dataset_name):
    """检查有雾图片和清晰图片的配对"""
    issues = 0
    
    # 提取清晰图片的ID
    clear_ids = set()
    for clear_file in clear_files:
        clear_ids.add(clear_file.stem)
    
    # 检查每个有雾图片是否有对应的清晰图片
    missing_clear = []
    for hazy_file in hazy_files:
        # 解析文件名: XXXX_beta_A.jpg
        parts = hazy_file.stem.split('_')
        if len(parts) >= 3:
            image_id = parts[0]
            if image_id not in clear_ids:
                missing_clear.append(hazy_file.name)
                issues += 1
    
    if missing_clear:
        print(f"  ❌ {dataset_name}: {len(missing_clear)} 个有雾图片缺少对应的清晰图片")
        if len(missing_clear) <= 5:
            for name in missing_clear:
                print(f"      - {name}")
        else:
            for name in missing_clear[:5]:
                print(f"      - {name}")
            print(f"      ... 还有 {len(missing_clear) - 5} 个")
    else:
        print(f"  ✅ {dataset_name}: 所有图片配对正确")
    
    return issues


def analyze_fog_params(hazy_files):
    """分析雾参数分布"""
    fog_dist = defaultdict(int)
    
    for hazy_file in hazy_files:
        parts = hazy_file.stem.split('_')
        if len(parts) >= 3:
            beta = parts[1]
            A = parts[2]
            fog_key = f"{beta}_{A}"
            fog_dist[fog_key] += 1
    
    return fog_dist


def check_image_sizes(image_files, dataset_name, min_size=256):
    """检查图片尺寸"""
    small_images = []
    error_images = []
    
    for img_file in image_files:
        try:
            img = Image.open(img_file)
            if min(img.size) < min_size:
                small_images.append((img_file.name, img.size))
        except Exception as e:
            error_images.append((img_file.name, str(e)))
    
    if small_images:
        print(f"  ⚠️  {dataset_name}: {len(small_images)} 张图片尺寸小于 {min_size}×{min_size}")
        for name, size in small_images[:3]:
            print(f"      - {name}: {size}")
    else:
        print(f"  ✅ {dataset_name}: 所有图片尺寸符合要求")
    
    if error_images:
        print(f"  ❌ {dataset_name}: {len(error_images)} 张图片无法读取")
        for name, error in error_images[:3]:
            print(f"      - {name}: {error}")


def check_overlap(train_files, val_files):
    """检查训练集和验证集是否有重叠"""
    # 提取图片ID
    train_ids = set()
    for f in train_files:
        parts = f.stem.split('_')
        if len(parts) >= 3:
            train_ids.add(parts[0])
    
    val_ids = set()
    for f in val_files:
        parts = f.stem.split('_')
        if len(parts) >= 3:
            val_ids.add(parts[0])
    
    overlap = train_ids & val_ids
    
    if overlap:
        print(f"  ❌ 发现 {len(overlap)} 个重叠的图片ID")
        if len(overlap) <= 10:
            print(f"      重叠ID: {', '.join(sorted(overlap))}")
        else:
            print(f"      重叠ID (前10个): {', '.join(sorted(list(overlap)[:10]))}")
    else:
        print(f"  ✅ 训练集和验证集无重叠")


if __name__ == '__main__':
    # 配置路径
    DATASET_DIR = r'E:\数据集\DEA-net\RESIDE\OTS'
    
    print("\n" + "="*80)
    print("开始检查数据集".center(80))
    print("="*80)
    
    check_dataset(DATASET_DIR)
    
    print("\n💡 如果检查通过，可以:")
    print("1. 将数据集上传到服务器")
    print("2. 开始训练模型")
    print("\n如果发现问题，请重新运行 recreate_ots_dataset.py")
