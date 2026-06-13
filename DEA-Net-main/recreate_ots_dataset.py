"""
重新筛选OTS数据集，确保多样化的雾参数
用于轻量化消融实验
"""
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def recreate_ots_dataset(
    hazy_dir,
    clear_dir,
    output_dir,
    train_size=2000,
    val_size=500,
    min_size=256
):
    """
    重新创建OTS数据集，确保验证集包含多种雾参数
    
    Args:
        hazy_dir: 有雾图片目录
        clear_dir: 清晰图片目录
        output_dir: 输出路径
        train_size: 训练集大小
        val_size: 验证集大小
        min_size: 最小图片尺寸
    """
    
    print("="*80)
    print("重新筛选OTS数据集".center(80))
    print("="*80)
    
    hazy_path = Path(hazy_dir)
    clear_path = Path(clear_dir)
    
    # 创建输出目录
    output_train_hazy = Path(output_dir) / 'train' / 'hazy'
    output_train_clear = Path(output_dir) / 'train' / 'clear'
    output_val_hazy = Path(output_dir) / 'val' / 'hazy'
    output_val_clear = Path(output_dir) / 'val' / 'clear'
    
    for dir_path in [output_train_hazy, output_train_clear, output_val_hazy, output_val_clear]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 有雾图片: {hazy_dir}")
    print(f"📂 清晰图片: {clear_dir}")
    print(f"📂 输出目录: {output_dir}")
    print(f"📊 训练集目标: {train_size} 对")
    print(f"📊 验证集目标: {val_size} 对")
    print(f"📏 最小尺寸: {min_size}×{min_size}\n")
    
    # 步骤1: 收集所有有雾图片，按雾参数分组
    print("步骤1: 分析雾参数分布...")
    fog_params_dict = defaultdict(list)  # {(beta, A): [image_ids]}
    
    for hazy_file in hazy_path.glob('*.jpg'):
        # 解析文件名: XXXX_beta_A.jpg
        parts = hazy_file.stem.split('_')
        if len(parts) >= 3:
            image_id = parts[0]
            beta = parts[1]
            A = parts[2]
            fog_key = f"{beta}_{A}"
            
            # 检查对应的清晰图片是否存在
            clear_file = clear_path / f"{image_id}.jpg"
            if clear_file.exists():
                fog_params_dict[fog_key].append(image_id)
    
    print(f"\n发现 {len(fog_params_dict)} 种不同的雾参数:")
    for fog_key, images in sorted(fog_params_dict.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
        print(f"  {fog_key}: {len(images)} 张图片")
    
    if len(fog_params_dict) == 0:
        print("\n❌ 错误: 没有找到任何有效的图片对！")
        return
    
    # 步骤2: 为验证集选择多样化的雾参数
    print(f"\n步骤2: 为验证集选择多样化的雾参数...")
    
    # 选择最常见的8种雾参数用于验证集
    num_fog_params_for_val = min(8, len(fog_params_dict))
    fog_params_for_val = sorted(fog_params_dict.keys(), 
                                 key=lambda x: len(fog_params_dict[x]), 
                                 reverse=True)[:num_fog_params_for_val]
    
    print(f"选择 {len(fog_params_for_val)} 种雾参数用于验证集:")
    for fog_key in fog_params_for_val:
        print(f"  {fog_key}: {len(fog_params_dict[fog_key])} 张可用")
    
    # 步骤3: 从每种雾参数中均匀采样验证集
    print(f"\n步骤3: 构建验证集...")
    val_images_per_param = val_size // len(fog_params_for_val)
    val_image_ids = set()
    val_selections = []
    
    for fog_key in fog_params_for_val:
        available_images = fog_params_dict[fog_key]
        selected = random.sample(available_images, min(val_images_per_param, len(available_images)))
        for image_id in selected:
            val_image_ids.add(image_id)
            val_selections.append((image_id, fog_key))
    
    # 如果还不够，从剩余的随机补充
    if len(val_selections) < val_size:
        remaining_needed = val_size - len(val_selections)
        all_available = []
        for fog_key, images in fog_params_dict.items():
            for img_id in images:
                if img_id not in val_image_ids:
                    all_available.append((img_id, fog_key))
        
        if all_available:
            additional = random.sample(all_available, min(remaining_needed, len(all_available)))
            val_selections.extend(additional)
            for img_id, _ in additional:
                val_image_ids.add(img_id)
    
    print(f"✅ 验证集: {len(val_selections)} 对图片")
    
    # 统计验证集的雾参数分布
    val_fog_distribution = defaultdict(int)
    for _, fog_key in val_selections:
        val_fog_distribution[fog_key] += 1
    
    print("\n验证集雾参数分布:")
    for fog_key, count in sorted(val_fog_distribution.items()):
        print(f"  {fog_key}: {count} 张")
    
    # 步骤4: 构建训练集（排除验证集中的图片）
    print(f"\n步骤4: 构建训练集...")
    train_selections = []
    
    for fog_key, images in fog_params_dict.items():
        for image_id in images:
            if image_id not in val_image_ids:
                train_selections.append((image_id, fog_key))
    
    # 随机采样训练集
    if len(train_selections) > train_size:
        train_selections = random.sample(train_selections, train_size)
    
    print(f"✅ 训练集: {len(train_selections)} 对图片")
    
    # 统计训练集的雾参数分布
    train_fog_distribution = defaultdict(int)
    for _, fog_key in train_selections:
        train_fog_distribution[fog_key] += 1
    
    print("\n训练集雾参数分布 (前10种):")
    for fog_key, count in sorted(train_fog_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {fog_key}: {count} 张")
    
    # 步骤5: 复制文件
    print(f"\n步骤5: 复制文件...")
    
    # 复制训练集
    print("复制训练集...")
    train_copied = 0
    train_skipped = 0
    
    for image_id, fog_key in train_selections:
        beta, A = fog_key.split('_')
        hazy_name = f"{image_id}_{beta}_{A}.jpg"
        clear_name = f"{image_id}.jpg"
        
        src_hazy = hazy_path / hazy_name
        src_clear = clear_path / clear_name
        
        if src_hazy.exists() and src_clear.exists():
            try:
                from PIL import Image
                img = Image.open(src_clear)
                if min(img.size) >= min_size:
                    shutil.copy2(src_hazy, output_train_hazy / hazy_name)
                    shutil.copy2(src_clear, output_train_clear / clear_name)
                    train_copied += 1
                else:
                    train_skipped += 1
            except Exception as e:
                train_skipped += 1
        else:
            train_skipped += 1
        
        if (train_copied + train_skipped) % 100 == 0:
            print(f"  进度: {train_copied + train_skipped}/{len(train_selections)}", end='\r')
    
    print(f"\n  ✅ 训练集复制完成: {train_copied} 对")
    if train_skipped > 0:
        print(f"  ⚠️  跳过: {train_skipped} 对（尺寸过小或文件缺失）")
    
    # 复制验证集
    print("\n复制验证集...")
    val_copied = 0
    val_skipped = 0
    
    for image_id, fog_key in val_selections:
        beta, A = fog_key.split('_')
        hazy_name = f"{image_id}_{beta}_{A}.jpg"
        clear_name = f"{image_id}.jpg"
        
        src_hazy = hazy_path / hazy_name
        src_clear = clear_path / clear_name
        
        if src_hazy.exists() and src_clear.exists():
            try:
                from PIL import Image
                img = Image.open(src_clear)
                if min(img.size) >= min_size:
                    shutil.copy2(src_hazy, output_val_hazy / hazy_name)
                    shutil.copy2(src_clear, output_val_clear / clear_name)
                    val_copied += 1
                else:
                    val_skipped += 1
            except Exception as e:
                val_skipped += 1
        else:
            val_skipped += 1
        
        if (val_copied + val_skipped) % 50 == 0:
            print(f"  进度: {val_copied + val_skipped}/{len(val_selections)}", end='\r')
    
    print(f"\n  ✅ 验证集复制完成: {val_copied} 对")
    if val_skipped > 0:
        print(f"  ⚠️  跳过: {val_skipped} 对（尺寸过小或文件缺失）")
    
    # 最终统计
    print("\n" + "="*80)
    print("数据集创建完成".center(80))
    print("="*80)
    print(f"\n📊 最终统计:")
    print(f"  训练集: {train_copied} 对")
    print(f"  验证集: {val_copied} 对")
    print(f"  验证集雾参数种类: {len(val_fog_distribution)}")
    print(f"\n📂 输出目录: {output_dir}")
    print(f"  ├── train/")
    print(f"  │   ├── hazy/ ({train_copied} 张)")
    print(f"  │   └── clear/ ({train_copied} 张)")
    print(f"  └── val/")
    print(f"      ├── hazy/ ({val_copied} 张)")
    print(f"      └── clear/ ({val_copied} 张)")
    print("\n✅ 数据集已准备好用于训练！")
    print("="*80)


if __name__ == '__main__':
    # 配置路径
    HAZY_DIR = r'E:\数据集\OTS_ALPHA\haze\OTS'
    CLEAR_DIR = r'E:\数据集\OTS_ALPHA\clear\clear_images'
    OUTPUT_DIR = r'E:\数据集\DEA-net\RESIDE\OTS'
    
    # 设置随机种子以保证可复现
    random.seed(42)
    
    print("\n" + "="*80)
    print("OTS数据集筛选工具".center(80))
    print("="*80)
    print(f"\n📂 有雾图片: {HAZY_DIR}")
    print(f"📂 清晰图片: {CLEAR_DIR}")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print("\n⚠️  脚本将:")
    print(f"1. 从源目录读取原始OTS数据集")
    print(f"2. 筛选出2000对训练图片（多种雾参数）")
    print(f"3. 筛选出500对验证图片（8种雾参数均匀分布）")
    print(f"4. 保存到输出目录")
    
    input("\n按Enter继续...")
    
    # 执行筛选
    recreate_ots_dataset(
        hazy_dir=HAZY_DIR,
        clear_dir=CLEAR_DIR,
        output_dir=OUTPUT_DIR,
        train_size=2000,
        val_size=500,
        min_size=256
    )
    
    print("\n💡 下一步:")
    print("1. 将新的 RESIDE/OTS 文件夹上传到服务器")
    print("2. 替换服务器上的 /root/sj-tmp/DEA-Net-main/dataset/RESIDE/OTS")
    print("3. 重新训练模型")
    print("\n预期结果:")
    print("  - PSNR: 32-35 dB（合理范围）")
    print("  - 验证集包含多种雾参数，性能更可靠")
    print("  - 适合进行轻量化消融实验")
