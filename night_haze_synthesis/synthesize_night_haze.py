"""
夜景多浓度雾图合成脚本
一键生成训练数据集

使用方法：
python synthesize_night_haze.py --input_dir night_clear --output_dir night_dataset

输入：清晰的夜景图片
输出：多浓度雾图 + 对应的清晰图
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import random


def add_haze(clear_image, t_value, A_value=None):
    """
    使用大气散射模型添加雾
    
    I(x) = J(x) * t(x) + A * (1 - t(x))
    
    Args:
        clear_image: 清晰图像 (numpy array, 0-255)
        t_value: 透射率 (0-1)，越小雾越浓
        A_value: 大气光值 (0-255)，如果为None则自动计算
    
    Returns:
        hazy_image: 有雾图像
    """
    # 归一化到 0-1
    clear = clear_image.astype(np.float32) / 255.0
    
    # 计算大气光 A（取图像中最亮的区域）
    if A_value is None:
        # 对于夜景，大气光通常较暗
        # 取图像中前10%最亮像素的平均值
        gray = cv2.cvtColor((clear * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        flat = gray.flatten()
        flat.sort()
        threshold = flat[int(len(flat) * 0.9)]
        mask = gray >= threshold
        
        A = np.zeros(3)
        for i in range(3):
            A[i] = np.mean(clear[:,:,i][mask])
        
        # 夜景大气光通常较暗，限制在0.3-0.6范围
        A = np.clip(A, 0.3, 0.6)
    else:
        A = np.array([A_value, A_value, A_value]) / 255.0
    
    # 生成透射率图（可以添加一些空间变化）
    h, w = clear.shape[:2]
    
    # 基础透射率
    t = np.ones((h, w)) * t_value
    
    # 添加一些空间变化（模拟不均匀雾）
    if random.random() < 0.5:
        # 生成渐变雾（从上到下或从左到右）
        if random.random() < 0.5:
            # 垂直渐变
            gradient = np.linspace(0.8, 1.2, h)
            t = t * gradient[:, np.newaxis]
        else:
            # 水平渐变
            gradient = np.linspace(0.8, 1.2, w)
            t = t * gradient[np.newaxis, :]
    
    # 限制透射率范围
    t = np.clip(t, 0.1, 1.0)
    
    # 扩展到3通道
    t = np.stack([t, t, t], axis=2)
    
    # 应用大气散射模型
    hazy = clear * t + A * (1 - t)
    
    # 转换回 0-255
    hazy = np.clip(hazy * 255, 0, 255).astype(np.uint8)
    
    return hazy


def process_single_image(clear_path, output_clear_dir, output_hazy_dir, 
                        density_levels, add_noise=True):
    """
    处理单张图片，生成多个浓度版本
    
    Args:
        clear_path: 清晰图路径
        output_clear_dir: 输出清晰图目录
        output_hazy_dir: 输出雾图目录
        density_levels: 浓度等级列表 [(name, t_min, t_max), ...]
        add_noise: 是否添加噪声（夜景通常噪声较多）
    """
    # 读取图像
    clear_img = cv2.imread(str(clear_path))
    if clear_img is None:
        print(f"⚠️  无法读取图像: {clear_path}")
        return 0
    
    # 获取文件名（不含扩展名）
    filename = Path(clear_path).stem
    ext = Path(clear_path).suffix
    
    generated_count = 0
    
    # 为每个浓度等级生成雾图
    for density_name, t_min, t_max in density_levels:
        # 随机选择透射率
        t = random.uniform(t_min, t_max)
        
        # 生成雾图
        hazy_img = add_haze(clear_img, t)
        
        # 可选：添加噪声（模拟夜景噪声）
        if add_noise:
            noise_level = random.uniform(5, 15)
            noise = np.random.normal(0, noise_level, hazy_img.shape)
            hazy_img = np.clip(hazy_img + noise, 0, 255).astype(np.uint8)
        
        # 保存文件
        output_filename = f"{filename}_{density_name}_t{t:.2f}{ext}"
        
        # 保存清晰图（复制）
        clear_output_path = os.path.join(output_clear_dir, output_filename)
        cv2.imwrite(clear_output_path, clear_img)
        
        # 保存雾图
        hazy_output_path = os.path.join(output_hazy_dir, output_filename)
        cv2.imwrite(hazy_output_path, hazy_img)
        
        generated_count += 1
    
    return generated_count


def synthesize_dataset(input_dir, output_dir, density_config='default', 
                      add_noise=True, max_images=None):
    """
    批量合成数据集
    
    Args:
        input_dir: 输入清晰图目录
        output_dir: 输出数据集目录
        density_config: 浓度配置 ('default', 'light', 'heavy', 'all')
        add_noise: 是否添加噪声
        max_images: 最多处理图片数（None表示全部）
    """
    print("="*80)
    print("夜景多浓度雾图合成系统".center(80))
    print("="*80)
    
    # 创建输出目录
    output_clear_dir = os.path.join(output_dir, 'clear')
    output_hazy_dir = os.path.join(output_dir, 'hazy')
    os.makedirs(output_clear_dir, exist_ok=True)
    os.makedirs(output_hazy_dir, exist_ok=True)
    
    # 浓度配置
    density_configs = {
        'default': [
            ('light', 0.6, 0.8),    # 轻雾
            ('medium', 0.4, 0.6),   # 中雾
            ('heavy', 0.2, 0.4),    # 浓雾
        ],
        'light': [
            ('light', 0.7, 0.9),
        ],
        'heavy': [
            ('heavy', 0.1, 0.3),
        ],
        'all': [
            ('verylight', 0.8, 0.95),
            ('light', 0.6, 0.8),
            ('medium', 0.4, 0.6),
            ('heavy', 0.2, 0.4),
            ('veryheavy', 0.05, 0.2),
        ]
    }
    
    density_levels = density_configs.get(density_config, density_configs['default'])
    
    print(f"\n📂 输入目录: {input_dir}")
    print(f"📂 输出目录: {output_dir}")
    print(f"🌫️  浓度配置: {density_config}")
    print(f"   浓度等级: {len(density_levels)} 个")
    for name, t_min, t_max in density_levels:
        print(f"      - {name}: t={t_min:.2f}-{t_max:.2f}")
    print(f"🔊 添加噪声: {'是' if add_noise else '否'}")
    print()
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"📷 找到 {len(image_files)} 张图片")
    print(f"📊 预计生成 {len(image_files) * len(density_levels)} 对图片\n")
    
    if len(image_files) == 0:
        print("❌ 没有找到图片文件！")
        print(f"   请确保 {input_dir} 目录中有图片")
        return
    
    # 处理每张图片
    total_generated = 0
    print("🚀 开始合成...\n")
    
    for img_path in tqdm(image_files, desc="合成进度"):
        count = process_single_image(
            img_path, 
            output_clear_dir, 
            output_hazy_dir,
            density_levels,
            add_noise
        )
        total_generated += count
    
    print(f"\n{'='*80}")
    print("✅ 合成完成！".center(80))
    print(f"{'='*80}")
    print(f"📊 统计信息:")
    print(f"   输入图片: {len(image_files)} 张")
    print(f"   生成图片对: {total_generated} 对")
    print(f"   清晰图目录: {output_clear_dir}")
    print(f"   雾图目录: {output_hazy_dir}")
    print(f"{'='*80}\n")
    
    # 显示示例
    print("📝 数据集结构:")
    print(f"{output_dir}/")
    print(f"├── clear/")
    print(f"│   ├── image1_light_t0.75.jpg")
    print(f"│   ├── image1_medium_t0.52.jpg")
    print(f"│   ├── image1_heavy_t0.28.jpg")
    print(f"│   └── ...")
    print(f"└── hazy/")
    print(f"    ├── image1_light_t0.75.jpg")
    print(f"    ├── image1_medium_t0.52.jpg")
    print(f"    ├── image1_heavy_t0.28.jpg")
    print(f"    └── ...")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='夜景多浓度雾图合成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基础用法（生成轻、中、浓三种雾）
  python synthesize_night_haze.py --input_dir night_clear --output_dir night_dataset
  
  # 只生成浓雾
  python synthesize_night_haze.py --input_dir night_clear --output_dir night_heavy --density heavy
  
  # 生成所有浓度等级（5种）
  python synthesize_night_haze.py --input_dir night_clear --output_dir night_all --density all
  
  # 不添加噪声
  python synthesize_night_haze.py --input_dir night_clear --output_dir night_dataset --no_noise
  
  # 只处理前100张图片
  python synthesize_night_haze.py --input_dir night_clear --output_dir night_dataset --max_images 100
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入清晰夜景图片目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出数据集目录')
    parser.add_argument('--density', type=str, default='default',
                       choices=['default', 'light', 'heavy', 'all'],
                       help='浓度配置: default(轻中浓), light(只轻雾), heavy(只浓雾), all(5种浓度)')
    parser.add_argument('--no_noise', action='store_true',
                       help='不添加噪声')
    parser.add_argument('--max_images', type=int, default=None,
                       help='最多处理图片数量')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 错误: 输入目录不存在: {args.input_dir}")
        return
    
    # 合成数据集
    synthesize_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        density_config=args.density,
        add_noise=not args.no_noise,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()
