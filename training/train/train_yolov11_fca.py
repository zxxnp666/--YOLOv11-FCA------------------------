"""
YOLOv11-FCA 训练脚本（中等方案）
使用DEA-Net去雾后的数据训练YOLOv11-FCA
"""

import argparse
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv11-FCA')
    parser.add_argument('--config', type=str, default='config/yolov11_fca_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--data', type=str, default='data/datasets/traffic/data.yaml',
                       help='数据集配置文件')
    parser.add_argument('--weights', type=str, default=None,
                       help='预训练权重路径')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch', type=int, default=None,
                       help='批量大小（覆盖配置文件）')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='图像大小')
    parser.add_argument('--device', type=str, default='0',
                       help='训练设备')
    parser.add_argument('--resume', action='store_true',
                       help='恢复训练')
    
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_yolov11_fca(args, config):
    
    if not ULTRALYTICS_AVAILABLE:
        print("Error: ultralytics not installed!")
        print("Install with: pip install ultralytics")
        return
    
    # 初始化模型
    if args.weights:
        print(f"Loading weights from {args.weights}")
        model = YOLO(args.weights)
    else:
        print("Initializing YOLOv11 model...")
        model = YOLO('yolov11n.yaml')  # 或其他YOLOv11变体
    
    # 训练参数
    train_args = {
        'data': args.data,
        'epochs': args.epochs or config['training']['epochs'],
        'batch': args.batch or config['training']['batch_size'],
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': config['training']['workers'],
        'patience': config['training'].get('patience', 50),  # 早停参数
        'optimizer': config['training']['optimizer'],
        'lr0': config['training']['lr0'],
        'lrf': config['training']['lrf'],
        'momentum': config['training']['momentum'],
        'weight_decay': config['training']['weight_decay'],
        'warmup_epochs': config['training']['warmup_epochs'],
        'mosaic': config['augmentation']['mosaic'],
        'mixup': config['augmentation']['mixup'],
        'hsv_h': config['augmentation']['hsv_h'],
        'hsv_s': config['augmentation']['hsv_s'],
        'hsv_v': config['augmentation']['hsv_v'],
        'degrees': config['augmentation']['degrees'],
        'translate': config['augmentation']['translate'],
        'scale': config['augmentation']['scale'],
        'fliplr': config['augmentation']['fliplr'],
        'project': config['save']['dir'],
        'name': 'train',
        'exist_ok': True,
        'resume': args.resume,
    }
    
    # 开始训练
    print("\n" + "="*60)
    print("Starting YOLOv11-FCA Training...")
    print("="*60)
    print(f"Model: YOLOv11-FCA")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {train_args['epochs']}")
    print(f"Batch size: {train_args['batch']}")
    print(f"Device: {args.device}")
    print("="*60 + "\n")
    
    results = model.train(**train_args)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    return results


def main():
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 训练模型
    train_yolov11_fca(args, config)


if __name__ == '__main__':
    main()
