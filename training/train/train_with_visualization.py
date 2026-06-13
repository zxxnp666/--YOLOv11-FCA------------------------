"""
带可视化的训练脚本
实时显示训练进度、损失曲线、性能指标
"""

import argparse
import yaml
import os
import sys
from datetime import datetime
from tqdm import tqdm
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed.")


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, log_dir='results/logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        # 打开日志文件
        self.log_fp = open(self.log_file, 'w', encoding='utf-8')
        
        print(f"📊 训练日志保存到: {self.log_file}")
        print(f"📊 TensorBoard日志: {log_dir}")
        print(f"📊 启动TensorBoard: tensorboard --logdir {log_dir} --port 6006")
        print("="*70)
        
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        self.log_fp.write(log_msg + '\n')
        self.log_fp.flush()
        
    def log_epoch(self, epoch, total_epochs, metrics):
        """记录每个epoch的信息"""
        self.log("="*70)
        self.log(f"📈 Epoch [{epoch}/{total_epochs}]")
        self.log("-"*70)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                self.log(f"  {key}: {value:.4f}")
            else:
                self.log(f"  {key}: {value}")
        
        self.log("="*70)
        
    def __del__(self):
        if hasattr(self, 'log_fp'):
            self.log_fp.close()


def parse_args():
    parser = argparse.ArgumentParser(description='训练YOLOv11（带可视化）')
    parser.add_argument('--config', type=str, default='config/yolov11_fca_config.yaml')
    parser.add_argument('--data', type=str, default='coco.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--weights', type=str, default='yolo11n.pt',
                       help='YOLOv11预训练权重')
    parser.add_argument('--fraction', type=float, default=0.1,
                       help='使用数据集的比例（0.1=10%）')
    
    return parser.parse_args()


def train_with_visualization(args):
    """带可视化的训练"""
    
    # 初始化可视化器
    visualizer = TrainingVisualizer()
    
    visualizer.log("🚀 开始训练YOLOv11（标准版，无FCA）")
    visualizer.log(f"⚙️  配置文件: {args.config}")
    visualizer.log(f"� 模型权:重: {args.weights}")
    visualizer.log(f"� 数p据集: {args.data}")
    visualizer.log(f"� Epoch s: {args.epochs}")
    visualizer.log(f"� Batch sizae: {args.batch}")
    visualizer.log(f"� Detvice: {args.device}")
    visualizer.log(f"📉 Data fraction: {args.fraction * 100}%")
    
    if not ULTRALYTICS_AVAILABLE:
        visualizer.log("❌ ultralytics未安装，无法训练")
        return
    
    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        visualizer.log(f"✓ 配置文件加载成功")
    else:
        visualizer.log(f"⚠️  配置文件不存在，使用默认配置")
        config = {}
    
    # 初始化模型
    visualizer.log(f"📦 初始化YOLOv11模型: {args.weights}...")
    # 使用 YOLOv11
    model = YOLO(args.weights)  # 使用命令行指定的权重
    visualizer.log("✓ YOLOv11模型初始化成功")
    
    # 训练参数
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': 640,
        'device': args.device,
        'fraction': args.fraction,
        'project': 'checkpoints/yolov11',  # 改为 yolov11（区分标准版和FCA版）
        'name': 'train',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
        'plots': True,  # 自动生成训练曲线图
    }
    
    visualizer.log("\n" + "="*70)
    visualizer.log("🎯 开始训练...")
    visualizer.log("="*70)
    visualizer.log(f"💡 提示: 可以打开TensorBoard查看实时训练曲线")
    visualizer.log(f"   命令: tensorboard --logdir results/logs --port 6006")
    visualizer.log("="*70 + "\n")
    
    try:
        # 开始训练
        results = model.train(**train_args)
        
        visualizer.log("\n" + "="*70)
        visualizer.log("🎉 YOLOv11（标准版）训练完成！")
        visualizer.log("="*70)
        visualizer.log(f"✓ 最佳模型保存在: checkpoints/yolov11/train/weights/best.pt")
        visualizer.log(f"✓ 最新模型保存在: checkpoints/yolov11/train/weights/last.pt")
        visualizer.log(f"✓ 训练曲线图: checkpoints/yolov11/train/")
        visualizer.log("="*70)
        
        # 显示最终指标
        if hasattr(results, 'results_dict'):
            visualizer.log("\n📊 最终训练结果:")
            for key, value in results.results_dict.items():
                visualizer.log(f"  {key}: {value}")
        
        return results
        
    except Exception as e:
        visualizer.log(f"\n❌ 训练出错: {str(e)}")
        import traceback
        visualizer.log(traceback.format_exc())
        raise


def main():
    args = parse_args()
    train_with_visualization(args)


if __name__ == '__main__':
    main()
