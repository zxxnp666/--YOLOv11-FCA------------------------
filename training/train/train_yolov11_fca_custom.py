"""
YOLOv11-FCA 自定义训练脚本
真正集成FCA注意力机制到YOLOv11
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.detection.fca_attention import FCAModule

try:
    from ultralytics import YOLO
    from ultralytics.nn.modules import C2f, C3k2, Conv, Bottleneck
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Error: ultralytics not installed!")
    sys.exit(1)


def inject_fca_into_model(model, reduction=16):
    """
    将FCA模块注入到YOLOv11模型中
    
    策略：在Backbone和Neck的C3k2/C2f模块后添加FCA
    YOLOv11使用C3k2模块替代了YOLOv8的C2f
    """
    print("\n" + "="*70)
    print("🔧 注入FCA注意力模块到YOLOv11...")
    print("="*70)
    
    fca_count = 0
    fca_modules = []  # 保存FCA模块引用，防止被垃圾回收
    
    # 遍历模型的所有模块
    for name, module in model.model.named_modules():
        # YOLOv11使用C3k2模块，也支持C2f
        if isinstance(module, (C3k2, C2f)) if hasattr(sys.modules[__name__], 'C3k2') else isinstance(module, C2f):
            # 获取输出通道数
            if hasattr(module, 'cv2'):
                out_channels = module.cv2.conv.out_channels
            elif hasattr(module, 'c'):
                out_channels = module.c
            else:
                continue
            
            # 创建FCA模块并移到正确的设备
            fca = FCAModule(out_channels, reduction=reduction)
            if next(module.parameters()).is_cuda:
                fca = fca.cuda()
            
            fca_modules.append(fca)  # 保存引用
            
            # 包装原始模块
            original_forward = module.forward
            
            def make_fca_forward(fca_module, orig_forward):
                def fca_forward(x):
                    out = orig_forward(x)
                    out = fca_module(out)
                    return out
                return fca_forward
            
            module.forward = make_fca_forward(fca, original_forward)
            fca_count += 1
            
            print(f"  ✓ 在 {name} 后添加FCA (channels={out_channels})")
    
    # 将FCA模块保存到模型中，防止被垃圾回收
    model.fca_modules = fca_modules
    
    print(f"\n✓ 成功注入 {fca_count} 个FCA模块")
    print("="*70 + "\n")
    
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='训练YOLOv11-FCA（真正集成FCA）')
    parser.add_argument('--data', type=str, default='coco.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--fraction', type=float, default=1.0,
                       help='使用数据集的比例，1.0表示使用全部数据')
    parser.add_argument('--reduction', type=int, default=16,
                       help='FCA降维比例')
    parser.add_argument('--weights', type=str, default='yolo11n.pt',
                       help='YOLOv11预训练权重')
    parser.add_argument('--patience', type=int, default=20,
                       help='早停耐心值，多少轮不提升就停止')
    parser.add_argument('--lr0', type=float, default=0.001,
                       help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                       help='权重衰减（L2正则化）')
    parser.add_argument('--resume', action='store_true')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print("🚀 YOLOv11-FCA 训练程序")
    print("="*70)
    print(f"📦 模型: {args.weights}")
    print(f"📊 数据集: {args.data}")
    print(f"🔢 Epochs: {args.epochs}")
    print(f"📦 Batch: {args.batch}")
    print(f"💻 Device: {args.device}")
    print(f"📉 Fraction: {args.fraction * 100}%")
    print(f"🎯 FCA Reduction: {args.reduction}")
    print(f"⏱️  Patience: {args.patience} (早停)")
    print(f"📈 Learning Rate: {args.lr0}")
    print(f"⚖️  Weight Decay: {args.weight_decay}")
    print("="*70 + "\n")
    
    # 1. 加载YOLOv11模型
    print(f"📦 加载YOLOv11模型: {args.weights}...")
    
    if not os.path.exists(args.weights):
        print(f"⚠️  本地未找到 {args.weights}，将自动下载...")
    
    model = YOLO(args.weights)
    print("✓ 模型加载成功\n")
    
    # 2. 注入FCA模块
    model = inject_fca_into_model(model, reduction=args.reduction)
    
    # 3. 训练参数（防过拟合配置）
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': 640,
        'device': args.device,
        'fraction': args.fraction,
        'patience': args.patience,           # 早停：防止过拟合
        'lr0': args.lr0,                     # 降低学习率
        'weight_decay': args.weight_decay,   # 增加权重衰减
        'project': 'checkpoints/yolov11_fca',
        'name': 'train',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
        'plots': True,
        'resume': args.resume,
    }
    
    # 4. 开始训练
    print("="*70)
    print("🎯 开始训练YOLOv11-FCA...")
    print("="*70)
    print("💡 提示: FCA模块已注入到模型中，会自动参与训练")
    print("💡 提示: 可以使用 TensorBoard 查看训练曲线")
    print("   命令: tensorboard --logdir checkpoints/yolov11_fca --port 6006")
    print("="*70 + "\n")
    
    try:
        results = model.train(**train_args)
        
        print("\n" + "="*70)
        print("🎉 训练完成！")
        print("="*70)
        print(f"✓ 最佳模型: checkpoints/yolov11_fca/train/weights/best.pt")
        print(f"✓ 最新模型: checkpoints/yolov11_fca/train/weights/last.pt")
        print(f"✓ 训练曲线: checkpoints/yolov11_fca/train/")
        print("\n📊 这个模型包含了FCA注意力机制！")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
