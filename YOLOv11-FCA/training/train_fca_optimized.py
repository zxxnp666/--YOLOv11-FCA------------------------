"""
优化版 YOLOv11-FCA 训练
- 更长的训练轮数
- 调整超参数
- 只在关键位置添加 FCA
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ultralytics import YOLO
from models.detection.fca_attention import FCAModule

try:
    from ultralytics.nn.modules import C3k2, C2f
except:
    from ultralytics.nn.modules import C2f
    C3k2 = None


def inject_fca_selective(model, reduction=16):
    """
    选择性注入 FCA：只在 Backbone 的深层和 Neck 中添加
    """
    print("\n" + "="*70)
    print("🔧 选择性注入 FCA 模块（优化版）")
    print("="*70)
    
    fca_count = 0
    fca_modules = []
    
    # 只在特定层添加 FCA
    target_layers = []
    
    for name, module in model.model.named_modules():
        if C3k2 and isinstance(module, C3k2):
            target_module = module
        elif isinstance(module, C2f):
            target_module = module
        else:
            continue
        
        # 只在 Backbone 的后半部分和 Neck 中添加 FCA
        # 跳过浅层特征（前2层）
        if fca_count < 2:
            fca_count += 1
            print(f"  ⊗ 跳过 {name} (浅层特征)")
            continue
        
        if hasattr(target_module, 'cv2'):
            out_channels = target_module.cv2.conv.out_channels
        elif hasattr(target_module, 'c'):
            out_channels = target_module.c
        else:
            continue
        
        fca = FCAModule(out_channels, reduction=reduction)
        if next(target_module.parameters()).is_cuda:
            fca = fca.cuda()
        
        fca_modules.append(fca)
        
        original_forward = target_module.forward
        
        def make_fca_forward(fca_module, orig_forward):
            def fca_forward(x):
                out = orig_forward(x)
                out = fca_module(out)
                return out
            return fca_forward
        
        target_module.forward = make_fca_forward(fca, original_forward)
        target_layers.append(name)
        
        print(f"  ✓ 在 {name} 后添加 FCA (channels={out_channels})")
    
    model.fca_modules = fca_modules
    
    print(f"\n✓ 成功注入 {len(fca_modules)} 个 FCA 模块")
    print("="*70 + "\n")
    
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description='YOLOv11-FCA 优化训练')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch', type=int, default=32, help='批次大小')
    parser.add_argument('--lr0', type=float, default=0.002, help='初始学习率')
    parser.add_argument('--reduction', type=int, default=8, help='FCA reduction')
    parser.add_argument('--device', type=str, default='0', help='GPU设备')
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 YOLOv11-FCA 优化训练")
    print("="*70)
    print("📦 模型: yolo11n.pt")
    print("📊 数据集: COCO")
    print(f"🔢 Epochs: {args.epochs}")
    print(f"📦 Batch: {args.batch}")
    print(f"📈 Learning Rate: {args.lr0}")
    print("⚖️  Weight Decay: 0.0005 (更小)")
    print(f"🎯 FCA Reduction: {args.reduction}")
    print("="*70 + "\n")
    
    # 加载模型
    model = YOLO('yolo11n.pt')
    
    # 选择性注入 FCA
    model = inject_fca_selective(model, reduction=args.reduction)
    
    # 优化的训练参数
    results = model.train(
        data='coco_fixed.yaml',
        epochs=args.epochs,      # 可配置
        batch=args.batch,        # 可配置
        imgsz=640,
        device=args.device,      # 可配置
        patience=30,             # 更大的耐心值
        lr0=args.lr0,            # 可配置
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,     # 更小的权重衰减
        warmup_epochs=5,         # 更长的预热
        project='checkpoints/yolov11_fca_optimized',
        name='train',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        plots=True,
    )
    
    print("\n" + "="*70)
    print("🎉 优化训练完成！")
    print("="*70)
    print(f"✓ 最佳模型: checkpoints/yolov11_fca_optimized/train/weights/best.pt")
    print("="*70)


if __name__ == '__main__':
    main()
