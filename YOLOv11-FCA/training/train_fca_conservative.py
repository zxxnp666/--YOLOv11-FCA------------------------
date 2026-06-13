"""
保守优化版 YOLOv11-FCA
只做最小改动：稍微增加训练轮数
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


def inject_fca_into_model(model, reduction=16):
    """将FCA模块注入到YOLOv11模型中（所有层）"""
    print("\n" + "="*70)
    print("🔧 注入FCA注意力模块到YOLOv11...")
    print("="*70)
    
    fca_count = 0
    fca_modules = []
    
    for name, module in model.model.named_modules():
        if C3k2 and isinstance(module, C3k2):
            target_module = module
        elif isinstance(module, C2f):
            target_module = module
        else:
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
        fca_count += 1
        
        print(f"  ✓ 在 {name} 后添加FCA (channels={out_channels})")
    
    model.fca_modules = fca_modules
    
    print(f"\n✓ 成功注入 {fca_count} 个FCA模块")
    print("="*70 + "\n")
    
    return model


def main():
    print("="*70)
    print("🚀 YOLOv11-FCA 保守优化训练")
    print("="*70)
    print("📦 模型: yolo11n.pt")
    print("📊 数据集: COCO")
    print("🔢 Epochs: 120 (稍微增加)")
    print("📦 Batch: 32")
    print("📈 Learning Rate: 0.001 (保持不变)")
    print("⚖️  Weight Decay: 0.001 (保持不变)")
    print("🎯 FCA Reduction: 16 (保持不变)")
    print("="*70 + "\n")
    
    # 加载模型
    model = YOLO('yolo11n.pt')
    
    # 注入 FCA（使用原版配置）
    model = inject_fca_into_model(model, reduction=16)
    
    # 保守的训练参数（只增加轮数）
    results = model.train(
        data='coco_fixed.yaml',
        epochs=120,              # 只增加 20 轮
        batch=32,
        imgsz=640,
        device=0,
        patience=25,             # 稍微增加耐心值
        lr0=0.001,               # 保持原来的学习率
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.001,      # 保持原来的权重衰减
        warmup_epochs=3,
        project='runs/detect/checkpoints/yolov11_fca_conservative',
        name='train',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        plots=True,
    )
    
    print("\n" + "="*70)
    print("🎉 保守优化训练完成！")
    print("="*70)
    print(f"✓ 最佳模型: runs/detect/checkpoints/yolov11_fca_conservative/train/weights/best.pt")
    print("="*70)


if __name__ == '__main__':
    main()
