"""
简化的 YOLOv11-FCA 训练脚本
直接指定数据集路径，避免路径查找问题
"""

import sys
import os
from pathlib import Path

# 添加模型路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ultralytics import YOLO
from models.detection.fca_attention import FCAModule

try:
    from ultralytics.nn.modules import C3k2, C2f
except:
    from ultralytics.nn.modules import C2f
    C3k2 = None


def inject_fca_into_model(model, reduction=16):
    """将FCA模块注入到YOLOv11模型中"""
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
    print("🚀 YOLOv11-FCA 简化训练程序")
    print("="*70)
    
    # 数据集配置（直接在代码中指定）
    data_config = {
        'path': '/root/sj-tmp/YOLOv11-FCA/training/data/datasets/coco',
        'train': 'train2017',
        'val': 'val2017',
        'nc': 80,
        'names': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
                  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
                  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    }
    
    # 保存临时配置文件
    import yaml
    temp_yaml = Path('temp_coco.yaml')
    with open(temp_yaml, 'w') as f:
        yaml.dump(data_config, f)
    
    print(f"📊 数据集: {data_config['path']}")
    print(f"📦 模型: yolo11n.pt")
    print(f"🔢 Epochs: 100")
    print(f"📦 Batch: 32")
    print("="*70 + "\n")
    
    # 加载模型
    print("📦 加载YOLOv11模型...")
    model = YOLO('yolo11n.pt')
    print("✓ 模型加载成功\n")
    
    # 注入FCA
    model = inject_fca_into_model(model, reduction=16)
    
    # 训练参数
    train_args = {
        'data': str(temp_yaml),
        'epochs': 100,
        'batch': 32,
        'imgsz': 640,
        'device': 0,
        'patience': 20,
        'lr0': 0.001,
        'weight_decay': 0.001,
        'project': 'checkpoints/yolov11_fca',
        'name': 'train',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
        'plots': True,
    }
    
    # 开始训练
    print("="*70)
    print("🎯 开始训练YOLOv11-FCA...")
    print("="*70 + "\n")
    
    try:
        results = model.train(**train_args)
        
        print("\n" + "="*70)
        print("🎉 训练完成！")
        print("="*70)
        print(f"✓ 最佳模型: checkpoints/yolov11_fca/train/weights/best.pt")
        print(f"✓ 最新模型: checkpoints/yolov11_fca/train/weights/last.pt")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时文件
        if temp_yaml.exists():
            temp_yaml.unlink()


if __name__ == '__main__':
    main()
