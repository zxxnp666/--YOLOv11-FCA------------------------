"""
测试YOLOv11-FCA模型
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='测试YOLOv11-FCA模型')
    parser.add_argument('--weights', type=str, required=True,
                       help='模型权重路径，例如：checkpoints/yolov11_fca/train/weights/best.pt')
    parser.add_argument('--data', type=str, default='coco.yaml',
                       help='数据集配置文件')
    parser.add_argument('--batch', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='0',
                       help='GPU设备ID')
    parser.add_argument('--conf', type=float, default=0.001,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='NMS IoU阈值')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print("🧪 YOLOv11-FCA 模型测试")
    print("="*70)
    print(f"📦 模型: {args.weights}")
    print(f"📊 数据集: {args.data}")
    print(f"📦 Batch: {args.batch}")
    print(f"💻 Device: {args.device}")
    print(f"🎯 Conf: {args.conf}")
    print(f"🎯 IoU: {args.iou}")
    print("="*70 + "\n")
    
    # 加载模型
    print(f"📦 加载模型: {args.weights}...")
    model = YOLO(args.weights)
    print("✓ 模型加载成功\n")
    
    # 验证模型
    print("="*70)
    print("🎯 开始验证...")
    print("="*70 + "\n")
    
    results = model.val(
        data=args.data,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("🎉 验证完成！")
    print("="*70)
    print(f"📊 mAP50: {results.box.map50:.4f}")
    print(f"📊 mAP50-95: {results.box.map:.4f}")
    print("="*70)


if __name__ == '__main__':
    main()
