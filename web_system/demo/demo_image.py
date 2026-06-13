"""
单张图像演示程序
展示完整的雾天交通目标检测流程
"""

import argparse
import cv2
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.fusion.end_to_end_model import FoggyTrafficSystem


def parse_args():
    parser = argparse.ArgumentParser(description='Foggy Traffic Detection Demo')
    parser.add_argument('--input', type=str, required=True,
                       help='输入雾天图像路径')
    parser.add_argument('--output', type=str, default='result.jpg',
                       help='输出结果路径')
    parser.add_argument('--deanet_weights', type=str,
                       default='checkpoints/deanet/ITS/PSNR4131_SSIM9945.pth',
                       help='DEA-Net权重路径')
    parser.add_argument('--yolo_weights', type=str,
                       default='checkpoints/yolov11_fca/best.pt',
                       help='YOLOv11-FCA权重路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='运行设备')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--save_dehazed', action='store_true',
                       help='保存去雾后的图像')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    print("="*60)
    print("Foggy Traffic Object Detection Demo")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # 初始化系统
    print("\nInitializing system...")
    try:
        system = FoggyTrafficSystem(
            deanet_weights=args.deanet_weights,
            yolov11_weights=args.yolo_weights,
            device=args.device
        )
        print("✓ System initialized successfully!")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        print("\nTips:")
        print("1. Make sure you have downloaded the pretrained weights")
        print("2. Check the weights paths in the arguments")
        return
    
    # 读取图像
    print(f"\nReading image: {args.input}")
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Failed to read image: {args.input}")
        return
    
    print(f"Image size: {image.shape}")
    
    # 处理图像
    print("\nProcessing...")
    print("  Step 1: Dehazing with DEA-Net...")
    print("  Step 2: Detecting with YOLOv11-FCA...")
    
    try:
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 端到端预测
        detections, dehazed = system.predict(image_rgb)
        
        print("✓ Processing completed!")
        
        # 保存去雾后的图像
        if args.save_dehazed:
            dehazed_path = args.output.replace('.jpg', '_dehazed.jpg')
            dehazed_bgr = cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dehazed_path, dehazed_bgr)
            print(f"✓ Dehazed image saved to: {dehazed_path}")
        
        # 绘制检测结果
        # TODO: 实现检测结果可视化
        
        # 保存结果
        output_image = image.copy()  # 这里应该绘制检测框
        cv2.imwrite(args.output, output_image)
        print(f"✓ Result saved to: {args.output}")
        
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
