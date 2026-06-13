"""
训练原始 YOLOv11n（不带 FCA）作为 baseline 对比
"""

from ultralytics import YOLO

print("="*70)
print("🚀 训练 YOLOv11n Baseline（不带FCA）")
print("="*70)
print("📦 模型: yolo11n.pt (原始)")
print("📊 数据集: COCO")
print("🔢 Epochs: 100")
print("📦 Batch: 32")
print("💻 Device: 0")
print("="*70 + "\n")

# 加载原始 YOLOv11n 模型（不注入 FCA）
model = YOLO('yolo11n.pt')

# 训练参数（与 FCA 版本保持一致，确保公平对比）
results = model.train(
    data='coco_fixed.yaml',
    epochs=100,
    batch=32,
    imgsz=640,
    device=0,
    patience=20,           # 早停
    lr0=0.001,             # 初始学习率
    weight_decay=0.001,    # 权重衰减
    project='checkpoints/yolov11_baseline',  # 保存到不同目录
    name='train',
    exist_ok=True,
    pretrained=True,
    verbose=True,
    plots=True,
)

print("\n" + "="*70)
print("🎉 Baseline 训练完成！")
print("="*70)
print(f"✓ 最佳模型: checkpoints/yolov11_baseline/train/weights/best.pt")
print(f"✓ 最新模型: checkpoints/yolov11_baseline/train/weights/last.pt")
print("\n📊 对比两个模型:")
print("  - YOLOv11n-FCA: checkpoints/yolov11_fca/train/weights/best.pt")
print("  - YOLOv11n (baseline): checkpoints/yolov11_baseline/train/weights/best.pt")
print("="*70)
