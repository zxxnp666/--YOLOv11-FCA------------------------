"""
显示训练的最终结果
"""

import pandas as pd
from pathlib import Path

results_file = Path('runs/detect/checkpoints/yolov11_fca/train/results.csv')

if not results_file.exists():
    print("❌ 结果文件不存在")
    exit(1)

# 读取结果
df = pd.read_csv(results_file)
df.columns = df.columns.str.strip()

# 获取最后一行（最终结果）
final = df.iloc[-1]

print("="*70)
print("🎯 YOLOv11-FCA 最终训练结果")
print("="*70)
print(f"\n📊 性能指标:")
print(f"  mAP50:        {final['metrics/mAP50(B)']:.4f} ({final['metrics/mAP50(B)']*100:.2f}%)")
print(f"  mAP50-95:     {final['metrics/mAP50-95(B)']:.4f} ({final['metrics/mAP50-95(B)']*100:.2f}%)")
print(f"  Precision:    {final['metrics/precision(B)']:.4f} ({final['metrics/precision(B)']*100:.2f}%)")
print(f"  Recall:       {final['metrics/recall(B)']:.4f} ({final['metrics/recall(B)']*100:.2f}%)")

print(f"\n📉 损失值:")
print(f"  Box Loss:     {final['train/box_loss']:.4f}")
print(f"  Cls Loss:     {final['train/cls_loss']:.4f}")
print(f"  DFL Loss:     {final['train/dfl_loss']:.4f}")

print(f"\n⏱️  训练信息:")
print(f"  Epoch:        {int(final['epoch'])}/100")

print("\n" + "="*70)

# 找到最佳 mAP50-95 的轮次
best_idx = df['metrics/mAP50-95(B)'].idxmax()
best = df.iloc[best_idx]

print(f"🏆 最佳性能 (Epoch {int(best['epoch'])}):")
print(f"  mAP50:        {best['metrics/mAP50(B)']:.4f} ({best['metrics/mAP50(B)']*100:.2f}%)")
print(f"  mAP50-95:     {best['metrics/mAP50-95(B)']:.4f} ({best['metrics/mAP50-95(B)']*100:.2f}%)")
print(f"  Precision:    {best['metrics/precision(B)']:.4f} ({best['metrics/precision(B)']*100:.2f}%)")
print(f"  Recall:       {best['metrics/recall(B)']:.4f} ({best['metrics/recall(B)']*100:.2f}%)")

print("="*70)
