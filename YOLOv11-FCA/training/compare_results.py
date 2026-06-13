"""
对比 YOLOv11n-FCA 和 YOLOv11n Baseline 的性能
"""

import pandas as pd
from pathlib import Path

print("="*70)
print("📊 YOLOv11n-FCA vs YOLOv11n Baseline 性能对比")
print("="*70 + "\n")

# 读取训练结果
fca_results = Path('checkpoints/yolov11_fca/train/results.csv')
baseline_results = Path('checkpoints/yolov11_baseline/train/results.csv')

if not fca_results.exists():
    print("❌ FCA 训练结果不存在")
    exit(1)

if not baseline_results.exists():
    print("❌ Baseline 训练结果不存在")
    exit(1)

# 读取 CSV
df_fca = pd.read_csv(fca_results)
df_baseline = pd.read_csv(baseline_results)

# 去除列名中的空格
df_fca.columns = df_fca.columns.str.strip()
df_baseline.columns = df_baseline.columns.str.strip()

# 获取最佳结果（最后一行通常是最佳的）
fca_best = df_fca.iloc[-1]
baseline_best = df_baseline.iloc[-1]

print("📈 最终性能对比:\n")

# 对比表格
metrics = {
    'metrics/mAP50(B)': 'mAP50',
    'metrics/mAP50-95(B)': 'mAP50-95',
    'metrics/precision(B)': 'Precision',
    'metrics/recall(B)': 'Recall',
    'train/box_loss': 'Box Loss',
    'train/cls_loss': 'Cls Loss',
}

print(f"{'指标':<20} {'YOLOv11n-FCA':<15} {'YOLOv11n (Baseline)':<20} {'差异':<10}")
print("-" * 70)

for key, name in metrics.items():
    if key in df_fca.columns and key in df_baseline.columns:
        fca_val = fca_best[key]
        baseline_val = baseline_best[key]
        diff = fca_val - baseline_val
        diff_str = f"{diff:+.4f}" if 'loss' not in key.lower() else f"{diff:+.4f}"
        
        print(f"{name:<20} {fca_val:<15.4f} {baseline_val:<20.4f} {diff_str:<10}")

print("\n" + "="*70)
print("📝 结论:")
print("="*70)

# 计算 mAP50-95 的差异
if 'metrics/mAP50-95(B)' in df_fca.columns:
    fca_map = fca_best['metrics/mAP50-95(B)']
    baseline_map = baseline_best['metrics/mAP50-95(B)']
    diff_percent = ((fca_map - baseline_map) / baseline_map) * 100
    
    if diff_percent > 0:
        print(f"✅ YOLOv11n-FCA 比 Baseline 高 {diff_percent:.2f}%")
        print("   FCA 注意力机制有效提升了模型性能！")
    elif diff_percent > -5:
        print(f"⚖️  YOLOv11n-FCA 比 Baseline 低 {abs(diff_percent):.2f}%")
        print("   性能接近，FCA 模块在保持轻量级的同时维持了性能。")
    else:
        print(f"⚠️  YOLOv11n-FCA 比 Baseline 低 {abs(diff_percent):.2f}%")
        print("   可能需要调整 FCA 的超参数或位置。")

print("="*70)

# 保存对比结果
comparison = pd.DataFrame({
    'Metric': list(metrics.values()),
    'YOLOv11n-FCA': [fca_best.get(k, 0) for k in metrics.keys()],
    'YOLOv11n-Baseline': [baseline_best.get(k, 0) for k in metrics.keys()],
})
comparison['Difference'] = comparison['YOLOv11n-FCA'] - comparison['YOLOv11n-Baseline']
comparison.to_csv('model_comparison.csv', index=False)

print(f"\n✓ 对比结果已保存到: model_comparison.csv")
