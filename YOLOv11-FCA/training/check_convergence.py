"""
检查训练是否已经收敛
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_file = Path('runs/detect/checkpoints/yolov11_fca/train/results.csv')

if not results_file.exists():
    print("❌ 结果文件不存在")
    exit(1)

# 读取结果
df = pd.read_csv(results_file)
df.columns = df.columns.str.strip()

# 获取最后 30 轮的数据
last_30 = df.tail(30)

print("="*70)
print("📊 最后 30 轮的 mAP50-95 变化")
print("="*70)

for idx, row in last_30.iterrows():
    epoch = int(row['epoch'])
    map_val = row['metrics/mAP50-95(B)']
    print(f"Epoch {epoch:3d}: mAP50-95 = {map_val:.4f}")

# 分析趋势
last_10 = df.tail(10)['metrics/mAP50-95(B)']
last_5 = df.tail(5)['metrics/mAP50-95(B)']

max_map = df['metrics/mAP50-95(B)'].max()
final_map = df.iloc[-1]['metrics/mAP50-95(B)']
best_epoch = df['metrics/mAP50-95(B)'].idxmax() + 1

print("\n" + "="*70)
print("📈 收敛分析")
print("="*70)
print(f"最佳 mAP50-95: {max_map:.4f} (Epoch {best_epoch})")
print(f"最终 mAP50-95: {final_map:.4f} (Epoch 100)")
print(f"差距: {(max_map - final_map):.4f}")

# 判断趋势
last_10_trend = last_10.iloc[-1] - last_10.iloc[0]
last_5_trend = last_5.iloc[-1] - last_5.iloc[0]

print(f"\n最后 10 轮变化: {last_10_trend:+.4f}")
print(f"最后 5 轮变化: {last_5_trend:+.4f}")

print("\n" + "="*70)
print("💡 建议")
print("="*70)

if last_5_trend > 0.002:
    print("✅ mAP 还在明显上升，建议增加 50-100 轮训练")
    print("   预期可提升到: {:.1f}%".format((final_map + 0.01) * 100))
elif last_5_trend > 0.0005:
    print("⚖️  mAP 缓慢上升，可以尝试增加 20-50 轮")
    print("   预期可提升到: {:.1f}%".format((final_map + 0.005) * 100))
elif abs(last_5_trend) < 0.0005:
    print("➡️  mAP 已经平稳，增加轮数提升有限")
    print("   当前结果已经很好: {:.1f}%".format(final_map * 100))
else:
    print("⚠️  mAP 在下降，可能已经过拟合")
    print("   建议使用 Epoch {} 的模型 (mAP={:.1f}%)".format(best_epoch, max_map * 100))

print("="*70)

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2)
plt.axhline(y=max_map, color='r', linestyle='--', label=f'Best: {max_map:.4f}')
plt.xlabel('Epoch')
plt.ylabel('mAP50-95')
plt.title('mAP50-95 Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convergence_analysis.png', dpi=150)
print(f"\n✓ 收敛曲线已保存到: convergence_analysis.png")
