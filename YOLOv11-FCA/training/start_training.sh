#!/bin/bash
# YOLOv11-FCA 训练启动脚本

echo "=========================================="
echo "🚀 启动 YOLOv11-FCA 训练"
echo "=========================================="

# 激活虚拟环境（如果使用）
# source venv/bin/activate

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 开始训练
python train/train_yolov11_fca_custom.py \
    --data coco.yaml \
    --epochs 100 \
    --batch 16 \
    --device 0 \
    --fraction 1.0 \
    --reduction 16 \
    --weights yolo11n.pt \
    --patience 20 \
    --lr0 0.001 \
    --weight_decay 0.001

echo "=========================================="
echo "✅ 训练完成"
echo "=========================================="
