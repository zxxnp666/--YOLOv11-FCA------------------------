#!/bin/bash
# COCO数据集配置脚本
# 解决ultralytics找不到labels的问题

echo "=========================================="
echo "🔧 COCO数据集配置脚本"
echo "=========================================="

# 设置路径
DATASET_DIR="/root/sj-tmp/biyesheji/datasets/coco"
ZIP_FILE="/root/sj-tmp/biyesheji/coco2017labels-segments.zip"

echo ""
echo "📂 检查数据集目录..."
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ 数据集目录不存在: $DATASET_DIR"
    exit 1
fi
echo "✓ 数据集目录存在"

echo ""
echo "📦 检查labels压缩包..."
if [ ! -f "$ZIP_FILE" ]; then
    echo "❌ labels压缩包不存在: $ZIP_FILE"
    echo "请先下载: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip"
    exit 1
fi
echo "✓ labels压缩包存在 ($(du -h $ZIP_FILE | cut -f1))"

echo ""
echo "📂 检查images目录..."
if [ ! -d "$DATASET_DIR/images/train2017" ]; then
    echo "❌ 训练图像目录不存在: $DATASET_DIR/images/train2017"
    exit 1
fi
if [ ! -d "$DATASET_DIR/images/val2017" ]; then
    echo "❌ 验证图像目录不存在: $DATASET_DIR/images/val2017"
    exit 1
fi
echo "✓ 训练图像: $(ls $DATASET_DIR/images/train2017 | wc -l) 张"
echo "✓ 验证图像: $(ls $DATASET_DIR/images/val2017 | wc -l) 张"

echo ""
echo "📦 解压labels..."
if [ -d "$DATASET_DIR/labels" ]; then
    echo "⚠️  labels目录已存在，是否删除重新解压？(y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        rm -rf "$DATASET_DIR/labels"
        echo "✓ 已删除旧的labels目录"
    else
        echo "跳过解压"
    fi
fi

if [ ! -d "$DATASET_DIR/labels" ]; then
    echo "正在解压 $ZIP_FILE ..."
    unzip -q "$ZIP_FILE" -d "$DATASET_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✓ labels解压成功"
        echo "✓ train2017 labels: $(ls $DATASET_DIR/labels/train2017 | wc -l) 个"
        echo "✓ val2017 labels: $(ls $DATASET_DIR/labels/val2017 | wc -l) 个"
    else
        echo "❌ 解压失败"
        exit 1
    fi
fi

echo ""
echo "🔗 创建符号链接..."

# 删除旧的符号链接
rm -rf "$DATASET_DIR/train" "$DATASET_DIR/val"

# 创建train目录和符号链接
mkdir -p "$DATASET_DIR/train"
ln -sf ../images/train2017 "$DATASET_DIR/train/images"
ln -sf ../labels/train2017 "$DATASET_DIR/train/labels"
echo "✓ train/images -> ../images/train2017"
echo "✓ train/labels -> ../labels/train2017"

# 创建val目录和符号链接
mkdir -p "$DATASET_DIR/val"
ln -sf ../images/val2017 "$DATASET_DIR/val/images"
ln -sf ../labels/val2017 "$DATASET_DIR/val/labels"
echo "✓ val/images -> ../images/val2017"
echo "✓ val/labels -> ../labels/val2017"

echo ""
echo "📝 创建coco.yaml配置文件..."
cat > "$DATASET_DIR/coco.yaml" << 'EOF'
# COCO 2017 数据集配置
# YOLOv11训练使用

# 数据集路径（相对于此yaml文件）
path: /root/sj-tmp/biyesheji/datasets/coco  # 数据集根目录
train: train/images  # 训练图像（相对于path）
val: val/images      # 验证图像（相对于path）

# 类别数量
nc: 80

# 类别名称（COCO 80类）
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush

# 下载配置（已有数据集，不需要下载）
download: |
  echo "数据集已存在，无需下载"
EOF

echo "✓ coco.yaml 创建成功"

echo ""
echo "✅ 验证最终目录结构..."
echo ""
echo "目录结构:"
echo "$DATASET_DIR/"
echo "├── images/"
echo "│   ├── train2017/ ($(ls $DATASET_DIR/images/train2017 | wc -l) 张)"
echo "│   └── val2017/ ($(ls $DATASET_DIR/images/val2017 | wc -l) 张)"
echo "├── labels/"
echo "│   ├── train2017/ ($(ls $DATASET_DIR/labels/train2017 | wc -l) 个)"
echo "│   └── val2017/ ($(ls $DATASET_DIR/labels/val2017 | wc -l) 个)"
echo "├── train/"
echo "│   ├── images -> ../images/train2017"
echo "│   └── labels -> ../labels/train2017"
echo "├── val/"
echo "│   ├── images -> ../images/val2017"
echo "│   └── labels -> ../labels/val2017"
echo "└── coco.yaml"

echo ""
echo "=========================================="
echo "✅ COCO数据集配置完成！"
echo "=========================================="
echo ""
echo "📝 下一步："
echo "1. 设置环境变量（避免磁盘空间问题）："
echo "   export YOLO_CONFIG_DIR=/root/sj-tmp/.ultralytics"
echo "   export TORCH_HOME=/root/sj-tmp/.torch"
echo ""
echo "2. 训练YOLOv11（标准版）："
echo "   python train/train_with_visualization.py \\"
echo "     --data $DATASET_DIR/coco.yaml \\"
echo "     --epochs 100 \\"
echo "     --batch 64 \\"
echo "     --weights yolo11n.pt"
echo ""
echo "3. 训练YOLOv11-FCA："
echo "   python train/train_yolov11_fca_custom.py \\"
echo "     --data $DATASET_DIR/coco.yaml \\"
echo "     --epochs 100 \\"
echo "     --batch 64 \\"
echo "     --weights yolo11n.pt \\"
echo "     --reduction 16"
echo ""
echo "=========================================="
