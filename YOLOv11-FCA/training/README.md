# YOLOv11-FCA 训练项目

基于YOLOv11集成FCA（Fast Channel Attention）注意力机制的目标检测模型训练代码。

## 📁 项目结构

```
training/
├── models/
│   └── detection/
│       ├── fca_attention.py          # FCA注意力模块
│       ├── yolov11_fca.py            # YOLOv11-FCA模型定义
│       └── __init__.py
├── train/
│   ├── train_yolov11_fca_custom.py   # 训练脚本（真正注入FCA）
│   └── __init__.py
├── config/
│   └── yolov11_fca_config.yaml       # 训练配置
├── coco.yaml                          # 数据集配置
├── requirements.txt                   # 依赖库
├── yolo11n.pt                         # 预训练权重
├── start_training.sh                  # 启动脚本
└── README.md                          # 本文件
```

## 📦 数据集准备

### 方式1：使用COCO数据集

将数据集放在以下位置：
```
training/data/datasets/coco/
├── train2017/          # 训练集图像
├── val2017/            # 验证集图像
└── labels/             # YOLO格式标注
    ├── train2017/
    └── val2017/
```

### 方式2：使用自定义数据集

1. 修改 `coco.yaml` 中的路径和类别
2. 确保数据集格式为YOLO格式（每张图片对应一个.txt标注文件）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

将数据集放在 `data/datasets/coco/` 目录下（或修改 `coco.yaml` 配置）

### 3. 开始训练

**方式1：使用启动脚本**
```bash
bash start_training.sh
```

**方式2：直接运行Python脚本**
```bash
python train/train_yolov11_fca_custom.py \
    --data coco.yaml \
    --epochs 100 \
    --batch 16 \
    --device 0 \
    --weights yolo11n.pt
```

### 4. 查看训练进度

```bash
tensorboard --logdir checkpoints/yolov11_fca --port 6006
```

## ⚙️ 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 数据集配置文件 | coco.yaml |
| `--epochs` | 训练轮数 | 100 |
| `--batch` | 批次大小 | 16 |
| `--device` | GPU设备ID | 0 |
| `--fraction` | 使用数据集比例 | 1.0 |
| `--reduction` | FCA降维比例 | 16 |
| `--weights` | 预训练权重 | yolo11n.pt |
| `--patience` | 早停耐心值 | 20 |
| `--lr0` | 初始学习率 | 0.001 |
| `--weight_decay` | 权重衰减 | 0.001 |

## 📊 训练输出

训练完成后，模型和日志保存在：
```
checkpoints/yolov11_fca/train/
├── weights/
│   ├── best.pt         # 最佳模型
│   └── last.pt         # 最新模型
├── results.png         # 训练曲线
└── ...
```

## 🔧 FCA注意力机制

FCA模块会自动注入到YOLOv11的Backbone和Neck中的C3k2/C2f模块后，无需手动修改模型结构。

训练脚本会输出注入的FCA模块数量和位置。

## 📝 注意事项

1. **GPU内存**：batch_size=16需要约8GB显存，根据GPU调整
2. **数据集格式**：确保使用YOLO格式标注（每行：class x_center y_center width height）
3. **早停机制**：patience=20表示20轮不提升就停止，防止过拟合
4. **学习率**：lr0=0.001较保守，可根据训练曲线调整

## 🐛 常见问题

**Q: 提示找不到数据集？**
A: 检查 `coco.yaml` 中的路径是否正确，确保数据集已放在指定位置。

**Q: 显存不足？**
A: 减小 `--batch` 参数，例如改为8或4。

**Q: 训练速度慢？**
A: 可以使用 `--fraction 0.1` 先用10%数据测试，确认无误后再用全量数据。

## 📧 联系方式

如有问题，请提交Issue或联系开发者。
