# COCO数据集目录

请将COCO数据集按以下结构放置：

```
training/data/datasets/coco/
├── train2017/              # 训练集图像（118,287张）
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
├── val2017/                # 验证集图像（5,000张）
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...
└── labels/                 # YOLO格式标注（ultralytics会自动生成）
    ├── train2017/
    │   ├── 000000000009.txt
    │   └── ...
    └── val2017/
        ├── 000000000139.txt
        └── ...
```

## 📥 下载COCO数据集

### 方式1：使用官方链接下载

```bash
# 训练集图像（18GB）
wget http://images.cocodataset.org/zips/train2017.zip

# 验证集图像（1GB）
wget http://images.cocodataset.org/zips/val2017.zip

# 标注文件（241MB）
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# 解压
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

### 方式2：让ultralytics自动下载

第一次训练时，ultralytics会自动下载COCO数据集到这个目录。

## ⚠️ 注意事项

1. **标注格式**：ultralytics会自动将COCO JSON格式转换为YOLO格式
2. **磁盘空间**：确保至少有30GB可用空间
3. **下载时间**：根据网络速度，可能需要1-3小时

## ✅ 验证数据集

确保目录结构正确后，运行训练脚本即可。ultralytics会自动验证数据集。
