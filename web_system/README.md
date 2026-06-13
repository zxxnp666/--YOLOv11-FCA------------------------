# Web系统说明

## 📋 系统列表

本目录包含三个Web可视化系统：

### 1. app_unified.py - 统一系统 ⭐推荐
**端口**: 7860  
**启动**: `启动统一Web系统.bat` 或 `python web_system/app_unified.py`

**特点**:
- ✅ 左侧导航栏
- ✅ 三大独立模块（完整系统、去雾、检测）
- ✅ 现代化界面
- ✅ 适合答辩展示

**适用场景**: 毕业设计答辩、完整功能展示

---

### 2. app.py - 完整系统
**端口**: 7861  
**启动**: `python web_system/app.py`

**特点**:
- ✅ 端到端处理
- ✅ 专业界面
- ✅ 批量处理

**适用场景**: 完整系统演示、批量处理

---

### 3. app_yolo_only.py - YOLO系统
**端口**: 7862  
**启动**: `启动YOLO检测系统.bat` 或 `python web_system/app_yolo_only.py`

**特点**:
- ✅ 简洁界面
- ✅ 快速启动
- ✅ 专注检测

**适用场景**: 查看训练结果、快速测试

---

## 🚀 快速开始

### 推荐使用统一系统

1. **准备权重文件**
   ```
   checkpoints/deanet/ITS/PSNR4131_SSIM9945.pth
   checkpoints/yolov11_fca/best.pt
   ```

2. **启动系统**
   ```bash
   # Windows
   启动统一Web系统.bat
   
   # 或命令行
   python web_system/app_unified.py
   ```

3. **访问系统**
   ```
   http://localhost:7860
   ```

---

## 📁 目录结构

```
web_system/
├── app_unified.py          # 统一系统（推荐）
├── app.py                  # 完整系统
├── app_yolo_only.py        # YOLO系统
├── models/                 # 模型定义
│   ├── dehazing/          # 去雾模型
│   ├── detection/         # 检测模型
│   └── fusion/            # 融合模型
├── utils/                  # 工具函数
├── uploads/                # 上传文件临时存储
└── README.md              # 本文档
```

---

## 🔗 相关文档

- [统一Web系统使用说明](../docs/统一Web系统使用说明.md)
- [Web系统对比说明](../docs/Web系统对比说明.md)
- [YOLO检测系统使用说明](../docs/YOLO检测系统使用说明.md)
- [权重文件说明](../checkpoints/README.md)

---

## 💡 选择建议

| 需求 | 推荐系统 |
|------|---------|
| 毕业设计答辩 | app_unified.py ⭐ |
| 日常开发测试 | app_unified.py |
| 查看训练结果 | app_yolo_only.py |
| 完整系统演示 | app.py |
| 批量处理 | app.py |

---

**最后更新**: 2025年1月
