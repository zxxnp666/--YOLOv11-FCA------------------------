"""
雾天交通目标检测与去雾增强系统
改进版 - 算家云视觉体系风格界面
"""

import gradio as gr
import cv2
import numpy as np
import torch
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from models.fusion.end_to_end_model import FoggyTrafficSystem
from models.dehazing.generic_dehaze import MPRNetLike


class FoggyTrafficApp:
    def __init__(self):
        self.full_system = None
        self.dehaze_model = None
        self.detect_model = None
        self.full_system_loaded = False
        self.dehaze_loaded = False
        self.detect_loaded = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建必要目录
        os.makedirs('checkpoints/deanet', exist_ok=True)
        os.makedirs('checkpoints/yolov11_fca', exist_ok=True)
        os.makedirs('web_system/uploads', exist_ok=True)

    def scan_deanet_models(self):
        """扫描DEA-Net模型文件"""
        search_paths = [
            "checkpoints/deanet",
            "DEA-Net-main/trained_models",
            "shared/checkpoints/deanet",
        ]
        
        models = []
        for base_path in search_paths:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.endswith(('.pth', '.pt')):
                            full_path = os.path.join(root, file)
                            models.append(os.path.relpath(full_path))
        
        return sorted(set(models)) if models else ["未找到模型文件"]
    
    def scan_yolo_models(self):
        """扫描YOLO模型文件"""
        search_paths = [
            "checkpoints/yolov11_fca",
            "training/results",
            "shared/checkpoints/yolov11_fca",
            ".",
        ]
        
        models = []
        for base_path in search_paths:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.endswith('.pt'):
                            full_path = os.path.join(root, file)
                            models.append(os.path.relpath(full_path))
        
        return sorted(set(models)) if models else ["未找到模型文件"]
    def load_full_system(self, deanet_path, yolo_path):
        """加载完整系统"""
        try:
            if not deanet_path or deanet_path == "未找到模型文件":
                return "请选择DEA-Net模型文件"
            if not yolo_path or yolo_path == "未找到模型文件":
                return "请选择YOLO模型文件"
            
            if not os.path.exists(deanet_path):
                return f"DEA-Net模型文件不存在: {deanet_path}"
            if not os.path.exists(yolo_path):
                return f"YOLO模型文件不存在: {yolo_path}"
            
            self.full_system = FoggyTrafficSystem(
                deanet_weights=deanet_path,
                yolov11_weights=yolo_path,
                device=self.device
            )
            self.full_system_loaded = True
            
            return "模型加载成功！系统已就绪"
            
        except Exception as e:
            return f"加载失败: {str(e)}"
    
    def process_full_system(self, image):
        """一站式处理：去雾+检测"""
        if not self.full_system_loaded:
            return None, None, None, "请先加载模型"
        
        if image is None:
            return None, None, None, "请上传雾天图片"
        
        try:
            # 转换为RGB
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    image_rgb = image
            else:
                image_rgb = np.array(image)
            
            # 端到端处理
            detections, dehazed = self.full_system.predict(image_rgb)
            
            # 绘制检测结果
            result_image = dehazed.copy()
            num_detections = 0
            
            if detections and len(detections) > 0:
                result = detections[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    for box in result.boxes:
                        if box.conf[0] >= 0.25:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            
                            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Class {cls}: {conf:.2f}"
                            cv2.putText(result_image, label, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            num_detections += 1
            
            status = f"处理完成！检测到 {num_detections} 个目标"
            return image_rgb, dehazed, result_image, status
            
        except Exception as e:
            return None, None, None, f"处理失败: {str(e)}"
    def load_dehaze_model(self, model_path):
        """加载去雾模型"""
        try:
            if not model_path or model_path == "未找到模型文件":
                return "请选择DEA-Net模型文件"
            
            if not os.path.exists(model_path):
                return f"模型文件不存在: {model_path}"
            
            self.dehaze_model = MPRNetLike()
            self.dehaze_model.load_pretrained(model_path, self.device)
            self.dehaze_model.to(self.device)
            self.dehaze_model.eval()
            self.dehaze_loaded = True
            
            return "去雾模型加载成功！"
            
        except Exception as e:
            return f"加载失败: {str(e)}"
    
    def process_dehaze(self, image):
        """单独去雾处理"""
        if not self.dehaze_loaded:
            return None, None, "请先加载去雾模型"
        
        if image is None:
            return None, None, "请上传雾天图片"
        
        try:
            # 转换为RGB
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    image_rgb = image
            else:
                image_rgb = np.array(image)
            
            # 归一化
            if image_rgb.max() > 1.0:
                image_tensor = torch.from_numpy(image_rgb).float() / 255.0
            else:
                image_tensor = torch.from_numpy(image_rgb).float()
            
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # 去雾
            with torch.no_grad():
                dehazed_tensor = self.dehaze_model(image_tensor)
                dehazed_tensor = torch.clamp(dehazed_tensor, 0, 1)
            
            # 转换回numpy
            dehazed_np = dehazed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            dehazed_np = (dehazed_np * 255).astype('uint8')
            
            return image_rgb, dehazed_np, "去雾处理完成！"
            
        except Exception as e:
            return None, None, f"处理失败: {str(e)}"
    def load_detect_model(self, model_path):
        """加载检测模型"""
        try:
            if not model_path or model_path == "未找到模型文件":
                return "请选择YOLO模型文件"
            
            if not os.path.exists(model_path):
                return f"模型文件不存在: {model_path}"
            
            from ultralytics import YOLO
            self.detect_model = YOLO(model_path)
            self.detect_loaded = True
            
            return "检测模型加载成功！"
            
        except Exception as e:
            return f"加载失败: {str(e)}"
    
    def process_detect(self, image):
        """单独检测处理"""
        if not self.detect_loaded:
            return None, None, "请先加载检测模型"
        
        if image is None:
            return None, None, "请上传清晰交通图片"
        
        try:
            # 转换为RGB
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    image_rgb = image
            else:
                image_rgb = np.array(image)
            
            # YOLO检测
            results = self.detect_model.predict(image_rgb, conf=0.25, verbose=False)
            
            # 绘制结果
            result_image = image_rgb.copy()
            num_detections = 0
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        class_name = self.detect_model.names[cls] if hasattr(self.detect_model, 'names') else f"Class {cls}"
                        
                        color = (0, 255, 0)
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(result_image, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        num_detections += 1
            
            status = f"检测完成！发现 {num_detections} 个目标"
            return image_rgb, result_image, status
            
        except Exception as e:
            return None, None, f"处理失败: {str(e)}"
def create_interface():
    """创建改进版算家云风格界面"""
    
    app = FoggyTrafficApp()
    
    # 改进版算家云风格CSS
    custom_css = """
    /* 算家云视觉体系 - 改进版 */
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
        background: #f0f2f5;
        margin: 0;
        padding: 0;
        min-height: 100vh;
    }
    
    /* 顶部导航栏 */
    .top-navbar {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 16px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    
    .navbar-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .navbar-title {
        font-size: 20px;
        font-weight: 700;
        margin: 0;
    }
    
    .navbar-subtitle {
        font-size: 14px;
        opacity: 0.9;
        margin-top: 2px;
    }
    
    .navbar-info {
        font-size: 13px;
        opacity: 0.8;
    }
    
    /* 主布局容器 */
    .main-layout {
        display: flex;
        min-height: calc(100vh - 80px);
        max-width: 1400px;
        margin: 0 auto;
        background: white;
        box-shadow: 0 0 20px rgba(0,0,0,0.05);
    }
    
    /* 左侧导航栏 - 改进版 */
    .sidebar {
        background: linear-gradient(180deg, #34495e 0%, #2c3e50 100%);
        width: 280px;
        min-height: calc(100vh - 80px);
        display: flex;
        flex-direction: column;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    
    .sidebar-header {
        padding: 32px 24px 24px 24px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.05);
    }
    
    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        color: #ecf0f1;
        margin: 0 0 8px 0;
        text-align: center;
    }
    
    .sidebar-subtitle {
        font-size: 14px;
        color: #bdc3c7;
        text-align: center;
        margin: 0;
    }
    
    /* 导航菜单 */
    .nav-menu {
        flex: 1;
        padding: 24px 0;
        display: flex;
        flex-direction: column;
    }
    
    .nav-item {
        display: block;
        width: 100%;
        padding: 18px 24px;
        border: none;
        background: transparent;
        text-align: left;
        font-size: 16px;
        font-weight: 500;
        color: #ecf0f1;
        cursor: pointer;
        transition: all 0.3s ease;
        border-left: 4px solid transparent;
        margin-bottom: 4px;
    }
    
    .nav-item:hover {
        background: rgba(255,255,255,0.1);
        color: #2ecc71;
        border-left-color: #2ecc71;
        transform: translateX(4px);
    }
    
    .nav-item.active {
        background: rgba(46,204,113,0.15);
        color: #2ecc71;
        border-left-color: #2ecc71;
        font-weight: 600;
    }
    
    /* 右侧主内容区 */
    .main-content {
        flex: 1;
        padding: 32px;
        background: #fafbfc;
        overflow-y: auto;
    }
    
    /* 内容卡片 */
    .content-card {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e8eaec;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    
    .card-title {
        font-size: 24px;
        font-weight: 700;
        color: #2c3e50;
        margin: 0 0 24px 0;
        padding-bottom: 16px;
        border-bottom: 2px solid #e8eaec;
        position: relative;
    }
    
    .card-title::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 60px;
        height: 2px;
        background: #27ae60;
    }
    
    /* 绿色主按钮 - 加大 */
    .primary-btn {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 16px 32px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        min-width: 160px !important;
        box-shadow: 0 4px 12px rgba(39,174,96,0.3) !important;
    }
    
    .primary-btn:hover {
        background: linear-gradient(135deg, #229954 0%, #27ae60 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(39,174,96,0.4) !important;
    }
    
    /* 次要按钮 */
    .secondary-btn {
        background: #6c757d !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 12px 20px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    
    .secondary-btn:hover {
        background: #5a6268 !important;
        transform: translateY(-1px) !important;
    }
    
    /* 上传区域 */
    .upload-area {
        border: 2px dashed #bdc3c7;
        border-radius: 12px;
        padding: 48px 24px;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.3s ease;
        margin: 16px 0;
    }
    
    .upload-area:hover {
        border-color: #27ae60;
        background: #f0fff4;
        transform: translateY(-2px);
    }
    
    /* 结果展示区 */
    .result-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 24px;
        margin-top: 24px;
    }
    
    .result-item {
        background: #ffffff;
        border: 1px solid #e8eaec;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    
    .result-item:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    .result-title {
        font-size: 16px;
        font-weight: 600;
        color: #495057;
        margin-bottom: 16px;
    }
    
    /* 进度步骤 */
    .progress-steps {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 24px 0;
        padding: 20px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        border: 1px solid #dee2e6;
    }
    
    .step-item {
        display: flex;
        align-items: center;
        font-size: 15px;
        font-weight: 500;
        color: #6c757d;
        padding: 8px 16px;
        border-radius: 20px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .step-item.active {
        color: #27ae60;
        background: #e8f5e8;
        font-weight: 600;
    }
    
    .step-arrow {
        margin: 0 16px;
        color: #bdc3c7;
        font-size: 18px;
        font-weight: bold;
    }
    
    /* 状态提示 */
    .status-success {
        color: #27ae60;
        background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        margin-top: 16px;
        border-left: 4px solid #27ae60;
    }
    
    .status-error {
        color: #e74c3c;
        background: linear-gradient(135deg, #fdf2f2 0%, #f8d7da 100%);
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        margin-top: 16px;
        border-left: 4px solid #e74c3c;
    }
    
    /* 隐藏Gradio默认元素 */
    .gr-box {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }
    
    .gr-form {
        border: none !important;
        background: transparent !important;
    }
    
    /* 响应式设计 */
    @media (max-width: 1024px) {
        .sidebar {
            width: 240px;
        }
        
        .main-content {
            padding: 24px;
        }
        
        .result-grid {
            grid-template-columns: 1fr;
        }
    }
    
    @media (max-width: 768px) {
        .navbar-content {
            flex-direction: column;
            text-align: center;
        }
        
        .sidebar {
            width: 200px;
        }
        
        .main-content {
            padding: 16px;
        }
        
        .content-card {
            padding: 24px;
        }
    }
    """
    
    # 创建主题
    theme = gr.themes.Soft(
        primary_hue="green",
        secondary_hue="gray",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
    )
    
    with gr.Blocks(title="雾天交通目标检测系统", theme=theme, css=custom_css) as interface:
        
        # 顶部导航栏
        gr.HTML("""
        <div class="top-navbar">
            <div class="navbar-content">
                <div>
                    <div class="navbar-title">智能视觉分析平台</div>
                    <div class="navbar-subtitle">Intelligent Vision Analysis Platform</div>
                </div>
                <div class="navbar-info">
                    算家云视觉体系 | 基于深度学习的雾天交通检测
                </div>
            </div>
        </div>
        """)
        
        # 主布局容器
        gr.HTML('<div class="main-layout">')
        
        # 使用Row布局实现左侧导航栏 + 右侧内容区
        with gr.Row():
            # 左侧导航栏
            with gr.Column(scale=1, min_width=280):
                gr.HTML("""
                <div class="sidebar">
                    <div class="sidebar-header">
                        <div class="sidebar-title">雾天交通检测系统</div>
                    </div>
                    <div class="nav-menu">
                """)
                
                # 导航选择 - 使用Radio实现选项卡切换
                nav_choice = gr.Radio(
                    choices=["一站式处理", "单独去雾", "单独检测"],
                    value="一站式处理",
                    label="",
                    show_label=False,
                    container=False
                )
                
                gr.HTML("""
                    </div>
                </div>
                """)
            
            # 右侧主内容区
            with gr.Column(scale=4):
                gr.HTML('<div class="main-content">')
                
                # 一站式处理页面
                with gr.Group(visible=True) as page_full:
                    gr.HTML('<div class="content-card">')
                    gr.HTML('<div class="card-title">雾天交通目标一站式处理</div>')
                    
                    # 模型配置区
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 模型配置")
                            deanet_dropdown = gr.Dropdown(
                                choices=app.scan_deanet_models(),
                                label="DEA-Net去雾模型",
                                allow_custom_value=True
                            )
                            yolo_dropdown = gr.Dropdown(
                                choices=app.scan_yolo_models(),
                                label="YOLOv11-FCA检测模型",
                                allow_custom_value=True
                            )
                            load_full_btn = gr.Button("加载模型", variant="primary")
                            load_full_status = gr.Textbox(label="状态", lines=3, interactive=False)
                    
                    gr.HTML("""
                    <div class="progress-steps">
                        <div class="step-item active">上传图片</div>
                        <div class="step-arrow">→</div>
                        <div class="step-item">去雾处理</div>
                        <div class="step-arrow">→</div>
                        <div class="step-item">目标检测</div>
                    </div>
                    """)
                    
                    # 操作区
                    with gr.Row():
                        with gr.Column():
                            full_input = gr.Image(label="上传雾天图片", type="numpy", height=320)
                            full_process_btn = gr.Button("启动去雾 + 检测", variant="primary", size="lg")
                            full_status = gr.Textbox(label="处理状态", interactive=False)
                    
                    # 结果展示区
                    gr.Markdown("### 处理结果")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**原始雾图**")
                            full_original = gr.Image(label="", height=280)
                        with gr.Column():
                            gr.Markdown("**去雾后图像**")
                            full_dehazed = gr.Image(label="", height=280)
                        with gr.Column():
                            gr.Markdown("**检测结果图**")
                            full_result = gr.Image(label="", height=280)
                    
                    gr.HTML('</div>')  # 关闭content-card
                
                # 单独去雾页面
                with gr.Group(visible=False) as page_dehaze:
                    gr.HTML('<div class="content-card">')
                    gr.HTML('<div class="card-title">单独去雾增强</div>')
                    
                    # 模型配置区
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 模型配置")
                            dehaze_dropdown = gr.Dropdown(
                                choices=app.scan_deanet_models(),
                                label="DEA-Net去雾模型",
                                allow_custom_value=True
                            )
                            load_dehaze_btn = gr.Button("加载模型", variant="primary")
                            load_dehaze_status = gr.Textbox(label="状态", lines=2, interactive=False)
                    
                    # 操作区
                    with gr.Row():
                        with gr.Column():
                            dehaze_input = gr.Image(label="上传雾天图片", type="numpy", height=320)
                            dehaze_process_btn = gr.Button("启动去雾", variant="primary", size="lg")
                            dehaze_status = gr.Textbox(label="处理状态", interactive=False)
                    
                    # 结果展示区
                    gr.Markdown("### 去雾结果对比")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**原始雾图**")
                            dehaze_original = gr.Image(label="", height=320)
                        with gr.Column():
                            gr.Markdown("**去雾后图像**")
                            dehaze_result = gr.Image(label="", height=320)
                    
                    gr.HTML('</div>')  # 关闭content-card
                
                # 单独检测页面
                with gr.Group(visible=False) as page_detect:
                    gr.HTML('<div class="content-card">')
                    gr.HTML('<div class="card-title">单独目标检测</div>')
                    
                    # 模型配置区
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 模型配置")
                            detect_dropdown = gr.Dropdown(
                                choices=app.scan_yolo_models(),
                                label="YOLOv11-FCA检测模型",
                                allow_custom_value=True
                            )
                            load_detect_btn = gr.Button("加载模型", variant="primary")
                            load_detect_status = gr.Textbox(label="状态", lines=2, interactive=False)
                    
                    # 操作区
                    with gr.Row():
                        with gr.Column():
                            detect_input = gr.Image(label="上传清晰交通图片", type="numpy", height=320)
                            detect_process_btn = gr.Button("启动检测", variant="primary", size="lg")
                            detect_status = gr.Textbox(label="处理状态", interactive=False)
                    
                    # 结果展示区
                    gr.Markdown("### 检测结果")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**原始图片**")
                            detect_original = gr.Image(label="", height=320)
                        with gr.Column():
                            gr.Markdown("**检测标注结果图**")
                            detect_result = gr.Image(label="", height=320)
                    
                    gr.HTML('</div>')  # 关闭content-card
                
                gr.HTML('</div>')  # 关闭main-content
        
        gr.HTML('</div>')  # 关闭main-layout
        
        # 页面切换逻辑
        def switch_page(choice):
            if choice == "一站式处理":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            elif choice == "单独去雾":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            elif choice == "单独检测":
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
        
        nav_choice.change(
            fn=switch_page,
            inputs=nav_choice,
            outputs=[page_full, page_dehaze, page_detect]
        )
        
        # 事件绑定 - 一站式处理
        load_full_btn.click(
            fn=app.load_full_system,
            inputs=[deanet_dropdown, yolo_dropdown],
            outputs=load_full_status
        )
        
        full_process_btn.click(
            fn=app.process_full_system,
            inputs=full_input,
            outputs=[full_original, full_dehazed, full_result, full_status]
        )
        
        # 事件绑定 - 单独去雾
        load_dehaze_btn.click(
            fn=app.load_dehaze_model,
            inputs=dehaze_dropdown,
            outputs=load_dehaze_status
        )
        
        dehaze_process_btn.click(
            fn=app.process_dehaze,
            inputs=dehaze_input,
            outputs=[dehaze_original, dehaze_result, dehaze_status]
        )
        
        # 事件绑定 - 单独检测
        load_detect_btn.click(
            fn=app.load_detect_model,
            inputs=detect_dropdown,
            outputs=load_detect_status
        )
        
        detect_process_btn.click(
            fn=app.process_detect,
            inputs=detect_input,
            outputs=[detect_original, detect_result, detect_status]
        )
    
    return interface


def main():
    """启动应用"""
    print("启动雾天交通检测系统...")
    print("改进版算家云视觉体系风格界面")
    print("- 顶部导航栏")
    print("- 改进的左侧导航栏")
    print("- 更大的字体和按钮")
    print("- 渐变背景色")
    
    try:
        interface = create_interface()
        print("界面创建成功，启动服务器...")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7862,
            share=False,
            inbrowser=True,
            show_error=True
        )
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()