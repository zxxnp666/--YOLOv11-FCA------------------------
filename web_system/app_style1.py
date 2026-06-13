"""
方案一：卡片式布局
- 顶部Tab切换三个功能模块
- 圆角卡片包裹各区域
- 图片展示区放大到400-500px
- 简洁清爽的视觉风格
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
        
        os.makedirs('checkpoints/deanet', exist_ok=True)
        os.makedirs('checkpoints/yolov11_fca', exist_ok=True)
        os.makedirs('web_system/uploads', exist_ok=True)

    def scan_deanet_models(self):
        search_paths = ["checkpoints/deanet", "DEA-Net-main/trained_models", "shared/checkpoints/deanet"]
        models = []
        for base_path in search_paths:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.endswith(('.pth', '.pt')):
                            models.append(os.path.relpath(os.path.join(root, file)))
        return sorted(set(models)) if models else ["未找到模型文件"]
    
    def scan_yolo_models(self):
        search_paths = ["checkpoints/yolov11_fca", "training/results", "shared/checkpoints/yolov11_fca", "."]
        models = []
        for base_path in search_paths:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.endswith('.pt'):
                            models.append(os.path.relpath(os.path.join(root, file)))
        return sorted(set(models)) if models else ["未找到模型文件"]

    def load_full_system(self, deanet_path, yolo_path):
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
        if not self.full_system_loaded:
            return None, None, None, "请先加载模型"
        if image is None:
            return None, None, None, "请上传雾天图片"
        
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    image_rgb = image
            else:
                image_rgb = np.array(image)
            
            detections, dehazed = self.full_system.predict(image_rgb)
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
            
            return image_rgb, dehazed, result_image, f"处理完成！检测到 {num_detections} 个目标"
        except Exception as e:
            return None, None, None, f"处理失败: {str(e)}"

    def load_dehaze_model(self, model_path):
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
        if not self.dehaze_loaded:
            return None, None, "请先加载去雾模型"
        if image is None:
            return None, None, "请上传雾天图片"
        
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    image_rgb = image
            else:
                image_rgb = np.array(image)
            
            if image_rgb.max() > 1.0:
                image_tensor = torch.from_numpy(image_rgb).float() / 255.0
            else:
                image_tensor = torch.from_numpy(image_rgb).float()
            
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                dehazed_tensor = self.dehaze_model(image_tensor)
                dehazed_tensor = torch.clamp(dehazed_tensor, 0, 1)
            
            dehazed_np = dehazed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            dehazed_np = (dehazed_np * 255).astype('uint8')
            return image_rgb, dehazed_np, "去雾处理完成！"
        except Exception as e:
            return None, None, f"处理失败: {str(e)}"

    def load_detect_model(self, model_path):
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
        if not self.detect_loaded:
            return None, None, "请先加载检测模型"
        if image is None:
            return None, None, "请上传清晰交通图片"
        
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    image_rgb = image
            else:
                image_rgb = np.array(image)
            
            results = self.detect_model.predict(image_rgb, conf=0.25, verbose=False)
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
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(result_image, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        num_detections += 1
            
            return image_rgb, result_image, f"检测完成！发现 {num_detections} 个目标"
        except Exception as e:
            return None, None, f"处理失败: {str(e)}"


def create_interface():
    """方案一：卡片式布局界面"""
    
    app = FoggyTrafficApp()
    
    custom_css = """
    .gradio-container {
        font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 20px;
    }
    
    /* 主容器卡片 */
    .main-card {
        background: white;
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        padding: 0;
        max-width: 1400px;
        margin: 0 auto;
        overflow: hidden;
    }
    
    /* 顶部标题栏 */
    .header-bar {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 24px 32px;
        text-align: center;
    }
    
    .header-title {
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 8px 0;
        letter-spacing: 2px;
    }
    
    .header-subtitle {
        font-size: 14px;
        color: #a0a0a0;
        margin: 0;
    }
    
    /* 内容区域 */
    .content-area {
        padding: 32px;
        background: #f8f9fa;
    }
    
    /* 功能卡片 */
    .func-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    
    .func-card-title {
        font-size: 16px;
        font-weight: 600;
        color: #1a1a2e;
        margin: 0 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid #667eea;
    }
    
    /* 图片展示卡片 */
    .image-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e9ecef;
    }
    
    .image-label {
        font-size: 14px;
        font-weight: 600;
        color: #495057;
        margin-bottom: 12px;
    }
    
    /* 主按钮样式 */
    .primary-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
    }
    
    .primary-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102,126,234,0.5) !important;
    }
    
    /* 次要按钮 */
    .secondary-btn {
        background: #f8f9fa !important;
        color: #495057 !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-size: 13px !important;
    }
    
    .secondary-btn:hover {
        background: #e9ecef !important;
    }
    
    /* 状态文本框 */
    .status-box textarea {
        background: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        font-size: 13px !important;
    }
    
    /* Tab样式优化 */
    .tabs {
        border: none !important;
    }
    
    .tab-nav {
        background: white !important;
        border-bottom: 2px solid #e9ecef !important;
        padding: 0 32px !important;
    }
    
    .tab-nav button {
        font-size: 15px !important;
        font-weight: 500 !important;
        padding: 16px 24px !important;
        border: none !important;
        background: transparent !important;
        color: #6c757d !important;
        border-bottom: 3px solid transparent !important;
        margin-bottom: -2px !important;
    }
    
    .tab-nav button.selected {
        color: #667eea !important;
        border-bottom-color: #667eea !important;
        font-weight: 600 !important;
    }
    
    /* 下拉框样式 */
    .dropdown-container select {
        border-radius: 8px !important;
        border: 1px solid #dee2e6 !important;
        padding: 10px 12px !important;
    }
    """
    
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate"
    )
    
    with gr.Blocks(title="雾天交通目标检测系统", theme=theme, css=custom_css) as interface:
        
        # 顶部标题
        gr.HTML("""
        <div class="main-card">
            <div class="header-bar">
                <h1 class="header-title">雾天交通目标检测系统</h1>
                <p class="header-subtitle">基于DEA-Net去雾与YOLOv11-FCA检测的端到端解决方案</p>
            </div>
        </div>
        """)
        
        # Tab切换
        with gr.Tabs() as tabs:
            
            # Tab1: 一站式处理
            with gr.TabItem("一站式处理", id=0):
                with gr.Row():
                    # 左侧：配置与上传
                    with gr.Column(scale=2):
                        gr.HTML('<div class="func-card-title">模型配置</div>')
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
                        load_full_btn = gr.Button("加载模型", variant="primary", size="lg")
                        load_full_status = gr.Textbox(label="加载状态", lines=2, interactive=False)
                        
                        gr.HTML('<div class="func-card-title" style="margin-top:20px;">上传图片</div>')
                        full_input = gr.Image(label="上传雾天图片", type="numpy", height=280)
                        full_process_btn = gr.Button("开始处理", variant="primary", size="lg")
                        full_status = gr.Textbox(label="处理状态", interactive=False)
                    
                    # 右侧：结果展示
                    with gr.Column(scale=3):
                        gr.HTML('<div class="func-card-title">处理结果</div>')
                        with gr.Row():
                            with gr.Column():
                                gr.HTML('<div class="image-label">原始雾图</div>')
                                full_original = gr.Image(label="", height=400, show_label=False)
                            with gr.Column():
                                gr.HTML('<div class="image-label">去雾结果</div>')
                                full_dehazed = gr.Image(label="", height=400, show_label=False)
                        with gr.Row():
                            with gr.Column():
                                gr.HTML('<div class="image-label">检测结果</div>')
                                full_result = gr.Image(label="", height=400, show_label=False)
            
            # Tab2: 单独去雾
            with gr.TabItem("单独去雾", id=1):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML('<div class="func-card-title">模型配置</div>')
                        dehaze_dropdown = gr.Dropdown(
                            choices=app.scan_deanet_models(),
                            label="DEA-Net去雾模型",
                            allow_custom_value=True
                        )
                        load_dehaze_btn = gr.Button("加载模型", variant="primary")
                        load_dehaze_status = gr.Textbox(label="加载状态", lines=2, interactive=False)
                        
                        gr.HTML('<div class="func-card-title" style="margin-top:20px;">上传图片</div>')
                        dehaze_input = gr.Image(label="上传雾天图片", type="numpy", height=280)
                        dehaze_process_btn = gr.Button("开始去雾", variant="primary", size="lg")
                        dehaze_status = gr.Textbox(label="处理状态", interactive=False)
                    
                    with gr.Column(scale=3):
                        gr.HTML('<div class="func-card-title">去雾效果对比</div>')
                        with gr.Row():
                            with gr.Column():
                                gr.HTML('<div class="image-label">原始雾图</div>')
                                dehaze_original = gr.Image(label="", height=450, show_label=False)
                            with gr.Column():
                                gr.HTML('<div class="image-label">去雾结果</div>')
                                dehaze_result = gr.Image(label="", height=450, show_label=False)
            
            # Tab3: 单独检测
            with gr.TabItem("单独检测", id=2):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML('<div class="func-card-title">模型配置</div>')
                        detect_dropdown = gr.Dropdown(
                            choices=app.scan_yolo_models(),
                            label="YOLOv11-FCA检测模型",
                            allow_custom_value=True
                        )
                        load_detect_btn = gr.Button("加载模型", variant="primary")
                        load_detect_status = gr.Textbox(label="加载状态", lines=2, interactive=False)
                        
                        gr.HTML('<div class="func-card-title" style="margin-top:20px;">上传图片</div>')
                        detect_input = gr.Image(label="上传交通图片", type="numpy", height=280)
                        detect_process_btn = gr.Button("开始检测", variant="primary", size="lg")
                        detect_status = gr.Textbox(label="处理状态", interactive=False)
                    
                    with gr.Column(scale=3):
                        gr.HTML('<div class="func-card-title">检测结果</div>')
                        with gr.Row():
                            with gr.Column():
                                gr.HTML('<div class="image-label">原始图片</div>')
                                detect_original = gr.Image(label="", height=450, show_label=False)
                            with gr.Column():
                                gr.HTML('<div class="image-label">检测标注</div>')
                                detect_result = gr.Image(label="", height=450, show_label=False)
        
        # 事件绑定
        load_full_btn.click(fn=app.load_full_system, inputs=[deanet_dropdown, yolo_dropdown], outputs=load_full_status)
        full_process_btn.click(fn=app.process_full_system, inputs=full_input, outputs=[full_original, full_dehazed, full_result, full_status])
        
        load_dehaze_btn.click(fn=app.load_dehaze_model, inputs=dehaze_dropdown, outputs=load_dehaze_status)
        dehaze_process_btn.click(fn=app.process_dehaze, inputs=dehaze_input, outputs=[dehaze_original, dehaze_result, dehaze_status])
        
        load_detect_btn.click(fn=app.load_detect_model, inputs=detect_dropdown, outputs=load_detect_status)
        detect_process_btn.click(fn=app.process_detect, inputs=detect_input, outputs=[detect_original, detect_result, detect_status])
    
    return interface


def main():
    print("启动方案一：卡片式布局界面...")
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7861, share=False, inbrowser=True)


if __name__ == '__main__':
    main()
