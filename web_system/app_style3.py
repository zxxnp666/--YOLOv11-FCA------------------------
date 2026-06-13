"""
方案三：向导式流程布局
- 步骤条引导用户操作
- 清晰的1-2-3流程
- 适合答辩演示
- 大图展示区域
"""
import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from models.fusion.end_to_end_model import FoggyTrafficSystem
from models.dehazing.generic_dehaze import MPRNetLike


def calculate_psnr(img1, img2):
    """计算PSNR（峰值信噪比）"""
    if img1.shape != img2.shape:
        return 0.0
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """计算SSIM（结构相似性）"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 转灰度计算
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
        img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))


def calculate_contrast(img):
    """计算图像对比度"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    return float(np.std(gray))


def calculate_entropy(img):
    """计算图像信息熵"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


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
                        if file.endswith(('.pth', '.pt', '.pk')):
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
                return "请选择DEA-Net模型文件", False
            if not yolo_path or yolo_path == "未找到模型文件":
                return "请选择YOLO模型文件", False
            if not os.path.exists(deanet_path):
                return f"DEA-Net模型文件不存在: {deanet_path}", False
            if not os.path.exists(yolo_path):
                return f"YOLO模型文件不存在: {yolo_path}", False
            
            self.full_system = FoggyTrafficSystem(
                deanet_weights=deanet_path,
                yolov11_weights=yolo_path,
                device=self.device
            )
            self.full_system_loaded = True
            return "模型加载成功！可以进行下一步", True
        except Exception as e:
            return f"加载失败: {str(e)}", False
    
    def process_full_system(self, image):
        """一站式处理：去雾+检测，带指标计算"""
        if not self.full_system_loaded:
            return None, None, None, "", "请先完成步骤1：加载模型"
        if image is None:
            return None, None, None, "", "请先完成步骤2：上传图片"
        
        try:
            start_time = time.time()
            
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    image_rgb = image
            else:
                image_rgb = np.array(image)
            
            h, w = image_rgb.shape[:2]
            
            # 端到端处理
            detections, dehazed = self.full_system.predict(image_rgb)
            
            dehaze_time = time.time() - start_time
            
            result_image = dehazed.copy()
            num_detections = 0
            
            if detections and len(detections) > 0:
                result = detections[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    line_width = max(3, int(min(h, w) / 200))
                    font_scale = max(0.8, min(h, w) / 800)
                    font_thickness = max(2, int(line_width * 0.8))
                    
                    for box in result.boxes:
                        if box.conf[0] >= 0.25:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), line_width)
                            label = f"Class {cls}: {conf:.2f}"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                            cv2.rectangle(result_image, (x1, y1-th-10), (x1+tw+4, y1), (0, 255, 0), -1)
                            cv2.putText(result_image, label, (x1+2, y1-6),
                                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                            num_detections += 1
            
            total_time = time.time() - start_time
            
            # 计算去雾质量指标
            contrast_before = calculate_contrast(image_rgb)
            contrast_after = calculate_contrast(dehazed)
            contrast_improve = ((contrast_after - contrast_before) / contrast_before * 100) if contrast_before > 0 else 0
            
            entropy_before = calculate_entropy(image_rgb)
            entropy_after = calculate_entropy(dehazed)
            
            psnr_value = calculate_psnr(image_rgb, dehazed)
            ssim_value = calculate_ssim(image_rgb, dehazed)
            
            # 构建指标文本
            metrics_text = f"""╔══════════════════════════════════════╗
║        端到端处理评估报告            ║
╠══════════════════════════════════════╣
║  图像尺寸:    {w} × {h}              
║  总处理耗时:  {total_time:.2f} 秒              
╠══════════════════════════════════════╣
║  【去雾效果评估】                    
║  ├─ 对比度提升:  {contrast_improve:+.1f}%          
║  ├─ 信息熵变化:  {entropy_before:.2f} → {entropy_after:.2f}   
║  ├─ PSNR:        {psnr_value:.2f} dB          
║  └─ SSIM:        {ssim_value:.4f}            
╠══════════════════════════════════════╣
║  【目标检测结果】                    
║  └─ 检测目标数:  {num_detections} 个            
╚══════════════════════════════════════╝"""
            
            status = f"处理完成！耗时{total_time:.2f}s，检测到{num_detections}个目标"
            return image_rgb, dehazed, result_image, metrics_text, status
            
        except Exception as e:
            return None, None, None, "", f"处理失败: {str(e)}"

    def load_dehaze_model(self, model_path):
        try:
            if not model_path or model_path == "未找到模型文件":
                return "请选择DEA-Net模型文件", False
            if not os.path.exists(model_path):
                return f"模型文件不存在: {model_path}", False
            
            self.dehaze_model = MPRNetLike()
            self.dehaze_model.load_pretrained(model_path, self.device)
            self.dehaze_model.to(self.device)
            self.dehaze_model.eval()
            self.dehaze_loaded = True
            return "去雾模型加载成功！", True
        except Exception as e:
            return f"加载失败: {str(e)}", False
    
    def process_dehaze(self, image):
        """去雾处理 - 带指标计算和速度优化"""
        if not self.dehaze_loaded:
            return None, None, "", "请先加载去雾模型"
        if image is None:
            return None, None, "", "请上传雾天图片"
        
        try:
            start_time = time.time()
            
            # 图像预处理
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    image_rgb = image
            else:
                image_rgb = np.array(image)
            
            original_size = image_rgb.shape[:2]
            
            # 速度优化：大图缩放处理
            max_size = 1024
            h, w = original_size
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                image_resized = image_rgb
                new_h, new_w = h, w
            
            # 转换为tensor
            if image_resized.max() > 1.0:
                image_tensor = torch.from_numpy(image_resized).float() / 255.0
            else:
                image_tensor = torch.from_numpy(image_resized).float()
            
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # 使用半精度加速（如果支持）
            use_fp16 = self.device == 'cuda' and torch.cuda.is_available()
            
            with torch.no_grad():
                if use_fp16:
                    with torch.cuda.amp.autocast():
                        dehazed_tensor = self.dehaze_model(image_tensor)
                else:
                    dehazed_tensor = self.dehaze_model(image_tensor)
                dehazed_tensor = torch.clamp(dehazed_tensor, 0, 1)
            
            # 转换回numpy
            dehazed_np = dehazed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            dehazed_np = (dehazed_np * 255).astype('uint8')
            
            # 如果缩放了，恢复原始尺寸
            if max(h, w) > max_size:
                dehazed_np = cv2.resize(dehazed_np, (w, h), interpolation=cv2.INTER_CUBIC)
            
            process_time = time.time() - start_time
            
            # 计算图像质量指标
            # 对比度提升
            contrast_before = calculate_contrast(image_rgb)
            contrast_after = calculate_contrast(dehazed_np)
            contrast_improve = ((contrast_after - contrast_before) / contrast_before * 100) if contrast_before > 0 else 0
            
            # 信息熵
            entropy_before = calculate_entropy(image_rgb)
            entropy_after = calculate_entropy(dehazed_np)
            
            # PSNR和SSIM（去雾前后对比，数值越高说明变化越大）
            psnr_value = calculate_psnr(image_rgb, dehazed_np)
            ssim_value = calculate_ssim(image_rgb, dehazed_np)
            
            # 构建指标显示文本
            metrics_text = f"""┌─────────────────────────────────────┐
│         去雾效果评估指标            │
├─────────────────────────────────────┤
│  处理耗时:     {process_time:.2f} 秒           │
│  图像尺寸:     {w} × {h}             │
├─────────────────────────────────────┤
│  【对比度】                         │
│    处理前:     {contrast_before:.1f}              │
│    处理后:     {contrast_after:.1f}              │
│    提升:       {contrast_improve:+.1f}%            │
├─────────────────────────────────────┤
│  【信息熵】                         │
│    处理前:     {entropy_before:.2f}              │
│    处理后:     {entropy_after:.2f}              │
├─────────────────────────────────────┤
│  【相似度指标】                     │
│    PSNR:       {psnr_value:.2f} dB           │
│    SSIM:       {ssim_value:.4f}              │
└─────────────────────────────────────┘"""
            
            status = f"去雾完成！耗时 {process_time:.2f}s，对比度提升 {contrast_improve:+.1f}%"
            return image_rgb, dehazed_np, metrics_text, status
            
        except Exception as e:
            return None, None, "", f"处理失败: {str(e)}"

    def load_detect_model(self, model_path):
        try:
            if not model_path or model_path == "未找到模型文件":
                return "请选择YOLO模型文件", False
            if not os.path.exists(model_path):
                return f"模型文件不存在: {model_path}", False
            
            from ultralytics import YOLO
            self.detect_model = YOLO(model_path)
            self.detect_loaded = True
            return "检测模型加载成功！", True
        except Exception as e:
            return f"加载失败: {str(e)}", False
    
    def process_detect(self, image):
        if not self.detect_loaded:
            return None, None, "请先加载检测模型"
        if image is None:
            return None, None, "请上传交通图片"
        
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
                    # 根据图片大小动态调整线宽和字体
                    h, w = result_image.shape[:2]
                    line_width = max(3, int(min(h, w) / 200))
                    font_scale = max(0.8, min(h, w) / 800)
                    font_thickness = max(2, int(line_width * 0.8))
                    
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = self.detect_model.names[cls] if hasattr(self.detect_model, 'names') else f"Class {cls}"
                        # 绘制更粗的边框
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), line_width)
                        label = f"{class_name}: {conf:.2f}"
                        # 绘制标签背景
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        cv2.rectangle(result_image, (x1, y1-th-10), (x1+tw+4, y1), (0, 255, 0), -1)
                        cv2.putText(result_image, label, (x1+2, y1-6),
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                        num_detections += 1
            
            return image_rgb, result_image, f"检测完成！发现 {num_detections} 个目标"
        except Exception as e:
            return None, None, f"处理失败: {str(e)}"


def create_interface():
    """方案三：专业级向导式布局"""
    
    app = FoggyTrafficApp()
    
    custom_css = """
    /* 全局样式 */
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif !important;
        background: linear-gradient(180deg, #f0f4f8 0%, #e2e8f0 100%) !important;
        min-height: 100vh;
        padding: 0 !important;
    }
    
    /* 顶部标题栏 */
    .header-bar {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        padding: 16px 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(30, 58, 95, 0.3);
        margin-bottom: 12px;
    }
    
    .header-bar h1 {
        color: white;
        font-size: 22px;
        font-weight: 600;
        margin: 0;
        letter-spacing: 2px;
    }
    
    /* 步骤指示器 */
    .steps-indicator {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 12px 20px;
        background: white;
        border-radius: 10px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    .step {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .step-num {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 13px;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.35);
    }
    
    .step-label {
        font-size: 13px;
        font-weight: 600;
        color: #1e3a5f;
    }
    
    .step-connector {
        width: 40px;
        height: 2px;
        background: #2563eb;
        margin: 0 10px;
        border-radius: 1px;
    }
    
    /* 图片卡片容器 */
    .img-card {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* 图片展示区标题 */
    .img-title {
        font-size: 13px;
        font-weight: 600;
        color: white;
        background: linear-gradient(135deg, #1e3a5f 0%, #334155 100%);
        padding: 10px 16px;
        text-align: center;
        letter-spacing: 1px;
        margin: 0;
    }
    
    /* 图片区域包裹 */
    .img-wrapper {
        background: #f8fafc;
        padding: 8px;
        min-height: 320px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* 主按钮 */
    button.primary {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: white !important;
        box-shadow: 0 3px 12px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    
    button.primary:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.45) !important;
    }
    
    /* Tab样式 */
    .tab-nav {
        background: white !important;
        border-radius: 10px 10px 0 0 !important;
        padding: 4px !important;
        gap: 4px !important;
    }
    
    .tab-nav button {
        border-radius: 8px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        padding: 12px 28px !important;
        margin: 2px !important;
        transition: all 0.2s ease !important;
    }
    
    .tab-nav button:hover {
        background: #f1f5f9 !important;
    }
    
    .tab-nav button.selected {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3) !important;
    }
    
    .tabitem {
        background: white !important;
        border-radius: 0 0 12px 12px !important;
        padding: 20px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
    }
    
    /* 隐藏footer */
    footer {
        display: none !important;
    }
    
    /* 图片组件 */
    .image-frame {
        border-radius: 8px !important;
        border: 2px solid #e2e8f0 !important;
    }
    
    /* 控制区样式 */
    .control-section {
        padding: 12px;
        background: #f8fafc;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Group卡片样式 */
    .gr-group {
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        background: white !important;
    }
    
    /* 图片标题紧贴Group顶部 */
    .img-title {
        margin: -1px -1px 0 -1px;
        border-radius: 10px 10px 0 0;
    }
    
    /* 指标显示框样式 */
    .metrics-box textarea {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        color: #4ade80 !important;
        border: none !important;
        border-radius: 0 0 10px 10px !important;
        padding: 16px !important;
        letter-spacing: 0.5px !important;
    }
    
    /* 下拉框样式 */
    select, .gr-dropdown {
        border-radius: 8px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 8px 12px !important;
        transition: all 0.2s ease !important;
    }
    
    select:focus, .gr-dropdown:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }
    
    /* 输入框样式 */
    input, textarea {
        border-radius: 8px !important;
        border: 2px solid #e2e8f0 !important;
    }
    
    input:focus, textarea:focus {
        border-color: #2563eb !important;
        outline: none !important;
    }
    """
    
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size="lg"
    )
    
    with gr.Blocks(title="雾天交通目标检测系统", theme=theme, css=custom_css) as interface:
        
        # 顶部标题栏
        gr.HTML("""
        <div class="header-bar">
            <h1>雾天交通目标检测系统</h1>
        </div>
        """)
        
        with gr.Tabs():
            
            # 一站式处理（向导式）
            with gr.TabItem("一站式处理"):
                
                # 步骤指示器
                gr.HTML("""
                <div class="steps-indicator">
                    <div class="step">
                        <div class="step-num">1</div>
                        <span class="step-label">加载模型</span>
                    </div>
                    <div class="step-connector"></div>
                    <div class="step">
                        <div class="step-num">2</div>
                        <span class="step-label">上传图片</span>
                    </div>
                    <div class="step-connector"></div>
                    <div class="step">
                        <div class="step-num">3</div>
                        <span class="step-label">查看结果</span>
                    </div>
                </div>
                """)
                
                # 模型加载区 - 四列对齐
                with gr.Row():
                    with gr.Column(scale=1):
                        deanet_dropdown = gr.Dropdown(
                            choices=app.scan_deanet_models(),
                            label="DEA-Net 去雾模型",
                            allow_custom_value=True
                        )
                    with gr.Column(scale=1):
                        yolo_dropdown = gr.Dropdown(
                            choices=app.scan_yolo_models(),
                            label="YOLOv11-FCA 检测模型",
                            allow_custom_value=True
                        )
                    with gr.Column(scale=1):
                        gr.HTML('<div style="height:24px"></div>')
                        load_full_btn = gr.Button("加载模型", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        load_full_status = gr.Textbox(label="加载状态", lines=1, interactive=False)
                
                gr.HTML('<div style="height:12px"></div>')
                
                # 图片展示区 - 四列
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.HTML('<div class="img-title">上传雾天图片</div>')
                            full_input = gr.Image(label="", type="numpy", height=280, show_label=False)
                        full_process_btn = gr.Button("开始处理（去雾+检测）", variant="primary", size="lg")
                        full_status = gr.Textbox(label="处理状态", lines=1, interactive=False)
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.HTML('<div class="img-title">原始雾图</div>')
                            full_original = gr.Image(label="", height=280, show_label=False)
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.HTML('<div class="img-title">去雾结果</div>')
                            full_dehazed = gr.Image(label="", height=280, show_label=False)
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.HTML('<div class="img-title">检测结果</div>')
                            full_result = gr.Image(label="", height=280, show_label=False)
                
                # 评估指标区域（放在下方）
                gr.HTML('<div style="height:12px"></div>')
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            gr.HTML('<div class="img-title">处理评估报告</div>')
                            full_metrics = gr.Textbox(
                                label="", 
                                lines=10, 
                                interactive=False, 
                                show_label=False,
                                elem_classes="metrics-box"
                            )
            
            # 单独去雾
            with gr.TabItem("单独去雾"):
                gr.HTML("""
                <div class="steps-indicator">
                    <div class="step">
                        <div class="step-num">1</div>
                        <span class="step-label">加载模型</span>
                    </div>
                    <div class="step-connector"></div>
                    <div class="step">
                        <div class="step-num">2</div>
                        <span class="step-label">上传处理</span>
                    </div>
                    <div class="step-connector"></div>
                    <div class="step">
                        <div class="step-num">3</div>
                        <span class="step-label">对比结果</span>
                    </div>
                </div>
                """)
                
                # 模型加载区 - 四列对齐
                with gr.Row():
                    with gr.Column(scale=1):
                        dehaze_dropdown = gr.Dropdown(
                            choices=app.scan_deanet_models(),
                            label="DEA-Net 去雾模型",
                            allow_custom_value=True
                        )
                    with gr.Column(scale=1):
                        gr.HTML('<div style="height:24px"></div>')
                        load_dehaze_btn = gr.Button("加载模型", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        load_dehaze_status = gr.Textbox(label="加载状态", lines=1, interactive=False)
                    with gr.Column(scale=1):
                        pass  # 占位保持对齐
                
                gr.HTML('<div style="height:12px"></div>')
                
                # 图片展示区 - 四列（含指标）
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.HTML('<div class="img-title">上传雾天图片</div>')
                            dehaze_input = gr.Image(label="", type="numpy", height=320, show_label=False)
                        dehaze_process_btn = gr.Button("开始去雾", variant="primary", size="lg")
                        dehaze_status = gr.Textbox(label="处理状态", lines=1, interactive=False)
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.HTML('<div class="img-title">原始雾图</div>')
                            dehaze_original = gr.Image(label="", height=320, show_label=False)
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.HTML('<div class="img-title">去雾结果</div>')
                            dehaze_result = gr.Image(label="", height=320, show_label=False)
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.HTML('<div class="img-title">质量评估指标</div>')
                            dehaze_metrics = gr.Textbox(
                                label="", 
                                lines=16, 
                                interactive=False, 
                                show_label=False,
                                elem_classes="metrics-box"
                            )
            
            # 单独检测
            with gr.TabItem("单独检测"):
                gr.HTML("""
                <div class="steps-indicator">
                    <div class="step">
                        <div class="step-num">1</div>
                        <span class="step-label">加载模型</span>
                    </div>
                    <div class="step-connector"></div>
                    <div class="step">
                        <div class="step-num">2</div>
                        <span class="step-label">上传检测</span>
                    </div>
                    <div class="step-connector"></div>
                    <div class="step">
                        <div class="step-num">3</div>
                        <span class="step-label">查看标注</span>
                    </div>
                </div>
                """)
                
                # 模型加载区 - 三列对齐
                with gr.Row():
                    with gr.Column(scale=1):
                        detect_dropdown = gr.Dropdown(
                            choices=app.scan_yolo_models(),
                            label="YOLOv11-FCA 检测模型",
                            allow_custom_value=True
                        )
                    with gr.Column(scale=1):
                        gr.HTML('<div style="height:24px"></div>')
                        load_detect_btn = gr.Button("加载模型", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        load_detect_status = gr.Textbox(label="加载状态", lines=1, interactive=False)
                
                gr.HTML('<div style="height:12px"></div>')
                
                # 图片展示区 - 三列
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.HTML('<div class="img-title">上传交通图片</div>')
                            detect_input = gr.Image(label="", type="numpy", height=360, show_label=False)
                        detect_process_btn = gr.Button("开始检测", variant="primary", size="lg")
                        detect_status = gr.Textbox(label="处理状态", lines=1, interactive=False)
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.HTML('<div class="img-title">原始图片</div>')
                            detect_original = gr.Image(label="", height=360, show_label=False)
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.HTML('<div class="img-title">检测标注</div>')
                            detect_result = gr.Image(label="", height=360, show_label=False)
        
        # 事件绑定
        load_full_btn.click(
            fn=lambda d, y: app.load_full_system(d, y)[0],
            inputs=[deanet_dropdown, yolo_dropdown],
            outputs=load_full_status
        )
        full_process_btn.click(
            fn=app.process_full_system,
            inputs=full_input,
            outputs=[full_original, full_dehazed, full_result, full_metrics, full_status]
        )
        
        load_dehaze_btn.click(
            fn=lambda m: app.load_dehaze_model(m)[0],
            inputs=dehaze_dropdown,
            outputs=load_dehaze_status
        )
        dehaze_process_btn.click(
            fn=app.process_dehaze,
            inputs=dehaze_input,
            outputs=[dehaze_original, dehaze_result, dehaze_metrics, dehaze_status]
        )
        
        load_detect_btn.click(
            fn=lambda m: app.load_detect_model(m)[0],
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
    print("启动方案三：向导式流程布局界面...")
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7862, share=False, inbrowser=True)


if __name__ == '__main__':
    main()
