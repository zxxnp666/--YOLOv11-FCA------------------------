"""
雾天交通目标检测与去雾增强系统 - Streamlit 版本

特点：
- 使用 Streamlit，界面简洁、偏工程化风格，弱化 AI 痕迹
- 左侧导航：端到端系统 / 仅去雾 / 仅检测
- 单张图片演示，一屏同时展示：原图 / 去雾图 / 检测结果
- 显示耗时、对比度提升、PSNR、SSIM 等指标（端到端、去雾模式）
- 支持导出三图拼接对比图（本地下载）
"""

import os
import random
import sys
import time
import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(__file__))

from models.fusion.end_to_end_model import FoggyTrafficSystem  # type: ignore
from models.dehazing.generic_dehaze import MPRNetLike  # type: ignore
from models.dehazing.deanet_official import DEANetOfficial  # type: ignore
from video_utils import process_video_with_system  # type: ignore


# ======================
# 图像指标计算函数
# ======================

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 PSNR（峰值信噪比）"""
    if img1.shape != img2.shape:
        return 0.0
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 SSIM（结构相似性）"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

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

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(np.mean(ssim_map))


def calculate_contrast(img: np.ndarray) -> float:
    """图像对比度（灰度标准差）"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    return float(np.std(gray))


def calculate_entropy(img: np.ndarray) -> float:
    """图像信息熵（灰度）"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-12)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def calculate_sharpness(img: np.ndarray) -> float:
    """清晰度（Laplacian 方差，越大通常越清晰）"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def calculate_brightness(img: np.ndarray) -> float:
    """平均亮度（灰度均值）"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    return float(np.mean(gray))


def is_night_image(img: np.ndarray, threshold: float = 80.0) -> bool:
    """
    判断图片是否为夜晚场景（基于平均亮度）
    
    Args:
        img: RGB 图像 (numpy array)
        threshold: 亮度阈值，低于此值认为是夜晚（默认 80.0）
    
    Returns:
        True 表示夜晚，False 表示白天
    """
    brightness = calculate_brightness(img)
    return brightness < threshold


# ======================
# 随机雾气合成工具
# ======================

def add_random_haze(
    img_bgr: np.ndarray,
    beta_range=(0.6, 2.0),
    A_range=(0.7, 1.0),
) -> tuple[np.ndarray, dict]:
    """给 BGR 图像添加随机雾气（简化大气散射模型）。"""
    h, w, _ = img_bgr.shape

    # 1) 随机采样雾参数
    beta = random.uniform(*beta_range)
    A = random.uniform(*A_range)

    # 2) 在较低分辨率生成平滑“深度图”，再上采样，加快速度
    max_dim = 512
    scale = max(1, int(max(h, w) / max_dim))
    small_h = max(1, h // scale)
    small_w = max(1, w // scale)

    noise_small = np.random.rand(small_h, small_w).astype(np.float32)
    sigma = max(small_h, small_w) / 10
    noise_small = cv2.GaussianBlur(noise_small, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_CUBIC)
    noise -= noise.min()
    noise /= (noise.max() + 1e-8)
    d = noise

    # 3) t(x) = exp(-beta * d(x))
    t = np.exp(-beta * d)[..., None]

    # 4) I = J * t + A * (1 - t)
    J = img_bgr.astype(np.float32) / 255.0
    A_vec = np.array([A, A, A], dtype=np.float32)
    I = J * t + A_vec * (1.0 - t)

    hazy_bgr = np.clip(I * 255.0, 0, 255).astype(np.uint8)

    info = {
        "beta_雾浓度系数": round(beta, 3),
        "A_大气光强度": round(A, 3),
    }
    return hazy_bgr, info


# ======================
# 模型封装类（复用原逻辑）
# ======================


class FoggyTrafficApp:
    def __init__(self) -> None:
        # 单模型模式（兼容旧代码）
        self.full_system = None
        self.dehaze_model = None
        self.detect_model = None
        
        # 双模型模式（白天/夜晚自动选择）
        self.full_system_day = None
        self.full_system_night = None
        self.dehaze_model_day = None
        self.dehaze_model_night = None
        
        self.full_system_loaded = False
        self.dual_system_loaded = False  # 双模型模式是否已加载
        self.dehaze_loaded = False
        self.dual_dehaze_loaded = False  # 双去雾模型模式是否已加载
        self.detect_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs("checkpoints/deanet", exist_ok=True)
        os.makedirs("checkpoints/yolov11_fca", exist_ok=True)

    def scan_deanet_models(self):
        search_paths = [
            "checkpoints/deanet",
            "DEA-Net-main/trained_models",
            "shared/checkpoints/deanet",
        ]
        models = []
        for base_path in search_paths:
            if os.path.exists(base_path):
                for root, _, files in os.walk(base_path):
                    for file in files:
                        if file.endswith((".pth", ".pt", ".pk")):
                            models.append(os.path.relpath(os.path.join(root, file)))
        return sorted(set(models)) if models else ["未找到模型文件"]

    def scan_yolo_models(self):
        search_paths = [
            "checkpoints/yolov11_fca",
            "training/results",
            "shared/checkpoints/yolov11_fca",
            ".",
        ]
        models = []
        for base_path in search_paths:
            if os.path.exists(base_path):
                for root, _, files in os.walk(base_path):
                    for file in files:
                        if file.endswith(".pt"):
                            models.append(os.path.relpath(os.path.join(root, file)))
        return sorted(set(models)) if models else ["未找到模型文件"]

    # ----- 端到端 -----
    def load_full_system(
        self,
        deanet_path: str,
        yolo_path: str,
        use_dual_models: bool = True,
        day_model_path: str | None = None,
        night_model_path: str | None = None,
    ) -> str:
        """
        加载端到端系统模型
        
        Args:
            deanet_path: 去雾模型路径（单模型模式）或 None（双模型模式）
            yolo_path: YOLO 检测模型路径
            use_dual_models: 是否使用双模型模式（自动加载白天.pk 和 夜晚.pk）
        """
        if not yolo_path or yolo_path == "未找到模型文件":
            return "请选择 YOLO 模型文件"
        if not os.path.exists(yolo_path):
            return f"YOLO 模型文件不存在：{yolo_path}"

        # 双模型模式：加载白天/夜晚两套权重（默认路径在 checkpoints/deanet 下）
        if use_dual_models:
            day_model_path = day_model_path or "checkpoints/deanet/白天.pk"
            night_model_path = night_model_path or "checkpoints/deanet/夜晚.pk"
            
            if not os.path.exists(day_model_path):
                return f"白天模型文件不存在：{day_model_path}"
            if not os.path.exists(night_model_path):
                return f"夜晚模型文件不存在：{night_model_path}"
            
            try:
                print(f"[WebSystem] 正在加载白天模型：{day_model_path}")
                self.full_system_day = FoggyTrafficSystem(
                    deanet_weights=day_model_path,
                    yolov11_weights=yolo_path,
                    device=self.device,
                )
                print(f"[WebSystem] ✅ 白天模型加载成功")
                
                print(f"[WebSystem] 正在加载夜晚模型：{night_model_path}")
                self.full_system_night = FoggyTrafficSystem(
                    deanet_weights=night_model_path,
                    yolov11_weights=yolo_path,
                    device=self.device,
                )
                print(f"[WebSystem] ✅ 夜晚模型加载成功")
                
                self.dual_system_loaded = True
                self.full_system_loaded = False  # 单模型模式未加载
                return f"双模型系统加载成功（白天：{day_model_path}，夜晚：{night_model_path}）"
            except Exception as e:
                return f"双模型加载失败：{str(e)}"
        
        # 单模型模式（兼容旧代码）
        else:
            if not deanet_path or deanet_path == "未找到模型文件":
                return "请选择 DEA-Net 模型文件"
            if not os.path.exists(deanet_path):
                return f"DEA-Net 模型文件不存在：{deanet_path}"

            self.full_system = FoggyTrafficSystem(
                deanet_weights=deanet_path,
                yolov11_weights=yolo_path,
                device=self.device,
            )
            self.full_system_loaded = True
            self.dual_system_loaded = False
            return "端到端系统模型加载成功"

    def process_full_system(
        self,
        image: np.ndarray,
        gt_image: np.ndarray | None = None,
        enable_pre_detect: bool = True,
        detection_conf_pre: float = 0.25,
        detection_conf_post: float = 0.25,
    ):
        # 检查模型是否已加载（支持单模型和双模型模式）
        if not self.full_system_loaded and not self.dual_system_loaded:
            return None, None, None, None, {}, {}
        if image is None:
            return None, None, None, None, {}, {}

        start = time.time()

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
        
        # 根据模式选择使用的系统
        if self.dual_system_loaded:
            # 双模型模式：根据图片亮度自动选择
            is_night = is_night_image(image_rgb, threshold=getattr(self, "night_threshold", 80.0))
            current_system = self.full_system_night if is_night else self.full_system_day
            model_type = "夜晚模型" if is_night else "白天模型"
            brightness = calculate_brightness(image_rgb)
            print(f"[WebSystem] 图片平均亮度：{brightness:.2f}，自动选择：{model_type}")
        else:
            # 单模型模式
            current_system = self.full_system
            model_type = "单模型"

        # 如果提供了清晰参考图，先对其尺寸进行对齐
        gt_rgb: np.ndarray | None = None
        if gt_image is not None:
            if isinstance(gt_image, np.ndarray):
                if len(gt_image.shape) == 2:
                    gt_rgb = cv2.cvtColor(gt_image, cv2.COLOR_GRAY2RGB)
                elif gt_image.shape[2] == 4:
                    gt_rgb = cv2.cvtColor(gt_image, cv2.COLOR_RGBA2RGB)
                else:
                    gt_rgb = gt_image
            else:
                gt_rgb = np.array(gt_image)
            if gt_rgb.shape[:2] != (h, w):
                gt_rgb = cv2.resize(gt_rgb, (w, h), interpolation=cv2.INTER_AREA)

        # 先对原始雾图做一次检测（用于对比，可选）
        pre_det_image = image_rgb.copy()
        pre_label_jobs: list[tuple[int, int, str, tuple[int, int, int], int]] = []
        pre_target_counter: dict[str, int] = {}
        pre_boxes_for_union: list[tuple[int, int, int, int, str]] = []

        if enable_pre_detect:
            try:
                # 使用当前选择的系统的 YOLO 模型
                yolo_model = current_system.yolo_model  # type: ignore[attr-defined]
                pre_results = yolo_model.predict(  # type: ignore[attr-defined]
                    image_rgb, conf=float(detection_conf_pre), verbose=False
                )
                if pre_results and len(pre_results) > 0:
                    pre_r = pre_results[0]
                    if hasattr(pre_r, "boxes") and len(pre_r.boxes) > 0:
                        base_lw = max(4, int(min(h, w) / 150))
                        base_thick = max(2, int(base_lw * 0.9))
                        base_font_px = max(16, int(min(h, w) / 35))

                        for box in pre_r.boxes:
                            if box.conf[0] >= detection_conf_pre:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])

                                if hasattr(yolo_model, "names"):  # type: ignore[attr-defined]
                                    name_map = yolo_model.names  # type: ignore[attr-defined]
                                    raw_name = str(name_map.get(cls, cls)).lower()
                                else:
                                    raw_name = str(cls)

                                if "person" in raw_name:
                                    color = (255, 0, 0)
                                    label_cn = "行人"
                                    lw = int(base_lw * 1.4)
                                elif any(
                                    k in raw_name
                                    for k in ["car", "truck", "bus", "van", "motorcycle", "bike"]
                                ):
                                    color = (255, 140, 0)
                                    label_cn = "车辆"
                                    lw = int(base_lw * 1.3)
                                else:
                                    color = (0, 180, 0)
                                    label_cn = "目标"
                                    lw = base_lw

                                cv2.rectangle(pre_det_image, (x1, y1), (x2, y2), color, lw)
                                label = f"{label_cn}  置信度 {conf:.2f}"
                                pre_label_jobs.append((x1, y1, label, color, base_font_px))
                                pre_target_counter[label_cn] = pre_target_counter.get(label_cn, 0) + 1
                                pre_boxes_for_union.append((x1, y1, x2, y2, label_cn))
            except Exception:
                # 如果原始检测出错，不影响后续流程，直接使用原图
                pre_label_jobs = []

        pre_det_image = _draw_labels_pil(pre_det_image, pre_label_jobs) if pre_label_jobs else image_rgb.copy()

        # 再进行端到端：去雾 + 检测（使用当前选择的系统）
        detections, dehazed = current_system.predict(image_rgb)

        # 确保去雾结果与原始雾图尺寸一致（某些网络可能因卷积/转置卷积造成1像素差异）
        if dehazed.shape[:2] != (h, w):
            dehazed = cv2.resize(dehazed, (w, h), interpolation=cv2.INTER_AREA)

        result_image = dehazed.copy()
        num_detections = 0
        label_jobs: list[tuple[int, int, str, tuple[int, int, int], int]] = []
        target_counter: dict[str, int] = {}
        post_boxes_for_union: list[tuple[int, int, int, int, str]] = []

        if detections and len(detections) > 0:
            result = detections[0]
            if hasattr(result, "boxes") and len(result.boxes) > 0:
                base_lw = max(4, int(min(h, w) / 150))
                base_thick = max(2, int(base_lw * 0.9))
                base_font_px = max(16, int(min(h, w) / 35))

                for box in result.boxes:
                    if box.conf[0] >= detection_conf_post:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])

                        # 根据类别区分颜色和粗细：行人/车辆更醒目
                        yolo_model = current_system.yolo_model  # type: ignore[attr-defined]
                        if hasattr(yolo_model, "names"):  # type: ignore[attr-defined]
                            name_map = yolo_model.names  # type: ignore[attr-defined]
                            raw_name = str(name_map.get(cls, cls)).lower()
                        else:
                            raw_name = str(cls)

                        # 注意：当前图像是 RGB；颜色也用 RGB
                        if "person" in raw_name:
                            color = (255, 0, 0)  # 红色突出行人
                            label_cn = "行人"
                            lw = int(base_lw * 1.4)
                        elif any(k in raw_name for k in ["car", "truck", "bus", "van", "motorcycle", "bike"]):
                            color = (255, 140, 0)  # 橙色突出车辆
                            label_cn = "车辆"
                            lw = int(base_lw * 1.3)
                        else:
                            color = (0, 180, 0)  # 其他目标保持绿色
                            # 其他类别优先显示模型类别名；如果拿不到，就用“目标”
                            label_cn = "目标"
                            lw = base_lw

                        font_thickness = base_thick

                        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, lw)
                        # 中文标签交给 PIL 画（OpenCV putText 会变成问号）
                        label = f"{label_cn}  置信度 {conf:.2f}"
                        label_jobs.append((x1, y1, label, color, base_font_px))
                        num_detections += 1
                        target_counter[label_cn] = target_counter.get(label_cn, 0) + 1
                        post_boxes_for_union.append((x1, y1, x2, y2, label_cn))

        # 用 PIL 绘制中文标签
        result_image = _draw_labels_pil(result_image, label_jobs)

        pre_total = int(sum(pre_target_counter.values()))
        post_total = int(num_detections)

        # 计算去雾前/后检测结果的“并集”数量（按 IoU 合并重复目标）
        def _iou(box1: tuple[int, int, int, int, str], box2: tuple[int, int, int, int, str]) -> float:
            x1, y1, x2, y2, _ = box1
            xx1, yy1, xx2, yy2, _ = box2
            inter_x1 = max(x1, xx1)
            inter_y1 = max(y1, yy1)
            inter_x2 = min(x2, xx2)
            inter_y2 = min(y2, yy2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return 0.0
            inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (xx2 - xx1) * (yy2 - yy1)
            union = area1 + area2 - inter + 1e-6
            return inter / union

        merged_boxes: list[tuple[int, int, int, int, str]] = []
        merged_boxes.extend(pre_boxes_for_union)
        for pb in post_boxes_for_union:
            overlap = False
            for mb in merged_boxes:
                # 同一类别且 IoU 足够大则认为是同一个目标
                if pb[4] == mb[4] and _iou(pb, mb) >= 0.5:
                    overlap = True
                    break
            if not overlap:
                merged_boxes.append(pb)

        merged_target_counter: dict[str, int] = {}
        for _, _, _, _, cls_name in merged_boxes:
            merged_target_counter[cls_name] = merged_target_counter.get(cls_name, 0) + 1
        merged_total = int(sum(merged_target_counter.values()))

        # 根据是否提供清晰参考图，选择不同的 PSNR/SSIM 计算方式
        if gt_rgb is not None:
            # 系统同款：参考图 vs 雾图 / 去雾图
            psnr_before = calculate_psnr(gt_rgb, image_rgb)
            ssim_before = calculate_ssim(gt_rgb, image_rgb)
            psnr_after = calculate_psnr(gt_rgb, dehazed)
            ssim_after = calculate_ssim(gt_rgb, dehazed)

            metrics = {
                "去雾前 PSNR (dB)": round(psnr_before, 2),
                "去雾后 PSNR (dB)": round(psnr_after, 2),
                "去雾前 SSIM": round(ssim_before, 4),
                "去雾后 SSIM": round(ssim_after, 4),
                "去雾前检测目标数": pre_total,
                "去雾后检测目标数": post_total,
                "综合检测目标数（并集）": merged_total,
            }
        else:
            # 无参考图：使用变化量指标，表明去雾前后发生了变化
            psnr_after = calculate_psnr(image_rgb, dehazed)
            ssim_after = calculate_ssim(image_rgb, dehazed)

            metrics = {
                "去雾前 PSNR (dB)": "∞",
                "去雾后 PSNR (dB)": round(psnr_after, 2),
                "去雾前 SSIM": 1.0,
                "去雾后 SSIM": round(ssim_after, 4),
                "去雾前检测目标数": pre_total,
                "去雾后检测目标数": post_total,
                "综合检测目标数（并集）": merged_total,
            }
        
        # 如果是双模型模式，在指标中添加使用的模型信息
        if self.dual_system_loaded:
            brightness = calculate_brightness(image_rgb)
            metrics["使用的模型"] = model_type
            metrics["图片平均亮度"] = round(brightness, 2)

        # 目标类型统计：同时展示去雾前/去雾后/并集的数量
        all_classes = sorted(set(list(pre_target_counter.keys()) + list(target_counter.keys())))
        targets = []
        for cls_name in all_classes:
            targets.append(
                {
                    "目标类型": cls_name,
                    "去雾前数量": int(pre_target_counter.get(cls_name, 0)),
                    "去雾后数量": int(target_counter.get(cls_name, 0)),
                    "综合数量（并集）": int(merged_target_counter.get(cls_name, 0)),
                }
            )

        return image_rgb, dehazed, pre_det_image, result_image, metrics, targets

    # ----- 去雾 -----
    def load_dehaze_model(self, model_path: str) -> str:
        if not model_path or model_path == "未找到模型文件":
            return "请选择 DEA-Net 模型文件"
        if not os.path.exists(model_path):
            return f"模型文件不存在：{model_path}"

        # 优先尝试使用官方 DEA-Net 结构加载 .pk 权重
        try:
            self.dehaze_model = DEANetOfficial(device=self.device)
            loaded, total = self.dehaze_model.load_pretrained(model_path, self.device)
            if loaded == 0:
                print(
                    "[WebSystem] DEA-Net 官方加载失败或未匹配任何权重，将回退到通用 MPRNetLike 结构。"
                )
                raise RuntimeError("DEA-Net weights not matched")
        except Exception as e:
            print(f"[WebSystem] 使用 DEANetOfficial 加载失败: {e}")
            print("[WebSystem] 回退到 MPRNetLike 封装加载方式。")
            self.dehaze_model = MPRNetLike()
            self.dehaze_model.load_pretrained(model_path, self.device)

        self.dehaze_model.to(self.device)
        self.dehaze_model.eval()
        self.dehaze_loaded = True
        return "去雾模型加载成功"

    def process_dehaze(self, image: np.ndarray, gt_image: np.ndarray | None = None):
        if not self.dehaze_loaded:
            return None, None, {}, {}
        if image is None:
            return None, None, {}, {}

        start = time.time()

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

        # 若提供清晰参考图，先对齐尺寸
        gt_rgb: np.ndarray | None = None
        if gt_image is not None:
            if isinstance(gt_image, np.ndarray):
                if len(gt_image.shape) == 2:
                    gt_rgb = cv2.cvtColor(gt_image, cv2.COLOR_GRAY2RGB)
                elif gt_image.shape[2] == 4:
                    gt_rgb = cv2.cvtColor(gt_image, cv2.COLOR_RGBA2RGB)
                else:
                    gt_rgb = gt_image
            else:
                gt_rgb = np.array(gt_image)
            if gt_rgb.shape[:2] != (h, w):
                gt_rgb = cv2.resize(gt_rgb, (w, h), interpolation=cv2.INTER_AREA)

        if image_rgb.max() > 1.0:
            image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        else:
            image_tensor = torch.from_numpy(image_rgb).float()

        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dehazed_tensor = self.dehaze_model(image_tensor)
            dehazed_tensor = torch.clamp(dehazed_tensor, 0, 1)

        dehazed_np = dehazed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        dehazed_np = (dehazed_np * 255).astype("uint8")

        # 确保去雾结果与原图尺寸一致
        if dehazed_np.shape[:2] != (h, w):
            dehazed_np = cv2.resize(dehazed_np, (w, h), interpolation=cv2.INTER_AREA)

        if gt_rgb is not None:
            # 系统同款：参考图 vs 雾图 / 去雾图
            psnr_before = calculate_psnr(gt_rgb, image_rgb)
            ssim_before = calculate_ssim(gt_rgb, image_rgb)
            psnr_after = calculate_psnr(gt_rgb, dehazed_np)
            ssim_after = calculate_ssim(gt_rgb, dehazed_np)

            metrics = {
                "去雾前 PSNR (dB)": round(psnr_before, 2),
                "去雾后 PSNR (dB)": round(psnr_after, 2),
                "去雾前 SSIM": round(ssim_before, 4),
                "去雾后 SSIM": round(ssim_after, 4),
            }
        else:
            # 无参考图：变化量指标
            psnr_after = calculate_psnr(image_rgb, dehazed_np)
            ssim_after = calculate_ssim(image_rgb, dehazed_np)

            metrics = {
                "去雾前 PSNR (dB)": "∞",
                "去雾后 PSNR (dB)": round(psnr_after, 2),
                "去雾前 SSIM": 1.0,
                "去雾后 SSIM": round(ssim_after, 4),
            }

        return image_rgb, dehazed_np, metrics, []

    # ----- 检测 -----
    def load_detect_model(self, model_path: str) -> str:
        if not model_path or model_path == "未找到模型文件":
            return "请选择 YOLO 模型文件"
        if not os.path.exists(model_path):
            return f"模型文件不存在：{model_path}"

        from ultralytics import YOLO  # type: ignore

        self.detect_model = YOLO(model_path)
        self.detect_loaded = True
        return "检测模型加载成功"

    def process_detect(self, image: np.ndarray, detection_conf: float = 0.25):
        if not self.detect_loaded:
            return None, None, 0, []
        if image is None:
            return None, None, 0, []

        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                image_rgb = image
        else:
            image_rgb = np.array(image)

        results = self.detect_model.predict(image_rgb, conf=float(detection_conf), verbose=False)
        result_image = image_rgb.copy()
        num_detections = 0
        label_jobs: list[tuple[int, int, str, tuple[int, int, int], int]] = []
        target_counter: dict[str, int] = {}

        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, "boxes") and len(result.boxes) > 0:
                h, w = result_image.shape[:2]
                base_lw = max(4, int(min(h, w) / 150))
                base_thick = max(2, int(base_lw * 0.9))
                base_font_px = max(16, int(min(h, w) / 35))

                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if conf < detection_conf:
                        continue
                    cls = int(box.cls[0])
                    if hasattr(self.detect_model, "names"):
                        raw_name = str(self.detect_model.names[cls]).lower()
                    else:
                        raw_name = str(cls)

                    if "person" in raw_name:
                        color = (255, 0, 0)
                        label_cn = "行人"
                        lw = int(base_lw * 1.4)
                    elif any(k in raw_name for k in ["car", "truck", "bus", "van", "motorcycle", "bike"]):
                        color = (255, 140, 0)
                        label_cn = "车辆"
                        lw = int(base_lw * 1.3)
                    else:
                        color = (0, 180, 0)
                        label_cn = "目标"
                        lw = base_lw

                    font_thickness = base_thick

                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, lw)
                    label = f"{label_cn}  置信度 {conf:.2f}"
                    label_jobs.append((x1, y1, label, color, base_font_px))
                    num_detections += 1
                    target_counter[label_cn] = target_counter.get(label_cn, 0) + 1

        result_image = _draw_labels_pil(result_image, label_jobs)
        targets = [
            {"目标类型": k, "数量": int(v)}
            for k, v in sorted(target_counter.items(), key=lambda x: (-x[1], x[0]))
        ]
        return image_rgb, result_image, num_detections, targets


# ======================
# Streamlit 页面
# ======================

SETTINGS_PATH = Path(__file__).resolve().parent / ".last_models.json"


def load_settings() -> dict:
    """读取上次使用的模型路径（用于下次默认加载）。"""
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def save_settings(data: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@st.cache_resource(show_spinner=False)
def get_app() -> FoggyTrafficApp:
    """创建并缓存应用实例。

    说明：Streamlit 每次交互都会重新执行脚本，如果不缓存对象，
    则每次点击按钮都会创建新的 FoggyTrafficApp 实例，之前加载的模型状态会丢失，
    导致“加载成功但开始处理时又提示未加载”。使用 cache_resource 可以保证
    整个会话中始终复用同一个应用实例和已加载的模型。
    """
    return FoggyTrafficApp()


def combine_images_horiz(*images: np.ndarray) -> Image.Image:
    """将多张 RGB 图像横向拼接为一张 PIL Image"""
    pil_images = [Image.fromarray(img) for img in images if img is not None]
    if not pil_images:
        raise ValueError("没有有效图像用于拼接")

    h = max(img.height for img in pil_images)
    resized = []
    for img in pil_images:
        new_w = int(img.width * h / img.height)
        resized.append(img.resize((new_w, h)))

    total_w = sum(img.width for img in resized)
    combined = Image.new("RGB", (total_w, h))
    x_offset = 0
    for img in resized:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    return combined


def _find_cn_font_path() -> str | None:
    """尽量寻找系统可用的中文字体（Windows 优先微软雅黑）。"""
    candidates: list[Path] = []
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates.extend(
        [
            Path(windir) / "Fonts" / "msyh.ttc",  # 微软雅黑
            Path(windir) / "Fonts" / "msyh.ttf",
            Path(windir) / "Fonts" / "simhei.ttf",  # 黑体
            Path(windir) / "Fonts" / "simsun.ttc",  # 宋体
        ]
    )
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _get_cn_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """获取中文字体；找不到时退化为默认字体（可能不支持中文，但不会报错）。"""
    font_path = _find_cn_font_path()
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_labels_pil(
    img_rgb: np.ndarray,
    labels: list[tuple[int, int, str, tuple[int, int, int], int]],
) -> np.ndarray:
    """
    在 RGB 图像上用 PIL 绘制中文标签，避免 OpenCV putText 变成问号。
    labels: (x1, y1, text, rgb_color, font_size)
    """
    if not labels:
        return img_rgb

    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    for x1, y1, text, color, font_size in labels:
        font = _get_cn_font(font_size)
        # 计算文字框大小
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        pad_x, pad_y = 8, 5
        x = max(0, x1)
        y = max(0, y1 - th - pad_y * 2 - 2)

        # 背景色块
        draw.rectangle([x, y, x + tw + pad_x * 2, y + th + pad_y * 2], fill=color)
        # 黑字更清晰
        draw.text((x + pad_x, y + pad_y), text, font=font, fill=(0, 0, 0))

    return np.array(pil_img)


def main():
    st.set_page_config(page_title="雾天交通目标实时检测与去雾增强系统", page_icon="🚦", layout="wide")

    app = get_app()
    settings = load_settings()

    # 全局样式（导航栏美化等）
    st.markdown(
        """
        <style>
        /* 侧边栏整体背景与阴影：柔和蓝绿渐变 */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #e0f2fe 0%, #e0f2fe 35%, #f9fafb 100%);
            color: #0f172a;
            box-shadow: 2px 0 10px rgba(15,23,42,0.12);
        }

        .sidebar-panel {
            padding: 22px 18px 24px 18px;
        }
        .sidebar-title-main {
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 12px;
            letter-spacing: 1px;
            color: #0f172a;
        }
        .sidebar-section-label {
            font-size: 14px;
            font-weight: 600;
            color: #6b7280;
            margin-bottom: 6px;
        }
        .sidebar-hint {
            font-size: 12px;
            color: #9ca3af;
            margin-top: 8px;
        }

        /* 功能模块单选按钮美化：圆角卡片 + 轻微动效 */
        div[data-testid="stRadio"] > label {
            font-size: 14px;
            font-weight: 500;
        }

        div[data-testid="stRadio"] > div {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        div[data-testid="stRadio"] label[data-baseweb="radio"] {
            border-radius: 999px;
            padding: 6px 10px;
            transition: all 0.18s ease-out;
            cursor: pointer;
        }

        div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
            background: rgba(191, 219, 254, 0.9);
            transform: translateX(2px);
        }

        /* 选中项的高亮：柔和蓝绿条 */
        div[data-testid="stRadio"] input[aria-checked="true"] + div {
            background: linear-gradient(90deg, #38bdf8, #4ade80);
            box-shadow: 0 0 8px rgba(56,189,248,0.5);
            border-radius: 999px;
            padding: 6px 12px;
            transform: translateX(3px);
        }

        div[data-testid="stRadio"] input[aria-checked="true"] + div p {
            color: #0f172a !important;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 左侧导航（美化版）
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-panel">
              <div class="sidebar-title-main">系统导航</div>
              <div class="sidebar-section-label">功能模块</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        mode = st.radio(
            "功能模块",
            [
                "端到端系统（去雾+检测）",
                "仅去雾增强",
                "仅目标检测",
                "随机雾气合成工具",
                "视频处理（去雾+检测）",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        # 互斥：自动加载上次模型 vs 智能识别亮度（只能选一个）
        if "auto_load" not in st.session_state:
            st.session_state["auto_load"] = True
        if "enable_smart_brightness" not in st.session_state:
            st.session_state["enable_smart_brightness"] = False

        def _on_auto_load_change():
            if st.session_state.get("auto_load"):
                st.session_state["enable_smart_brightness"] = False

        def _on_smart_brightness_change():
            if st.session_state.get("enable_smart_brightness"):
                st.session_state["auto_load"] = False

        auto_load = st.checkbox(
            "启动时自动加载上次模型",
            key="auto_load",
            on_change=_on_auto_load_change,
        )
        enable_pre_detect = st.checkbox("显示去雾前检测对比（略慢）", value=True)
        enable_smart_brightness = st.checkbox(
            "智能识别亮度（自动选择白天/夜晚去雾模型）",
            key="enable_smart_brightness",
            on_change=_on_smart_brightness_change,
        )
        detect_conf_pre = st.slider(
            "去雾前检测置信度阈值（越大目标越少但更可靠）",
            0.05,
            0.8,
            0.25,
            0.01,
        )
        detect_conf_post = st.slider(
            "去雾后/仅检测置信度阈值",
            0.05,
            0.8,
            0.25,
            0.01,
        )
        night_threshold = st.slider(
            "夜晚判断亮度阈值（越大越容易判为夜晚）",
            30.0,
            140.0,
            80.0,
            1.0,
            disabled=not enable_smart_brightness,
        )
        st.markdown(
            '<div class="sidebar-hint">提示：建议在 GPU 环境下运行以获得更好体验。</div>',
            unsafe_allow_html=True,
        )

    # 如果是随机雾气合成工具，直接走合成流程，跳过模型加载与检测逻辑
    if mode == "随机雾气合成工具":
        st.subheader("随机雾气合成工具")
        st.markdown("上传一张图片，生成不同雾浓度和分布的合成雾天图像，可用于数据扩充演示。")

        # 参数区域
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.markdown("#### 雾气参数范围")
            beta_min, beta_max = st.slider(
                "雾浓度系数 β 范围（越大雾越浓）",
                min_value=0.1,
                max_value=3.0,
                value=(0.6, 2.0),
                step=0.1,
            )
            A_min, A_max = st.slider(
                "大气光强度 A 范围（越靠近 1 越偏白、越亮）",
                min_value=0.4,
                max_value=1.0,
                value=(0.7, 1.0),
                step=0.05,
            )

        with col_right:
            uploaded_haze = st.file_uploader(
                "上传一张清晰图片（JPG/PNG）用于合成雾天图像", type=["jpg", "jpeg", "png"]
            )
            run_haze = st.button("生成随机雾图", type="primary")

            if uploaded_haze is None:
                st.info("请先上传一张图片。")
                return

            image = Image.open(uploaded_haze).convert("RGB")
            img_rgb = np.array(image)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            if not run_haze:
                st.info("点击上方“生成随机雾图”按钮查看效果。")
                st.image(img_rgb, width="stretch", caption="原始图像")
                return

            with st.spinner("正在生成雾图，请稍候..."):
                hazy_bgr, info = add_random_haze(
                    img_bgr,
                    beta_range=(beta_min, beta_max),
                    A_range=(A_min, A_max),
                )
            hazy_rgb = cv2.cvtColor(hazy_bgr, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**原始图像**")
                st.image(img_rgb, width="stretch")
            with col2:
                st.markdown("**合成雾天图像**")
                st.image(hazy_rgb, width="stretch")

            st.markdown("### 本次随机雾气参数")
            st.table([{"参数": k, "数值": v} for k, v in info.items()])

            buf = BytesIO()
            Image.fromarray(hazy_rgb).save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                "下载当前雾图（PNG）",
                data=buf,
                file_name="hazy_random.png",
                mime="image/png",
            )
            return

    # ======================
    # 视频处理模式：上传视频 -> 去雾+检测 -> 输出新视频
    # ======================
    if mode == "视频处理（去雾+检测）":
        st.subheader("视频处理（去雾 + 目标检测）")
        st.markdown("上传一段雾天交通视频，系统将对每一帧进行去雾和目标检测，并生成处理后的视频。")

        uploaded_video = st.file_uploader(
            "上传视频文件（MP4/AVI/MOV）", type=["mp4", "avi", "mov"], key="video_upload"
        )
        resize_factor = st.slider(
            "分辨率缩放比例（用于提速，值越小越快）", 0.25, 1.0, 0.5, 0.05, key="video_resize"
        )

        if uploaded_video is not None:
            if st.button("开始处理视频", type="primary", key="process_video_btn"):
                import tempfile

                # 1. 保存上传视频到临时文件
                tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_in.write(uploaded_video.read())
                tmp_in.close()

                tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_out.close()

                app = get_app()
                if app.full_system is None:
                    st.error("请先在左侧 '端到端系统（去雾+检测）' 模式中加载模型权重。")
                    return

                with st.spinner("视频处理中，请稍候（取决于视频长度和分辨率）..."):
                    process_video_with_system(
                        app.full_system,
                        input_path=tmp_in.name,
                        output_path=tmp_out.name,
                        resize_factor=resize_factor,
                    )

                # 2. 读取处理后视频并展示
                with open(tmp_out.name, "rb") as f:
                    video_bytes = f.read()

                st.success("视频处理完成！")
                st.video(video_bytes)
                st.download_button(
                    "下载处理后视频",
                    data=video_bytes,
                    file_name="output_dehaze_detect.mp4",
                    mime="video/mp4",
                )
        else:
            st.info("请先上传视频文件。")

        return

    col_config, col_upload = st.columns([2, 3])

    with col_config:
        st.subheader("模型与参数")

        # 去雾模型：端到端根据“智能识别亮度”开关，决定单模型/智能模式（白天/夜晚权重固定路径）
        if mode in ["端到端系统（去雾+检测）"]:
            if enable_smart_brightness:
                # 智能模式下不在这里“选择/保存”白天或夜晚权重；固定使用 checkpoints/deanet/白天.pk 与 夜晚.pk
                day_model_path = "checkpoints/deanet/白天.pk"
                night_model_path = "checkpoints/deanet/夜晚.pk"
                st.info(
                    "已启用智能识别亮度：系统将根据图片亮度自动选择去雾模型。\n\n"
                    f"- 白天模型：`{day_model_path}`\n"
                    f"- 夜晚模型：`{night_model_path}`"
                )
                if not os.path.exists(day_model_path):
                    st.warning(f"未找到白天模型文件：{day_model_path}")
                if not os.path.exists(night_model_path):
                    st.warning(f"未找到夜晚模型文件：{night_model_path}")

                deanet_choice = None
                day_choice = day_model_path
                night_choice = night_model_path
            else:
                # 关闭智能识别：回到之前的“单去雾模型选择/上传”
                deanet_models = app.scan_deanet_models()
                last_deanet = settings.get("last_deanet_path")
                if isinstance(last_deanet, str) and last_deanet and last_deanet not in deanet_models:
                    if os.path.exists(last_deanet):
                        deanet_models = [last_deanet] + deanet_models
                st.session_state.setdefault("deanet_model", last_deanet if last_deanet in deanet_models else deanet_models[0])
                deanet_choice = st.selectbox("选择去雾模型文件（预设路径）", deanet_models, key="deanet_model")

                deanet_upload = st.file_uploader("或从本地选择去雾模型（.pth/.pt/.pk）", type=["pth", "pt", "pk"], key="deanet_upload")
                if deanet_upload is not None:
                    upload_dir = Path("checkpoints/deanet")
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    temp_path = upload_dir / f"user_{deanet_upload.name}"
                    with open(temp_path, "wb") as f:
                        f.write(deanet_upload.getbuffer())
                    deanet_choice = str(temp_path)
                    st.info(f"已保存去雾模型：{deanet_choice}")

                day_choice = None
                night_choice = None

        # 去雾模型：支持从预设路径选择或本地上传（仅去雾增强）
        elif mode in ["仅去雾增强"]:
            deanet_models = app.scan_deanet_models()
            last_deanet = settings.get("last_deanet_path")
            if isinstance(last_deanet, str) and last_deanet and last_deanet not in deanet_models:
                if os.path.exists(last_deanet):
                    deanet_models = [last_deanet] + deanet_models
            if isinstance(last_deanet, str) and last_deanet in deanet_models:
                st.session_state.setdefault("deanet_model", last_deanet)
            deanet_choice = st.selectbox("选择去雾模型文件（预设路径）", deanet_models, key="deanet_model")

            deanet_upload = st.file_uploader(
                "或从本地选择去雾模型（.pth/.pt/.pk）",
                type=["pth", "pt", "pk"],
                key="deanet_upload",
            )
            if deanet_upload is not None:
                upload_dir = Path("checkpoints/deanet")
                upload_dir.mkdir(parents=True, exist_ok=True)
                temp_path = upload_dir / f"user_{deanet_upload.name}"
                with open(temp_path, "wb") as f:
                    f.write(deanet_upload.getbuffer())
                deanet_choice = str(temp_path)
                st.info(f"已保存去雾模型：{deanet_choice}")
        else:
            deanet_choice = None
            day_choice = None
            night_choice = None

        # 检测模型：支持从预设路径选择或本地上传
        if mode in ["端到端系统（去雾+检测）", "仅目标检测"]:
            yolo_models = app.scan_yolo_models()
            last_yolo = settings.get("last_yolo_path")
            if isinstance(last_yolo, str) and last_yolo and last_yolo not in yolo_models:
                if os.path.exists(last_yolo):
                    yolo_models = [last_yolo] + yolo_models
            if isinstance(last_yolo, str) and last_yolo in yolo_models:
                st.session_state.setdefault("yolo_model", last_yolo)
            yolo_choice = st.selectbox("选择检测模型文件（预设路径）", yolo_models, key="yolo_model")

            yolo_upload = st.file_uploader(
                "或从本地选择检测模型（.pt）",
                type=["pt"],
                key="yolo_upload",
            )
            if yolo_upload is not None:
                upload_dir = Path("checkpoints/yolov11_fca")
                upload_dir.mkdir(parents=True, exist_ok=True)
                temp_path = upload_dir / f"user_{yolo_upload.name}"
                with open(temp_path, "wb") as f:
                    f.write(yolo_upload.getbuffer())
                yolo_choice = str(temp_path)
                st.info(f"已保存检测模型：{yolo_choice}")
        else:
            yolo_choice = None

        def do_load_models(is_auto: bool) -> bool:
            """加载当前模式所需模型；成功则写入配置并返回 True。"""
            messages: list[str] = []

            if mode == "端到端系统（去雾+检测）":
                print(
                    f"[WEB_SYSTEM] 加载端到端模型 "
                    f"(去雾权重: {deanet_choice}, 检测权重: {yolo_choice}, 自动加载: {is_auto})"
                )
                if enable_smart_brightness:
                    # 设置亮度阈值到 app 实例，供自动选择使用
                    app.night_threshold = float(night_threshold)
                    messages.append(
                        app.load_full_system(
                            str(deanet_choice),
                            str(yolo_choice),
                            use_dual_models=True,
                            day_model_path=str(day_choice),
                            night_model_path=str(night_choice),
                        )
                    )
                else:
                    messages.append(
                        app.load_full_system(
                            str(deanet_choice),
                            str(yolo_choice),
                            use_dual_models=False,
                        )
                    )
            elif mode == "仅去雾增强":
                print(
                    f"[WEB_SYSTEM] 加载去雾模型 "
                    f"(去雾权重: {deanet_choice}, 自动加载: {is_auto})"
                )
                messages.append(app.load_dehaze_model(str(deanet_choice)))
            else:  # 仅目标检测
                print(
                    f"[WEB_SYSTEM] 加载检测模型 "
                    f"(检测权重: {yolo_choice}, 自动加载: {is_auto})"
                )
                messages.append(app.load_detect_model(str(yolo_choice)))

            ok = True
            for msg in messages:
                if "失败" in msg or "不存在" in msg or "请选择" in msg:
                    st.error(msg)
                    ok = False
                else:
                    if not is_auto:
                        st.success(msg)

            if ok:
                if mode == "端到端系统（去雾+检测）":
                    # 只保存“去雾模型 + 检测模型”两项；智能模式不保存白天/夜晚路径
                    if isinstance(deanet_choice, str) and deanet_choice:
                        settings["last_deanet_path"] = deanet_choice
                else:
                    if isinstance(deanet_choice, str) and deanet_choice:
                        settings["last_deanet_path"] = deanet_choice
                if isinstance(yolo_choice, str) and yolo_choice:
                    settings["last_yolo_path"] = yolo_choice
                save_settings(settings)
            return ok

        # 自动加载：每个模式在一次会话里只尝试一次，避免反复重载
        auto_key = f"_auto_loaded_once::{mode}"
        if auto_load and not st.session_state.get(auto_key, False):
            # 仅当存在历史模型且文件存在时才自动加载
            can_auto = True
            if mode in ["端到端系统（去雾+检测）"]:
                # 与“智能识别亮度”互斥：能自动加载时只校验单去雾权重
                p = settings.get("last_deanet_path")
                can_auto = can_auto and isinstance(p, str) and os.path.exists(p)
            elif mode in ["仅去雾增强"]:
                p = settings.get("last_deanet_path")
                can_auto = can_auto and isinstance(p, str) and os.path.exists(p)
            if mode in ["端到端系统（去雾+检测）", "仅目标检测"]:
                p = settings.get("last_yolo_path")
                can_auto = can_auto and isinstance(p, str) and os.path.exists(p)

            if can_auto:
                ok = do_load_models(is_auto=True)
                if ok:
                    st.info("已自动加载上次使用的模型。")
                st.session_state[auto_key] = True

        if st.button("加载模型", type="primary"):
            do_load_models(is_auto=False)

    with col_upload:
        st.subheader("图像上传")
        uploaded_file = st.file_uploader(
            "上传一张雾天或交通场景图片（JPG/PNG）", type=["jpg", "jpeg", "png"], key="hazy_upload"
        )
        # 可选：上传清晰参考图，用于真实 PSNR/SSIM 计算
        uploaded_gt = st.file_uploader(
            "（可选）上传对应清晰参考图（用于真实 PSNR/SSIM 计算）",
            type=["jpg", "jpeg", "png"],
            key="gt_upload",
        )
        run = st.button("开始处理", type="primary")

    if not run or uploaded_file is None:
        st.info("请先在左侧加载模型，然后上传图片并点击“开始处理”。")
        return

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    gt_np = None
    if uploaded_gt is not None:
        gt_img = Image.open(uploaded_gt).convert("RGB")
        gt_np = np.array(gt_img)

    with st.spinner("处理中，请稍候..."):
        if mode == "端到端系统（去雾+检测）":
            # 为兼容旧版本（不支持 detection_conf_* 参数）的缓存实例，这里做一次回退尝试
            try:
                result = app.process_full_system(
                    image_np,
                    gt_image=gt_np,
                    enable_pre_detect=enable_pre_detect,
                    detection_conf_pre=float(detect_conf_pre),
                    detection_conf_post=float(detect_conf_post),
                )
            except TypeError:
                # 旧实例：不带 detection_conf 参数
                result = app.process_full_system(
                    image_np,
                    gt_image=gt_np,
                    enable_pre_detect=enable_pre_detect,
                )
            # 兼容旧版本（只返回5个值）的情况
            if isinstance(result, tuple) and len(result) == 6:
                orig, dehazed, det_before, det_after, metrics, targets = result
            else:
                orig, dehazed, det_after, metrics, targets = result  # type: ignore[misc]
                det_before = None
        elif mode == "仅去雾增强":
            orig, dehazed, metrics, targets = app.process_dehaze(image_np, gt_image=gt_np)
            det_before = None
            det_after = None
        else:  # 仅目标检测
            orig, det_after, num, targets = app.process_detect(
                image_np, detection_conf=float(detect_conf_post)
            )
            det_before = None
            dehazed = None
            metrics = {"检测目标数": int(num)} if orig is not None else {}

    if orig is None:
        st.error("处理失败，请检查模型是否已正确加载。")
        return

    # 图像对比区域
    st.markdown("### 图像对比")

    if mode == "端到端系统（去雾+检测）":
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**原始图像**")
            st.image(orig, width="stretch")

        with col2:
            st.markdown("**去雾结果**" if dehazed is not None else "（无去雾结果）")
            if dehazed is not None:
                st.image(dehazed, width="stretch")

        with col3:
            st.markdown("**去雾前检测结果**" if det_before is not None else "（无检测结果）")
            if det_before is not None:
                st.image(det_before, width="stretch")

        with col4:
            st.markdown("**去雾后检测结果**" if det_after is not None else "（无检测结果）")
            if det_after is not None:
                st.image(det_after, width="stretch")
    else:
        # 其他模式保持三列布局
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**原始图像**")
            st.image(orig, width="stretch")

        with col2:
            st.markdown("**去雾结果**" if dehazed is not None else "（无去雾结果）")
            if dehazed is not None:
                st.image(dehazed, width="stretch")

        with col3:
            st.markdown("**检测结果**" if det_after is not None else "（无检测结果）")
            if det_after is not None:
                st.image(det_after, width="stretch")

    # 指标展示
    if metrics:
        st.markdown("### 指标与统计")
        cols = st.columns(len(metrics))
        for (k, v), c in zip(metrics.items(), cols):
            with c:
                st.metric(label=k, value=v)

    # 检测到的目标类型统计
    if targets:
        st.markdown("### 检测目标类型统计")
        st.table(targets)

    # 导出对比图
    export_images = [img for img in [orig, dehazed, det_before, det_after] if img is not None]
    if len(export_images) >= 2:
        st.markdown("### 导出对比图")
        combined = combine_images_horiz(*export_images)
        buf = BytesIO()
        combined.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label="下载当前对比图（PNG）",
            data=buf,
            file_name="comparison.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()

