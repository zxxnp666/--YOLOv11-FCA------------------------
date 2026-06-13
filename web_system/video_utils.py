import cv2
import numpy as np
import torch


def process_video_with_system(system, input_path: str, output_path: str, resize_factor: float = 1.0):
    """
    使用端到端系统对整段视频做去雾 + 检测。

    Args:
        system: FoggyTrafficSystem 实例（已加载好去雾和检测权重）
        input_path: 输入视频路径
        output_path: 输出视频路径
        resize_factor: 分辨率缩放比例，用于提速（0.25~1.0）
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开输入视频: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if resize_factor != 1.0:
        out_w, out_h = int(w * resize_factor), int(h * resize_factor)
    else:
        out_w, out_h = w, h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频: {output_path}")

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        if resize_factor != 1.0:
            frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            detections, dehazed_rgb = system.predict(frame_rgb)

        # 在去雾后图像上绘制检测结果
        if detections and len(detections) > 0:
            result = detections[0]
            vis_bgr = result.plot()  # ultralytics 的 plot 返回 BGR 图像
        else:
            vis_bgr = cv2.cvtColor(dehazed_rgb, cv2.COLOR_RGB2BGR)

        if vis_bgr.shape[1] != out_w or vis_bgr.shape[0] != out_h:
            vis_bgr = cv2.resize(vis_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

        out.write(vis_bgr)

    cap.release()
    out.release()

