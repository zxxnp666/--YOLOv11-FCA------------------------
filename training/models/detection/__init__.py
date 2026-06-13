"""
目标检测模型模块
"""

from .fca_attention import FCAModule
from .yolov11_fca import YOLOv11FCA

__all__ = ['FCAModule', 'YOLOv11FCA']
