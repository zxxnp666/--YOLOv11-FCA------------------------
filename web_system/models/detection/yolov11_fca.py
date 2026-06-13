"""
YOLOv11-FCA: YOLOv11 with Fast Channel Attention
基于YOLOv11集成FCA注意力机制的目标检测模型
"""

import torch
import torch.nn as nn
from .fca_attention import FCAModule

try:
    from ultralytics import YOLO
    from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLOv11FCA will use simplified version.")


class C2f_FCA(nn.Module):
    """
    C2f模块集成FCA注意力
    """
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)])
        
        # 添加FCA注意力
        self.fca = FCAModule(c2, reduction=16)
        
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        
        # 应用FCA注意力
        out = self.fca(out)
        
        return out


class Bottleneck(nn.Module):
    """Standard bottleneck"""
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class YOLOv11FCA(nn.Module):
    """
    YOLOv11-FCA检测模型
    
    在YOLOv11的Backbone和Neck中集成FCA注意力机制
    
    Args:
        cfg: 模型配置文件路径或配置字典
        num_classes: 检测类别数
        pretrained: 预训练权重路径
    """
    
    def __init__(self, cfg='yolov11n.yaml', num_classes=80, pretrained=None):
        super(YOLOv11FCA, self).__init__()
        
        self.num_classes = num_classes
        
        if ULTRALYTICS_AVAILABLE:
            # 使用ultralytics的YOLOv11
            self._init_with_ultralytics(cfg, pretrained)
        else:
            # 使用简化版本
            self._init_simplified()
            
    def _init_with_ultralytics(self, cfg, pretrained):
        """使用ultralytics初始化"""
        # 加载YOLOv11模型
        if pretrained:
            self.model = YOLO(pretrained)
        else:
            self.model = YOLO(cfg)
            
        # TODO: 在模型中插入FCA模块
        # 这需要修改ultralytics的模型结构
        print("Warning: FCA integration with ultralytics is not fully implemented yet.")
        print("Please use the custom training script or modify the model manually.")
        
    def _init_simplified(self):
        """简化版本初始化（用于代码框架）"""
        # 这是一个简化的框架，实际使用时需要完整实现
        self.backbone = nn.Sequential(
            Conv(3, 64, 6, 2, 2),
            Conv(64, 128, 3, 2),
            C2f_FCA(128, 128, 3),
            Conv(128, 256, 3, 2),
            C2f_FCA(256, 256, 6),
            Conv(256, 512, 3, 2),
            C2f_FCA(512, 512, 6),
            Conv(512, 1024, 3, 2),
            C2f_FCA(1024, 1024, 3),
        )
        
        print("Using simplified YOLOv11FCA framework.")
        print("For full implementation, please install ultralytics: pip install ultralytics")
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像, shape [B, 3, H, W]
            
        Returns:
            如果是训练模式，返回损失
            如果是推理模式，返回检测结果
        """
        if ULTRALYTICS_AVAILABLE and hasattr(self, 'model'):
            return self.model(x)
        else:
            # 简化版本的前向传播
            return self.backbone(x)
    
    def load_pretrained(self, weights_path):
        """加载预训练权重"""
        print(f"Loading weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        if 'model' in checkpoint:
            self.load_state_dict(checkpoint['model'], strict=False)
        else:
            self.load_state_dict(checkpoint, strict=False)
            
        print("Weights loaded successfully!")
    
    def save(self, save_path):
        """保存模型"""
        torch.save({
            'model': self.state_dict(),
            'num_classes': self.num_classes,
        }, save_path)
        print(f"Model saved to {save_path}")


class Conv(nn.Module):
    """Standard convolution with BN and activation"""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


if __name__ == '__main__':
    # 测试代码
    print("Testing YOLOv11FCA...")
    
    # 创建模型
    model = YOLOv11FCA(num_classes=10)
    
    # 测试前向传播
    x = torch.randn(1, 3, 640, 640)
    
    try:
        out = model(x)
        print(f"Input shape: {x.shape}")
        if isinstance(out, torch.Tensor):
            print(f"Output shape: {out.shape}")
        print("Forward pass successful!")
    except Exception as e:
        print(f"Forward pass failed: {e}")
    
    # 计算参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params / 1e6:.2f}M")
