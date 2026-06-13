"""
端到端雾天交通目标检测系统
DEA-Net + YOLOv11-FCA
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.dehazing.deanet import DEANet
from models.dehazing.generic_dehaze import MPRNetLike
from models.dehazing.deanet_official import DEANetOfficial  # type: ignore
from models.detection.yolov11_fca import YOLOv11FCA


class FoggyTrafficSystem(nn.Module):
    """
    雾天交通目标实时检测与去雾增强系统
    
    将去雾模块和YOLOv11-FCA检测模块集成为端到端系统
    
    Args:
        deanet_weights: 去雾模型预训练权重路径
        yolov11_weights: YOLOv11-FCA预训练权重路径
        num_classes: 检测类别数
        device: 运行设备 ('cuda' 或 'cpu')
        fusion_mode: 融合模式 ('sequential' 或 'joint')
    """
    
    def __init__(self, deanet_weights=None, yolov11_weights=None, 
                 num_classes=10, device='cuda', fusion_mode='sequential'):
        super(FoggyTrafficSystem, self).__init__()
        
        self.num_classes = num_classes
        self.device = device
        self.fusion_mode = fusion_mode
        self.use_dehazing = False
        
        # 初始化去雾模块
        print("🔧 初始化去雾模块...")
        try:
            if deanet_weights:
                # 优先尝试使用官方 DEA-Net 结构加载 .pk 权重
                self.deanet = DEANetOfficial(device=device)
                loaded, total = self.deanet.load_pretrained(deanet_weights, device)
                if loaded == 0:
                    raise RuntimeError("DEA-Net weights not matched")
                self.use_dehazing = True
                print("✅ 去雾模块加载成功 (DEANetOfficial)")
            else:
                raise ValueError("去雾权重路径未提供")
        except Exception as e:
            # 回退到兼容版 MPRNetLike 结构
            print(f"[FoggyTrafficSystem] 使用 DEANetOfficial 加载失败: {e}")
            print("[FoggyTrafficSystem] 回退到 MPRNetLike 去雾网络。")
            try:
                self.deanet = MPRNetLike()
                if deanet_weights:
                    self.deanet.load_pretrained(deanet_weights, device)
                    self.deanet.to(device)
                    self.use_dehazing = True
                    print("✅ 去雾模块加载成功 (MPRNetLike)")
            except Exception as e2:
                print(f"❌ 去雾模块加载失败: {str(e2)}")
                import traceback
                traceback.print_exc()
                raise Exception(f"去雾模块加载失败，请检查权重文件: {str(e2)}")
            
        # 初始化检测模块
        print("🔧 初始化检测模块...")
        from ultralytics import YOLO
        try:
            if yolov11_weights:
                self.yolo_model = YOLO(yolov11_weights)
                print("✅ YOLO检测模块加载成功")
            else:
                raise ValueError("YOLO权重路径未提供")
        except Exception as e:
            print(f"❌ YOLO加载失败: {str(e)}")
            raise
        self.to(device)
        
    def forward(self, hazy_image, return_dehazed=False):
        """
        前向传播
        
        Args:
            hazy_image: 雾天图像, shape [B, 3, H, W]
            return_dehazed: 是否返回去雾后的图像
            
        Returns:
            如果return_dehazed=True: (detections, dehazed_image)
            否则: detections
        """
        # Step 1: 去雾
        dehazed_image = self.deanet(hazy_image)
        
        # Step 2: 目标检测
        detections = self.yolov11_fca(dehazed_image)
        
        if return_dehazed:
            return detections, dehazed_image
        else:
            return detections
    
    def dehaze(self, hazy_image):
        """
        单独执行去雾
        
        Args:
            hazy_image: 雾天图像, shape [B, 3, H, W]
            
        Returns:
            dehazed_image: 去雾后的图像, shape [B, 3, H, W]
        """
        with torch.no_grad():
            dehazed_image = self.deanet(hazy_image)
        return dehazed_image
    
    def detect(self, image):
        """
        单独执行检测
        
        Args:
            image: 输入图像, shape [B, 3, H, W]
            
        Returns:
            detections: 检测结果
        """
        with torch.no_grad():
            detections = self.yolov11_fca(image)
        return detections
    
    def predict(self, hazy_image):
        """
        端到端预测（推理模式）
        
        Args:
            hazy_image: 雾天图像, numpy array [H, W, 3] (RGB)
            
        Returns:
            detections: YOLO检测结果
            dehazed_image: 去雾后的图像 numpy array
        """
        # 只设置去雾模块为eval模式，不要调用self.eval()
        if self.use_dehazing and self.deanet is not None:
            self.deanet.eval()
        
        # 转换numpy array到tensor
        if not isinstance(hazy_image, torch.Tensor):
            # 归一化到[0, 1]
            if hazy_image.max() > 1.0:
                hazy_image = hazy_image.astype('float32') / 255.0
            
            # HWC -> CHW
            hazy_image = torch.from_numpy(hazy_image).permute(2, 0, 1).float()
            hazy_image = hazy_image.unsqueeze(0)  # Add batch dimension
                
        hazy_image = hazy_image.to(self.device)
        
        with torch.no_grad():
            # Step 1: 去雾
            if self.use_dehazing and self.deanet is not None:
                dehazed_tensor = self.deanet(hazy_image)
                # 强制限制到[0, 1]范围
                dehazed_tensor = torch.clamp(dehazed_tensor, 0, 1)
            else:
                dehazed_tensor = hazy_image
            
            # 转换为numpy用于YOLO
            dehazed_np = dehazed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # 确保在[0, 255]范围
            dehazed_np = np.clip(dehazed_np * 255, 0, 255).astype('uint8')
            
            # Step 2: YOLO检测 - 直接调用predict方法
            detections = self.yolo_model.predict(dehazed_np, verbose=False)
            
        return detections, dehazed_np
    
    def freeze_deanet(self):
        """冻结DEA-Net参数（只训练检测器）"""
        for param in self.deanet.parameters():
            param.requires_grad = False
        print("DEA-Net parameters frozen.")
        
    def unfreeze_deanet(self):
        """解冻DEA-Net参数"""
        for param in self.deanet.parameters():
            param.requires_grad = True
        print("DEA-Net parameters unfrozen.")
        
    def freeze_detector(self):
        """冻结检测器参数"""
        for param in self.yolov11_fca.parameters():
            param.requires_grad = False
        print("Detector parameters frozen.")
        
    def unfreeze_detector(self):
        """解冻检测器参数"""
        for param in self.yolov11_fca.parameters():
            param.requires_grad = True
        print("Detector parameters unfrozen.")
    
    def get_trainable_params(self):
        """获取可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_num_params(self):
        """获取模型总参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }
    
    def save(self, save_path):
        """保存模型"""
        torch.save({
            'deanet': self.deanet.state_dict(),
            'yolov11_fca': self.yolov11_fca.state_dict(),
            'num_classes': self.num_classes,
            'fusion_mode': self.fusion_mode
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load(self, checkpoint_path):
        """加载模型"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.deanet.load_state_dict(checkpoint['deanet'])
        self.yolov11_fca.load_state_dict(checkpoint['yolov11_fca'])
        
        print("Checkpoint loaded successfully!")


if __name__ == '__main__':
    # 测试代码
    print("Testing FoggyTrafficSystem...")
    
    # 创建系统
    system = FoggyTrafficSystem(num_classes=10, device='cpu')
    
    # 测试前向传播
    x = torch.randn(1, 3, 640, 640)
    
    try:
        detections, dehazed = system.predict(x)
        print(f"Input shape: {x.shape}")
        print(f"Dehazed shape: {dehazed.shape}")
        print("System test successful!")
    except Exception as e:
        print(f"System test failed: {e}")
    
    # 打印参数信息
    params_info = system.get_num_params()
    print(f"\nParameter info:")
    print(f"  Total: {params_info['total'] / 1e6:.2f}M")
    print(f"  Trainable: {params_info['trainable'] / 1e6:.2f}M")
    print(f"  Frozen: {params_info['frozen'] / 1e6:.2f}M")
