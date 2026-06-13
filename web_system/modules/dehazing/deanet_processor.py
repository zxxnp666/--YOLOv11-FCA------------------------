"""
DEA-Net去雾处理器
封装DEA-Net模型，提供简单易用的API
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.dehazing.deanet import DEANet


class DEANetProcessor:
    """
    DEA-Net去雾处理器
    
    Args:
        checkpoint_path: 预训练权重路径
        device: 运行设备
    """
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 初始化模型
        self.model = DEANet()
        self.model.load_pretrained(checkpoint_path, self.device)
        self.model.eval()
        self.model.to(self.device)
        
        print(f"DEA-Net initialized on {self.device}")
        
    def process(self, image):
        """
        处理单张图像
        
        Args:
            image: PIL Image, numpy array, 或 torch.Tensor
            
        Returns:
            dehazed: 去雾后的图像 (与输入格式相同)
        """
        # 记录输入格式
        input_type = type(image)
        
        # 转换为tensor
        if isinstance(image, Image.Image):
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            if len(image.shape) == 3:
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            else:
                image = torch.from_numpy(image)
        
        image = image.to(self.device)
        
        # 去雾
        with torch.no_grad():
            dehazed = self.model(image)
            dehazed = torch.clamp(dehazed, 0, 1)
        
        # 转换回输入格式
        if input_type == Image.Image:
            dehazed = dehazed.squeeze(0).permute(1, 2, 0).cpu().numpy()
            dehazed = (dehazed * 255).astype(np.uint8)
            dehazed = Image.fromarray(dehazed)
        elif input_type == np.ndarray:
            dehazed = dehazed.squeeze(0).permute(1, 2, 0).cpu().numpy()
            dehazed = (dehazed * 255).astype(np.uint8)
        
        return dehazed
    
    def process_batch(self, images):
        """批量处理"""
        results = []
        for img in images:
            dehazed = self.process(img)
            results.append(dehazed)
        return results


if __name__ == '__main__':
    # 测试代码
    print("Testing DEANetProcessor...")
    
    # 创建处理器
    # processor = DEANetProcessor('checkpoints/deanet/ITS/PSNR4131_SSIM9945.pth')
    
    # 创建测试图像
    test_img = Image.new('RGB', (256, 256), color='gray')
    
    # dehazed = processor.process(test_img)
    # print(f"Dehazed image size: {dehazed.size}")
    
    print("Please provide a valid checkpoint path to test.")
