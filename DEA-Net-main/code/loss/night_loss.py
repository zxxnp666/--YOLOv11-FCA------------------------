"""
夜晚场景专用损失函数
针对低光照、光源干扰、颜色偏移等问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NightAwareLoss(nn.Module):
    """
    夜晚感知损失
    结合亮度自适应、光源抑制、颜色校正
    """
    def __init__(self, w_brightness=0.3, w_light_suppress=0.2, w_color=0.1):
        super(NightAwareLoss, self).__init__()
        self.w_brightness = w_brightness
        self.w_light_suppress = w_light_suppress
        self.w_color = w_color
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测图像 [B, 3, H, W]
            target: 目标图像 [B, 3, H, W]
        """
        # 1. 基础L1损失
        loss_l1 = self.l1_loss(pred, target)
        
        # 2. 亮度自适应损失（对暗区域更敏感）
        loss_brightness = self.brightness_adaptive_loss(pred, target)
        
        # 3. 光源抑制损失（抑制过亮区域）
        loss_light = self.light_suppression_loss(pred, target)
        
        # 4. 颜色一致性损失
        loss_color = self.color_consistency_loss(pred, target)
        
        # 总损失
        total_loss = (loss_l1 + 
                     self.w_brightness * loss_brightness +
                     self.w_light_suppress * loss_light +
                     self.w_color * loss_color)
        
        return total_loss
    
    def brightness_adaptive_loss(self, pred, target):
        """
        亮度自适应损失
        对暗区域（低光照）给予更高权重
        """
        # 计算亮度（灰度图）
        pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        
        # 亮度权重：暗区域权重稍高，但不要过大
        # 使用更温和的权重函数，避免极端值
        brightness_weight = 1.0 / (target_gray + 0.3)  # 从0.1改为0.3，减少权重差异
        brightness_weight = torch.clamp(brightness_weight, 0.5, 3.0)  # 限制权重范围
        brightness_weight = brightness_weight / brightness_weight.mean()  # 归一化
        
        # 加权L1损失
        diff = torch.abs(pred_gray - target_gray)
        weighted_loss = (diff * brightness_weight).mean()
        
        return weighted_loss
    
    def light_suppression_loss(self, pred, target):
        """
        光源抑制损失
        抑制过亮区域（车灯、路灯）的影响
        """
        # 检测过亮区域（阈值0.8）
        pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        
        # 过亮区域mask
        bright_mask = (target_gray > 0.8).float()
        
        # 对过亮区域降低权重
        weight = 1.0 - 0.5 * bright_mask  # 过亮区域权重减半
        
        # 加权损失
        diff = torch.abs(pred - target).mean(dim=1)  # [B, H, W]
        weighted_loss = (diff * weight).mean()
        
        return weighted_loss
    
    def color_consistency_loss(self, pred, target):
        """
        颜色一致性损失
        保持RGB通道间的比例关系
        """
        # 计算RGB通道比例
        pred_ratio_rg = pred[:, 0] / (pred[:, 1] + 1e-6)
        pred_ratio_rb = pred[:, 0] / (pred[:, 2] + 1e-6)
        
        target_ratio_rg = target[:, 0] / (target[:, 1] + 1e-6)
        target_ratio_rb = target[:, 0] / (target[:, 2] + 1e-6)
        
        # 比例差异
        loss_rg = F.l1_loss(pred_ratio_rg, target_ratio_rg)
        loss_rb = F.l1_loss(pred_ratio_rb, target_ratio_rb)
        
        return (loss_rg + loss_rb) / 2


class IlluminationEnhancementLoss(nn.Module):
    """
    光照增强损失
    鼓励模型提升暗区域亮度
    """
    def __init__(self, target_brightness=0.5):
        super(IlluminationEnhancementLoss, self).__init__()
        self.target_brightness = target_brightness
    
    def forward(self, pred, target):
        """
        鼓励预测图像的暗区域亮度接近目标图像的亮度（而不是固定值）
        """
        # 计算亮度
        pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        
        # 找出暗区域（亮度<0.3）
        dark_mask = (target_gray < 0.3).float()
        
        if dark_mask.sum() > 0:
            # 计算暗区域的预测和目标亮度
            pred_dark = pred_gray * dark_mask
            target_dark = target_gray * dark_mask
            
            # 让预测的暗区域亮度接近目标的暗区域亮度
            loss = F.l1_loss(pred_dark, target_dark)
        else:
            loss = torch.tensor(0.0, device=pred.device)
        
        return loss


class PerceptualLoss(nn.Module):
    """
    感知损失（使用VGG特征）
    保持高层语义信息
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # 使用预训练VGG16的前几层
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features
            self.feature_extractor = nn.Sequential(*list(vgg.children())[:16]).eval()
            
            # 冻结参数
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        except:
            print("Warning: VGG16 not available, PerceptualLoss disabled")
            self.feature_extractor = None
    
    def forward(self, pred, target):
        if self.feature_extractor is None:
            return torch.tensor(0.0, device=pred.device)
        
        # 提取特征
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        # L2损失
        loss = F.mse_loss(pred_features, target_features)
        
        return loss


class NightDehazeLoss(nn.Module):
    """
    夜晚去雾综合损失
    整合所有夜晚场景优化
    """
    def __init__(self, 
                 w_l1=1.0,
                 w_night_aware=0.3,  # 降低权重
                 w_illumination=0.2,  # 降低权重
                 w_perceptual=0.05,  # 降低权重
                 w_cr=0.1):
        super(NightDehazeLoss, self).__init__()
        
        self.w_l1 = w_l1
        self.w_night_aware = w_night_aware
        self.w_illumination = w_illumination
        self.w_perceptual = w_perceptual
        self.w_cr = w_cr
        
        self.l1_loss = nn.L1Loss()
        self.night_aware_loss = NightAwareLoss(w_brightness=0.2, w_light_suppress=0.1, w_color=0.05)
        self.illumination_loss = IlluminationEnhancementLoss()
        self.perceptual_loss = PerceptualLoss()
        
        # CR损失（如果需要）
        try:
            from loss.cr import ContrastLoss
            self.cr_loss = ContrastLoss()
        except:
            try:
                from loss import ContrastLoss
                self.cr_loss = ContrastLoss()
            except:
                print("Warning: ContrastLoss not available")
                self.cr_loss = None
    
    def forward(self, pred, target, input_img=None):
        """
        综合损失计算
        
        Args:
            pred: 预测图像
            target: 目标图像
            input_img: 输入图像（用于CR损失）
        """
        losses = {}
        
        # 1. L1损失
        losses['l1'] = self.l1_loss(pred, target)
        
        # 2. 夜晚感知损失
        losses['night_aware'] = self.night_aware_loss(pred, target)
        
        # 3. 光照增强损失
        losses['illumination'] = self.illumination_loss(pred, target)
        
        # 4. 感知损失
        losses['perceptual'] = self.perceptual_loss(pred, target)
        
        # 5. CR损失
        if self.cr_loss is not None and self.w_cr > 0 and input_img is not None:
            losses['cr'] = self.cr_loss(pred, target, input_img)
        else:
            losses['cr'] = torch.tensor(0.0, device=pred.device)
        
        # 总损失
        total_loss = (self.w_l1 * losses['l1'] +
                     self.w_night_aware * losses['night_aware'] +
                     self.w_illumination * losses['illumination'] +
                     self.w_perceptual * losses['perceptual'] +
                     self.w_cr * losses['cr'])
        
        losses['total'] = total_loss
        
        return total_loss, losses


# 便捷函数
def get_night_loss(loss_type='full'):
    """
    获取夜晚场景损失函数
    
    Args:
        loss_type: 'full', 'night_aware', 'illumination'
    """
    if loss_type == 'full':
        return NightDehazeLoss()
    elif loss_type == 'night_aware':
        return NightAwareLoss()
    elif loss_type == 'illumination':
        return IlluminationEnhancementLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
