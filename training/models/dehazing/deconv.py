"""
DEConv: Detail-Enhanced Convolution
通过重参数化技术实现零额外参数和计算开销
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DEConv(nn.Module):
    """
    Detail-Enhanced Convolution
    
    在训练时使用多分支结构增强细节表达能力
    在推理时通过重参数化合并为单个卷积，实现零额外开销
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        groups: 分组卷积
        bias: 是否使用偏置
        deploy: 是否为部署模式（重参数化后）
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, groups=1, bias=True, deploy=False):
        super(DEConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.deploy = deploy
        
        if deploy:
            # 部署模式：单个卷积
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                stride, padding, groups=groups, bias=True)
        else:
            # 训练模式：多分支结构
            # 主分支：标准卷积
            self.conv_main = nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding, groups=groups, bias=False)
            
            # 细节增强分支：1x1卷积
            self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                     stride=stride, groups=groups, bias=False)
            
            # 恒等映射分支（当输入输出通道相同时）
            if in_channels == out_channels and stride == 1:
                self.identity = nn.BatchNorm2d(in_channels)
            else:
                self.identity = None
                
            # Batch Normalization
            self.bn = nn.BatchNorm2d(out_channels)
            
    def forward(self, x):
        """前向传播"""
        if self.deploy:
            # 部署模式：直接使用重参数化后的卷积
            return self.conv(x)
        else:
            # 训练模式：多分支融合
            out = self.conv_main(x)
            out += self.conv_1x1(x)
            
            if self.identity is not None:
                out += self.identity(x)
                
            out = self.bn(out)
            return out
    
    def switch_to_deploy(self):
        """
        切换到部署模式（重参数化）
        将多分支结构融合为单个卷积
        """
        if self.deploy:
            return
            
        # 获取融合后的权重和偏置
        kernel, bias = self._get_equivalent_kernel_bias()
        
        # 创建新的卷积层
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 
                             self.kernel_size, self.stride, self.padding,
                             groups=self.groups, bias=True)
        
        # 加载融合后的权重
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        
        # 删除训练时的多分支
        self.__delattr__('conv_main')
        self.__delattr__('conv_1x1')
        if self.identity is not None:
            self.__delattr__('identity')
        self.__delattr__('bn')
        
        self.deploy = True
        
    def _get_equivalent_kernel_bias(self):
        """获取等价的卷积核和偏置"""
        # 主分支的kernel
        kernel_main = self.conv_main.weight
        
        # 1x1分支的kernel，需要padding到3x3
        kernel_1x1 = self.conv_1x1.weight
        if self.kernel_size > 1:
            pad = (self.kernel_size - 1) // 2
            kernel_1x1 = F.pad(kernel_1x1, [pad, pad, pad, pad])
        
        # 恒等映射的kernel
        if self.identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_identity = torch.zeros((self.in_channels, input_dim, 
                                         self.kernel_size, self.kernel_size),
                                        dtype=kernel_main.dtype,
                                        device=kernel_main.device)
            for i in range(self.in_channels):
                kernel_identity[i, i % input_dim, 
                              self.kernel_size // 2, 
                              self.kernel_size // 2] = 1
        else:
            kernel_identity = 0
            
        # 融合所有分支
        kernel = kernel_main + kernel_1x1 + kernel_identity
        
        # 融合BN
        std = (self.bn.running_var + self.bn.eps).sqrt()
        t = (self.bn.weight / std).reshape(-1, 1, 1, 1)
        kernel = kernel * t
        bias = self.bn.bias - self.bn.running_mean * self.bn.weight / std
        
        return kernel, bias


if __name__ == '__main__':
    # 测试代码
    print("Testing DEConv...")
    
    # 创建训练模式的DEConv
    deconv = DEConv(64, 64, kernel_size=3, stride=1, padding=1, deploy=False)
    x = torch.randn(1, 64, 32, 32)
    
    # 训练模式
    y_train = deconv(x)
    print(f"Training mode output shape: {y_train.shape}")
    
    # 切换到部署模式
    deconv.switch_to_deploy()
    y_deploy = deconv(x)
    print(f"Deploy mode output shape: {y_deploy.shape}")
    
    # 验证结果一致性
    diff = (y_train - y_deploy).abs().max()
    print(f"Max difference: {diff.item():.6f}")
