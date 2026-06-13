"""
去雾效果可视化UI
支持选择模型权重和测试图片，实时查看去雾效果
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import ImageTk

# 添加code目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
from model import DEANet


class DehazeUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DEA-Net 去雾效果查看器")
        self.root.geometry("1400x800")
        
        self.model = None
        self.current_image = None
        self.dehazed_image = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # 顶部控制面板
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # 模型选择
        ttk.Label(control_frame, text="模型权重:").pack(side=tk.LEFT, padx=5)
        self.model_path_var = tk.StringVar()
        model_entry = ttk.Entry(control_frame, textvariable=self.model_path_var, width=50)
        model_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="选择模型", command=self.select_model).pack(side=tk.LEFT, padx=5)
        
        # Base dim选择
        ttk.Label(control_frame, text="Base Dim:").pack(side=tk.LEFT, padx=5)
        self.base_dim_var = tk.StringVar(value="32")
        base_dim_combo = ttk.Combobox(control_frame, textvariable=self.base_dim_var, 
                                      values=["24", "32"], width=5, state="readonly")
        base_dim_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="加载模型", command=self.load_model).pack(side=tk.LEFT, padx=5)
        
        # 图片选择
        ttk.Button(control_frame, text="选择图片", command=self.select_image).pack(side=tk.LEFT, padx=20)
        ttk.Button(control_frame, text="去雾处理", command=self.process_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="保存结果", command=self.save_result).pack(side=tk.LEFT, padx=5)
        
        # 状态栏
        status_frame = ttk.Frame(self.root, padding="5")
        status_frame.pack(side=tk.TOP, fill=tk.X)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var, foreground="blue").pack(side=tk.LEFT)
        
        # 图片显示区域
        image_frame = ttk.Frame(self.root)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 原图
        left_frame = ttk.LabelFrame(image_frame, text="原图（有雾）", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.original_label = ttk.Label(left_frame)
        self.original_label.pack(fill=tk.BOTH, expand=True)
        
        # 去雾后
        right_frame = ttk.LabelFrame(image_frame, text="去雾后", padding="10")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.dehazed_label = ttk.Label(right_frame)
        self.dehazed_label.pack(fill=tk.BOTH, expand=True)
        
        # 信息显示
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.info_text = tk.Text(info_frame, height=4, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X)
        self.info_text.insert(1.0, "使用说明：\n1. 选择模型权重文件（.pk文件）\n2. 选择Base Dim（24或32）\n3. 点击'加载模型'\n4. 选择要去雾的图片\n5. 点击'去雾处理'查看效果")
        self.info_text.config(state=tk.DISABLED)
    
    def select_model(self):
        filename = filedialog.askopenfilename(
            title="选择模型权重文件",
            filetypes=[("PyTorch模型", "*.pk"), ("所有文件", "*.*")],
            initialdir="./experiment"
        )
        if filename:
            self.model_path_var.set(filename)
    
    def load_model(self):
        model_path = self.model_path_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("错误", "请选择有效的模型文件")
            return
        
        try:
            base_dim = int(self.base_dim_var.get())
            self.status_var.set(f"正在加载模型 (base_dim={base_dim})...")
            self.root.update()
            
            # 创建模型
            self.model = DEANet(base_dim=base_dim)
            
            # 加载权重
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            
            # 处理DataParallel前缀
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # 显示模型信息
            info = f"✓ 模型加载成功\n"
            info += f"  Base Dim: {base_dim}\n"
            info += f"  参数量: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M\n"
            if 'max_psnr' in checkpoint:
                info += f"  训练PSNR: {checkpoint['max_psnr']:.2f} dB\n"
            if 'epoch' in checkpoint:
                info += f"  训练Epoch: {checkpoint['epoch']}\n"
            
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
            self.info_text.config(state=tk.DISABLED)
            
            self.status_var.set("模型加载成功，请选择图片")
            
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败：{str(e)}")
            self.status_var.set("模型加载失败")
    
    def select_image(self):
        filename = filedialog.askopenfilename(
            title="选择测试图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")]
        )
        if filename:
            try:
                self.current_image = Image.open(filename).convert('RGB')
                self.display_image(self.current_image, self.original_label)
                self.status_var.set(f"已加载图片: {os.path.basename(filename)}")
                
                # 清空去雾结果
                self.dehazed_label.config(image='')
                self.dehazed_image = None
                
            except Exception as e:
                messagebox.showerror("错误", f"图片加载失败：{str(e)}")
    
    def process_image(self):
        if self.model is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        if self.current_image is None:
            messagebox.showwarning("警告", "请先选择图片")
            return
        
        try:
            self.status_var.set("正在处理...")
            self.root.update()
            
            # 转换为tensor
            to_tensor = ToTensor()
            img_tensor = to_tensor(self.current_image).unsqueeze(0)
            
            # Padding到4的倍数
            _, _, h, w = img_tensor.shape
            mod_pad_h = (4 - h % 4) % 4
            mod_pad_w = (4 - w % 4) % 4
            img_tensor = F.pad(img_tensor, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            
            # 推理
            with torch.no_grad():
                output = self.model(img_tensor)
                output = output[:, :, :h, :w].clamp(0, 1)
            
            # 转换回PIL图像
            to_pil = ToPILImage()
            self.dehazed_image = to_pil(output.squeeze(0))
            
            # 显示结果
            self.display_image(self.dehazed_image, self.dehazed_label)
            self.status_var.set("处理完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"处理失败：{str(e)}")
            self.status_var.set("处理失败")
    
    def display_image(self, pil_image, label):
        # 调整图像大小以适应显示区域
        display_size = (600, 600)
        pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # 转换为PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        label.config(image=photo)
        label.image = photo  # 保持引用
    
    def save_result(self):
        if self.dehazed_image is None:
            messagebox.showwarning("警告", "没有可保存的结果")
            return
        
        filename = filedialog.asksaveasfilename(
            title="保存去雾结果",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                self.dehazed_image.save(filename)
                self.status_var.set(f"已保存到: {filename}")
                messagebox.showinfo("成功", "结果已保存")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败：{str(e)}")


def main():
    root = tk.Tk()
    app = DehazeUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
