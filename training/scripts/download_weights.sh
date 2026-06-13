#!/bin/bash
# DEA-Net 预训练权重下载脚本

echo "========================================"
echo "下载 DEA-Net 预训练权重"
echo "========================================"
echo ""

# 创建权重目录
mkdir -p checkpoints/deanet/{ITS,OTS,HAZE4K}

echo "预训练权重下载链接："
echo "- Google Drive: https://drive.google.com/drive/folders/1Rjb8dpyNnvvr0XLvIX9fg8Hdru_MhMCj"
echo "- 百度网盘: https://pan.baidu.com/s/1retfKIs_Om-D4zA45sL6Kg (密码: dcyb)"
echo ""
echo "请手动下载以下文件并放置到对应目录："
echo ""
echo "1. PSNR4131_SSIM9945.pth (ITS权重)"
echo "   → checkpoints/deanet/ITS/"
echo ""
echo "2. PSNR3659_SSIM9897.pth (OTS权重)"
echo "   → checkpoints/deanet/OTS/"
echo ""
echo "3. PSNR3426_SSIM9985.pth (HAZE4K权重)"
echo "   → checkpoints/deanet/HAZE4K/"
echo ""
echo "========================================"
echo "推荐使用 ITS 权重（性能最好）"
echo "========================================"
echo ""

# 如果有 gdown 可以自动下载（需要 Google Drive 文件ID）
# pip install gdown
# gdown --id <FILE_ID> -O checkpoints/deanet/ITS/PSNR4131_SSIM9945.pth
