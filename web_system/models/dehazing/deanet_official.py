import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_dea_root() -> Path:
    """返回 DEA-Net-main/code 目录路径。"""
    root = Path(__file__).resolve().parents[3]  # 项目根目录 D:\windsurf\biyesheji
    dea_code = root / "DEA-Net-main" / "code"
    return dea_code


class DEANetOfficial(nn.Module):
    """
    封装 DEA-Net-main 中的 DEANet 结构，用于在 Web 端直接加载 .pk 权重。
    """

    def __init__(self, base_dim: int = 32, device: str = "cuda"):
        super().__init__()

        dea_code = _get_dea_root()
        sys.path.append(str(dea_code))
        # 延迟导入，避免全局环境污染
        from model.backbone_train import DEANet as _DEANet  # type: ignore

        self.net = _DEANet(base_dim=base_dim)
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DEA-Net 要求输入尺寸在编码/解码过程中保持对齐，
        # 对于任意尺寸的图像，这里先 pad 到 4 的倍数，再在输出时裁剪回原尺寸。
        b, c, h, w = x.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h != 0 or pad_w != 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_padded = x

        out = self.net(x_padded)

        if pad_h != 0 or pad_w != 0:
            out = out[:, :, :h, :w]
        return out

    def load_pretrained(self, weights_path: str, device: str = "cuda"):
        """按 DEA-Net 官方方式加载 .pk 权重，并输出简要统计信息。"""
        print(f"[DEANetOfficial] Loading DEA-Net weights from: {weights_path}")
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)

        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        model_dict = self.net.state_dict()
        loaded, skipped = 0, 0
        new_dict = {}
        for k, v in state_dict.items():
            # 原 DEA-Net 训练时通常以 "module." 前缀保存
            k_fixed = k.replace("module.", "", 1) if k.startswith("module.") else k
            if k_fixed in model_dict and model_dict[k_fixed].shape == v.shape:
                new_dict[k_fixed] = v
                loaded += 1
            else:
                skipped += 1

        model_dict.update(new_dict)
        self.net.load_state_dict(model_dict)

        print(f"[DEANetOfficial] Loaded {loaded}/{len(state_dict)} keys (skipped {skipped}).")
        if loaded == 0:
            print(
                "[DEANetOfficial] ❌ 未成功加载任何 DEA-Net 权重，请检查权重文件是否来自 DEA-Net-main 的训练结果。"
            )
        elif loaded < len(state_dict) * 0.5:
            print(
                "[DEANetOfficial] ⚠️  仅加载了少部分权重，去雾效果可能弱于论文结果。"
            )

        self.to(device)
        self.eval()
        return loaded, len(state_dict)

