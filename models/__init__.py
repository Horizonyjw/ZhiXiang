"""模型模块：统一对外接口为 [B, Tin, C, H, W] -> [B, Tout, C, H, W]。"""

from .unet import RadarUNet, ClassicUNet2D

__all__ = ["RadarUNet", "ClassicUNet2D"]
