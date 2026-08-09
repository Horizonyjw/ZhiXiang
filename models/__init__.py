"""模型模块：统一对外接口为 [B, Tin, C, H, W] -> [B, Tout, C, H, W]。"""

from .fno import FNO2d, RadarFNO
from .persistence import Persistence
from .unet import ClassicUNet2D, RadarUNet

__all__ = [
    "ClassicUNet2D",
    "FNO2d",
    "Persistence",
    "RadarFNO",
    "RadarUNet",
]
