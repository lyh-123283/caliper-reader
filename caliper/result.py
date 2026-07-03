"""
游标卡尺识别 — 数据结构定义
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class TickInfo:
    """单条刻度线信息"""
    x: int             # x 坐标（像素）
    y_start: int       # 刻线起点 y
    y_end: int         # 刻线终点 y
    y_mid: int         # 刻线中点 y
    length: int        # 刻线长度（像素）
    is_long: bool = False  # 是否为长刻线（cm标记）


@dataclass
class DigitInfo:
    """OCR 识别的数字信息"""
    x: int             # 数字中心 x 坐标
    y: int             # 数字中心 y 坐标
    value: int         # 数字值
    text: str          # 原始识别文本
    confidence: float  # 识别置信度
    bbox: tuple = None # (x1, y1, x2, y2) 边界框


@dataclass
class CaliperResult:
    """最终识别结果"""
    main_scale: float        # 主尺整数读数 (mm)
    vernier_scale: float     # 游标小数读数 (mm)
    total: float             # 总读数 (mm)
    precision: float         # 检测精度，当前固定为 0.02mm
    confidence: float        # 总置信度 (0~1)
    image_annotated: np.ndarray = None  # 最终标注图像
    debug_images: dict = field(default_factory=dict)  # 各步骤中间图像
    extra_info: dict = field(default_factory=dict)    # 额外信息
