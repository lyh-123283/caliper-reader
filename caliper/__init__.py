"""
游标卡尺读数识别 — caliper 包

流水线架构:
  步骤 0: 图像预处理（增强 + 滤波 + 二值化）
  步骤 1: ROI提取 + 方向矫正
  步骤 2: 区域分离（主尺 / 游标尺）
  步骤 3: 主尺识别（刻度线 + 数字OCR）
  步骤 4: 游标尺识别（刻度线 + 对齐 + 数字OCR）
  步骤 5: 读数合并 + 最终标注
"""

from .result import CaliperResult, TickInfo, DigitInfo
from .pipeline import CaliperPipeline, read_caliper, read_caliper_from_array
from .config import config

__version__ = "4.0.0"
__all__ = [
    'CaliperPipeline',
    'CaliperResult',
    'TickInfo',
    'DigitInfo',
    'config',
    'read_caliper',
    'read_caliper_from_array',
]
