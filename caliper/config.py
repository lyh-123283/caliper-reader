"""
游标卡尺识别 — 集中调参配置文件
==================================

使用方法：
    from caliper.config import config

    # 查看/修改参数
    config.preprocess.gamma = 0.85         # 调亮暗部
    config.roi.aspect_min = 8.0            # 放宽长宽比下限
    config.main_scale.min_tick_count = 5   # 放宽最小刻线数

    # 重置为默认值
    config.reset()

    # 打印所有参数（调试用）
    config.summary()
"""


# ═════════════════════════════════════════════════════════════
#  0. 图像预处理 — preprocess.py
# ═════════════════════════════════════════════════════════════

class PreprocessConfig:
    """预处理参数：灰度 → gamma 校正 → 双边滤波 → 中值滤波 → CLAHE → 钝化 → 二值化"""

    # ── 幂律变换 (gamma 校正) ──
    #     公式: s = 255 * (r/255)^(1/gamma)
    #     1.0 = 不改变；<1 提亮暗部（刻线区域更可见）；>1 压暗高光（抑制反光）
    #     典型范围: 0.6 ~ 1.5
    gamma: float = 1.5

    # ── 高斯模糊前置（在双边之前轻量去纹理噪声）──
    #     ksize: 核尺寸（奇数，典型 3~5）；设为 0 跳过
    #     sigma: 标准差（典型 0.5~1.0）
    gauss_pre_ksize: int = 0
    gauss_pre_sigma: float = 0.8

    # ── 双边滤波（保边去噪）──
    #     直径: 像素邻域直径，越大平滑越强但越慢
    #     sigma: 颜色/空间标准差，越大平滑越强
    bilateral_d: int = 11
    bilateral_sigma: float = 60.0

    # ── 中值滤波（脉冲/椒盐噪声）──
    #     核尺寸（奇数），≥3 生效；设为 0 跳过中值滤波
    median_ksize: int = 9

    # ── CLAHE 对比度增强（局部自适应直方图均衡）──
    #     clip_limit: 对比度限制，越大对比度越强（也放大噪声）
    #     tile_grid: 分块大小，(8,8) 是常规默认
    clahe_clip_limit: float = 1.5
    clahe_tile_w: int = 8
    clahe_tile_h: int = 8

    # ── 非锐化掩膜锐化 ──
    #     v6.5: 公式已修正为标准 unsharp mask：sharp = orig + amount × (orig - blur)
    #     amount=0   → 不变（跳过锐化）
    #     amount=0.5 → 轻微锐化（默认，平衡刻度可见性 vs 噪声放大）
    #     amount=1.5 → 强锐化（可能引入噪声，下游过检）
    unsharp_amount: float = 0.5
    unsharp_blur_sigma: float = 1.5

    # ── 自适应阈值二值化 ──
    #     block_size: 局部邻域大小（奇数），越大对光照鲁棒但细节损失
    #     C: 从均值中减去的常数，越大二值化越保守（白像素越少）
    adaptive_block_size: int = 31
    adaptive_C: int = 5

    # ── 后处理：形态学开运算（二值化后去噪）──
    #     先腐蚀再膨胀，消除孤立小噪点
    morph_open_enabled: bool = True
    morph_open_kernel_size: int = 3   # 核尺寸（椭圆核）
    morph_open_iterations: int = 1    # 迭代次数

    # ── 后处理：连通域过滤（二值化后去噪）──
    #     剔除面积小于阈值的孤立连通域（白连通域=噪声斑块）
    #     v6.5: 从 15 增加到 50，更激进地过滤背景墙椒盐噪声
    #     (主尺刻度线连通域面积一般 > 200，安全裕量充足)
    cc_filter_enabled: bool = True
    cc_min_area: int = 50             # 最小面积（像素）


# ═════════════════════════════════════════════════════════════
#  1. ROI 提取 — roi_extract.py
# ═════════════════════════════════════════════════════════════

class ROIExtractConfig:
    """ROI 提取参数：COM 质心法定位 → 应急轮廓兜底"""

    # ── COM 质心法：以投影质心为轴，扩展的宽度/高度比例 ──
    #     预期读数区约占图宽 30%、图高 22%
    x_center_span_ratio: float = 0.30
    y_center_span_ratio: float = 0.22

    # ── 边界余量 ──
    #     在检测到的边界外扩的比例
    y_pad_ratio: float = 0.10
    x_pad_ratio: float = 0.15

    # ── ROI 尺寸下限（像素）──
    min_roi_height: int = 15
    min_roi_width: int = 30

    # ── 轮廓评分（纯轮廓应急兜底）──
    #     面积比（轮廓外接矩形 / 全图）
    contour_area_ratio_min: float = 0.05
    contour_area_ratio_max: float = 0.60
    #     长宽比（宽/高）
    contour_aspect_min: float = 6.0
    contour_aspect_max: float = 30.0
    #     矩形度（轮廓面积 / 外接矩形面积）
    contour_rectangularity_min: float = 0.65
    #     最低总分（低于则放弃轮廓法）
    contour_min_score: float = 0.15

    # ── 评分权重 ──
    score_weight_area: float = 0.25
    score_weight_aspect: float = 0.40
    score_weight_rect: float = 0.25
    score_weight_position: float = 0.10

    # ── 形态学闭运算核宽（占图宽比例）──
    morph_kernel_ratio: float = 0.025  # 水平核宽 = 图宽 * ratio


# ═════════════════════════════════════════════════════════════
#  1b. 方向矫正 — roi_extract.py / orient_caliper()
# ═════════════════════════════════════════════════════════════

class OrientConfig:
    """方向矫正参数：Canny 边缘 → HoughLinesP 检测 → 角度统计 → 旋转"""

    # ── Canny 边缘检测 ──
    canny_low: int = 40
    canny_high: int = 150

    # ── HoughLinesP 概率霍夫 ──
    hough_threshold: int = 50   # 累加器阈值（越高要求越严格）
    hough_min_length: int = 25  # 线段最小长度（像素）
    hough_max_gap: int = 6      # 同一直线的最大间断

    # ── 角度过滤 ──
    #     只保留角度在此范围内的线（刻度线近似垂直）
    angle_min: float = 55.0     # 与水平方向夹角下限
    angle_max: float = 125.0    # 与水平方向夹角上限

    # ── 角度合并 ──
    #     缩尾比例（去掉最极端的两端）
    trim_ratio: float = 0.1     # 两端各去掉 10%

    # ── 旋转阈值 ──
    #     小于此角度的不做旋转（避免微小抖动 + HoughLinesP 随机噪声）
    rotate_min_angle: float = 0.3
    #     大于此角度认为是检测错误
    rotate_max_angle: float = 80.0


# ═════════════════════════════════════════════════════════════
#  2. 区域分离 — region_split.py
# ═════════════════════════════════════════════════════════════

class RegionSplitConfig:
    """区域分离参数：投影突变法 → 梯度法 → 闭运算投影法"""

    # ── CLAHE 增强 ──
    clahe_clip_limit: float = 2.5
    clahe_tile_w: int = 8
    clahe_tile_h: int = 8

    # ── 搜索范围（占图像高度比例）──
    #     v5.7: 放宽到 [10%, 75%] —— 主尺刻度行可能很靠上（如 12% 处），
    #     游标行可能在 50% 处，分界点在两者之间任何位置都要能搜到
    search_lo_ratio: float = 0.10
    search_hi_ratio: float = 0.75

    # ── tick 密度验证 ──
    #     band = max(h / band_ratio_denom, band_min)
    #     v5.8: band 设为 ROI 高度的 ~10%（一行刻度行高度的 60%~80%）
    #     太大 → 跨多行刻度，打分失真；太小 → 噪声影响大
    density_band_ratio_denom: int = 12
    density_band_min: int = 25
    density_min_score: int = 4  # 最低分（两侧各至少2条线）

    # ── 闭运算核宽（二分图法回退）──
    close_kernel_ratio: float = 0.33  # 核宽 = 图宽 * ratio（最小值 30）

    # ── 梯度法阈值（波峰检测）──
    gradient_threshold_factor: float = 1.8  # 阈值 = 均值 × factor（最小值 0.04）
    gradient_min_thresh: float = 0.04

    # ── 最终回退比例（所有方案失败时）──
    #     物理先验：主尺约占 ROI 高度的 55%~65%
    fallback_split_ratio: float = 0.60

    # ── 游标区域最小高度比例 ──
    #     游标区至少有 ROI 高度的 28%，否则强制重分配
    min_vernier_height_ratio: float = 0.28


# ═════════════════════════════════════════════════════════════
#  3. 主尺识别 — main_scale.py
# ═════════════════════════════════════════════════════════════
class MainScaleConfig:

    """主尺识别参数：自适应阈值 → 垂直投影 → 峰值检测 → 刻线提取 → 等间距补全 → OCR"""

    # ── 二值化（自适应阈值）──
    #     blockSize: 局部邻域大小（奇数）
    #     C: 从均值中减去的常数，越小前景越多（刻线越多）
    adaptive_block_size: int = 31
    adaptive_C: int = 2

    # ── 峰值检测 ──
    #     find_peaks_adaptive 的 min_dist（相邻峰最小间距）
    peak_min_dist: int = 3
    #     find_peaks_adaptive 的 threshold_factor（越小越敏感）
    #     v6: 提高到 0.20（之前 0.12 太敏感，把噪声峰也算入）
    peak_threshold_factor: float = 0.20
    # ── 最小刻线数 ──
    min_tick_count: int = 3

    # ── 长/短刻线判定 ──
    #     is_long = length > median_length * factor
    long_tick_factor: float = 1.3

    # ── 等间距补全 & 校验 (refine_ticks_by_spacing) ──
    #     是否启用（建议开启，利用物理等间距约束提升鲁棒性）
    spacing_refine_enabled: bool = True
    #     网格匹配容差比例（0.30 = 30% 间距）
    spacing_tolerance: float = 0.30
    #     间距 > S*gap_factor 触发补全（1.30 = 更敏感）
    spacing_gap_factor: float = 1.30
    #     间距 < S*dup_factor 触发去重（伪影过滤）
    spacing_dup_factor: float = 0.50
    #     网格吸附容差（0.28 = 偏移超过 28% 间距则保留原位）
    spacing_snap_ratio: float = 0.28


# ═════════════════════════════════════════════════════════════
#  4. 游标尺识别 — vernier_scale.py
# ═════════════════════════════════════════════════════════════

class VernierScaleConfig:
    """Vernier recognition params: projection -> fixed 0.02mm -> zero -> spacing fill -> alignment."""

    # ── 二值化（自适应阈值）──
    #     blockSize: 局部邻域大小（奇数）
    #     C: 从均值中减去的常数，越小前景越多（刻线越多）
    adaptive_block_size: int = 31
    adaptive_C: int = 4

    # ── 峰值检测 ──
    peak_min_dist: int = 2
    #     越小越敏感，游标尺更密所以比主尺略高
    peak_threshold_factor: float = 0.15

    # ── 最小刻线数 ──
    min_tick_count: int = 3


    # ── 零线验证 ──
    #     候选零线刻线长度需 >= median_len * factor
    zero_length_factor: float = 0.4

    # ── 零线 OCR 数字 0 验证 ──
    #     搜索区域: y_start = int(h * ratio)，取下半个图
    zero_digit_search_ratio: float = 0.50
    #     搜索窗口半宽: max(min_px, int(w * ratio))
    zero_digit_half_w_min: int = 12
    zero_digit_half_w_ratio: float = 0.04
    #     OCR 最低置信度
    zero_digit_conf_min: float = 0.3
    #     CLAHE 增强
    zero_digit_clahe_clip: float = 2.5

    # ── 等间距补全 & 校验 (refine_ticks_by_spacing) ──
    #     是否启用（游标尺刻线更密，建议开启以补全漏检）
    spacing_refine_enabled: bool = True
    #     网格匹配容差比例（游标尺间距更小，容差稍大）
    spacing_tolerance: float = 0.35
    #     间距 > S*gap_factor 触发补全
    spacing_gap_factor: float = 1.35
    #     间距 < S*dup_factor 触发去重
    spacing_dup_factor: float = 0.50
    #     网格吸附容差
    spacing_snap_ratio: float = 0.30

    # ── 对齐查找 — 亚像素插值 ──
    #     main_gap 最小值（防止除零）
    min_main_gap: float = 10.0

    # ── 对齐置信度阈值 ──
    align_conf_perfect: float = 0.95  # 误差 <= 0.5px
    align_conf_strong: float = 0.9    # 邻居/最优 >= 3
    align_conf_moderate: float = 0.7  # 邻居/最优 >= 2
    align_conf_weak: float = 0.5      # 邻居/最优 >= 1.5
    align_conf_bad: float = 0.3       # 更差

    # ── 长/短刻线判定 ──
    long_tick_factor: float = 1.3


# ═════════════════════════════════════════════════════════════
#  5. OCR 数字识别 — ocr.py
# ═════════════════════════════════════════════════════════════

class OCRConfig:
    """OCR 参数：投影强峰值选长刻度 → 上方连通域搜索 → 自适应框 → 识别"""

    # ── 投影强峰值筛选 ──
    #     垂直投影值高于 mean + std * factor 的刻线视为"长刻度"（数字候选）
    projection_strong_factor: float = 0.5
    #     最少保留几条强刻度线
    projection_min_strong: int = 3

    # ── 数字搜索区域（在长刻度线 x 附近，上方区域）──
    #     x 搜索半宽 = tick_gap * ratio
    search_tick_gap_ratio: float = 0.50
    #     y 搜索起点：图顶（0），终点：刻线上方至少此图高比例
    search_y_min_ratio: float = 0.25
    #     搜索框最小高度（像素）
    search_y_min_height: int = 15

    # ── 连通域过滤 ──
    cc_min_area: int = 8                 # 降低以捕获弱对比度数字
    cc_min_width: int = 3
    cc_min_height: int = 5
    cc_aspect_min: float = 0.0           # 去掉宽高比限制
    cc_aspect_max: float = 999.0

    # ── 连通域合并（处理多笔画数字如 0, 8, 6, 4）──
    #     y 方向有重叠 + x 方向间距 < tick_gap * ratio → 合并为同一数字
    merge_x_gap_ratio: float = 0.30

    # ── 自适应 padding ──
    #     pad = max(min_pad, min(cc_w, cc_h) // divisor)
    pad_min: int = 3
    pad_divisor: int = 4

    # ── 回退补丁（连通域法失败时）──
    #     在长刻度线上方固定比例区域取图
    fallback_x_half_ratio: float = 0.18  # 半宽 = tick_gap * ratio 或 min_px
    fallback_x_half_min: int = 10
    fallback_y_end_ratio: float = 0.40   # y 终点 = 图高 * ratio（即刻度线上方）
    fallback_y_h_ratio: float = 0.20     # 搜索高度 = 图高 * ratio

    # ── 补丁增强 ──
    patch_resize_factor: int = 3         # 放大倍数（v6.5: 2→3，小 patch 更清晰）
    patch_clahe_clip: float = 2.5
    patch_adaptive_block: int = 11
    patch_adaptive_C: int = 3

    # ── Tesseract ──
    tesseract_psm: str = '8'             # Page Segmentation Mode: 8=单行
    tesseract_whitelist: str = '0123456789'

    # ── EasyOCR ──
    easyocr_allowlist: str = '0123456789'
    easyocr_min_size: int = 5
    easyocr_text_threshold: float = 0.3
    easyocr_low_text: float = 0.2
    easyocr_min_conf: float = 0.2


# ═════════════════════════════════════════════════════════════
#  5. 读数合并 — merger.py
# ═════════════════════════════════════════════════════════════

class MergerConfig:
    """读数合并 & 置信度参数"""

    # ── 置信度评分 ──
    #     主尺刻线数不足阈值
    conf_main_tick_min: int = 5
    #     游标刻线数不足阈值
    conf_vernier_tick_min: int = 5
    #     间距变异系数阈值（超过则降级）
    conf_gap_cv_threshold: float = 0.15  # CV = std/mean

    # ── 绘制 ──
    #     标注线的颜色 BGR
    draw_main_tick_color: tuple = (0, 235, 100)     # 绿色
    draw_main_long_color: tuple = (0, 255, 80)       # 亮绿（长刻度）
    draw_vernier_tick_color: tuple = (255, 200, 50)  # 橙色
    draw_zero_line_color: tuple = (50, 150, 255)     # 蓝色
    draw_alignment_color: tuple = (50, 255, 150)     # 青绿


# ═════════════════════════════════════════════════════════════
#  主配置类
# ═════════════════════════════════════════════════════════════

class CaliperConfig:
    """游标卡尺识别总配置"""

    def __init__(self):
        self.preprocess = PreprocessConfig()
        self.roi = ROIExtractConfig()
        self.orient = OrientConfig()
        self.region_split = RegionSplitConfig()
        self.main_scale = MainScaleConfig()
        self.vernier_scale = VernierScaleConfig()
        self.ocr = OCRConfig()
        self.merger = MergerConfig()

    def reset(self):
        """将所有参数重置为默认值"""
        self.preprocess = PreprocessConfig()
        self.roi = ROIExtractConfig()
        self.orient = OrientConfig()
        self.region_split = RegionSplitConfig()
        self.main_scale = MainScaleConfig()
        self.vernier_scale = VernierScaleConfig()
        self.ocr = OCRConfig()
        self.merger = MergerConfig()

    def summary(self) -> str:
        """打印所有配置参数（调试用）"""
        lines = ["=== 游标卡尺识别参数配置 ==="]
        for group_name in ['preprocess', 'roi', 'orient', 'region_split',
                            'main_scale', 'vernier_scale', 'ocr', 'merger']:
            group = getattr(self, group_name)
            lines.append(f"\n── {group_name} ──")
            for k, v in group.__dict__.items():
                lines.append(f"  {k} = {v}")
        return "\n".join(lines)


# ── 全局单例 ──
config = CaliperConfig()
