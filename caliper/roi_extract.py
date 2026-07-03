"""
步骤 1 — ROI 提取 + 方向矫正

① ROI 提取：投影法定位刻度区域 → 裁剪（+ 轮廓验证回退）
② 方向矫正：HoughLinesP 检测刻线方向 → 旋转使刻线垂直
"""

import cv2
import numpy as np
from typing import Tuple, List

from .utils import rotate_image, draw_projection_plot
from .config import config


# ═══════════════════════════════════════════════════════════
#  ① ROI 提取：投影法定位刻度区域
# ═══════════════════════════════════════════════════════════

def extract_roi(img_color: np.ndarray,
                img_binary: np.ndarray,
                img_enhanced: np.ndarray = None) -> dict:
    """
    利用二值图投影找到刻度密集区域并裁剪。
    不旋转——旋转留给 orient_caliper() 处理。

    策略（v2 — 双二值源 + 轮廓验证）:
      a) 优先：自适应二值图水平投影 → y 范围 → 垂直投影 → x 范围
      b) 验证：对投影得到的 ROI 做轮廓打分，长宽比/矩形度不达标则回退
      c) 回退：Sobel X 二值图 → 形态学闭运算连接刻线 → 轮廓筛选 → ROI
      d) 最终：裁剪彩色图 + 增强图 + 二值图

    Returns:
        dict: roi_color, roi_gray, roi_binary, mask_vis, angle_roi
    """
    h, w = img_color.shape[:2]
    enhanced = img_enhanced if img_enhanced is not None else cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # ── 预处理输出的二值图为 THRESH_BINARY（黑=前景/白=背景）。
    #     投影与轮廓分析需要白前景，故内部反色处理。
    img_binary_for_proj = cv2.bitwise_not(img_binary) if img_binary is not None else None

    # ── 生成 Sobel 垂直边缘二值图（备用源，天然过滤水平干扰）──
    binary_vedge, sobel_grad_u8 = _binary_vertical_edges(enhanced)

    # ── 诊断记录 ──
    diag: dict = {
        'source': 'adaptive',
        'binary_adapt': img_binary_for_proj,  # 白前景版本（投影用）
        'binary_vedge': binary_vedge,
        'contour_vis': None,
        'contour_scores': [],
        'sobel_gradient': sobel_grad_u8,      # Sobel X 梯度归一化 uint8（用于热力图可视化）
        'enhanced': enhanced,                  # 增强灰度图引用（供 Sobel 可视化参考）
        'com_y_diag': None,                    # COM Y 质心法可视化数据
        'com_x_diag': None,                    # X 方向最长等间距序列可视化数据
    }

    if img_binary is None:
        y1, y2 = _proj_find_y_range(binary_vedge, h)
        diag['com_y_diag'] = _collect_com_y_diag(binary_vedge, h, y1, y2)
        x1, x2, x_diag = _proj_find_x_range(binary_vedge, y1, y2, w)
        diag['com_x_diag'] = x_diag
        diag['source'] = 'sobel'
    else:
        y1, y2 = _proj_find_y_range(img_binary_for_proj, h)
        diag['com_y_diag'] = _collect_com_y_diag(img_binary_for_proj, h, y1, y2)
        x1, x2, x_diag = _proj_find_x_range(img_binary_for_proj, y1, y2, w)
        diag['com_x_diag'] = x_diag

    # ── 应急兜底：COM 法始终产生有效范围，仅极端情况下触发轮廓法 ──
    if y2 - y1 < config.roi.min_roi_height or x2 - x1 < config.roi.min_roi_width:
        cy1, cy2, cx1, cx2, diag_contour = \
            _extract_roi_by_contour_with_diag(binary_vedge, h, w)
        diag['contour_vis'] = diag_contour
        if cy2 - cy1 > 0 and cx2 - cx1 > 0:
            y1, y2, x1, x2 = cy1, cy2, cx1, cx2
            diag['source'] = 'contour'

    # ── d) 裁剪 ──
    if y2 - y1 < config.roi.min_roi_height or x2 - x1 < config.roi.min_roi_width:
        return _full_roi_result(img_color, enhanced)

    # ── v6: 用游标压块精确边界做最终 ROI 精修 ──
    #     用户需求：完整框出游标卡尺读数区（主尺刻度 + 游标尺压块 + 游标尺数字），
    #     向左可少量超出，向右、向上、向下都不要超出。
    #     压块是 ROI 内最大的深色金属块，bbox 能给出读数区的精确几何范围。
    refined = _refine_roi_by_vernier_block(enhanced, y1, y2, x1, x2)
    if refined is not None:
        y1, y2, x1, x2 = refined
        diag['source'] = diag['source'] + '+block_refine'

    diag['y1'], diag['y2'], diag['x1'], diag['x2'] = y1, y2, x1, x2

    roi_color = img_color[y1:y2 + 1, x1:x2 + 1]
    roi_gray = enhanced[y1:y2 + 1, x1:x2 + 1]
    roi_binary = img_binary[y1:y2 + 1, x1:x2 + 1] if img_binary is not None else None

    # ── 可视化 ──
    mask_vis = _make_roi_vis_v3(img_color, roi_color, diag)

    return {
        'roi_color': roi_color,
        'roi_gray': roi_gray,
        'roi_binary': roi_binary,
        'mask_vis': mask_vis,
        'angle_roi': 0.0,
    }


# ═══════════════════════════════════════════════════════════
#  Sobel X 垂直边缘二值化（过滤水平边框/文字干扰）
# ═══════════════════════════════════════════════════════════

def _binary_vertical_edges(gray: np.ndarray):
    """
    Sobel X 水平方向梯度 → 只保留垂直边缘 → 归一化 → OTSU 二值化。

    物理依据：刻度线是垂直的，Sobel X 只对竖直方向亮度变化敏感，
    天然过滤卡尺水平边框、背景水平线条、文字笔画中的水平段。

    Returns:
        (binary, grad_u8) — OTSU 二值图 和 梯度归一化 uint8 图（用于可视化热力图）
    """
    # 1. 水平方向 Sobel（检测垂直边缘）
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_grad = np.abs(sobel_x)

    # 2. 归一化到 0~255（保存一份归一化图用于可视化）
    g_max = float(np.max(abs_grad))
    grad_u8 = np.zeros_like(gray, dtype=np.uint8)
    if g_max > 0:
        grad_u8 = (abs_grad / g_max * 255).astype(np.uint8)

    # 3. OTSU 二值化（保留强垂直边缘）— 用 grad_u8 做阈值
    _, binary = cv2.threshold(grad_u8, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, grad_u8


# ═══════════════════════════════════════════════════════════
#  轮廓验证 + 回退（长宽比 / 矩形度 / 面积评分）
# ═══════════════════════════════════════════════════════════

def _score_contour_as_scale_roi(cnt: np.ndarray, img_h: int, img_w: int) -> float:
    """
    对轮廓打分，判断它是否像一个"刻度区域"。

    评分维度：
      1) 面积比（轮廓面积 / 全图面积）：5%~60% → 权重 0.25
      2) 长宽比（宽/高）：6~30 → 权重 0.40（核心特征：刻度区长条状）
      3) 矩形度（轮廓面积 / 最小外接矩形面积）：> 0.65 → 权重 0.25
      4) 中心位置（轮廓中心偏离图像中心的程度）：权重 0.10

    Returns: 0.0 ~ 1.0，越高越像刻度区
    """
    img_area = img_h * img_w
    x, y, bw, bh = cv2.boundingRect(cnt)
    rect_area = bw * bh

    if bw <= 0 or bh <= 0 or rect_area <= 0:
        return 0.0

    # ── 1. 面积比 ──
    area_ratio = rect_area / img_area
    if config.roi.contour_area_ratio_min <= area_ratio <= config.roi.contour_area_ratio_max:
        score_area = 1.0
    elif area_ratio > 0:
        score_area = max(0.0, 1.0 - abs(area_ratio - 0.2) / 0.5)
    else:
        score_area = 0.0

    # ── 2. 长宽比（宽/高）──
    aspect = bw / bh
    if config.roi.contour_aspect_min <= aspect <= config.roi.contour_aspect_max:
        score_aspect = 1.0
    else:
        # 离理想区间越远分越低
        mid_aspect = (config.roi.contour_aspect_min + config.roi.contour_aspect_max) / 2
        dist = abs(aspect - mid_aspect) / mid_aspect
        score_aspect = max(0.0, 1.0 - dist)

    # ── 3. 矩形度（轮廓面积 / 外接矩形面积）──
    cnt_area = cv2.contourArea(cnt)
    rectangularity = cnt_area / rect_area if rect_area > 0 else 0.0
    if rectangularity >= config.roi.contour_rectangularity_min:
        score_rect = 1.0
    else:
        score_rect = rectangularity / config.roi.contour_rectangularity_min

    # ── 4. 中心位置 ──
    cx = x + bw / 2
    cy = y + bh / 2
    dx = abs(cx - img_w / 2) / (img_w / 2)
    dy = abs(cy - img_h / 2) / (img_h / 2)
    score_pos = max(0.0, 1.0 - (dx * 0.4 + dy * 0.6))

    return (
        config.roi.score_weight_area * score_area +
        config.roi.score_weight_aspect * score_aspect +
        config.roi.score_weight_rect * score_rect +
        config.roi.score_weight_position * score_pos
    )


def _extract_roi_by_contour_with_diag(binary_vedge: np.ndarray,
                                       img_h: int, img_w: int):
    """
    轮廓法 ROI 提取 + 诊断可视化。

    Returns: (y1, y2, x1, x2, diag_vis)
      diag_vis: RGB 图像，展示形态学变换 + 轮廓打分
    """
    # 1. 水平长条形结构元素闭运算 → 连接分散的垂直刻度线
    kernel_w = max(12, int(img_w * config.roi.morph_kernel_ratio))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    closed = cv2.morphologyEx(binary_vedge, cv2.MORPH_CLOSE, kernel_close)

    # 2. 小核开运算 → 去除残留噪点
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # 3. 提取最外层轮廓
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── 构建诊断图：Sobel二值 → 闭运算 → 开运算 → 轮廓 ──
    diag_vis = _make_contour_diag_vis(binary_vedge, closed, opened, contours,
                                       img_h, img_w)

    if not contours:
        return 0, 0, 0, 0, diag_vis

    # 4. 对每个轮廓打分，选最高分
    best_cnt, best_score = None, -1.0
    for cnt in contours:
        score = _score_contour_as_scale_roi(cnt, img_h, img_w)
        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None or best_score < config.roi.contour_min_score:
        return 0, 0, 0, 0, diag_vis

    # 5. 计算外接矩形并加余量
    x, y, bw, bh = cv2.boundingRect(best_cnt)
    pad_x = max(3, bw // 30)
    pad_y = max(2, bh // 20)
    y1 = max(0, y - pad_y)
    y2 = min(img_h - 1, y + bh + pad_y)
    x1 = max(0, x - pad_x)
    x2 = min(img_w - 1, x + bw + pad_x)

    return y1, y2, x1, x2, diag_vis


# ═══════════════════════════════════════════════════════════
#  COM 质心法投影定位
# ═══════════════════════════════════════════════════════════

def _proj_find_y_range(binary: np.ndarray, h: int) -> Tuple[int, int]:
    """COM 质心法：以投影加权中心为轴，扩展固定比例高度"""
    if binary is None:
        return 0, h - 1

    hproj = np.sum(binary, axis=1).astype(float)
    total = hproj.sum()
    if total <= 0:
        return 0, h - 1

    # 质心
    ys = np.arange(h)
    com_y = float(np.dot(ys, hproj) / total)

    # 以质心为轴，各扩展 span_ratio/2
    half_h = int(h * config.roi.y_center_span_ratio / 2)
    y1 = int(com_y) - half_h
    y2 = int(com_y) + half_h

    # 边界约束
    if y1 < 0:
        y2 = min(h - 1, y2 - y1)
        y1 = 0
    if y2 >= h:
        y1 = max(0, y1 - (y2 - h + 1))
        y2 = h - 1

    pad_y = max(2, int((y2 - y1) * config.roi.y_pad_ratio))
    y1 = max(0, y1 - pad_y)
    y2 = min(h - 1, y2 + pad_y)
    return y1, y2


def _collect_com_y_diag(binary: np.ndarray, h: int, y1: int, y2: int) -> dict:
    """收集 COM Y 质心法可视化所需数据（避免重复计算）"""
    if binary is None:
        return None
    hproj = np.sum(binary, axis=1).astype(float)
    total = hproj.sum()
    if total <= 0:
        return None
    ys = np.arange(h)
    com_y = float(np.dot(ys, hproj) / total)
    half_h = int(h * config.roi.y_center_span_ratio / 2)
    span_lo = int(com_y) - half_h
    span_hi = int(com_y) + half_h
    return {
        'hproj': hproj,
        'com_y': com_y,
        'span_lo': max(0, span_lo),
        'span_hi': min(h - 1, span_hi),
        'y1': y1,
        'y2': y2,
    }


def _proj_find_x_range(binary: np.ndarray, y1: int, y2: int, w: int):
    """
    x 方向 ROI 边界：用"最长等间距强峰序列"识别刻度区。

    v5.2 算法（替代失控的网格扩展）：
      1. 在 y 带内做垂直投影 → 自适应峰值检测
      2. 用强度过滤：只保留 vproj_norm ≥ 强度阈值（0.20）的峰
      3. 寻找"最长等间距强峰序列"：相邻 gap 在 [median*0.6, median*1.6] 内
      4. 序列首尾即 ROI 的 x 边界，再加固定 pad

    物理依据：游标卡尺刻度线严格等间距，这是与背景墙噪点、卡尺杆水平纹路、
    螺丝孔等的根本区别——后者要么间距不规整，要么强度比刻线弱很多。
    """
    if binary is None or y2 <= y1:
        return 0, w - 1, None

    strip = binary[y1:y2 + 1, :]
    vproj = np.sum(strip, axis=0).astype(float)
    total = vproj.sum()
    if total <= 0:
        return 0, w - 1, None

    vmax = float(np.max(vproj))
    if vmax <= 0:
        return 0, w - 1, None
    vproj_norm = vproj / vmax

    # 用于回退方案
    xs_all = np.arange(w)
    com_x = float(np.dot(xs_all, vproj) / total)

    # ── 1. 全图垂直投影峰值检测 ──
    from .utils import find_peaks_adaptive
    peaks_all = find_peaks_adaptive(vproj_norm, min_dist=3, threshold_factor=0.3)
    if len(peaks_all) < 10:
        fx1, fx2 = _fallback_com_x_range(com_x, w)
        return fx1, fx2, None

    # ── 2. 强度过滤：保留中等及以上强度的峰 ──
    strength = vproj_norm[peaks_all]
    th_strength = max(0.15, float(np.percentile(strength, 30)))
    strong_mask = strength >= th_strength
    strong_peaks = peaks_all[strong_mask]
    if len(strong_peaks) < 5:
        strong_peaks = peaks_all

    # ── 3. 估算 tick_gap：用相邻 gap 的中位数 ──
    diffs = np.diff(strong_peaks)
    if len(diffs) == 0:
        fx1, fx2 = _fallback_com_x_range(com_x, w)
        return fx1, fx2, None
    tick_gap = float(np.median(diffs))
    if tick_gap < 3.0:
        fx1, fx2 = _fallback_com_x_range(com_x, w)
        return fx1, fx2, None

    # ── 4. 寻找"最长等间距强峰序列" ──
    lo_gap = tick_gap * 0.50
    hi_gap = tick_gap * 1.80
    max_irregular = 2

    segments = []
    cur_start = 0
    irregular_run = 0
    for i, d in enumerate(diffs):
        if lo_gap <= d <= hi_gap:
            irregular_run = 0
        else:
            irregular_run += 1
            if irregular_run > max_irregular:
                end_idx = i - irregular_run
                if end_idx - cur_start >= 3:
                    segments.append((cur_start, end_idx))
                cur_start = i + 1
                irregular_run = 0
    if len(strong_peaks) - 1 - cur_start >= 3:
        segments.append((cur_start, len(strong_peaks) - 1))

    # ── 构建可视化数据（即使在回退前就收集好）──
    x_diag = {
        'vproj_norm': vproj_norm,
        'peaks_all': peaks_all,
        'strong_peaks': strong_peaks,
        'tick_gap': tick_gap,
        'th_strength': th_strength,
        'segments': [list(s) for s in segments],
        'all_segments': [list(s) for s in segments],
        'seg_lo': None,
        'seg_hi': None,
    }

    if not segments:
        fx1, fx2 = _fallback_com_x_range(com_x, w)
        return fx1, fx2, x_diag

    # ── 4b. 合并相邻段 ──
    merge_dist = tick_gap * 8.0
    merged = [list(segments[0])]
    for seg in segments[1:]:
        prev_end_x = float(strong_peaks[merged[-1][1]])
        cur_start_x = float(strong_peaks[seg[0]])
        if cur_start_x - prev_end_x < merge_dist:
            merged[-1][1] = seg[1]
        else:
            merged.append(list(seg))

    best_seg = max(merged, key=lambda s: s[1] - s[0])
    seg_lo = int(strong_peaks[best_seg[0]])
    seg_hi = int(strong_peaks[best_seg[1]])
    x_diag['seg_lo'] = seg_lo
    x_diag['seg_hi'] = seg_hi
    x_diag['best_seg'] = [best_seg[0], best_seg[1]]

    # ── 5. 加 pad 返回 ──
    pad_x = max(int(tick_gap * 1.5),
                int((seg_hi - seg_lo) * config.roi.x_pad_ratio))
    x1 = max(0, seg_lo - pad_x)
    x2 = min(w - 1, seg_hi + pad_x)

    # ── 6. v5.4: 以"游标尺压块"为中心收窄 ROI ──
    vernier_x_center = _locate_vernier_block_x_by_gray(binary, x1, x2, y1, y2)
    if vernier_x_center is not None:
        half_w = max(int(tick_gap * 20), int((x2 - x1) * 0.35))
        nx1 = max(x1, int(vernier_x_center) - half_w)
        nx2 = min(x2, int(vernier_x_center) + half_w)
        if nx2 - nx1 >= tick_gap * 8:
            x1, x2 = nx1, nx2
    x_diag['x1'] = x1
    x_diag['x2'] = x2

    return x1, x2, x_diag


def _refine_roi_by_vernier_block(enhanced: np.ndarray,
                                   y1: int, y2: int,
                                   x1: int, x2: int):
    """
    v6.1: 用"卡尺金属面上下沿"+"游标压块右缘"精修 ROI。

    用户需求：
      • y 上下不要超出卡尺金属面（不框背景墙、卡尺杆）
      • x 左侧可少量超出（含 DELIXI Logo）
      • x 右侧不要超出游标压块右缘

    算法：
      1. y 边界：Sobel Y → 全图水平投影 → 找两条最强水平边缘
         （= 卡尺上沿 + 下沿）
      2. x 右边界：在 y 范围内找游标压块右缘
         （从右往左扫描，第一个"垂直边缘强度跳变 + 下方为压块"位置）
      3. x 左边界：当前 x1（来自等间距搜索）通常已合理，向左略扩留一点

    Returns:
        (y1, y2, x1, x2) 或 None
    """
    if enhanced is None:
        return None
    H, W = enhanced.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    if x2 - x1 < 100 or y2 - y1 < 100:
        return None

    # ── 1. y 边界：用读数区左中段找水平边缘 ──
    # 候选 x 段过宽时，右侧长尺尾部/滑块水平边会把上沿拉偏。
    # y 边界只需要读数区的代表性水平边，因此先收窄用于估计的 x 窗口。
    y_edge_x1, y_edge_x2 = _select_y_edge_x_window(x1, x2, W)
    new_y1, new_y2 = _find_caliper_y_edges(enhanced, y_edge_x1, y_edge_x2, y1, y2)
    if new_y1 is None or new_y2 is None or new_y2 - new_y1 < 80:
        new_y1, new_y2 = y1, y2
    else:
        # v6.4: 给下方留出游标尺数字行空间（约 30% 卡尺高度）
        caliper_h = new_y2 - new_y1
        extra_below = int(caliper_h * 0.35)
        new_y2 = min(H - 1, new_y2 + extra_below)

    # ── 2. x 右边界：用 y 上沿水平边缘的右端 x 作为压块右缘 ──
    new_x2 = _find_caliper_right_edge(enhanced, new_y1, new_y2, x1)
    if new_x2 is None or new_x2 <= x1 + 200:
        new_x2 = x2

    # ── 3. x 左边界：保持原值（来自等间距搜索，已合理）──
    new_x1 = x1

    # 给 y 上下各加一点点余量（防止误差切到刻度线本身）
    pad_y = max(5, (new_y2 - new_y1) // 60)
    new_y1 = max(0, new_y1 - pad_y)
    new_y2 = min(H - 1, new_y2 + pad_y)

    if new_x2 - new_x1 < 100 or new_y2 - new_y1 < 80:
        return None
    return new_y1, new_y2, new_x1, new_x2


def _select_y_edge_x_window(x1: int, x2: int, image_w: int) -> Tuple[int, int]:
    """Choose a stable x window for ROI y-edge detection."""
    x1 = max(0, int(x1))
    x2 = min(image_w - 1, int(x2))
    span = x2 - x1
    if span <= 0:
        return x1, x2

    # Wide spans often include the right ruler tail and upper slider edge.
    # Keep the left/middle readout area, where main/vernier scales overlap.
    if span > image_w * 0.55:
        nx1 = x1 + int(span * 0.02)
        nx2 = x1 + int(span * 0.85)
        if nx2 - nx1 >= 300:
            return nx1, nx2
    return x1, x2


def _find_caliper_y_edges(enhanced: np.ndarray,
                            x_lo: int, x_hi: int,
                            y_search_lo: int, y_search_hi: int):
    """
    在 enhanced 全图上用 Sobel Y 找卡尺金属面的上沿和下沿。

    v6.3 策略：取"投影最强 + 距它 ≥ MIN_SEP 像素且自身最强的"两条 y。
    最强水平边缘一定是卡尺真实边界之一（金属/背景对比最大），
    再在距它远端找次强（卡尺另一侧边界）。

    步骤：
      1. Sobel Y → |grad| → OTSU 二值 → 水平投影
      2. anchor = 搜索区内投影最强 y
      3. 在 [anchor + MIN_SEP, end] 和 [start, anchor - MIN_SEP] 两段中
         分别找投影最强 y，取强度较大者
      4. 上沿=较小 y，下沿=较大 y
    """
    if enhanced is None:
        return None, None
    H, W = enhanced.shape[:2]
    x_lo = max(0, x_lo); x_hi = min(W - 1, x_hi)
    y_search_lo = max(0, y_search_lo); y_search_hi = min(H - 1, y_search_hi)

    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=5)
    abs_y = np.abs(sobel_y)
    g_max = float(np.max(abs_y))
    if g_max <= 0:
        return None, None
    abs_y_u8 = (abs_y / g_max * 255).astype(np.uint8)
    _, bw = cv2.threshold(abs_y_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    strip = bw[:, x_lo:x_hi + 1]
    hproj = np.sum(strip == 255, axis=1).astype(float)

    region_proj = hproj[y_search_lo:y_search_hi + 1]
    if len(region_proj) < 20:
        return None, None

    # ── 1. anchor = 投影最强的 y ──
    anchor_rel = int(np.argmax(region_proj))
    anchor = anchor_rel + y_search_lo

    # ── 2. 必须最小间距（卡尺金属面厚度 ≥ 150 px）──
    MIN_SEP = 150

    # ── 3. 在 anchor 远端找次强 ──
    best_far = -1
    best_far_val = 0
    # 左侧远端：[y_search_lo, anchor - MIN_SEP]
    if anchor - MIN_SEP > y_search_lo:
        left_region = hproj[y_search_lo:anchor - MIN_SEP + 1]
        if len(left_region) > 0:
            li = int(np.argmax(left_region))
            if left_region[li] > best_far_val:
                best_far_val = float(left_region[li])
                best_far = li + y_search_lo
    # 右侧远端：[anchor + MIN_SEP, y_search_hi]
    if anchor + MIN_SEP <= y_search_hi:
        right_region = hproj[anchor + MIN_SEP:y_search_hi + 1]
        if len(right_region) > 0:
            ri = int(np.argmax(right_region))
            if right_region[ri] > best_far_val:
                best_far_val = float(right_region[ri])
                best_far = ri + anchor + MIN_SEP

    if best_far < 0:
        return None, None

    top_y = min(anchor, best_far)
    bot_y = max(anchor, best_far)
    return top_y, bot_y


def _find_caliper_right_edge(enhanced: np.ndarray,
                               y_lo: int, y_hi: int,
                               x_lo: int):
    """
    在 [y_lo, y_hi] 行带内找卡尺金属面/压块右缘 x 坐标。

    v6.4 物理依据：游标卡尺金属面（含压块）的右缘是一条强水平边缘的右端。
    Sobel Y 后，把每行做"右连续段最右端 x"统计，取最大值即可。

    步骤：
      1. Sobel Y → |grad| → OTSU 二值（强水平边缘 = 1）
      2. 限定到 [y_lo, y_hi] 行
      3. 对每行，从右往左扫描，找"该行最右侧 1 像素 x"
      4. 取所有行的"最右 x"中的最大值（即整个金属面的右缘）
    """
    if enhanced is None:
        return None
    H, W = enhanced.shape[:2]
    y_lo = max(0, y_lo); y_hi = min(H - 1, y_hi)
    x_lo = max(0, x_lo)
    if y_hi - y_lo < 50:
        return None

    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=5)
    abs_y = np.abs(sobel_y)
    g_max = float(np.max(abs_y))
    if g_max <= 0:
        return None
    abs_y_u8 = (abs_y / g_max * 255).astype(np.uint8)
    _, bw = cv2.threshold(abs_y_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 只看 y 范围内的行
    strip = bw[y_lo:y_hi + 1, :]

    # 对每行找"最右侧白像素 x"
    right_xs = []
    for row in strip:
        idx = np.where(row == 255)[0]
        if len(idx) > 0 and idx[-1] > x_lo:
            right_xs.append(int(idx[-1]))

    if not right_xs:
        return None

    # 取所有右端 x 的 90 分位（去掉个别噪声/反光，保留真实右缘）
    right_x = int(np.percentile(right_xs, 90))
    if right_x <= x_lo + 200:
        return None
    return right_x


def _locate_vernier_block_x_by_gray(binary: np.ndarray,
                                      x_lo: int, x_hi: int,
                                      y_lo: int, y_hi: int) -> float:
    """
    v5.5: 用 binary 在水平方向"白像素总和最低且最连续的中央区段"识别游标压块。

    实际上 binary（自适应阈值 THRESH_BINARY，黑前景）里，主尺金属面板大部分
    是白色（背景）+ 少量黑刻线；而游标压块部分大量是黑色金属体表面 + 阴影 →
    该 x 区段的 "白像素列总数" 也较低，但**关键是要排除 ROI 两端的边缘伪低谷**
    （如 DELIXI Logo 文字密集区也会拉低白像素数）。

    策略：
      1. 求 binary 在 ROI 中下半的"白像素列计数"
      2. 平滑后取 ROI 中央 60% 范围内的最低连续段（边缘 20% 不参与）
      3. 段宽必须 > ROI 宽度 8% 才接受
    """
    if binary is None:
        return None
    h, w = binary.shape
    x_lo = max(0, x_lo)
    x_hi = min(w - 1, x_hi)
    y_lo = max(0, y_lo)
    y_hi = min(h - 1, y_hi)
    if x_hi - x_lo < 50 or y_hi - y_lo < 20:
        return None

    # 用 ROI 的中下半部分计算（游标压块通常在主尺刻度下方）
    y_mid = (y_lo + y_hi) // 2
    strip = binary[y_mid:y_hi + 1, x_lo:x_hi + 1]
    vproj = np.sum(strip == 255, axis=0).astype(float)  # 白像素列计数

    # 平滑
    win = max(15, (x_hi - x_lo) // 25)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    vproj_smooth = np.convolve(vproj, kernel, mode='same')

    # 限制搜索到中央 60%（剔除两端 DELIXI Logo / 卡尺杆等边缘干扰）
    roi_w = x_hi - x_lo + 1
    center_lo = int(roi_w * 0.20)
    center_hi = int(roi_w * 0.80)
    if center_hi <= center_lo + 50:
        return None
    central = vproj_smooth[center_lo:center_hi]

    median_v = float(np.median(vproj_smooth))
    threshold = median_v * 0.55
    below = central < threshold
    if not np.any(below):
        return None

    # 找最长连续段
    segments = []
    in_seg = False
    seg_start = 0
    for i, b in enumerate(below):
        if b and not in_seg:
            seg_start = i
            in_seg = True
        elif not b and in_seg:
            segments.append((seg_start, i))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(below)))

    if not segments:
        return None

    longest = max(segments, key=lambda s: s[1] - s[0])
    seg_len = longest[1] - longest[0]
    if seg_len < roi_w * 0.08:
        return None

    # 段中心（先转回 ROI 内部坐标，再加 x_lo）
    seg_center_in_central = (longest[0] + longest[1]) / 2.0
    center = seg_center_in_central + center_lo + x_lo
    return center


def _fallback_com_x_range(com_x: float, w: int):
    """信号过弱时回退到原 COM 固定比例宽度算法"""
    half_w = int(w * config.roi.x_center_span_ratio / 2)
    x1 = int(com_x) - half_w
    x2 = int(com_x) + half_w
    if x1 < 0:
        x2 = min(w - 1, x2 - x1)
        x1 = 0
    if x2 >= w:
        x1 = max(0, x1 - (x2 - w + 1))
        x2 = w - 1
    pad_x = max(3, int((x2 - x1) * config.roi.x_pad_ratio))
    x1 = max(0, x1 - pad_x)
    x2 = min(w - 1, x2 + pad_x)
    return x1, x2, None


def _full_roi_result(img_color, enhanced):
    """ROI 失败时返回全图"""
    return {
        'roi_color': img_color,
        'roi_gray': enhanced,
        'roi_binary': None,
        'mask_vis': img_color,
        'angle_roi': 0.0,
    }


# ═══════════════════════════════════════════════════════════
#  ② 方向矫正
# ═══════════════════════════════════════════════════════════

def orient_caliper(roi_color: np.ndarray,
                    roi_gray: np.ndarray,
                    roi_binary: np.ndarray = None) -> dict:
    """
    检测刻度线主导方向，旋转图像使刻线垂直。

    Returns:
        dict: rotated_color, rotated_gray, rotated_binary, orient_angle, orient_vis
    """
    gray = roi_gray if roi_gray is not None else cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    # Use a temporary contrast-normalized image only for orientation detection.
    # The downstream scale recognizers should receive the preprocessed ROI gray
    # directly, not a second CLAHE pass.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    orient_gray = clahe.apply(gray)

    edges = cv2.Canny(orient_gray, config.orient.canny_low, config.orient.canny_high)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=config.orient.hough_threshold,
                            minLineLength=config.orient.hough_min_length,
                            maxLineGap=config.orient.hough_max_gap)

    if lines is None or len(lines) == 0:
        return {
            'rotated_color': roi_color,
            'rotated_gray': gray,
            'rotated_binary': roi_binary,
            'orient_angle': 0.0,
            'orient_vis': roi_color,
        }

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        a = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if a < -180: a += 360
        if a > 180: a -= 360
        # 只保留近似垂直的线（刻度线方向），忽略水平方向的噪声
        # 垂直 = ±90°，允许偏差由 config 控制
        if config.orient.angle_min <= abs(a) <= config.orient.angle_max:
            angles.append(a)

    if not angles:
        return {
            'rotated_color': roi_color,
            'rotated_gray': gray,
            'rotated_binary': roi_binary,
            'orient_angle': 0.0,
            'orient_vis': roi_color,
        }

    # 计算偏离垂直方向的角度
    deviations = []
    for a in angles:
        if a >= 0:
            deviations.append(90.0 - a)
        else:
            deviations.append(-90.0 - a)

    # 缩尾均值
    deviations.sort()
    n = len(deviations)
    trim = max(1, int(n * config.orient.trim_ratio))
    trimmed = deviations[trim:n - trim] if n > trim * 2 else deviations
    angle = float(np.median(trimmed))

    # 小角度不旋转：< 0.3° 时偏差 < ~5px（1.5° = 26px 投影洒散太大）
    # HoughLinesP 统计有 ±0.1° 随机噪声，0.3° 以下转了反而抖动
    if abs(angle) < config.orient.rotate_min_angle or abs(angle) > config.orient.rotate_max_angle:
        angle = 0.0

    rotated_color = rotate_image(roi_color, angle)
    rotated_gray = rotate_image(gray, angle)
    rotated_binary = rotate_image(roi_binary, angle) if roi_binary is not None else None

    orient_vis = _make_orient_vis(roi_color, rotated_color, angle)

    return {
        'rotated_color': rotated_color,
        'rotated_gray': rotated_gray,
        'rotated_binary': rotated_binary,
        'orient_angle': angle,
        'orient_vis': orient_vis,
    }


# ═══════════════════════════════════════════════════════════
#  可视化 — v3 完整三列布局
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
#  可视化 — v7: Sobel X + COM Y + 最长等间距序列
# ═══════════════════════════════════════════════════════════


def _make_sobel_vis(enhanced: np.ndarray,
                    grad_u8: np.ndarray,
                    binary: np.ndarray,
                    panel_w: int = 420) -> np.ndarray:
    """Sobel X 垂直边缘检测可视化：原图 → 梯度热力图 → OTSU 二值图"""
    h, w = enhanced.shape[:2]
    pw = panel_w
    gap = 2
    thumb_w = (pw - gap * 2) // 3
    thumb_h = int(h * thumb_w / max(w, 1))

    def _thumb(img):
        if len(img.shape) == 2:
            t = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

    # Panel 1: 原始增强灰度图
    p1 = _thumb(enhanced)
    cv2.putText(p1, "Original Gray", (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

    # Panel 2: Sobel X 梯度热力图
    grad_color = cv2.applyColorMap(grad_u8, cv2.COLORMAP_JET)
    p2 = cv2.resize(grad_color, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    cv2.putText(p2, "Sobel X |grad|", (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

    # Panel 3: OTSU 二值图
    p3 = _thumb(binary)
    cv2.putText(p3, "OTSU Binary", (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

    row = np.zeros((thumb_h, pw, 3), dtype=np.uint8)
    row[:] = (30, 30, 35)
    row[:, :thumb_w] = p1
    row[:, thumb_w + gap:2 * thumb_w + gap] = p2
    col3_start = 2 * thumb_w + 2 * gap
    row[:, col3_start:col3_start + thumb_w] = p3

    # 底部标签
    label_h = 16
    vis = np.zeros((thumb_h + label_h, pw, 3), dtype=np.uint8)
    vis[:] = (30, 30, 35)
    vis[:thumb_h, :] = row
    cv2.putText(vis, "Sobel X — 垂直边缘检测 | 刻度线垂直 → 对竖直亮度变化敏感",
                (4, thumb_h + label_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 170), 1)
    return vis


def _make_com_y_vis(com_diag: dict, total_w: int, full_h: int) -> np.ndarray:
    """Y 方向 COM 质心法可视化：每行白像素总数投影图 + COM 质心 + ROI 边界标注"""
    if com_diag is None:
        empty = np.zeros((140, total_w, 3), dtype=np.uint8)
        empty[:] = (30, 30, 35)
        cv2.putText(empty, "COM Y: no data", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        return empty

    hproj = com_diag['hproj']
    com_y = com_diag['com_y']
    span_lo = com_diag['span_lo']
    span_hi = com_diag['span_hi']
    y1 = com_diag['y1']
    y2 = com_diag['y2']

    plot_h = 160
    label_h = 20
    margin_left = 50
    plot_w = total_w - margin_left - 16
    vis = np.zeros((plot_h + label_h, total_w, 3), dtype=np.uint8)
    vis[:] = (30, 30, 35)

    h_max = float(np.max(hproj))
    h_norm = hproj / h_max if h_max > 0 else hproj

    # 缩放：x=像素值, y=行号
    scale_y = (plot_h - 20) / full_h
    scale_v = (plot_w - 10) / max(h_max, 1)

    plot = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)
    plot[:] = (22, 22, 28)
    # 网格线
    for gy in range(0, plot_h, 30):
        cv2.line(plot, (0, gy), (plot_w, gy), (35, 35, 40), 1)
    for gx in range(0, plot_w, 40):
        cv2.line(plot, (gx, 0), (gx, plot_h), (35, 35, 40), 1)

    # 曲线：y 轴 = 行号，x 轴 = 白像素总数
    pts = []
    for yi, val in enumerate(hproj):
        px = int(val * scale_v / max(h_max, 1) * (plot_w - 15) + 8)
        py = int(yi * scale_y) + 10
        pts.append((px, py))

    # 填充 + 曲线
    for i in range(len(pts)):
        cv2.line(plot, (8, pts[i][1]), (pts[i][0], pts[i][1]),
                 (60, 100, 160, 60), 1, cv2.LINE_AA)
    for i in range(len(pts) - 1):
        cv2.line(plot, pts[i], pts[i + 1], (100, 180, 255), 1)

    # COM 质心线（红色虚线）
    com_py = int(com_y * scale_y) + 10
    cv2.line(plot, (0, com_py), (plot_w, com_py), (80, 60, 200), 1, cv2.LINE_AA)

    # span 范围半透明带
    sp_lo = int(span_lo * scale_y) + 10
    sp_hi = int(span_hi * scale_y) + 10
    overlay = np.zeros_like(plot)
    cv2.rectangle(overlay, (0, sp_lo), (plot_w, sp_hi), (40, 80, 40), -1)
    plot = cv2.addWeighted(plot, 0.85, overlay, 0.15, 0)

    # 最终 ROI 边界（黄色虚线）
    ry1 = int(y1 * scale_y) + 10
    ry2 = int(y2 * scale_y) + 10
    cv2.line(plot, (0, ry1), (plot_w, ry1), (50, 200, 220), 1, cv2.LINE_AA)
    cv2.line(plot, (0, ry2), (plot_w, ry2), (50, 200, 220), 1, cv2.LINE_AA)

    # 放置 plot 到 vis（右移 margin_left 给 y 轴标签留空间）
    vis[:plot_h, margin_left:margin_left + plot_w] = plot

    # y 轴标签
    cv2.putText(vis, "Y", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(vis, "0", (margin_left - 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (140, 140, 140), 1)
    cv2.putText(vis, f"{full_h}", (margin_left - 28, plot_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (140, 140, 140), 1)

    # 底部文字
    cv2.putText(vis,
                f"Y COM Centroid — 质心: y={com_y:.1f}  |  span: [{span_lo},{span_hi}]  |  ROI: [{y1},{y2}]",
                (4, plot_h + label_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 170), 1)

    return vis


def _make_longest_seq_x_vis(x_diag: dict, total_w: int) -> np.ndarray:
    """X 方向最长等间距序列法可视化：垂直投影 + 全峰 + 强峰 + 选中段"""
    if x_diag is None:
        empty = np.zeros((260, total_w, 3), dtype=np.uint8)
        empty[:] = (30, 30, 35)
        cv2.putText(empty, "Longest Equidistant Sequence: no data", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        return empty

    vproj = x_diag.get('vproj_norm')
    peaks_all = x_diag.get('peaks_all')
    strong_peaks = x_diag.get('strong_peaks')
    tick_gap = x_diag.get('tick_gap', 0)
    th_strength = x_diag.get('th_strength', 0)
    seg_lo = x_diag.get('seg_lo')
    seg_hi = x_diag.get('seg_hi')

    plot_h = 200
    info_h = 50
    total_h = plot_h + info_h
    n = len(vproj) if vproj is not None else 0
    if n == 0:
        empty = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        empty[:] = (30, 30, 35)
        return empty

    vis = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    vis[:] = (30, 30, 35)

    plot_w = total_w - 16
    ox, oy = 8, 8

    plot = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)
    plot[:] = (22, 22, 28)
    # 网格
    for gy in range(0, plot_h, 25):
        cv2.line(plot, (0, gy), (plot_w, gy), (32, 32, 36), 1)

    scale_x = (plot_w - 10) / max(n, 1)

    def px(x):
        return int(x * scale_x) + 5

    def py(v):
        return plot_h - 8 - int(v * (plot_h - 20))

    # 投影曲线
    for i in range(min(n - 1, plot_w - 2)):
        x0, x1 = px(i), px(i + 1)
        y0, y1 = py(vproj[i]), py(vproj[i + 1])
        cv2.line(plot, (x0, y0), (x1, y1), (120, 120, 140), 1)

    # 全峰（蓝色小点）
    for pi in peaks_all:
        if 0 <= pi < n:
            cv2.circle(plot, (px(pi), py(vproj[pi])), 3, (180, 120, 60), -1)

    # 强度阈值线
    th_py = py(th_strength)
    cv2.line(plot, (0, th_py), (plot_w, th_py), (120, 120, 200), 1, cv2.LINE_AA)
    cv2.putText(plot, f"th={th_strength:.2f}", (2, th_py - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (120, 120, 200), 1)

    # 强峰（橙色大点）
    for pi in strong_peaks:
        if 0 <= pi < n:
            cv2.circle(plot, (px(pi), py(vproj[pi])), 5, (80, 180, 255), -1)

    # 选中最长段（绿色半透明条）
    if seg_lo is not None and seg_hi is not None:
        sx1 = px(seg_lo)
        sx2 = px(seg_hi)
        overlay = np.zeros_like(plot)
        cv2.rectangle(overlay, (sx1, 0), (sx2, plot_h), (0, 80, 0), -1)
        plot = cv2.addWeighted(plot, 0.88, overlay, 0.12, 0)
        # 边界
        cv2.line(plot, (sx1, 0), (sx1, plot_h), (50, 200, 220), 1, cv2.LINE_AA)
        cv2.line(plot, (sx2, 0), (sx2, plot_h), (50, 200, 220), 1, cv2.LINE_AA)
        cv2.putText(plot, f"sel [{seg_lo},{seg_hi}]", (sx1 + 2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (50, 220, 100), 1)

    vis[:plot_h, ox:ox + plot_w] = plot

    # 信息栏
    seq_len = len(strong_peaks)
    seg_span = (seg_hi - seg_lo) if (seg_lo is not None and seg_hi is not None) else 0
    info_text = (f"tick_gap={tick_gap:.1f}px  |  "
                 f"强峰: {seq_len}  |  "
                 f"选中段跨度: {seg_span}px  |  "
                 f"阈值: {th_strength:.2f}")
    if x_diag.get('x1') is not None:
        info_text += f"  |  ROI x: [{x_diag['x1']},{x_diag['x2']}]"

    cv2.putText(vis, info_text, (4, plot_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(vis, "X 方向 — 最长等间距强峰序列法 | 蓝=全峰 橙=强峰 绿带=选中最长段",
                (4, plot_h + info_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (140, 140, 145), 1)

    return vis


def _make_roi_vis_v3(original: np.ndarray,
                      roi_cropped: np.ndarray,
                      diag: dict) -> np.ndarray:
    """
    ROI 提取全流程可视化 (v4)。

    布局（两列）:
      左列 — 原图 + 裁剪框（撑满整列高度）
      右列 — 上 = ROI 裁剪结果，下 = 识别过程（状态条 + 二值面板 + 投影）
    """
    h, w = original.shape[:2]
    rh, rw = roi_cropped.shape[:2]
    gap = 3

    LEFT_W = min(w, 520)
    RIGHT_W = min(w, 420)

    # ── 左列：原图 + 裁剪框 ──
    y1, y2, x1, x2 = diag.get('y1', 0), diag.get('y2', 0), diag.get('x1', 0), diag.get('x2', 0)

    left_scale = LEFT_W / w
    left_disp_h = int(h * left_scale)
    left_panel = cv2.resize(original.copy(), (LEFT_W, left_disp_h), interpolation=cv2.INTER_AREA)
    sy1, sx1 = int(y1 * left_scale), int(x1 * left_scale)
    sy2, sx2 = int(y2 * left_scale), int(x2 * left_scale)
    cv2.rectangle(left_panel, (sx1, sy1), (sx2, sy2), (0, 255, 220), 3)
    cv2.putText(left_panel, f"({x1},{y1})", (sx1, max(sy1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 220), 1)
    cv2.putText(left_panel, f"({x2},{y2})", (max(sx2 - 50, 4), min(sy2 + 16, left_disp_h - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 220), 1)

    # 左列标题
    col_height = max(20, left_disp_h + 16)
    left_col = np.zeros((col_height, LEFT_W, 3), dtype=np.uint8)
    left_col[:] = (30, 30, 35)
    left_col[:left_disp_h, :] = left_panel
    cv2.putText(left_col, "Original + Crop Box", (6, left_disp_h + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    # ── 右列上：ROI 裁剪结果（缩小）──
    roi_title_h = 16
    roi_disp_w = int(RIGHT_W * 0.55)  # 缩小宽度
    roi_disp_h = max(int(rh * roi_disp_w / max(rw, 1)), 22)
    roi_disp = cv2.resize(roi_cropped, (roi_disp_w, roi_disp_h), interpolation=cv2.INTER_AREA)
    # 居中放在 RIGHT_W 面板上
    roi_pad_left = (RIGHT_W - roi_disp_w) // 2
    roi_panel = np.zeros((roi_disp_h + roi_title_h, RIGHT_W, 3), dtype=np.uint8)
    roi_panel[:] = (30, 30, 35)
    roi_panel[roi_title_h:roi_title_h + roi_disp_h, roi_pad_left:roi_pad_left + roi_disp_w] = roi_disp
    cv2.putText(roi_panel, f"ROI Crop [{rw}x{rh}] px", (6, roi_title_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    # ── 右列下：识别过程可视化 ──
    source_label = diag.get('source', 'adaptive')
    source_map = {
        'adaptive':       ('Adaptive Bin Projection',      (0, 220, 80)),
        'sobel':          ('Sobel X Vertical Edges',        (100, 200, 255)),
        'sobel_fallback': ('Fallback: Sobel X',             (255, 180, 60)),
        'contour':        ('Fallback: Contour scoring',     (255, 120, 80)),
    }
    status_text, status_color = source_map.get(source_label, source_map['adaptive'])

    bar_h = 24
    status_bar = np.zeros((bar_h, RIGHT_W, 3), dtype=np.uint8)
    if source_label == 'adaptive':
        status_bar[:] = (35, 60, 35)
    elif 'sobel' in source_label:
        status_bar[:] = (55, 45, 25)
    else:
        status_bar[:] = (55, 30, 30)
    cv2.putText(status_bar, status_text, (6, bar_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1)

    binary_adapt = diag.get('binary_adapt')
    seg_adapt = _make_binary_row(binary_adapt, RIGHT_W, h, w, y1, y2, x1, x2,
                                  title="Adaptive Threshold Binary",
                                  source_used=(source_label == 'adaptive'))

    binary_vedge = diag.get('binary_vedge')
    seg_sobel = _make_binary_row(binary_vedge, RIGHT_W, h, w, y1, y2, x1, x2,
                                  title="Sobel X Vertical Edges",
                                  source_used=('sobel' in source_label))

    contour_vis = diag.get('contour_vis')
    seg_contour = None
    if contour_vis is not None and contour_vis.size > 0:
        ch_s, cw_s = contour_vis.shape[:2]
        sc = RIGHT_W / max(cw_s, 1)
        seg_contour = cv2.resize(contour_vis, (RIGHT_W, max(int(ch_s * sc), 36)), interpolation=cv2.INTER_AREA)

    # ── v7 新面板：Sobel X 梯度可视化 ──
    sobel_grad = diag.get('sobel_gradient')
    enh_ref = diag.get('enhanced')
    seg_sobel_vis = None
    if sobel_grad is not None and enh_ref is not None and binary_vedge is not None:
        seg_sobel_vis = _make_sobel_vis(enh_ref, sobel_grad, binary_vedge, RIGHT_W)

    # ── v7 新面板：COM Y 质心法可视化 ──
    com_y_diag = diag.get('com_y_diag')
    seg_com_y = _make_com_y_vis(com_y_diag, RIGHT_W, h)

    # ── v7 新面板：X 方向最长等间距序列可视化 ──
    com_x_diag = diag.get('com_x_diag')
    seg_com_x = _make_longest_seq_x_vis(com_x_diag, RIGHT_W)

    process_panels = [status_bar, seg_adapt, seg_sobel]
    if seg_sobel_vis is not None:
        process_panels.append(seg_sobel_vis)
    process_panels.append(seg_com_y)
    process_panels.append(seg_com_x)
    if seg_contour is not None:
        process_panels.append(seg_contour)
    process_h = sum(p.shape[0] for p in process_panels) + gap * (len(process_panels) - 1)

    process_col = np.zeros((process_h, RIGHT_W, 3), dtype=np.uint8)
    process_col[:] = (30, 30, 35)
    yc = 0
    for p in process_panels:
        ph = p.shape[0]
        process_col[yc:yc + ph, :] = p
        yc += ph + gap

    # ── 右列组装 ──
    right_h = roi_panel.shape[0] + gap + process_col.shape[0]
    right_col = np.zeros((right_h, RIGHT_W, 3), dtype=np.uint8)
    right_col[:] = (30, 30, 35)
    right_col[:roi_panel.shape[0], :] = roi_panel
    right_col[roi_panel.shape[0] + gap:roi_panel.shape[0] + gap + process_h, :] = process_col

    # ── 两列拼接 ──
    total_h = max(col_height, right_h) + 18
    total_w = LEFT_W + gap + RIGHT_W
    vis = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    vis[:] = (30, 30, 35)
    vis[:col_height, :LEFT_W] = left_col
    vis[:right_h, LEFT_W + gap:total_w] = right_col

    cv2.putText(vis, "STEP 1a: ROI Extraction", (5, total_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 125), 1)
    return vis


def _make_binary_row(binary: np.ndarray,
                      disp_w: int,
                      full_h: int, full_w: int,
                      y1: int, y2: int, x1: int, x2: int,
                      title: str = "Binary",
                      source_used: bool = False) -> np.ndarray:
    """
    紧凑单行：左=缩略图(90px) | 中=H投影 | 右=V投影，高度 ~80px。
    """
    gap = 2
    row_h = 80
    thumb_w = 90
    proj_w = max((disp_w - thumb_w - gap * 2) // 2, 30)

    row = np.zeros((row_h, disp_w, 3), dtype=np.uint8)
    row[:] = (30, 30, 35)

    if binary is not None and binary.size > 0:
        t = cv2.resize(binary, (thumb_w, row_h), interpolation=cv2.INTER_AREA)
        row[:row_h, :thumb_w] = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    if source_used:
        cv2.putText(row, "USED", (2, row_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 100), 1)

    x_off = thumb_w + gap
    if binary is not None and binary.size > 0:
        hproj = _safe_hproj(binary, full_h)
    else:
        hproj = np.zeros(full_h)
    hplot = _draw_1d_projection(hproj, proj_w, row_h,
                                 markers=[(y1, (255, 180, 60, 140)),
                                          (y2, (255, 180, 60, 140))])
    row[:, x_off:x_off + proj_w] = hplot
    cv2.putText(row, "H", (x_off + 2, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140, 140, 140), 1)

    x_off2 = x_off + proj_w + gap
    if binary is not None and y2 > y1:
        strip = binary[y1:y2 + 1, :]
        vproj = _safe_vproj(strip, full_w)
    else:
        vproj = np.zeros(full_w)
    vplot = _draw_1d_projection(vproj, proj_w, row_h,
                                 markers=[(x1, (0, 210, 100, 140)),
                                          (x2, (0, 210, 100, 140))])
    row[:, x_off2:x_off2 + proj_w] = vplot
    cv2.putText(row, "V", (x_off2 + 2, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140, 140, 140), 1)

    cv2.putText(row, title, (thumb_w + gap + 4, row_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (160, 160, 170), 1)
    return row


def _make_contour_diag_vis(binary_vedge: np.ndarray,
                            closed: np.ndarray,
                            opened: np.ndarray,
                            contours: list,
                            img_h: int, img_w: int) -> np.ndarray:
    """
    构建轮廓诊断可视化图：
    Sobel二值 | 闭运算 | 开运算 | 开运算+轮廓评分
    """
    gap = 2

    # 统一缩放到合适高度
    panel_h = 120
    scale = panel_h / max(img_h, 1)
    panel_w = int(img_w * scale)

    def _thumb(bin_img):
        t = cv2.resize(bin_img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

    t1 = _thumb(binary_vedge)
    t2 = _thumb(closed)
    t3 = _thumb(opened)
    t4 = _thumb(opened)

    # t4 叠加轮廓 + 打分
    for cnt in contours:
        score = _score_contour_as_scale_roi(cnt, img_h, img_w)
        color = (0, 255, 100) if score >= 0.5 else (255, 180, 60) if score >= 0.2 else (100, 120, 255)
        # 缩放到面板坐标
        cnt_scaled = (cnt.astype(float) * scale).astype(np.int32)
        cv2.drawContours(t4, [cnt_scaled], -1, color, 1)
        if score >= 0.2:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'] * scale)
                cy = int(M['m01'] / M['m00'] * scale)
                cv2.putText(t4, f"{score:.2f}", (cx - 12, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    n_contours = len(contours) if contours else 0
    titles = [
        "Sobel X Binary",
        f"Morph Close (kw={max(12, int(img_w * config.roi.morph_kernel_ratio))})",
        "Morph Open (denoise)",
        f"Contours + Score ({n_contours} found)",
    ]
    panels = [t1, t2, t3, t4]

    # 标题条
    title_h = 14
    total_w = panel_w * 4 + gap * 3
    diag = np.zeros((panel_h + title_h, total_w, 3), dtype=np.uint8)
    diag[:] = (30, 30, 35)

    for i, (p, ttl) in enumerate(zip(panels, titles)):
        x_off = i * (panel_w + gap)
        diag[title_h:title_h + panel_h, x_off:x_off + panel_w] = p
        cv2.putText(diag, ttl, (x_off + 3, title_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    return diag


def _safe_hproj(binary: np.ndarray, h: int) -> np.ndarray:
    if binary is None or binary.size == 0:
        return np.zeros(h)
    p = np.sum(binary, axis=1).astype(float)
    if np.max(p) > 0:
        p /= np.max(p)
    return p


def _safe_vproj(strip: np.ndarray, w: int) -> np.ndarray:
    if strip is None or strip.size == 0:
        return np.zeros(w)
    p = np.sum(strip, axis=0).astype(float)
    if np.max(p) > 0:
        p /= np.max(p)
    return p


def _draw_1d_projection(signal: np.ndarray, width: int, height: int,
                          markers: list = None,
                          title: str = "") -> np.ndarray:
    """绘制一维投影曲线"""
    plot = np.zeros((height, width, 3), dtype=np.uint8)
    plot[:] = (25, 25, 30)
    n = len(signal)
    if n == 0 or width <= 10:
        return plot
    s_max = float(np.max(signal))
    s_norm = signal / s_max if s_max > 0 else signal
    plot_w = width - 10
    for i in range(min(n - 1, plot_w - 1)):
        x0 = int(i * plot_w / n) + 5
        x1 = int((i + 1) * plot_w / n) + 5
        y0 = height - 3 - int(s_norm[i] * (height - 8))
        y1 = height - 3 - int(s_norm[i + 1] * (height - 8))
        cv2.line(plot, (x0, y0), (x1, y1), (255, 180, 60), 1)
    if markers:
        for item in markers:
            idx, color = item[0], item[1]
            c = tuple(int(x) for x in color[:3])  # BGR only
            if 0 <= idx < n:
                px = int(idx * plot_w / n) + 5
                cv2.line(plot, (px, 0), (px, height - 1), c, 1)
    if title:
        cv2.putText(plot, title, (8, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    return plot


def _make_orient_vis(before: np.ndarray, after: np.ndarray, angle: float) -> np.ndarray:
    """方向矫正前后对比"""
    from .utils import make_comparison_vis
    vis = make_comparison_vis(
        before, after,
        f"Before (deviation: {angle:.1f} deg)",
        "After Correction"
    )
    h = vis.shape[0]
    cv2.putText(vis, "STEP 1b: Orientation Correction", (5, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 125), 1)
    return vis
