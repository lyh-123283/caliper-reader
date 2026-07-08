"""
步骤 1 — ROI 提取 + 方向矫正

① ROI 提取：投影法定位刻度区域 → 裁剪（+ 轮廓验证回退）
② 方向矫正：HoughLinesP 检测刻线方向 → 旋转使刻线垂直
"""

import cv2
import numpy as np
from typing import Tuple, List

from .utils import rotate_image
from .config import config
from .vernier_rectify import _find_vernier_body_x_range


# ═══════════════════════════════════════════════════════════
#  ① ROI 提取：投影法定位刻度区域
# ═══════════════════════════════════════════════════════════

def locate_roi_lowres(img_color: np.ndarray,
                      max_width: int = 1600) -> dict:
    h, w = img_color.shape[:2]
    scale = min(1.0, float(max_width) / float(w)) if w > 0 else 1.0
    if scale < 1.0:
        small = cv2.resize(
            img_color, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA)
    else:
        small = img_color.copy()

    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    enhanced = _make_lowres_roi_enhanced(gray)
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=_odd_between(min(sh, sw) // 18, 15, 51),
        C=9,
    )
    fg = cv2.bitwise_not(binary)

    y1, y2 = _proj_find_y_range(fg, sh)
    x1, x2, x_diag = _proj_find_x_range(fg, y1, y2, sw)
    if y2 - y1 < config.roi.min_roi_height or x2 - x1 < config.roi.min_roi_width:
        return _lowres_roi_failure(img_color, small, enhanced, fg, (x1, y1, x2, y2))

    refined = _refine_roi_by_vernier_block(enhanced, y1, y2, x1, x2)
    if refined is not None:
        y1, y2, x1, x2 = refined
    reading_refined = _refine_roi_to_reading_window(enhanced, y1, y2, x1, x2, x_diag)
    if reading_refined is not None:
        y1, y2, x1, x2 = reading_refined

    inv_scale = 1.0 / scale if scale > 0 else 1.0
    ox1 = int(np.floor(x1 * inv_scale))
    oy1 = int(np.floor(y1 * inv_scale))
    ox2 = int(np.ceil((x2 + 1) * inv_scale))
    oy2 = int(np.ceil((y2 + 1) * inv_scale))

    roi_w = max(1, ox2 - ox1)
    roi_h = max(1, oy2 - oy1)
    ox1 = max(0, ox1 - max(30, int(roi_w * 0.015)))
    oy1 = max(0, oy1 - max(15, int(roi_h * 0.040)))
    ox2 = min(w, ox2 + max(30, int(roi_w * 0.015)))
    oy2 = min(h, oy2 + max(15, int(roi_h * 0.040)))

    if ox2 - ox1 < config.roi.min_roi_width or oy2 - oy1 < config.roi.min_roi_height:
        return _lowres_roi_failure(img_color, small, enhanced, fg, (x1, y1, x2, y2))

    debug = _make_lowres_roi_debug(
        img_color, small, enhanced, fg,
        (x1, y1, x2, y2), (ox1, oy1, ox2, oy2))
    return {
        'roi_color': img_color[oy1:oy2, ox1:ox2].copy(),
        'x_offset': ox1,
        'y_offset': oy1,
        'roi_box_original': (ox1, oy1, ox2, oy2),
        'roi_box_lowres': (x1, y1, x2, y2),
        'scale': scale,
        'lowres_debug': debug,
        'locate_failed': False,
    }


def _make_lowres_roi_enhanced(gray: np.ndarray) -> np.ndarray:
    gamma = config.preprocess.gamma
    if gamma is not None and abs(float(gamma) - 1.0) > 1e-6:
        inv_gamma = 1.0 / float(gamma)
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype(np.uint8)
        gray = cv2.LUT(gray, table)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _odd_between(value: int, lo: int, hi: int) -> int:
    value = max(lo, min(hi, int(value)))
    if value % 2 == 0:
        value += 1
    return min(hi if hi % 2 == 1 else hi - 1, value)


def _lowres_roi_failure(img_color: np.ndarray,
                        small: np.ndarray,
                        enhanced: np.ndarray,
                        fg: np.ndarray,
                        box: tuple) -> dict:
    debug = _make_lowres_roi_debug(
        img_color, small, enhanced, fg, box,
        (0, 0, img_color.shape[1], img_color.shape[0]))
    return {
        'roi_color': None,
        'x_offset': 0,
        'y_offset': 0,
        'roi_box_original': None,
        'roi_box_lowres': box,
        'scale': 1.0,
        'lowres_debug': debug,
        'locate_failed': True,
    }


def _make_lowres_roi_debug(original: np.ndarray,
                           small: np.ndarray,
                           enhanced: np.ndarray,
                           fg: np.ndarray,
                           lowres_box: tuple,
                           original_box: tuple) -> np.ndarray:
    sw = small.shape[1]
    x1, y1, x2, y2 = [int(v) for v in lowres_box]
    ox1, oy1, ox2, oy2 = [int(v) for v in original_box]

    small_vis = small.copy()
    cv2.rectangle(small_vis, (x1, y1), (x2, y2), (0, 120, 255), 2)
    cv2.putText(small_vis, "lowres ROI", (max(4, x1), max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2, cv2.LINE_AA)

    original_h = max(1, int(original.shape[0] * sw / original.shape[1]))
    original_vis = cv2.resize(original, (sw, original_h), interpolation=cv2.INTER_AREA)
    sx = sw / float(original.shape[1])
    sy = original_h / float(original.shape[0])
    cv2.rectangle(original_vis, (int(ox1 * sx), int(oy1 * sy)),
                  (int(ox2 * sx), int(oy2 * sy)), (0, 255, 120), 2)
    cv2.putText(original_vis, "mapped original ROI", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2, cv2.LINE_AA)

    enhanced_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    fg_vis = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(enhanced_vis, (x1, y1), (x2, y2), (0, 120, 255), 2)
    cv2.rectangle(fg_vis, (x1, y1), (x2, y2), (0, 120, 255), 2)

    rows = [small_vis, original_vis, enhanced_vis, fg_vis]
    out_w = max(r.shape[1] for r in rows)
    padded = []
    for row in rows:
        if row.shape[1] < out_w:
            pad = np.zeros((row.shape[0], out_w - row.shape[1], 3), dtype=np.uint8)
            pad[:] = (30, 30, 35)
            row = np.hstack([row, pad])
        padded.append(row)
    gap = np.zeros((6, out_w, 3), dtype=np.uint8)
    gap[:] = (30, 30, 35)
    return np.vstack([padded[0], gap, padded[1], gap, padded[2], gap, padded[3]])


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


def _refine_roi_to_reading_window(enhanced: np.ndarray,
                                  y1: int, y2: int,
                                  x1: int, x2: int,
                                  x_diag: dict = None):
    if enhanced is None:
        return None
    H, W = enhanced.shape[:2]
    x1 = max(0, int(x1))
    x2 = min(W - 1, int(x2))
    y1 = max(0, int(y1))
    y2 = min(H - 1, int(y2))
    if x2 - x1 < 300 or y2 - y1 < 80:
        return None

    crop = enhanced[y1:y2 + 1, x1:x2 + 1]
    h, w = crop.shape[:2]
    lower_y = int(h * 0.45)
    if lower_y >= h - 20:
        return None

    body_x1, body_x2 = _find_vernier_body_x_range(crop[lower_y:, :])
    body_w = body_x2 - body_x1
    if body_w < max(120, int(w * 0.18)):
        return None

    tick_gap = _reading_window_tick_gap(x_diag, body_w)
    left_margin = max(int(tick_gap * 20), int(body_w * 0.35), 220)
    right_margin = max(int(tick_gap * 10), int(body_w * 0.12), 120)

    nx1 = max(0, int(body_x1) - left_margin)
    nx2 = min(w - 1, int(body_x2) + right_margin)
    if nx2 - nx1 < max(int(tick_gap * 30), int(body_w * 1.10)):
        return None
    if nx2 - nx1 > w * 0.92:
        return None

    return y1, y2, x1 + nx1, x1 + nx2


def _reading_window_tick_gap(x_diag: dict, body_w: int) -> float:
    if x_diag:
        try:
            tick_gap = float(x_diag.get('tick_gap', 0.0))
            if tick_gap >= 3.0:
                return tick_gap
        except (TypeError, ValueError):
            pass
    return max(8.0, float(body_w) / 55.0)


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


# ═══════════════════════════════════════════════════════════
#  ② 方向矫正
# ═══════════════════════════════════════════════════════════

def _white_foreground_binary(binary: np.ndarray, gray: np.ndarray) -> np.ndarray:
    if binary is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        work = clahe.apply(gray)
        _, out = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return out
    out = binary.copy()
    if float(np.mean(out > 0)) > 0.5:
        out = cv2.bitwise_not(out)
    return out


def _estimate_orient_angle_from_tick_pixels(gray: np.ndarray,
                                            binary: np.ndarray) -> Tuple[float, dict]:
    h, w = gray.shape[:2]
    if h < 40 or w < 80:
        return None, {}

    fg = _white_foreground_binary(binary, gray)
    try:
        from .region_split import _split_by_vernier_tick_band, _split_by_gray_seam
        split_y = _split_by_vernier_tick_band(gray, fg, h, w)
        if split_y is None:
            split_y = _split_by_gray_seam(gray, h, w)
    except Exception:
        split_y = None
    if split_y is None:
        split_y = int(h * 0.58)

    band_h = max(36, min(90, int(h * 0.22)))
    y1 = max(0, int(split_y) - band_h)
    y2 = max(y1 + 1, min(h, int(split_y) - 4))
    band = fg[y1:y2, :]
    if band.size == 0:
        return None, {'split_y': split_y, 'count': 0}

    kernel_h = max(9, min(31, ((y2 - y1) // 2) * 2 + 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    vertical = cv2.morphologyEx(band, cv2.MORPH_OPEN, vertical_kernel)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (vertical > 0).astype('uint8'), 8)

    deviations = []
    min_h = max(10, int((y2 - y1) * 0.28))
    for label in range(1, labels_count):
        x, y, bw, bh, area = stats[label]
        if bh < min_h or bw > 14 or area < bh * 0.6:
            continue
        ys, xs = np.where(labels == label)
        if len(xs) < 8:
            continue
        rows = []
        centers = []
        for yy in np.unique(ys):
            xx = xs[ys == yy]
            if len(xx) > 0:
                rows.append(float(yy + y1))
                centers.append(float(np.mean(xx)))
        if len(rows) < 6:
            continue
        slope, _ = np.polyfit(np.asarray(rows), np.asarray(centers), 1)
        deviation = float(np.degrees(np.arctan(slope)))
        if abs(deviation) <= 4.0:
            deviations.append(deviation)

    diag = {'split_y': int(split_y), 'band_y1': y1, 'band_y2': y2,
            'count': len(deviations)}
    if len(deviations) < 30:
        return None, diag

    arr = np.asarray(deviations, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    keep = np.abs(arr - median) <= max(0.35, 3.0 * mad)
    if np.count_nonzero(keep) >= 20:
        median = float(np.median(arr[keep]))
    diag.update({'raw_deviation': median, 'mad': mad,
                 'kept': int(np.count_nonzero(keep))})
    if mad > 0.9:
        return None, diag

    angle = -median
    if abs(angle) < 0.15 or abs(angle) > min(config.orient.rotate_max_angle, 4.0):
        angle = 0.0
    return angle, diag


def _estimate_orient_angle_from_hough(gray: np.ndarray) -> float:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    orient_gray = clahe.apply(gray)
    edges = cv2.Canny(orient_gray, config.orient.canny_low, config.orient.canny_high)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=max(25, int(config.orient.hough_threshold * 0.65)),
                            minLineLength=max(config.orient.hough_min_length, int(gray.shape[1] * 0.03)),
                            maxLineGap=max(config.orient.hough_max_gap, 12))
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        a = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if a < -180:
            a += 360
        if a > 180:
            a -= 360
        if config.orient.angle_min <= abs(a) <= config.orient.angle_max:
            angles.append(a)
    if not angles:
        return 0.0

    deviations = []
    for a in angles:
        if a >= 0:
            deviations.append(a - 90.0)
        else:
            deviations.append(a + 90.0)
    deviations.sort()
    n = len(deviations)
    trim = max(1, int(n * config.orient.trim_ratio))
    trimmed = deviations[trim:n - trim] if n > trim * 2 else deviations
    angle = float(np.median(trimmed))
    if abs(angle) < config.orient.rotate_min_angle or abs(angle) > min(config.orient.rotate_max_angle, 4.0):
        angle = 0.0
    return angle


def _estimate_seam_angle_from_gradient(gray: np.ndarray,
                                       binary: np.ndarray) -> Tuple[float, dict]:
    h, w = gray.shape[:2]
    if h < 40 or w < 160:
        return None, {}

    fg = _white_foreground_binary(binary, gray)
    try:
        from .region_split import _split_by_vernier_tick_band, _split_by_gray_seam
        split_y = _split_by_vernier_tick_band(gray, fg, h, w)
        if split_y is None:
            split_y = _split_by_gray_seam(gray, h, w)
    except Exception:
        split_y = None
    if split_y is None:
        split_y = int(h * 0.58)

    band_half = max(28, h // 12)
    y1 = max(0, int(split_y) - band_half)
    y2 = min(h, int(split_y) + band_half)
    crop = gray[y1:y2, :]
    if crop.size == 0:
        return None, {'split_y': int(split_y), 'points': 0}

    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    grad_y = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
    score = np.abs(grad_y)
    score = cv2.blur(score, (max(9, w // 180), 1))

    xs = []
    ys = []
    weights = []
    step = max(4, w // 450)
    margin = max(20, w // 80)
    threshold = max(4.0, float(np.percentile(score, 78)))
    for x0 in range(margin, w - margin, step):
        x1b = max(0, x0 - step)
        x2b = min(w, x0 + step + 1)
        col = np.mean(score[:, x1b:x2b], axis=1)
        if col.size == 0:
            continue
        yi = int(np.argmax(col))
        val = float(col[yi])
        if val < threshold:
            continue
        xs.append(float(x0))
        ys.append(float(yi + y1))
        weights.append(val)

    diag = {'split_y': int(split_y), 'points': len(xs)}
    if len(xs) < 80:
        return None, diag

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    weight = np.asarray(weights, dtype=float)
    keep = np.ones(len(x), dtype=bool)
    coef = None
    mad = 0.0
    for _ in range(4):
        coef = np.polyfit(x[keep], y[keep], 1, w=weight[keep])
        resid = y - (coef[0] * x + coef[1])
        good_resid = resid[keep]
        mad = float(np.median(np.abs(good_resid - np.median(good_resid))))
        tol = max(3.0, 3.5 * mad)
        keep = np.abs(resid) <= tol
        if np.count_nonzero(keep) < 50:
            break

    kept = int(np.count_nonzero(keep))
    coverage = float((np.max(x[keep]) - np.min(x[keep])) / max(w, 1)) if kept else 0.0
    diag.update({'kept': kept, 'mad': mad, 'coverage': coverage})
    if kept < 80 or coverage < 0.45:
        return None, diag

    coef = np.polyfit(x[keep], y[keep], 1, w=weight[keep])
    angle = float(np.degrees(np.arctan(coef[0])))
    diag['raw_angle'] = angle
    if abs(angle) < 0.05 or abs(angle) > 0.85 or mad > 4.0:
        return None, diag
    return angle, diag


def orient_caliper(roi_color: np.ndarray,
                    roi_gray: np.ndarray,
                    roi_binary: np.ndarray = None) -> dict:
    """
    检测刻度线主导方向，旋转图像使刻线垂直。

    Returns:
        dict: rotated_color, rotated_gray, rotated_binary, orient_angle, orient_vis
    """
    gray = roi_gray if roi_gray is not None else cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    angle, orient_diag = _estimate_orient_angle_from_tick_pixels(gray, roi_binary)
    if angle is None:
        angle = _estimate_orient_angle_from_hough(gray)
        orient_diag = {'method': 'hough'}
    else:
        orient_diag['method'] = 'tick_pixels'

    coarse_angle = float(angle)
    rotated_color = rotate_image(roi_color, coarse_angle)
    rotated_gray = rotate_image(gray, coarse_angle)
    rotated_binary = rotate_image(roi_binary, coarse_angle) if roi_binary is not None else None

    fine_angle, seam_diag = _estimate_seam_angle_from_gradient(rotated_gray, rotated_binary)
    if fine_angle is not None:
        angle = coarse_angle + float(fine_angle)
        orient_diag['method'] = f"{orient_diag.get('method', 'unknown')}+seam"
    else:
        fine_angle = 0.0
        angle = coarse_angle

    rotated_color = rotate_image(roi_color, angle)
    rotated_gray = rotate_image(gray, angle)
    rotated_binary = rotate_image(roi_binary, angle) if roi_binary is not None else None

    orient_diag['coarse_tick_angle'] = coarse_angle
    orient_diag['fine_seam_angle'] = float(fine_angle)
    orient_diag['final_angle'] = float(angle)
    orient_diag['seam'] = seam_diag
    orient_vis = _make_orient_vis(roi_color, rotated_color, angle)

    return {
        'rotated_color': rotated_color,
        'rotated_gray': rotated_gray,
        'rotated_binary': rotated_binary,
        'orient_angle': angle,
        'orient_diag': orient_diag,
        'orient_vis': orient_vis,
    }

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
