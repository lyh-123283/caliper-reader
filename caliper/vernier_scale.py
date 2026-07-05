"""
步骤 4 — 游标尺识别（刻度线检测 + 固定 0.02mm 精度 + 对齐查找）

流程:
  1. 垂直投影 → 游标刻度线检测
  2. 固定使用 0.02mm 精度
  3. 零线定位（最左侧刻线）
  4. 网格法找最佳对齐线 → 小数读数
"""

import cv2
import numpy as np
from typing import List, Tuple

from .utils import (
    find_peaks_adaptive, extract_ticks_from_binary, draw_projection_plot,
    refine_ticks_by_spacing, extract_ticks_from_anchor_band,
)
from .config import config




# ═══════════════════════════ 对齐查找 v2 ═══════════════════════════

def _verify_zero_by_digit(region: dict, zero_x: float) -> bool:
    """
    验证候选零线位置附近是否存在数字 "0"。

    物理依据：游标卡尺的 0 刻度线向下应穿过数字 "0"（印在游标尺面板下方）。
    在候选零线的 x 位置附近截取游标尺下半部分数字区域，跑 OCR 看是否
    能识别出 "0"。
    """
    img = region['image']
    h, w = img.shape

    # 数字区域：游标尺数字印在刻线下方（靠近游标尺底边），取下半部分
    y_start = int(h * config.vernier_scale.zero_digit_search_ratio)

    # 以 zero_x 为中心取搜索窗口
    half_w = max(config.vernier_scale.zero_digit_half_w_min,
                 int(w * config.vernier_scale.zero_digit_half_w_ratio))
    x1 = max(0, int(zero_x) - half_w)
    x2 = min(w, int(zero_x) + half_w)

    if x2 - x1 < 10:
        return False

    patch = img[y_start:, x1:x2]
    if patch.size == 0:
        return False

    # 增强处理：放大 + CLAHE + 二值化
    patch = cv2.resize(patch, (patch.shape[1] * 2, patch.shape[0] * 2),
                       interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=config.vernier_scale.zero_digit_clahe_clip, tileGridSize=(4, 4))
    patch = clahe.apply(patch)
    _, patch_bin = cv2.threshold(patch, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 用现有 OCR 引擎识别
    from .ocr import get_ocr_reader_singleton
    reader = get_ocr_reader_singleton()
    reader._ensure_engine()
    results = reader._ocr_single_patch(patch_bin)

    for text, conf in results:
        if text == '0' and conf > config.vernier_scale.zero_digit_conf_min:
            return True
    return False


def _find_zero_tick(vernier_ticks: List[dict], region: dict = None):
    """
    从游标刻线列表中定位零线（物理上=最左侧那条有效刻线）。

    v6 改进：
      1) 长度过滤 — 候选 tick 长度需 ≥ 中位数 × factor
      2) 位置约束 — zero_x 必须 ≥ ROI 宽度 × 5%（避免选到 ROI 最左缘伪影）
      3) 面板检测 — 候选 tick 下方 30% 区域灰度均值需偏暗（游标尺金属面板）
      4) 数字验证 — 零线向下应穿过数字 "0"；OCR 确认则增强可信度

    优先级：满足 1+2+3+4 → 满足 1+2+3 → 兜底回退首条

    Returns:
        (zero_tick, zero_digit_found)
    """
    if not vernier_ticks:
        return None, False
    v_sorted = sorted(vernier_ticks, key=lambda t: t['x'])
    lengths = [t.get('length', 0) for t in v_sorted]
    median_len = float(np.median(lengths)) if lengths else 0

    # ── 位置下限：游标尺零线一定不会贴 ROI 最左边 ──
    # v6: 提高到 17% 防止主尺残余刻线/DELIXI Logo 区域被当成零线
    img_w = region['image'].shape[1] if region is not None and 'image' in region else 0
    min_x = max(80, int(img_w * 0.17))

    # ── 第一轮：满足长度 + 位置 + 面板 + 数字 0 验证 ──
    if region is not None:
        for t in v_sorted:
            if t['x'] < min_x:
                continue
            length = t.get('length', 0)
            if length < median_len * config.vernier_scale.zero_length_factor and len(v_sorted) > 3:
                continue
            if not _has_dark_panel_below(region, t['x']):
                continue
            if _verify_zero_by_digit(region, t['x']):
                return t, True

    # ── 第二轮：满足长度 + 位置 + 面板（无 OCR 验证）──
    if region is not None:
        for t in v_sorted:
            if t['x'] < min_x:
                continue
            length = t.get('length', 0)
            if length < median_len * config.vernier_scale.zero_length_factor and len(v_sorted) > 3:
                continue
            if _has_dark_panel_below(region, t['x']):
                return t, False

    # ── 第三轮：放宽到只看长度 + 位置 ──
    for t in v_sorted:
        if t['x'] < min_x:
            continue
        length = t.get('length', 0)
        if length >= median_len * config.vernier_scale.zero_length_factor or len(v_sorted) <= 3:
            return t, False

    # ── 兜底 ──
    return v_sorted[0], False


def _has_dark_panel_below(region: dict, x_pos: float) -> bool:
    """
    判断候选零线 x 位置下方是否为游标尺金属面板（深色连续区）。

    物理依据：真正的游标尺零线下方紧邻游标尺金属体（深色块），
    而主尺溢出的伪刻线下方通常是亮金属面或背景。

    采样：在 [x-3, x+3] × [h*0.30, h*0.70] 范围取灰度均值；
    判定：均值 < 全 region 灰度均值 × 0.85 → 是面板。
    """
    if region is None or 'image' not in region:
        return True  # 无 region 信息时不阻塞
    img = region['image']
    h, w = img.shape[:2]
    x = int(round(x_pos))
    if x < 3 or x >= w - 3:
        return False

    y1 = max(0, int(h * 0.30))
    y2 = min(h - 1, int(h * 0.70))
    if y2 <= y1:
        return True
    patch = img[y1:y2 + 1, max(0, x - 3):min(w, x + 4)]
    if patch.size == 0:
        return False

    patch_mean = float(np.mean(patch))
    global_mean = float(np.mean(img))
    return patch_mean < global_mean * 0.85

def find_best_alignment(vernier_ticks: List[dict],
                         precision: float,
                         main_ticks: List[dict]) -> Tuple[float, dict, float]:
    """
    找到与主尺刻度线最对齐的游标刻线（游标读数）。

    物理原理:
      游标尺有 N 条等距刻线，对应主尺 (N-1) 条刻线跨度。
      游标第 i 条线与主尺某条线对齐时，小数读数 = i * precision。

    Args:
        vernier_ticks: 游标刻线列表
        precision:     卡尺精度，当前固定为 0.02
        main_ticks:    主尺刻线列表

    Returns:
        (vernier_reading, aligned_tick, confidence)
    """
    v_sorted = sorted(vernier_ticks, key=lambda t: t['x'])
    n_all = len(v_sorted)
    expected_lines = int(round(1.0 / precision)) + 1 if precision and precision > 0 else n_all
    n = min(n_all, max(2, expected_lines))
    v_sorted = v_sorted[:n]
    if n < 2:
        return 0.0, None, 0.0

    # 计算游标区域 y 范围（用于 Y 方向过滤）
    vy_range = None
    if v_sorted:
        vy_all_start = min(t.get('y_start', 0) for t in v_sorted)
        vy_all_end = max(t.get('y_end', 0) for t in v_sorted)
        vy_range = (vy_all_start, vy_all_end)

    # 对每条游标刻线，计算与最近主尺刻线的像素误差
    errors = np.zeros(n)
    for i, vt in enumerate(v_sorted):
        errors[i] = _compute_alignment_error(vt['x'], main_ticks, None)

    # ── 找误差最小的游标线 ──
    best_idx = int(np.argmin(errors))

    # ── 亚像素抛物线插值 ──
    # 在 best_idx 附近拟合抛物线，得到更精确的零点
    sub_idx = float(best_idx)
    if 0 < best_idx < n - 1:
        e0, e1, e2 = errors[best_idx - 1], errors[best_idx], errors[best_idx + 1]
        denom = e0 - 2 * e1 + e2
        if abs(denom) > 1e-9:
            sub_idx = best_idx + 0.5 * (e0 - e2) / denom
    sub_idx = max(0.0, min(float(n - 1), sub_idx))

    vernier_reading = round(sub_idx * precision, 2)

    # ── 置信度评分 ──
    confidence = _alignment_confidence(errors, best_idx, n)

    return vernier_reading, v_sorted[best_idx], confidence


def _compute_alignment_error(vx: float,
                             main_ticks: List[dict],
                             vernier_region_y_range: tuple = None) -> float:
    """计算游标线 vx 与最近主尺刻度线的像素距离。只在 y 方向有重叠的线对之间计算。"""
    best = float('inf')
    for mt in main_ticks:
        # Y 方向过滤：游标刻线必须与主尺刻线有垂直重叠才可能物理对齐
        if vernier_region_y_range is not None:
            vy_min, vy_max = vernier_region_y_range
            mt_ymin = mt.get('y_start', 0)
            mt_ymax = mt.get('y_end', 0)
            # 两条线 y 区间有交集才算
            if mt_ymax < vy_min or mt_ymin > vy_max:
                continue
        d = abs(vx - mt['x'])
        if d < best:
            best = d
    # 若无重叠刻线，回退到不过滤
    if best == float('inf'):
        for mt in main_ticks:
            d = abs(vx - mt['x'])
            if d < best:
                best = d
    return best


def _alignment_confidence(errors: np.ndarray,
                           best_idx: int, n: int) -> float:
    """
    评估对齐结果的置信度。

    好的对齐 = 最小误差显著低于邻居（尖锐谷）。
    差的   = 多条线误差相近（平底谷 = 模糊）。

    Returns: 0~1
    """
    best_err = errors[best_idx]
    if best_err <= 0.5:
        return config.vernier_scale.align_conf_perfect  # 几乎完美对齐

    # 检查邻居
    neighbor_errs = []
    for offset in [-2, -1, 1, 2]:
        ni = best_idx + offset
        if 0 <= ni < n:
            neighbor_errs.append(errors[ni])

    if not neighbor_errs:
        return 0.5

    median_neighbor = float(np.median(neighbor_errs))
    if median_neighbor < 0.5:
        return 0.5

    # 信号比值：邻居误差 / 最优误差，越大说明最优越突出
    ratio = median_neighbor / max(best_err, 0.5)
    if ratio >= 3.0:
        return config.vernier_scale.align_conf_strong
    elif ratio >= 2.0:
        return config.vernier_scale.align_conf_moderate
    elif ratio >= 1.5:
        return config.vernier_scale.align_conf_weak
    return config.vernier_scale.align_conf_bad


# ═══════════════════════════ 可视化 ═══════════════════════════

def _draw_vernier_ticks(region: dict,
                         binary: np.ndarray,
                         vernier_ticks: List[dict],
                         vproj: np.ndarray,
                         peaks: np.ndarray,
                         zero_x: float = 0,
                         zero_digit_found: bool = False,
                         band_detection: dict = None) -> np.ndarray:
    if band_detection:
        return _draw_vernier_ticks_on_band(
            region, vernier_ticks, zero_x, zero_digit_found, band_detection
        )

    """绘制游标尺刻度线检测 — 灰度底图 + 右下角二值图小窗"""
    img = region['image']
    h, w = img.shape

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for t in vernier_ticks:
        is_long = t.get('is_long', False)
        color = (255, 200, 50) if is_long else (200, 160, 40)
        thickness = 3 if is_long else 2
        cv2.line(vis, (t['x'], t['y_start']), (t['x'], t['y_end']), color, thickness)
        if is_long:
            cv2.circle(vis, (t['x'], t['y_mid']), 4, (255, 255, 100), -1)

    # 零线高亮
    zx = int(zero_x) if zero_x > 0 else vernier_ticks[0]['x']
    cv2.line(vis, (zx, 0), (zx, h - 1), (50, 150, 255), 3)
    cv2.putText(vis, "ZERO", (zx + 4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 150, 255), 1)

    # ── 零线数字0验证状态提示 ──
    if zero_digit_found:
        status_text = "Zero Digit [0]: FOUND (OCR verified)"
        status_color = (0, 220, 100)
    else:
        status_text = "Zero Digit [0]: NOT FOUND (fallback by position)"
        status_color = (100, 180, 255)
    cv2.putText(vis, status_text, (5, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

    # ── 右下角二值图小窗（显示检测器实际看到的图像）──
    bnw = max(50, w // 4)
    bnh = int(h * bnw / w)
    bthumb = cv2.resize(binary, (bnw, bnh), interpolation=cv2.INTER_AREA)
    bthumb_3 = cv2.cvtColor(bthumb, cv2.COLOR_GRAY2BGR)
    bx2, by2 = w - bnw, h - bnh
    vis[by2:by2 + bnh, bx2:bx2 + bnw] = bthumb_3
    cv2.rectangle(vis, (bx2, by2), (bx2 + bnw, by2 + bnh), (255, 255, 255), 1)
    cv2.putText(vis, "BIN", (bx2 + 3, by2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # 投影图
    proj_vis = draw_projection_plot(vproj, peaks, width=w,
                                     title=f"Vertical Projection ({len(vernier_ticks)} ticks)")
    ph = proj_vis.shape[0]

    gap = 2
    out = np.zeros((h + ph + gap, w, 3), dtype=np.uint8)
    out[:] = (30, 30, 35)
    out[:h, :w] = vis
    out[h + gap:h + gap + ph, :w] = proj_vis

    cv2.putText(out, "STEP 4: Vernier Scale Ticks (gray + binary overlay)", (5, out.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 125), 1)

    return out


def _draw_vernier_ticks_on_band(region: dict,
                                vernier_ticks: List[dict],
                                zero_x: float,
                                zero_digit_found: bool,
                                band_detection: dict) -> np.ndarray:
    """Draw vernier tick labels on the same narrow band used for detection."""
    img = region['image']
    x1 = int(band_detection['x1'])
    x2 = int(band_detection['x2'])
    y1 = int(band_detection['band_y1'])
    y2 = int(band_detection['band_y2'])
    gray_band = img[y1:y2, x1:x2]
    if gray_band.size == 0:
        return np.zeros((100, 300, 3), dtype=np.uint8)

    band_h, band_w = gray_band.shape[:2]
    scale_y = max(2, min(4, int(np.ceil(150 / max(1, band_h)))))
    disp_h = band_h * scale_y
    vis = cv2.cvtColor(
        cv2.resize(gray_band, (band_w, disp_h), interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_GRAY2BGR
    )

    face_left = int(band_detection.get('face_left_x', 0))
    if 0 < face_left < band_w:
        cv2.line(vis, (face_left, 0), (face_left, disp_h - 1),
                 (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(vis, "EDGE", (face_left + 3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    for t in vernier_ticks:
        lx = int(round(t['x'] - x1))
        if not (0 <= lx < band_w):
            continue
        is_zero = abs(float(t['x']) - float(zero_x)) <= 3.0
        is_long = t.get('is_long', False)
        color = (50, 150, 255) if is_zero else ((255, 220, 80) if is_long else (80, 230, 255))
        thickness = 3 if is_zero or is_long else 2
        cv2.line(vis, (lx, 0), (lx, disp_h - 1), color, thickness, cv2.LINE_AA)
        if is_long:
            cv2.circle(vis, (lx, disp_h // 2), 4, (255, 255, 120), -1)

    zx = int(round(zero_x - x1))
    if 0 <= zx < band_w:
        cv2.line(vis, (zx, 0), (zx, disp_h - 1), (50, 150, 255), 3, cv2.LINE_AA)
        cv2.putText(vis, "ZERO", (min(zx + 4, band_w - 60), 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 150, 255), 2)

    status_text = "Zero Digit [0]: FOUND" if zero_digit_found else "Zero from band sequence"
    cv2.putText(vis, status_text, (5, disp_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 180, 255), 1)
    cv2.putText(vis, "STEP 4: Vernier ticks on detected narrow band", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    plot_peaks = np.array(band_detection.get('tick_xs_local', []), dtype=int)
    proj_vis = draw_projection_plot(
        band_detection.get('proj_norm', np.array([])),
        plot_peaks,
        width=band_w,
        title=f"Narrow-band projection ({len(vernier_ticks)} ticks)"
    )
    gap = 2
    out = np.zeros((disp_h + proj_vis.shape[0] + gap, band_w, 3), dtype=np.uint8)
    out[:] = (30, 30, 35)
    out[:disp_h, :band_w] = vis
    out[disp_h + gap:disp_h + gap + proj_vis.shape[0], :band_w] = proj_vis
    return out


def _draw_alignment(region: dict,
                     color_region: np.ndarray,
                     vernier_ticks: List[dict],
                     main_gap: float,
                     zero_x: float,
                     aligned_tick: dict,
                     align_conf: float = 0.0,
                     full_color: np.ndarray = None,
                     split_y: int = 0,
                     main_ticks: List[dict] = None) -> np.ndarray:
    """绘制对齐检测可视化

    背景图优先级: full_color (整张 ROI + 主尺网格) > color_region > 灰度
    """
    y_off = region.get('y_offset', 0)

    if full_color is not None:
        vis = full_color.copy()
        use_full = True
    elif color_region is not None:
        vis = color_region.copy()
        use_full = False
    else:
        vis = cv2.cvtColor(region['image'], cv2.COLOR_GRAY2BGR)
        use_full = False

    h, w = vis.shape[:2]

    # ── 分割线（全图模式下画出主尺/游标分界）──
    if use_full and split_y > 0:
        cv2.line(vis, (0, split_y), (w, split_y), (255, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(vis, "MAIN SCALE", (10, split_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)
        cv2.putText(vis, "VERNIER", (10, split_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    # ── 主尺真实网格线（全图模式下画在主尺半区）──
    if use_full and main_ticks and split_y > 0:
        for mt in main_ticks:
            mx = mt['x']
            my1 = mt.get('y_start', 0)
            my2 = min(mt.get('y_end', split_y), split_y)
            if 0 <= mx < w and my1 < my2:
                cv2.line(vis, (mx, my1), (mx, my2), (80, 80, 90), 1, cv2.LINE_AA)
    elif main_gap > 0:
        # 回退：合成网格线
        grid_offset = zero_x % main_gap
        for gx in np.arange(grid_offset, w, main_gap):
            gx = int(gx)
            if 0 <= gx < w:
                cv2.line(vis, (gx, 0), (gx, h), (80, 80, 90), 1, cv2.LINE_AA)

    # 游标刻线（全图模式下加上 y_offset = split_y）
    for i, t in enumerate(vernier_ticks):
        dy = y_off if use_full else 0
        pt1 = (t['x'], t['y_start'] + dy)
        pt2 = (t['x'], t['y_end'] + dy)
        cv2.line(vis, pt1, pt2, (200, 160, 40), 1)

    # 零线（亮蓝粗线贯穿全图 + 标注 x 坐标）
    zx = int(zero_x)
    cv2.line(vis, (zx, 0), (zx, h - 1), (50, 150, 255), 3)
    cv2.putText(vis, f"ZERO (x={zx})", (zx + 4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 150, 255), 2)
    if use_full:
        cv2.putText(vis, f"x={zx}", (zx + 4, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 150, 255), 1)

    # 对齐线高亮（全图模式下加上 y_offset，标注序号）
    if aligned_tick:
        ax = aligned_tick['x']
        dy = y_off if use_full else 0
        ay1 = aligned_tick['y_start'] + dy
        ay2 = aligned_tick['y_end'] + dy
        aym = aligned_tick['y_mid'] + dy
        cv2.line(vis, (ax, ay1), (ax, ay2), (0, 255, 80), 3)
        cv2.circle(vis, (ax, aym), 8, (0, 255, 80), 2)
        # 计算对齐线是第几条（序号）
        v_sorted = sorted(vernier_ticks, key=lambda t: t['x'])
        aligned_idx = next((i for i, t in enumerate(v_sorted) if t['x'] == ax), -1)
        label = f"ALIGNED! tick#{aligned_idx}" if aligned_idx >= 0 else "ALIGNED!"
        cv2.putText(vis, label, (ax + 5, aym),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 2)

    # 对齐置信度
    if align_conf > 0:
        cc = (0, 255, 100) if align_conf > 0.7 else (255, 200, 50) if align_conf > 0.4 else (255, 120, 120)
        cv2.putText(vis, f"conf: {align_conf:.2f}", (w - 160, vis.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, cc, 1)

    cv2.putText(vis, "STEP 4: Vernier Alignment v2", (5, vis.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 125), 1)

    return vis


def _empty_vernier_result() -> dict:
    empty_img = np.zeros((100, 300, 3), dtype=np.uint8)
    return {
        'vernier_ticks': [], 'precision': 0.02, 'vernier_reading': 0.0,
        'zero_x': 0.0, 'aligned_tick': None,
        'vis_ticks': empty_img, 'vis_alignment': empty_img, 'vis_digits': empty_img,
        'vproj_norm': None, 'vernier_peaks': None, 'zero_digit_found': False,
    }


def _find_vernier_body_x_range(img: np.ndarray) -> tuple:
    """Find the bright vernier body strip so main-scale remnants are ignored."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return 0, w

    y2 = max(12, min(h, int(h * 0.28)))
    band = img[:y2, :]
    col_mean = np.mean(band, axis=0).astype(float)
    win = max(15, w // 120)
    if win % 2 == 0:
        win += 1
    smooth = np.convolve(col_mean, np.ones(win, dtype=float) / win, mode='same')
    grad = np.gradient(smooth)

    left_lo, left_hi = int(w * 0.25), int(w * 0.45)
    if left_hi <= left_lo:
        return int(w * 0.30), int(w * 0.70)

    left_slice = grad[left_lo:left_hi]
    if left_slice.size == 0:
        return int(w * 0.30), int(w * 0.70)
    left_x = left_lo + int(np.argmax(left_slice))
    if grad[left_x] < 1.2:
        left_x = int(w * 0.34)

    right_lo, right_hi = int(w * 0.60), int(w * 0.76)
    right_x = int(w * 0.69)
    if right_hi > right_lo:
        right_slice = grad[right_lo:right_hi]
        if right_slice.size:
            candidate = right_lo + int(np.argmin(right_slice))
            if grad[candidate] < -1.0:
                right_x = candidate

    min_width = int(w * 0.24)
    max_width = int(w * 0.42)
    if right_x - left_x < min_width:
        right_x = left_x + min_width
    if right_x - left_x > max_width:
        right_x = left_x + max_width

    pad = max(6, int(w * 0.006))
    return max(0, left_x - pad), min(w, right_x + pad)


def _filter_vernier_ticks_by_grid(vernier_ticks: List[dict], main_gap: float) -> List[dict]:
    """Keep the longest stable vernier grid for fixed 0.02 mm calipers."""
    if not vernier_ticks or len(vernier_ticks) < 4 or main_gap <= 0:
        return vernier_ticks

    expected = main_gap * 0.98
    if expected < 3:
        return vernier_ticks

    ticks = sorted(vernier_ticks, key=lambda t: t['x'])

    deduped = []
    dup_tol = max(3.0, expected * 0.60)
    for t in ticks:
        if deduped and t['x'] - deduped[-1]['x'] < dup_tol:
            if _tick_grid_priority(t) > _tick_grid_priority(deduped[-1]):
                deduped[-1] = t
        else:
            deduped.append(t)

    if len(deduped) < 4:
        return vernier_ticks

    best = []
    tol = max(3.0, expected * 0.35)
    for start_idx, start_tick in enumerate(deduped):
        seq = [start_tick]
        current_x = float(start_tick['x'])
        for _ in range(1, 60):
            target = current_x + expected
            candidates = [t for t in deduped
                          if target - tol <= t['x'] <= target + tol
                          and t['x'] > current_x + expected * 0.45]
            if not candidates:
                break
            nxt = min(candidates, key=lambda t: (
                abs(t['x'] - target),
                -_tick_grid_priority(t)[0],
                -_tick_grid_priority(t)[1],
            ))
            if nxt in seq:
                break
            seq.append(nxt)
            current_x = float(nxt['x'])

        if len(seq) > len(best):
            best = seq

    median_len = float(np.median([t.get('length', 0) for t in deduped])) if deduped else 0.0
    min_start_len = median_len * 0.70
    min_initial = 5
    min_total = max(12, config.vernier_scale.min_tick_count)
    for start_tick in deduped:
        if start_tick.get('length', 0) < min_start_len:
            continue
        seq = _collect_vernier_grid_sequence(deduped, start_tick, expected, tol)
        if len(seq) < min_total:
            continue
        if _initial_grid_streak(seq, expected, tol) >= min_initial:
            return sorted(seq, key=lambda t: t['x'])

    if len(best) >= max(6, config.vernier_scale.min_tick_count):
        return sorted(best, key=lambda t: t['x'])
    return vernier_ticks


def _collect_vernier_grid_sequence(ticks: List[dict],
                                   start_tick: dict,
                                   expected: float,
                                   tol: float) -> List[dict]:
    seq = [start_tick]
    used = {id(start_tick)}
    start_x = float(start_tick['x'])
    prev_x = start_x
    for k in range(1, 60):
        target = start_x + expected * k
        candidates = [
            t for t in ticks
            if id(t) not in used
            and abs(float(t['x']) - target) <= tol
            and float(t['x']) > prev_x + expected * 0.35
        ]
        if not candidates:
            continue
        nxt = min(candidates, key=lambda t: (
            abs(float(t['x']) - target),
            -_tick_grid_priority(t)[0],
            -_tick_grid_priority(t)[1],
        ))
        seq.append(nxt)
        used.add(id(nxt))
        prev_x = float(nxt['x'])
    return sorted(seq, key=lambda t: t['x'])


def _initial_grid_streak(seq: List[dict], expected: float, tol: float) -> int:
    if not seq:
        return 0
    xs = [float(t['x']) for t in sorted(seq, key=lambda item: item['x'])]
    start_x = xs[0]
    streak = 0
    for k in range(0, 20):
        target = start_x + expected * k
        if any(abs(x - target) <= tol for x in xs):
            streak += 1
        else:
            break
    return streak


def _sync_band_detection_ticks(band_detection: dict, vernier_ticks: List[dict]) -> dict:
    if not band_detection or not vernier_ticks:
        return band_detection

    x1 = int(band_detection.get('x1', 0))
    band = band_detection.get('band')
    band_w = band.shape[1] if band is not None and band.size else None
    local_xs = []
    global_xs = []
    for t in sorted(vernier_ticks, key=lambda item: item['x']):
        gx = int(round(t['x']))
        lx = gx - x1
        if band_w is not None and (lx < 0 or lx >= band_w):
            continue
        local_xs.append(lx)
        global_xs.append(gx)

    if len(local_xs) >= config.vernier_scale.min_tick_count:
        band_detection['tick_xs_local'] = local_xs
        band_detection['tick_xs_global'] = global_xs
    return band_detection


def _tick_grid_priority(tick: dict) -> tuple:
    source_score = {
        'band_projection_refined': 3,
        'anchor_band': 2,
        'projection': 1,
    }.get(tick.get('source', 'projection'), 1)
    return source_score, tick.get('length', 0)


def _map_tick_to_original(tick: dict) -> dict:
    """Map a tick detected on the vernier body crop back to split ROI coords."""
    x_offset = int(tick.get('_x_offset', 0))
    mapped = dict(tick)
    mapped['x_local'] = int(round(tick['x']))
    mapped['x'] = int(round(tick['x'])) + x_offset
    return mapped


def _map_ticks_to_original(ticks: List[dict], region: dict) -> List[dict]:
    x_offset = int(region.get('x_offset', 0)) if region else 0
    ticks_with_offset = []
    for t in ticks:
        tt = dict(t)
        tt['_x_offset'] = x_offset
        ticks_with_offset.append(tt)
    return sorted([_map_tick_to_original(t) for t in ticks_with_offset], key=lambda t: t['x'])


def _map_x_to_original(x: float, region: dict, y: float = None) -> float:
    x_offset = int(region.get('x_offset', 0)) if region else 0
    return float(x + x_offset)


def _contiguous_true_segments(mask: np.ndarray, min_len: int = 1) -> List[Tuple[int, int]]:
    """Return [start, end) segments for a 1-D boolean mask."""
    segments = []
    start = None
    for i, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_len:
                segments.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segments.append((start, len(mask)))
    return segments


def _find_vernier_tick_band(binary: np.ndarray, x1: int, x2: int) -> Tuple[int, int]:
    """Find the narrow row band containing downward vernier tick strokes."""
    h, w = binary.shape[:2]
    if h <= 0 or w <= 0:
        return 0, 0

    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    search_h = max(18, min(h, int(h * 0.62)))
    crop = binary[:search_h, x1:x2]
    if crop.size == 0:
        fallback_h = max(12, min(h, int(h * 0.30)))
        return 0, fallback_h

    kernel_h = max(5, min(21, search_h // 10))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    vertical = cv2.morphologyEx(crop, cv2.MORPH_OPEN, vertical_kernel)
    if np.count_nonzero(vertical) < max(20, crop.size * 0.0005):
        vertical = crop

    row_score = np.mean(vertical > 0, axis=1).astype(float)
    if np.max(row_score) <= 0:
        fallback_h = max(12, min(h, int(h * 0.30)))
        return 0, fallback_h

    win = max(3, min(15, search_h // 25))
    if win % 2 == 0:
        win += 1
    smooth = np.convolve(row_score, np.ones(win, dtype=float) / win, mode='same')
    th = max(float(np.median(smooth) + 0.45 * np.std(smooth)),
             float(np.max(smooth) * 0.22))
    segments = _contiguous_true_segments(smooth >= th,
                                         min_len=max(5, min(18, search_h // 18)))
    if not segments:
        fallback_h = max(12, min(h, int(h * 0.30)))
        return 0, fallback_h

    max_mean = max(float(np.mean(smooth[s:e])) for s, e in segments)
    strong = [(s, e) for s, e in segments
              if float(np.mean(smooth[s:e])) >= max_mean * 0.55]
    top_limit = int(search_h * 0.45)
    near_top = [(s, e) for s, e in strong if s <= top_limit]
    selected = min(near_top or strong, key=lambda seg: (seg[0], -(seg[1] - seg[0])))

    pad = max(3, min(8, kernel_h // 2))
    y1 = max(0, selected[0] - pad)
    y2 = min(search_h, selected[1] + pad)
    min_band_h = max(12, min(36, int(h * 0.12)))
    if y2 - y1 < min_band_h:
        extra = min_band_h - (y2 - y1)
        y1 = max(0, y1 - extra // 2)
        y2 = min(search_h, y2 + extra - extra // 2)
    return y1, max(y1 + 1, y2)


def _dedupe_tick_xs(xs: List[int], tol: float) -> List[int]:
    if not xs:
        return []
    groups = []
    for x in sorted(int(v) for v in xs):
        if not groups or x - groups[-1][-1] > tol:
            groups.append([x])
        else:
            groups[-1].append(x)
    return [int(round(float(np.median(g)))) for g in groups]


def _estimate_vernier_tick_gap(tick_xs: List[int], main_gap: float) -> float:
    if main_gap and main_gap > 3:
        return float(main_gap) * 0.98
    if len(tick_xs) < 3:
        return 0.0
    diffs = np.diff(sorted(tick_xs))
    diffs = diffs[diffs >= 3]
    if len(diffs) == 0:
        return 0.0
    lo, hi = np.percentile(diffs, [20, 80])
    core = diffs[(diffs >= lo) & (diffs <= hi)]
    return float(np.median(core if len(core) else diffs))


def _find_vernier_face_left_edge(gray: np.ndarray,
                                 band_y1: int,
                                 band_y2: int,
                                 x1: int,
                                 x2: int,
                                 expected_gap: float) -> int:
    """Find the left edge of the vernier face inside the crop.

    The face edge is visually strong and can look like a tick in projection,
    so zero detection must skip peaks that sit on this edge.
    """
    if gray is None or gray.size == 0 or expected_gap <= 0:
        return 0

    h, w = gray.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    band_y1 = max(0, min(h - 1, int(band_y1)))
    band_y2 = max(band_y1 + 1, min(h, int(band_y2)))
    crop = gray[band_y1:band_y2, x1:x2]
    if crop.size == 0 or crop.shape[1] < 8:
        return 0

    search_w = min(crop.shape[1], max(80, int(expected_gap * 5.0)))
    col = np.mean(crop[:, :search_w], axis=0).astype(float)
    win = max(5, min(21, int(expected_gap // 2) * 2 + 1))
    if win >= len(col):
        win = max(3, len(col) | 1)
    smooth = np.convolve(col, np.ones(win, dtype=float) / win, mode='same')
    grad = np.diff(smooth)
    if len(grad) == 0:
        return 0

    limit = min(len(grad), max(30, int(expected_gap * 3.0)))
    idx = int(np.argmax(grad[:limit])) + 1
    strength = float(grad[idx - 1])
    noise = float(np.std(grad[:limit])) if limit > 3 else 0.0
    if strength < max(2.0, noise * 1.2):
        return 0
    return idx


def _has_top_connected_vernier_stroke(band: np.ndarray, x: int) -> bool:
    """True ticks are connected downward from the seam-side top of the band."""
    if band is None or band.size == 0:
        return False
    h, w = band.shape[:2]
    if h <= 0 or w <= 0:
        return False
    x = int(x)
    if x < 0 or x >= w:
        return False

    strip = band[:, max(0, x - 2):min(w, x + 3)] > 0
    if strip.size == 0:
        return False
    row_on = np.mean(strip, axis=1) > 0.18
    if not np.any(row_on):
        return False

    first = int(np.argmax(row_on))
    run = 0
    for value in row_on[first:]:
        if value:
            run += 1
        elif run >= 3:
            break
        else:
            run = 0

    top_h = min(10, h)
    top_density = float(np.mean(strip[:top_h])) if top_h > 0 else 0.0
    min_run = max(7, int(h * 0.13))
    return first <= max(2, int(h * 0.08)) and run >= min_run and top_density >= 0.12


def _tick_xs_from_projection_segments(proj_norm: np.ndarray,
                                      smooth: np.ndarray,
                                      h_th: float,
                                      dedupe_tol: float) -> List[int]:
    """Find tick x positions from above-threshold peak segments.

    Smooth projection is good for thresholding, but exact x localization must
    use the raw projection so flat/plateau peaks do not disappear.
    """
    if proj_norm is None or smooth is None or len(proj_norm) == 0:
        return []

    mask = smooth >= h_th
    segments = _contiguous_true_segments(mask, min_len=1)
    xs = []
    for s, e in segments:
        raw = proj_norm[s:e]
        if raw.size == 0 or float(np.max(raw)) <= 0:
            continue

        raw_th = max(0.18, float(np.max(raw)) * 0.45)
        raw_segments = _contiguous_true_segments(raw >= raw_th, min_len=1)
        if not raw_segments:
            raw_segments = [(0, len(raw))]

        for rs, re in raw_segments:
            local = raw[rs:re]
            if local.size == 0:
                continue
            local_xs = np.arange(rs, re, dtype=float)
            weight_sum = float(np.sum(local))
            if weight_sum > 1e-6:
                x = s + int(round(float(np.sum(local_xs * local) / weight_sum)))
            else:
                x = s + int(round((rs + re - 1) / 2.0))
            xs.append(x)

    return _dedupe_tick_xs(xs, dedupe_tol)


def _detect_vernier_band_projection(binary: np.ndarray,
                                    body_x1: int,
                                    body_x2: int,
                                    main_gap: float,
                                    gray: np.ndarray = None) -> dict:
    """Detect vernier tick x positions once from the narrow band below seam."""
    if binary is None or binary.size == 0:
        return None

    h, w = binary.shape[:2]
    x1 = max(0, min(w - 1, int(body_x1)))
    x2 = max(x1 + 1, min(w, int(body_x2)))
    band_y1, band_y2 = _find_vernier_tick_band(binary, x1, x2)
    band = binary[band_y1:band_y2, x1:x2]
    if band.size == 0:
        return None

    proj = np.sum(band > 0, axis=0).astype(float)
    if np.max(proj) <= 0:
        return None

    proj_norm = proj / np.max(proj)
    win = max(3, min(11, len(proj_norm) // 80))
    if win % 2 == 0:
        win += 1
    smooth = np.convolve(proj_norm, np.ones(win, dtype=float) / win, mode='same')
    n = len(smooth)
    min_dist = max(3, n // 200)

    peaks = []
    valleys = []
    for i in range(min_dist, n - min_dist):
        left = smooth[i - min_dist:i]
        right = smooth[i + 1:i + min_dist + 1]
        if smooth[i] > max(float(np.max(left)), float(np.max(right))):
            peaks.append((i, float(smooth[i])))
        if smooth[i] < min(float(np.min(left)), float(np.min(right))):
            valleys.append((i, float(smooth[i])))

    if len(peaks) < 3:
        return None

    peak_vals = sorted([v for _, v in peaks], reverse=True)
    top80_n = max(1, int(len(peak_vals) * 0.8))
    A = float(np.median(peak_vals[:top80_n]))

    B = 0.0
    if valleys:
        valley_vals = sorted([v for _, v in valleys])
        top80_nv = max(1, int(len(valley_vals) * 0.8))
        B = float(np.median(valley_vals[:top80_nv]))
    h_th = (A + B) / 2.0

    expected_gap = _estimate_vernier_tick_gap([], main_gap)
    dedupe_tol = max(3.0, expected_gap * 0.35) if expected_gap > 0 else 3.0
    peak_tick_xs = _dedupe_tick_xs([x for x, v in peaks if v >= h_th], dedupe_tol)
    segment_tick_xs = _tick_xs_from_projection_segments(
        proj_norm, smooth, h_th, dedupe_tol
    )
    raw_tick_xs = segment_tick_xs if len(segment_tick_xs) >= 3 else peak_tick_xs
    if expected_gap <= 0:
        expected_gap = _estimate_vernier_tick_gap(raw_tick_xs, main_gap)
    if len(raw_tick_xs) < 3 or expected_gap <= 2.0:
        return None

    face_left_x = _find_vernier_face_left_edge(gray, band_y1, band_y2, x1, x2, expected_gap)
    ignore_until = int(round(face_left_x + expected_gap * 0.55)) if face_left_x > 0 else int(round(expected_gap * 0.75))
    early_filter_until = int(round(ignore_until + expected_gap * 3.0))
    tick_xs = []
    for x in raw_tick_xs:
        if x < ignore_until:
            continue
        if x < early_filter_until and not _has_top_connected_vernier_stroke(band, x):
            continue
        tick_xs.append(x)
    if len(tick_xs) < 3:
        tick_xs = raw_tick_xs
        ignore_until = 0

    return {
        'x1': x1,
        'x2': x2,
        'band_y1': band_y1,
        'band_y2': band_y2,
        'band': band,
        'proj': proj,
        'proj_norm': proj_norm,
        'smooth': smooth,
        'peaks': peaks,
        'valleys': valleys,
        'peak_tick_xs_local': peak_tick_xs,
        'segment_tick_xs_local': segment_tick_xs,
        'raw_tick_xs_local': raw_tick_xs,
        'tick_xs_local': tick_xs,
        'tick_xs_global': [x1 + x for x in tick_xs],
        'h_th': h_th,
        'A': A,
        'B': B,
        'expected_gap': expected_gap,
        'face_left_x': face_left_x,
        'ignore_until': ignore_until,
        'early_filter_until': early_filter_until,
    }


def _build_ticks_from_band_detection(band_detection: dict,
                                     long_tick_factor: float = None) -> List[dict]:
    """Build tick dictionaries directly from the narrow-band projection result."""
    if not band_detection:
        return []
    band = band_detection.get('band')
    if band is None or band.size == 0:
        return []
    if long_tick_factor is None:
        long_tick_factor = config.vernier_scale.long_tick_factor

    y_offset = int(band_detection.get('band_y1', 0))
    x_offset = int(band_detection.get('x1', 0))
    h, w = band.shape[:2]
    ticks = []
    for x in band_detection.get('tick_xs_local', []):
        x = int(x)
        if x < 0 or x >= w:
            continue
        refined = _refine_vernier_tick_from_band(
            band, x, band_detection.get('expected_gap', 0.0)
        )
        if refined is None:
            strip = band[:, max(0, x - 2):min(w, x + 3)] > 0
            row_on = np.mean(strip, axis=1) > 0.16 if strip.size else np.zeros(h, dtype=bool)
            ys = np.where(row_on)[0]
            if len(ys) > 0:
                y_start_local = int(ys[0])
                y_end_local = int(ys[-1])
            else:
                y_start_local = 0
                y_end_local = h - 1
            x_refined = float(x)
            x_top = float(x)
            x_bottom = float(x)
            slope = 0.0
        else:
            x_refined = float(refined['x'])
            y_start_local = int(refined['y_start'])
            y_end_local = int(refined['y_end'])
            x_top = float(refined['x_top'])
            x_bottom = float(refined['x_bottom'])
            slope = float(refined['slope'])
        y_start = y_start_local + y_offset
        y_end = y_end_local + y_offset
        length = max(1, y_end - y_start)
        ticks.append({
            'x': int(round(x_refined)) + x_offset,
            'x_projection': x + x_offset,
            'x_refined_local': x_refined,
            'x_top': x_top + x_offset,
            'x_bottom': x_bottom + x_offset,
            'y_start': y_start,
            'y_end': y_end,
            'y_mid': int(round((y_start + y_end) / 2.0)),
            'length': length,
            'fit_slope': slope,
            'source': 'band_projection_refined',
        })

    if ticks:
        median_len = float(np.median([t['length'] for t in ticks]))
        for t in ticks:
            t['is_long'] = t['length'] > median_len * long_tick_factor
    return sorted(ticks, key=lambda t: t['x'])


def _refine_vernier_tick_from_band(band: np.ndarray,
                                   approx_x: int,
                                   expected_gap: float) -> dict:
    """Refine one vernier tick by fitting local foreground stroke centers."""
    if band is None or band.size == 0:
        return None
    h, w = band.shape[:2]
    if h <= 0 or w <= 0:
        return None

    approx_x = int(round(approx_x))
    radius = max(4, min(10, int(round(expected_gap * 0.35)) if expected_gap > 0 else 7))
    x1 = max(0, approx_x - radius)
    x2 = min(w - 1, approx_x + radius)
    if x2 <= x1:
        return None

    crop = band[:, x1:x2 + 1] > 0
    if crop.size == 0:
        return None

    centers = []
    rows = []
    weights = []
    local_approx = approx_x - x1
    max_seg_w = max(2, int(radius * 0.9))
    for y in range(h):
        xs = np.where(crop[y])[0]
        if len(xs) == 0:
            continue

        segs = _contiguous_int_segments(xs)
        if not segs:
            continue
        seg = min(
            segs,
            key=lambda s: (
                abs(((s[0] + s[1]) / 2.0) - local_approx),
                -(s[1] - s[0] + 1),
            )
        )
        seg_w = seg[1] - seg[0] + 1
        if seg_w > max_seg_w:
            continue

        center = x1 + (seg[0] + seg[1]) / 2.0
        if abs(center - approx_x) > radius * 0.85:
            continue
        centers.append(center)
        rows.append(float(y))
        weights.append(float(seg_w))

    min_points = max(5, int(h * 0.14))
    if len(centers) < min_points:
        return None

    ys = np.array(rows, dtype=float)
    xs = np.array(centers, dtype=float)
    ws = np.array(weights, dtype=float)
    y_start = int(round(float(np.min(ys))))
    y_end = int(round(float(np.max(ys))))
    if y_end <= y_start:
        return None

    if len(xs) >= 3 and float(np.ptp(ys)) >= 2.0:
        slope, intercept = np.polyfit(ys, xs, 1, w=ws)
    else:
        slope = 0.0
        intercept = float(np.average(xs, weights=ws))

    length = y_end - y_start
    y_ref = y_start + min(max(2, int(round(length * 0.25))), 10)
    y_ref = max(y_start, min(y_end, y_ref))
    x_ref = float(slope * y_ref + intercept)
    x_top = float(slope * y_start + intercept)
    x_bottom = float(slope * y_end + intercept)

    return {
        'x': max(0.0, min(float(w - 1), x_ref)),
        'x_top': max(0.0, min(float(w - 1), x_top)),
        'x_bottom': max(0.0, min(float(w - 1), x_bottom)),
        'y_start': y_start,
        'y_end': y_end,
        'slope': float(slope),
    }


def _contiguous_int_segments(xs: np.ndarray) -> List[tuple]:
    if xs is None or len(xs) == 0:
        return []
    xs = np.array(xs, dtype=int)
    segments = []
    start = int(xs[0])
    prev = int(xs[0])
    for value in xs[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        segments.append((start, prev))
        start = value
        prev = value
    segments.append((start, prev))
    return segments


def _count_right_grid_hits(start_x: float,
                           tick_xs: List[int],
                           expected_gap: float,
                           tol: float) -> int:
    if expected_gap <= 0 or not tick_xs:
        return 0
    xs = np.array(sorted(float(x) for x in tick_xs), dtype=float)
    max_x = float(xs[-1])
    hits = 0
    misses = 0
    for k in range(1, 55):
        target = float(start_x) + expected_gap * k
        if target > max_x + tol:
            break
        if np.min(np.abs(xs - target)) <= tol:
            hits += 1
            misses = 0
        else:
            misses += 1
            if misses >= 3 and k >= 6:
                break
    return hits


def _find_corresponding_mapped_tick(corrected_tick: dict,
                                    corrected_ticks: List[dict],
                                    mapped_ticks: List[dict]) -> dict:
    if corrected_tick is None or not corrected_ticks or not mapped_ticks:
        return None
    best_idx = min(
        range(min(len(corrected_ticks), len(mapped_ticks))),
        key=lambda i: abs(corrected_ticks[i]['x'] - corrected_tick['x'])
    )
    return mapped_ticks[best_idx]


def _apply_near_integer_snap(mapped_ticks: List[dict],
                             corrected_ticks: List[dict],
                             zero_x: float,
                             zero_x_corrected: float,
                             main_ticks: List[dict],
                             main_gap: float) -> tuple:
    """Compensate small systematic parallax when zero is nearly on a cm tick."""
    snap_info = {'applied': False}
    if not mapped_ticks or not corrected_ticks or not main_ticks or main_gap <= 0:
        return mapped_ticks, corrected_ticks, zero_x, zero_x_corrected, snap_info

    long_ticks = [t for t in main_ticks if t.get('is_long')]
    lengths = [float(t.get('length', 0)) for t in main_ticks]
    if lengths:
        med_len = float(np.median(lengths))
        secondary = [t for t in main_ticks if float(t.get('length', 0)) >= med_len * 1.18]
        if len(secondary) > len(long_ticks):
            long_ticks = secondary
    if not long_ticks:
        return mapped_ticks, corrected_ticks, zero_x, zero_x_corrected, snap_info

    nearest_long = min(long_ticks, key=lambda t: abs(float(t['x']) - float(zero_x)))
    shift = float(nearest_long['x']) - float(zero_x)
    snap_tol = max(2.0, main_gap * 0.18)
    if abs(shift) > snap_tol or abs(shift) < 0.5:
        return mapped_ticks, corrected_ticks, zero_x, zero_x_corrected, snap_info

    main_xs = sorted(float(t['x']) for t in main_ticks)
    sample = mapped_ticks[:min(5, len(mapped_ticks))]
    if len(sample) < 3:
        return mapped_ticks, corrected_ticks, zero_x, zero_x_corrected, snap_info

    offsets = []
    for tick in sample:
        tx = float(tick['x'])
        nearest = min(main_xs, key=lambda x: abs(x - tx))
        offsets.append(tx - nearest)
    median_offset = float(np.median(offsets))
    spread = float(np.median(np.abs(np.array(offsets, dtype=float) - median_offset)))
    if abs(median_offset + shift) > max(1.5, main_gap * 0.08):
        return mapped_ticks, corrected_ticks, zero_x, zero_x_corrected, snap_info
    if spread > max(1.5, main_gap * 0.10):
        return mapped_ticks, corrected_ticks, zero_x, zero_x_corrected, snap_info

    corrected_shift = shift
    if len(mapped_ticks) >= 2 and len(corrected_ticks) >= 2:
        mapped_gap = float(np.median(np.diff([t['x'] for t in mapped_ticks[:min(8, len(mapped_ticks))]])))
        corrected_gap = float(np.median(np.diff([t['x'] for t in corrected_ticks[:min(8, len(corrected_ticks))]])))
        if abs(mapped_gap) > 1e-6:
            corrected_shift = shift * corrected_gap / mapped_gap

    snapped_mapped = []
    for tick in mapped_ticks:
        out = dict(tick)
        out['x_before_snap'] = out.get('x')
        out['x'] = int(round(float(out['x']) + shift))
        snapped_mapped.append(out)

    snapped_corrected = []
    for tick in corrected_ticks:
        out = dict(tick)
        out['x_before_snap'] = out.get('x')
        out['x'] = int(round(float(out['x']) + corrected_shift))
        if 'x_refined_local' in out:
            out['x_refined_local'] = float(out['x_refined_local']) + corrected_shift
        snapped_corrected.append(out)

    snap_info = {
        'applied': True,
        'shift': shift,
        'corrected_shift': corrected_shift,
        'target_main_x': float(nearest_long['x']),
        'median_offset': median_offset,
        'spread': spread,
    }
    return (
        snapped_mapped,
        snapped_corrected,
        float(zero_x) + shift,
        float(zero_x_corrected) + corrected_shift,
        snap_info,
    )


def _find_zero_from_band_detection(vernier_ticks: List[dict],
                                   band_detection: dict):
    """Locate vernier zero from the already computed narrow-band ticks."""
    if not band_detection or not vernier_ticks:
        return None, False, None

    x1 = int(band_detection['x1'])
    x2 = int(band_detection['x2'])
    band = band_detection['band']
    proj_norm = band_detection['proj_norm']
    smooth = band_detection['smooth']
    peaks = band_detection['peaks']
    valleys = band_detection['valleys']
    tick_xs = list(band_detection['tick_xs_local'])
    h_th = float(band_detection['h_th'])
    A = float(band_detection['A'])
    B = float(band_detection['B'])
    typical_gap = float(band_detection['expected_gap'])
    face_left_x = int(band_detection.get('face_left_x', 0))

    if len(tick_xs) < 3 or typical_gap <= 2.0:
        return None, False, None

    valley_candidates = []
    fallback_sequence = []
    tol = max(3.0, typical_gap * 0.35)
    min_hits = 5 if len(tick_xs) < 14 else 8
    for idx, tx in enumerate(tick_xs):
        prev_x = tick_xs[idx - 1] if idx > 0 else None
        left_ref = prev_x if prev_x is not None else face_left_x
        leading_gap = float(tx - left_ref) if left_ref and tx > left_ref else float(tx)

        left_a = max(0, int(round(tx - typical_gap * 1.8)))
        left_b = max(0, int(round(tx - typical_gap * 0.45)))
        left_energy = float(np.mean(smooth[left_a:left_b])) if left_b > left_a else 0.0
        first_after_face = (
            idx == 0
            and face_left_x > 0
            and tx - face_left_x >= typical_gap * 0.55
        )
        blank_ok = (
            tx >= typical_gap * 0.75
            and (
                first_after_face
                or leading_gap >= typical_gap * 1.45
                or left_energy <= h_th * 0.65
            )
        )

        hits = _count_right_grid_hits(tx, tick_xs, typical_gap, tol)
        record = (idx, leading_gap, left_ref or 0, tx, hits)
        if hits >= min_hits:
            fallback_sequence.append(record)
        if blank_ok and hits >= min_hits:
            valley_candidates.append(record)

    if valley_candidates:
        best_valley = valley_candidates[0]
    elif fallback_sequence:
        valley_candidates = fallback_sequence
        best_valley = fallback_sequence[0]
    else:
        return None, False, None

    zero_local_x = best_valley[3]
    zero_global_x = x1 + zero_local_x
    ticks_in_range = [t for t in vernier_ticks if x1 - typical_gap <= t['x'] <= x2 + typical_gap]
    if ticks_in_range:
        nearest = min(ticks_in_range, key=lambda t: abs(t['x'] - zero_global_x))
        found_tick = nearest if abs(nearest['x'] - zero_global_x) <= tol else {'x': zero_global_x}
    else:
        found_tick = {'x': zero_global_x}

    vis = _make_valley_projection_vis(
        band, proj_norm, smooth, peaks, valleys, tick_xs, x1,
        h_th, A, B, typical_gap,
        valley_candidates, best_valley, found_tick)
    return found_tick, False, vis


def _find_zero_from_projection_valley(binary: np.ndarray,
                                      vernier_ticks: List[dict],
                                      body_x1: int,
                                      body_x2: int,
                                      main_gap: float,
                                      gray: np.ndarray = None,
                                      band_detection: dict = None):
    """投影谷底法定位零线（v7 重写）。

    算法：
      1. 游标区顶部 45% 二值图水平投影 → 归一化 + 平滑
      2. 找平滑曲线上所有局部极大值（峰）和局部极小值（谷）
      3. 取前 80% 最高的峰的中位数 A，前 80% 最深的谷绝对值的中位数 B
      4. h = (A + B) / 2
      5. 高于 h 的峰 = 游标刻度线位置
      6. 相邻峰间距明显偏大的位置 → 谷底候选
      7. 第一个谷底候选区右侧第一条峰 = 零线

    Returns:
        (zero_tick, zero_digit_found, vis_projection)
    """
    if binary is None or binary.size == 0 or not vernier_ticks:
        return None, False, None
    if band_detection is not None:
        return _find_zero_from_band_detection(vernier_ticks, band_detection)

    h, w = binary.shape[:2]
    x1 = max(0, min(w - 1, int(body_x1)))
    x2 = max(x1 + 1, min(w, int(body_x2)))
    band_y1, band_y2 = _find_vernier_tick_band(binary, x1, x2)
    band = binary[band_y1:band_y2, x1:x2]
    if band.size == 0:
        return None, False, None

    proj = np.sum(band > 0, axis=0).astype(float)
    if np.max(proj) <= 0:
        return None, False, None

    proj_norm = proj / np.max(proj)
    win = max(3, min(11, len(proj_norm) // 80))
    if win % 2 == 0:
        win += 1
    smooth = np.convolve(proj_norm, np.ones(win, dtype=float) / win, mode='same')
    n = len(smooth)

    # ── 1. 找局部极大值（峰）和局部极小值（谷）──
    min_dist = max(3, n // 200)
    peaks = []   # (x, value)
    valleys = [] # (x, value)
    for i in range(min_dist, n - min_dist):
        left = smooth[i - min_dist:i]
        right = smooth[i + 1:i + min_dist + 1]
        if smooth[i] > max(float(np.max(left)), float(np.max(right))):
            peaks.append((i, float(smooth[i])))
        if smooth[i] < min(float(np.min(left)), float(np.min(right))):
            valleys.append((i, float(smooth[i])))

    if len(peaks) < 3:
        return None, False, None

    # ── 2. 前 80% 中位数：峰取最高 80% 的中位数 A，谷取最深 80% 的中位数 B ──
    peak_vals = sorted([v for _, v in peaks], reverse=True)
    top80_n = max(1, int(len(peak_vals) * 0.8))
    A = float(np.median(peak_vals[:top80_n]))

    B = 0.0
    if valleys:
        valley_vals = sorted([v for _, v in valleys])  # 升序，最小值在前
        top80_nv = max(1, int(len(valley_vals) * 0.8))
        B = float(np.median(valley_vals[:top80_nv]))

    h_th = (A + B) / 2.0

    # ── 3. 筛选高于 h_th 的峰 → 刻度线 x 坐标 ──
    expected_gap = _estimate_vernier_tick_gap([], main_gap)
    dedupe_tol = max(3.0, expected_gap * 0.35) if expected_gap > 0 else 3.0
    tick_xs = _dedupe_tick_xs([x for x, v in peaks if v >= h_th], dedupe_tol)
    if len(tick_xs) < 3:
        return None, False, None

    if expected_gap <= 0:
        expected_gap = _estimate_vernier_tick_gap(tick_xs, main_gap)
    if expected_gap <= 2.0:
        return None, False, None

    # ── 4. 相邻峰间距 → 找显著偏大的谷底候选 ──
    diffs = np.diff(tick_xs)
    if len(diffs) < 2:
        return None, False, None

    typical_gap = float(expected_gap)
    if typical_gap < 2.0:
        return None, False, None

    # Find the first candidate after a blank lead-in, then require a stable
    # vernier spacing sequence to the right. Projection still supplies the
    # candidates, but it no longer decides zero only by a valley.
    valley_candidates = []
    fallback_sequence = []
    tol = max(3.0, typical_gap * 0.35)
    min_hits = 5 if len(tick_xs) < 14 else 8
    for idx, tx in enumerate(tick_xs):
        prev_x = tick_xs[idx - 1] if idx > 0 else None
        leading_gap = float(tx if prev_x is None else tx - prev_x)

        left_a = max(0, int(round(tx - typical_gap * 1.8)))
        left_b = max(0, int(round(tx - typical_gap * 0.45)))
        left_energy = float(np.mean(smooth[left_a:left_b])) if left_b > left_a else 0.0
        blank_ok = (
            tx >= typical_gap * 0.75
            and (leading_gap >= typical_gap * 1.45 or left_energy <= h_th * 0.65)
        )

        hits = _count_right_grid_hits(tx, tick_xs, typical_gap, tol)
        record = (idx, leading_gap, prev_x or 0, tx, hits)
        if hits >= min_hits:
            fallback_sequence.append(record)
        if blank_ok and hits >= min_hits:
            valley_candidates.append(record)

    if valley_candidates:
        best_valley = valley_candidates[0]
    elif fallback_sequence:
        valley_candidates = fallback_sequence
        best_valley = fallback_sequence[0]
    else:
        return None, False, None

    # ── 5. 第一个谷底候选区右侧第一条峰 = 零线 ──
    zero_local_x = best_valley[3]
    zero_global_x = x1 + zero_local_x

    # 在 vernier_ticks 中找最接近 zero_global_x 的 tick
    ticks_in_range = [t for t in vernier_ticks if x1 - typical_gap <= t['x'] <= x2 + typical_gap]
    if ticks_in_range:
        nearest = min(ticks_in_range, key=lambda t: abs(t['x'] - zero_global_x))
        found_tick = nearest if abs(nearest['x'] - zero_global_x) <= tol else {'x': zero_global_x}
    else:
        found_tick = {'x': zero_global_x}

    vis = _make_valley_projection_vis(
        band, proj_norm, smooth, peaks, valleys, tick_xs, x1,
        h_th, A, B, typical_gap,
        valley_candidates, best_valley, found_tick)

    return found_tick, False, vis


def _make_valley_projection_vis(band: np.ndarray,
                                 proj_norm: np.ndarray,
                                 smooth: np.ndarray,
                                 peaks: list,
                                 valleys: list,
                                 tick_xs: list,
                                 x_offset: int,
                                 h_th: float,
                                 A: float,
                                 B: float,
                                 typical_gap: float,
                                 valley_candidates: list,
                                 best_valley: tuple,
                                 found_tick: dict = None) -> np.ndarray:
    """可视化：投影曲线 + 峰/谷标记 + 阈值 h + 刻度线 + 谷底候选 + 零线"""
    band_h, band_w = band.shape[:2]
    n = len(smooth)
    plot_w = max(700, n + 80)
    plot_h = 380
    margin = 40

    vis = np.zeros((plot_h + band_h + 30, plot_w, 3), dtype=np.uint8)
    vis[:] = (20, 20, 28)

    # ── 顶部：二值图缩略 ──
    bw_display = cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)
    thumb_w = min(band_w, plot_w - margin * 2)
    thumb_h = band_h
    if band_h > 0 and band_w > 0:
        thumb = cv2.resize(bw_display, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        ox = margin
        vis[:thumb_h, ox:ox + thumb_w] = thumb
        cv2.putText(vis, f"Auto tick band (h={band_h}px, w={band_w}px)", (margin, thumb_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 150), 1)

    # ── 图表区域 ──
    chart_y0 = band_h + 28
    chart_h = plot_h - 8
    chart_w = n
    chart_left = margin

    cv2.rectangle(vis, (chart_left, chart_y0),
                  (chart_left + chart_w, chart_y0 + chart_h), (35, 35, 42), -1)

    # 坐标轴
    cv2.line(vis, (chart_left, chart_y0), (chart_left, chart_y0 + chart_h), (80, 80, 90), 1)
    mid_y = chart_y0 + chart_h // 2
    cv2.line(vis, (chart_left, mid_y), (chart_left + chart_w, mid_y), (50, 50, 58), 1, cv2.LINE_AA)

    # 原始投影（浅灰柱状）
    for i in range(0, n, max(1, n // 700)):
        bar_h = int(proj_norm[i] * chart_h * 0.5)
        x = chart_left + i
        cv2.line(vis, (x, mid_y), (x, mid_y - bar_h), (50, 50, 62), 1)

    # 平滑曲线
    pts = []
    for i in range(n):
        y = chart_y0 + int(chart_h * 0.5 - smooth[i] * chart_h * 0.5)
        pts.append((chart_left + i, y))
    for i in range(len(pts) - 1):
        cv2.line(vis, pts[i], pts[i + 1], (100, 200, 255), 2, cv2.LINE_AA)

    # 峰标记（绿色圆点）和谷标记（红色圆点）
    for x, v in peaks:
        px = chart_left + x
        py = chart_y0 + int(chart_h * 0.5 - v * chart_h * 0.5)
        cv2.circle(vis, (px, py), 2, (0, 255, 100), -1)
    for x, v in valleys:
        px = chart_left + x
        py = chart_y0 + int(chart_h * 0.5 - v * chart_h * 0.5)
        cv2.circle(vis, (px, py), 2, (100, 140, 255), -1)

    # 阈值线 h (黄色)
    h_y = chart_y0 + int(chart_h * 0.5 - h_th * chart_h * 0.5)
    cv2.line(vis, (chart_left, h_y), (chart_left + chart_w, h_y),
             (0, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(vis, f"h_th={h_th:.3f} (A={A:.3f} + B={B:.3f}) / 2", (chart_left + 4, h_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 220), 1)

    # A 线 (绿色) 和 B 线 (蓝色)
    a_y = chart_y0 + int(chart_h * 0.5 - A * chart_h * 0.5)
    cv2.line(vis, (chart_left, a_y), (chart_left + chart_w, a_y),
             (0, 180, 80), 1, cv2.LINE_AA)
    cv2.putText(vis, f"peak_median_top80%={A:.3f}", (chart_left + chart_w - 260, a_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 180, 80), 1)
    b_y = chart_y0 + int(chart_h * 0.5 - B * chart_h * 0.5)
    cv2.line(vis, (chart_left, b_y), (chart_left + chart_w, b_y),
             (255, 140, 80), 1, cv2.LINE_AA)
    cv2.putText(vis, f"valley_median_top80%={B:.3f}", (chart_left + chart_w - 260, b_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 140, 80), 1)

    # 筛选后的刻度线（高于 h_th 的峰）
    for tx in tick_xs:
        px = chart_left + tx
        cv2.line(vis, (px, chart_y0), (px, chart_y0 + chart_h), (0, 200, 100), 1, cv2.LINE_AA)

    # 谷底候选区（浅红半透明高亮）
    for cand in valley_candidates:
        _, _, left_x, right_x, *_ = cand
        cv2.rectangle(vis, (chart_left + left_x, chart_y0),
                      (chart_left + right_x, chart_y0 + chart_h), (60, 20, 30), -1)

    # 最佳谷底（选中的：第一个满足条件的，品红色框 + 标签）
    if best_valley:
        _, _, best_left_x, best_right_x, *rest = best_valley
        hits = rest[0] if rest else 0
        cv2.rectangle(vis, (chart_left + best_left_x, chart_y0),
                      (chart_left + best_right_x, chart_y0 + chart_h), (220, 50, 120), 2, cv2.LINE_AA)
        mid_gap_x = chart_left + (best_left_x + best_right_x) // 2
        cv2.putText(vis, f"ZERO CAND hits={hits}", (mid_gap_x - 40, chart_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 50, 120), 2)

    # 找零线高亮（粗亮绿线）
    if found_tick:
        zx = int(found_tick['x']) - x_offset
        if 0 <= zx < n:
            zpx = chart_left + zx
            cv2.line(vis, (zpx, chart_y0 - 10), (zpx, chart_y0 + chart_h),
                     (50, 255, 50), 3, cv2.LINE_AA)
            cv2.circle(vis, (zpx, chart_y0 + 16), 8, (50, 255, 50), -1)
            cv2.putText(vis, "ZERO", (zpx + 10, chart_y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 2)

    # 间距标注
    cv2.putText(vis, f"typical_gap={typical_gap:.0f}px | {len(tick_xs)} ticks above h | {len(valley_candidates)} valley candidates",
                (chart_left + 4, chart_y0 + chart_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 170), 1)

    # 标题
    cv2.putText(vis, "Zero-Line Detection (auto tick band + right grid check)", (margin, band_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 220), 1)

    # 图例
    legend_y = chart_y0 + chart_h + 2
    cv2.circle(vis, (margin + 6, legend_y - 6), 3, (0, 255, 100), -1)
    cv2.putText(vis, "peak", (margin + 14, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 100), 1)
    cx = margin + 60
    cv2.circle(vis, (cx + 6, legend_y - 6), 3, (100, 140, 255), -1)
    cv2.putText(vis, "valley", (cx + 14, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 140, 255), 1)
    cx += 80
    cv2.line(vis, (cx, legend_y), (cx + 24, legend_y), (0, 220, 220), 2)
    cv2.putText(vis, "h_th", (cx + 28, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 220, 220), 1)
    cx += 70
    cv2.rectangle(vis, (cx, legend_y - 8), (cx + 24, legend_y + 8), (220, 50, 120), 2)
    cv2.putText(vis, "gap", (cx + 28, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 50, 120), 1)

    if found_tick:
        status = "ZERO FOUND"
        sc = (50, 255, 50)
    else:
        status = "NOT FOUND"
        sc = (255, 100, 100)
    cv2.putText(vis, f"Result: {status}", (plot_w - 200, legend_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, sc, 1)

    return vis


def recognize_vernier_scale(region: dict,
                             main_gap: float,
                             color_region: np.ndarray = None,
                             main_ticks: List[dict] = None) -> dict:
    """Vernier detection: body-range crop + split-anchored tick localization."""
    img = region['image']
    h, w = img.shape

    if main_gap <= 0:
        main_gap = 10.0

    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=config.vernier_scale.adaptive_block_size,
        C=config.vernier_scale.adaptive_C
    )
    if np.sum(binary > 0) < img.shape[0] * img.shape[1] * 0.03:
        _, binary = cv2.threshold(img, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    region['binary'] = binary

    vproj = np.sum(binary, axis=0).astype(float)
    vproj_norm = vproj / np.max(vproj) if np.max(vproj) > 0 else vproj

    if region.get('is_body_crop'):
        body_x1, body_x2 = 0, w
    else:
        body_x1, body_x2 = _find_vernier_body_x_range(img)

    band_detection = _detect_vernier_band_projection(
        binary, body_x1, body_x2, main_gap, img
    )
    vernier_xs = np.array(
        band_detection['tick_xs_global'] if band_detection else [],
        dtype=int
    )
    vernier_ticks = []
    if len(vernier_xs) >= config.vernier_scale.min_tick_count:
        vernier_ticks = _build_ticks_from_band_detection(
            band_detection,
            long_tick_factor=config.vernier_scale.long_tick_factor
        )

    if len(vernier_ticks) < config.vernier_scale.min_tick_count:
        fallback_xs = find_peaks_adaptive(
            vproj_norm,
            min_dist=config.vernier_scale.peak_min_dist,
            threshold_factor=config.vernier_scale.peak_threshold_factor,
        )
        if len(fallback_xs) >= config.vernier_scale.min_tick_count and config.vernier_scale.spacing_refine_enabled:
            refined_xs = refine_ticks_by_spacing(
                fallback_xs, binary,
                spacing_tolerance=config.vernier_scale.spacing_tolerance,
                gap_factor=config.vernier_scale.spacing_gap_factor,
                dup_factor=config.vernier_scale.spacing_dup_factor,
                snap_ratio=config.vernier_scale.spacing_snap_ratio,
            )
            if len(refined_xs) >= config.vernier_scale.min_tick_count:
                fallback_xs = refined_xs
        if len(fallback_xs) >= config.vernier_scale.min_tick_count:
            vernier_ticks = extract_ticks_from_binary(
                binary, fallback_xs,
                min_length_ratio=0.04,
                long_tick_factor=config.vernier_scale.long_tick_factor)
            vernier_ticks.sort(key=lambda t: t['x'])

        if len(vernier_ticks) < config.vernier_scale.min_tick_count:
            work_binary = binary[:, body_x1:body_x2]
            vernier_ticks = extract_ticks_from_anchor_band(
                work_binary,
                direction="down",
                min_length_ratio=0.04,
                band_ratio=0.45,
                peak_min_dist=config.vernier_scale.peak_min_dist,
                peak_threshold_factor=0.05,
                long_tick_factor=config.vernier_scale.long_tick_factor,
            )
            for t in vernier_ticks:
                t['x'] += body_x1
            vernier_ticks.sort(key=lambda t: t['x'])
        vernier_xs = np.array([t['x'] for t in vernier_ticks], dtype=int)
        if len(vernier_ticks) < config.vernier_scale.min_tick_count:
            return _empty_vernier_result()

    precision = 0.02
    grid_ticks = _filter_vernier_ticks_by_grid(vernier_ticks, main_gap)
    if len(grid_ticks) >= config.vernier_scale.min_tick_count:
        vernier_ticks = grid_ticks
        band_detection = _sync_band_detection_ticks(band_detection, vernier_ticks)
    vernier_xs = np.array([t['x'] for t in vernier_ticks], dtype=int)
    v_gap = float(np.median(np.diff([t['x'] for t in vernier_ticks]))) if len(vernier_ticks) >= 2 else 0.0

    zero_tick, zero_digit_found, valley_vis = _find_zero_from_projection_valley(
        binary, vernier_ticks, body_x1, body_x2, main_gap,
        gray=img, band_detection=band_detection
    )
    if zero_tick is not None:
        zero_x = float(zero_tick['x'])
        clean_vernier = [t for t in vernier_ticks
                         if t['x'] >= zero_x - max(v_gap, 1.0) * 0.4]
        if len(clean_vernier) >= config.vernier_scale.min_tick_count:
            vernier_ticks = sorted(clean_vernier, key=lambda t: t['x'])
            vernier_xs = np.array([t['x'] for t in vernier_ticks], dtype=int)
            v_gap = float(np.median(np.diff([t['x'] for t in vernier_ticks]))) if len(vernier_ticks) >= 2 else v_gap

    if zero_tick is None:
        zero_tick, zero_digit_found = _find_zero_tick(vernier_ticks, region)
        zero_x = float(zero_tick['x']) if zero_tick else float(vernier_ticks[0]['x'])

    clean_vernier = [t for t in vernier_ticks if t['x'] >= zero_x - max(v_gap, 1.0) * 0.4]
    if len(clean_vernier) >= config.vernier_scale.min_tick_count:
        vernier_ticks = clean_vernier
    vernier_ticks.sort(key=lambda t: t['x'])

    corrected_ticks = [dict(t) for t in vernier_ticks]
    zero_x_corrected = float(zero_x)

    mapped_ticks = _map_ticks_to_original(corrected_ticks, region)
    zero_y = None
    if zero_tick is not None:
        zero_y = zero_tick.get('y_mid', None)
    if zero_y is None and corrected_ticks:
        zero_y = float(np.median([t.get('y_mid', 0) for t in corrected_ticks]))
    zero_x = _map_x_to_original(zero_x_corrected, region, zero_y)
    mapped_ticks, corrected_ticks, zero_x, zero_x_corrected, integer_snap = _apply_near_integer_snap(
        mapped_ticks, corrected_ticks, zero_x, zero_x_corrected, main_ticks, main_gap
    )
    vernier_xs = np.array([t['x'] for t in corrected_ticks], dtype=int)

    vis_ticks = _draw_vernier_ticks(
        region, binary, corrected_ticks, vproj_norm, vernier_xs,
        zero_x_corrected, zero_digit_found, band_detection=band_detection
    )

    vernier_reading, aligned_tick, align_conf = find_best_alignment(
        mapped_ticks, precision, main_ticks
    )
    aligned_tick_corrected = _find_corresponding_mapped_tick(
        aligned_tick, mapped_ticks, corrected_ticks
    )

    vis_alignment = _draw_alignment(
        region, color_region, corrected_ticks,
        main_gap, zero_x_corrected, aligned_tick_corrected, align_conf
    )

    return {
        'vernier_ticks': mapped_ticks,
        'precision': precision,
        'vernier_reading': vernier_reading,
        'zero_x': zero_x,
        'aligned_tick': aligned_tick,
        'alignment_confidence': align_conf,
        'vis_ticks': vis_ticks,
        'vis_alignment': vis_alignment,
        'vis_valley': valley_vis,
        'vproj_norm': vproj_norm,
        'vernier_peaks': vernier_xs,
        'vernier_band_detection': band_detection,
        'integer_snap': integer_snap,
        'zero_digit_found': zero_digit_found,
    }
