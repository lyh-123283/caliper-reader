"""
步骤 3 — 主尺识别（刻度线检测 + 数字定位识别）

流程:
  1. 垂直投影 → 刻度线位置检测
  2. 精密提取每条刻线的起止点
  3. OCR 识别刻度数字（2, 3, 4, 5... cm 标记）
  4. 结合零线位置计算整数读数
"""

import cv2
import numpy as np
from typing import List, Tuple

from .result import DigitInfo
from .utils import (
    find_peaks_adaptive, extract_ticks_from_binary, draw_projection_plot,
    refine_ticks_by_spacing,
)
from .config import config


def recognize_main_scale(region: dict,
                          color_region: np.ndarray = None) -> dict:
    """
    主尺识别主函数

    Args:
        region:       主尺区域 dict {image, binary, y_offset, height}
        color_region: 对应的彩色区域（用于 OCR）

    Returns:
        dict with keys:
            'main_ticks':   刻度线列表
            'main_gap':     主尺间距（像素）
            'main_digits':  OCR 识别的数字列表
            'main_reading': 整数读数（暂用间距估算）
            'vis_ticks':    刻度线可视化
            'vis_digits':   数字识别可视化
    """
    img = region['image']
    h, w = img.shape

    # ── 1. 自适应二值化（比 OTSU 更鲁棒，避免低对比度时全部消失）──
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=config.main_scale.adaptive_block_size,
        C=config.main_scale.adaptive_C
    )
    # 回退：自适应阈值得到的前景太少（全部淹没），改用 OTSU
    if np.sum(binary > 0) < w * h * 0.03:
        _, binary = cv2.threshold(img, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    region['binary'] = binary

    # ── 2. 垂直投影 → 峰值检测 ──
    vproj = np.sum(binary, axis=0).astype(float)
    if np.max(vproj) > 0:
        vproj_norm = vproj / np.max(vproj)
    else:
        vproj_norm = vproj

    main_xs = find_peaks_adaptive(vproj_norm, min_dist=config.main_scale.peak_min_dist,
                                   threshold_factor=config.main_scale.peak_threshold_factor)
    if len(main_xs) < config.main_scale.min_tick_count:
        return _empty_main_result()

    # ── 2.5 等间距补全 & 校验 ──
    if config.main_scale.spacing_refine_enabled:
        refined_xs = refine_ticks_by_spacing(
            main_xs, binary,
            spacing_tolerance=config.main_scale.spacing_tolerance,
            gap_factor=config.main_scale.spacing_gap_factor,
            dup_factor=config.main_scale.spacing_dup_factor,
            snap_ratio=config.main_scale.spacing_snap_ratio,
        )
        if len(refined_xs) >= config.main_scale.min_tick_count:
            main_xs = refined_xs

    # ── 3. 精密提取刻线 ──
    main_ticks = extract_ticks_from_binary(
        binary, main_xs,
        long_tick_factor=config.main_scale.long_tick_factor)
    if len(main_ticks) < config.main_scale.min_tick_count:
        return _empty_main_result()

    main_ticks.sort(key=lambda t: t['x'])
    main_xs = np.array([t['x'] for t in main_ticks], dtype=int)
    main_gap = float(np.median(np.diff([t['x'] for t in main_ticks])))

    # v6.5: OCR 数字识别已迁移到 merger（在拿到 zero_x 后定向识别）
    # OCR is handled later in merger after zero_x is known. Keep main_digits empty.
    # 避免 pipeline.py 报错。
    region['main_ticks'] = main_ticks

    # ── 4. 主尺读数（合并阶段计算）──
    main_reading = 0.0

    # ── 可视化 ──
    vis_ticks = _draw_main_ticks(region, binary, main_ticks, vproj_norm, main_xs)

    # v6.5: OCR 已迁出，vis_digits 用空图占位
    empty_vis = np.zeros((100, 300, 3), dtype=np.uint8)

    return {
        'main_ticks': main_ticks,
        'main_gap': main_gap,
        'main_digits': [],  # v6.5: 留空，由 merger 定向填充
        'main_reading': main_reading,
        'vis_ticks': vis_ticks,
        'vis_digits': empty_vis,
    }


# ═══════════════════════════ 内部函数 ═══════════════════════════

def find_nearest_cm_digit_region(main_ticks: List[dict],
                                     main_gap: float,
                                     zero_x: float,
                                     binary: np.ndarray) -> tuple:
    if not main_ticks or main_gap <= 0 or zero_x <= 0 or binary is None:
        return None, 0, 0

    H, W = binary.shape[:2]

    y_starts = [t['y_start'] for t in main_ticks if 'y_start' in t]
    if len(y_starts) < 3:
        return None, 0, 0
    y_top_tick = max(y_starts)

    y_top = max(0, y_top_tick - int(4 * main_gap))
    y_bottom = max(y_top + 8, y_top_tick - int(1 * main_gap))
    y_bottom = min(H, y_bottom)

    cm_px = int(main_gap * 10)
    x_left = max(0, int(zero_x - 1.7 * cm_px))
    x_right = min(W, int(zero_x + 0.4 * cm_px))
    if x_right - x_left < 8:
        return None, 0, 0

    binary_crop = binary[y_top:y_bottom, x_left:x_right].copy()
    return binary_crop, x_left, y_top


def find_digit_cc_candidates(binary_crop: np.ndarray,
                             x_offset: int, y_offset: int,
                             zero_x: float = None,
                             min_area: int = 700,
                             max_area: int = 1200,
                             min_aspect: float = 0.6,
                             max_aspect: float = 3.5) -> list:
    """Return all plausible digit connected components in the OCR crop."""
    if binary_crop is None or binary_crop.size == 0:
        return []

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_crop, connectivity=8)
    if num_labels < 2:
        return []

    H, W = binary_crop.shape
    dynamic_min_area = max(250, int(H * H * 0.09))
    effective_min_area = min(min_area, dynamic_min_area)
    candidates = []
    for j in range(1, num_labels):
        x = int(stats[j, cv2.CC_STAT_LEFT])
        y = int(stats[j, cv2.CC_STAT_TOP])
        w = int(stats[j, cv2.CC_STAT_WIDTH])
        h = int(stats[j, cv2.CC_STAT_HEIGHT])
        area = int(stats[j, cv2.CC_STAT_AREA])
        if area < effective_min_area or area > max_area:
            continue
        if w < 3 or h < 5:
            continue
        aspect = h / max(w, 1)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        y_center_ratio = (y + h / 2) / H
        x_center_ratio = (x + w / 2) / W
        confidence = (
            0.4 * min(1.0, area / 200) +
            0.3 * (1.0 - abs(aspect - 1.5) / 2.0) +
            0.3 * y_center_ratio
        )
        confidence = max(0.0, min(1.0, confidence))

        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)
        bbox = (x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset)
        candidates.append({
            'idx': j,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'area': area,
            'aspect': aspect,
            'y_ratio': y_center_ratio,
            'x_ratio': x_center_ratio,
            'center_x': x + w / 2 + x_offset,
            'bbox': bbox,
            'confidence': confidence,
            'digit_crop': binary_crop[y1:y2, x1:x2],
        })

    return sorted(candidates, key=lambda c: c['center_x'])






def _draw_main_ticks(region: dict,
                      binary: np.ndarray,
                      main_ticks: List[dict],
                      vproj: np.ndarray,
                      peaks: np.ndarray) -> np.ndarray:
    """绘制主尺刻度线检测结果 — 灰度底图 + 右侧二值图小窗"""
    img = region['image']
    h, w = img.shape

    # 主图：增强灰度图上叠加刻线
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 画刻度线
    for t in main_ticks:
        color = (0, 255, 100) if t.get('is_long', False) else (0, 180, 80)
        thickness = 3 if t.get('is_long', False) else 2
        cv2.line(vis, (t['x'], t['y_start']), (t['x'], t['y_end']), color, thickness)
        if t.get('is_long', False):
            cv2.circle(vis, (t['x'], t['y_mid']), 5, (255, 255, 0), -1)

    # ── 右侧二值图小窗（显示检测器实际看到的二值图）──
    bin_thumb_w = max(50, w // 4)
    bin_thumb_h = int(h * bin_thumb_w / w)
    bin_thumb = cv2.resize(binary, (bin_thumb_w, bin_thumb_h), interpolation=cv2.INTER_AREA)
    bin_thumb_3ch = cv2.cvtColor(bin_thumb, cv2.COLOR_GRAY2BGR)
    # 放在右下角
    bx, by = w - bin_thumb_w, h - bin_thumb_h
    vis[by:by + bin_thumb_h, bx:bx + bin_thumb_w] = bin_thumb_3ch
    cv2.rectangle(vis, (bx, by), (bx + bin_thumb_w, by + bin_thumb_h), (255, 255, 255), 1)
    cv2.putText(vis, "BIN", (bx + 3, by + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # 下方追加投影图
    proj_vis = draw_projection_plot(vproj, peaks, width=w,
                                     title=f"Vertical Projection ({len(main_ticks)} ticks)")
    ph = proj_vis.shape[0]

    gap = 2
    out = np.zeros((h + ph + gap, w, 3), dtype=np.uint8)
    out[:] = (30, 30, 35)
    out[:h, :w] = vis
    out[h + gap:h + gap + ph, :w] = proj_vis

    cv2.putText(out, "STEP 3: Main Scale Ticks (gray + binary overlay)", (5, out.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 125), 1)

    return out


def _empty_main_result() -> dict:
    empty_img = np.zeros((100, 300, 3), dtype=np.uint8)
    return {
        'main_ticks': [],
        'main_gap': 0.0,
        'main_digits': [],
        'main_reading': 0.0,
        'vis_ticks': empty_img,
        'vis_digits': empty_img,
    }
