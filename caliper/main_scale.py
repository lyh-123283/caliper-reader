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
from .ocr import DigitReader
from .config import config


# 全局 OCR 实例（单例，避免重复加载）
_ocr_reader: DigitReader = None


def get_ocr_reader() -> DigitReader:
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = DigitReader()
    return _ocr_reader


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
    # 这里不再调用 _recognize_main_digits()。但保留 main_digits 字段（空列表）
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
    """v6.6: 圈出"zero_x 上方"的备选区二进制图，找游标 0 刻度线**左侧最近**主尺数字。

    物理依据（用户提供）：
      - 主尺数字在刻度线**上方**（在主尺行最顶刻度线之上）
      - 数字下方对应主尺 cm 整数刻度位置
      - 备选区宽度：zero_x - 1.2cm 到 zero_x + 0.3cm
        （右边界略过 zero_x，确保 zero_x 投影到的 cm 数字也在区内）
      - 备选区高度：主尺最顶刻度线 y - 2×tick_gap 到 y_top_tick

    Args:
        main_ticks: 主尺 tick 列表（每条含 x, y_start, y_end）
        main_gap:   1mm 像素间距
        zero_x:     游标 0 刻度线 x
        binary:     主尺区域二值图（THRESH_BINARY_INV，黑前景=255）

    Returns:
        (binary_crop, x_offset, y_offset) 或 (None, 0, 0)
        - binary_crop: 备选区二值图
        - x_offset, y_offset: 在原图中的偏移
    """
    if not main_ticks or main_gap <= 0 or zero_x <= 0 or binary is None:
        return None, 0, 0

    H, W = binary.shape[:2]

    # 1. 找主尺刻度线上沿 y 位置（取最大值，因为数字在上方，下探越深越安全）
    y_starts = [t['y_start'] for t in main_ticks if 'y_start' in t]
    if len(y_starts) < 3:
        return None, 0, 0
    y_top_tick = max(y_starts)

    # 2. 备选区 y 范围：整体上移 1*tick_gap
    y_top = max(0, y_top_tick - int(4 * main_gap))
    y_bottom = max(y_top + 8, y_top_tick - int(1 * main_gap))
    y_bottom = min(H, y_bottom)

    # 3. 备选区 x 范围
    cm_px = int(main_gap * 10)  # 1cm 像素
    x_left = max(0, int(zero_x - 1.2 * cm_px))
    x_right = min(W, int(zero_x + 0.3 * cm_px))
    if x_right - x_left < 8:
        return None, 0, 0

    # 4. 裁剪
    binary_crop = binary[y_top:y_bottom, x_left:x_right].copy()
    return binary_crop, x_left, y_top


def find_largest_digit_cc(binary_crop: np.ndarray,
                            x_offset: int, y_offset: int,
                            zero_x: float = None,
                            min_area: int = 700,
                            max_area: int = 1200,
                            min_aspect: float = 0.6,
                            max_aspect: float = 3.5) -> tuple:
    """v6.6: 在备选区二值图中找**最像数字**的连通域。

    数字连通域的特征（v6.5 经验）：
      - 面积: 700 ~ 1200 px²
      - 高/宽比: 0.6 ~ 3.5（数字大致 1:1 ~ 1:1.5，但 "1" 窄长）
      - 位置: 偏向备选区下半（数字底部贴刻度线）
      - 优先选零线左侧且最靠右的（最近 cm 数字）

    Args:
        binary_crop: 备选区二值图
        x_offset, y_offset: 在原图中的偏移
        zero_x: 零线 x 坐标（用于优先左侧候选）
        min/max_area, min/max_aspect: 连通域筛选阈值

    Returns:
        (digit_crop, bbox, confidence) 或 (None, None, 0.0)
        - digit_crop: 数字 patch 灰度图（裁剪后）
        - bbox: 数字在原图中的 (x1, y1, x2, y2)
        - confidence: 0~1 置信度
    """
    if binary_crop is None or binary_crop.size == 0:
        return None, None, 0.0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_crop, connectivity=8)
    if num_labels < 2:
        return None, None, 0.0

    H, W = binary_crop.shape
    candidates = []
    for j in range(1, num_labels):
        x = int(stats[j, cv2.CC_STAT_LEFT])
        y = int(stats[j, cv2.CC_STAT_TOP])
        w = int(stats[j, cv2.CC_STAT_WIDTH])
        h = int(stats[j, cv2.CC_STAT_HEIGHT])
        area = int(stats[j, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if w < 3 or h < 5:
            continue
        aspect = h / max(w, 1)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        # 偏向备选区下半：占比 = (y + h/2) / H
        y_center_ratio = (y + h / 2) / H
        # x 偏向右侧（zero_x 附近）
        x_center_ratio = (x + w / 2) / W
        candidates.append({
            'idx': j, 'x': x, 'y': y, 'w': w, 'h': h, 'area': area,
            'aspect': aspect, 'y_ratio': y_center_ratio, 'x_ratio': x_center_ratio,
        })

    if not candidates:
        return None, None, 0.0

    # 优先级：零线左侧 > 零线右侧；同侧选 x 更大（更靠近零线）
    candidates.sort(key=lambda c: (
        (c['x'] + c['w'] / 2 + x_offset) >= zero_x if zero_x is not None else False,
        -(c['x'] + c['w'] / 2)
    ))
    best = candidates[0]

    # 提取 patch（含 padding）
    pad = 2
    x1 = max(0, best['x'] - pad)
    y1 = max(0, best['y'] - pad)
    x2 = min(W, best['x'] + best['w'] + pad)
    y2 = min(H, best['y'] + best['h'] + pad)
    digit_crop = binary_crop[y1:y2, x1:x2]

    # bbox 映射到原图坐标
    bbox = (x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset)

    # confidence
    confidence = (
        0.4 * min(1.0, best['area'] / 200) +    # 面积分（~200 px² 是中等数字）
        0.3 * (1.0 - abs(best['aspect'] - 1.5) / 2.0) +  # 高宽比接近 1.5 最佳
        0.3 * best['y_ratio']  # 越靠下越好（数字贴刻度线）
    )
    confidence = max(0.0, min(1.0, confidence))

    return digit_crop, bbox, confidence


def _recognize_main_digits(region: dict,
                            color_region: np.ndarray,
                            long_tick_xs: List[int],
                            vproj_norm: np.ndarray = None) -> Tuple[List[DigitInfo], np.ndarray]:
    """识别主尺上的刻度数字，返回 (digits, debug_vis)"""
    reader = get_ocr_reader()
    img = region['image']
    color_zone = color_region

    digits, debug_vis = reader.read(
        img, color_zone, long_tick_xs,
        digit_zone='above',
        tick_infos=region.get('main_ticks', []),
        binary=region.get('binary'),
        vproj_norm=vproj_norm,
    )

    return digits, debug_vis


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
