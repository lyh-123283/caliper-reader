"""
步骤 5 — 读数合并 + 最终标注

合并主尺整数读数和游标小数读数，计算总读数和置信度，
在图像上绘制完整的标注结果。
"""

import cv2
import numpy as np
from typing import List

from .result import CaliperResult, DigitInfo


def merge_readings(main_result: dict,
                    vernier_result: dict,
                    rotated_color: np.ndarray,
                    region_main: dict,
                    region_vernier: dict,
                    split_y: int) -> CaliperResult:
    """
    合并主尺和游标尺读数，生成最终结果

    Args:
        main_result:     主尺识别结果
        vernier_result:  游标尺识别结果
        rotated_color:   旋转矫正后的彩色图（用于标注）
        region_main:     主尺区域信息
        region_vernier:  游标尺区域信息
        split_y:         分割线 y 坐标

    Returns:
        CaliperResult
    """
    main_ticks = main_result['main_ticks']
    main_gap = main_result['main_gap']
    main_digits = main_result['main_digits']
    vernier_ticks = vernier_result['vernier_ticks']
    vernier_digits = vernier_result.get('vernier_digits', [])
    precision = vernier_result['precision']
    vernier_reading = vernier_result['vernier_reading']
    zero_x = vernier_result['zero_x']

    # v6.5: 传 gray + binary 给 merger 让它做定向 OCR
    main_gray = region_main.get('image')
    main_binary = region_main.get('binary')

    # ── 计算主尺整数读数（附带推导诊断信息）──
    main_reading, main_derivation = _compute_main_reading_with_info(
        main_ticks, main_digits, main_gap, zero_x,
        gray_region=main_gray, binary_region=main_binary)

    # ── 游标对齐索引 ──
    aligned_tick = vernier_result.get('aligned_tick')
    vernier_aligned_idx = -1
    if aligned_tick and vernier_ticks:
        v_sorted = sorted(vernier_ticks, key=lambda t: t['x'])
        for i, vt in enumerate(v_sorted):
            if vt is aligned_tick or vt['x'] == aligned_tick['x']:
                vernier_aligned_idx = i
                break

    # ── 总读数 ──
    total = main_reading + vernier_reading

    # ── 置信度 ──
    confidence = calc_confidence(main_ticks, vernier_ticks, precision)

    # ── 最终标注图（含读数推导可视化）──
    annotated = draw_final_annotation(
        rotated_color, region_main, region_vernier,
        main_ticks, vernier_ticks, main_gap, split_y,
        main_reading, vernier_reading, total, precision,
        zero_x, aligned_tick, main_digits, vernier_digits,
        main_derivation, vernier_aligned_idx
    )

    # ── 读数推导专用可视化（单独一张调试图像）──
    derivation_vis = draw_reading_derivation(
        rotated_color, region_main, region_vernier,
        main_ticks, vernier_ticks, main_gap, split_y,
        main_reading, vernier_reading, total, precision,
        zero_x, aligned_tick, main_digits,
        main_derivation, vernier_aligned_idx,
        vernier_result.get('alignment_confidence', 0.0)
    )

    return CaliperResult(
        main_scale=main_reading,
        vernier_scale=vernier_reading,
        total=total,
        precision=precision,
        confidence=confidence,
        image_annotated=annotated,
        extra_info={
            'main_ticks_count': len(main_ticks),
            'vernier_ticks_count': len(vernier_ticks),
            'main_gap_px': main_gap,
            'zero_x': zero_x,
            'main_digits': [(d.text, d.value, d.x) for d in main_digits],
            'main_derivation': main_derivation,
            'derivation_vis': derivation_vis,
        },
    )


# ═══════════════════════════ 置信度计算 ═══════════════════════════

def calc_confidence(main_ticks: List[dict],
                     vernier_ticks: List[dict],
                     precision: float) -> float:
    """估算识别置信度"""
    s = 0.5

    # 主尺刻线数量
    if len(main_ticks) >= 10:
        s += 0.12
    elif len(main_ticks) >= 5:
        s += 0.06

    # 游标刻线数量 vs 期望值
    expected_v = max(int(1.0 / precision) if precision > 0 else 50, 10)
    v_ratio = min(len(vernier_ticks) / expected_v, 1.0)
    s += v_ratio * 0.2

    # 主尺刻线间距均匀性
    if len(main_ticks) >= 3:
        gaps = np.diff([t['x'] for t in sorted(main_ticks, key=lambda t: t['x'])])
        if len(gaps) > 2:
            cv_val = float(np.std(gaps) / np.mean(gaps)) if np.mean(gaps) > 0 else 1.0
            if cv_val < 0.15:
                s += 0.1
            elif cv_val < 0.3:
                s += 0.05

    return min(s, 0.95)


# ═══════════════════════════ 读数推导辅助 ═══════════════════════════

def _compute_main_reading_with_info(main_ticks: List[dict],
                                     main_digits: List[DigitInfo],
                                     main_gap: float,
                                     zero_x: float,
                                     gray_region: np.ndarray = None,
                                     binary_region: np.ndarray = None,
                                     main_color_region: np.ndarray = None) -> tuple:
    if main_gap <= 0:
        return _ocr_failed_reading('invalid_main_gap')

    main_xs = _dedupe_main_xs([t['x'] for t in main_ticks], main_gap) if main_ticks else []

    if main_xs and zero_x > 0 and gray_region is not None and binary_region is not None:
        from .ocr import get_ocr_reader_singleton
        from .main_scale import find_nearest_cm_digit_region, find_digit_cc_candidates

        binary_crop, x_off, y_off = find_nearest_cm_digit_region(
            main_ticks, main_gap, zero_x, binary_region)
        if binary_crop is None:
            return _ocr_failed_reading('no_digit_region')

        cc_candidates = find_digit_cc_candidates(binary_crop, x_off, y_off, zero_x)
        if not cc_candidates:
            return _ocr_failed_reading('no_digit_component')

        reader = get_ocr_reader_singleton()
        engine = reader.engine_status() if hasattr(reader, 'engine_status') else reader.engine_name()
        char_candidates = []
        for cc in cc_candidates:
            digit = reader.ocr_patch_to_digit(cc['digit_crop'], cc['bbox'], gray_region)
            if digit is None or digit.value < 0:
                continue
            char_candidates.append({
                'digit': digit,
                'value': digit.value,
                'text': digit.text,
                'confidence': digit.confidence,
                'bbox': cc['bbox'],
                'cc_confidence': cc['confidence'],
                'center_x': cc['center_x'],
                'source': 'single_char',
            })

        ocr_candidates = _group_main_ocr_labels(char_candidates, main_ticks, main_gap)
        if not ocr_candidates:
            return _ocr_failed_reading('ocr_no_digit', ocr_engine=engine)

        side_tol = max(4.0, main_gap * 0.20)
        usable = [c for c in ocr_candidates if c['ref_tick_x'] <= zero_x + side_tol]
        if not usable:
            return _ocr_failed_reading('no_ocr_digit_left_of_zero', ocr_engine=engine)
        selected = max(
            usable,
            key=lambda c: (c['ref_tick_x'], c['confidence'], c['cc_confidence'])
        )
        digit = selected['digit']
        ref_x = selected['ref_tick_x']
        extra_ticks = sum(1 for x in main_xs if ref_x + main_gap * 0.3 < x <= zero_x)
        reading = float(digit.value) * 10 + extra_ticks
        return reading, {
            'nearest_digit': digit,
            'extra_ticks': extra_ticks,
            'strategy': 'ocr',
            'ref_tick_x': ref_x,
            'ocr_text': digit.text,
            'ocr_confidence': digit.confidence,
            'ocr_engine': engine,
            'ocr_candidates': _summarize_ocr_candidates(ocr_candidates, selected),
        }

    return _ocr_failed_reading('missing_ocr_inputs')


def _ocr_failed_reading(reason: str,
                        ocr_engine: str = None) -> tuple:
    return 0.0, {
        'nearest_digit': None,
        'extra_ticks': 0,
        'strategy': 'ocr_failed',
        'ocr_reason': reason,
        'ocr_engine': ocr_engine,
    }


def _bind_digit_to_cm_tick(digit: DigitInfo,
                           main_ticks: List[dict],
                           main_gap: float) -> dict:
    if digit is None or not main_ticks or main_gap <= 0:
        return None
    return _bind_label_x_to_cm_tick(float(digit.x), main_ticks, main_gap, tolerance_ratio=0.65)


def _bind_label_x_to_cm_tick(label_x: float,
                             main_ticks: List[dict],
                             main_gap: float,
                             tolerance_ratio: float = 0.65) -> dict:
    if label_x is None or not main_ticks or main_gap <= 0:
        return None
    cm_ticks = _main_cm_ticks(main_ticks, main_gap)
    if not cm_ticks:
        return None
    center_x = float(label_x)
    nearest = min(cm_ticks, key=lambda t: abs(float(t['x']) - center_x))
    if abs(float(nearest['x']) - center_x) > max(8.0, main_gap * tolerance_ratio):
        return None
    return nearest


def _group_main_ocr_labels(char_candidates: list,
                           main_ticks: List[dict],
                           main_gap: float) -> list:
    if not char_candidates:
        return []

    chars = sorted(char_candidates, key=lambda c: (c.get('bbox') or (c.get('center_x', 0),))[0])
    grouped = []
    gap_tol = max(6.0, main_gap * 0.45)
    i = 0
    while i < len(chars):
        cur = chars[i]
        cur_bbox = cur.get('bbox')
        next_item = chars[i + 1] if i + 1 < len(chars) else None
        if cur.get('value') == 1 and cur_bbox and next_item and next_item.get('bbox'):
            nxt = next_item
            nxt_bbox = nxt['bbox']
            x_gap = float(nxt_bbox[0] - cur_bbox[2])
            combined_value = 10 + int(nxt.get('value', -99))
            if 0 <= x_gap <= gap_tol and 10 <= combined_value <= 15:
                bbox = (
                    min(cur_bbox[0], nxt_bbox[0]),
                    min(cur_bbox[1], nxt_bbox[1]),
                    max(cur_bbox[2], nxt_bbox[2]),
                    max(cur_bbox[3], nxt_bbox[3]),
                )
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0
                conf = min(float(cur.get('confidence', 0.0)),
                           float(nxt.get('confidence', 0.0)))
                cc_conf = min(float(cur.get('cc_confidence', 0.0)),
                              float(nxt.get('cc_confidence', 0.0)))
                grouped.append({
                    'digit': DigitInfo(
                        x=int(round(center_x)),
                        y=int(round(center_y)),
                        value=combined_value,
                        text=str(combined_value),
                        confidence=conf,
                        bbox=bbox,
                    ),
                    'value': combined_value,
                    'text': str(combined_value),
                    'confidence': conf,
                    'bbox': bbox,
                    'cc_confidence': cc_conf,
                    'center_x': center_x,
                    'source': 'grouped_2digit',
                    'children': [cur, nxt],
                })
                i += 2
                continue

        grouped.append(dict(cur))
        i += 1

    labels = []
    for label in grouped:
        tolerance_ratio = 1.2 if label.get('source') == 'grouped_2digit' else 0.65
        ref_tick = _bind_label_x_to_cm_tick(label.get('center_x'), main_ticks, main_gap, tolerance_ratio)
        if ref_tick is None:
            continue
        label = dict(label)
        label['ref_tick_x'] = float(ref_tick['x'])
        labels.append(label)
    return labels


def _main_cm_ticks(main_ticks: List[dict], main_gap: float) -> List[dict]:
    if not main_ticks:
        return []
    long_ticks = [t for t in main_ticks if t.get('is_long')]
    lengths = [float(t.get('length', 0)) for t in main_ticks]
    if not lengths:
        return []
    median_len = float(np.median(lengths))
    secondary = [t for t in main_ticks if float(t.get('length', 0)) >= median_len * 1.18]
    if len(secondary) > len(long_ticks):
        long_ticks = secondary
    if not long_ticks:
        return []

    tol = max(3.0, main_gap * 0.25)
    groups = []
    for tick in sorted(long_ticks, key=lambda t: float(t['x'])):
        if not groups or float(tick['x']) - float(groups[-1][-1]['x']) > tol:
            groups.append([tick])
        else:
            groups[-1].append(tick)

    deduped = []
    for group in groups:
        deduped.append(max(group, key=lambda t: float(t.get('length', 0))))
    return deduped


def _dedupe_main_xs(xs: list, main_gap: float) -> list:
    """Deduplicate nearby main-scale tick x positions before counting mm ticks."""
    if not xs:
        return []
    tol = max(2.0, main_gap * 0.25) if main_gap and main_gap > 0 else 3.0
    groups = []
    for x in sorted(float(v) for v in xs):
        if not groups or x - groups[-1][-1] > tol:
            groups.append([x])
        else:
            groups[-1].append(x)
    return [float(np.median(group)) for group in groups]


def _summarize_ocr_candidates(candidates: list, selected: dict = None) -> list:
    selected_ref = selected.get('ref_tick_x') if selected else None
    out = []
    for c in candidates:
        ref_tick_x = c.get('ref_tick_x')
        out.append({
            'text': c.get('text'),
            'value': c.get('value'),
            'confidence': c.get('confidence'),
            'bbox': c.get('bbox'),
            'center_x': c.get('center_x'),
            'ref_tick_x': ref_tick_x,
            'source': c.get('source'),
            'selected': selected_ref is not None and ref_tick_x is not None
                        and abs(ref_tick_x - selected_ref) < 1e-6,
        })
    return out






# ═══════════════════════════ 最终标注 ═══════════════════════════

def draw_final_annotation(rotated_color: np.ndarray,
                           region_main: dict,
                           region_vernier: dict,
                           main_ticks: List[dict],
                           vernier_ticks: List[dict],
                           main_gap: float,
                           split_y: int,
                           main_reading: float,
                           vernier_reading: float,
                           total: float,
                           precision: float,
                           zero_x: float,
                           aligned_tick: dict = None,
                           main_digits: List[DigitInfo] = None,
                           vernier_digits: List[DigitInfo] = None,
                           main_derivation: dict = None,
                           vernier_aligned_idx: int = -1) -> np.ndarray:
    """绘制最终标注图：全刻线 + 全数字 + 读数推导"""
    ann = rotated_color.copy()
    H, W = ann.shape[:2]

    # ── 区域分界线 ──
    cv2.line(ann, (0, split_y), (W, split_y), (200, 200, 200), 1)
    cv2.putText(ann, "MAIN", (6, split_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 100), 1)
    cv2.putText(ann, "VERNIER", (6, split_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)

    # ── 主尺刻线（绿色实线，长刻度更粗更亮）──
    for t in main_ticks:
        pt1 = (t['x'], t['y_start'] + region_main['y_offset'])
        pt2 = (t['x'], min(t['y_end'] + region_main['y_offset'], split_y - 1))
        if t.get('is_long'):
            cv2.line(ann, pt1, pt2, (0, 255, 80), 3)
        else:
            cv2.line(ann, pt1, pt2, (0, 160, 50), 2)

    # ── 游标刻线（橙色/黄色实线）──
    vy_off = region_vernier.get('y_offset', split_y)
    for t in vernier_ticks:
        pt1 = (t['x'], t['y_start'] + vy_off)
        pt2 = (t['x'], t['y_end'] + vy_off)
        if t.get('is_long'):
            cv2.line(ann, pt1, pt2, (0, 200, 255), 3)
        else:
            cv2.line(ann, pt1, pt2, (0, 150, 210), 2)

    # ── 主尺 OCR 数字标注 ──
    if main_digits:
        for d in main_digits:
            if d.value >= 0:
                dy = d.y + region_main['y_offset']
                dx = d.x
                _draw_label_box(ann, min(dx, W - 30), max(dy, 14), str(d.value),
                                (30, 50, 80), (100, 255, 160), 8, 0.55)

    # ── 游标 OCR 数字标注 ──
    if vernier_digits:
        for d in vernier_digits:
            if d.value >= 0:
                dy = d.y + vy_off
                dx = d.x
                _draw_label_box(ann, min(dx, W - 30), max(dy, split_y + 14), str(d.value),
                                (60, 40, 20), (255, 200, 100), 8, 0.55)

    # ── 零线（红色粗线，贯穿全图）──
    zx = int(zero_x)
    cv2.line(ann, (zx, 0), (zx, H - 1), (50, 50, 255), 3)
    cv2.putText(ann, "ZERO (0)", (zx + 5, split_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 255), 2)

    # ═══════════════════════════════════════════
    #  主尺读数推导标注 (增强版)
    # ═══════════════════════════════════════════
    if main_derivation and main_derivation.get('strategy') == 'ocr':
        nd = main_derivation['nearest_digit']
        extra = main_derivation['extra_ticks']
        dx, dy = nd.x, nd.y + region_main['y_offset']

        # 用到的 OCR 数字：半透明亮黄底色框 + 白色大字 + 描边
        pad = max(W // 60, 14)
        _draw_label_box(ann, dx, dy, str(nd.value),
                        (50, 180, 255), (255, 255, 255), pad, font_scale=0.7)

        # 高亮用到的参考长刻度线
        ref_x = int(main_derivation.get('ref_tick_x', dx))
        cv2.line(ann, (ref_x, 0), (ref_x, split_y), (50, 180, 255), 2, cv2.LINE_AA)

        # 从参考刻线 → 零线的粗线 + 大箭头
        arrow_y = split_y - 22
        _draw_arrow_between(ann, ref_x, zx, arrow_y, (50, 180, 255), thickness=2)

        # 刻度计数：洋红色大字 + 白描边
        mid_x = (ref_x + zx) // 2
        tick_label = f"+{extra} mm"
        _draw_label_box(ann, mid_x, arrow_y - 10, tick_label,
                        (180, 50, 180), (255, 255, 255), max(pad - 2, 10), font_scale=0.55)

        # 公式（顶部居中，深色半透明底条）
        formula_text = f"MAIN = {nd.value} + {extra} = {main_reading:.1f} mm"
        _draw_top_banner(ann, formula_text, (40, 160, 40))

    # ═══════════════════════════════════════════
    #  游标读数推导标注 (增强版)
    # ═══════════════════════════════════════════
    if aligned_tick and vernier_aligned_idx >= 0:
        ax = aligned_tick['x']
        ay = aligned_tick['y_mid'] + vy_off

        # 对齐刻线高亮：画一根粗的高亮竖条覆盖原刻线
        cv2.line(ann, (ax, aligned_tick['y_start'] + vy_off),
                 (ax, aligned_tick['y_end'] + vy_off), (0, 255, 80), 4)
        # 大圈
        cv2.circle(ann, (ax, ay), max(14, H // 30), (0, 255, 80), 3)
        # 圆圈内部填充半透明
        overlay = ann.copy()
        cv2.circle(overlay, (ax, ay), max(13, H // 30), (0, 80, 0), -1)

        # 标签
        _draw_label_box(ann, ax + 16, ay - 8,
                        f"ALIGNED i={vernier_aligned_idx}",
                        (0, 220, 80), (255, 255, 255), 12, font_scale=0.5)

        # 公式（游标区域底部居中）
        v_read = vernier_aligned_idx * precision
        v_text = f"VERNIER = {vernier_aligned_idx} x {precision:.2f} = {v_read:.2f} mm"
        _draw_bottom_banner(ann, v_text, split_y, (0, 180, 100))

    # ── 对齐标记 ──

    # ── 主尺网格线（浅灰短线）──
    if main_gap > 0:
        grid_offset = zero_x % main_gap
        for gx in np.arange(grid_offset, W, main_gap):
            gx = int(gx)
            if 0 <= gx < W:
                cv2.line(ann, (gx, split_y), (gx, split_y + 12),
                         (100, 100, 100), 1, cv2.LINE_AA)

    # ── 精度指示（右上角小标）──
    prec_text = f"precision: {precision:.2f} mm"
    (pw, ph), _ = cv2.getTextSize(prec_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    _draw_label_box(ann, W - pw // 2 - 8, ph + 6, prec_text,
                    (30, 30, 45), (180, 180, 190), 6, 0.45)

    # ── 底部结果面板 ──
    panel_h = 95
    cv2.rectangle(ann, (0, H - panel_h), (W, H), (12, 12, 22), -1)
    cv2.line(ann, (0, H - panel_h), (W, H - panel_h), (80, 80, 90), 2)

    cv2.putText(ann, f"Main:  {main_reading:.1f} mm", (15, H - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 2)
    cv2.putText(ann, f"Vernier: +{vernier_reading:.2f} mm", (15, H - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 230, 255), 1)

    # 总读数（大字 + 白描边）
    total_text = f"= {total:.2f} mm"
    (ttw, tth), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
    ttx, tty = W - ttw - 20, H - 48
    cv2.putText(ann, total_text, (ttx, tty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4)
    cv2.putText(ann, total_text, (ttx, tty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 255, 120), 3)

    return ann


# ═══════════════════════════ 可视化辅助函数 ═══════════════════════════

def _draw_label_box(img, cx, cy, text, bg_color, fg_color, pad, font_scale=0.55):
    """在 (cx,cy) 处画一个半透明圆角标签框 + 白描边文字"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)
    x1, y1 = cx - tw // 2 - pad, cy - th // 2 - pad
    x2, y2 = cx + tw // 2 + pad, cy + th // 2 + pad
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)

    # 半透明底框
    roi = img[y1:y2, x1:x2]
    overlay_box = np.full_like(roi, bg_color, dtype=np.uint8)
    cv2.rectangle(overlay_box, (0, 0), (x2 - x1, y2 - y1), bg_color, -1)
    img[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.45, overlay_box, 0.55, 0)

    # 白色描边文字
    tx, ty = cx - tw // 2, cy + th // 2
    cv2.putText(img, text, (tx, ty), font, font_scale, (255, 255, 255), 3)
    cv2.putText(img, text, (tx, ty), font, font_scale, fg_color, 2)


def _draw_arrow_between(img, x1, x2, y, color, thickness=2):
    """画水平粗线 + 箭头（两端都画，暗示方向）"""
    cv2.line(img, (x1, y), (x2, y), color, thickness, cv2.LINE_AA)
    # 右端箭头
    tip = 8
    cv2.arrowedLine(img, (x2 - tip, y), (x2, y), color, thickness + 1, tipLength=0.6)
    # 左端反向箭头
    cv2.arrowedLine(img, (x1 + tip, y), (x1, y), color, thickness + 1, tipLength=0.6)


def _draw_top_banner(img, text, color):
    """在图像顶部画半透明横幅 + 白色描边大字"""
    H, W = img.shape[:2]
    bh = max(28, H // 16)
    roi = img[:bh, :]
    overlay_b = np.full_like(roi, (15, 15, 20), dtype=np.uint8)
    img[:bh, :] = cv2.addWeighted(roi, 0.35, overlay_b, 0.65, 0)
    cv2.line(img, (0, bh), (W, bh), color, 2)
    # 白描边 + 彩色字
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.55
    (tw, th), _ = cv2.getTextSize(text, font, fs, 2)
    tx, ty = (W - tw) // 2, (bh + th) // 2
    cv2.putText(img, text, (tx, ty), font, fs, (255, 255, 255), 3)
    cv2.putText(img, text, (tx, ty), font, fs, color, 2)


def _draw_bottom_banner(img, text, split_y, color):
    """在游标区域（split_y 下方）底部画半透明横幅"""
    H, W = img.shape[:2]
    bh = max(28, H // 18)
    y0 = H - bh
    roi = img[y0:H, :]
    overlay_b = np.full_like(roi, (15, 15, 20), dtype=np.uint8)
    img[y0:H, :] = cv2.addWeighted(roi, 0.35, overlay_b, 0.65, 0)
    cv2.line(img, (0, y0), (W, y0), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    (tw, th), _ = cv2.getTextSize(text, font, fs, 2)
    tx, ty = (W - tw) // 2, y0 + (bh + th) // 2
    cv2.putText(img, text, (tx, ty), font, fs, (255, 255, 255), 3)
    cv2.putText(img, text, (tx, ty), font, fs, color, 2)


# ═══════════════════════════ 读数推导专用可视化 ═══════════════════════════

def draw_reading_derivation(rotated_color: np.ndarray,
                              region_main: dict,
                              region_vernier: dict,
                              main_ticks: List[dict],
                              vernier_ticks: List[dict],
                              main_gap: float,
                              split_y: int,
                              main_reading: float,
                              vernier_reading: float,
                              total: float,
                              precision: float,
                              zero_x: float,
                              aligned_tick: dict,
                              main_digits: List[DigitInfo],
                              main_derivation: dict,
                              vernier_aligned_idx: int,
                              align_conf: float) -> np.ndarray:
    """
    读数推导专用图：上半 = 主尺推导，下半 = 游标推导。
    深色背景 + 高对比度标注。
    """
    roi_main = rotated_color[:split_y, :].copy()
    roi_vernier = rotated_color[split_y:, :].copy()
    main_h, main_w = roi_main.shape[:2]
    vern_h, vern_w = roi_vernier.shape[:2]
    zx = int(zero_x)

    # ═══ 1. 主尺推导 ═══
    for t in main_ticks:
        y_end = min(t['y_end'] + region_main['y_offset'], split_y)
        c = (0, 255, 100) if t.get('is_long') else (0, 170, 70)
        cv2.line(roi_main, (t['x'], t['y_start'] + region_main['y_offset']),
                 (t['x'], y_end), c, 3 if t.get('is_long') else 2)

    # 零线
    cv2.line(roi_main, (zx, 0), (zx, main_h), (80, 80, 255), 3)
    _draw_label_box(roi_main, zx + 6, 18, "ZERO (0)", (80, 40, 40), (255, 255, 255), 8, 0.5)

    if main_derivation and main_derivation.get('strategy') == 'ocr':
        nd = main_derivation['nearest_digit']
        extra = main_derivation['extra_ticks']
        dx, dy = nd.x, nd.y + region_main['y_offset']
        ref_x = int(main_derivation.get('ref_tick_x', dx))

        _draw_label_box(roi_main, dx, dy, str(nd.value),
                        (50, 160, 255), (255, 255, 255), 14, 0.7)

        # 高亮参考刻度线
        cv2.line(roi_main, (ref_x, 0), (ref_x, main_h), (50, 160, 255), 2, cv2.LINE_AA)

        arrow_y = main_h - 30
        _draw_arrow_between(roi_main, ref_x, zx, arrow_y, (50, 160, 255), thickness=2)
        mid_x = (ref_x + zx) // 2
        _draw_label_box(roi_main, mid_x, arrow_y - 10,
                        f"+{extra} mm", (160, 40, 160), (255, 255, 255), 10, 0.55)

        _draw_top_banner(roi_main,
                         f"MAIN = {nd.value} + {extra} = {main_reading:.1f} mm",
                         (40, 180, 60))

    # ═══ 2. 游标推导 ═══
    vy_off = region_vernier.get('y_offset', split_y)
    for t in vernier_ticks:
        cv2.line(roi_vernier, (t['x'], t['y_start']),
                 (t['x'], t['y_end']), (0, 200, 220), 1)

    cv2.line(roi_vernier, (zx, 0), (zx, vern_h), (80, 80, 255), 3)

    # 主尺网格
    if main_gap > 0:
        go = zero_x % main_gap
        for gx in np.arange(go, vern_w, main_gap):
            gx = int(gx)
            if 0 <= gx < vern_w:
                cv2.line(roi_vernier, (gx, 0), (gx, 12), (140, 140, 140), 1, cv2.LINE_AA)

    if aligned_tick and vernier_aligned_idx >= 0:
        ax = aligned_tick['x']
        ay = aligned_tick['y_mid']
        cv2.line(roi_vernier, (ax, aligned_tick['y_start']),
                 (ax, aligned_tick['y_end']), (0, 255, 80), 4)
        r = max(14, vern_h // 10)
        cv2.circle(roi_vernier, (ax, ay), r, (0, 255, 80), 3)
        _draw_label_box(roi_vernier, ax + r + 10, ay - 4,
                        f"i={vernier_aligned_idx}", (0, 200, 80), (255, 255, 255), 10, 0.5)

    v_read = vernier_aligned_idx * precision if vernier_aligned_idx >= 0 else vernier_reading
    conf_str = f" conf={align_conf:.2f}" if align_conf > 0 else ""
    _draw_bottom_banner(roi_vernier,
                        f"VERNIER = {vernier_aligned_idx} x {precision:.2f}"
                        f" = {v_read:.2f} mm{conf_str}",
                        split_y, (0, 220, 120))

    # ═══ 3. 拼接 ═══
    gap = 4
    out_w = max(main_w, vern_w)
    out_h = main_h + vern_h + gap + 30
    vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    vis[:] = (15, 15, 22)
    vis[:main_h, :main_w] = roi_main
    vis[main_h + gap:main_h + gap + vern_h, :vern_w] = roi_vernier

    # 分隔线 + 标签
    sep_y = main_h + gap // 2
    cv2.line(vis, (0, sep_y), (out_w, sep_y), (80, 80, 90), 1)
    _draw_label_box(vis, out_w // 2, sep_y - 6, "▼ split line ▼",
                    (30, 30, 40), (180, 180, 180), 8, 0.4)

    # 总读数
    cv2.rectangle(vis, (0, out_h - 30), (out_w, out_h), (10, 10, 16), -1)
    cv2.line(vis, (0, out_h - 30), (out_w, out_h - 30), (100, 255, 100), 2)
    total_text = f"TOTAL = {main_reading:.1f} + {vernier_reading:.2f} = {total:.2f} mm"
    (ttw, tth), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(vis, total_text, ((out_w - ttw) // 2, out_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(vis, total_text, ((out_w - ttw) // 2, out_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 2)

    return vis
