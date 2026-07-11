"""
步骤 1 — ROI 提取 + 方向矫正

① ROI 提取：投影法定位刻度区域 → 裁剪（+ 轮廓验证回退）
② 方向矫正：HoughLinesP 检测刻线方向 → 旋转使刻线垂直
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Tuple, List

from .utils import rotate_image
from .config import config
from .vernier_rectify import _find_vernier_body_x_range

_ROI_SCREW_TEMPLATE_CACHE = None


def _read_image_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _load_roi_screw_template():
    global _ROI_SCREW_TEMPLATE_CACHE
    if _ROI_SCREW_TEMPLATE_CACHE is not None:
        return _ROI_SCREW_TEMPLATE_CACHE
    path = Path(__file__).resolve().parent.parent / 'templates' / 'roi_screw_template.png'
    img = _read_image_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        _ROI_SCREW_TEMPLATE_CACHE = False
        return None
    _ROI_SCREW_TEMPLATE_CACHE = img
    return img


def _locate_roi_by_screw_template(img_color: np.ndarray) -> dict:
    template_gray = _load_roi_screw_template()
    if template_gray is None or template_gray is False:
        return None
    if img_color is None or img_color.size == 0:
        return None

    timings = {}

    def mark(key: str, start_time: float):
        timings[key] = (time.perf_counter() - start_time) * 1000.0

    h, w = img_color.shape[:2]
    if h <= 0 or w <= 0:
        return None

    scale = min(1.0, 600.0 / float(w))
    t0 = time.perf_counter()
    if scale < 1.0:
        small_color = cv2.resize(
            img_color, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_LINEAR)
    else:
        small_color = img_color
    small_gray = cv2.cvtColor(small_color, cv2.COLOR_BGR2GRAY)
    tw = max(8, int(round(template_gray.shape[1] * scale)))
    th = max(8, int(round(template_gray.shape[0] * scale)))
    small_template = cv2.resize(template_gray, (tw, th), interpolation=cv2.INTER_AREA)
    mark('template_resize_gray', t0)

    t0 = time.perf_counter()
    search_gray, search_offset = _screw_template_search_window(small_gray)
    candidates = _screw_template_candidates(
        search_gray, small_template,
        scales=(1.0,),
        per_scale_k=12)
    if search_offset != (0, 0):
        ox, oy = search_offset
        for candidate in candidates:
            x, y = candidate['loc']
            candidate['loc'] = (x + ox, y + oy)
    mark('template_match', t0)
    if not candidates:
        return None

    t0 = time.perf_counter()
    pool = _nms_template_candidates(candidates, top_k=20, iou_thresh=0.25)
    geometry = _find_two_screw_rows(
        pool,
        y_tolerance=55.0 * scale,
        min_spacing=180.0 * scale,
        spacing_tolerance=0.45,
        row_gap_min=180.0 * scale,
        row_gap_max=850.0 * scale,
        x_align_ratio=0.45)
    mark('template_geometry', t0)
    if not _screw_geometry_is_valid(geometry):
        t0 = time.perf_counter()
        candidates = _screw_template_candidates(
            search_gray, small_template,
            scales=(0.8, 0.9, 1.0, 1.1, 1.2),
            per_scale_k=20)
        if search_offset != (0, 0):
            ox, oy = search_offset
            for candidate in candidates:
                x, y = candidate['loc']
                candidate['loc'] = (x + ox, y + oy)
        timings['template_match_fallback'] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        pool = _nms_template_candidates(candidates, top_k=50, iou_thresh=0.25)
        geometry = _find_two_screw_rows(
            pool,
            y_tolerance=55.0 * scale,
            min_spacing=180.0 * scale,
            spacing_tolerance=0.45,
            row_gap_min=180.0 * scale,
            row_gap_max=850.0 * scale,
            x_align_ratio=0.45)
        timings['template_geometry_fallback'] = (time.perf_counter() - t0) * 1000.0
        if not _screw_geometry_is_valid(geometry):
            return None

    t0 = time.perf_counter()
    polygon = _screw_roi_polygon(
        geometry,
        left_pad_ratio=1.10,
        right_pad_ratio=0.65,
        top_down_ratio=-0.08,
        bottom_pad_ratio=0.22)
    if polygon is None:
        return None

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    inv_scale = 1.0 / scale if scale > 0 else 1.0
    ox1 = max(0, int(np.floor(min(xs) * inv_scale)))
    oy1 = max(0, int(np.floor(min(ys) * inv_scale)))
    ox2 = min(w, int(np.ceil(max(xs) * inv_scale)))
    oy2 = min(h, int(np.ceil(max(ys) * inv_scale)))
    if ox2 - ox1 < config.roi.min_roi_width or oy2 - oy1 < config.roi.min_roi_height:
        return None
    crop = img_color[oy1:oy2, ox1:ox2].copy()
    mark('template_map_and_crop', t0)
    t0 = time.perf_counter()
    roi_debug = _make_roi_location_vis(
        small_color,
        (min(xs), min(ys), max(xs), max(ys)),
        crop,
        'screw_template'
    )
    mark('roi_debug_vis', t0)

    return {
        'roi_color': crop,
        'x_offset': ox1,
        'y_offset': oy1,
        'roi_box_original': (ox1, oy1, ox2, oy2),
        'roi_box_lowres': (min(xs), min(ys), max(xs), max(ys)),
        'roi_polygon_lowres': tuple(polygon),
        'scale': scale,
        'lowres_debug': roi_debug,
        'roi_timings': timings,
        'locate_failed': False,
        'roi_source': 'screw_template',
    }


def _screw_template_search_window(gray: np.ndarray) -> tuple:
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return gray, (0, 0)
    x1 = int(round(w * 0.25))
    x2 = int(round(w * 0.75))
    y1 = int(round(h * 0.25))
    y2 = int(round(h * 0.75))
    if x2 <= x1 or y2 <= y1:
        return gray, (0, 0)
    return gray[y1:y2, x1:x2], (x1, y1)


def _screw_geometry_is_valid(geometry: dict) -> bool:
    if geometry is None:
        return False
    items = geometry['top']['items'] + geometry['bottom']['items']
    scores = [float(candidate.get('score', 0.0)) for candidate in items]
    return len(scores) >= 6 and min(scores) >= 0.42


def _screw_template_candidates(target_gray: np.ndarray,
                               template_gray: np.ndarray,
                               scales: tuple,
                               per_scale_k: int) -> list:
    candidates = []
    th0, tw0 = template_gray.shape[:2]
    for scale in scales:
        tw = max(8, int(round(tw0 * scale)))
        th = max(8, int(round(th0 * scale)))
        if tw >= target_gray.shape[1] or th >= target_gray.shape[0]:
            continue
        resized = cv2.resize(template_gray, (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(target_gray, resized, cv2.TM_CCOEFF_NORMED)
        work = result.copy()
        suppress_x = max(1, tw // 2)
        suppress_y = max(1, th // 2)
        for _ in range(max(1, per_scale_k)):
            _, max_val, _, max_loc = cv2.minMaxLoc(work)
            candidates.append({
                'score': float(max_val),
                'loc': max_loc,
                'size': (tw, th),
                'scale': float(scale),
            })
            x, y = max_loc
            x1 = max(0, x - suppress_x)
            y1 = max(0, y - suppress_y)
            x2 = min(work.shape[1], x + suppress_x + 1)
            y2 = min(work.shape[0], y + suppress_y + 1)
            work[y1:y2, x1:x2] = -1.0
    return sorted(candidates, key=lambda item: item['score'], reverse=True)


def _template_box(candidate: dict) -> tuple:
    x, y = candidate['loc']
    tw, th = candidate['size']
    return int(x), int(y), int(x + tw), int(y + th)


def _template_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _nms_template_candidates(candidates: list, top_k: int, iou_thresh: float) -> list:
    selected = []
    for candidate in candidates:
        box = _template_box(candidate)
        if any(_template_iou(box, _template_box(prev)) > iou_thresh for prev in selected):
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def _template_center(candidate: dict) -> tuple:
    x1, y1, x2, y2 = _template_box(candidate)
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _find_screw_row_triples(candidates: list,
                            y_tolerance: float,
                            min_spacing: float,
                            spacing_tolerance: float) -> list:
    rows = []
    n = len(candidates)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triple = [candidates[i], candidates[j], candidates[k]]
                centers = [_template_center(c) for c in triple]
                ordered = sorted(zip(centers, triple), key=lambda item: item[0][0])
                xs = [item[0][0] for item in ordered]
                ys = [item[0][1] for item in ordered]
                y_spread = max(ys) - min(ys)
                if y_spread > y_tolerance:
                    continue
                dx1 = xs[1] - xs[0]
                dx2 = xs[2] - xs[1]
                if dx1 < min_spacing or dx2 < min_spacing:
                    continue
                spacing_ratio = abs(dx1 - dx2) / max(dx1, dx2, 1.0)
                if spacing_ratio > spacing_tolerance:
                    continue
                score = sum(1.0 - item[1]['score'] for item in ordered)
                score += (y_spread / max(y_tolerance, 1.0)) * 0.05
                score += spacing_ratio * 0.10
                rows.append({
                    'items': [item[1] for item in ordered],
                    'xs': xs,
                    'ys': ys,
                    'y': float(np.median(ys)),
                    'spacing': (dx1 + dx2) * 0.5,
                    'score': score,
                    'spacing_ratio': spacing_ratio,
                })
    return sorted(rows, key=lambda item: item['score'])


def _find_two_screw_rows(candidates: list,
                         y_tolerance: float,
                         min_spacing: float,
                         spacing_tolerance: float,
                         row_gap_min: float,
                         row_gap_max: float,
                         x_align_ratio: float):
    rows = _find_screw_row_triples(candidates, y_tolerance, min_spacing, spacing_tolerance)
    best = None
    row_limit = min(len(rows), 300)
    for a_idx in range(row_limit):
        for b_idx in range(a_idx + 1, row_limit):
            row_a = rows[a_idx]
            row_b = rows[b_idx]
            top, bottom = (row_a, row_b) if row_a['y'] <= row_b['y'] else (row_b, row_a)
            row_gap = bottom['y'] - top['y']
            if row_gap < row_gap_min or row_gap > row_gap_max:
                continue
            avg_spacing = max(1.0, (top['spacing'] + bottom['spacing']) * 0.5)
            x_align = float(np.median([abs(tx - bx) for tx, bx in zip(top['xs'], bottom['xs'])]))
            if x_align > avg_spacing * x_align_ratio:
                continue
            spacing_diff = abs(top['spacing'] - bottom['spacing']) / avg_spacing
            if spacing_diff > spacing_tolerance:
                continue
            pair_score = top['score'] + bottom['score']
            pair_score += (x_align / max(avg_spacing * x_align_ratio, 1.0)) * 0.20
            pair_score += spacing_diff * 0.15
            pair_score += (row_gap / max(row_gap_max, 1.0)) * 0.02
            if best is None or pair_score < best['score']:
                best = {
                    'top': top,
                    'bottom': bottom,
                    'score': pair_score,
                    'row_gap': row_gap,
                    'x_align': x_align,
                    'spacing_diff': spacing_diff,
                }
    return best


def _screw_roi_polygon(geometry: dict,
                       left_pad_ratio: float,
                       right_pad_ratio: float,
                       top_down_ratio: float,
                       bottom_pad_ratio: float):
    top_centers = [np.array(_template_center(c), dtype=np.float32) for c in geometry['top']['items']]
    bottom_centers = [np.array(_template_center(c), dtype=np.float32) for c in geometry['bottom']['items']]
    all_centers = top_centers + bottom_centers
    row_vec = ((top_centers[-1] - top_centers[0]) +
               (bottom_centers[-1] - bottom_centers[0])) * 0.5
    row_norm = float(np.linalg.norm(row_vec))
    if row_norm < 1e-6:
        return None
    x_axis = row_vec / row_norm
    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)
    row_down = np.mean(bottom_centers, axis=0) - np.mean(top_centers, axis=0)
    if float(np.dot(y_axis, row_down)) < 0:
        y_axis = -y_axis

    us = [float(np.dot(p, x_axis)) for p in all_centers]
    top_vs = [float(np.dot(p, y_axis)) for p in top_centers]
    bottom_vs = [float(np.dot(p, y_axis)) for p in bottom_centers]
    left_u = min(us)
    right_u = max(us)
    top_v = float(np.median(top_vs))
    bottom_v = float(np.median(bottom_vs))
    spacing = max(1.0, (geometry['top']['spacing'] + geometry['bottom']['spacing']) * 0.5)
    row_gap = max(1.0, bottom_v - top_v)

    widths = []
    heights = []
    for candidate in geometry['top']['items'] + geometry['bottom']['items']:
        x1, y1, x2, y2 = _template_box(candidate)
        widths.append(x2 - x1)
        heights.append(y2 - y1)
    half_w = float(np.median(widths)) * 0.5
    half_h = float(np.median(heights)) * 0.5

    u_min = left_u - spacing * left_pad_ratio - half_w
    u_max = right_u + spacing * right_pad_ratio + half_w
    v_min = top_v + row_gap * top_down_ratio - half_h
    v_max = bottom_v + row_gap * bottom_pad_ratio + half_h

    corners = []
    for u, v in [(u_min, v_min), (u_max, v_min), (u_max, v_max), (u_min, v_max)]:
        p = x_axis * u + y_axis * v
        corners.append((int(round(float(p[0]))), int(round(float(p[1])))))
    return corners


# ═══════════════════════════════════════════════════════════
#  ① ROI 提取：投影法定位刻度区域
# ═══════════════════════════════════════════════════════════

def locate_roi_lowres(img_color: np.ndarray,
                      max_width: int = 1600) -> dict:
    template_result = _locate_roi_by_screw_template(img_color)
    if template_result is not None:
        return template_result

    timings = {}

    def mark(key: str, start_time: float):
        timings[key] = (time.perf_counter() - start_time) * 1000.0

    h, w = img_color.shape[:2]
    scale = min(1.0, float(max_width) / float(w)) if w > 0 else 1.0
    t0 = time.perf_counter()
    full_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    mark('gray_full', t0)

    t0 = time.perf_counter()
    if scale < 1.0:
        gray = cv2.resize(
            full_gray, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_LINEAR)
    else:
        gray = full_gray.copy()
    mark('resize_gray_linear', t0)

    sh, sw = gray.shape[:2]
    t0 = time.perf_counter()
    enhanced = _make_lowres_roi_enhanced(gray)
    mark('enhance_gamma_clahe', t0)

    t0 = time.perf_counter()
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=_odd_between(min(sh, sw) // 18, 15, 51),
        C=9,
    )
    fg = cv2.bitwise_not(binary)
    mark('adaptive_threshold', t0)

    t0 = time.perf_counter()
    y1, y2 = _proj_find_y_range(fg, sh)
    mark('horizontal_projection', t0)

    t0 = time.perf_counter()
    x1, x2, x_diag = _proj_find_x_range(fg, y1, y2, sw)
    mark('vertical_projection', t0)
    if y2 - y1 < config.roi.min_roi_height or x2 - x1 < config.roi.min_roi_width:
        return _lowres_roi_failure(timings, (x1, y1, x2, y2), scale)

    t0 = time.perf_counter()
    refined = _refine_roi_by_vernier_block(enhanced, y1, y2, x1, x2, timings)
    if refined is not None:
        y1, y2, x1, x2 = refined
    mark('refine_vernier_block', t0)

    t0 = time.perf_counter()
    reading_refined = _refine_roi_to_reading_window(enhanced, y1, y2, x1, x2, x_diag)
    if reading_refined is not None:
        y1, y2, x1, x2 = reading_refined
    mark('refine_reading_window', t0)

    t0 = time.perf_counter()
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
        mark('map_and_crop', t0)
        return _lowres_roi_failure(timings, (x1, y1, x2, y2), scale)

    crop = img_color[oy1:oy2, ox1:ox2].copy()
    mark('map_and_crop', t0)
    t0 = time.perf_counter()
    roi_debug = _make_roi_location_vis(
        img_color, (ox1, oy1, ox2, oy2), crop, 'lowres_projection'
    )
    mark('roi_debug_vis', t0)
    return {
        'roi_color': crop,
        'x_offset': ox1,
        'y_offset': oy1,
        'roi_box_original': (ox1, oy1, ox2, oy2),
        'roi_box_lowres': (x1, y1, x2, y2),
        'scale': scale,
        'lowres_debug': roi_debug,
        'roi_timings': timings,
        'locate_failed': False,
        'roi_source': 'lowres_projection',
    }


def _make_roi_location_vis(img_color: np.ndarray,
                           roi_box: tuple,
                           roi_crop: np.ndarray,
                           source: str) -> np.ndarray:
    if img_color is None or roi_box is None:
        return None
    h, w = img_color.shape[:2]
    if h <= 0 or w <= 0:
        return None

    top_w = 640
    scale = min(1.0, top_w / float(w))
    view_w = max(1, int(round(w * scale)))
    view_h = max(1, int(round(h * scale)))
    if abs(scale - 1.0) < 1e-6:
        overview = img_color.copy()
    else:
        overview = cv2.resize(img_color, (view_w, view_h), interpolation=cv2.INTER_AREA)

    x1, y1, x2, y2 = roi_box
    p1 = (int(round(x1 * scale)), int(round(y1 * scale)))
    p2 = (int(round(x2 * scale)), int(round(y2 * scale)))
    cv2.rectangle(overview, p1, p2, (0, 255, 120), 3, cv2.LINE_AA)
    cv2.putText(
        overview, f"ROI: {source}", (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        overview, f"ROI: {source}", (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
    )

    if roi_crop is None or roi_crop.size == 0:
        return overview
    ch, cw = roi_crop.shape[:2]
    crop_w = view_w
    crop_h = max(1, int(round(ch * (crop_w / float(max(cw, 1))))))
    crop_h = min(crop_h, 160)
    crop_w = max(1, int(round(cw * (crop_h / float(max(ch, 1))))))
    crop_view = cv2.resize(roi_crop, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

    gap = 8
    out_w = max(view_w, crop_w)
    out_h = view_h + gap + crop_h
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    out[:] = (28, 28, 32)
    out[:view_h, :view_w] = overview
    crop_x = (out_w - crop_w) // 2
    out[view_h + gap:view_h + gap + crop_h, crop_x:crop_x + crop_w] = crop_view
    cv2.rectangle(
        out, (crop_x, view_h + gap),
        (crop_x + crop_w - 1, view_h + gap + crop_h - 1),
        (0, 255, 120), 2, cv2.LINE_AA
    )
    return out


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


def _lowres_roi_failure(timings: dict,
                        box: tuple,
                        scale: float) -> dict:
    return {
        'roi_color': None,
        'x_offset': 0,
        'y_offset': 0,
        'roi_box_original': None,
        'roi_box_lowres': box,
        'scale': scale,
        'lowres_debug': None,
        'roi_timings': timings,
        'locate_failed': True,
    }


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
                                   x1: int, x2: int,
                                   timings: dict = None):
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
    t0 = time.perf_counter()
    pad_x = max(50, int((x2 - x1) * 0.50))
    pad_y = max(50, int((y2 - y1) * 0.35))
    edge_x1 = max(0, x1 - pad_x)
    edge_x2 = min(W - 1, x2 + pad_x)
    edge_y1 = max(0, y1 - pad_y)
    edge_y2 = min(H - 1, y2 + pad_y)
    edge_crop = enhanced[edge_y1:edge_y2 + 1, edge_x1:edge_x2 + 1]

    t0 = time.perf_counter()
    edge_bw = _make_sobel_y_binary(edge_crop)
    if timings is not None:
        timings['refine_make_edge_map'] = (time.perf_counter() - t0) * 1000.0
    if edge_bw is None:
        return None

    t0 = time.perf_counter()
    y_edge_x1, y_edge_x2 = _select_y_edge_x_window(x1, x2, W)
    y_edge_x1 = max(0, y_edge_x1 - edge_x1)
    y_edge_x2 = min(edge_bw.shape[1] - 1, y_edge_x2 - edge_x1)
    if timings is not None:
        timings['refine_select_y_edge_window'] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    search_y1 = max(0, y1 - edge_y1)
    search_y2 = min(edge_bw.shape[0] - 1, y2 - edge_y1)
    new_y1, new_y2 = _find_caliper_y_edges(
        edge_bw, y_edge_x1, y_edge_x2, search_y1, search_y2
    )
    if timings is not None:
        timings['refine_find_y_edges'] = (time.perf_counter() - t0) * 1000.0
    if new_y1 is None or new_y2 is None or new_y2 - new_y1 < 80:
        new_y1, new_y2 = y1, y2
    else:
        new_y1 += edge_y1
        new_y2 += edge_y1
        # v6.4: 给下方留出游标尺数字行空间（约 30% 卡尺高度）
        caliper_h = new_y2 - new_y1
        extra_below = int(caliper_h * 0.35)
        new_y2 = min(H - 1, new_y2 + extra_below)

    # ── 2. x 右边界：用 y 上沿水平边缘的右端 x 作为压块右缘 ──
    t0 = time.perf_counter()
    right_y1 = max(0, new_y1 - edge_y1)
    right_y2 = min(edge_bw.shape[0] - 1, new_y2 - edge_y1)
    right_x1 = max(0, x1 - edge_x1)
    new_x2 = _find_caliper_right_edge(edge_bw, right_y1, right_y2, right_x1)
    if new_x2 is not None:
        new_x2 += edge_x1
    if timings is not None:
        timings['refine_find_right_edge'] = (time.perf_counter() - t0) * 1000.0
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


def _make_sobel_y_binary(enhanced: np.ndarray):
    if enhanced is None:
        return None
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=5)
    abs_y = np.abs(sobel_y)
    g_max = float(np.max(abs_y))
    if g_max <= 0:
        return None
    abs_y_u8 = (abs_y / g_max * 255).astype(np.uint8)
    _, bw = cv2.threshold(abs_y_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _find_caliper_y_edges(edge_bw: np.ndarray,
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
    if edge_bw is None:
        return None, None
    H, W = edge_bw.shape[:2]
    x_lo = max(0, x_lo); x_hi = min(W - 1, x_hi)
    y_search_lo = max(0, y_search_lo); y_search_hi = min(H - 1, y_search_hi)

    strip = edge_bw[:, x_lo:x_hi + 1]
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


def _find_caliper_right_edge(edge_bw: np.ndarray,
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
    if edge_bw is None:
        return None
    H, W = edge_bw.shape[:2]
    y_lo = max(0, y_lo); y_hi = min(H - 1, y_hi)
    x_lo = max(0, x_lo)
    if y_hi - y_lo < 50:
        return None

    # 只看 y 范围内的行
    strip = edge_bw[y_lo:y_hi + 1, :]

    mask = strip == 255
    row_has_edge = np.any(mask[:, x_lo + 1:], axis=1) if x_lo + 1 < W else np.zeros(mask.shape[0], dtype=bool)
    if not np.any(row_has_edge):
        return None

    reversed_idx = np.argmax(mask[row_has_edge, ::-1], axis=1)
    right_xs = W - 1 - reversed_idx
    right_xs = right_xs[right_xs > x_lo]
    if right_xs.size == 0:
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


def _estimate_seam_angle_ransac(gray: np.ndarray,
                                binary: np.ndarray) -> Tuple[float, dict]:
    h, w = gray.shape[:2]
    if h < 40 or w < 160:
        return None, {}

    fg = _white_foreground_binary(binary, gray)
    seeds = _seam_seed_candidates(gray, fg)
    results = []
    for seed_y in seeds:
        result = _detect_seam_line_for_seed(gray, seed_y)
        if result is not None:
            results.append(result)
            if _is_high_quality_seam(result):
                angle = float(result['angle'])
                diag = {
                    'method': 'seam_ransac',
                    'seeds': [int(v) for v in seeds],
                    'candidates': [_summarize_seam_candidate(r) for r in results],
                    'early_stop': True,
                }
                diag.update(_summarize_seam_candidate(result))
                diag['raw_angle'] = angle
                return angle, diag

    diag = {
        'method': 'seam_ransac',
        'seeds': [int(v) for v in seeds],
        'candidates': [_summarize_seam_candidate(r) for r in results],
        'early_stop': False,
    }
    valid = [r for r in results if r.get('valid')]
    if not valid:
        diag['reason'] = 'no_valid_seam'
        return None, diag

    best = max(valid, key=lambda r: r['rank'])
    angle = float(best['angle'])
    diag.update(_summarize_seam_candidate(best))
    diag['raw_angle'] = angle
    if abs(angle) > 2.0:
        diag['reason'] = 'angle_out_of_range'
        return None, diag
    return angle, diag


def _is_high_quality_seam(result: dict) -> bool:
    return (
        bool(result.get('valid')) and
        int(result.get('kept', 0)) >= 160 and
        float(result.get('coverage', 0.0)) >= 0.55 and
        float(result.get('mad', 999.0)) <= 0.25 and
        abs(float(result.get('angle', 999.0))) <= 2.0
    )


def _seam_seed_candidates(gray: np.ndarray,
                          fg: np.ndarray) -> List[int]:
    h, w = gray.shape[:2]
    seeds = []
    try:
        from .region_split import _split_by_vernier_tick_band, _split_by_gray_seam
        first = _split_by_vernier_tick_band(gray, fg, h, w)
        second = _split_by_gray_seam(gray, h, w)
        if first is not None:
            seeds.append(int(first))
        if second is not None:
            seeds.append(int(second))
    except Exception:
        pass
    seeds.extend([int(h * 0.58), int(h * 0.64), int(h * 0.70)])

    out = []
    for seed in seeds:
        seed = max(0, min(h - 1, int(seed)))
        if all(abs(seed - old) > 15 for old in out):
            out.append(seed)
    return out


def _detect_seam_line_for_seed(gray: np.ndarray,
                               seed_y: int) -> dict:
    h, w = gray.shape[:2]
    band_half = max(35, int(h * 0.08))
    y1 = max(0, int(seed_y) - band_half)
    y2 = min(h, int(seed_y) + band_half)
    crop = gray[y1:y2, :]
    if crop.size == 0:
        return None

    blur = cv2.GaussianBlur(crop, (3, 3), 0)
    score = np.abs(cv2.Scharr(blur, cv2.CV_32F, 0, 1))
    score = cv2.blur(score, (max(9, (w // 180) | 1), 1))
    threshold = max(6.0, float(np.percentile(score, 83)))
    step = max(6, w // 320)
    margin = max(20, w // 80)

    points = []
    for x0 in range(margin, w - margin, step):
        x1b = max(0, x0 - step)
        x2b = min(w, x0 + step + 1)
        col = np.mean(score[:, x1b:x2b], axis=1)
        for yi, val in _local_1d_peaks(col, min_sep=6, top_k=2):
            if val >= threshold:
                points.append((float(x0), float(yi + y1), float(val)))

    if len(points) < 50:
        return {
            'seed_y': int(seed_y), 'band_y1': y1, 'band_y2': y2,
            'points': len(points), 'valid': False, 'reason': 'few_points'
        }

    xs = np.asarray([p[0] for p in points], dtype=float)
    ys = np.asarray([p[1] for p in points], dtype=float)
    weights = np.asarray([p[2] for p in points], dtype=float)
    best = _ransac_seam_line(xs, ys, weights, seed_y, h, w)
    if best is None:
        return {
            'seed_y': int(seed_y), 'band_y1': y1, 'band_y2': y2,
            'points': len(points), 'valid': False, 'reason': 'no_ransac'
        }

    a, b, keep = best
    for _ in range(3):
        fit = _weighted_line_fit(xs[keep], ys[keep], weights[keep])
        if fit is None:
            break
        a, b = fit
        resid = ys - (a * xs + b)
        good_resid = resid[keep]
        mad0 = _median_abs_dev(good_resid)
        new_keep = np.abs(resid) <= max(max(3.0, h * 0.004), 3.5 * mad0)
        if np.count_nonzero(new_keep) < 40:
            break
        keep = new_keep

    resid = ys - (a * xs + b)
    in_res = np.abs(resid[keep])
    mad = _median_abs_dev(in_res) if in_res.size else 999.0
    kept = int(np.count_nonzero(keep))
    coverage = float((np.max(xs[keep]) - np.min(xs[keep])) / max(w, 1)) if kept else 0.0
    angle = float(np.degrees(np.arctan(a)))
    mid_y = float(a * (w * 0.5) + b)
    valid = (
        kept >= 70 and coverage >= 0.45 and mad <= 4.0 and
        abs(angle) <= 2.0 and 0.48 * h <= mid_y <= 0.78 * h
    )
    rank = float(np.sum(weights[keep]) * coverage / (1.0 + mad)) if valid else -1.0
    return {
        'seed_y': int(seed_y),
        'band_y1': y1,
        'band_y2': y2,
        'points': len(points),
        'kept': kept,
        'coverage': coverage,
        'mad': mad,
        'angle': angle,
        'mid_y': mid_y,
        'rank': rank,
        'valid': valid,
        'line': (float(a), float(b)),
        'threshold': threshold,
    }


def _local_1d_peaks(values: np.ndarray,
                    min_sep: int = 5,
                    top_k: int = 3) -> List[Tuple[int, float]]:
    if values is None or len(values) < 3:
        return []
    candidates = []
    for i in range(1, len(values) - 1):
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            candidates.append((float(values[i]), i))
    candidates.sort(reverse=True)
    peaks = []
    used = []
    for val, idx in candidates:
        if all(abs(idx - old) >= min_sep for old in used):
            peaks.append((idx, val))
            used.append(idx)
            if len(peaks) >= top_k:
                break
    return peaks


def _ransac_seam_line(xs: np.ndarray,
                      ys: np.ndarray,
                      weights: np.ndarray,
                      seed_y: int,
                      h: int,
                      w: int):
    if len(xs) < 2:
        return None
    rng = np.random.default_rng(1234)
    tol = max(3.0, h * 0.004)
    mid_x = w * 0.5
    n = len(xs)
    sample_count = 450
    i = rng.integers(0, n, size=sample_count)
    j = rng.integers(0, n, size=sample_count)
    valid_pair = (i != j) & (np.abs(xs[i] - xs[j]) >= w * 0.15)
    if not np.any(valid_pair):
        return None

    i = i[valid_pair]
    j = j[valid_pair]
    a = (ys[j] - ys[i]) / (xs[j] - xs[i])
    angle = np.degrees(np.arctan(a))
    b = ys[i] - a * xs[i]
    mid_y = a * mid_x + b
    seed_tol = max(35, int(h * 0.08)) * 0.75
    valid = (
        (np.abs(angle) <= 2.0) &
        (mid_y >= 0.48 * h) &
        (mid_y <= 0.78 * h) &
        (np.abs(mid_y - seed_y) <= seed_tol)
    )
    if not np.any(valid):
        return None

    a = a[valid]
    b = b[valid]
    resid = np.abs(ys[None, :] - (a[:, None] * xs[None, :] + b[:, None]))
    keep = resid <= tol
    counts = np.count_nonzero(keep, axis=1)
    valid_rows = counts >= 40
    if not np.any(valid_rows):
        return None

    keep = keep[valid_rows]
    a = a[valid_rows]
    b = b[valid_rows]
    x_kept = np.where(keep, xs[None, :], np.nan)
    x_min = np.nanmin(x_kept, axis=1)
    x_max = np.nanmax(x_kept, axis=1)
    coverage = (x_max - x_min) / max(w, 1)
    valid_rows = coverage >= 0.35
    if not np.any(valid_rows):
        return None

    keep = keep[valid_rows]
    a = a[valid_rows]
    b = b[valid_rows]
    coverage = coverage[valid_rows]
    score = np.sum(keep * weights[None, :], axis=1) * coverage
    best_idx = int(np.argmax(score))
    return float(a[best_idx]), float(b[best_idx]), keep[best_idx]


def _weighted_line_fit(xs: np.ndarray,
                       ys: np.ndarray,
                       weights: np.ndarray):
    if len(xs) < 2:
        return None
    try:
        coef = np.polyfit(xs, ys, 1, w=np.maximum(weights, 1e-3))
        return float(coef[0]), float(coef[1])
    except Exception:
        return None


def _median_abs_dev(values: np.ndarray) -> float:
    if values is None or len(values) == 0:
        return 999.0
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _summarize_seam_candidate(result: dict) -> dict:
    keys = (
        'seed_y', 'band_y1', 'band_y2', 'points', 'kept',
        'coverage', 'mad', 'angle', 'mid_y', 'rank', 'valid', 'reason'
    )
    return {k: result[k] for k in keys if k in result}


def orient_caliper(roi_color: np.ndarray,
                    roi_gray: np.ndarray,
                    roi_binary: np.ndarray = None,
                    make_debug: bool = True) -> dict:
    """
    检测刻度线主导方向，旋转图像使刻线垂直。

    Returns:
        dict: rotated_color, rotated_gray, rotated_binary, orient_angle, orient_vis
    """
    gray = roi_gray if roi_gray is not None else cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    angle_scale = float(getattr(config.orient, 'angle_detection_scale', 1.0) or 1.0)
    if 0 < angle_scale < 0.999:
        h, w = gray.shape[:2]
        small_w = max(1, int(round(w * angle_scale)))
        small_h = max(1, int(round(h * angle_scale)))
        angle_gray = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        angle_binary = (
            cv2.resize(roi_binary, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            if roi_binary is not None else None
        )
        seam_angle, seam_diag = _estimate_seam_angle_ransac(angle_gray, angle_binary)
        if isinstance(seam_diag, dict):
            seam_diag = dict(seam_diag)
            seam_diag['angle_detection_scale'] = angle_scale
    else:
        seam_angle, seam_diag = _estimate_seam_angle_ransac(gray, roi_binary)
    if seam_angle is None:
        seam_angle = 0.0
    angle = float(seam_angle)

    rotated_color = rotate_image(roi_color, angle)
    rotated_gray = rotate_image(gray, angle)
    rotated_binary = rotate_image(roi_binary, angle) if roi_binary is not None else None

    orient_diag = {
        'method': 'seam_ransac',
        'seam_angle': float(seam_angle),
        'residual_angle': None,
        'final_angle': float(angle),
        'seam': seam_diag,
        'residual': None,
    }
    orient_vis = _make_orient_vis(roi_color, rotated_color, angle) if make_debug else None

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
