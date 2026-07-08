"""
步骤 2 — 区域分离：主尺 / 游标尺

策略（v4 — 投影突变法优先）：
  ROI 矫正后刻线垂直，主尺与游标尺之间是深色分界带。
  直接做水平像素投影，找中间区域最深的波谷（突变点）即为分界线。

  方案A（优先）：水平像素投影 → 找中间区域最深波谷（突变点）
    原理：刻度区像素值高（亮），分界带像素值低（暗），投影曲线在分界处骤降。
  方案B（回退）：灰度图 Sobel Y 方向梯度 → 水平投影 → 找最大梯度峰
    原理：黑线上下沿产生强烈的垂直梯度，投影后形成峰值。
  方案C（回退）：二值图 + 宽水平核闭运算 → 连通刻线 → 投影找窄谷
    原理：用超宽核把刻线连成片，黑线变成唯一的水平断点。
"""

import cv2
import numpy as np

from .utils import draw_projection_plot
from .config import config


def split_scales(rotated_gray: np.ndarray,
                  rotated_binary: np.ndarray = None,
                  rotated_color: np.ndarray = None) -> dict:
    """沿 y 轴切分主尺和游标尺区域"""
    h, w = rotated_gray.shape

    # 准备二值图（验证刻线密度用）
    if rotated_binary is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(rotated_gray)
        _, binary = cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = rotated_binary

    # Use the physical seam itself as the split.  Downstream recognition should
    # work from the true main/vernier boundary instead of an artificially
    # shifted crop line.
    split_y = _split_by_vernier_tick_band(rotated_gray, binary, h, w)
    if split_y is None:
        split_y = _split_by_gray_seam(rotated_gray, h, w)
    if split_y is None:
        split_y = _split_by_candidate_scan(rotated_gray, binary, h, w)

    # ── 最终回退（基于物理先验：主尺约占ROI高度的60%）──
    if split_y is None:
        split_y = int(h * config.region_split.fallback_split_ratio)

    # ── 游标区域高度校验：不能太小（至少占ROI的 min_ratio）──
    min_vernier_h = int(h * config.region_split.min_vernier_height_ratio)
    if h - split_y < min_vernier_h:
        split_y = h - min_vernier_h

    band_info = _analyze_horizontal_tick_bands(rotated_gray, binary, split_y)
    main_band = band_info.get('main_tick_band', (max(0, split_y - max(24, h // 3)), split_y))
    vernier_band = band_info.get('vernier_tick_band', (split_y, min(h, split_y + max(24, h // 4))))

    # ── 切分 ──
    img_upper = rotated_gray[:split_y, :]
    img_lower = rotated_gray[split_y:, :]
    bin_upper = binary[:split_y, :]
    bin_lower = binary[split_y:, :]

    main_band_local = (max(0, main_band[0]), min(split_y, main_band[1]))
    vernier_band_local = (
        max(0, vernier_band[0] - split_y),
        min(h - split_y, vernier_band[1] - split_y),
    )
    region_main = {
        'image': img_upper, 'binary': bin_upper,
        'y_offset': 0, 'height': split_y,
        'tick_band': main_band_local,
        'tick_band_global': main_band,
    }
    region_vernier = {
        'image': img_lower, 'binary': bin_lower,
        'y_offset': split_y, 'height': h - split_y,
        'tick_band': vernier_band_local,
        'tick_band_global': vernier_band,
    }

    split_vis = _make_split_vis(rotated_color if rotated_color is not None
                                  else rotated_gray,
                                  rotated_gray, binary, split_y, band_info)

    return {
        'region_main': region_main,
        'region_vernier': region_vernier,
        'split_y': split_y,
        'tick_bands': band_info,
        'split_vis': split_vis,
    }


def _split_by_vernier_tick_band(gray: np.ndarray, binary: np.ndarray,
                                h: int, w: int):
    """用中心区域梯度最大处定位分界线。"""
    lo, hi = int(h * 0.42), int(h * 0.84)
    if hi <= lo:
        return None

    x1, x2 = int(w * 0.28), int(w * 0.70)
    if x2 <= x1:
        x1, x2 = 0, w

    row_mean = np.mean(gray[:, x1:x2], axis=1).astype(float)
    win = max(7, h // 70)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / win
    smooth = np.convolve(row_mean, kernel, mode='same')
    grad = np.abs(np.gradient(smooth))

    # 只取搜索窗口内的梯度
    grad[0:lo] = 0
    grad[hi:] = 0

    best_y = int(np.argmax(grad))

    if float(grad[best_y]) < 2.0:
        return None

    return best_y


def _split_by_gray_seam(gray: np.ndarray, h: int, w: int):
    if gray is None or h < 20 or w < 20:
        return None

    lo, hi = int(h * 0.38), int(h * 0.78)
    if hi <= lo:
        return None

    x1, x2 = int(w * 0.18), int(w * 0.86)
    if x2 - x1 < max(20, w * 0.20):
        x1, x2 = 0, w

    crop = gray[:, x1:x2]
    row_mean = np.mean(crop, axis=1).astype(float)
    scharr_y = cv2.Scharr(crop, cv2.CV_32F, 0, 1)
    row_edge = np.mean(np.abs(scharr_y), axis=1).astype(float)

    win = max(5, h // 90)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / win
    smooth_mean = np.convolve(row_mean, kernel, mode='same')
    smooth_edge = np.convolve(row_edge, kernel, mode='same')

    mean_win = smooth_mean[lo:hi + 1]
    edge_win = smooth_edge[lo:hi + 1]
    if mean_win.size == 0 or edge_win.size == 0:
        return None

    dark = float(np.max(mean_win)) - smooth_mean
    dark_win = dark[lo:hi + 1]
    dark_norm = _norm01(dark_win)
    edge_norm = _norm01(edge_win)
    local_contrast = _row_local_contrast(smooth_mean, lo, hi, max(4, h // 80))

    score = 0.48 * edge_norm + 0.32 * dark_norm + 0.20 * local_contrast
    if score.size == 0:
        return None

    best_rel = int(np.argmax(score))
    best_y = lo + best_rel
    if float(score[best_rel]) < 0.28:
        return None
    if float(edge_norm[best_rel]) < 0.20 and float(dark_norm[best_rel]) < 0.35:
        return None

    return best_y


def _norm01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < 1e-6:
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def _row_local_contrast(row: np.ndarray, lo: int, hi: int, radius: int) -> np.ndarray:
    out = []
    n = len(row)
    for y in range(lo, hi + 1):
        a1 = max(0, y - radius)
        a2 = max(a1 + 1, y - max(1, radius // 3))
        b1 = min(n - 1, y + max(1, radius // 3))
        b2 = min(n, y + radius)
        above = float(np.mean(row[a1:a2])) if a2 > a1 else float(row[y])
        below = float(np.mean(row[b1:b2])) if b2 > b1 else float(row[y])
        out.append(abs(above - below))
    return _norm01(np.array(out, dtype=float))


def _foreground_binary(binary: np.ndarray, gray: np.ndarray = None) -> np.ndarray:
    if binary is None:
        if gray is None:
            return None
        _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return fg
    fg = binary.copy()
    if float(np.mean(fg > 0)) > 0.5:
        fg = cv2.bitwise_not(fg)
    return fg


def _analyze_horizontal_tick_bands(gray: np.ndarray,
                                   binary: np.ndarray,
                                   split_y: int) -> dict:
    h, w = gray.shape[:2]
    fg = _foreground_binary(binary, gray)
    if fg is None or fg.size == 0:
        return {
            'main_tick_band': (max(0, split_y - max(24, h // 3)), split_y),
            'vernier_tick_band': (split_y, min(h, split_y + max(24, h // 4))),
        }

    kernel_h = max(7, min(31, h // 18))
    if kernel_h % 2 == 0:
        kernel_h += 1
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    vertical = cv2.morphologyEx(fg, cv2.MORPH_OPEN, vertical_kernel)
    if np.count_nonzero(vertical) < max(20, fg.size * 0.0004):
        vertical = fg

    row_score = np.mean(vertical > 0, axis=1).astype(float)
    row_coverage = _row_horizontal_coverage(vertical)
    win = max(5, min(21, h // 45))
    if win % 2 == 0:
        win += 1
    smooth = np.convolve(row_score, np.ones(win, dtype=float) / win, mode='same')
    coverage_smooth = np.convolve(row_coverage, np.ones(win, dtype=float) / win, mode='same')

    main_band = _find_tick_band_from_rows(
        smooth, 0, split_y, 'main', h, coverage_smooth)
    vernier_band = _find_tick_band_from_rows(
        smooth, split_y, h, 'vernier', h, coverage_smooth)

    if main_band is None:
        main_band = (max(0, split_y - max(24, int(h * 0.28))), split_y)
    if vernier_band is None:
        vernier_band = (split_y, min(h, split_y + max(24, int(h * 0.22))))

    return {
        'main_tick_band': main_band,
        'vernier_tick_band': vernier_band,
        'row_projection': row_score,
        'row_projection_smooth': smooth,
        'row_coverage': row_coverage,
        'row_coverage_smooth': coverage_smooth,
        'vertical_binary': vertical,
    }


def _row_horizontal_coverage(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape[:2]
    if h <= 0 or w <= 0:
        return np.array([], dtype=float)
    block_w = max(12, w // 120)
    n_blocks = max(1, w // block_w)
    trimmed = binary[:, :n_blocks * block_w] > 0
    if trimmed.size == 0:
        return np.zeros(h, dtype=float)
    blocks = trimmed.reshape(h, n_blocks, block_w)
    block_density = np.mean(blocks, axis=2)
    return np.mean(block_density > 0.01, axis=1).astype(float)


def _find_tick_band_from_rows(row_score: np.ndarray,
                              lo: int,
                              hi: int,
                              side: str,
                              full_h: int,
                              row_coverage: np.ndarray = None):
    lo = max(0, int(lo))
    hi = min(len(row_score), int(hi))
    if hi - lo < 8:
        return None

    values = row_score[lo:hi]
    vmax = float(np.max(values)) if values.size else 0.0
    if vmax <= 0:
        return None

    positive = values[values > 0]
    base = float(np.percentile(positive, 62)) if positive.size else 0.0
    th = max(base, vmax * 0.18)
    min_len = max(7, min(30, int(full_h * 0.025)))
    segments = _contiguous_segments_1d(values >= th, min_len=min_len)
    if not segments:
        th = vmax * 0.12
        segments = _contiguous_segments_1d(values >= th, min_len=max(5, min_len // 2))
    if not segments:
        return None

    scored = []
    for s, e in segments:
        gs, ge = lo + s, lo + e
        length = max(1, ge - gs)
        mean_score = float(np.mean(row_score[gs:ge]))
        length_score = min(1.0, length / max(12.0, full_h * 0.18))
        if side == 'main':
            proximity = 1.0 - min(1.0, abs(hi - ge) / max(12.0, full_h * 0.30))
        else:
            proximity = 1.0 - min(1.0, abs(gs - lo) / max(12.0, full_h * 0.22))
        score = 0.62 * (mean_score / vmax) + 0.23 * length_score + 0.15 * proximity
        scored.append((score, gs, ge))

    _, y1, y2 = max(scored, key=lambda item: item[0])
    pad = max(3, min(12, full_h // 80))
    y1 = max(lo, y1 - pad)
    y2 = min(hi, y2 + pad)
    min_h = max(12, min(48, int(full_h * 0.06)))
    if y2 - y1 < min_h:
        extra = min_h - (y2 - y1)
        y1 = max(lo, y1 - extra // 2)
        y2 = min(hi, y2 + extra - extra // 2)
    if side == 'main':
        y1 = _extend_main_band_to_long_ticks(row_score, row_coverage, y1, y2, lo, hi, vmax, full_h)
    return int(y1), int(max(y1 + 1, y2))


def _extend_main_band_to_long_ticks(row_score: np.ndarray,
                                    row_coverage: np.ndarray,
                                    y1: int,
                                    y2: int,
                                    lo: int,
                                    hi: int,
                                    vmax: float,
                                    full_h: int) -> int:
    if y2 <= y1 or vmax <= 0:
        return y1
    if row_coverage is None or len(row_coverage) != len(row_score):
        row_coverage = np.ones_like(row_score, dtype=float)

    search_lo = max(lo, hi - max(50, int(full_h * 0.45)))
    positive = row_score[search_lo:y2][row_score[search_lo:y2] > 0]
    if positive.size == 0:
        return y1

    low_th = max(vmax * 0.11, float(np.percentile(positive, 28)) * 0.75)
    cov_positive = row_coverage[search_lo:y2][row_coverage[search_lo:y2] > 0]
    cov_th = max(0.22, float(np.percentile(cov_positive, 45)) * 0.85) if cov_positive.size else 0.22
    max_gap = max(8, min(28, int(full_h * 0.055)))
    candidate = y1
    gap = 0
    seen = False

    for y in range(y1 - 1, search_lo - 1, -1):
        if float(row_score[y]) >= low_th and float(row_coverage[y]) >= cov_th:
            candidate = y
            gap = 0
            seen = True
        elif seen:
            gap += 1
            if gap > max_gap:
                break

    if y1 - candidate < max(6, int(full_h * 0.025)):
        return y1
    return max(lo, candidate - max(4, min(12, full_h // 45)))


def _contiguous_segments_1d(mask: np.ndarray, min_len: int = 1):
    segments = []
    start = None
    for idx, val in enumerate(mask.astype(bool)):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            if idx - start >= min_len:
                segments.append((start, idx))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segments.append((start, len(mask)))
    return segments


def _split_by_candidate_scan(gray: np.ndarray, binary: np.ndarray,
                             h: int, w: int):
    """候选扫描：用亮度空带 + 刻线间距 + 双侧覆盖联合打分。"""
    lo, hi = int(h * config.region_split.search_lo_ratio), int(h * config.region_split.search_hi_ratio)
    if hi <= lo:
        return None

    band = max(h // config.region_split.density_band_ratio_denom,
               config.region_split.density_band_min)
    gray_means = np.mean(gray, axis=1).astype(float)
    row_mean_min = float(np.min(gray_means[lo:hi + 1]))
    row_mean_max = float(np.max(gray_means[lo:hi + 1]))
    row_mean_span = max(row_mean_max - row_mean_min, 1.0)

    best_y = None
    best_score = -1.0
    for cy in range(lo + band, hi - band):
        y1 = max(0, cy - band // 2)
        y2 = min(h, cy + band // 2)
        if y2 <= y1:
            continue

        band_density = float(np.mean(binary[y1:y2, :] > 0))
        if band_density > 0.45:
            continue

        above_zone = binary[max(0, cy - band):cy, :]
        below_zone = binary[cy:min(h, cy + band), :]
        above_cov, above_gap = _equispaced_coverage(above_zone, w)
        below_cov, below_gap = _equispaced_coverage(below_zone, w)
        if above_cov < 0.22 or below_cov < 0.22:
            continue
        if above_gap <= 0 or below_gap <= 0:
            continue

        ratio = above_gap / below_gap if above_gap > below_gap else below_gap / above_gap
        if ratio < 1.15 or ratio > 4.5:
            continue
        ratio_score = max(0.0, 1.0 - abs(ratio - 2.0) / 1.5)
        cover_score = min(1.0, above_cov * below_cov * 1.5)
        gap_score = 1.0 - band_density
        bright_score = (gray_means[cy] - row_mean_min) / row_mean_span
        score = 0.42 * ratio_score + 0.28 * cover_score + 0.22 * gap_score + 0.08 * bright_score

        if score > best_score:
            best_score = score
            best_y = cy

    if best_y is None or best_score < 0.35:
        return None

    return _snap_to_brightest_gap(gray, best_y, band, lo, hi)


def _snap_to_brightest_gap(gray: np.ndarray, center_y: int, band: int,
                           lo: int, hi: int) -> int:
    """在 center_y 附近吸附到最亮的空带行。"""
    half = max(4, band // 2)
    win_lo = max(lo, center_y - half)
    win_hi = min(hi, center_y + half)
    if win_hi <= win_lo:
        return center_y
    row_means = np.mean(gray[win_lo:win_hi + 1, :], axis=1)
    return win_lo + int(np.argmax(row_means))


def _equispaced_coverage(zone_binary: np.ndarray, w: int):
    """
    估算一个 y 带内的"等间距连续刻线段跨度 / ROI 宽度" + 估算的 tick_gap。

    Returns:
        (coverage_ratio, tick_gap_px)
        coverage_ratio: 0~1，最长等间距段跨度 / ROI 宽度
        tick_gap_px:    该段刻线相邻间距中位值（无信号时返回 0）
    """
    if zone_binary is None or zone_binary.size == 0:
        return 0.0, 0.0
    vproj = np.sum(zone_binary, axis=0).astype(float)
    vmax = float(np.max(vproj))
    if vmax <= 0:
        return 0.0, 0.0
    vproj_norm = vproj / vmax

    from .utils import find_peaks_adaptive
    peaks = find_peaks_adaptive(vproj_norm, min_dist=3, threshold_factor=0.3)
    if len(peaks) < 5:
        return 0.0, 0.0

    strength = vproj_norm[peaks]
    th = max(0.15, float(np.percentile(strength, 30)))
    strong = peaks[strength >= th]
    if len(strong) < 5:
        strong = peaks

    diffs = np.diff(strong)
    if len(diffs) == 0:
        return 0.0, 0.0
    tick_gap = float(np.median(diffs))
    if tick_gap < 3.0:
        return 0.0, 0.0

    lo_g, hi_g = tick_gap * 0.5, tick_gap * 1.8
    max_irr = 2
    segments = []
    cur_start = 0
    irr_run = 0
    for i, d in enumerate(diffs):
        if lo_g <= d <= hi_g:
            irr_run = 0
        else:
            irr_run += 1
            if irr_run > max_irr:
                end_idx = i - irr_run
                if end_idx - cur_start >= 3:
                    segments.append((cur_start, end_idx))
                cur_start = i + 1
                irr_run = 0
    if len(strong) - 1 - cur_start >= 3:
        segments.append((cur_start, len(strong) - 1))

    if not segments:
        return 0.0, 0.0

    best = max(segments, key=lambda s: s[1] - s[0])
    span = int(strong[best[1]]) - int(strong[best[0]])
    # 重新估算该最长段内部的 tick_gap（更精确）
    seg_diffs = np.diff(strong[best[0]:best[1] + 1])
    seg_diffs_valid = seg_diffs[(seg_diffs >= lo_g) & (seg_diffs <= hi_g)]
    if len(seg_diffs_valid) >= 2:
        tick_gap = float(np.median(seg_diffs_valid))
    return min(1.0, span / float(w)), tick_gap


# ═══════════════════════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════════════════════

def _make_split_vis(color_bg: np.ndarray,   # BGR 彩色图（若不可用则为灰度）
                     gray: np.ndarray,
                     binary: np.ndarray,
                     split_y: int,
                     band_info: dict = None) -> np.ndarray:
    """生成区域分离的可视化图：彩色原图+分割线 + 梯度投影图 + 闭运算投影图"""
    h, w = gray.shape

    # ── 上：用彩色背景（清晰可辨）或灰度背景 + 分割线 ──
    if len(color_bg.shape) == 3 and color_bg.shape[2] == 3:
        vis_gray = color_bg.copy()
    else:
        vis_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if band_info:
        overlay = vis_gray.copy()
        main_band = band_info.get('main_tick_band')
        vernier_band = band_info.get('vernier_tick_band')
        if main_band:
            cv2.rectangle(overlay, (0, int(main_band[0])), (w - 1, int(main_band[1])),
                          (0, 180, 80), -1)
        if vernier_band:
            cv2.rectangle(overlay, (0, int(vernier_band[0])), (w - 1, int(vernier_band[1])),
                          (255, 160, 40), -1)
        vis_gray = cv2.addWeighted(vis_gray, 0.82, overlay, 0.18, 0)
        if main_band:
            cv2.line(vis_gray, (0, int(main_band[0])), (w, int(main_band[0])), (0, 180, 80), 1)
            cv2.line(vis_gray, (0, int(main_band[1])), (w, int(main_band[1])), (0, 180, 80), 1)
        if vernier_band:
            cv2.line(vis_gray, (0, int(vernier_band[0])), (w, int(vernier_band[0])), (255, 160, 40), 1)
            cv2.line(vis_gray, (0, int(vernier_band[1])), (w, int(vernier_band[1])), (255, 160, 40), 1)

    cv2.line(vis_gray, (0, split_y), (w, split_y), (0, 255, 255), 2)
    cv2.putText(vis_gray, "MAIN SCALE", (10, split_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis_gray, "VERNIER", (10, split_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # ── 中：梯度投影图 ──
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad = np.abs(sobel_y)
    grad_proj = np.sum(abs_grad, axis=1).astype(float)
    if np.max(grad_proj) > 0:
        grad_proj /= np.max(grad_proj)
    grad_plot = draw_projection_plot(grad_proj, title="Sobel-Y Gradient Projection (peak = split)")
    if len(grad_proj) > 0:
        px = int(split_y * (grad_plot.shape[1] - 40) / len(grad_proj)) + 20
        cv2.line(grad_plot, (px, 0), (px, grad_plot.shape[0]),
                 (0, 255, 255), 1, cv2.LINE_AA)

    # ── 下：二值图闭运算投影 ──
    if binary is not None:
        closed = band_info.get('vertical_binary') if band_info else None
        if closed is None:
            fg = _foreground_binary(binary, gray)
            kernel_h = max(7, min(31, h // 18))
            if kernel_h % 2 == 0:
                kernel_h += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
            closed = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        hproj = np.sum(closed > 0, axis=1).astype(float)
        if np.max(hproj) > 0:
            hproj /= np.max(hproj)
        bin_plot = draw_projection_plot(hproj,
                                         title="Horizontal Projection of Vertical Tick Pixels")
        if len(hproj) > 0:
            px2 = int(split_y * (bin_plot.shape[1] - 40) / len(hproj)) + 20
            cv2.line(bin_plot, (px2, 0), (px2, bin_plot.shape[0]),
                     (0, 255, 255), 1, cv2.LINE_AA)

        # 闭运算效果预览（缩略图）
        closed_thumb = cv2.resize(closed, (min(w, 600), min(h, 80)),
                                   interpolation=cv2.INTER_AREA)
        closed_vis = cv2.cvtColor(closed_thumb, cv2.COLOR_GRAY2BGR)
    else:
        bin_plot = np.zeros((200, 200, 3), dtype=np.uint8)
        closed_vis = np.zeros((80, 200, 3), dtype=np.uint8)

    # ── 合并 ──
    gap = 4
    plot_h = grad_plot.shape[0]
    plot_h2 = bin_plot.shape[0]
    thumb_h = closed_vis.shape[0]

    out_h = h + plot_h + plot_h2 + thumb_h + gap * 4
    out_w = max(w, grad_plot.shape[1], bin_plot.shape[1])
    vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    vis[:] = (30, 30, 35)

    vis[:h, :w] = vis_gray
    y = h + gap
    vis[y:y + plot_h, :grad_plot.shape[1]] = grad_plot
    y += plot_h + gap
    vis[y:y + thumb_h, :closed_vis.shape[1]] = closed_vis
    y += thumb_h + gap
    vis[y:y + plot_h2, :bin_plot.shape[1]] = bin_plot

    cv2.putText(vis, "STEP 2: Region Split (projection + gradient + binary)",
                (5, out_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 125), 1)

    return vis
