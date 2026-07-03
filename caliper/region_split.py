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
        split_y = _split_by_candidate_scan(rotated_gray, binary, h, w)

    # ── 最终回退（基于物理先验：主尺约占ROI高度的60%）──
    if split_y is None:
        split_y = int(h * config.region_split.fallback_split_ratio)

    # ── 游标区域高度校验：不能太小（至少占ROI的 min_ratio）──
    min_vernier_h = int(h * config.region_split.min_vernier_height_ratio)
    if h - split_y < min_vernier_h:
        split_y = h - min_vernier_h

    # ── 切分 ──
    img_upper = rotated_gray[:split_y, :]
    img_lower = rotated_gray[split_y:, :]
    bin_upper = binary[:split_y, :]
    bin_lower = binary[split_y:, :]

    region_main = {
        'image': img_upper, 'binary': bin_upper,
        'y_offset': 0, 'height': split_y,
    }
    region_vernier = {
        'image': img_lower, 'binary': bin_lower,
        'y_offset': split_y, 'height': h - split_y,
    }

    split_vis = _make_split_vis(rotated_color if rotated_color is not None
                                  else rotated_gray,
                                  rotated_gray, binary, split_y)

    return {
        'region_main': region_main,
        'region_vernier': region_vernier,
        'split_y': split_y,
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
                     split_y: int) -> np.ndarray:
    """生成区域分离的可视化图：彩色原图+分割线 + 梯度投影图 + 闭运算投影图"""
    h, w = gray.shape

    # ── 上：用彩色背景（清晰可辨）或灰度背景 + 分割线 ──
    if len(color_bg.shape) == 3 and color_bg.shape[2] == 3:
        vis_gray = color_bg.copy()
    else:
        vis_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

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
        kernel_w = max(w // 3, 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        hproj = np.sum(closed, axis=1).astype(float)
        if np.max(hproj) > 0:
            hproj /= np.max(hproj)
        bin_plot = draw_projection_plot(hproj,
                                         title=f"Binary-Close Projection (kernel={kernel_w}px)")
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
