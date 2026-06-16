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

    # ── 提取所有刻线的垂直分布（用于验证两侧 tick 密度）──
    tick_xs = _extract_vertical_feature_positions(binary, w)

    # ── 方案A（优先）：水平投影突变法 ──
    split_y = _split_by_projection(rotated_gray, binary, tick_xs, h, w)

    # ── 方案B 回退：灰度梯度法 ──
    if split_y is None:
        split_y = _split_by_gradient(rotated_gray, binary, tick_xs, h, w)

    # ── 方案C 回退：二值闭运算投影法 ──
    if split_y is None:
        split_y = _split_by_binary_close(binary, tick_xs, h, w)

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


# ═══════════════════════════════════════════════════════════
#  方案A（优先）：水平投影突变法
# ═══════════════════════════════════════════════════════════

def _split_by_projection(gray: np.ndarray, binary: np.ndarray,
                          tick_xs: np.ndarray, h: int, w: int):
    """
    主尺/游标分界检测 — v6.1: Sobel Y 找面板上沿，再回退到面板上方的窄白缝。

    最强水平边缘 = 游标尺面板**上沿**（金属→暗压块跳变），但游标尺
    刻度线本身也跨过这条边缘（从主尺数字行底部一直到面板下方），
    若直接以此为 split_y，会把游标尺刻度切成两半。

    正确做法：在 best_y 上方往上找"hproj 行像素均值的局部最大（亮带）" →
    即主尺数字行底部和游标尺面板顶部之间的窄白缝。

    步骤：
      1. Sobel Y → OTSU → 水平投影找最强 y（游标尺面板上沿，记为 panel_top_y）
      2. 在 [panel_top_y - 60, panel_top_y - 5] 范围内做 gray 行均值
      3. 找该范围内 gray 均值最大的 y（白缝处）即为 split_y
    """
    if gray is None:
        return None
    h_full = h

    lo, hi = int(h * config.region_split.search_lo_ratio), int(h * config.region_split.search_hi_ratio)
    if hi <= lo:
        return None

    # Sobel Y → 强水平边缘
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    abs_y = np.abs(sobel_y)
    g_max = float(np.max(abs_y))
    if g_max <= 0:
        return None
    abs_y_u8 = (abs_y / g_max * 255).astype(np.uint8)
    _, bw = cv2.threshold(abs_y_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hproj_edge = np.sum(bw == 255, axis=1).astype(float)

    region_proj = hproj_edge[lo:hi + 1]
    if len(region_proj) < 10:
        return None
    best_rel = int(np.argmax(region_proj))
    panel_top_y = best_rel + lo
    best_strength = float(region_proj[best_rel])
    if best_strength < float(np.max(hproj_edge)) * 0.30:
        return None

    # ── v6.2: 在 panel_top_y 上方做固定回退（这种卡尺主尺刻度行紧贴游标尺面板，
    #     没有白缝；强行回退 ~10% ROI 高度，让主尺刻度行完整归入主尺区）──
    retreat_px = max(15, int(h_full * 0.07))
    split_y = max(lo, panel_top_y - retreat_px)
    return split_y


# ═══════════════════════════════════════════════════════════
#  方案B（回退）：灰度梯度法
# ═══════════════════════════════════════════════════════════

def _split_by_gradient(gray: np.ndarray, binary: np.ndarray,
                        tick_xs: np.ndarray, h: int, w: int):
    """梯度法找分界线 + 刻线密度验证"""
    clahe = cv2.createCLAHE(
        clipLimit=config.region_split.clahe_clip_limit,
        tileGridSize=(config.region_split.clahe_tile_w, config.region_split.clahe_tile_h))
    enhanced = clahe.apply(gray)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad = np.abs(sobel_y)
    grad_proj = np.sum(abs_grad, axis=1).astype(float)
    if np.max(grad_proj) > 0:
        grad_proj /= np.max(grad_proj)

    lo, hi = int(h * config.region_split.search_lo_ratio), int(h * config.region_split.search_hi_ratio)
    if hi <= lo:
        return None

    # 收集多个候选峰值
    mean_val = float(np.mean(grad_proj[lo:hi]))
    thresh = max(mean_val * config.region_split.gradient_threshold_factor,
                 config.region_split.gradient_min_thresh)
    grad_region = grad_proj[lo:hi]
    candidates = _find_local_peaks(grad_region, min_dist=5, threshold=thresh)
    candidates = [lo + p for p in candidates]

    if not candidates:
        return None

    # 按 tick-density 打分，取最优
    band = max(h // config.region_split.density_band_ratio_denom,
               config.region_split.density_band_min)
    best_y, best_score = None, -1
    for cy in candidates:
        score = _tick_density_score(binary, tick_xs, h, w, cy, band)
        if score > best_score:
            best_score = score
            best_y = cy

    if best_score < config.region_split.density_min_score:
        return None

    return best_y


def _split_by_binary_close(binary: np.ndarray, tick_xs: np.ndarray,
                            h: int, w: int):
    """二值图闭运算投影找分界线 + 刻线密度验证"""
    kernel_w = max(int(w * config.region_split.close_kernel_ratio), 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    hproj = np.sum(closed, axis=1).astype(float)
    if np.max(hproj) > 0:
        hproj /= np.max(hproj)

    lo, hi = int(h * config.region_split.search_lo_ratio), int(h * config.region_split.search_hi_ratio)
    if hi <= lo:
        return None

    # 收集多个谷值候选
    hproj_region = 1.0 - hproj[lo:hi]  # 反转：谷变峰
    candidates = _find_local_peaks(hproj_region, min_dist=5, threshold=0.02)
    candidates = [lo + p for p in candidates]

    if not candidates:
        return None

    band = max(h // 20, 8)
    best_y, best_score = None, -1
    for cy in candidates:
        score = _tick_density_score(binary, tick_xs, h, w, cy, band)
        if score > best_score:
            best_score = score
            best_y = cy

    if best_score < 4:
        return None
    return best_y


def _tick_density_score(binary: np.ndarray, tick_xs: np.ndarray,
                         h: int, w: int, cy: int, band: int) -> float:
    """
    对候选分割线打分。

    v5.7 关键洞察：游标尺刻度密度 ≈ 主尺密度的 1.7~2 倍（20 分度卡尺）。
    因此分界点的物理特征是：
      • 上方 band 内是"间距 g_above 的等间距刻线行"
      • 下方 band 内是"间距 g_below 的等间距刻线行"
      • g_above / g_below ∈ [1.4, 3.5]（主尺/游标的合理比值范围）
    切在同一行刻度内部时 g_above ≈ g_below ≈ 同一值 → 比值 ≈ 1 → 低分。
    """
    y1 = max(0, cy - band)
    y2 = min(h - 1, cy + band)
    if y2 <= y1:
        return 0.0

    above_zone = binary[y1:cy + 1, :]
    below_zone = binary[cy:y2 + 1, :]

    # ── 计算上下两侧的"等间距覆盖系数 + tick_gap" ──
    above_cov, above_gap = _equispaced_coverage(above_zone, w)
    below_cov, below_gap = _equispaced_coverage(below_zone, w)

    # 任一侧没有"足够等间距覆盖"（< 30%）→ 直接淘汰
    if above_cov < 0.30 or below_cov < 0.30:
        return 0.0

    if above_gap <= 0 or below_gap <= 0:
        return 0.0

    # ── 主尺/游标比值打分 ──
    #     比值 ≈ 1.0 → 切在同一行内部（淘汰）
    #     比值 ≈ 1.7~2.0 → 完美主尺/游标分界（最高分）
    #     比值 > 3.5 或 < 1.0 → 不合理
    ratio = above_gap / below_gap if above_gap > below_gap else below_gap / above_gap
    if ratio < 1.30 or ratio > 3.50:
        return 0.0

    # 距离理想比值 2.0 的近度（0~1，越接近 2.0 越高）
    closeness = max(0.0, 1.0 - abs(ratio - 2.0) / 1.5)

    # 综合得分 = 双侧覆盖乘积 × 比值近度 × 100
    base = above_cov * below_cov  # 0~1
    return float(base * (0.5 + closeness * 1.5) * 100.0)


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


def _extract_vertical_feature_positions(binary: np.ndarray, w: int) -> np.ndarray:
    """从二值图提取有垂直刻线特征的 x 坐标列表"""
    vproj = np.sum(binary, axis=0).astype(float)
    if np.max(vproj) > 0:
        vproj /= np.max(vproj)
    # 找局部峰值（垂直刻线位置），低门槛确保不遗漏
    from .utils import find_peaks_adaptive
    return find_peaks_adaptive(vproj, min_dist=2, threshold_factor=0.1)


def _find_local_peaks(signal: np.ndarray, min_dist: int = 5,
                       threshold: float = 0.02) -> list:
    """在 1D 信号中找局部峰值，返回索引列表"""
    n = len(signal)
    peaks = []
    for i in range(min_dist, n - min_dist):
        if signal[i] <= threshold:
            continue
        if all(signal[i] > signal[j]
               for j in range(i - min_dist, i + min_dist + 1) if j != i):
            peaks.append(i)
    return peaks


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
