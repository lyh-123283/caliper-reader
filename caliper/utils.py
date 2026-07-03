"""
游标卡尺识别 — 通用工具函数
"""

import cv2
import numpy as np
from typing import List
from .config import config


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """旋转图像，不裁剪，空白区域填白色"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    ca, sa = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(h * sa + w * ca), int(h * ca + w * sa)
    M[0, 2] += nW / 2 - w / 2
    M[1, 2] += nH / 2 - h / 2
    if len(img.shape) == 3:
        border = (255, 255, 255)
    else:
        border = 255
    return cv2.warpAffine(img, M, (nW, nH),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border)


def find_peaks_adaptive(signal: np.ndarray, min_dist: int = 3,
                         threshold_factor: float = 0.3) -> np.ndarray:
    """自适应阈值找峰值：阈值为 signal 均值 + threshold_factor×标准差"""
    mu = float(np.mean(signal))
    sigma = float(np.std(signal))
    th = max(mu + threshold_factor * sigma, 0.02)

    peaks = []
    n = len(signal)
    for i in range(min_dist, n - min_dist):
        if signal[i] <= th:
            continue
        is_peak = True
        for d in range(1, min_dist + 1):
            if signal[i] <= signal[i - d] or signal[i] <= signal[i + d]:
                is_peak = False
                break
        if is_peak:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def _tick_row_threshold(col: np.ndarray,
                        max_factor: float = 0.40,
                        single_stroke_cap: float = 0.80) -> float:
    """Threshold a 7px-wide vertical strip without rejecting 1px tick strokes."""
    if col is None or len(col) == 0:
        return 30.0
    dynamic = float(np.max(col)) * max_factor
    one_pixel_cap = 255.0 * single_stroke_cap
    return max(30.0, min(dynamic, one_pixel_cap))


def extract_ticks_from_binary(binary: np.ndarray,
                               approx_xs: np.ndarray,
                               min_length_ratio: float = 0.25,
                               long_tick_factor: float = None) -> List[dict]:
    """在指定 x 坐标附近精确提取刻线起止点。

    v6: 加入"刻线长度 ≥ 区域高度 × min_length_ratio"硬约束，
        过滤掉短伪刻线（如 OCR 数字笔画、噪声、阴影边缘）。
    v6.5: long_tick_factor 改为参数，调用方传入主尺/游标尺各自的 config 值。
    """
    h, w = binary.shape
    min_len_px = max(6, int(h * min_length_ratio))
    if long_tick_factor is None:
        long_tick_factor = config.main_scale.long_tick_factor
    ticks = []

    for px in approx_xs:
        x = int(px)
        if x < 3 or x >= w - 3:
            continue

        # 取 x 附近的列（±3像素），求和得到该位置的垂直投影
        strip = binary[:, max(0, x - 3):min(w, x + 4)]
        col = np.sum(strip, axis=1)

        # v6: 列强度阈值收紧到 max*0.40（之前 0.25 太宽松）
        threshold = _tick_row_threshold(col)
        indices = np.where(col > threshold)[0]

        if len(indices) < min_len_px // 2:
            continue

        segs = contiguous_segments(indices, min_len=5)
        if not segs:
            continue

        ys, ye = max(segs, key=lambda s: s[1] - s[0])
        length = ye - ys
        if length < min_len_px:
            continue

        x_refined = _refine_tick_x(binary, x, ys, ye,
                                   search_radius=max(4, min(10, min_len_px // 2)))
        ticks.append({
            'x': int(x_refined),
            'y_start': int(ys),
            'y_end': int(ye),
            'y_mid': int((ys + ye) / 2),
            'length': int(length),
        })

    if ticks:
        ml = float(np.median([t['length'] for t in ticks]))
        for t in ticks:
            t['is_long'] = t['length'] > ml * long_tick_factor

    return ticks


def extract_ticks_from_anchor_band(binary: np.ndarray,
                                   direction: str,
                                   min_length_ratio: float = 0.20,
                                   band_ratio: float = 0.35,
                                   peak_min_dist: int = 3,
                                   peak_threshold_factor: float = 0.15,
                                   long_tick_factor: float = None) -> List[dict]:
    """Detect vertical tick marks from the band next to the split line.

    direction="up"   finds main-scale ticks growing upward from the split.
    direction="down" finds vernier ticks growing downward from the split.
    """
    h, w = binary.shape[:2]
    if h == 0 or w == 0:
        return []

    band_h = max(12, min(h, int(h * band_ratio)))
    if direction == "up":
        anchor_band = binary[h - band_h:h, :]
        anchor_limit = h - max(6, int(h * 0.08))
    else:
        anchor_band = binary[:band_h, :]
        anchor_limit = max(6, int(h * 0.08))

    vproj = np.sum(anchor_band > 0, axis=0).astype(float)
    if np.max(vproj) <= 0:
        return []
    vproj_norm = vproj / np.max(vproj)
    approx_xs = find_peaks_adaptive(
        vproj_norm,
        min_dist=peak_min_dist,
        threshold_factor=peak_threshold_factor,
    )

    min_len_px = max(6, int(h * min_length_ratio))
    ticks = []
    for px in approx_xs:
        x = int(px)
        if x < 3 or x >= w - 3:
            continue

        strip = binary[:, max(0, x - 3):min(w, x + 4)]
        col = np.sum(strip, axis=1)
        threshold = _tick_row_threshold(col)
        indices = np.where(col > threshold)[0]
        if len(indices) < min_len_px // 2:
            continue

        segs = contiguous_segments(indices, min_len=5)
        if not segs:
            continue

        if direction == "up":
            anchored = [s for s in segs if s[1] >= anchor_limit]
            seg = max(anchored, key=lambda s: s[1] - s[0]) if anchored else max(segs, key=lambda s: s[1])
        else:
            anchored = [s for s in segs if s[0] <= anchor_limit]
            seg = max(anchored, key=lambda s: s[1] - s[0]) if anchored else min(segs, key=lambda s: s[0])

        ys, ye = seg
        length = ye - ys
        if length < min_len_px:
            continue

        x_refined = _refine_tick_x(binary, x, ys, ye,
                                   search_radius=max(4, min(10, min_len_px // 2)))
        ticks.append({
            'x': int(x_refined),
            'y_start': int(ys),
            'y_end': int(ye),
            'y_mid': int((ys + ye) / 2),
            'length': int(length),
        })

    ticks = _dedupe_ticks_by_x(ticks)
    if ticks:
        ml = float(np.median([t['length'] for t in ticks]))
        factor = long_tick_factor if long_tick_factor is not None else config.main_scale.long_tick_factor
        for t in ticks:
            t['is_long'] = t['length'] > ml * factor
    return ticks


def _dedupe_ticks_by_x(ticks: List[dict], min_gap: int = 3) -> List[dict]:
    if not ticks:
        return []
    out = []
    for t in sorted(ticks, key=lambda item: item['x']):
        if out and abs(t['x'] - out[-1]['x']) <= min_gap:
            if t['length'] > out[-1]['length']:
                out[-1] = t
        else:
            out.append(t)
    return out


def _refine_tick_x(binary: np.ndarray,
                   approx_x: int,
                   y_start: int,
                   y_end: int,
                   search_radius: int = 6) -> int:
    """Refine a coarse x position by looking for the densest vertical stroke."""
    h, w = binary.shape[:2]
    if h == 0 or w == 0:
        return int(approx_x)

    x1 = max(0, int(approx_x) - search_radius)
    x2 = min(w - 1, int(approx_x) + search_radius)
    if x2 <= x1:
        return int(approx_x)

    y1 = max(0, int(y_start) - 1)
    y2 = min(h - 1, int(y_end) + 1)
    if y2 <= y1:
        return int(approx_x)

    crop = binary[y1:y2 + 1, x1:x2 + 1]
    if crop.size == 0:
        return int(approx_x)

    col_scores = np.sum(crop > 0, axis=0).astype(float)
    if not np.any(col_scores > 0):
        return int(approx_x)

    best = np.max(col_scores)
    best_idx = np.where(col_scores == best)[0]
    if len(best_idx) == 0:
        return int(approx_x)

    refined = x1 + int(round(float(np.mean(best_idx))))
    return max(0, min(w - 1, refined))


def contiguous_segments(indices: np.ndarray, min_len: int = 5) -> List[tuple]:
    """将连续索引归并为线段"""
    if len(indices) < 2:
        return []
    segs, start = [], indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            if int(indices[i - 1]) - int(start) + 1 >= min_len:
                segs.append((int(start), int(indices[i - 1])))
            start = indices[i]
    if int(indices[-1]) - int(start) + 1 >= min_len:
        segs.append((int(start), int(indices[-1])))
    return segs


def refine_ticks_by_spacing(x_positions: np.ndarray,
                            binary: np.ndarray,
                            spacing_tolerance: float = 0.30,
                            gap_factor: float = 1.55,
                            dup_factor: float = 0.50,
                            snap_ratio: float = 0.28) -> np.ndarray:
    """
    利用刻度线严格等间距的物理特性，补全遗漏 + 过滤误检。

    核心思路
    --------
    游标卡尺的刻度线（无论主尺还是游标尺）相邻间距严格相等。
    垂直投影峰值检测偶尔会漏掉某条线（对比度低、被噪声淹没），
    或者把伪影误判为刻线。利用等间距约束可以极大提升鲁棒性。

    算法步骤
    --------
    1) 由已检测刻线计算中位数间距 S（对少量异常值鲁棒）
    2) 逐对扫描相邻刻线间距：
       - gap > gap_factor * S  → 判定中间遗漏了刻线 → 在期望位置补入
       - gap < dup_factor * S  → 判定其中一条是伪影 → 保留列信号更强的
    3) 在二值图的期望 x 位置附近微搜索，精确定位补入的刻线
    4) 全局网格快照：把每个刻线吸附到最近的等间距网格点上（容差 snap_ratio*S）

    Args:
        x_positions:      已排序的刻线 x 坐标 (int/float)
        binary:           二值图（0/255, 白前景黑背景）
        spacing_tolerance: 网格匹配容差比例（默认 0.30，即 30% 间距）
        gap_factor:       间距 > S*gap_factor 时触发补全（默认 1.55）
        dup_factor:       间距 < S*dup_factor 时触发去重（默认 0.50）
        snap_ratio:       网格吸附容差比例（默认 0.28）

    Returns:
        精炼后的刻线 x 坐标 np.ndarray（float, 已排序）
    """
    xs = np.array(sorted(set(float(x) for x in x_positions)), dtype=float)
    n = len(xs)
    if n < 3:
        return xs

    h, w = binary.shape

    # ── 1. 中位数间距（过滤异常值：剔除 > 2.5*中位数 的 gap） ──
    diffs_all = np.diff(xs)
    median_raw = float(np.median(diffs_all))
    if median_raw < 2.0:
        return xs
    valid_diffs = diffs_all[diffs_all < median_raw * 2.5]
    spacing = float(np.median(valid_diffs)) if len(valid_diffs) >= 2 else median_raw
    if spacing < 2.0:
        return xs

    tol = spacing * spacing_tolerance
    gap_th = spacing * gap_factor
    dup_th = spacing * dup_factor

    # ── 辅助：在 x 附近搜索二值图中的最强列 ──
    def _search_column(nominal_x: float, search_radius: int = None) -> float:
        if search_radius is None:
            search_radius = max(2, int(spacing * 0.28))
        lo = max(0, int(nominal_x) - search_radius)
        hi = min(w - 1, int(nominal_x) + search_radius)
        if hi <= lo:
            return nominal_x
        strip = binary[:, lo:hi + 1]
        col_sum = np.sum(strip, axis=0).astype(float)
        best_offset = int(np.argmax(col_sum))
        # 必须有足够信号才算有效
        if col_sum[best_offset] < 255 * 3:
            return nominal_x  # 信号太弱，不强行补
        return float(lo + best_offset)

    # ── 辅助：计算某 x 处的列信号强度 ──
    def _column_strength(xx: float) -> float:
        col = int(round(xx))
        if col < 0 or col >= w:
            return 0.0
        return float(np.sum(binary[:, col]))

    # ── 2. 逐对扫描：补全遗漏 ──
    filled = []
    i = 0
    while i < n:
        filled.append(xs[i])
        if i + 1 < n:
            gap = xs[i + 1] - xs[i]
            missing_count = round(gap / spacing) - 1
            if missing_count >= 1 and gap > gap_th:
                # 在 xs[i] 和 xs[i+1] 之间等距插入缺失的刻线
                step = gap / (missing_count + 1)
                for k in range(1, missing_count + 1):
                    expected_x = xs[i] + k * step
                    refined_x = _search_column(expected_x)
                    filled.append(refined_x)
        i += 1

    # 排序 + 去重
    filled = np.array(sorted(set(round(x, 1) for x in filled)), dtype=float)

    # ── 3. 逐对扫描：去重（过密的伪影） ──
    cleaned = [filled[0]]
    for j in range(1, len(filled)):
        gap = filled[j] - cleaned[-1]
        if gap < dup_th:
            # 保留列信号更强的那个
            str_new = _column_strength(filled[j])
            str_old = _column_strength(cleaned[-1])
            if str_new > str_old:
                cleaned[-1] = filled[j]
            # 否则丢弃 filled[j]
        else:
            cleaned.append(filled[j])

    cleaned = np.array(cleaned, dtype=float)

    # ── 4. 全局网格吸附 ──
    if len(cleaned) >= 3:
        # 重新估算间距（用清理后的数据）
        diffs2 = np.diff(cleaned)
        spacing2 = float(np.median(diffs2[diffs2 < np.median(diffs2) * 2.5]))
        if spacing2 >= 2.0:
            # 以最左侧刻线为基准，生成完整网格
            origin = cleaned[0]
            snapped = []
            for x in cleaned:
                k = round((x - origin) / spacing2)
                grid_x = origin + k * spacing2
                if abs(x - grid_x) <= spacing2 * snap_ratio:
                    snapped.append(grid_x)
                else:
                    # 偏离太大，保留原值（可能是真实偏移）
                    snapped.append(x)
            cleaned = np.array(sorted(set(round(x, 1) for x in snapped)), dtype=float)

    # ── 5. 网格边缘延伸：从最左/最右向图像边界扩展 ──
    #     物理依据：刻度线布满整个主尺/游标尺区域，检测到的第一条/最后
    #     一条线未必是真正的第一/最后一条。沿等间距网格向两侧延伸到图像
    #     边界，在预期位置搜索列信号，有足够强度即确认该刻线存在。
    if len(cleaned) >= 3:
        spacing3 = float(np.median(np.diff(cleaned)))
        if spacing3 >= 2.0:
            origin = cleaned[0] if len(cleaned) > 0 else 0.0

            # 向左延伸
            x = origin - spacing3
            while x >= spacing3 * 0.3:  # 至少离左边界还有一定距离
                found_x = _search_column(x, search_radius=max(2, int(spacing3 * 0.3)))
                if found_x != x:  # _search_column 找到了有效信号
                    cleaned = np.append(cleaned, found_x)
                x -= spacing3

            # 向右延伸
            last_x = cleaned[-1] if len(cleaned) > 0 else origin
            x = last_x + spacing3
            while x <= w - spacing3 * 0.3:
                found_x = _search_column(x, search_radius=max(2, int(spacing3 * 0.3)))
                if found_x != x:
                    cleaned = np.append(cleaned, found_x)
                x += spacing3

            cleaned = np.array(sorted(set(round(x, 1) for x in cleaned)), dtype=float)

    return cleaned


# ═════════════════════════════════════════════════════════════
#  可视化工具
# ═════════════════════════════════════════════════════════════

def make_comparison_vis(left_img: np.ndarray, right_img: np.ndarray,
                         left_label: str = "", right_label: str = "",
                         bg_color: tuple = (40, 40, 45)) -> np.ndarray:
    """生成左右对比图"""
    h1, w1 = left_img.shape[:2]
    h2, w2 = right_img.shape[:2]

    # 确保都是 3 通道
    if len(left_img.shape) == 2:
        left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
    if len(right_img.shape) == 2:
        right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)

    gap = 4
    out_h = max(h1, h2)
    out_w = w1 + w2 + gap
    vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    vis[:] = bg_color

    vis[:h1, :w1] = left_img
    vis[:h2, w1 + gap:w1 + gap + w2] = right_img

    if left_label:
        cv2.putText(vis, left_label, (5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if right_label:
        cv2.putText(vis, right_label, (w1 + gap + 5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return vis


def draw_projection_plot(signal: np.ndarray, peaks: np.ndarray = None,
                          width: int = 800, height: int = 200,
                          title: str = "") -> np.ndarray:
    """绘制投影曲线图（用于调试可视化）"""
    if len(signal) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    # 归一化
    s_max = float(np.max(signal))
    if s_max > 0:
        s_norm = signal / s_max
    else:
        s_norm = signal

    n = len(s_norm)
    plot = np.ones((height, width, 3), dtype=np.uint8) * 30

    # 绘制曲线
    for i in range(min(n - 1, width - 2)):
        x0 = int(i * (width - 40) / n) + 20
        x1 = int((i + 1) * (width - 40) / n) + 20
        y0 = height - 20 - int(s_norm[i] * (height - 50))
        y1 = height - 20 - int(s_norm[i + 1] * (height - 50))
        cv2.line(plot, (x0, y0), (x1, y1), (100, 200, 255), 1)

    # 绘制峰值
    if peaks is not None and len(peaks) > 0:
        for p in peaks:
            pi = int(p)
            if 0 <= pi < n:
                px = int(pi * (width - 40) / n) + 20
                py = height - 20 - int(s_norm[pi] * (height - 50))
                cv2.circle(plot, (px, py), 4, (0, 255, 100), -1)

    # 标题
    if title:
        cv2.putText(plot, title, (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return plot


def draw_legend(image: np.ndarray, items: list,
                x: int = 8, y_start: int = 50, line_h: int = 18) -> np.ndarray:
    """
    在图像上叠加图例。
    items: [(标签, 颜色, 样式), ...]
      样式: 'line', 'rect', 'circle', 'text'
    返回原地修改的图像引用。
    """
    for i, (label, color, style) in enumerate(items):
        cy = y_start + i * line_h
        if style == 'line':
            cv2.line(image, (x, cy), (x + 24, cy), color, 2)
        elif style == 'rect':
            cv2.rectangle(image, (x, cy - 6), (x + 24, cy + 6), color, 1)
        elif style == 'circle':
            cv2.circle(image, (x + 12, cy), 5, color, -1)
        cv2.putText(image, label, (x + 30, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
    return image


def draw_legend_below(image: np.ndarray, items: list, line_h: int = 18) -> np.ndarray:
    """
    在图像下方拼接一个图例面板（不覆盖原图）。
    """
    h, w = image.shape[:2]
    panel_h = len(items) * line_h + 16
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 35)
    for i, (label, color, style) in enumerate(items):
        cy = 10 + i * line_h
        if style == 'line':
            cv2.line(panel, (8, cy), (32, cy), color, 2)
        elif style == 'rect':
            cv2.rectangle(panel, (8, cy - 6), (32, cy + 6), color, 1)
        elif style == 'circle':
            cv2.circle(panel, (20, cy), 5, color, -1)
        cv2.putText(panel, label, (38, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    return np.vstack([image, panel])
