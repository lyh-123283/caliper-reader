"""
步骤 4 — 游标尺识别（刻度线检测 + 精度推断 + 对齐查找 + 数字OCR）

流程:
  1. 垂直投影 → 游标刻度线检测
  2. 游标间距 → 推断精度 (0.02/0.05/0.1)
  3. 零线定位（最左侧刻线）
  4. 网格法找最佳对齐线 → 小数读数
  5. OCR 识别游标数字
"""

import cv2
import numpy as np
from typing import List, Tuple

from .utils import (
    find_peaks_adaptive, extract_ticks_from_binary, draw_projection_plot,
    refine_ticks_by_spacing, extract_ticks_from_anchor_band,
)
from .ocr import DigitReader
from .config import config


def recognize_vernier_scale(region: dict,
                             main_gap: float,
                             color_region: np.ndarray = None,
                             main_ticks: List[dict] = None) -> dict:
    """
    游标尺识别主函数

    Args:
        region:       游标尺区域 dict {image, binary, y_offset, height}
        main_gap:     主尺刻度间距（像素）
        color_region: 对应的彩色区域
        main_ticks:   主尺刻线列表（可选；None 则退化为合成网格法）

    Returns:
        dict with keys:
            'vernier_ticks':    游标刻度线列表
            'precision':        推断精度
            'vernier_reading':  小数读数
            'zero_x':           零线 x 坐标
            'aligned_tick':     对齐的游标刻线
            'vis_ticks':        刻度线可视化
            'vis_alignment':    对齐可视化
            'vis_digits':       数字识别可视化
    """
    img_full = region['image']
    h_full, w = img_full.shape

    # ── 0. 顶部裁剪：主尺刻度线底部可能混入游标区最上方几行 ──
    #        裁剪掉灰度偏亮、前景像素多（主尺残余）的头几行，
    #        避免它们污染垂直投影和零线检测。
    trim_top = max(0, min(10, h_full // 8))
    img = img_full[trim_top:, :]
    h, w = img.shape

    if main_gap <= 0:
        main_gap = 10.0

    # ── 1. 自适应二值化（比 OTSU 更鲁棒，避免低对比度时全部消失）──
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=config.vernier_scale.adaptive_block_size,
        C=config.vernier_scale.adaptive_C
    )
    # 回退：自适应阈值得到的前景太少，改用 OTSU
    if np.sum(binary > 0) < img.shape[0] * img.shape[1] * 0.03:
        _, binary = cv2.threshold(img, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    region['binary'] = binary

    # ── 2. 垂直投影 → 游标刻度检测 ──
    vproj = np.sum(binary, axis=0).astype(float)
    if np.max(vproj) > 0:
        vproj_norm = vproj / np.max(vproj)
    else:
        vproj_norm = vproj

    vernier_xs = find_peaks_adaptive(vproj_norm, min_dist=config.vernier_scale.peak_min_dist,
                                      threshold_factor=config.vernier_scale.peak_threshold_factor)
    if len(vernier_xs) < config.vernier_scale.min_tick_count:
        return _empty_vernier_result()

    # ── 2.5 等间距补全 & 校验 ──
    #     物理约束：游标尺刻线间距严格相等（虽然比主尺略密）。
    #     游标尺刻线更密，垂直投影漏检概率更高，等间距补全至关重要。
    if config.vernier_scale.spacing_refine_enabled:
        refined_xs = refine_ticks_by_spacing(
            vernier_xs, binary,
            spacing_tolerance=config.vernier_scale.spacing_tolerance,
            gap_factor=config.vernier_scale.spacing_gap_factor,
            dup_factor=config.vernier_scale.spacing_dup_factor,
            snap_ratio=config.vernier_scale.spacing_snap_ratio,
        )
        if len(refined_xs) >= config.vernier_scale.min_tick_count:
            vernier_xs = refined_xs

    # ── 3. 精密提取刻线 ──
    vernier_ticks = extract_ticks_from_binary(
        binary, vernier_xs,
        long_tick_factor=config.vernier_scale.long_tick_factor)
    vernier_ticks.sort(key=lambda t: t['x'])

    if len(vernier_ticks) < config.vernier_scale.min_tick_count:
        return _empty_vernier_result()

    # The product now only supports 0.02 mm calipers.
    v_gap = float(np.median(np.diff([t['x'] for t in vernier_ticks])))
    precision = 0.02

    # ── 5. 零线 = 最长等间距连续段起点（避开双重检测和主尺残余）──
    #     游标尺刻度严格等间距。vernier_ticks 中往往含"双重检测"（间距 ≈ 0）
    #     和"主尺刻度残余"（间距远大于游标 gap）。找最长等间距连续段，
    #     段起点即为可信零线 x。
    zero_x_from_seq = _find_zero_from_longest_run(vernier_ticks, v_gap)
    if zero_x_from_seq is not None:
        zero_x = float(zero_x_from_seq)
        # 在 vernier_ticks 中找最接近 zero_x 的作为 zero_tick
        zero_tick = min(vernier_ticks, key=lambda t: abs(t['x'] - zero_x))
        zero_digit_found = _verify_zero_by_digit(region, zero_x)
    else:
        zero_tick, zero_digit_found = _find_zero_tick(vernier_ticks, region)
        zero_x = float(zero_tick['x']) if zero_tick else float(vernier_ticks[0]['x'])

    # ── 5.5 过滤零线左侧的伪刻线 ──
    #     游标尺刻度线必然在零线右侧。零线左侧检测到的刻线实际
    #     是主尺的刻度线（区域分离时混入游标区的），必须剔除。
    #     容差 = 游标间距 × 0.4（避免把零线自身也删掉）。
    clean_vernier = [t for t in vernier_ticks if t['x'] >= zero_x - v_gap * 0.4]
    if len(clean_vernier) >= config.vernier_scale.min_tick_count:
        vernier_ticks = clean_vernier
    vernier_ticks.sort(key=lambda t: t['x'])

    # v6.5: 过滤后重算 v_gap；precision 固定为 0.02 mm
    if len(vernier_ticks) >= 2:
        v_gap = float(np.median(np.diff([t['x'] for t in vernier_ticks])))
        precision = 0.02

    # ── 6. 对齐查找 ──
    vernier_reading, aligned_tick, align_conf = find_best_alignment(
        vernier_ticks, precision, main_ticks
    )

    # ── 可视化 ──
    vis_ticks = _draw_vernier_ticks(region, binary, vernier_ticks, vproj_norm, vernier_xs,
                                     zero_x, zero_digit_found)
    vis_alignment = _draw_alignment(region, color_region, vernier_ticks,
                                     main_gap, zero_x, aligned_tick, align_conf)

    return {
        'vernier_ticks': vernier_ticks,
        'precision': precision,
        'vernier_reading': vernier_reading,
        'zero_x': zero_x,
        'aligned_tick': aligned_tick,
        'alignment_confidence': align_conf,
        'vis_ticks': vis_ticks,
        'vis_alignment': vis_alignment,
        # 供流水线零线概览使用
        'vproj_norm': vproj_norm,
        'vernier_peaks': vernier_xs,
        'zero_digit_found': zero_digit_found,
    }


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
    from .main_scale import get_ocr_reader
    reader = get_ocr_reader()
    reader._ensure_engine()
    results = reader._ocr_single_patch(patch_bin)

    for text, conf in results:
        if text == '0' and conf > config.vernier_scale.zero_digit_conf_min:
            return True
    return False


def _find_zero_from_longest_run(vernier_ticks: list, v_gap: float):
    """
    在游标刻线序列中找最长等间距连续段，返回段起点 x 作为零线。

    流程：
      1. 去重（合并相邻间距 < v_gap_robust × 0.3 的双重检测）
      2. 估算 v_gap_robust（去掉极小极大后的中位 gap）
      3. 找最长"相邻 gap ∈ [0.6×v_gap, 1.5×v_gap]" 段
      4. 段起点 = 零线 x

    Args:
        vernier_ticks: 游标刻线列表（含 'x'）
        v_gap:         全局中位 gap（不可信，仅作初始估计）

    Returns:
        段起点 x 或 None
    """
    if not vernier_ticks or len(vernier_ticks) < 4:
        return None
    xs = sorted([int(t['x']) for t in vernier_ticks])

    # ── 1. 初步估算 v_gap（用 trimmed median）──
    diffs = np.diff(xs)
    if len(diffs) == 0:
        return None
    sorted_diffs = sorted(diffs)
    trim = max(1, len(sorted_diffs) // 5)
    trimmed = sorted_diffs[trim:-trim] if len(sorted_diffs) > 2*trim else sorted_diffs
    v_gap_robust = float(np.median(trimmed)) if trimmed else float(v_gap)
    if v_gap_robust < 2.0:
        return None

    # ── 2. 去重：合并相邻间距 < v_gap_robust × 0.3 的双重检测 ──
    dedup_th = v_gap_robust * 0.3
    deduped = [xs[0]]
    for x in xs[1:]:
        if x - deduped[-1] < dedup_th:
            # 取平均位置作为合并点
            deduped[-1] = (deduped[-1] + x) // 2
        else:
            deduped.append(x)

    if len(deduped) < 4:
        return None

    # ── 3. 重新估算 v_gap（去重后更准）──
    diffs2 = np.diff(deduped)
    sorted_d2 = sorted(diffs2)
    trim2 = max(1, len(sorted_d2) // 5)
    trimmed2 = sorted_d2[trim2:-trim2] if len(sorted_d2) > 2*trim2 else sorted_d2
    v_gap2 = float(np.median(trimmed2)) if trimmed2 else v_gap_robust
    if v_gap2 < 2.0:
        return None

    lo_th = v_gap2 * 0.55
    hi_th = v_gap2 * 1.60

    # ── 4. 找最长等间距连续段 ──
    segments = []
    cur_start = 0
    for i, d in enumerate(diffs2):
        if not (lo_th <= d <= hi_th):
            if i - cur_start >= 3:  # 至少 4 个刻度
                segments.append((cur_start, i))
            cur_start = i + 1
    if len(deduped) - 1 - cur_start >= 3:
        segments.append((cur_start, len(deduped) - 1))

    if not segments:
        return None

    # 最长段
    longest = max(segments, key=lambda s: s[1] - s[0])
    return deduped[longest[0]]


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
        precision:     卡尺精度 (0.02/0.05/0.1)
        main_ticks:    主尺刻线列表

    Returns:
        (vernier_reading, aligned_tick, confidence)
    """
    v_sorted = sorted(vernier_ticks, key=lambda t: t['x'])
    n = len(v_sorted)
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
        errors[i] = _compute_alignment_error(vt['x'], main_ticks, vy_range)

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
                         zero_digit_found: bool = False) -> np.ndarray:
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

    # Remove duplicate detections around the same physical tick.
    deduped = []
    dup_tol = max(3.0, expected * 0.35)
    for t in ticks:
        if deduped and t['x'] - deduped[-1]['x'] < dup_tol:
            if t.get('length', 0) > deduped[-1].get('length', 0):
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
            nxt = min(candidates, key=lambda t: abs(t['x'] - target))
            if nxt in seq:
                break
            seq.append(nxt)
            current_x = float(nxt['x'])

        if len(seq) > len(best):
            best = seq

    if len(best) >= max(6, config.vernier_scale.min_tick_count):
        return sorted(best, key=lambda t: t['x'])
    return vernier_ticks


def _find_zero_from_projection_valley(binary: np.ndarray,
                                      vernier_ticks: List[dict],
                                      body_x1: int,
                                      body_x2: int,
                                      main_gap: float):
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

    h, w = binary.shape[:2]
    x1 = max(0, min(w - 1, int(body_x1)))
    x2 = max(x1 + 1, min(w, int(body_x2)))
    band_h = max(12, min(h, int(h * 0.45)))
    band = binary[:band_h, x1:x2]
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
    tick_xs = sorted([x for x, v in peaks if v >= h_th])
    if len(tick_xs) < 3:
        return None, False, None

    # ── 4. 相邻峰间距 → 找显著偏大的谷底候选 ──
    diffs = np.diff(tick_xs)
    if len(diffs) < 2:
        return None, False, None

    # 用 trimmed median 估算正常刻度间距
    sorted_d = sorted(diffs)
    trim = max(1, len(sorted_d) // 5)
    trimmed = sorted_d[trim:-trim] if len(sorted_d) > 2 * trim else sorted_d
    typical_gap = float(np.median(trimmed))
    if typical_gap < 2.0:
        return None, False, None

    # 找间距 > typical_gap × 1.8 的位置，选间距最大的作为谷底
    valley_candidates = []
    for i, d in enumerate(diffs):
        if d > typical_gap * 1.8:
            valley_candidates.append((i, d, tick_xs[i], tick_xs[i + 1]))

    if not valley_candidates:
        return None, False, None

    # 选择 x 最小（最靠左）的那个谷底候选
    best_valley = min(valley_candidates, key=lambda v: v[2])  # v[2] = tick_xs[i] (左侧峰的全局x)
    gap_left_x = best_valley[2]   # 谷底左侧最后一条峰
    gap_right_x = best_valley[3]  # 谷底右侧第一条峰

    # ── 5. 第一个谷底候选区右侧第一条峰 = 零线 ──
    zero_local_x = gap_right_x
    zero_global_x = x1 + zero_local_x

    # 在 vernier_ticks 中找最接近 zero_global_x 的 tick
    ticks_in_range = [t for t in vernier_ticks if x1 - typical_gap <= t['x'] <= x2 + typical_gap]
    if ticks_in_range:
        found_tick = min(ticks_in_range, key=lambda t: abs(t['x'] - zero_global_x))
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
        cv2.putText(vis, f"Binary top band (h={band_h}px, w={band_w}px)", (margin, thumb_h + 14),
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
    from .utils import draw_projection_plot  # unused but keep import style consistent
    for cand in valley_candidates:
        _, _, left_x, right_x = cand
        cv2.rectangle(vis, (chart_left + left_x, chart_y0),
                      (chart_left + right_x, chart_y0 + chart_h), (60, 20, 30), -1)

    # 最佳谷底（选中的：第一个满足条件的，品红色框 + 标签）
    if best_valley:
        _, _, best_left_x, best_right_x = best_valley
        cv2.rectangle(vis, (chart_left + best_left_x, chart_y0),
                      (chart_left + best_right_x, chart_y0 + chart_h), (220, 50, 120), 2, cv2.LINE_AA)
        mid_gap_x = chart_left + (best_left_x + best_right_x) // 2
        cv2.putText(vis, f"1st GAP {best_right_x - best_left_x:.0f}px", (mid_gap_x - 40, chart_y0 - 8),
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
    cv2.putText(vis, "Zero-Line Valley Detection v7 (peak/valley top80% median)", (margin, band_h + 22),
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


def _count_forward_grid_hits(start_tick: dict, ticks: List[dict], expected_gap: float) -> int:
    """Count how many following ticks match the expected vernier spacing."""
    if expected_gap <= 0:
        return 0
    xs = sorted(float(t['x']) for t in ticks if t['x'] > start_tick['x'])
    if not xs:
        return 0

    tol = max(3.0, expected_gap * 0.32)
    hits = 0
    current = float(start_tick['x'])
    for _ in range(8):
        target = current + expected_gap
        candidates = [x for x in xs if target - tol <= x <= target + tol]
        if not candidates:
            break
        nxt = min(candidates, key=lambda x: abs(x - target))
        hits += 1
        current = nxt
        xs = [x for x in xs if x > current + expected_gap * 0.35]
    return hits


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

    body_x1, body_x2 = _find_vernier_body_x_range(img)
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

    if len(vernier_ticks) < config.vernier_scale.min_tick_count:
        vernier_xs = find_peaks_adaptive(
            vproj_norm,
            min_dist=config.vernier_scale.peak_min_dist,
            threshold_factor=config.vernier_scale.peak_threshold_factor,
        )
        if len(vernier_xs) < config.vernier_scale.min_tick_count:
            return _empty_vernier_result()
        if config.vernier_scale.spacing_refine_enabled:
            refined_xs = refine_ticks_by_spacing(
                vernier_xs, binary,
                spacing_tolerance=config.vernier_scale.spacing_tolerance,
                gap_factor=config.vernier_scale.spacing_gap_factor,
                dup_factor=config.vernier_scale.spacing_dup_factor,
                snap_ratio=config.vernier_scale.spacing_snap_ratio,
            )
            if len(refined_xs) >= config.vernier_scale.min_tick_count:
                vernier_xs = refined_xs
        vernier_ticks = extract_ticks_from_binary(
            binary, vernier_xs,
            long_tick_factor=config.vernier_scale.long_tick_factor)
        vernier_ticks.sort(key=lambda t: t['x'])
        if len(vernier_ticks) < config.vernier_scale.min_tick_count:
            return _empty_vernier_result()

    vernier_xs = np.array([t['x'] for t in vernier_ticks], dtype=int)
    precision = 0.02
    v_gap = float(np.median(np.diff([t['x'] for t in vernier_ticks]))) if len(vernier_ticks) >= 2 else 0.0

    zero_tick, zero_digit_found, valley_vis = _find_zero_from_projection_valley(
        binary, vernier_ticks, body_x1, body_x2, main_gap
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
        grid_ticks = _filter_vernier_ticks_by_grid(vernier_ticks, main_gap)
        if len(grid_ticks) >= config.vernier_scale.min_tick_count:
            vernier_ticks = grid_ticks
            vernier_xs = np.array([t['x'] for t in vernier_ticks], dtype=int)
            v_gap = float(np.median(np.diff([t['x'] for t in vernier_ticks]))) if len(vernier_ticks) >= 2 else v_gap

        zero_tick, zero_digit_found = _find_zero_tick(vernier_ticks, region)
        zero_x = float(zero_tick['x']) if zero_tick else float(vernier_ticks[0]['x'])

    clean_vernier = [t for t in vernier_ticks if t['x'] >= zero_x - max(v_gap, 1.0) * 0.4]
    if len(clean_vernier) >= config.vernier_scale.min_tick_count:
        vernier_ticks = clean_vernier
    vernier_ticks.sort(key=lambda t: t['x'])

    vernier_reading, aligned_tick, align_conf = find_best_alignment(
        vernier_ticks, precision, main_ticks
    )

    vis_ticks = _draw_vernier_ticks(
        region, binary, vernier_ticks, vproj_norm, vernier_xs,
        zero_x, zero_digit_found
    )
    vis_alignment = _draw_alignment(
        region, color_region, vernier_ticks,
        main_gap, zero_x, aligned_tick, align_conf
    )

    return {
        'vernier_ticks': vernier_ticks,
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
        'zero_digit_found': zero_digit_found,
    }
