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

from .utils import find_peaks_adaptive, extract_ticks_from_binary, draw_projection_plot, refine_ticks_by_spacing
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
    img = region['image']
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

    # ── 4. 间距 → 推断精度 ──
    v_gap = float(np.median(np.diff([t['x'] for t in vernier_ticks])))
    precision = infer_precision(main_gap, v_gap, len(vernier_ticks))

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

    # v6.5: 过滤后重算 v_gap + precision（之前 v_gap 用过滤前数据算的）
    if len(vernier_ticks) >= 2:
        v_gap = float(np.median(np.diff([t['x'] for t in vernier_ticks])))
        precision = infer_precision(main_gap, v_gap, len(vernier_ticks))

    # ── 6. 对齐查找 v2 ──
    vernier_reading, aligned_tick, align_conf = find_best_alignment(
        vernier_ticks, main_gap, precision, zero_x, main_ticks
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


# ═══════════════════════════ 精度推断 ═══════════════════════════

def infer_precision(main_gap: float, v_gap: float, tick_count: int = 0) -> float:
    """从刻线数量 + 间距比推断精度（10/20/50分度）"""
    if main_gap <= 0:
        return 0.02

    # ── 方法1（最可靠）：直接数游标尺上的刻线条数 ──
    # 50分度 = ~50条线, 20分度 = ~20条线, 10分度 = ~10条线
    if tick_count >= config.vernier_scale.precision_tick_50:
        return 0.02
    elif tick_count >= config.vernier_scale.precision_tick_20:
        return 0.05
    elif tick_count >= config.vernier_scale.precision_tick_10:
        return 0.1

    # ── 方法2（回退）：间距比 ──
    if v_gap <= 0:
        return 0.02
    ratio = min(v_gap / main_gap, 0.99)
    candidates = {10: 0.900, 20: 0.950, 50: 0.980}
    best_div, best_diff = 50, float('inf')
    for div, expected in candidates.items():
        diff = abs(ratio - expected)
        if diff < best_diff:
            best_diff = diff
            best_div = div
    p = 1.0 / best_div
    if p >= 0.09: return 0.1
    elif p >= 0.045: return 0.05
    return 0.02


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
                         main_gap: float,
                         precision: float,
                         zero_x: float,
                         main_ticks: List[dict] = None) -> Tuple[float, dict, float]:
    """
    找到与主尺刻度线最对齐的游标刻线（游标读数）。

    v2 改进:
      a) 使用真实主尺刻度线位置，而非合成网格线
      b) 对每条游标刻线，在游标区域内找最近的主尺刻线，计算误差
      c) 亚像素抛物线插值细化对齐位置
      d) 输出置信度评分

    物理原理:
      游标尺有 N 条等距刻线，对应主尺 (N-1) 条刻线跨度。
      游标第 i 条线与主尺某条线对齐时，小数读数 = i * precision。

    Args:
        vernier_ticks: 游标刻线列表
        main_gap:      主尺像素间距
        precision:     卡尺精度 (0.02/0.05/0.1)
        zero_x:        零线 x 坐标
        main_ticks:    主尺刻线列表（可选；None 则退化为合成网格法）

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

    # ── 对每条游标刻线，计算与最近主尺刻线的像素误差 ──
    errors = np.zeros(n)
    for i, vt in enumerate(v_sorted):
        if main_ticks and len(main_ticks) >= 2:
            # v2: 用真实主尺刻线（增加 Y 方向过滤）
            errors[i] = _compute_alignment_error_real(vt['x'], main_ticks, vy_range)
        else:
            # 回退：合成网格
            errors[i] = _compute_alignment_error_grid(vt['x'], main_gap, zero_x)

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


def _compute_alignment_error_real(vx: float,
                                   main_ticks: List[dict],
                                   vernier_region_y_range: tuple = None) -> float:
    """
    计算游标线 vx 与最近主尺刻度线的像素距离。
    只在 y 方向有物理重叠的刻线之间计算对齐误差。
    """
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


def _compute_alignment_error_grid(vx: float, main_gap: float,
                                   zero_x: float) -> float:
    """合成主尺网格线，计算游标线 vx 与最近网格线的距离"""
    offset = zero_x % main_gap
    nearest_grid = round((vx - offset) / main_gap) * main_gap + offset
    return abs(vx - nearest_grid)


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
