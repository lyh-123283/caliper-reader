"""
OCR 模块 — 游标卡尺刻度数字识别 v4
策略: 垂直投影强峰值 → 长刻度线上方连通域搜索 → 多笔画合并 → 自适应框 → OCR
返回 (digits, debug_vis)
"""

import cv2
import numpy as np
from typing import List, Tuple
import warnings
from .result import DigitInfo
from .config import config

_HAS_TESSERACT = False
try:
    import pytesseract
    _HAS_TESSERACT = True
except ImportError:
    pass

_HAS_EASYOCR = False
try:
    import easyocr
    _HAS_EASYOCR = True
except ImportError:
    pass


# ═════════════════════════════════════════════════════════════
#  辅助函数（模块级）
# ═════════════════════════════════════════════════════════════

def _select_strong_ticks_by_projection(tick_infos: list,
                                        vproj_norm: np.ndarray,
                                        strong_factor: float = 0.5,
                                        min_keep: int = 3) -> list:
    """用垂直投影峰值强度筛选长刻度线

    原理：长刻度线在二值图中像素多 → 垂直投影值高。
    对每条刻线取其在投影曲线中的值，高于 mean + strong_factor*std 的视为"长刻度"。

    Args:
        tick_infos:    刻线列表 [{'x':...}, ...]
        vproj_norm:    归一化垂直投影 (0~1)
        strong_factor: 阈值 = mean + factor * std
        min_keep:      最少保留条数

    Returns:
        强峰值刻线列表（已按 x 排序）
    """
    n = len(vproj_norm)
    scored = []
    for t in tick_infos:
        x = int(round(t['x']))
        if 0 <= x < n:
            scored.append((float(vproj_norm[x]), t))

    if not scored:
        return []

    values = [s[0] for s in scored]
    mean_v = float(np.mean(values))
    std_v = float(np.std(values)) if len(values) >= 2 else 0.01
    threshold = mean_v + strong_factor * std_v

    strong = [t for v, t in scored if v >= threshold]

    # 保证最少条数
    if len(strong) < min_keep and len(scored) >= min_keep:
        scored.sort(key=lambda s: s[0], reverse=True)
        strong = [t for _, t in scored[:max(min_keep, len(scored) // 2)]]

    strong.sort(key=lambda t: t['x'])
    return strong


def _merge_nearby_ccs(ccs: list, tick_gap: float,
                       x_gap_ratio: float = 0.25) -> dict:
    """合并位置接近的连通域为单一数字外接矩形

    处理多笔画数字（0 中间的孔、8 上下两个圈、6/4 的分离笔画等）。
    策略：y 方向有重叠 + x 方向间距 < tick_gap * x_gap_ratio → 视为同一数字。

    合并时使用组的动态边界（而非第一个 cc），确保 3+ 组件连锁合并。
    """
    if not ccs:
        return {'x': 0, 'y': 0, 'w': 0, 'h': 0}
    if len(ccs) == 1:
        c = ccs[0]
        return {'x': c['x'], 'y': c['y'], 'w': c['w'], 'h': c['h']}

    x_gap_th = tick_gap * x_gap_ratio

    ccs_sorted = sorted(ccs, key=lambda c: c['x'])
    used = set()
    groups = []

    for i, first in enumerate(ccs_sorted):
        if i in used:
            continue
        group = [first]
        used.add(i)
        # 组边界随合并动态扩展
        g_x_min, g_x_max = first['x'], first['x'] + first['w']
        g_y_min, g_y_max = first['y'], first['y'] + first['h']

        # 多轮扫描直到没有新成员加入
        changed = True
        while changed:
            changed = False
            for j in range(i + 1, len(ccs_sorted)):
                if j in used:
                    continue
                other = ccs_sorted[j]
                # y 与组边界有重叠
                y_overlap = not (g_y_max < other['y'] or
                                 other['y'] + other['h'] < g_y_min)
                if not y_overlap:
                    continue
                # x 与组边界足够近（other 在 group 右侧或重叠）
                x_gap = other['x'] - g_x_max
                if x_gap >= 0 and x_gap < x_gap_th:
                    group.append(other)
                    used.add(j)
                    g_x_max = max(g_x_max, other['x'] + other['w'])
                    g_x_min = min(g_x_min, other['x'])
                    g_y_max = max(g_y_max, other['y'] + other['h'])
                    g_y_min = min(g_y_min, other['y'])
                    changed = True
                # other 在 group 左侧
                x_gap_left = g_x_min - (other['x'] + other['w'])
                if x_gap_left >= 0 and x_gap_left < x_gap_th:
                    group.append(other)
                    used.add(j)
                    g_x_min = min(g_x_min, other['x'])
                    g_x_max = max(g_x_max, other['x'] + other['w'])
                    g_y_max = max(g_y_max, other['y'] + other['h'])
                    g_y_min = min(g_y_min, other['y'])
                    changed = True
        groups.append(group)

    # 取总面积最大的组
    best = max(groups, key=lambda g: sum(c['area'] for c in g))

    x_min = min(c['x'] for c in best)
    y_min = min(c['y'] for c in best)
    x_max = max(c['x'] + c['w'] for c in best)
    y_max = max(c['y'] + c['h'] for c in best)

    return {'x': x_min, 'y': y_min, 'w': x_max - x_min, 'h': y_max - y_min}


class DigitReader:
    """v2 — locate patches around ticks then OCR each"""

    def __init__(self):
        self._easyocr = None
        self._engine = None  # None = 未初始化，在 _ensure_engine 中决定
        self._debug_patches = []
        self._engine_status = ""  # 诊断信息

    def engine_name(self) -> str:
        self._ensure_engine()
        return self._engine

    def engine_status(self) -> str:
        """返回引擎初始化诊断信息（供 GUI 显示）"""
        self._ensure_engine()
        return self._engine_status

    def read(self, gray_region: np.ndarray,
             color_region: np.ndarray = None,
             long_tick_xs: List[int] = None,
             digit_zone: str = 'above',
             tick_infos: List[dict] = None,
             binary: np.ndarray = None,
             vproj_norm: np.ndarray = None) -> Tuple[List[DigitInfo], np.ndarray]:
        """主入口: 返回 (digits, debug_vis)

        v4: 利用垂直投影强峰值自动筛选长刻度线，做连通域搜索，
            通过合并多笔画连通域 + 自适应外接矩形来精准定位数字框。
        v6.5: digit_zone 默认 'above'（在长刻度上方的 1mm 主刻度区搜数字）。
            本卡尺游标尺文字（0~9 数字）虽然视觉上在刻度下方，但 OCR 处理的
            是主尺数字（cm 标记 0~12），它们在主尺刻度的上方（与主尺刻度
            共占主尺行的上半部分）。'auto' 双向会引入游标尺的伪识别，
            默认 'above' 效果最稳。
        """
        self._ensure_engine()
        self._debug_patches = []
        work_color = (color_region if color_region is not None
                      else cv2.cvtColor(gray_region, cv2.COLOR_GRAY2BGR))

        candidates = self._locate_candidate_regions(
            gray_region, work_color, long_tick_xs, digit_zone,
            tick_infos, binary, vproj_norm)

        digits = []
        for cand in candidates:
            ocr = self._ocr_single_patch(cand['patch'])
            for text, conf in ocr:
                digits.append(DigitInfo(
                    x=cand['cx'], y=cand['cy'], value=int(text),
                    text=text, confidence=conf,
                    bbox=(cand['x1'], cand['y1'], cand['x2'], cand['y2'])))

        if not digits and long_tick_xs:
            digits = self._infer_from_positions(gray_region, long_tick_xs)

        debug_vis = self._make_digit_vis(work_color, candidates, digits,
                                          long_tick_xs, digit_zone)
        return sorted(digits, key=lambda d: d.x), debug_vis

    # ---- locate patches (v4: 投影强峰值 + 上方连通域自适应框) ----

    def _locate_candidate_regions(self, gray, color, long_tick_xs, digit_zone,
                                    tick_infos, binary, vproj_norm):
        """v4: 垂直投影强峰值 → 长刻度线 → 连通域搜索 → 自适应外接框
        v6.5: 支持 digit_zone='above'/'below'/'auto'（双向都搜 + 合并候选）
        """
        h, w = gray.shape
        candidates = []
        if not tick_infos:
            return candidates

        # ── 1. 筛选长刻度线：优先用投影峰值强度 ──
        if vproj_norm is not None and len(vproj_norm) > 0:
            strong_ticks = _select_strong_ticks_by_projection(
                tick_infos, vproj_norm,
                strong_factor=config.ocr.projection_strong_factor,
                min_keep=config.ocr.projection_min_strong,
            )
            # 投影筛选为空时回退到 is_long 判定
            if not strong_ticks and long_tick_xs:
                strong_ticks = [t for t in tick_infos if t.get('is_long', False)]
            if not strong_ticks and tick_infos:
                strong_ticks = [tick_infos[0], tick_infos[-1]]
        elif long_tick_xs:
            # 回退：用 extract_ticks 的 is_long 判定
            strong_ticks = [t for t in tick_infos if t.get('is_long', False)]
            if not strong_ticks and tick_infos:
                strong_ticks = [tick_infos[0], tick_infos[-1]]  # 至少取首尾
        else:
            return candidates

        if not strong_ticks:
            return candidates

        # ── 2. 准备二值图 ──
        if binary is not None:
            bw = binary.copy()
        else:
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        tick_gap = float(np.median(np.diff(sorted([t['x'] for t in tick_infos]))))
        if tick_gap <= 0:
            tick_gap = w / 10.0

        locked_cy = None  # 首个成功定位的 y 中心，后续复用

        for t in strong_ticks:
            tx = int(t['x'])
            if tx < 3 or tx > w - 3:
                continue

            # ── 3. 搜索窗口：根据 digit_zone 决定 y 方向 ──
            #     v6.5: 支持 'above' / 'below' / 'auto'（双向）
            search_x1 = max(0, tx - int(tick_gap * config.ocr.search_tick_gap_ratio))
            search_x2 = min(w, tx + int(tick_gap * config.ocr.search_tick_gap_ratio))
            tick_top = t.get('y_start', int(h * 0.6))
            tick_bottom = t.get('y_end', int(h * 0.6))

            # 根据 zone 决定 y 范围
            if digit_zone == 'above':
                search_y1 = 0
                search_y2 = max(tick_top + int(tick_gap * 0.15),
                                int(h * config.ocr.search_y_min_ratio),
                                config.ocr.search_y_min_height)
            elif digit_zone == 'below':
                search_y1 = tick_bottom - int(tick_gap * 0.15)
                search_y2 = min(h, tick_bottom + int(tick_gap * 2.0))
            else:
                # auto：双向都搜（合并为一个大搜索框，让连通域分析自己选）
                # 这样不论数字在上方还是下方都能找到
                search_y1 = 0
                search_y2 = min(h, tick_bottom + int(tick_gap * 2.0))

            if search_x2 <= search_x1 or search_y2 <= search_y1:
                continue

            region_bw = bw[search_y1:search_y2, search_x1:search_x2]
            rH, rW = region_bw.shape
            if rH < 8 or rW < 8:
                continue

            # ── 4. 连通域搜索，收集所有候选 ──
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                region_bw, connectivity=8)

            valid_ccs = []
            for j in range(1, num_labels):
                area = int(stats[j, cv2.CC_STAT_AREA])
                rx = int(stats[j, cv2.CC_STAT_LEFT])
                ry = int(stats[j, cv2.CC_STAT_TOP])
                rw_c = int(stats[j, cv2.CC_STAT_WIDTH])
                rh_c = int(stats[j, cv2.CC_STAT_HEIGHT])

                if area < config.ocr.cc_min_area:
                    continue
                if rw_c < config.ocr.cc_min_width or rh_c < config.ocr.cc_min_height:
                    continue
                aspect = rh_c / max(rw_c, 1)
                if aspect < config.ocr.cc_aspect_min or aspect > config.ocr.cc_aspect_max:
                    continue

                valid_ccs.append({
                    'x': rx, 'y': ry, 'w': rw_c, 'h': rh_c,
                    'area': area, 'cx': rx + rw_c // 2, 'cy': ry + rh_c // 2,
                })

            if valid_ccs:
                # ── 5a. 合并多笔画连通域 → 自适应外接矩形 ──
                merged = _merge_nearby_ccs(valid_ccs, tick_gap,
                                            x_gap_ratio=config.ocr.merge_x_gap_ratio)

                rx, ry, rw_c, rh_c = merged['x'], merged['y'], merged['w'], merged['h']
                cx = search_x1 + rx + rw_c // 2
                cy = search_y1 + ry + rh_c // 2

                # 自适应 padding：按数字大小比例缩放
                pad = max(config.ocr.pad_min,
                          min(rw_c, rh_c) // config.ocr.pad_divisor)
                x1 = max(0, search_x1 + max(0, rx - pad))
                y1 = max(0, search_y1 + max(0, ry - pad))
                x2 = min(w, search_x1 + min(rx + rw_c + pad, rW))
                y2 = min(h, search_y1 + min(ry + rh_c + pad, rH))

                if locked_cy is None:
                    locked_cy = cy

            elif locked_cy is not None:
                # ── 5b. 回退 A：用之前锁定的 y ──
                half_w = max(config.ocr.fallback_x_half_min,
                             int(tick_gap * config.ocr.fallback_x_half_ratio))
                half_h = max(config.ocr.fallback_x_half_min,
                             int(h * config.ocr.fallback_y_h_ratio))
                x1 = max(0, tx - half_w)
                x2 = min(w, tx + half_w)
                y2_limit = int(h * config.ocr.fallback_y_end_ratio)
                y2_fb = min(y2_limit, locked_cy + half_h)
                y1 = max(0, y2_fb - 2 * half_h)
                y2 = y2_fb
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

            else:
                # ── 5c. 回退 B：固定比例取刻线上方区域 ──
                half_w = max(config.ocr.fallback_x_half_min,
                             int(tick_gap * config.ocr.fallback_x_half_ratio))
                x1 = max(0, tx - half_w)
                x2 = min(w, tx + half_w)
                y2_limit = int(h * config.ocr.fallback_y_end_ratio)
                half_h = max(config.ocr.fallback_x_half_min,
                             int(h * config.ocr.fallback_y_h_ratio))
                y1 = max(0, y2_limit - 2 * half_h)
                y2 = min(h, y2_limit + half_h // 2)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

            # ── 6. 提取补丁 → 增强 → 记入候选 ──
            if y2 <= y1 or x2 <= x1:
                continue
            patch = self._enhance_patch(gray[y1:y2, x1:x2])
            self._debug_patches.append(patch)
            cv2.rectangle(color, (x1, y1), (x2, y2), (255, 160, 40), 2)
            candidates.append({
                'cx': cx, 'cy': cy, 'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'pw': x2 - x1, 'ph': y2 - y1, 'patch': patch,
            })
        return candidates

    def read_digit_at(self, gray_region: np.ndarray,
                      target_x: int,
                      binary: np.ndarray = None,
                      tick_gap: float = None) -> 'DigitInfo':
        """v6.5: 定向 OCR — 识别指定 x 位置**正上方**的数字（不再扫所有 tick）。

        用途：游标 0 刻度线 (zero_x) 左侧最近的主尺 cm 整数标记。
        物理：主尺数字在刻线上方 0~30% 主尺行高范围（DELIXI 下方）。

        Args:
            gray_region: 主尺 region 灰度图
            target_x:   目标数字的 x 像素坐标（cm 整数刻度位置）
            binary:     区域二值图（用于连通域分析；None 则内部 OTSU）
            tick_gap:   1mm 像素间距（用于搜索半宽；None 估算）

        Returns:
            DigitInfo | None（识别失败返回 None）
        """
        h, w = gray_region.shape[:2]
        if target_x < 5 or target_x > w - 5:
            return None

        # 估算 tick_gap（如果没传）
        if tick_gap is None or tick_gap <= 0:
            tick_gap = max(20.0, w / 100.0)

        # 准备二值图
        if binary is None:
            binary = cv2.adaptiveThreshold(
                gray_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 5)
        bw = binary

        # 搜索范围：x ± tick_gap*1.0，y 从 0 到 h*0.55（数字大致在主尺行上半部）
        half_w = int(tick_gap * 1.0)
        y2_limit = int(h * 0.55)
        if y2_limit < 8 or half_w < 4:
            return None
        search_x1 = max(0, target_x - half_w)
        search_x2 = min(w, target_x + half_w)
        if search_x2 <= search_x1:
            return None
        region_bw = bw[:y2_limit, search_x1:search_x2]
        rH, rW = region_bw.shape

        # 找连通域
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(region_bw, connectivity=8)
        valid_ccs = []
        for j in range(1, num_labels):
            area = int(stats[j, cv2.CC_STAT_AREA])
            rx = int(stats[j, cv2.CC_STAT_LEFT])
            ry = int(stats[j, cv2.CC_STAT_TOP])
            rw_c = int(stats[j, cv2.CC_STAT_WIDTH])
            rh_c = int(stats[j, cv2.CC_STAT_HEIGHT])
            # 数字最小尺寸（防噪点）
            if area < 8 or rw_c < 3 or rh_c < 5:
                continue
            valid_ccs.append({
                'x': rx, 'y': ry, 'w': rw_c, 'h': rh_c,
                'area': area, 'cx': rx + rw_c // 2, 'cy': ry + rh_c // 2,
            })

        if not valid_ccs:
            return None

        # 合并多笔画（0 8 6 等）
        merged = _merge_nearby_ccs(
            valid_ccs, tick_gap,
            x_gap_ratio=config.ocr.merge_x_gap_ratio)

        rx, ry, rw_c, rh_c = merged['x'], merged['y'], merged['w'], merged['h']
        cx = search_x1 + rx + rw_c // 2
        cy = ry + rh_c // 2

        # 自适应 padding
        pad = max(config.ocr.pad_min,
                  min(rw_c, rh_c) // config.ocr.pad_divisor)
        x1 = max(0, search_x1 + max(0, rx - pad))
        y1 = max(0, ry - pad)
        x2 = min(w, search_x1 + min(rx + rw_c + pad, rW))
        y2 = min(h, ry + rh_c + pad)

        if y2 <= y1 or x2 <= x1:
            return None

        # 增强 patch
        patch = self._enhance_patch(gray_region[y1:y2, x1:x2])

        # OCR
        results = self._ocr_single_patch(patch)
        for text, conf in results:
            if not text.isdigit():
                continue
            value = int(text)
            if value > 15:
                continue
            return DigitInfo(
                x=cx, y=cy, value=value,
                text=text, confidence=conf,
                bbox=(x1, y1, x2, y2))
        return None

    def _enhance_patch(self, gp):
        ph, pw = gp.shape
        if ph < 8 or pw < 8: return gp
        w = cv2.resize(gp, (pw * config.ocr.patch_resize_factor, ph * config.ocr.patch_resize_factor),
                       interpolation=cv2.INTER_CUBIC)
        w = cv2.createCLAHE(clipLimit=config.ocr.patch_clahe_clip, tileGridSize=(4, 4)).apply(w)
        # v6.5: block_size 按 patch 大小自适应（避免 11/24=46% 局部窗口过大）
        # 公式：block = clamp(min(resized_w, resized_h) / 5, [3, 11])，保证奇数
        resized_min = min(w.shape[:2])
        block = max(3, min(11, resized_min // 5)) | 1
        b = cv2.adaptiveThreshold(w, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block,
                                   config.ocr.patch_adaptive_C)
        if np.sum(b < 128) > np.sum(b >= 128):
            b = cv2.bitwise_not(b)
        return b

    # ---- OCR ----

    def _ocr_single_patch(self, patch):
        if self._engine == 'tesseract':
            return self._ocr_tess(patch)
        if self._engine == 'easyocr' and self._easyocr:
            return self._ocr_easy(patch)
        return []

    def ocr_patch_to_digit(self, patch: np.ndarray,
                              bbox: tuple,
                              gray_region: np.ndarray = None) -> 'DigitInfo':
        """v6.6: 接收已选定的数字 patch + bbox，OCR 后返回 DigitInfo。

        适用于上游已经通过连通域分析
        选定了数字位置+裁剪好 patch 的场景。

        Args:
            patch:       二值或灰度数字 patch
            bbox:        patch 在原图中的 (x1, y1, x2, y2)
            gray_region: 原始灰度图（用于 patch 增强前取对应区域）

        Returns:
            DigitInfo 或 None
        """
        if patch is None or patch.size == 0:
            return None
        self._ensure_engine()
        patch_variants = [patch.copy()]
        if len(patch.shape) == 2:
            patch_variants.append(cv2.bitwise_not(patch))
        if gray_region is not None and bbox is not None:
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, gray_region.shape[1] - 1))
            x2 = max(0, min(x2, gray_region.shape[1]))
            y1 = max(0, min(y1, gray_region.shape[0] - 1))
            y2 = max(0, min(y2, gray_region.shape[0]))
            gray_patch = gray_region[y1:y2, x1:x2]
            if gray_patch.size > 0:
                patch_variants.append(self._enhance_patch(gray_patch))

        results = []
        for candidate_patch in patch_variants:
            results = self._ocr_single_patch(candidate_patch)
            if results:
                break
        for text, conf in results:
            if not text.isdigit():
                continue
            value = int(text)
            if value > 15:
                continue
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            return DigitInfo(
                x=cx, y=cy, value=value,
                text=text, confidence=conf,
                bbox=(x1, y1, x2, y2))
        return None

    def _ocr_tess(self, patch):
        """v6.5: 保持 PSM 8 单字符识别（最稳定），不取置信度。"""
        try:
            t = pytesseract.image_to_string(
                patch,
                config='--psm {psm} -c tessedit_char_whitelist={wl}'.format(
                    psm=config.ocr.tesseract_psm,
                    wl=config.ocr.tesseract_whitelist)).strip()
        except Exception:
            return []
        return [(t, 0.7)] if t.isdigit() else []

    def _ocr_easy(self, patch):
        inp = patch if len(patch.shape) == 3 else cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        try:
            res = self._easyocr.readtext(inp, allowlist=config.ocr.easyocr_allowlist,
                                          paragraph=False, min_size=config.ocr.easyocr_min_size,
                                          text_threshold=config.ocr.easyocr_text_threshold,
                                          low_text=config.ocr.easyocr_low_text)
        except Exception:
            return []
        return [(t.strip(), c) for _, t, c in res
                if t.strip().isdigit() and c > config.ocr.easyocr_min_conf]

    def _ensure_engine(self):
        """延迟初始化 OCR 引擎，按优先级尝试：tesseract → easyocr → fallback"""
        if self._engine is not None:
            return

        # ── 1) 尝试 tesseract ──
        if _HAS_TESSERACT:
            try:
                # 验证 tesseract 二进制可用
                v = pytesseract.get_tesseract_version()
                self._engine = 'tesseract'
                self._engine_status = f"Tesseract {v}"
                return
            except Exception as e:
                warnings.warn(f"pytesseract 已安装但 Tesseract OCR 二进制不可用: {e}")

        # ── 2) 尝试 easyocr ──
        if _HAS_EASYOCR:
            try:
                self._easyocr = easyocr.Reader(['en'], gpu=False, verbose=False)
                self._engine = 'easyocr'
                self._engine_status = "EasyOCR (CPU)"
                return
            except Exception as e:
                warnings.warn(f"EasyOCR 初始化失败: {e}")

        # ── 3) fallback ──
        self._engine = 'fallback'
        self._engine_status = "无 OCR 引擎 (仅位置推断)"
        warnings.warn("未找到可用的 OCR 引擎。请安装: pip install pytesseract + Tesseract二进制, "
                       "或 pip install easyocr")

    def _infer_from_positions(self, gray, ts):
        h = gray.shape[0]
        xs = sorted(ts)
        if len(xs) < 2: return []
        g = np.median(np.diff(xs))
        if g <= 0: return []
        half_w = max(config.ocr.fallback_x_half_min,
                     int(g * config.ocr.fallback_x_half_ratio))
        y2 = int(h * config.ocr.fallback_y_end_ratio)
        half_h = max(config.ocr.fallback_x_half_min,
                     int(h * config.ocr.fallback_y_h_ratio))
        y1 = max(0, y2 - 2 * half_h)
        return [DigitInfo(
            x=int(tx - half_w), y=(y1 + y2) // 2, value=-1,
            text="?", confidence=0.3,
            bbox=(int(tx - half_w), y1, int(tx + half_w), min(y2, h)))
                for tx in xs]

    # ---- visualization ----

    def _make_digit_vis(self, color, candidates, digits, ts, zone):
        h, w = color.shape[:2]
        left = color.copy()
        if ts:
            for t in ts:
                cv2.line(left, (int(t), 0), (int(t), h - 1), (80, 80, 90), 1)
        for c in candidates:
            cv2.rectangle(left, (c['x1'], c['y1']), (c['x2'], c['y2']), (255, 160, 40), 2)
        for d in digits:
            if d.value >= 0:
                cv2.putText(left, str(d.value), (d.x - 10, d.y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 3)
        n_ok = sum(1 for d in digits if d.value >= 0)
        cv2.putText(left, f"S1: {len(candidates)} regions", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 180, 50), 1)
        cv2.putText(left, f"S2: {n_ok} digits ({self._engine})", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 100), 1)
        right = self._patches_panel(h)
        gap = 4
        oh = max(h, right.shape[0])
        ow = w + right.shape[1] + gap
        out = np.zeros((oh, ow, 3), dtype=np.uint8)
        out[:] = (30, 30, 35)
        out[:h, :w] = left
        out[:right.shape[0], w + gap:w + gap + right.shape[1]] = right
        cv2.putText(out, f"OCR ({zone})", (5, oh - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 125), 1)
        return out

    def _patches_panel(self, th):
        if not self._debug_patches:
            return np.zeros((th, 110, 3), dtype=np.uint8)
        pw, ph = 90, 36
        vis = np.zeros((th, pw + 20, 3), dtype=np.uint8)
        vis[:] = (25, 25, 30)
        y = 6
        for i, p in enumerate(self._debug_patches):
            if y + ph > th: break
            if len(p.shape) == 2: p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
            vis[y:y + ph, 8:8 + pw] = cv2.resize(p, (pw, ph), interpolation=cv2.INTER_AREA)
            cv2.putText(vis, f"#{i}", (2, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1)
            y += ph + 3
        cv2.putText(vis, "Patches", (2, th - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 105), 1)
        return vis


# v6.5: 供 merger 直接调定向 OCR 的全局单例 + 懒加载
_OCR_READER_SINGLETON: 'DigitReader' = None


def get_ocr_reader_singleton() -> 'DigitReader':
    """懒加载全局 OCR 单例。"""
    global _OCR_READER_SINGLETON
    if _OCR_READER_SINGLETON is None:
        _OCR_READER_SINGLETON = DigitReader()
    return _OCR_READER_SINGLETON
