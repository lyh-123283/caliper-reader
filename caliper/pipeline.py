"""
流水线主控 — 串联所有步骤

流水线:
  步骤 0: 图像预处理（增强 + 滤波 + 二值化）
  步骤 1: ROI提取 + 方向矫正
  步骤 2: 区域分离（主尺 / 游标尺）
  步骤 3: 主尺识别（刻度线 + 数字OCR）
  步骤 4: 游标尺识别（刻度线 + 对齐 + 数字OCR）
  步骤 5: 读数合并 + 最终标注
"""

import cv2
import numpy as np

from .result import CaliperResult
from .preprocess import preprocess
from .roi_extract import locate_roi_lowres, orient_caliper
from .region_split import split_scales
from .main_scale import recognize_main_scale
from .vernier_scale import recognize_vernier_scale
from .merger import merge_readings
from .utils import draw_legend_below, draw_projection_plot
from .config import config


class CaliperPipeline:
    """游标卡尺识别流水线"""

    def __init__(self):
        self.debug_images = {}
        self.step_results = {}

        # ── 预处理参数（可通过 config.preprocess.xxx 修改）──
        self.preprocess_params = {
            'clip_limit': config.preprocess.clahe_clip_limit,
            'bilateral_d': config.preprocess.bilateral_d,
            'bilateral_sigma': config.preprocess.bilateral_sigma,
            'gamma': config.preprocess.gamma,
            'median_ksize': config.preprocess.median_ksize,
        }

    def _emit_progress(self, progress_callback, step_key: str, status: str):
        if progress_callback is None:
            return
        image = self.debug_images.get(step_key)
        if image is not None:
            progress_callback(step_key, image, status)

    def run(self, img: np.ndarray, progress_callback=None) -> CaliperResult:
        """
        执行完整流水线

        Args:
            img: BGR 彩色图像 (numpy array)

        Returns:
            CaliperResult 识别结果（含所有中间调试图像）
        """
        self.debug_images = {}
        self.step_results = {}
        original = img.copy()

        # ═══════════════════════════════════════
        #  步骤 0: 图像预处理
        # ═══════════════════════════════════════
        roi_result = locate_roi_lowres(img)
        if roi_result['roi_color'] is None:
            return self._fail(original, "ROI 提取失败")
        self.debug_images['1a_ROI定位'] = roi_result.get('lowres_debug')
        self._emit_progress(progress_callback, '1a_ROI定位', 'ROI 定位完成')

        pp = preprocess(roi_result['roi_color'], **self.preprocess_params)
        roi_result['roi_color'] = pp['color']
        roi_result['roi_gray'] = pp['enhanced']
        roi_result['roi_binary'] = pp['binary_adaptive']
        self.debug_images['0_预处理'] = pp['debug_vis']
        self.step_results['preprocess'] = pp
        self._emit_progress(progress_callback, '0_预处理', '预处理完成')
        self.step_results['roi'] = roi_result

        orient_result = orient_caliper(
            roi_result['roi_color'],
            roi_result['roi_gray'],
            roi_result['roi_binary']
        )
        self.debug_images['1b_方向校正'] = orient_result['orient_vis']
        self.step_results['orient'] = orient_result
        self._emit_progress(progress_callback, '1b_方向校正', '方向校正完成')
        return self._run_remainder(original, orient_result, progress_callback)

    def _run_remainder(self, original: np.ndarray,
                       orient_result: dict,
                       progress_callback=None) -> CaliperResult:
        """执行流水线剩余部分（步骤 2~5）。"""
        rotated_color = orient_result['rotated_color']
        rotated_gray = orient_result['rotated_gray']
        rotated_binary = orient_result['rotated_binary']

        # 步骤 2
        split_result = split_scales(rotated_gray, rotated_binary, rotated_color)
        self.debug_images['2_区域分离'] = split_result['split_vis']
        self.step_results['split'] = split_result
        self._emit_progress(progress_callback, '2_区域分离', '区域分离完成')
        region_main = split_result['region_main']
        region_vernier = split_result['region_vernier']
        split_y = split_result['split_y']

        # 步骤 3
        main_color = rotated_color[:split_y, :]
        main_result = recognize_main_scale(region_main, main_color)
        self.debug_images['3a_主尺刻度线'] = main_result['vis_ticks']
        self.step_results['main'] = main_result
        self._emit_progress(progress_callback, '3a_主尺刻度线', '主尺刻线识别完成')

        # 步骤 4
        vernier_color = rotated_color[split_y:, :]
        vernier_result = recognize_vernier_scale(
            region_vernier, main_result['main_gap'], vernier_color,
            main_result['main_ticks']
        )
        # ── 生成零线验证概览 ──
        overview = _make_zero_overview(rotated_color, main_result, vernier_result,
                                        split_y, region_main, region_vernier)
        self.debug_images['3a_主尺刻度线'] = overview
        self._emit_progress(progress_callback, '3a_主尺刻度线', '零线总览完成')
        self.debug_images['4b_游标刻度线'] = vernier_result['vis_ticks']
        self._emit_progress(progress_callback, '4b_游标刻度线', '游标刻线识别完成')
        # ── 对齐可视化：用整张 ROI 图（含主尺真实网格）──
        vernier_result['vis_alignment'] = _regenerate_alignment_vis(
            vernier_result, vernier_color, rotated_color,
            split_y, main_result['main_ticks'], main_result['main_gap'])
        self.debug_images['4c_游标对齐'] = vernier_result['vis_alignment']
        self.step_results['vernier'] = vernier_result
        self._emit_progress(progress_callback, '4c_游标对齐', '游标对齐完成')

        # ── 添加图例（在对齐图重新生成之后）──
        _add_legends(main_result, vernier_result)

        # 步骤 5
        final = merge_readings(
            main_result, vernier_result,
            rotated_color, region_main, region_vernier, split_y
        )

        # ── 生成 OCR 调试图（替换空白占位）──
        ocr_debug_vis = _make_ocr_debug_vis(
            rotated_color, split_y, region_main,
            main_result, vernier_result, final)
        if ocr_debug_vis is not None:
            self.debug_images['3b_主尺数字OCR'] = ocr_debug_vis
            self._emit_progress(progress_callback, '3b_主尺数字OCR', 'OCR 调试图完成')

        final.debug_images = self.debug_images
        self.debug_images['5_最终标注'] = final.image_annotated
        self._emit_progress(progress_callback, '5_最终标注', '最终标注完成')
        # 添加读数推导可视化
        deriv_vis = final.extra_info.get('derivation_vis')
        if deriv_vis is not None:
            self.debug_images['5b_读数推导'] = deriv_vis
            self._emit_progress(progress_callback, '5b_读数推导', '读数推导完成')

        return final

    def _fail(self, img: np.ndarray, reason: str) -> CaliperResult:
        """生成失败结果"""
        result = CaliperResult(
            main_scale=0.0,
            vernier_scale=0.0,
            total=0.0,
            precision=0.02,
            confidence=0.0,
            image_annotated=img,
            debug_images={'error': img},
            extra_info={'error': reason},
        )
        return result


# ═══════════════════════════════════════════════════
#  图例 & 概览辅助函数
# ═══════════════════════════════════════════════════

def _regenerate_alignment_vis(vernier_result: dict,
                                vernier_color: np.ndarray,
                                rotated_color: np.ndarray,
                                split_y: int,
                                main_ticks: list,
                                main_gap: float) -> np.ndarray:
    """用整张 ROI 彩色图重新生成对齐可视化（含主尺真实网格线）"""
    from .vernier_scale import _draw_alignment
    return _draw_alignment(
        {'y_offset': split_y},
        vernier_color,               # color_region 回退
        vernier_result['vernier_ticks'],
        main_gap,
        vernier_result['zero_x'],
        vernier_result.get('aligned_tick'),
        vernier_result.get('alignment_confidence', 0.0),
        full_color=rotated_color,
        split_y=split_y,
        main_ticks=main_ticks,
    )


def _add_legends(main_result: dict, vernier_result: dict):
    """在关键可视化图像下方添加中文图例面板"""
    # ── 游标刻度线图 ──
    vis_vt = vernier_result.get('vis_ticks')
    if vis_vt is not None and vis_vt.size > 0:
        items = [
            ("orange line = vernier tick", (200, 160, 40), 'line'),
            ("blue line = zero (0th tick)", (50, 150, 255), 'line'),
        ]
        vernier_result['vis_ticks'] = draw_legend_below(vis_vt, items)

    # ── 游标对齐图 ──
    vis_va = vernier_result.get('vis_alignment')
    if vis_va is not None and vis_va.size > 0:
        items = [
            ("gray line = main scale tick", (80, 80, 90), 'line'),
            ("orange line = vernier tick", (200, 160, 40), 'line'),
            ("blue line = zero (crossing full ROI)", (50, 150, 255), 'line'),
            ("thick green = best alignment", (0, 255, 80), 'line'),
            ("yellow dash = main/vernier split", (255, 255, 100), 'line'),
        ]
        vernier_result['vis_alignment'] = draw_legend_below(vis_va, items)

    # ── 主尺数字OCR图（图例稍后由 _make_ocr_debug_vis 自行处理）──


def _make_ocr_debug_vis(rotated_color: np.ndarray,
                         split_y: int,
                         region_main: dict,
                         main_result: dict,
                         vernier_result: dict,
                         final_result) -> np.ndarray:
    """生成主尺 OCR 调试图 — 三合一纵向拼接：
       上: 主尺彩色图 + 备选区红框 + 零线黄线
       中: 备选区裁出 blow up
       下: 连通域筛选 + OCR 结果
    """
    from .main_scale import find_nearest_cm_digit_region

    main_color = rotated_color[:split_y, :]
    main_gray = region_main.get('image')
    main_binary = region_main.get('binary')
    main_ticks = main_result.get('main_ticks', [])
    main_gap = main_result.get('main_gap', 0)
    zero_x = vernier_result.get('zero_x', 0)

    H_main, W_main = main_color.shape[:2]
    if H_main < 10 or W_main < 10:
        return None

    # ── 1. 获取备选区 ──
    binary_crop, x_off, y_off = find_nearest_cm_digit_region(
        main_ticks, main_gap, zero_x, main_binary)
    if binary_crop is None or binary_crop.size == 0:
        # 退化：只显示主尺 + 零线
        fallback = main_color.copy()
        cv2.line(fallback, (int(zero_x), 0), (int(zero_x), H_main - 1),
                 (0, 255, 255), 2)
        cv2.putText(fallback, "NO BACKUP REGION", (10, H_main - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1)
        return fallback

    ch, cw = binary_crop.shape

    # ── 2. 连通域筛选 ──

    # ── 3. OCR ──

    # ── 4. 策略信息（从 final_result 取）──
    extra = final_result.extra_info if final_result else {}
    main_deriv = extra.get('main_derivation', {}) if hasattr(final_result, 'extra_info') else {}
    strategy = main_deriv.get('strategy', '?') if isinstance(main_deriv, dict) else '?'
    eng = main_deriv.get('ocr_engine', '?') if isinstance(main_deriv, dict) else '?'
    ocr_candidates = main_deriv.get('ocr_candidates', []) if isinstance(main_deriv, dict) else []
    selected_candidates = [c for c in ocr_candidates if c.get('selected')]
    sel_bbox = selected_candidates[0].get('bbox') if selected_candidates else None
    if selected_candidates:
        selected = selected_candidates[0]
        ocr_line = "OCR => '{}' ref_x={}".format(
            selected.get('text'), int(round(float(selected.get('ref_tick_x', 0))))
        )
        ocr_color = (0, 255, 100)
    elif ocr_candidates:
        ocr_line = "OCR candidates found, none selected"
        ocr_color = (0, 160, 255)
    else:
        ocr_line = "OCR => no candidate"
        ocr_color = (100, 100, 255)

    # ═══════════════════════════════════
    #  Panel A: 主尺彩色 + 红框 + 零线
    # ═══════════════════════════════════
    panel_a = main_color.copy()
    # 刻度线（半透明绿）
    for t in main_ticks:
        cv2.line(panel_a, (t['x'], max(0, t.get('y_start', 0))),
                 (t['x'], min(H_main - 1, t.get('y_end', H_main))), (0, 160, 60), 1)
    # 零线（黄色粗线）
    cv2.line(panel_a, (int(zero_x), 0), (int(zero_x), H_main - 1), (0, 255, 255), 2)
    cv2.putText(panel_a, f"ZERO x={int(zero_x)}", (int(zero_x) + 4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    # 备选区红框
    cv2.rectangle(panel_a, (x_off, y_off), (x_off + cw, y_off + ch), (0, 0, 255), 2)
    cv2.putText(panel_a, f"backup ({cw}x{ch})", (x_off + 3, y_off + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # y_top_tick 线（青蓝色虚线），即 max(y_start)
    y_top_tick = max(t.get('y_start', 0) for t in main_ticks) if main_ticks else 0
    for x in range(0, W_main, 10):
        x2 = min(W_main, x + 5)
        cv2.line(panel_a, (x, y_top_tick), (x2, y_top_tick), (255, 200, 50), 1)
    cv2.putText(panel_a, f"y_top_tick={y_top_tick} (max y_start)", (4, y_top_tick - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 50), 1)

    # ═══════════════════════════════════
    #  Panel B: 备选区 blow up 4x
    # ═══════════════════════════════════
    blow = cv2.resize(binary_crop, (cw * 4, ch * 4), interpolation=cv2.INTER_NEAREST)
    panel_b = cv2.cvtColor(blow, cv2.COLOR_GRAY2BGR)

    # ═══════════════════════════════════
    #  Panel C: 备选区彩色底 + 连通域框 + OCR
    # ═══════════════════════════════════
    panel_c = main_color[y_off:y_off + ch, x_off:x_off + cw].copy()
    if len(panel_c.shape) == 2:
        panel_c = cv2.cvtColor(panel_c, cv2.COLOR_GRAY2BGR)

    # 所有连通域（蓝色细框 + 面积标注）
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_crop, connectivity=8)
    for j in range(1, num_labels):
        x = int(stats[j, cv2.CC_STAT_LEFT])
        y = int(stats[j, cv2.CC_STAT_TOP])
        w_cc = int(stats[j, cv2.CC_STAT_WIDTH])
        h_cc = int(stats[j, cv2.CC_STAT_HEIGHT])
        area = int(stats[j, cv2.CC_STAT_AREA])
        cv2.rectangle(panel_c, (x, y), (x + w_cc, y + h_cc), (255, 140, 40), 1)
        # 只标面积 > 10 的
        if area > 10:
            cv2.putText(panel_c, f"{area}", (x, max(y - 1, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, (255, 140, 40), 1)

    # 选中连通域（绿色粗框）
    if sel_bbox is not None:
        bx1 = sel_bbox[0] - x_off
        by1 = sel_bbox[1] - y_off
        bx2 = sel_bbox[2] - x_off
        by2 = sel_bbox[3] - y_off
        cv2.rectangle(panel_c, (bx1, by1), (bx2, by2), (0, 255, 100), 3)

    # OCR 结果文字
    # ═══════════════════════════════════
    #  纵向拼接三面板
    # ═══════════════════════════════════
    gap = 3
    panel_w = max(W_main, cw * 4)

    # Panel A 保持原宽
    pa = panel_a

    # Panel B 可能需要 pad 到 panel_w
    if panel_b.shape[1] < panel_w:
        pb = np.zeros((panel_b.shape[0], panel_w, 3), dtype=np.uint8)
        pb[:] = (20, 20, 25)
        pb[:, :panel_b.shape[1]] = panel_b
    else:
        pb = panel_b

    # Panel C
    if panel_c.shape[1] < panel_w:
        pc = np.zeros((panel_c.shape[0], panel_w, 3), dtype=np.uint8)
        pc[:] = (20, 20, 25)
        pc[:, :panel_c.shape[1]] = panel_c
    else:
        pc = panel_c

    # 面板标题
    label_h = 20
    total_h = (H_main + gap) + (label_h + pb.shape[0] + gap) + (label_h + pc.shape[0] + gap) + 36
    combined = np.zeros((total_h, panel_w, 3), dtype=np.uint8)
    combined[:] = (22, 22, 28)

    y = 0
    # A
    combined[y:y + H_main, :W_main] = pa
    cv2.putText(combined, "A: Main Scale + Backup Region (red) + Zero (yellow)",
                (4, y + H_main - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1)
    y += H_main + gap

    # B label
    cv2.putText(combined, f"B: Backup Region ({cw}x{ch}) x4",
                (4, y + label_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    y += label_h
    combined[y:y + pb.shape[0], :] = pb
    y += pb.shape[0] + gap

    # C label
    cv2.putText(combined, f"C: CC Analysis ({num_labels - 1} CCs) | {ocr_line}",
                (4, y + label_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, ocr_color, 1)
    y += label_h
    combined[y:y + pc.shape[0], :] = pc
    y += pc.shape[0] + gap

    # ── 图例 ──
    legend_items = [
        ("red rect = backup region", (0, 0, 255), 'rect'),
        ("yellow line = zero (vernier x=0)", (0, 255, 255), 'line'),
        ("orange box = all CCs", (255, 140, 40), 'rect'),
        ("green thick = selected CC", (0, 255, 100), 'rect'),
    ]
    combined = draw_legend_below(combined, legend_items)

    # 顶部大标题
    cv2.putText(combined, f"STEP 3b: Main Scale OCR [{eng}]  strategy={strategy}",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return combined


def _make_zero_overview(rotated_color: np.ndarray,
                         main_result: dict,
                         vernier_result: dict,
                         split_y: int,
                         region_main: dict,
                         region_vernier: dict) -> np.ndarray:
    """生成零线验证概览：整张 ROI + 零线 + 主尺/游标刻度叠加 + 图例
       下方追加：游标尺垂直投影 + 数字0验证状态"""
    H, W = rotated_color.shape[:2]
    overview = rotated_color.copy()

    # ── 区域分界线 ──
    cv2.line(overview, (0, split_y), (W, split_y), (255, 255, 255), 1)
    cv2.putText(overview, "MAIN", (4, split_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 100), 1)
    cv2.putText(overview, "VERNIER", (4, split_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1)

    # ── 主尺刻度线（绿色，长刻度加粗）──
    for t in main_result.get('main_ticks', []):
        y_off = region_main.get('y_offset', 0)
        is_long = t.get('is_long', False)
        color = (0, 255, 80) if is_long else (0, 180, 60)
        thickness = 4 if is_long else 3
        cv2.line(overview, (t['x'], t['y_start'] + y_off),
                 (t['x'], min(t['y_end'] + y_off, split_y)), color, thickness)

    # ── 游标刻度线（橙色，长刻度加粗）──
    vy_off = region_vernier.get('y_offset', split_y)
    for t in vernier_result.get('vernier_ticks', []):
        is_long = t.get('is_long', False)
        color = (255, 200, 50) if is_long else (200, 150, 40)
        thickness = 3 if is_long else 2
        cv2.line(overview, (t['x'], t['y_start'] + vy_off),
                 (t['x'], t['y_end'] + vy_off), color, thickness)

    # ── 零线（亮蓝粗线，贯穿全图，标注 x 坐标）──
    zero_x = int(vernier_result.get('zero_x', 0))
    cv2.line(overview, (zero_x, 0), (zero_x, H - 1), (255, 60, 60), 4)
    cv2.putText(overview, f"ZERO LINE (x={zero_x})", (zero_x + 6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 60, 60), 2)
    # 底部也标
    cv2.putText(overview, f"x={zero_x}", (zero_x + 6, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 60, 60), 2)

    # ── 图例（拼到图下方）──
    legend_items = [
        ("thick red = zero line (vernier 0)", (255, 60, 60), 'line'),
        ("green line = main scale tick", (0, 180, 60), 'line'),
        ("orange line = vernier tick", (200, 150, 40), 'line'),
        ("white dash = main/vernier split", (255, 255, 255), 'line'),
    ]

    # ── 零线数字0验证状态条 ──
    zero_digit_found = vernier_result.get('zero_digit_found', False)
    vproj_norm = vernier_result.get('vproj_norm')
    vernier_peaks = vernier_result.get('vernier_peaks')

    # 构造验证状态横幅
    bar_h = 28
    status_bar = np.zeros((bar_h, W, 3), dtype=np.uint8)
    if zero_digit_found:
        status_bar[:] = (40, 70, 40)
        status_text = "[OK] Zero Digit '0' Verified by OCR"
        status_color = (0, 255, 100)
    else:
        status_bar[:] = (60, 50, 30)
        status_text = "[WARN] Zero Digit '0' NOT Found — using position fallback"
        status_color = (100, 180, 255)
    cv2.putText(status_bar, status_text, (8, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

    # ── 游标尺垂直投影图 ──
    proj_panel = None
    if vproj_norm is not None and vernier_peaks is not None and len(vernier_peaks) > 0:
        proj_panel = draw_projection_plot(
            vproj_norm, vernier_peaks, width=W,
            title="Vernier Vertical Projection (for zero-line detection)")
        # 在投影图上标注零线位置
        if zero_x > 0 and proj_panel is not None:
            zx_proj = int(zero_x)
            ph = proj_panel.shape[0]
            if 0 <= zx_proj < W:
                cv2.line(proj_panel, (zx_proj, 0), (zx_proj, ph - 1),
                         (50, 150, 255), 1, cv2.LINE_AA)
                cv2.putText(proj_panel, "ZERO", (zx_proj + 3, ph - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 150, 255), 1)

    # ── 竖向拼接：概览图 + 状态条 + 投影图 ──
    gap = 2
    panels = [overview]
    # 先在图例前插入状态条
    _panels_height = H
    panels.append(status_bar)
    _panels_height += gap + bar_h
    if proj_panel is not None:
        panels.append(proj_panel)
        _panels_height += gap + proj_panel.shape[0]

    total_h = _panels_height
    combined = np.zeros((total_h, W, 3), dtype=np.uint8)
    combined[:] = (30, 30, 35)
    y_cursor = 0
    combined[y_cursor:y_cursor + H, :] = overview; y_cursor += H + gap
    combined[y_cursor:y_cursor + bar_h, :] = status_bar; y_cursor += bar_h + gap
    if proj_panel is not None:
        ph = proj_panel.shape[0]
        combined[y_cursor:y_cursor + ph, :] = proj_panel; y_cursor += ph + gap

    # ── 图例（拼到最底部）──
    combined = draw_legend_below(combined, legend_items)

    cv2.putText(combined, "Step 3: Main + Zero Line (full ROI)", (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return combined


def read_caliper(image_path: str) -> CaliperResult:
    """从文件路径读取（支持中文路径）"""
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    pipeline = CaliperPipeline()
    return pipeline.run(img)


def read_caliper_from_array(img: np.ndarray) -> CaliperResult:
    """从 numpy 数组读取（兼容旧接口）"""
    pipeline = CaliperPipeline()
    return pipeline.run(img)
