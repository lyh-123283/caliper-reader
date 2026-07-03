"""
步骤 0 — 图像预处理（增强 + 滤波 + 二值化 + 后处理）

流程:
  1. 灰度化
  2. 幂律变换（gamma 校正）— 全局亮度/对比度调节
  3. 高斯模糊前置（轻量去纹理）
  4. 双边滤波（保边去噪）
  5. 中值滤波（去除脉冲/椒盐噪声，与双边互补）
  6. CLAHE 对比度增强（局部自适应）
  7. 非锐化掩膜（unsharp mask）锐化
  8. 自适应阈值二值化
  9. 形态学开运算（可选，清除孤立噪点）
 10. 轮廓连通块过滤（可选，剔除小面积噪声斑块）
"""

import cv2
import numpy as np
from .config import config


def preprocess(img: np.ndarray,
               clip_limit: float = None,
               bilateral_d: int = None,
               bilateral_sigma: float = None,
               gamma: float = None,
               median_ksize: int = None) -> dict:
    """
    图像预处理主函数

    Args:
        img:             BGR 彩色图像
        clip_limit:      CLAHE 对比度限制（None=使用 config.preprocess.clahe_clip_limit）
        bilateral_d:     双边滤波直径（None=使用 config）
        bilateral_sigma: 双边滤波 sigma（None=使用 config）
        gamma:           幂律变换 gamma 值（None=使用 config → 1.0=不变；<1 提亮暗部；>1 压暗高光）
        median_ksize:    中值滤波核尺寸（None=使用 config；奇数≥3；设为 0 跳过中值滤波）

    Returns:
        dict with keys:
            'color':             原始彩色图 (BGR)
            'gray':              原始灰度图
            'enhanced':          增强后的灰度图 (gamma → gaussian → bilateral → median → CLAHE → unsharp)
            'binary_adaptive':   自适应阈值二值图 (THRESH_BINARY: 黑=刻度前景, 白=背景)
            'debug_vis':         可视化对比图（全步骤网格）
            'intermediates':     dict 中间步骤图像，供外部可视化使用
    """
    # ── 参数默认值来自 config ──
    if clip_limit is None:
        clip_limit = config.preprocess.clahe_clip_limit
    if bilateral_d is None:
        bilateral_d = config.preprocess.bilateral_d
    if bilateral_sigma is None:
        bilateral_sigma = config.preprocess.bilateral_sigma
    if gamma is None:
        gamma = config.preprocess.gamma
    if median_ksize is None:
        median_ksize = config.preprocess.median_ksize
    result = {'color': img.copy()}
    intermediates = {}  # 收集中间步骤图像，供可视化使用

    # ── 1. 灰度化 ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result['gray'] = gray
    intermediates['01_gray'] = gray

    # ── 2. 幂律变换（gamma 校正）──
    #     物理意义：s = c * r^(1/gamma)
    #     gamma < 1 → 扩展暗部、提亮图像 → 刻线区域更可见
    #     gamma > 1 → 压缩暗部、压暗图像 → 抑制过曝反光
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype(np.uint8)
        gray = cv2.LUT(gray, table)
    intermediates['02_gamma'] = gray.copy()

    # ── 2b. 高斯模糊前置（轻量去纹理噪声，双边之前）──
    gk = config.preprocess.gauss_pre_ksize
    if gk >= 3:
        gk = gk if gk % 2 == 1 else gk + 1
        gray = cv2.GaussianBlur(gray, (gk, gk), config.preprocess.gauss_pre_sigma)
    intermediates['03_gauss'] = gray.copy()

    # ── 3. 双边滤波（保边去噪）──
    denoised = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigma, bilateral_sigma)
    intermediates['04_bilateral'] = denoised.copy()

    # ── 4. 中值滤波（去除脉冲/椒盐噪声，与双边互补）──
    if median_ksize >= 3:
        # 确保核尺寸为奇数
        ksize = median_ksize if median_ksize % 2 == 1 else median_ksize + 1
        denoised = cv2.medianBlur(denoised, ksize)
    intermediates['05_median'] = denoised.copy()

    # ── 5. CLAHE 对比度增强（局部自适应直方图均衡）──
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(config.preprocess.clahe_tile_w, config.preprocess.clahe_tile_h))
    enhanced = clahe.apply(denoised)
    intermediates['06_clahe'] = enhanced.copy()

    # ── 6. 非锐化掩膜锐化（unsharp mask）──
    #     标准公式：sharp = orig + amount × (orig - blur)
    #     amount=0 → 不变；amount=1 → 标准锐化；amount=1.5 → 强锐化
    if config.preprocess.unsharp_amount > 0.01:
        blur = cv2.GaussianBlur(enhanced, (0, 0), config.preprocess.unsharp_blur_sigma)
        # cv2.addWeighted(orig, 1+a, blur, -a, 0) 实现 orig + a*(orig - blur)
        a = config.preprocess.unsharp_amount
        enhanced = cv2.addWeighted(enhanced, 1.0 + a, blur, -a, 0)
    intermediates['07_unsharp'] = enhanced.copy()
    result['enhanced'] = enhanced

    # ── 7. 自适应阈值二值化 ──
    #     输出 THRESH_BINARY：黑字/黑线 = 前景（刻度），白底 = 背景
    #     便于人眼直观看懂；下游 ROI 投影步骤内部会反色处理。
    binary_adaptive = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,           # ← 黑前景/白背景（人眼友好）
        blockSize=config.preprocess.adaptive_block_size,
        C=config.preprocess.adaptive_C
    )
    intermediates['08_adaptive_bin'] = binary_adaptive.copy()

    # ── 9. 后处理：形态学开运算（二值化后清除孤立噪点）──
    if config.preprocess.morph_open_enabled:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.preprocess.morph_open_kernel_size, config.preprocess.morph_open_kernel_size),
        )
        binary_adaptive = cv2.morphologyEx(
            binary_adaptive, cv2.MORPH_OPEN, kernel,
            iterations=config.preprocess.morph_open_iterations,
        )
    intermediates['09_morph_open'] = binary_adaptive.copy()

    # ── 10. 后处理：轮廓连通块过滤（剔除小面积噪声斑块）──
    if config.preprocess.cc_filter_enabled:
        min_area = config.preprocess.cc_min_area

        # 既过滤黑前景里夹的白岛，也过滤白背景上的孤立黑点。
        # 这里用 findContours + 局部像素计数替代 connectedComponents 的全图标签扫描。
        binary_adaptive = _filter_small_components_by_contour(
            binary_adaptive, min_area)

    # ── 后处理结束后，同步更新 result ──
    intermediates['10_cc_filter'] = binary_adaptive.copy()
    result['binary_adaptive'] = binary_adaptive

    # ── 生成可视化（全步骤网格）──
    result['intermediates'] = intermediates
    result['debug_vis'] = _make_preprocess_vis(img, intermediates, gamma, median_ksize)

    return result


def _filter_small_components_by_contour(binary: np.ndarray,
                                        min_area: int) -> np.ndarray:
    """Remove small white and black components without repeated full-image scans."""
    out = binary.copy()

    # Small white islands are turned black.
    _fill_small_contours(out, out.copy(), min_area, 0)

    # Small black specks are white in the inverted mask and are turned white.
    inverted = cv2.bitwise_not(out)
    _fill_small_contours(out, inverted, min_area, 255)

    return out


def _fill_small_contours(target: np.ndarray,
                         source_mask: np.ndarray,
                         min_area: int,
                         fill_value: int) -> None:
    contours, _ = cv2.findContours(
        source_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        if w * h < min_area:
            pixel_area = int(cv2.countNonZero(source_mask[y:y + h, x:x + w]))
        elif cv2.contourArea(contour) >= min_area:
            continue
        else:
            local_mask = np.zeros((h, w), dtype=np.uint8)
            shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
            cv2.drawContours(local_mask, [shifted], -1, 255, -1)
            pixel_area = int(cv2.countNonZero(
                cv2.bitwise_and(source_mask[y:y + h, x:x + w], local_mask)))

        if pixel_area < min_area:
            cv2.drawContours(target, [contour], -1, fill_value, -1)


def _make_preprocess_vis(original: np.ndarray,
                          intermediates: dict,
                          gamma: float = 1.0,
                          median_ksize: int = 3) -> np.ndarray:
    """生成预处理关键步骤可视化，保留少量大图方便肉眼检查。"""
    h, w = original.shape[:2]

    cell_w = 520
    scale = cell_w / w
    cell_h = max(int(h * scale), 40)

    def _resize_gray(img):
        """安全缩放灰度图并转 BGR"""
        if img is None:
            return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        if len(img.shape) == 3 and img.shape[2] == 3:
            r = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            return r
        r = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

    bg_color = (30, 30, 35)
    gap = 6
    label_h = 24
    cols = 2
    steps = [
        ("Original", original, (210, 210, 210)),
        ("Enhanced", intermediates.get('07_unsharp', intermediates.get('06_clahe')), (120, 200, 255)),
        ("Adaptive Binary", intermediates.get('08_adaptive_bin'), (100, 255, 160)),
        ("Final Filtered", intermediates.get('10_cc_filter'), (255, 180, 120)),
    ]
    rows = 2
    out_w = cols * cell_w + gap * (cols - 1) + 16
    out_h = rows * (cell_h + label_h + gap) + gap + 30
    vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    vis[:] = bg_color

    for idx, (name, img_i, color) in enumerate(steps):
        r = idx // cols
        c = idx % cols
        x0 = 8 + c * (cell_w + gap)
        y0 = gap + r * (cell_h + label_h + gap)
        img_bgr = _resize_gray(img_i)

        vis[y0:y0 + cell_h, x0:x0 + cell_w] = img_bgr

        bar_y = y0 + cell_h
        bar = np.zeros((label_h, cell_w, 3), dtype=np.uint8)
        bar[:] = (35, 35, 40)
        cv2.putText(bar, name, (6, label_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        vis[bar_y:bar_y + label_h, x0:x0 + cell_w] = bar

    cv2.putText(vis, "STEP 0: Preprocess key views - black=ticks/text, white=background",
                (8, out_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 145), 1)

    return vis
