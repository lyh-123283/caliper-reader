"""
步骤 0 — 图像预处理
"""

import os
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from .config import config


def preprocess(img: np.ndarray,
               clip_limit: float = None,
               bilateral_d: int = None,
               bilateral_sigma: float = None,
               gamma: float = None,
               median_ksize: int = None,
               make_debug: bool = True) -> dict:
    """
    图像预处理主函数

    Args:
        img:             BGR 彩色图像
        clip_limit:      CLAHE 对比度限制（None=使用 config.preprocess.clahe_clip_limit）
        bilateral_d:     双边滤波直径（None=使用 config）
        bilateral_sigma: 双边滤波 sigma（None=使用 config）
        gamma:           幂律变换 gamma 值（None=使用 config → 1.0=不变；<1 提亮暗部；>1 压暗高光）
        median_ksize:    中值滤波核尺寸（None=使用 config；奇数>=3；设为 0 跳过）

    Returns:
        dict with keys:
            'color':             原始彩色图 (BGR)
            'gray':              原始灰度图
            'enhanced':          增强后的灰度图
            'binary_adaptive':   自适应阈值二值图 (THRESH_BINARY: 黑=刻度前景, 白=背景)
            'debug_vis':         可视化对比图（全步骤网格）
            'intermediates':     dict 中间步骤图像，供外部可视化使用
    """
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
    intermediates = {}
    timings = {}

    def mark(key: str, start_time: float):
        timings[key] = (time.perf_counter() - start_time) * 1000.0

    t0 = time.perf_counter()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mark('gray', t0)
    result['gray'] = gray
    if make_debug:
        intermediates['01_gray'] = gray

    t0 = time.perf_counter()
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype(np.uint8)
        gray = cv2.LUT(gray, table)
    mark('gamma', t0)
    if make_debug:
        intermediates['02_gamma'] = gray.copy()

    t0 = time.perf_counter()
    denoised = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigma, bilateral_sigma)
    mark('bilateral', t0)
    if make_debug:
        intermediates['03_bilateral'] = denoised.copy()

    t0 = time.perf_counter()
    if median_ksize >= 3:
        ksize = median_ksize if median_ksize % 2 == 1 else median_ksize + 1
        denoised = cv2.medianBlur(denoised, ksize)
        if make_debug:
            intermediates['04_median'] = denoised.copy()
    mark('median', t0)

    t0 = time.perf_counter()
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(config.preprocess.clahe_tile_w, config.preprocess.clahe_tile_h))
    enhanced = clahe.apply(denoised)
    mark('clahe', t0)
    if make_debug:
        intermediates['05_clahe'] = enhanced.copy()

    t0 = time.perf_counter()
    if config.preprocess.unsharp_amount > 0.01:
        blur = cv2.GaussianBlur(enhanced, (0, 0), config.preprocess.unsharp_blur_sigma)
        a = config.preprocess.unsharp_amount
        enhanced = cv2.addWeighted(enhanced, 1.0 + a, blur, -a, 0)
    mark('unsharp', t0)
    if make_debug:
        intermediates['06_unsharp'] = enhanced.copy()
    result['enhanced'] = enhanced

    t0 = time.perf_counter()
    binary_scale = float(getattr(config.preprocess, 'adaptive_binary_scale', 1.0) or 1.0)
    if 0 < binary_scale < 0.999:
        h, w = enhanced.shape[:2]
        small_w = max(1, int(round(w * binary_scale)))
        small_h = max(1, int(round(h * binary_scale)))
        enhanced_for_binary = cv2.resize(
            enhanced, (small_w, small_h), interpolation=cv2.INTER_AREA)
        block_size = max(3, int(round(config.preprocess.adaptive_block_size * binary_scale)))
        if block_size % 2 == 0:
            block_size += 1
        max_block = min(small_h, small_w)
        if max_block % 2 == 0:
            max_block -= 1
        block_size = min(block_size, max(3, max_block))
        binary_small = cv2.adaptiveThreshold(
            enhanced_for_binary, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block_size,
            C=config.preprocess.adaptive_C
        )
        binary_adaptive = cv2.resize(
            binary_small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        binary_adaptive = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=config.preprocess.adaptive_block_size,
            C=config.preprocess.adaptive_C
        )
    mark('adaptive_threshold', t0)
    if make_debug:
        intermediates['07_adaptive_bin'] = binary_adaptive.copy()

    t0 = time.perf_counter()
    if config.preprocess.morph_open_enabled:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.preprocess.morph_open_kernel_size, config.preprocess.morph_open_kernel_size),
        )
        binary_adaptive = cv2.morphologyEx(
            binary_adaptive, cv2.MORPH_OPEN, kernel,
            iterations=config.preprocess.morph_open_iterations,
        )
    mark('morph_open', t0)
    if make_debug:
        intermediates['08_morph_open'] = binary_adaptive.copy()

    t0 = time.perf_counter()
    if config.preprocess.cc_filter_enabled:
        min_area = config.preprocess.cc_min_area
        binary_adaptive = _filter_small_components_by_contour(
            binary_adaptive, min_area)
    mark('cc_filter', t0)

    if make_debug:
        intermediates['09_cc_filter'] = binary_adaptive.copy()
    result['binary_adaptive'] = binary_adaptive

    result['intermediates'] = intermediates
    result['step_timings'] = timings
    result['debug_vis'] = _make_preprocess_vis(img, intermediates, gamma, median_ksize) if make_debug else None

    return result


def _filter_small_components_by_contour(binary: np.ndarray,
                                        min_area: int) -> np.ndarray:
    out = binary.copy()
    _fill_small_contours(out, out.copy(), min_area, 0)
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
    h, w = original.shape[:2]

    cell_w = 520
    scale = cell_w / w
    cell_h = max(int(h * scale), 40)

    def _resize_gray(img):
        if img is None:
            return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        if len(img.shape) == 3 and img.shape[2] == 3:
            r = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            return r
        r = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

    bg_color = (30, 30, 35)
    gap = 6
    label_h = 28
    cols = 2
    steps = [
        ("原图", original, (210, 210, 210)),
        ("增强图", intermediates.get('06_unsharp', intermediates.get('05_clahe')), (120, 200, 255)),
        ("自适应二值图", intermediates.get('07_adaptive_bin'), (100, 255, 160)),
        ("最终过滤图", intermediates.get('09_cc_filter'), (255, 180, 120)),
    ]
    rows = 2
    out_w = cols * cell_w + gap * (cols - 1) + 16
    out_h = rows * (cell_h + label_h + gap) + gap + 34
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
        _draw_chinese_text(bar, name, (8, 5), font_size=16, color=color)
        vis[bar_y:bar_y + label_h, x0:x0 + cell_w] = bar

    _draw_chinese_text(
        vis,
        "步骤0：图像预处理关键视图（黑色=刻线/文字，白色=背景）",
        (8, out_h - 26),
        font_size=16,
        color=(140, 140, 145),
    )

    return vis


def _draw_chinese_text(img: np.ndarray,
                       text: str,
                       xy: tuple,
                       font_size: int = 16,
                       color=(220, 220, 220)) -> None:
    font = _get_chinese_font(font_size)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    b, g, r = color
    draw.text(xy, text, font=font, fill=(r, g, b))
    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _get_chinese_font(size: int):
    font_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    candidates = [
        os.path.join(font_dir, "msyh.ttc"),
        os.path.join(font_dir, "simhei.ttf"),
        os.path.join(font_dir, "simsun.ttc"),
        os.path.join(font_dir, "msyh.ttf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()
