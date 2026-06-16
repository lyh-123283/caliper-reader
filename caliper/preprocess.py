"""
步骤 0 — 图像预处理（增强 + 滤波 + 二值化 + 后处理）

流程:
  1. 灰度化
  2. 幂律变换（gamma 校正）— 全局亮度/对比度调节
  3. 双边滤波（保边去噪）
  4. 中值滤波（去除脉冲/椒盐噪声，与双边互补）
  5. NLM 非局部均值去噪（可选，强保边去噪）
  6. CLAHE 对比度增强（局部自适应）
  7. 非锐化掩膜（unsharp mask）锐化
  8. 自适应阈值二值化
  9. 形态学开运算（可选，清除孤立噪点）
 10. 连通域过滤（可选，剔除小面积噪声斑块）
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
            'enhanced':          增强后的灰度图 (gamma → bilateral → median → CLAHE → unsharp)
            'binary_adaptive':   自适应阈值二值图 (THRESH_BINARY: 黑=刻度前景, 白=背景)
            'debug_vis':         可视化对比图（3合1，标注每幅图的处理链）
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

    # ── 1. 灰度化 ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result['gray'] = gray

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

    # ── 2b. 高斯模糊前置（轻量去纹理噪声，双边之前）──
    gk = config.preprocess.gauss_pre_ksize
    if gk >= 3:
        gk = gk if gk % 2 == 1 else gk + 1
        gray = cv2.GaussianBlur(gray, (gk, gk), config.preprocess.gauss_pre_sigma)

    # ── 3. 双边滤波（保边去噪）──
    denoised = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigma, bilateral_sigma)

    # ── 4. 中值滤波（去除脉冲/椒盐噪声，与双边互补）──
    if median_ksize >= 3:
        # 确保核尺寸为奇数
        ksize = median_ksize if median_ksize % 2 == 1 else median_ksize + 1
        denoised = cv2.medianBlur(denoised, ksize)

    # ── 5. CLAHE 对比度增强（局部自适应直方图均衡）──
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(config.preprocess.clahe_tile_w, config.preprocess.clahe_tile_h))
    enhanced = clahe.apply(denoised)

    # ── 6. 非锐化掩膜锐化（unsharp mask）──
    #     标准公式：sharp = orig + amount × (orig - blur)
    #     amount=0 → 不变；amount=1 → 标准锐化；amount=1.5 → 强锐化
    #     旧代码错用 addWeighted(orig, a, blur, 1-a)，amount<1 时反而模糊
    if config.preprocess.unsharp_amount > 0.01:
        blur = cv2.GaussianBlur(enhanced, (0, 0), config.preprocess.unsharp_blur_sigma)
        # cv2.addWeighted(orig, 1+a, blur, -a, 0) 实现 orig + a*(orig - blur)
        a = config.preprocess.unsharp_amount
        enhanced = cv2.addWeighted(enhanced, 1.0 + a, blur, -a, 0)
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
    result['binary_adaptive'] = binary_adaptive

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

    # ── 10. 后处理：双向连通域过滤（剔除小面积噪声斑块）──
    if config.preprocess.cc_filter_enabled:
        min_area = config.preprocess.cc_min_area

        # v6.5: 既过滤"黑前景里夹的白岛"，也过滤"白背景上的孤立黑点"
        #       前者：抹黑（变成前景）；后者：抹白（变成背景）

        # (a) 过滤白色小斑块（黑刻线里夹的小白点）→ 抹黑
        num_w, labels_w, stats_w, _ = cv2.connectedComponentsWithStats(
            binary_adaptive, connectivity=8)
        white_mask = np.zeros_like(binary_adaptive)
        for i in range(1, num_w):
            if stats_w[i, cv2.CC_STAT_AREA] < min_area:
                white_mask[labels_w == i] = 255
        if white_mask.any():
            binary_adaptive = cv2.subtract(binary_adaptive, white_mask)

        # (b) 过滤黑色小斑块（白背景上的椒盐黑点）→ 抹白
        inverted = cv2.bitwise_not(binary_adaptive)
        num_b, labels_b, stats_b, _ = cv2.connectedComponentsWithStats(
            inverted, connectivity=8)
        black_mask = np.zeros_like(binary_adaptive)
        for i in range(1, num_b):
            if stats_b[i, cv2.CC_STAT_AREA] < min_area:
                black_mask[labels_b == i] = 255
        if black_mask.any():
            binary_adaptive = cv2.add(binary_adaptive, black_mask)

    # ── 后处理结束后，同步更新 result ──
    result['binary_adaptive'] = binary_adaptive

    # ── 生成可视化（3合1：原图 / 增强 / 自适应阈值）──
    result['debug_vis'] = _make_preprocess_vis(img, enhanced, binary_adaptive, gamma, median_ksize)

    return result


def _make_preprocess_vis(original: np.ndarray,
                          enhanced: np.ndarray,
                          bin_adaptive: np.ndarray,
                          gamma: float = 1.0,
                          median_ksize: int = 3) -> np.ndarray:
    """生成预处理 3 合 1 可视化，每幅图下方标注所经历的处理步骤"""
    h, w = original.shape[:2]

    scale = min(500 / max(h, w), 1.0)
    nw, nh = int(w * scale), int(h * scale)

    def _resize(img):
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    # 三张图横向排列
    orig_r = _resize(original)
    enh_r = cv2.cvtColor(_resize(enhanced), cv2.COLOR_GRAY2BGR)
    ba_r = cv2.cvtColor(_resize(bin_adaptive), cv2.COLOR_GRAY2BGR)

    # ── 构建各面板的处理标签 ──
    # 第 1 幅：原图（未处理）
    label1 = "1. Original (BGR)"

    # 第 2 幅：增强图 — 列出所有启用步骤
    enhance_steps = []
    if gamma != 1.0:
        enhance_steps.append("γ=%.2f" % gamma)
    enhance_steps.append("Bilateral")
    if median_ksize >= 3:
        enhance_steps.append("Median")
    enhance_steps.append("CLAHE")
    enhance_steps.append("UnsharpMask")
    label2 = "2. " + " → ".join(enhance_steps)

    # 第 3 幅：自适应阈值二值图
    post_parts = []
    if config.preprocess.morph_open_enabled:
        post_parts.append("MorphOpen(k=%d)" % config.preprocess.morph_open_kernel_size)
    if config.preprocess.cc_filter_enabled:
        post_parts.append("CCFilter(min=%d)" % config.preprocess.cc_min_area)
    post_suffix = " → " + " → ".join(post_parts) if post_parts else ""
    label3 = "3. AdaptiveBinary(blk=%d,C=%d)%s → 黑=刻度/白=背景" % (
        config.preprocess.adaptive_block_size,
        config.preprocess.adaptive_C,
        post_suffix)

    labels = [label1, label2, label3]
    imgs = [orig_r, enh_r, ba_r]

    gap = 4
    out_h = nh + 10
    out_w = nw * 3 + gap * 2
    vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    vis[:] = (30, 30, 35)

    for i, (img, label) in enumerate(zip(imgs, labels)):
        x0 = i * (nw + gap)
        ch, cw = img.shape[:2]
        vis[:ch, x0:x0 + cw] = img
        cv2.putText(vis, label, (x0 + 8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(vis, "STEP 0: Preprocessing", (5, out_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 125), 1)

    return vis
