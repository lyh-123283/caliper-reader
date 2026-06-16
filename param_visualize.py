"""
参数可视化脚本：展示不同预处理参数对图像的影响
生成按参数分组的综合对比图
"""
import os
import cv2
import numpy as np
from pathlib import Path
from caliper import config
from caliper.preprocess import preprocess

# ── 读取测试图像 ──
img_path = Path(__file__).parent / "test.jpg"
data = img_path.read_bytes()
img_arr = np.frombuffer(data, dtype=np.uint8)
img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

out_dir = Path(__file__).parent / "param_visualizations"
out_dir.mkdir(parents=True, exist_ok=True)


def make_label(text, width=500, height=40):
    """创建标签图像"""
    lbl = np.ones((height, width, 3), dtype=np.uint8) * 40
    cv2.putText(lbl, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (200, 200, 255), 2, cv2.LINE_AA)
    return lbl


def make_comparison_grid(results_dict, param_name, out_name):
    """
    将一组(参数值, 增强图, 二值图) 排列成网格并保存
    results_dict: {param_value: (enhanced_img, binary_img)}
    """
    n = len(results_dict)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    # 统一缩放到相同尺寸
    h, w = img.shape[:2]
    scale = min(300 / max(h, w), 1.0)
    tw, th = int(w * scale), int(h * scale)

    cell_h = th + 35
    cell_w = tw

    # 创建大画布
    canvas_h = rows * cell_h + 40
    canvas_w = cols * cell_w + 40
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 50

    sorted_keys = sorted(results_dict.keys())
    for idx, (key, (enhanced, binary)) in enumerate(sorted(results_dict.items())):
        row = idx // cols
        col = idx % cols

        x0 = 10 + col * cell_w
        y0 = 10 + row * cell_h

        # 缩放增强图
        enh_r = cv2.resize(enhanced, (tw, th))
        if len(enh_r.shape) == 2 or enh_r.shape[2] == 1:
            enh_r = cv2.cvtColor(enh_r, cv2.COLOR_GRAY2BGR)

        # 参数标签
        cv2.putText(canvas,
                    f"{param_name}={key}",
                    (x0 + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)

        # 放置图像
        canvas[y0 + 30:y0 + 30 + th, x0:x0 + tw] = enh_r

    cv2.putText(canvas, f"Effect of {param_name} (Gray=Enhanced)", (10, canvas_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / out_name), canvas)
    print(f"  Saved: {out_name}")


def make_full_grid(param_list, out_name):
    """
    将多种参数组合的 debug_vis 排列成 3×N 网格
    param_list: [(label, debug_vis), ...]
    """
    n = len(param_list)
    if n == 0:
        return

    # 获取第一张 debug_vis 尺寸来确定单元格尺寸
    sample = param_list[0][1]
    cell_h, cell_w = sample.shape[:2]

    # 每行放 2 个
    cols = 2
    rows = (n + cols - 1) // cols

    margin = 20
    canvas_h = rows * (cell_h + 30) + margin
    canvas_w = cols * (cell_w + 10) + margin
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 50

    for idx, (label_text, vis) in enumerate(param_list):
        row = idx // cols
        col = idx % cols

        x0 = margin // 2 + col * (cell_w + 10)
        y0 = margin // 2 + row * (cell_h + 30)

        h, w = vis.shape[:2]
        canvas[y0:y0 + h, x0:x0 + w] = vis

        # 标签
        cv2.putText(canvas,
                    label_text,
                    (x0 + 10, y0 + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / out_name), canvas)
    print(f"  Saved: {out_name}")


# ═══════════════════════════════════════════════════════════════
#  默认参数
# ═══════════════════════════════════════════════════════════════

D = {
    'gamma': 0.8,
    'bilateral_d': 5,
    'bilateral_sigma': 25.0,
    'median_ksize': 0,
    'clahe_clip': 4.0,
    'unsharp_amount': 1.5,
    'clahe_tile_w': 8,
    'clahe_tile_h': 8,
    'adaptive_block_size': 31,
    'adaptive_C': 5,
    'unsharp_blur_sigma': 1.5,
}

print("=" * 60)
print("Generating EXTENDED parameter comparison visualizations...")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════
#  第一部分：单参数独立对比（扩大取值范围）
# ═══════════════════════════════════════════════════════════════

# ── 01. Gamma 校正（10档）──
print("\n[01] Gamma: 0.4 / 0.5 / 0.6 / 0.7 / 0.8 / 0.9 / 1.0 / 1.2 / 1.5 / 2.0")
r = {}
for g in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]:
    o = preprocess(img, clip_limit=D['clahe_clip'], bilateral_d=D['bilateral_d'],
                   bilateral_sigma=D['bilateral_sigma'], gamma=g, median_ksize=D['median_ksize'])
    r[g] = (o['enhanced'], o['binary_adaptive'])
make_comparison_grid(r, 'gamma', '01_gamma_comparison.png')

# ── 02. 双边滤波直径（6档）──
print("\n[02] Bilateral d: 5 / 7 / 9 / 11 / 15 / 21")
r = {}
for d in [5, 7, 9, 11, 15, 21]:
    o = preprocess(img, clip_limit=D['clahe_clip'], bilateral_d=d,
                   bilateral_sigma=D['bilateral_sigma'], gamma=D['gamma'], median_ksize=D['median_ksize'])
    r[d] = (o['enhanced'], o['binary_adaptive'])
make_comparison_grid(r, 'bilateral_d', '02_bilateral_d_comparison.png')

# ── 03. 双边滤波 Sigma（5档）──
print("\n[03] Bilateral sigma: 10 / 25 / 50 / 75 / 100")
r = {}
for s in [10.0, 25.0, 50.0, 75.0, 100.0]:
    o = preprocess(img, clip_limit=D['clahe_clip'], bilateral_d=D['bilateral_d'],
                   bilateral_sigma=s, gamma=D['gamma'], median_ksize=D['median_ksize'])
    r[s] = (o['enhanced'], o['binary_adaptive'])
make_comparison_grid(r, 'bilateral_sigma', '03_bilateral_sigma_comparison.png')

# ── 04. 中值滤波核尺寸（5档）──
print("\n[04] Median ksize: 0 / 3 / 5 / 7 / 9")
r = {}
for k in [0, 3, 5, 7, 9]:
    o = preprocess(img, clip_limit=D['clahe_clip'], bilateral_d=D['bilateral_d'],
                   bilateral_sigma=D['bilateral_sigma'], gamma=D['gamma'], median_ksize=k)
    r[k] = (o['enhanced'], o['binary_adaptive'])
make_comparison_grid(r, 'median_ksize', '04_median_ksize_comparison.png')

# ── 05. CLAHE clip limit（6档）──
print("\n[05] CLAHE clip: 1.0 / 2.0 / 3.0 / 4.0 / 6.0 / 8.0")
r = {}
for c in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
    o = preprocess(img, clip_limit=c, bilateral_d=D['bilateral_d'],
                   bilateral_sigma=D['bilateral_sigma'], gamma=D['gamma'], median_ksize=D['median_ksize'])
    r[c] = (o['enhanced'], o['binary_adaptive'])
make_comparison_grid(r, 'clahe_clip', '05_clahe_clip_comparison.png')

# ── 06. Unsharp amount（6档）──
print("\n[06] Unsharp amount: 0 / 0.5 / 1.0 / 1.5 / 2.0 / 3.0")
r = {}
for a in [0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    o = preprocess(img, clip_limit=D['clahe_clip'], bilateral_d=D['bilateral_d'],
                   bilateral_sigma=D['bilateral_sigma'], gamma=D['gamma'], median_ksize=D['median_ksize'])
    r[a] = (o['enhanced'], o['binary_adaptive'])
make_comparison_grid(r, 'unsharp_amount', '06_unsharp_amount_comparison.png')

# ── 07. CLAHE tile grid 对比 ──
print("\n[07] CLAHE tile grid: (4,4) / (8,8) / (12,12) / (16,16) / (24,24)")
r = {}
for tw, th in [(4,4), (8,8), (12,12), (16,16), (24,24)]:
    tmp_tw, tmp_th = config.preprocess.clahe_tile_w, config.preprocess.clahe_tile_h
    config.preprocess.clahe_tile_w, config.preprocess.clahe_tile_h = tw, th
    o = preprocess(img, clip_limit=D['clahe_clip'], bilateral_d=D['bilateral_d'],
                   bilateral_sigma=D['bilateral_sigma'], gamma=D['gamma'], median_ksize=D['median_ksize'])
    config.preprocess.clahe_tile_w, config.preprocess.clahe_tile_h = tmp_tw, tmp_th
    r[f"{tw}x{th}"] = (o['enhanced'], o['binary_adaptive'])
make_comparison_grid(r, 'clahe_tile', '07_clahe_tile_comparison.png')

# ── 08. Adaptive block_size ──
print("\n[08] Adaptive block_size: 11 / 21 / 31 / 41 / 51 / 71")
r = {}
for bs in [11, 21, 31, 41, 51, 71]:
    tmp = config.preprocess.adaptive_block_size
    config.preprocess.adaptive_block_size = bs
    o = preprocess(img, clip_limit=D['clahe_clip'], bilateral_d=D['bilateral_d'],
                   bilateral_sigma=D['bilateral_sigma'], gamma=D['gamma'], median_ksize=D['median_ksize'])
    config.preprocess.adaptive_block_size = tmp
    # 直接用二值图展示
    h, w = img.shape[:2]
    s = min(300 / max(h, w), 1.0)
    tw, th = int(w*s), int(h*s)
    bin_r = cv2.resize(o['binary_adaptive'], (tw, th))
    bin_r = cv2.cvtColor(bin_r, cv2.COLOR_GRAY2BGR)
    r[bs] = (bin_r, bin_r)  # enhanced slot used for binary display
make_comparison_grid(r, 'block_size', '08_adaptive_block_comparison.png')

# ── 09. Adaptive C ──
print("\n[09] Adaptive C: 0 / 2 / 5 / 8 / 12 / 16")
r = {}
for c in [0, 2, 5, 8, 12, 16]:
    tmp = config.preprocess.adaptive_C
    config.preprocess.adaptive_C = c
    o = preprocess(img, clip_limit=D['clahe_clip'], bilateral_d=D['bilateral_d'],
                   bilateral_sigma=D['bilateral_sigma'], gamma=D['gamma'], median_ksize=D['median_ksize'])
    config.preprocess.adaptive_C = tmp
    h, w = img.shape[:2]
    s = min(300 / max(h, w), 1.0)
    tw, th = int(w*s), int(h*s)
    bin_r = cv2.resize(o['binary_adaptive'], (tw, th))
    bin_r = cv2.cvtColor(bin_r, cv2.COLOR_GRAY2BGR)
    r[c] = (bin_r, bin_r)
make_comparison_grid(r, 'C', '09_adaptive_C_comparison.png')

# ── 10. Unsharp blur sigma ──
print("\n[10] Unsharp blur sigma: 0.5 / 1.0 / 1.5 / 2.0 / 3.0 / 5.0")
r = {}
for ub in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    tmp = config.preprocess.unsharp_blur_sigma
    config.preprocess.unsharp_blur_sigma = ub
    o = preprocess(img, clip_limit=D['clahe_clip'], bilateral_d=D['bilateral_d'],
                   bilateral_sigma=D['bilateral_sigma'], gamma=D['gamma'], median_ksize=D['median_ksize'])
    config.preprocess.unsharp_blur_sigma = tmp
    r[ub] = (o['enhanced'], o['binary_adaptive'])
make_comparison_grid(r, 'unsharp_sigma', '10_unsharp_sigma_comparison.png')


# ═══════════════════════════════════════════════════════════════
#  第二部分：参数交叉组合对比
# ═══════════════════════════════════════════════════════════════

# ── 11. Gamma × CLAHE clip 交叉对比 ──
print("\n[11] Gamma × CLAHE cross comparison")
r = {}
for g in [0.5, 0.8, 1.0, 1.5]:
    for cc in [1.0, 2.0, 4.0, 8.0]:
        o = preprocess(img, clip_limit=cc, bilateral_d=D['bilateral_d'],
                       bilateral_sigma=D['bilateral_sigma'], gamma=g, median_ksize=D['median_ksize'])
        r[f"g={g},cc={cc}"] = (o['enhanced'], o['binary_adaptive'])
make_comparison_grid(r, 'gamma×clahe', '11_gamma_x_clahe.png')

# ── 12. Bilateral × Median 交叉对比 ──
print("\n[12] Bilateral × Median cross comparison")
r = {}
for bd in [5, 15]:
    for bs in [25.0, 75.0]:
        for mk in [0, 5]:
            o = preprocess(img, clip_limit=D['clahe_clip'], bilateral_d=bd,
                           bilateral_sigma=bs, gamma=D['gamma'], median_ksize=mk)
            r[f"d={bd},s={bs},mk={mk}"] = (o['enhanced'], o['binary_adaptive'])
make_comparison_grid(r, 'bilateral×median', '12_bilateral_x_median.png')

# ── 13. CLAHE tile × clip 交叉对比 ──
print("\n[13] CLAHE tile × clip cross comparison")
r = {}
for tw_th in [(4,4), (8,8), (16,16)]:
    for cc in [1.0, 4.0, 8.0]:
        tmp_tw, tmp_th = config.preprocess.clahe_tile_w, config.preprocess.clahe_tile_h
        config.preprocess.clahe_tile_w, config.preprocess.clahe_tile_h = tw_th
        o = preprocess(img, clip_limit=cc, bilateral_d=D['bilateral_d'],
                       bilateral_sigma=D['bilateral_sigma'], gamma=D['gamma'], median_ksize=D['median_ksize'])
        config.preprocess.clahe_tile_w, config.preprocess.clahe_tile_h = tmp_tw, tmp_th
        r[f"tile={tw_th[0]}x{tw_th[1]},clip={cc}"] = (o['enhanced'], o['binary_adaptive'])
make_comparison_grid(r, 'tile×clip', '13_clahe_tile_x_clip.png')

# ── 14. block_size × C（自适应阈值交叉）──
print("\n[14] block_size × C cross comparison")
r = {}
for bs in [11, 31, 51]:
    for c_val in [0, 5, 12]:
        tmp_bs = config.preprocess.adaptive_block_size
        tmp_c = config.preprocess.adaptive_C
        config.preprocess.adaptive_block_size = bs
        config.preprocess.adaptive_C = c_val
        o = preprocess(img, clip_limit=D['clahe_clip'], bilateral_d=D['bilateral_d'],
                       bilateral_sigma=D['bilateral_sigma'], gamma=D['gamma'], median_ksize=D['median_ksize'])
        config.preprocess.adaptive_block_size = tmp_bs
        config.preprocess.adaptive_C = tmp_c
        h, w = img.shape[:2]
        s = min(300 / max(h, w), 1.0)
        tw, th = int(w*s), int(h*s)
        bin_r = cv2.resize(o['binary_adaptive'], (tw, th))
        bin_r = cv2.cvtColor(bin_r, cv2.COLOR_GRAY2BGR)
        r[f"blk={bs},C={c_val}"] = (bin_r, bin_r)
make_comparison_grid(r, 'block×C', '14_adaptive_block_x_C.png')


# ═══════════════════════════════════════════════════════════════
#  第三部分：Top / Bottom 组合 Debug Vis
# ═══════════════════════════════════════════════════════════════

# ── 15. Top 12 组合全景 ──
print("\n[15] Top 12 parameter combinations debug vis")
top_12 = [
    ("#1: g=0.8,d=5,s=25,mk=0,cc=4,ua=1.5", 0.8, 5, 25.0, 0, 4.0, 1.5),
    ("#2: g=1.0,d=5,s=25,mk=0,cc=4,ua=1.5", 1.0, 5, 25.0, 0, 4.0, 1.5),
    ("#3: g=0.6,d=5,s=25,mk=0,cc=4,ua=1.5", 0.6, 5, 25.0, 0, 4.0, 1.5),
    ("#4: g=1.2,d=5,s=25,mk=0,cc=4,ua=1.5", 1.2, 5, 25.0, 0, 4.0, 1.5),
    ("#5: g=0.8,d=5,s=25,mk=3,cc=4,ua=1.5", 0.8, 5, 25.0, 3, 4.0, 1.5),
    ("#6: g=1.0,d=5,s=25,mk=3,cc=4,ua=1.5", 1.0, 5, 25.0, 3, 4.0, 1.5),
    ("#7: g=0.8,d=5,s=50,mk=0,cc=4,ua=1.5", 0.8, 5, 50.0, 0, 4.0, 1.5),
    ("#8: g=0.6,d=5,s=25,mk=3,cc=4,ua=1.5", 0.6, 5, 25.0, 3, 4.0, 1.5),
    ("#9: g=1.0,d=5,s=50,mk=0,cc=4,ua=1.5", 1.0, 5, 50.0, 0, 4.0, 1.5),
    ("#10: g=0.8,d=5,s=75,mk=0,cc=4,ua=1.5", 0.8, 5, 75.0, 0, 4.0, 1.5),
    ("#11: g=0.6,d=5,s=50,mk=0,cc=4,ua=1.5", 0.6, 5, 50.0, 0, 4.0, 1.5),
    ("#12: g=0.8,d=5,s=25,mk=0,cc=2,ua=1.5", 0.8, 5, 25.0, 0, 2.0, 1.5),
]
debug_list = []
for label, g, bd, bs, mk, cc, ua in top_12:
    o = preprocess(img, clip_limit=cc, bilateral_d=bd,
                   bilateral_sigma=bs, gamma=g, median_ksize=mk)
    debug_list.append((label, o['debug_vis']))
make_full_grid(debug_list, '15_top12_combinations.png')

# ── 16. Bottom 6 组合（低评分参数）──
print("\n[16] Bottom 6 parameter combinations debug vis")
bottom_6 = [
    ("Low: g=1.2,d=15,s=75,mk=5,cc=1,ua=0.5", 1.2, 15, 75.0, 5, 1.0, 0.5),
    ("Low: g=2.0,d=21,s=100,mk=9,cc=1,ua=0", 2.0, 21, 100.0, 9, 1.0, 0),
    ("Low: g=0.4,d=21,s=75,mk=7,cc=2,ua=0", 0.4, 21, 75.0, 7, 2.0, 0),
    ("Low: g=1.5,d=21,s=75,mk=9,cc=1,ua=0.5", 1.5, 21, 75.0, 9, 1.0, 0.5),
    ("Low: g=0.5,d=15,s=100,mk=5,cc=1,ua=0.5", 0.5, 15, 100.0, 5, 1.0, 0.5),
    ("Low: g=2.0,d=15,s=50,mk=7,cc=2,ua=0.5", 2.0, 15, 50.0, 7, 2.0, 0.5),
]
debug_list = []
for label, g, bd, bs, mk, cc, ua in bottom_6:
    o = preprocess(img, clip_limit=cc, bilateral_d=bd,
                   bilateral_sigma=bs, gamma=g, median_ksize=mk)
    debug_list.append((label, o['debug_vis']))
make_full_grid(debug_list, '16_bottom6_combinations.png')


# ═══════════════════════════════════════════════════════════════
#  第四部分：Debug Vis 参数独立对比
# ═══════════════════════════════════════════════════════════════

# ── 17. Gamma debug grid ──
print("\n[17] Gamma debug vis grid (0.4 ~ 2.0)")
gamma_debug = []
for g in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]:
    o = preprocess(img, clip_limit=D['clahe_clip'], bilateral_d=D['bilateral_d'],
                   bilateral_sigma=D['bilateral_sigma'], gamma=g, median_ksize=D['median_ksize'])
    gamma_debug.append((f"gamma={g}", o['debug_vis']))
make_full_grid(gamma_debug, '17_gamma_debug_grid.png')

# ── 18. CLAHE clip debug grid ──
print("\n[18] CLAHE clip debug vis grid")
cc_debug = []
for c in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
    o = preprocess(img, clip_limit=c, bilateral_d=D['bilateral_d'],
                   bilateral_sigma=D['bilateral_sigma'], gamma=D['gamma'], median_ksize=D['median_ksize'])
    cc_debug.append((f"clahe_clip={c}", o['debug_vis']))
make_full_grid(cc_debug, '18_clahe_debug_grid.png')

# ── 19. Median ksize debug grid ──
print("\n[19] Median ksize debug vis grid")
mk_debug = []
for k in [0, 3, 5, 7, 9]:
    o = preprocess(img, clip_limit=D['clahe_clip'], bilateral_d=D['bilateral_d'],
                   bilateral_sigma=D['bilateral_sigma'], gamma=D['gamma'], median_ksize=k)
    mk_debug.append((f"median_ksize={k}", o['debug_vis']))
make_full_grid(mk_debug, '19_median_debug_grid.png')

# ── 20. Bilateral debug grid ──
print("\n[20] Bilateral d×sigma debug vis grid")
bilat_debug = []
for d in [5, 9, 15]:
    for s in [25.0, 50.0, 75.0]:
        o = preprocess(img, clip_limit=D['clahe_clip'], bilateral_d=d,
                       bilateral_sigma=s, gamma=D['gamma'], median_ksize=D['median_ksize'])
        bilat_debug.append((f"d={d},s={s}", o['debug_vis']))
make_full_grid(bilat_debug, '20_bilateral_debug_grid.png')


# ═══════════════════════════════════════════════════════════════
#  第五部分：极值组合对比（最佳 vs 最差）
# ═══════════════════════════════════════════════════════════════

# ── 21. Optimal vs Over-processed vs Under-processed ──
print("\n[21] Optimal vs Extreme processing styles")
styles = [
    ("OPTIMAL: g=0.8,d=5,s=25,mk=0,cc=4,ua=1.5", 0.8, 5, 25.0, 0, 4.0, 1.5),
    ("OVER_SHARP: g=0.5,d=5,s=10,mk=0,cc=8,ua=3.0", 0.5, 5, 10.0, 0, 8.0, 3.0),
    ("OVER_SMOOTH: g=1.5,d=21,s=100,mk=9,cc=1,ua=0", 1.5, 21, 100.0, 9, 1.0, 0),
    ("OVER_DARK: g=0.3,d=5,s=25,mk=0,cc=8,ua=2.0", 0.3, 5, 25.0, 0, 8.0, 2.0),
    ("OVER_BRIGHT: g=3.0,d=5,s=25,mk=0,cc=1,ua=0", 3.0, 5, 25.0, 0, 1.0, 0),
    ("MID_CONTRAST: g=1.0,d=9,s=50,mk=3,cc=3,ua=1.0", 1.0, 9, 50.0, 3, 3.0, 1.0),
]
debug_list = []
for label, g, bd, bs, mk, cc, ua in styles:
    o = preprocess(img, clip_limit=cc, bilateral_d=bd,
                   bilateral_sigma=bs, gamma=g, median_ksize=mk)
    debug_list.append((label, o['debug_vis']))
make_full_grid(debug_list, '21_optimal_vs_extreme.png')

print(f"\n{'='*60}")
print(f"All {21} visualizations saved to: {out_dir}")
print(f"{'='*60}")
