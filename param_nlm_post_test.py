"""
NLM 去噪 + 二值化后处理 专项对比测试
"""
import cv2
import numpy as np
from pathlib import Path
from caliper import config
from caliper.preprocess import preprocess

# ── 读取图像 ──
img_path = Path(__file__).parent / "test.jpg"
data = img_path.read_bytes()
img = np.frombuffer(data, dtype=np.uint8)
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

out_dir = Path(__file__).parent / "param_nlm_post_test"
out_dir.mkdir(parents=True, exist_ok=True)
for f in out_dir.glob("*.png"):
    f.unlink()


def make_grid(images_with_labels, out_name):
    n = len(images_with_labels)
    if n == 0:
        return
    cell_h, cell_w = images_with_labels[0][1].shape[:2]
    cols = 2
    rows = (n + cols - 1) // cols
    m = 20
    cw, ch = cell_w + 10, cell_h + 35
    canvas = np.ones((rows * ch + m, cols * cw + m, 3), dtype=np.uint8) * 50
    for idx, (label, vis) in enumerate(images_with_labels):
        r, c = idx // cols, idx % cols
        x0, y0 = m // 2 + c * cw, m // 2 + r * ch
        h, w = vis.shape[:2]
        canvas[y0:y0 + h, x0:x0 + w] = vis
        cv2.putText(canvas, label, (x0 + 10, y0 + h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / out_name), canvas)
    print(f"  -> {out_name}")


def apply_config(nlm_en=False, nlm_h=10.0, nlm_tw=7, nlm_sw=21,
                 morph_en=False, morph_k=2, morph_it=1,
                 cc_en=False, cc_min=8):
    """临时设置 config 参数"""
    config.preprocess.nlm_enabled = nlm_en
    config.preprocess.nlm_h = nlm_h
    config.preprocess.nlm_template_win = nlm_tw
    config.preprocess.nlm_search_win = nlm_sw
    config.preprocess.morph_open_enabled = morph_en
    config.preprocess.morph_open_kernel_size = morph_k
    config.preprocess.morph_open_iterations = morph_it
    config.preprocess.cc_filter_enabled = cc_en
    config.preprocess.cc_min_area = cc_min


# ════════════════════════════════════════════════
#  测试 1: NLM 强度对比（h=3 / 6 / 10 / 15）
# ════════════════════════════════════════════════
print("=" * 60)
print("[1] NLM h 强度对比")
print("=" * 60)

debug_list = []
for h in [0, 3, 6, 10, 15]:
    if h == 0:
        apply_config(nlm_en=False)
        label = "Baseline (无 NLM)"
    else:
        apply_config(nlm_en=True, nlm_h=h)
        label = f"NLM h={h}"
    out = preprocess(img)
    debug_list.append((label, out['debug_vis']))
make_grid(debug_list, "01_nlm_h_comparison.png")


# ════════════════════════════════════════════════
#  测试 2: 形态学开运算对比
# ════════════════════════════════════════════════
print("\n[2] 形态学开运算对比")

debug_list = []
# baseline (no post)
apply_config(nlm_en=False, morph_en=False, cc_en=False)
out = preprocess(img)
debug_list.append(("Baseline (无后处理)", out['debug_vis']))

# morph open only
for k in [2, 3, 4]:
    apply_config(nlm_en=False, morph_en=True, morph_k=k, cc_en=False)
    out = preprocess(img)
    debug_list.append((f"MorphOpen k={k}", out['debug_vis']))
make_grid(debug_list, "02_morph_open_comparison.png")


# ════════════════════════════════════════════════
#  测试 3: 连通域过滤对比
# ════════════════════════════════════════════════
print("\n[3] 连通域过滤对比")

debug_list = []
apply_config(nlm_en=False, morph_en=False, cc_en=False)
out = preprocess(img)
debug_list.append(("Baseline (无后处理)", out['debug_vis']))

for min_a in [5, 10, 20, 40]:
    apply_config(nlm_en=False, morph_en=False, cc_en=True, cc_min=min_a)
    out = preprocess(img)
    debug_list.append((f"CCFilter min={min_a}", out['debug_vis']))
make_grid(debug_list, "03_cc_filter_comparison.png")


# ════════════════════════════════════════════════
#  测试 4: NLM + 后处理 组合对比
# ════════════════════════════════════════════════
print("\n[4] NLM + 后处理组合对比")

combos = [
    ("Baseline",             False, 0,       False, 0, 0,  False, 0),
    ("NLM h=6",              True,  6,       False, 0, 0,  False, 0),
    ("MorphOpen k=2",        False, 0,       True,  2, 1,  False, 0),
    ("CCFilter min=10",      False, 0,       False, 0, 0,  True,  10),
    ("NLM + MorphOpen",      True,  6,       True,  2, 1,  False, 0),
    ("NLM + CCFilter",       True,  6,       False, 0, 0,  True,  10),
    ("Morph + CC",           False, 0,       True,  2, 1,  True,  10),
    ("NLM + Morph + CC",     True,  6,       True,  2, 1,  True,  10),
    ("NLM h=10 + Morph k=3", True,  10,      True,  3, 1,  False, 0),
    ("全开: h=10 + M k=3 + CC 15", True, 10, True, 3, 1, True, 15),
]

debug_list = []
for label, nlm, nh, m_en, mk, mi, cc_en, cc_min in combos:
    apply_config(nlm_en=nlm, nlm_h=nh,
                 morph_en=m_en, morph_k=mk, morph_it=mi,
                 cc_en=cc_en, cc_min=cc_min)
    out = preprocess(img)
    debug_list.append((label, out['debug_vis']))
make_grid(debug_list, "04_all_combinations.png")


# ════════════════════════════════════════════════
#  测试 5: 二值化放大特写 (最佳 6 组)
# ════════════════════════════════════════════════
print("\n[5] 二值化放大特写")

h, w = img.shape[:2]
roi_y0, roi_y1 = h // 3, h // 3 + 100
roi_x0, roi_x1 = 0, w

best_combos = [
    ("基线",        False, 0,  False, 0, 0,  False, 0),
    ("NLM h=6",     True,  6,  False, 0, 0,  False, 0),
    ("NLM h=10",    True,  10, False, 0, 0,  False, 0),
    ("Morph k=2",   False, 0,  True,  2, 1,  False, 0),
    ("CC min=10",   False, 0,  False, 0, 0,  True,  10),
    ("NLM+Morph+CC",True, 10,  True,  3, 1,  True,  15),
]

binary_rows = []
for label, nlm, nh, m_en, mk, mi, cc_en, cc_min in best_combos:
    apply_config(nlm_en=nlm, nlm_h=nh,
                 morph_en=m_en, morph_k=mk, morph_it=mi,
                 cc_en=cc_en, cc_min=cc_min)
    out = preprocess(img)
    bw = out['binary_adaptive']
    crop = bw[roi_y0:roi_y1, roi_x0:roi_x1]
    crop = cv2.resize(crop, (w * 2, 300), interpolation=cv2.INTER_NEAREST)
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    cv2.putText(crop_bgr, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    binary_rows.append(crop_bgr)

binary_pano = np.vstack(binary_rows)
cv2.imwrite(str(out_dir / "05_binary_zoom.png"), binary_pano)
print("  -> 05_binary_zoom.png")

# 恢复默认
apply_config(nlm_en=False, morph_en=False, cc_en=False)

print(f"\n{'='*60}")
print(f"Done! All results in: {out_dir}")
print(f"{'='*60}")
