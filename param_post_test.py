"""
后处理（形态学开运算 + 连通域过滤）专项对比测试
NLM 在小尺度图上测试（大图太慢）
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
img_full = cv2.imdecode(img, cv2.IMREAD_COLOR)

# 小图用于 NLM 加速
img_small = cv2.resize(img_full, (400, 300))

out_dir = Path(__file__).parent / "param_post_test"
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / out_name), canvas)
    print(f"  -> {out_name}")


def make_zoom_grid(rows_data, out_name):
    """二值放大特写竖向拼接"""
    binary_rows = []
    h, w = img_full.shape[:2]
    roi_y0, roi_y1 = h // 3, h // 3 + 120
    for label, bw in rows_data:
        crop = bw[roi_y0:roi_y1, :]
        crop = cv2.resize(crop, (w * 2, 280), interpolation=cv2.INTER_NEAREST)
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        cv2.putText(crop_bgr, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
        binary_rows.append(crop_bgr)
    pano = np.vstack(binary_rows)
    cv2.imwrite(str(out_dir / out_name), pano)
    print(f"  -> {out_name}")


def apply_post(morph_en=False, morph_k=2, morph_it=1, cc_en=False, cc_min=8):
    config.preprocess.morph_open_enabled = morph_en
    config.preprocess.morph_open_kernel_size = morph_k
    config.preprocess.morph_open_iterations = morph_it
    config.preprocess.cc_filter_enabled = cc_en
    config.preprocess.cc_min_area = cc_min

def reset_post():
    config.preprocess.morph_open_enabled = False
    config.preprocess.cc_filter_enabled = False


# ════════════════════════════════════════════════
#  测试 1: 形态学开运算（大图）kernel 尺寸对比
# ════════════════════════════════════════════════
print("=" * 60)
print("[1] 形态学开运算 kernel 对比")
reset_post()
out_base = preprocess(img_full)

debug_list = [("Baseline (无后处理)", out_base['debug_vis'])]
zoom_rows = [("Baseline", out_base['binary_adaptive'])]

for k in [2, 3, 4]:
    apply_post(morph_en=True, morph_k=k)
    out = preprocess(img_full)
    debug_list.append((f"MorphOpen k={k}", out['debug_vis']))
    zoom_rows.append((f"MorphOpen k={k}", out['binary_adaptive']))

make_grid(debug_list, "01_morph_open.png")
make_zoom_grid(zoom_rows, "01_morph_open_zoom.png")


# ════════════════════════════════════════════════
#  测试 2: 连通域过滤 min_area 对比
# ════════════════════════════════════════════════
print("\n[2] 连通域过滤 min_area 对比")
reset_post()
debug_list = [("Baseline", out_base['debug_vis'])]
zoom_rows = [("Baseline", out_base['binary_adaptive'])]

for min_a in [5, 10, 20, 40, 80]:
    apply_post(cc_en=True, cc_min=min_a)
    out = preprocess(img_full)
    debug_list.append((f"CCFilter min={min_a}", out['debug_vis']))
    zoom_rows.append((f"CCFilter min={min_a}", out['binary_adaptive']))

make_grid(debug_list, "02_cc_filter.png")
make_zoom_grid(zoom_rows, "02_cc_filter_zoom.png")


# ════════════════════════════════════════════════
#  测试 3: Morph + CC 组合对比
# ════════════════════════════════════════════════
print("\n[3] Morph + CC 组合对比")
reset_post()
combos = [
    ("Baseline",            False, 0,  False, 0),
    ("Morph k=2",           True,  2,  False, 0),
    ("CC min=10",           False, 0,  True,  10),
    ("CC min=20",           False, 0,  True,  20),
    ("Morph k=2 + CC 10",   True,  2,  True,  10),
    ("Morph k=2 + CC 20",   True,  2,  True,  20),
    ("Morph k=3 + CC 10",   True,  3,  True,  10),
    ("Morph k=3 + CC 20",   True,  3,  True,  20),
]

debug_list = []
zoom_rows = []
for label, m_en, mk, cc_en, cc_min in combos:
    apply_post(morph_en=m_en, morph_k=mk, cc_en=cc_en, cc_min=cc_min)
    out = preprocess(img_full)
    debug_list.append((label, out['debug_vis']))
    zoom_rows.append((label, out['binary_adaptive']))

make_grid(debug_list, "03_combo_morph_cc.png")
make_zoom_grid(zoom_rows, "03_combo_morph_cc_zoom.png")


# ════════════════════════════════════════════════
#  测试 4: NLM 强度对比（小图加速）
# ════════════════════════════════════════════════
print("\n[4] NLM h 强度对比 (小图 400x300)")
reset_post()
config.preprocess.nlm_enabled = False
out_base = preprocess(img_small)

debug_list_small = [("Baseline (无 NLM)", out_base['debug_vis'])]

for h in [5, 10, 15, 20]:
    config.preprocess.nlm_enabled = True
    config.preprocess.nlm_h = h
    config.preprocess.nlm_template_win = 7
    config.preprocess.nlm_search_win = 21
    out = preprocess(img_small)
    debug_list_small.append((f"NLM h={h}", out['debug_vis']))

config.preprocess.nlm_enabled = False
make_grid(debug_list_small, "04_nlm_h_small.png")


# ════════════════════════════════════════════════
#  测试 5: NLM（小图）+ Morph + CC 组合
# ════════════════════════════════════════════════
print("\n[5] NLM + Morph + CC (小图)")
reset_post()
config.preprocess.nlm_enabled = False

combos_small = [
    ("Baseline",              False, 0),
    ("NLM h=10",              True,  10),
    ("NLM h=10 + Morph k=2",  True,  10, True, 2, False, 0),
    ("NLM h=10 + CC 10",      True,  10, False, 0, True, 10),
    ("NLM h=10 + M2 + CC10",  True,  10, True, 2, True, 10),
    ("NLM h=15 + M2 + CC10",  True,  15, True, 2, True, 10),
]

debug_list_small = []
for item in combos_small:
    label = item[0]
    nlm_en, nlm_h = item[1], item[2]
    m_en = item[3] if len(item) > 3 else False
    mk = item[4] if len(item) > 4 else 0
    cc_en = item[5] if len(item) > 5 else False
    cc_min = item[6] if len(item) > 6 else 0

    config.preprocess.nlm_enabled = nlm_en
    config.preprocess.nlm_h = nlm_h
    apply_post(morph_en=m_en, morph_k=mk, cc_en=cc_en, cc_min=cc_min)
    out = preprocess(img_small)
    debug_list_small.append((label, out['debug_vis']))

config.preprocess.nlm_enabled = False
reset_post()
make_grid(debug_list_small, "05_nlm_morph_cc_small.png")

# ════════════════════════════════════════════════
#  测试 6: NLM h=10 大图确认测试
# ════════════════════════════════════════════════
print("\n[6] NLM h=10 大图实际测试 (可能需要30秒)")
reset_post()
config.preprocess.nlm_enabled = True
config.preprocess.nlm_h = 10
config.preprocess.nlm_template_win = 7
config.preprocess.nlm_search_win = 21
out_nlm = preprocess(img_full)
config.preprocess.nlm_enabled = False

debug_list = [
    ("Baseline (当前配置)", out_base['debug_vis']),
    ("NLM h=10 (全图)", out_nlm['debug_vis']),
]

apply_post(morph_en=True, morph_k=2, cc_en=True, cc_min=10)
config.preprocess.nlm_enabled = True
config.preprocess.nlm_h = 10
out_full = preprocess(img_full)
config.preprocess.nlm_enabled = False
reset_post()

debug_list.append(("NLM + Morph k=2 + CC 10", out_full['debug_vis']))
make_grid(debug_list, "06_nlm_fullsize.png")

# 放大特写
zoom_rows = [
    ("Baseline",              out_base['binary_adaptive']),
    ("NLM h=10",              out_nlm['binary_adaptive']),
    ("NLM + M2 + CC10",       out_full['binary_adaptive']),
]
make_zoom_grid(zoom_rows, "06_nlm_fullsize_zoom.png")

print(f"\n{'='*60}")
print(f"Done! All results: {out_dir}")
print(f"{'='*60}")
