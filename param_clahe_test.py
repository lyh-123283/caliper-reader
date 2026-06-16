"""
CLAHE clip_limit 参数专项测试
固定其他参数，仅变化 clip_limit，观察噪声放大/抑制效果
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

out_dir = Path(__file__).parent / "param_clahe_clip_test"
out_dir.mkdir(parents=True, exist_ok=True)
for f in out_dir.glob("*.png"):
    f.unlink()

# ── 固定参数（当前 config 值）──
FIXED = {
    'gamma':            config.preprocess.gamma,
    'bilateral_d':      config.preprocess.bilateral_d,
    'bilateral_sigma':  config.preprocess.bilateral_sigma,
    'median_ksize':     config.preprocess.median_ksize,
    'unsharp_amount':   config.preprocess.unsharp_amount,
}


def make_debug_grid(images_with_labels, out_name):
    """debug_vis 网格，每行2个"""
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


# ════════════════════════════════════════════════
#  10 个 clip_limit 值：0.5 到 10.0
# ════════════════════════════════════════════════

clip_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0]

print(f"测试 {len(clip_values)} 个 clip_limit 值...")
print(f"固定参数: gamma={FIXED['gamma']}, b_d={FIXED['bilateral_d']}, "
      f"b_s={FIXED['bilateral_sigma']}, mk={FIXED['median_ksize']}, "
      f"ua={FIXED['unsharp_amount']}")

debug_list = []
for cc in clip_values:
    out = preprocess(
        img,
        clip_limit=cc,
        bilateral_d=FIXED['bilateral_d'],
        bilateral_sigma=FIXED['bilateral_sigma'],
        gamma=FIXED['gamma'],
        median_ksize=FIXED['median_ksize'],
    )
    label = f"clip_limit={cc}"
    debug_list.append((label, out['debug_vis']))

make_debug_grid(debug_list, "clip_limit_0.5_to_10.png")

# ════════════════════════════════════════════════
#  额外：clip_limit 对二值化效果的放大特写
# ════════════════════════════════════════════════
print("\n生成二值化放大特写...")
h, w = img.shape[:2]
roi_h, roi_y0, roi_y1 = 80, h // 3, h // 3 + 80
roi_w, roi_x0, roi_x1 = w, 0, w

binary_rows = []
for cc in clip_values:
    out = preprocess(
        img,
        clip_limit=cc,
        bilateral_d=FIXED['bilateral_d'],
        bilateral_sigma=FIXED['bilateral_sigma'],
        gamma=FIXED['gamma'],
        median_ksize=FIXED['median_ksize'],
    )
    bw = out['binary_adaptive']
    # 截取中间一段放大
    crop = bw[roi_y0:roi_y1, roi_x0:roi_x1]
    crop = cv2.resize(crop, (w * 2, roi_h * 3), interpolation=cv2.INTER_NEAREST)
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    # 加标签
    cv2.putText(crop_bgr, f"clip={cc}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    binary_rows.append(crop_bgr)

# 竖向拼接
binary_pano = np.vstack(binary_rows)
cv2.imwrite(str(out_dir / "clip_limit_binary_zoom.png"), binary_pano)
print("  -> clip_limit_binary_zoom.png")

print("\nDone!")
