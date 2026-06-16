"""
多组参数对比测试：围绕"降噪不损清晰度"的目标测试若干组合
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

out_dir = Path(__file__).parent / "param_test_compare"
out_dir.mkdir(parents=True, exist_ok=True)

# 清空旧文件
for f in out_dir.glob("*.png"):
    f.unlink()


def make_grid(images_with_labels, out_name, title=""):
    """多张 debug_vis 排列成网格"""
    n = len(images_with_labels)
    if n == 0:
        return
    cell_h, cell_w = images_with_labels[0][1].shape[:2]
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    margin = 20
    canvas = np.ones((rows * (cell_h + 35) + margin, cols * (cell_w + 10) + margin, 3), dtype=np.uint8) * 50

    for idx, (label, vis) in enumerate(images_with_labels):
        row, col = idx // cols, idx % cols
        x0, y0 = margin // 2 + col * (cell_w + 10), margin // 2 + row * (cell_h + 35)
        h, w = vis.shape[:2]
        canvas[y0:y0 + h, x0:x0 + w] = vis
        cv2.putText(canvas, label, (x0 + 10, y0 + h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)

    if title:
        cv2.putText(canvas, title, (10, canvas.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / out_name), canvas)
    print(f"  -> {out_name}")


# ═══════════════════════════════════════
#  8 组参数测试
# ═══════════════════════════════════════
# 格式: (标签, gamma, bilateral_d, bilateral_sigma, median_ksize, clahe_clip, unsharp_amount)

combos = [
    ("A 当前配置: d=7 s=35 mk=3 cc=3 ua=1.3",         0.8, 7, 35.0, 3, 3.0, 1.3),
    ("B 原始默认: d=5 s=25 mk=0 cc=4 ua=1.5",          0.8, 5, 25.0, 0, 4.0, 1.5),
    ("C Top1参数: d=5 s=25 mk=0 cc=4 ua=1.5",          0.8, 5, 25.0, 0, 4.0, 1.5),
    ("D 轻降噪: d=5 s=20 mk=3 cc=3 ua=1.2",            0.8, 5, 20.0, 3, 3.0, 1.2),
    ("E 中降噪: d=7 s=40 mk=5 cc=3 ua=1.3",            0.8, 7, 40.0, 5, 3.0, 1.3),
    ("F 强降噪: d=9 s=60 mk=5 cc=2.5 ua=1.0",          0.8, 9, 60.0, 5, 2.5, 1.0),
    ("G 极保守: d=3 s=15 mk=0 cc=3 ua=1.0",            0.8, 3, 15.0, 0, 3.0, 1.0),
    ("H 锐化加强: d=7 s=35 mk=3 cc=4 ua=1.8",          0.8, 7, 35.0, 3, 4.0, 1.8),
]

print("生成 8 组参数对比...")
debug_vis_list = []
for label, g, bd, bs, mk, cc, ua in combos:
    out = preprocess(img, clip_limit=cc, bilateral_d=bd,
                     bilateral_sigma=bs, gamma=g, median_ksize=mk)
    debug_vis_list.append((label, out['debug_vis']))

make_grid(debug_vis_list, "00_all_8_combinations.png", "8 组预处理参数对比 (原图 | 增强 | 二值化)")
print("\nDone!")
