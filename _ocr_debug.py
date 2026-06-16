"""
OCR 诊断脚本 — 生成三张主尺数字识别过程图
  1. 主尺原图 + 备选区红框 + 零线
  2. 备选区裁出图 (blow up 4x)
  3. OCR 调试图 (连通域 + 识别结果)
"""

import cv2
import numpy as np
import os

from caliper import CaliperPipeline
from caliper.main_scale import find_nearest_cm_digit_region, find_largest_digit_cc
from caliper.ocr import get_ocr_reader_singleton

IMG_PATH = r"e:\朱\caliper-reader-master\youbiaokachi.jpg"
OUT_DIR  = r"e:\朱\caliper-reader-master\_debug_out"

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. 跑流水线 ──
img = cv2.imdecode(np.fromfile(IMG_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
print(f"原图: {img.shape}")

pipeline = CaliperPipeline()
result = pipeline.run(img)

print(f"总读数: {result.total:.2f} mm | 主尺: {result.main_scale:.2f} | "
      f"游标: {result.vernier_scale:.3f} | 精度: {result.precision}")

# ── 2. 取出步骤数据 ──
sr     = pipeline.step_results
main   = sr['main']
split  = sr['split']
orient = sr['orient']

region_main   = split['region_main']          # {'image','binary','y_offset','height'}
main_ticks    = main['main_ticks']
main_gap      = main['main_gap']
zero_x        = result.extra_info['zero_x']
main_gray     = region_main['image']           # 灰度图 (已旋转矫正后)
main_binary   = region_main['binary']          # 二值图
main_color    = orient['rotated_color'][:split['split_y'], :]  # 彩色主尺区域

print(f"main_gap={main_gap:.1f}px  zero_x={zero_x:.0f}  ticks={len(main_ticks)}")

# ════════════════════════════════════════════════════════
#  图1: 主尺彩色图 + 备选区红框 + 零线黄线
# ════════════════════════════════════════════════════════
disp1 = main_color.copy()
H1, W1 = disp1.shape[:2]

# 画零线（黄色竖线）
cv2.line(disp1, (int(zero_x), 0), (int(zero_x), H1 - 1), (0, 255, 255), 2)
cv2.putText(disp1, f"ZERO x={int(zero_x)}", (int(zero_x) + 5, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# 获取备选区
binary_crop, x_off, y_off = find_nearest_cm_digit_region(
    main_ticks, main_gap, zero_x, main_binary)

if binary_crop is not None:
    ch, cw = binary_crop.shape
    # 画红框（备选区）
    cv2.rectangle(disp1, (x_off, y_off), (x_off + cw, y_off + ch), (0, 0, 255), 3)
    cv2.putText(disp1, f"backup region ({cw}x{ch})",
                (x_off + 4, y_off + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    print(f"备选区: x={x_off}..{x_off+cw}, y={y_off}..{y_off+ch}, size={cw}x{ch}")
else:
    print("⚠️ 备选区为空！")
    ch, cw = 0, 0

# 标注主尺刻线（半透明绿色短线）
for t in main_ticks:
    cv2.line(disp1, (t['x'], max(0, t['y_start'])),
             (t['x'], min(H1 - 1, t['y_end'])), (0, 200, 80), 1)

# 顶部标题
cv2.putText(disp1, "Main Scale + Backup Region (red) + Zero Line (yellow)",
            (6, H1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

cv2.imencode('.png', disp1)[1].tofile(
    os.path.join(OUT_DIR, "ocr_1_main_with_backup_region.png"))
print(f"  saved: ocr_1_main_with_backup_region.png  {disp1.shape}")

# ════════════════════════════════════════════════════════
#  图2: 备选区裁出图 (blow up 4x)
# ════════════════════════════════════════════════════════
if binary_crop is not None and binary_crop.size > 0:
    blow = cv2.resize(binary_crop,
                      (cw * 4, ch * 4),
                      interpolation=cv2.INTER_NEAREST)
    # 转彩色以便画标注
    blow_color = cv2.cvtColor(blow, cv2.COLOR_GRAY2BGR)

    # 标注尺寸信息
    cv2.putText(blow_color, f"Backup Region ({cw}x{ch})  x4",
                (4, blow_color.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    cv2.imencode('.png', blow_color)[1].tofile(
        os.path.join(OUT_DIR, "ocr_2_backup_region_blowup.png"))
    print(f"  saved: ocr_2_backup_region_blowup.png  {blow_color.shape}")
else:
    # 空图占位
    empty = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(empty, "NO BACKUP REGION", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
    cv2.imencode('.png', empty)[1].tofile(
        os.path.join(OUT_DIR, "ocr_2_backup_region_blowup.png"))
    print("  saved: ocr_2_backup_region_blowup.png (empty)")

# ════════════════════════════════════════════════════════
#  图3: OCR 调试图 — 备选区 + 所有连通域 + 选中 + OCR 结果
# ════════════════════════════════════════════════════════
if binary_crop is not None and binary_crop.size > 0:
    # 用彩色底图（从主尺彩色图中裁出备选区对应位置）
    disp3_color = main_color[y_off:y_off + ch, x_off:x_off + cw].copy()
    # 如果裁出来是灰度，转彩色
    if len(disp3_color.shape) == 2:
        disp3_color = cv2.cvtColor(disp3_color, cv2.COLOR_GRAY2BGR)

    # ── 画出所有连通域（蓝色框）──
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary_crop, connectivity=8)
    for j in range(1, num_labels):
        x = int(stats[j, cv2.CC_STAT_LEFT])
        y = int(stats[j, cv2.CC_STAT_TOP])
        w_cc = int(stats[j, cv2.CC_STAT_WIDTH])
        h_cc = int(stats[j, cv2.CC_STAT_HEIGHT])
        area = int(stats[j, cv2.CC_STAT_AREA])
        # 蓝色框 = 所有连通域
        cv2.rectangle(disp3_color, (x, y), (x + w_cc, y + h_cc), (255, 150, 50), 1)
        # 标面积
        cv2.putText(disp3_color, f"{area}", (x, max(y - 2, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 150, 50), 1)

    # ── 用 find_largest_digit_cc 找最佳数字连通域 ──
    digit_crop, bbox, cc_conf = find_largest_digit_cc(binary_crop, x_off, y_off)

    if bbox is not None:
        # 映射回备选区坐标
        bx1 = bbox[0] - x_off
        by1 = bbox[1] - y_off
        bx2 = bbox[2] - x_off
        by2 = bbox[3] - y_off
        # 绿色粗框 = 选中的连通域
        cv2.rectangle(disp3_color, (bx1, by1), (bx2, by2), (0, 255, 80), 3)
        cv2.putText(disp3_color, f"SELECTED conf={cc_conf:.2f}",
                    (bx1, max(by1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 80), 2)

        # ── OCR 识别 ──
        reader = get_ocr_reader_singleton()
        eng = reader.engine_status()
        digit = reader.ocr_patch_to_digit(digit_crop, bbox, main_gray)

        if digit is not None and digit.value >= 0:
            ocr_text = f"OCR: {digit.value} (conf={digit.confidence:.2f}) [{eng}]"
            ocr_color = (0, 255, 100)
        else:
            ocr_text = f"OCR: FAILED [{eng}]"
            ocr_color = (0, 100, 255)
    else:
        ocr_text = "NO CC FOUND (all filtered out)"
        ocr_color = (0, 100, 255)

    # 顶部标识
    cv2.putText(disp3_color, f"CCs={num_labels - 1} | {ocr_text}",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ocr_color, 2)

    cv2.imencode('.png', disp3_color)[1].tofile(
        os.path.join(OUT_DIR, "ocr_3_digit_debug.png"))
    print(f"  saved: ocr_3_digit_debug.png  {disp3_color.shape}")
else:
    empty = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(empty, "NO BACKUP REGION", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
    cv2.imencode('.png', empty)[1].tofile(
        os.path.join(OUT_DIR, "ocr_3_digit_debug.png"))
    print("  saved: ocr_3_digit_debug.png (empty)")

print("\n=== 图3 连通域诊断 ===")
if binary_crop is not None and binary_crop.size > 0:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_crop, connectivity=8)
    print(f"连通域总数: {num_labels - 1}")
    for j in range(1, num_labels):
        x = int(stats[j, cv2.CC_STAT_LEFT])
        y = int(stats[j, cv2.CC_STAT_TOP])
        w_cc = int(stats[j, cv2.CC_STAT_WIDTH])
        h_cc = int(stats[j, cv2.CC_STAT_HEIGHT])
        area = int(stats[j, cv2.CC_STAT_AREA])
        aspect = h_cc / max(w_cc, 1)
        print(f"  CC#{j}: x={x} y={y} w={w_cc} h={h_cc} area={area} aspect={aspect:.2f}")

    # 也打印 find_largest_digit_cc 的筛选过程
    print("\n=== find_largest_digit_cc 筛选 ===")
    for j in range(1, num_labels):
        x = int(stats[j, cv2.CC_STAT_LEFT])
        y = int(stats[j, cv2.CC_STAT_TOP])
        w_cc = int(stats[j, cv2.CC_STAT_WIDTH])
        h_cc = int(stats[j, cv2.CC_STAT_HEIGHT])
        area = int(stats[j, cv2.CC_STAT_AREA])
        aspect = h_cc / max(w_cc, 1)
        y_ratio = (y + h_cc/2) / ch
        x_ratio = (x + w_cc/2) / cw
        flags = []
        if area < 50:   flags.append("area<50")
        if area > 600:  flags.append("area>600")
        if aspect < 0.6: flags.append("aspect<0.6")
        if aspect > 3.5: flags.append("aspect>3.5")
        status = "REJECT: " + ",".join(flags) if flags else "PASS"
        score = x_ratio * 0.6 + y_ratio * 0.2 + (area / 600) * 0.2 if not flags else 0
        print(f"  CC#{j}: area={area} aspect={aspect:.2f} y_ratio={y_ratio:.2f} x_ratio={x_ratio:.2f} score={score:.3f} → {status}")

print("\n✅ Done.")
