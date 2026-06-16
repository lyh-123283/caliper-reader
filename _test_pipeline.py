"""
独立测试脚本：对 youbiaokachi.jpg 跑完整流水线，
把每一步中间图像保存到 _debug_out/ 目录，方便逐步评估效果。
"""

import cv2
import numpy as np
import os
import sys
import traceback

# 防止 OCR 慢启动
os.environ.setdefault("OMP_NUM_THREADS", "2")

from caliper import CaliperPipeline

IMG_PATH = r"e:\朱\caliper-reader-master\youbiaokachi.jpg"
OUT_DIR = r"e:\朱\caliper-reader-master\_debug_out"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 读取图像（中文路径用 imdecode）
    img = cv2.imdecode(np.fromfile(IMG_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERR] 无法读取图像：{IMG_PATH}")
        sys.exit(1)
    print(f"[OK] 读取图像 {img.shape[1]}x{img.shape[0]}")

    # 保存原图（resize 小一点便于查看）
    h, w = img.shape[:2]
    sc = min(1.0, 1200 / max(h, w))
    if sc < 1.0:
        small = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    else:
        small = img
    cv2.imencode('.jpg', small)[1].tofile(os.path.join(OUT_DIR, "00_原图.jpg"))

    # 2. 跑流水线
    pipeline = CaliperPipeline()
    try:
        result = pipeline.run(img)
    except Exception as e:
        traceback.print_exc()
        print(f"[ERR] 流水线运行失败: {e}")
        sys.exit(2)

    # 3. 打印结果
    print("\n========== 识别结果 ==========")
    print(f"  主尺读数 main_scale  = {result.main_scale:.2f} mm")
    print(f"  游标读数 vernier     = {result.vernier_scale:.3f} mm")
    print(f"  总读数  total       = {result.total:.3f} mm")
    print(f"  检测精度 precision   = {result.precision} mm")
    print(f"  置信度  confidence   = {result.confidence:.2f}")
    print(f"  额外信息 extra_info  = {result.extra_info}")

    # ── 4. 保存每步中间图像（OCR 调试图现已内置于流水线中）──
    print("\n========== 中间图像 ==========")
    for k, v in result.debug_images.items():
        if v is None:
            continue
        # 文件名清理
        safe_name = k.replace('/', '_').replace('\\', '_')
        fp = os.path.join(OUT_DIR, f"{safe_name}.png")
        ok, buf = cv2.imencode('.png', v)
        if ok:
            buf.tofile(fp)
            print(f"  saved: {safe_name}.png  shape={v.shape}")
        else:
            print(f"  [SKIP] {safe_name}  encode 失败")

    # 5. 最终标注
    if result.image_annotated is not None:
        cv2.imencode('.png', result.image_annotated)[1].tofile(
            os.path.join(OUT_DIR, "99_最终标注.png"))
        print("  saved: 99_最终标注.png")


if __name__ == "__main__":
    main()
