import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caliper.template_ocr import normalize_digit_image


def imread_unicode(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or '.png', img)
    if ok:
        buf.tofile(str(path))


def load_boxes(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict):
        data = data.get('boxes', [])
    if not isinstance(data, list):
        raise ValueError('template box file must be a list or {"boxes": [...]}')
    return data


def refine_digit_crop(img, bbox):
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w_img - 1, x1))
    x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img - 1, y1))
    y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    H, W = bw.shape[:2]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num_labels < 2:
        return crop, (x1, y1, x2, y2)

    cx0 = W / 2.0
    cy0 = H / 2.0
    valid = []
    crop_area = max(1, W * H)
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(8, int(crop_area * 0.006)):
            continue
        if area > crop_area * 0.75 or w < 2 or h < 5:
            continue
        aspect = h / max(w, 1)
        if aspect < 0.6 or aspect > 5.0:
            continue
        ccx = x + w / 2.0
        ccy = y + h / 2.0
        center_dist = abs(ccx - cx0) / max(W, 1) + abs(ccy - cy0) / max(H, 1)
        score = area * (1.0 - min(0.85, center_dist))
        valid.append((score, x, y, w, h, area))

    if not valid:
        return crop, (x1, y1, x2, y2)

    valid.sort(reverse=True)
    _, bx, by, bw_c, bh_c, best_area = valid[0]
    selected = []
    best_cx = bx + bw_c / 2.0
    best_cy = by + bh_c / 2.0
    for _, x, y, w, h, area in valid:
        if area < best_area * 0.18:
            continue
        ccx = x + w / 2.0
        ccy = y + h / 2.0
        vertical_overlap = max(0, min(by + bh_c, y + h) - max(by, y))
        overlap_ratio = vertical_overlap / max(1, min(bh_c, h))
        close_x = abs(ccx - best_cx) <= max(W * 0.28, bw_c * 1.4)
        close_y = abs(ccy - best_cy) <= max(H * 0.25, bh_c * 0.8)
        if (close_x and close_y) or overlap_ratio > 0.35:
            selected.append((x, y, w, h))

    sx1 = min(x for x, y, w, h in selected)
    sy1 = min(y for x, y, w, h in selected)
    sx2 = max(x + w for x, y, w, h in selected)
    sy2 = max(y + h for x, y, w, h in selected)
    pad = max(2, min(W, H) // 12)
    sx1 = max(0, sx1 - pad)
    sy1 = max(0, sy1 - pad)
    sx2 = min(W, sx2 + pad)
    sy2 = min(H, sy2 + pad)
    refined = crop[sy1:sy2, sx1:sx2]
    return refined, (x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('boxes_json', help='JSON with image/digit/bbox entries')
    parser.add_argument('--out', default='templates/digits')
    args = parser.parse_args()

    boxes_path = Path(args.boxes_json)
    out_dir = Path(args.out)
    boxes = load_boxes(boxes_path)
    image_cache = {}
    counters = {}

    for item in boxes:
        image_path = Path(item['image'])
        if not image_path.is_absolute():
            image_path = boxes_path.parent / image_path
            if not image_path.exists():
                image_path = Path(item['image'])
        digit = str(item['digit'])
        if digit not in {str(i) for i in range(10)}:
            raise ValueError(f'invalid digit label: {digit}')
        x1, y1, x2, y2 = [int(round(v)) for v in item['bbox']]
        key = str(image_path.resolve())
        if key not in image_cache:
            image_cache[key] = imread_unicode(image_path)
        img = image_cache[key]
        if img is None:
            raise ValueError(f'cannot read image: {image_path}')
        h, w = img.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop, refined_bbox = refine_digit_crop(img, (x1, y1, x2, y2))
        norm = normalize_digit_image(crop)
        if norm is None:
            continue
        counters[digit] = counters.get(digit, 0) + 1
        name = item.get('name') or f"{image_path.stem}_{counters[digit]:02d}.png"
        imwrite_unicode(out_dir / digit / name, norm)
        print(f"{digit}: rough={(x1, y1, x2, y2)} refined={refined_bbox} -> {out_dir / digit / name}")

    print('templates written to', out_dir)
    print('counts:', counters)


if __name__ == '__main__':
    main()
