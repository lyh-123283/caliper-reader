import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_existing(path: Path):
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict):
        return data.get('boxes', [])
    if isinstance(data, list):
        return data
    return []


def save_boxes(path: Path, boxes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(boxes, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )


def fit_scale(w: int, h: int, max_w: int, max_h: int) -> float:
    return min(1.0, max_w / max(w, 1), max_h / max(h, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='image to label')
    parser.add_argument('--out', default='templates/digit_boxes.json')
    parser.add_argument('--max-width', type=int, default=1600)
    parser.add_argument('--max-height', type=int, default=900)
    args = parser.parse_args()

    image_path = Path(args.image)
    out_path = Path(args.out)
    img = imread_unicode(image_path)
    if img is None:
        raise SystemExit(f'cannot read image: {image_path}')

    h, w = img.shape[:2]
    scale = fit_scale(w, h, args.max_width, args.max_height)
    disp_w = max(1, int(round(w * scale)))
    disp_h = max(1, int(round(h * scale)))
    base = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    boxes = load_existing(out_path)
    image_key = str(image_path)

    drawing = False
    start = (0, 0)
    current = None

    def to_display_box(box):
        x1, y1, x2, y2 = box
        return (
            int(round(x1 * scale)),
            int(round(y1 * scale)),
            int(round(x2 * scale)),
            int(round(y2 * scale)),
        )

    def to_original_box(box):
        x1, y1, x2, y2 = box
        ox1 = int(round(min(x1, x2) / scale))
        oy1 = int(round(min(y1, y2) / scale))
        ox2 = int(round(max(x1, x2) / scale))
        oy2 = int(round(max(y1, y2) / scale))
        ox1 = max(0, min(w - 1, ox1))
        oy1 = max(0, min(h - 1, oy1))
        ox2 = max(0, min(w, ox2))
        oy2 = max(0, min(h, oy2))
        return [ox1, oy1, ox2, oy2]

    def redraw():
        canvas = base.copy()
        for item in boxes:
            if Path(item.get('image', '')) != image_path:
                continue
            x1, y1, x2, y2 = to_display_box(item['bbox'])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(canvas, str(item.get('digit', '?')), (x1, max(18, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
        if current is not None:
            x1, y1, x2, y2 = current
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 80, 255), 2)
        cv2.putText(canvas, 'Drag box, press 0-9 to label | u undo | s save | q quit',
                    (10, disp_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('digit labeler', canvas)

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, start, current
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start = (x, y)
            current = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current = (start[0], start[1], x, y)
            redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            current = (start[0], start[1], x, y)
            redraw()

    cv2.namedWindow('digit labeler', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('digit labeler', disp_w, disp_h)
    cv2.setMouseCallback('digit labeler', on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue
        if ord('0') <= key <= ord('9') and current is not None:
            bbox = to_original_box(current)
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                digit = chr(key)
                boxes.append({
                    'image': image_key,
                    'digit': digit,
                    'bbox': bbox,
                })
                print(f'add digit {digit}: {bbox}')
                current = None
                redraw()
        elif key == ord('u'):
            for i in range(len(boxes) - 1, -1, -1):
                if Path(boxes[i].get('image', '')) == image_path:
                    removed = boxes.pop(i)
                    print('undo:', removed)
                    break
            redraw()
        elif key == ord('s'):
            save_boxes(out_path, boxes)
            print('saved:', out_path)
        elif key == ord('q') or key == 27:
            save_boxes(out_path, boxes)
            print('saved:', out_path)
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
