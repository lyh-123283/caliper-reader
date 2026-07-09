from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


TEMPLATE_SIZE = (32, 48)


def normalize_digit_image(img: np.ndarray,
                          size: Tuple[int, int] = TEMPLATE_SIZE) -> Optional[np.ndarray]:
    if img is None or img.size == 0:
        return None
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(bw < 128) < np.sum(bw >= 128):
        bw = cv2.bitwise_not(bw)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_area = int(np.max(areas))
        cleaned = np.zeros_like(bw)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            ww = int(stats[label, cv2.CC_STAT_WIDTH])
            hh = int(stats[label, cv2.CC_STAT_HEIGHT])
            aspect = hh / max(ww, 1)
            if area >= max(8, best_area * 0.18) and aspect <= 5.5:
                cleaned[labels == label] = 255
        if np.any(cleaned):
            bw = cleaned

    ys, xs = np.where(bw > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = bw[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    if h < 3 or w < 2:
        return None

    target_w, target_h = size
    scale = min((target_w - 6) / max(w, 1), (target_h - 6) / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    ox = (target_w - new_w) // 2
    oy = (target_h - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return (canvas > 127).astype(np.uint8) * 255


def _is_prebuilt_template(img: np.ndarray) -> bool:
    if img is None or img.shape != (TEMPLATE_SIZE[1], TEMPLATE_SIZE[0]):
        return False
    vals = np.unique(img)
    return vals.size <= 3 and np.all((vals < 8) | (vals > 247))


def _template_features(norm: np.ndarray) -> dict:
    binary = (norm > 127).astype(np.uint8) * 255
    vec = binary.astype(np.float32).reshape(-1)
    vec -= float(vec.mean())
    vec_norm = float(np.linalg.norm(vec))
    return {
        'image': binary,
        'fg': binary > 0,
        'vec': vec,
        'vec_norm': vec_norm,
    }


class TemplateDigitRecognizer:
    def __init__(self, template_dir: Path = None):
        root = Path(__file__).resolve().parent.parent
        self.template_dir = Path(template_dir) if template_dir else root / 'templates' / 'digits'
        self.templates = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        self._loaded = True
        self.templates = {}
        if not self.template_dir.exists():
            return
        for digit_dir in self.template_dir.iterdir():
            if not digit_dir.is_dir() or not digit_dir.name.isdigit():
                continue
            digit = int(digit_dir.name)
            if digit < 0 or digit > 9:
                continue
            for path in sorted(digit_dir.glob('*.png')):
                img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                norm = (img > 127).astype(np.uint8) * 255 if _is_prebuilt_template(img) else normalize_digit_image(img)
                if norm is not None:
                    self.templates.setdefault(digit, []).append(_template_features(norm))

    def available(self) -> bool:
        self.load()
        return any(self.templates.values())

    def recognize(self, patch: np.ndarray) -> Optional[Tuple[str, float]]:
        self.load()
        if not self.templates:
            return None
        norm = normalize_digit_image(patch)
        if norm is None:
            return None

        best_digit = None
        best_score = -1.0
        fg = norm > 0
        vec = norm.astype(np.float32).reshape(-1)
        vec -= float(vec.mean())
        vec_norm = float(np.linalg.norm(vec))
        for digit, templates in self.templates.items():
            for templ in templates:
                tg = templ['fg']
                inter = np.count_nonzero(fg & tg)
                union = np.count_nonzero(fg | tg)
                iou = inter / union if union else 0.0

                denom = vec_norm * templ['vec_norm']
                corr = float(np.dot(vec, templ['vec']) / denom) if denom > 1e-6 else 0.0
                score = 0.65 * iou + 0.35 * max(0.0, corr)
                if score > best_score:
                    best_score = score
                    best_digit = digit

        if best_digit is None or best_score < 0.38:
            return None
        return str(best_digit), float(best_score)
