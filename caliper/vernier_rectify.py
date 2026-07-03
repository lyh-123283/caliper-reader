"""Crop the vernier body after the main/vernier split."""

import cv2
import numpy as np


def rectify_vernier_region(region: dict, color_region: np.ndarray = None) -> dict:
    """Return the vernier body crop used by the tick recognizer.

    The current recognizer works directly on a narrow tick band, so this step
    only removes unrelated left/right background while preserving geometry.
    """
    gray = region.get('image')
    if gray is None or gray.size == 0:
        return {
            'region': region,
            'color': color_region,
            'angle': 0.0,
            'matrix': None,
            'inverse_matrix': None,
            'vis': _empty_vis(),
        }

    color_gray = cv2.cvtColor(color_region, cv2.COLOR_BGR2GRAY) if color_region is not None else None
    body_source = color_gray if color_gray is not None else gray
    body_x1, body_x2 = _find_vernier_body_x_range(body_source)

    cropped_gray = body_source[:, body_x1:body_x2]
    cropped_color = color_region[:, body_x1:body_x2] if color_region is not None else None
    binary = region.get('binary')
    cropped_binary = binary[:, body_x1:body_x2] if binary is not None else None

    cropped_region = dict(region)
    cropped_region['image'] = cropped_gray
    cropped_region['binary'] = cropped_binary
    cropped_region['height'] = cropped_gray.shape[0]
    cropped_region['x_offset'] = region.get('x_offset', 0) + body_x1
    cropped_region['is_body_crop'] = True
    cropped_region['body_x_range'] = (body_x1, body_x2)

    vis = _make_crop_vis(body_source, cropped_gray, color_region, cropped_color, body_x1, body_x2)
    return {
        'region': cropped_region,
        'color': cropped_color,
        'angle': 0.0,
        'matrix': None,
        'inverse_matrix': None,
        'vis': vis,
    }


def _find_vernier_body_x_range(img: np.ndarray) -> tuple:
    """Find the bright vernier body strip within the lower split region."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return 0, w

    y2 = max(12, min(h, int(h * 0.28)))
    band = img[:y2, :]
    col_mean = np.mean(band, axis=0).astype(float)
    win = max(15, w // 120)
    if win % 2 == 0:
        win += 1
    smooth = np.convolve(col_mean, np.ones(win, dtype=float) / win, mode='same')
    grad = np.gradient(smooth)

    left_lo, left_hi = int(w * 0.18), int(w * 0.45)
    if left_hi <= left_lo:
        return int(w * 0.22), int(w * 0.72)

    left_slice = grad[left_lo:left_hi]
    if left_slice.size == 0:
        return int(w * 0.30), int(w * 0.70)
    left_x = left_lo + int(np.argmax(left_slice))
    if grad[left_x] < 1.2:
        left_x = int(w * 0.24)

    right_lo, right_hi = int(w * 0.60), int(w * 0.80)
    right_x = int(w * 0.69)
    if right_hi > right_lo:
        right_slice = grad[right_lo:right_hi]
        if right_slice.size:
            candidate = right_lo + int(np.argmin(right_slice))
            if grad[candidate] < -1.0:
                right_x = candidate

    min_width = int(w * 0.24)
    max_width = int(w * 0.50)
    if right_x - left_x < min_width:
        right_x = left_x + min_width
    if right_x - left_x > max_width:
        right_x = left_x + max_width

    pad = max(6, int(w * 0.006))
    return max(0, left_x - pad), min(w, right_x + pad)


def _make_crop_vis(src_gray, cropped_gray, src_color, cropped_color, x1: int, x2: int) -> np.ndarray:
    left = src_color.copy() if src_color is not None else cv2.cvtColor(src_gray, cv2.COLOR_GRAY2BGR)
    right = cropped_color.copy() if cropped_color is not None else cv2.cvtColor(cropped_gray, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(left, (x1, 0), (max(x1, x2 - 1), left.shape[0] - 1), (0, 220, 255), 2)

    h = max(left.shape[0], right.shape[0])
    w_left = left.shape[1]
    w_right = right.shape[1]
    gap = 6
    label_h = 24
    vis = np.full((h + label_h, w_left + gap + w_right, 3), 30, dtype=np.uint8)
    vis[:left.shape[0], :w_left] = left
    vis[:right.shape[0], w_left + gap:w_left + gap + w_right] = right
    cv2.putText(vis, "Vernier body crop", (6, h + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
    cv2.putText(vis, f"x={x1}:{x2}", (w_left + gap + 6, h + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
    return vis


def _empty_vis() -> np.ndarray:
    img = np.zeros((80, 240, 3), dtype=np.uint8)
    cv2.putText(img, "No vernier image", (12, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    return img
