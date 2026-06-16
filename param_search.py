"""
参数搜索脚本：对 `caliper.preprocess.preprocess` 在默认参数范围做网格搜索，
计算拉普拉斯方差与 Tenengrad 指标，保存结果 CSV 与 top N 可视化图。

用法示例：
    python param_search.py --img path/to/test.jpg --out results --top 3
"""
import os
import cv2
import argparse
import numpy as np
import csv
from pathlib import Path

from caliper import config
from caliper.preprocess import preprocess


def variance_of_laplacian(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


def tenengrad(img_gray):
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    fm = gx**2 + gy**2
    return float(np.mean(fm))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--img', required=True)
    p.add_argument('--out', default='param_search_results')
    p.add_argument('--top', type=int, default=3)
    args = p.parse_args()

    img_path = Path(args.img)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 使用二进制读取再解码以兼容包含非 ASCII 路径（例如中文路径）
    data = img_path.read_bytes()
    img_arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError('Failed to read image: ' + str(img_path))

    # 默认搜索范围（已确认）
    gammas = [0.6, 0.8, 1.0, 1.2]
    bilateral_ds = [5, 9, 15]
    bilateral_sigmas = [25.0, 50.0, 75.0]
    median_ksizes = [0, 3, 5]
    clahe_clips = [2.0, 4.0]
    unsharp_amounts = [0.5, 1.0, 1.5]

    results = []

    total = (len(gammas) * len(bilateral_ds) * len(bilateral_sigmas) *
             len(median_ksizes) * len(clahe_clips) * len(unsharp_amounts))
    idx = 0
    print(f'Starting grid search: {total} combinations')

    for gamma in gammas:
        for bd in bilateral_ds:
            for bs in bilateral_sigmas:
                for mk in median_ksizes:
                    for cc in clahe_clips:
                        for ua in unsharp_amounts:
                            idx += 1
                            print(f'[{idx}/{total}] g={gamma},bd={bd},bs={bs},mk={mk},cc={cc},ua={ua}')

                            # 修改全局 config 的可影响项
                            config.preprocess.clahe_clip_limit = cc
                            config.preprocess.unsharp_amount = ua
                            config.preprocess.unsharp_blur_sigma = config.preprocess.unsharp_blur_sigma

                            out = preprocess(img, clip_limit=cc, bilateral_d=bd,
                                             bilateral_sigma=bs, gamma=gamma,
                                             median_ksize=mk)

                            enhanced = out['enhanced']
                            binary = out['binary_adaptive']
                            debug_vis = out['debug_vis']

                            lap = variance_of_laplacian(enhanced)
                            ten = tenengrad(enhanced)

                            results.append({
                                'gamma': gamma,
                                'bilateral_d': bd,
                                'bilateral_sigma': bs,
                                'median_ksize': mk,
                                'clahe_clip': cc,
                                'unsharp_amount': ua,
                                'lap_var': float(lap),
                                'tenengrad': float(ten),
                                'debug_vis': debug_vis,
                            })

    # 归一化并合成得分
    laps = np.array([r['lap_var'] for r in results], dtype=np.float64)
    tens = np.array([r['tenengrad'] for r in results], dtype=np.float64)
    # 防止常数数组导致除以零
    lap_range = laps.max() - laps.min()
    ten_range = tens.max() - tens.min()
    laps_n = (laps - laps.min()) / (lap_range if lap_range != 0 else 1.0)
    tens_n = (tens - tens.min()) / (ten_range if ten_range != 0 else 1.0)
    scores = laps_n + tens_n  # 简单相加

    for i, r in enumerate(results):
        r['score'] = float(scores[i])

    # 将结果按 score 排序（降序）
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)

    # 保存 CSV
    csv_path = out_dir / 'results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['gamma','bilateral_d','bilateral_sigma','median_ksize',
                         'clahe_clip','unsharp_amount','lap_var','tenengrad','score'])
        for r in results_sorted:
            writer.writerow([r['gamma'], r['bilateral_d'], r['bilateral_sigma'], r['median_ksize'],
                             r['clahe_clip'], r['unsharp_amount'], r['lap_var'], r['tenengrad'], r['score']])

    # 保存 top N 的可视化图
    top_n = min(args.top, len(results_sorted))
    for i in range(top_n):
        r = results_sorted[i]
        vis = r['debug_vis']
        out_p = out_dir / f'top_{i+1}_score_{r["score"]:.4f}.png'
        cv2.imwrite(str(out_p), vis)

    print('Done. Results saved to', out_dir)


if __name__ == '__main__':
    main()
