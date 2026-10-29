"""Đánh giá nhận diện nhiều tờ trong một ảnh (lặp SIFT + RANSAC, detect_all) trên tập có nhãn.

Mỗi khung dự đoán (khung bao thẳng trục của tứ giác homography, cắt theo ảnh) được ghép với khung bao trong
nhãn theo IoU giảm dần (ghép một–một, IoU >= --iou). Các chỉ số:
  - Vị trí: recall = số tờ được ghép / tổng số tờ; precision = số khung được ghép / tổng số khung dự đoán.
  - Vị trí + mệnh giá: như trên nhưng chỉ tính cặp ghép đúng cả mệnh giá.
  - Theo ảnh: tỷ lệ ảnh đếm đúng số tờ, tỷ lệ ảnh tính đúng tổng số tiền.

Chạy:  python src/danh_gia_nhieu_to.py --split valid --phong-to 1 1.5 2 --workers 10   # chọn mức phóng to
       python src/danh_gia_nhieu_to.py --split test --phong-to 2                       # đánh giá cuối
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

from danh_gia_sift import ROOT, detect_all, load_split, load_templates
from mau_sac import ColorModel

# In được tiếng Việt cả khi chạy trong IDE hoặc chuyển hướng đầu ra trên Windows
sys.stdout.reconfigure(encoding="utf-8")

_w = {}


def _init(parallel=False):
    if parallel:
        cv2.setNumThreads(1)  # nhiều tiến trình cùng chạy: tránh mỗi tiến trình OpenCV lại mở nhiều luồng
    _w["templates"] = load_templates()
    _w["color"] = ColorModel()


def _run(job):
    path, scale = job
    img = cv2.imread(path)
    start = time.perf_counter()
    found = detect_all(img, _w["templates"], _w["color"], upscale=scale)
    return found, time.perf_counter() - start


def bbox_of(quad, W, H):
    q = np.float32(quad)
    x0, y0 = np.clip(q.min(0), 0, [W, H])
    x1, y1 = np.clip(q.max(0), 0, [W, H])
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, thr):
    """Ghép một–một theo IoU giảm dần; trả về danh sách (chỉ số dự đoán, chỉ số nhãn)."""
    pairs = sorted(((iou(p[1], g[1]), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)), reverse=True)
    used_p, used_g, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i), used_g.add(j), out.append((i, j))
    return out


def evaluate(split, scale, workers, thr, out_dir=None, draw=0):
    split_dir, cat, images, anns = load_split(split)
    per = defaultdict(list)
    for a in anns:
        per[a["image_id"]].append(a)
    ids = sorted(per)
    jobs = [(str(split_dir / images[i]["file_name"]), scale) for i in ids]
    if workers > 1:
        results = []
        with Pool(workers, initializer=_init, initargs=(True,)) as pool:
            for i, r in enumerate(pool.imap(_run, jobs, chunksize=2), 1):
                results.append(r)
                if i % 50 == 0:
                    print(f"  ×{scale:g}: {i}/{len(jobs)} ảnh", flush=True)
    else:
        _init()
        results = []
        for i, j in enumerate(jobs, 1):
            results.append(_run(j))
            if i % 50 == 0:
                print(f"  ×{scale:g}: {i}/{len(jobs)} ảnh", flush=True)

    rows, tot = [], defaultdict(float)
    for iid, (found, secs) in zip(ids, results):
        im = images[iid]
        W, H = im["width"], im["height"]
        gts = [(cat[a["category_id"]], a["bbox"]) for a in per[iid]]
        preds = [(lab, bbox_of(q, W, H), q) for lab, _, q in found]
        m = match(preds, gts, thr)
        tp = len(m)
        tpc = sum(preds[i][0] == gts[j][0] for i, j in m)
        sum_gt = sum(int(g[0]) for g in gts)
        sum_pred = sum(int(p[0]) for p in preds)
        rows.append({"file": im["file_name"], "so_to_that": len(gts), "so_to_du_doan": len(preds),
                     "dung_vi_tri": tp, "dung_ca_menh_gia": tpc, "tong_tien_that": sum_gt,
                     "tong_tien_du_doan": sum_pred, "thoi_gian": round(secs, 3)})
        key = "1 tờ" if len(gts) == 1 else "nhiều tờ"
        for k in ("tất cả", key):
            tot[k, "n"] += len(gts); tot[k, "p"] += len(preds); tot[k, "tp"] += tp; tot[k, "tpc"] += tpc
            tot[k, "img"] += 1; tot[k, "dem"] += len(preds) == len(gts); tot[k, "tien"] += sum_pred == sum_gt
            tot[k, "t"] += secs
        if out_dir and draw and len(gts) > 1 and draw > sum(1 for _ in (out_dir / "minh_hoa").glob("*.jpg")):
            img = cv2.imread(str(split_dir / im["file_name"]))
            for g in gts:
                x, y, w, h = map(int, g[1])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 160, 0), 1)
            ok_pred = {i for i, j in m if preds[i][0] == gts[j][0]}
            for i, (lab, _, q) in enumerate(preds):
                color = (0, 200, 0) if i in ok_pred else (0, 0, 255)
                cv2.polylines(img, [np.int32(q)], True, color, 2)
                x, y = np.int32(np.float32(q).min(0))
                cv2.putText(img, lab, (max(x, 0), max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imwrite(str(out_dir / "minh_hoa" / im["file_name"]), img)

    def line(k):
        n, p, tp, tpc = tot[k, "n"], tot[k, "p"], tot[k, "tp"], tot[k, "tpc"]
        r, pr = tpc / n, (tpc / p if p else 0)
        f1 = 2 * r * pr / (r + pr) if r + pr else 0
        return (f"{k:<9} {int(tot[k, 'img']):>4} ảnh, {int(n):>4} tờ, {int(p):>4} khung | vị trí: recall "
                f"{tp / n:.2%} precision {(tp / p if p else 0):.2%} | vị trí + mệnh giá: recall {r:.2%} "
                f"precision {pr:.2%} F1 {f1:.2%} | đếm đúng số tờ {tot[k, 'dem'] / tot[k, 'img']:.2%}, "
                f"đúng tổng tiền {tot[k, 'tien'] / tot[k, 'img']:.2%} | {1000 * tot[k, 't'] / tot[k, 'img']:.0f} ms/ảnh")

    text = "\n".join([f"Tập {split}, phóng to ×{scale:g}, IoU >= {thr}"] + [line(k) for k in ("tất cả", "1 tờ", "nhiều tờ")])
    return text, rows, tot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--phong-to", type=float, nargs="+", default=[2.0])
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=1, help=">1: chạy song song (thời gian đo kém chính xác)")
    parser.add_argument("--minh-hoa", type=int, default=12, help="Số ảnh nhiều tờ được vẽ kết quả")
    args = parser.parse_args()

    out_dir = ROOT / "ket_qua" / f"nhieu_to_{args.split}"
    (out_dir / "minh_hoa").mkdir(parents=True, exist_ok=True)
    reports = []
    for scale in args.phong_to:
        last = scale == args.phong_to[-1]
        text, rows, _ = evaluate(args.split, scale, args.workers, args.iou, out_dir if last else None,
                                 args.minh_hoa if last else 0)
        print(text + "\n", flush=True)
        reports.append(text)
        if last:
            with open(out_dir / "du_doan.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    (out_dir / "bao_cao.txt").write_text("\n\n".join(reports), encoding="utf-8")
    print("Kết quả lưu tại:", out_dir)


if __name__ == "__main__":
    main()
