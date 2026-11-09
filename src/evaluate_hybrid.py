"""Đánh giá chế độ kết hợp (hybrid) của realtime.py: YOLO định vị từng tờ, SIFT + màu nhận diện mệnh giá.

So sánh trên cùng khung YOLO (ngưỡng 0,45, NMS không phân lớp – như realtime.py) ba cách gán nhãn:
  - yolo          : nhãn của YOLO.
  - hybrid        : nhãn SIFT + màu; nếu SIFT + màu không kết luận thì giữ nhãn YOLO (realtime.py).
  - hybrid_strict : nhãn SIFT + màu; nếu không kết luận thì bỏ khung đó (không hiển thị).
Chỉ số giống evaluate_multi_note.py (khung đúng khi IoU >= 0,5 và đúng mệnh giá) + F1 từng mệnh giá trên các
khung đúng vị trí.

Chạy:  python src/evaluate_hybrid.py --split valid test --workers 10
Kết quả: results/hybrid_<split>/report.txt và predictions.csv
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import cv2

from evaluate_multi_note import match
from evaluate_sift import (MIN_MARGIN, RANSAC_THRESH, ROOT, UNKNOWN, candidates, crop_box, fuse, load_split,
                           load_templates, resize_max)
from color_model import ColorModel, hsv_hist

sys.stdout.reconfigure(encoding="utf-8")

YOLO_CONF, YOLO_AGNOSTIC = 0.45, True
# --so-sanh-luat: các luật khoảng cách thử trên cùng ứng viên SIFT (tên, exclusive, min_margin)
MARGIN_RULES = [("tổng inlier >= 1,3x (mặc định)", False, MIN_MARGIN)] + [
    (f"inlier riêng >= {m:g}x".replace(".", ","), True, m) for m in (1.3, 1.5, 2.0, 3.0)]
_w = {}


def _init(crop_max=0):
    cv2.setNumThreads(1)
    _w["crop_max"] = crop_max
    _w["t"] = load_templates()
    _w["c"] = ColorModel()


def _candidates(job):
    """Ứng viên SIFT + xếp hạng màu của một vùng cắt (quyết định nhãn ở tiến trình chính, để thử được nhiều luật)."""
    path, box = job
    crop = crop_box(cv2.imread(path), box)
    if _w["crop_max"]:
        crop = resize_max(crop, _w["crop_max"])
    start = time.perf_counter()
    cands = candidates(crop, _w["t"], (RANSAC_THRESH,))[RANSAC_THRESH]
    rank = _w["c"].rank(hsv_hist(crop))
    return cands, rank, time.perf_counter() - start


def yolo_boxes(split):
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(8)
    model = YOLO(str(ROOT / "results" / "yolo" / "yolo11n" / "weights" / "best.pt"))
    split_dir, cat, images, anns = load_split(split)
    per = defaultdict(list)
    for a in anns:
        per[a["image_id"]].append((cat[a["category_id"]], a["bbox"]))
    out = []
    for iid in sorted(per):
        path = str(split_dir / images[iid]["file_name"])
        r = model.predict(path, conf=YOLO_CONF, imgsz=640, agnostic_nms=YOLO_AGNOSTIC, device="cpu", verbose=False)[0]
        boxes = [(model.names[int(c)], [b[0], b[1], b[2] - b[0], b[3] - b[1]], s)
                 for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        out.append({"file": images[iid]["file_name"], "path": path, "gts": per[iid], "boxes": boxes})
    return out


def metrics(items, key, classes):
    n = p = tpc = money = 0
    per_cls = {c: [0, 0, 0] for c in classes}  # tp, số dự đoán, số thật (trên khung đúng vị trí)
    for it in items:
        preds = [(lab, box) for lab, box in zip(it[key], (b[1] for b in it["boxes"])) if lab is not None]
        gts = it["gts"]
        m = match(preds, gts, 0.5)
        ok = sum(preds[i][0] == gts[j][0] for i, j in m)
        n += len(gts); p += len(preds); tpc += ok
        money += sum(int(x[0]) for x in preds) == sum(int(g[0]) for g in gts)
        for i, j in m:
            t, q = gts[j][0], preds[i][0]
            per_cls[t][2] += 1
            if q in per_cls:
                per_cls[q][1] += 1
            if q == t:
                per_cls[t][0] += 1
    r, pr = tpc / n, (tpc / p if p else 0)
    f1c = {}
    for c, (tp, np_, nt) in per_cls.items():
        a, b = (tp / np_ if np_ else 0), (tp / nt if nt else 0)
        f1c[c] = 2 * a * b / (a + b) if a + b else 0
    return {"recall": r, "precision": pr, "f1": 2 * r * pr / (r + pr) if r + pr else 0,
            "money": money / len(items), "f1_class": f1c}


def compare_rules(split, raw, items, classes, out):
    """So sánh các luật khoảng cách (MARGIN_RULES) trên cùng ứng viên SIFT + màu -> report_margin.txt."""
    lines = [f"Tập {split}: so sánh luật khoảng cách trên {len(raw)} khung YOLO (cùng ứng viên SIFT + màu)",
             'hybrid chặt = không kết luận thì bỏ khung (realtime.py hiện "?"); hybrid = giữ nhãn YOLO', "",
             f"{'Luật':<32} {'Không KL':>8} {'chặt: P':>9} {'R':>7} {'F1':>7} {'hybrid F1':>10}   "
             + " ".join(f"{c:>7}" for c in classes)]
    for name, excl, margin in MARGIN_RULES:
        labs = [fuse(c, r, min_margin=margin, exclusive=excl)[0] for c, r, _ in raw]
        k = 0
        for it in items:
            sift = labs[k:k + len(it["boxes"])]
            k += len(it["boxes"])
            it["r_strict"] = [x if x != UNKNOWN else None for x in sift]
            it["r_hybrid"] = [x if x != UNKNOWN else y for x, y in zip(sift, it["yolo"])]
        ms, mh = metrics(items, "r_strict", classes), metrics(items, "r_hybrid", classes)
        lines.append(f"{name:<32} {sum(x == UNKNOWN for x in labs):>8} {ms['precision']:>9.2%} "
                     f"{ms['recall']:>7.2%} {ms['f1']:>7.2%} {mh['f1']:>10.2%}   "
                     + " ".join(f"{ms['f1_class'][c]:>7.1%}" for c in classes))
    lines.append("(các cột mệnh giá: F1 của hybrid chặt theo mệnh giá, trên khung đúng vị trí)")
    text = "\n".join(lines)
    print(text + "\n", flush=True)
    (out / "report_margin.txt").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", nargs="+", default=["valid", "test"])
    parser.add_argument("--crop-max", type=int, default=0,
                        help="Thu vùng cắt về cạnh dài tối đa N px trước SIFT (0 = giữ nguyên, như realtime.py cũ)")
    parser.add_argument("--so-sanh-luat", action="store_true",
                        help="So sánh các luật khoảng cách (MARGIN_RULES) trên cùng ứng viên SIFT -> report_margin.txt")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    classes = ["1000", "2000", "5000", "10000", "20000", "50000", "100000", "200000", "500000"]

    for split in args.split:
        t0 = time.perf_counter()
        items = yolo_boxes(split)
        jobs = [(it["path"], b[1]) for it in items for b in it["boxes"]]
        print(f"{split}: {len(items)} ảnh, {len(jobs)} khung YOLO ({time.perf_counter() - t0:.0f}s); chạy SIFT + màu...",
              flush=True)
        with Pool(args.workers, initializer=_init, initargs=(args.crop_max,)) as pool:
            raw = pool.map(_candidates, jobs, chunksize=4)
        res = [fuse(c, r) + (t,) for c, r, t in raw]  # luật mặc định, giống classify()
        k = 0
        for it in items:
            it["yolo"] = [b[0] for b in it["boxes"]]
            it["sift"] = [r[0] for r in res[k:k + len(it["boxes"])]]
            k += len(it["boxes"])
            it["hybrid"] = [s if s != UNKNOWN else y for s, y in zip(it["sift"], it["yolo"])]
            it["hybrid_strict"] = [s if s != UNKNOWN else None for s in it["sift"]]

        lines = [(f"Vùng cắt thu về cạnh dài tối đa {args.crop_max} px. " if args.crop_max else "") + f"Tập {split}: {len(items)} ảnh, {sum(len(i['gts']) for i in items)} tờ, {len(jobs)} khung YOLO "
                 f"(ngưỡng {YOLO_CONF}, NMS không phân lớp). SIFT + màu trung bình "
                 f"{1000 * sum(r[2] for r in res) / max(len(res), 1):.0f} ms / khung (đo khi chạy song song nhiều tiến trình, nên chậm hơn so với khi chạy một tiến trình).",
                 f"SIFT + màu không kết luận: {sum(r[0] == UNKNOWN for r in res)}/{len(res)} khung", "",
                 f"{'Cách gán nhãn':<15} {'Recall':>8} {'Precision':>10} {'F1':>8} {'Đúng tổng tiền':>15}   "
                 + " ".join(f"{c:>7}" for c in classes)]
        for key in ("yolo", "hybrid", "hybrid_strict"):
            m = metrics(items, key, classes)
            lines.append(f"{key:<15} {m['recall']:>8.2%} {m['precision']:>10.2%} {m['f1']:>8.2%} {m['money']:>15.2%}   "
                         + " ".join(f"{m['f1_class'][c]:>7.1%}" for c in classes))
        lines.append("(các cột mệnh giá: F1 nhận diện mệnh giá trên khung đúng vị trí)")
        text = "\n".join(lines)
        print(text + "\n", flush=True)
        out = ROOT / "results" / (f"hybrid_{split}" + (f"_crop{args.crop_max}" if args.crop_max else ""))
        out.mkdir(parents=True, exist_ok=True)
        if args.so_sanh_luat:
            compare_rules(split, raw, items, classes, out)
        (out / "report.txt").write_text(text, encoding="utf-8")
        with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "box", "yolo", "yolo_conf", "sift_mau"])
            for it in items:
                for b, s in zip(it["boxes"], it["sift"]):
                    w.writerow([it["file"], json.dumps([round(v, 1) for v in b[1]]), b[0], round(b[2], 3), s])


if __name__ == "__main__":
    main()
