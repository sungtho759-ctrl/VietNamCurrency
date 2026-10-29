"""Dò tham số nhận diện trên tập valid: số ảnh mẫu train, ngưỡng SIFT và luật kết hợp màu.

SIFT + đối sánh chỉ chạy một lần cho mỗi tờ tiền với toàn bộ ảnh mẫu (gốc + K_MAX ảnh mẫu train mỗi
mệnh giá), chạy song song nhiều tiến trình. Ứng viên và xếp hạng màu được lưu vào
ket_qua/do_tham_so/ung_vien_<split>_v3.json; sau đó thử mọi tổ hợp tham số trên dữ liệu đã lưu.

Quy tắc chọn: accuracy cao nhất trong các bộ tham số có precision
(accuracy trên mẫu có kết quả) >= --min-precision.

Chạy:  python src/tinh_chinh_nguong.py --split valid
"""
import argparse
import csv
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from danh_gia_sift import ROOT, UNKNOWN, candidates, crop_box, fuse, load_split, load_templates
from mau_sac import ColorModel, hsv_hist

# In được tiếng Việt cả khi chạy trong IDE hoặc chuyển hướng đầu ra trên Windows
sys.stdout.reconfigure(encoding="utf-8")

K_MAX = 8
RANSAC = (8.0, 10.0, 12.0)
K_TRAIN = (0, 2, 4, 8)
MIN_INLIERS = (6, 8, 10, 12)
MIN_RATIO = (0.0,)
MAX_AREA = (10.0,)
MARGIN = (1.0, 1.1, 1.3)
COLOR_VETO = (0, 2, 3)
AGREE_MIN = (0, 5)
COLOR_FALLBACK = (0.0, 0.7, 0.8)
KEYS = ("k_train", "ransac", "min_inliers", "min_ratio", "max_area", "margin", "veto", "agree", "fallback")
REFERENCE = {
    "An toàn (chỉ SIFT)": (0, 10.0, 8, 0.0, 10.0, 1.1, 0, 0, 0.0),
    "Cân bằng (chỉ SIFT)": (0, 12.0, 8, 0.0, 10.0, 1.0, 0, 0, 0.0),
}

_worker = {}


def _init_worker():
    _worker["templates"] = load_templates(K_MAX)
    _worker["color"] = ColorModel()


def _process(job):
    path, bbox, ann_id, true = job
    crop = crop_box(cv2.imread(path), bbox)
    cands = candidates(crop, _worker["templates"], RANSAC)
    return {"ann_id": ann_id, "true": true, "color": _worker["color"].rank(hsv_hist(crop)),
            "cands": {str(k): v for k, v in cands.items()}}


def collect(split, limit, workers):
    split_dir, cat, images, anns = load_split(split)
    if limit:
        anns = anns[:limit]
    jobs = [(str(split_dir / images[a["image_id"]]["file_name"]), a["bbox"], a["id"], cat[a["category_id"]])
            for a in anns]
    items = []
    start = time.perf_counter()
    with Pool(workers, initializer=_init_worker) as pool:
        for i, item in enumerate(pool.imap(_process, jobs, chunksize=4), 1):
            items.append(item)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}  ({time.perf_counter() - start:.0f}s)", flush=True)
    return items


def template_rank(name):
    """Thứ hạng ảnh mẫu train (<mệnh giá>_t<r>.jpg); ảnh mẫu gốc luôn được dùng (-1)."""
    stem = name.rsplit(".", 1)[0]
    return int(stem.split("_t")[1]) if "_t" in stem else -1


def evaluate(items, filtered, params):
    k, ransac, min_inliers, min_ratio, max_area, margin, veto, agree, fallback = params
    classes = sorted({it["true"] for it in items}, key=int)
    cand_lists = filtered[(k, ransac)]
    preds = [fuse(c, it["color"], min_inliers, min_ratio, max_area, margin, veto, agree, fallback)[0]
             for it, c in zip(items, cand_lists)]
    trues = [it["true"] for it in items]
    n = len(items)
    correct = sum(t == p for t, p in zip(trues, preds))
    answered = sum(p != UNKNOWN for p in preds)
    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(trues, preds))
        npred = sum(p == c for p in preds)
        ntrue = sum(t == c for t in trues)
        pr = tp / npred if npred else 0
        rc = tp / ntrue if ntrue else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    row = dict(zip(KEYS, params))
    row.update({"accuracy": correct / n, "coverage": answered / n,
                "precision": correct / answered if answered else 0,
                "macro_f1": sum(f1s) / len(f1s), "sai": answered - correct})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="valid", choices=["train", "valid"])
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--lam-moi", action="store_true", help="Bỏ qua tệp ứng viên đã lưu, chạy lại SIFT")
    args = parser.parse_args()

    out_dir = ROOT / "ket_qua" / "do_tham_so"
    out_dir.mkdir(parents=True, exist_ok=True)
    # v3: ứng viên với cả ảnh mẫu train (có tên mẫu) + xếp hạng màu
    cache = out_dir / f"ung_vien_{args.split}_v3.json"

    if cache.exists() and not args.lam_moi and not args.limit:
        items = json.loads(cache.read_text(encoding="utf-8"))
        print(f"Đọc {len(items)} tờ từ {cache.name}")
    else:
        items = collect(args.split, args.limit, args.workers)
        if not args.limit:
            cache.write_text(json.dumps(items), encoding="utf-8")

    filtered = {(k, r): [[c for c in it["cands"][str(r)] if template_rank(c[4]) < k] for it in items]
                for k in K_TRAIN for r in RANSAC}
    grid = list(itertools.product(K_TRAIN, RANSAC, MIN_INLIERS, MIN_RATIO, MAX_AREA, MARGIN,
                                  COLOR_VETO, AGREE_MIN, COLOR_FALLBACK))
    print(f"Thử {len(grid)} bộ tham số...", flush=True)
    results = [evaluate(items, filtered, params) for params in grid]
    with open(out_dir / f"luoi_nguong_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    index = {tuple(r[k] for k in KEYS): r for r in results}
    ok = [r for r in results if r["precision"] >= args.min_precision]
    if not ok:
        raise SystemExit(f"Không bộ tham số nào đạt precision >= {args.min_precision:.0%}; thử giảm --min-precision.")

    def pick(rows):
        # Khi hòa, ưu tiên ít ảnh mẫu (nhanh hơn) và ngưỡng chặt hơn
        return max(rows, key=lambda r: (r["accuracy"], r["precision"], -r["k_train"], r["margin"],
                                        r["min_inliers"], -r["ransac"]))

    best = pick(ok)

    def line(name, r):
        return (f"{name:<22} K={r['k_train']} RANSAC={r['ransac']:<4} inlier>={r['min_inliers']:<3} "
                f"khoảng cách>={r['margin']:<4} phủ quyết màu={r['veto']} đồng thuận>={r['agree']} "
                f"dự phòng màu={r['fallback']:<4}| accuracy {r['accuracy']:.2%}  coverage {r['coverage']:.2%}  "
                f"precision {r['precision']:.2%}  macro-F1 {r['macro_f1']:.2%}  sai {r['sai']}")

    lines = [f"Tập {args.split}: {len(items)} tờ tiền, {len(results)} bộ tham số",
             f"Quy tắc chọn: accuracy cao nhất với precision >= {args.min_precision:.0%}", ""]
    lines += [line(name, index[params]) for name, params in REFERENCE.items()]
    lines += [line("Tốt nhất", best), "", "Tốt nhất theo mức precision yêu cầu:"]
    for level in (0.95, 0.97, 0.98, 0.99):
        reach = [r for r in results if r["precision"] >= level]
        if reach:
            lines.append(line(f">= {level:.0%}", pick(reach)))
    lines += ["", "Tác dụng từng thành phần (tốt nhất ở precision >= 98%):"]
    variants = {
        "Chỉ SIFT, mẫu gốc": lambda r: r["k_train"] == 0 and not (r["veto"] or r["agree"] or r["fallback"]),
        "SIFT + mẫu train": lambda r: r["k_train"] > 0 and not (r["veto"] or r["agree"] or r["fallback"]),
        "SIFT mẫu gốc + màu": lambda r: r["k_train"] == 0,
        "SIFT + mẫu train + màu": lambda r: True,
    }
    for name, cond in variants.items():
        reach = [r for r in results if cond(r) and r["precision"] >= 0.98]
        if reach:
            lines.append(line(name, pick(reach)))
    text = "\n".join(lines)
    print(text)
    (out_dir / f"ket_qua_{args.split}.txt").write_text(text, encoding="utf-8")
    (out_dir / "nguong_tot_nhat.json").write_text(json.dumps({k: best[k] for k in KEYS}, indent=2),
                                                  encoding="utf-8")

    # Đánh đổi precision – coverage
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, color in zip(K_TRAIN, ["#9ecae1", "#4292c6", "#08519c", "#08306b"]):
        pts = [r for r in results if r["k_train"] == k]
        ax.scatter([r["coverage"] for r in pts], [r["precision"] for r in pts], s=8, alpha=0.6,
                   color=color, label=f"{k} ảnh mẫu train / mệnh giá")
    for (name, params), marker in zip(REFERENCE.items(), ["s", "D"]):
        r = index[params]
        ax.scatter(r["coverage"], r["precision"], s=110, marker=marker, color="#d62728", label=name)
    ax.scatter(best["coverage"], best["precision"], s=160, marker="*", color="#2ca02c", label="Được chọn")
    ax.axhline(args.min_precision, color="gray", ls="--", lw=1)
    ax.set_xlabel("Tỷ lệ đưa ra kết quả (coverage)")
    ax.set_ylabel("Accuracy trên mẫu có kết quả")
    ax.set_title(f"Dò tham số trên tập {args.split}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"danh_doi_{args.split}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
