"""Tính lại các chỉ số trên tập test "sạch" (bỏ ảnh test gần trùng valid / cùng ảnh gốc, xem check_leakage.py).

Không chạy lại SIFT hay YOLO: lọc các file dự đoán đã có trong results/ theo danh sách ảnh sạch, rồi tính
lại bằng cùng định nghĩa chỉ số. So sánh ba phần: toàn bộ test, phần sạch và phần rò rỉ – chênh lệch giữa
phần sạch và phần rò rỉ cho thấy mức lạc quan do rò rỉ.

Chạy:  python src/check_leakage.py          # tạo data/test_clean.json (một lần)
       python src/evaluate_clean.py [--nguong-trung 50]
Kết quả: results/clean_test/ (summary.md + báo cáo đầy đủ trên phần sạch cho từng cấu hình)
"""
import argparse
import csv
import json
import sys
from collections import defaultdict

import numpy as np

from evaluate_multi_note import match
from evaluate_sift import ROOT, UNKNOWN, load_split, report

sys.stdout.reconfigure(encoding="utf-8")

LEAK_PATH = ROOT / "data" / "test_clean.json"
OUT_DIR = ROOT / "results" / "clean_test"

# (tên hiển thị, thư mục kết quả trong results/, tên thư mục đầu ra)
CONFIGS = [
    ("Ban đầu (SIFT, RANSAC 5 px)", "comparison/baseline", "baseline"),
    ("Tinh chỉnh ngưỡng + nới diện tích (SIFT)", "comparison/tuned_thresholds", "tuned_thresholds"),
    ("Thêm luật khoảng cách 1,1 (SIFT)", "comparison/margin_rule", "margin_rule"),
    ("Chỉ màu, luôn trả lời", "comparison/color_only", "color_only"),
    ("Chỉ màu, từ chối khi d1/d2 > 0,9", "comparison/color_only_reject", "color_only_reject"),
    ("**SIFT + màu (mặc định)**", "sift_test", "sift_color_default"),
    ("SIFT + 8 ảnh mẫu train + màu", "comparison/train_templates_color", "train_templates_color"),
    ("YOLO11n (vùng cắt, ngưỡng valid)", "yolo/evaluation_test", "yolo"),
    ("YOLO11n (vùng cắt, luôn trả lời)", "yolo/evaluation_test/always_answer", "yolo_always_answer"),
    ("YOLO11n (ảnh đầy đủ, khung nhãn)", "yolo/evaluation_test/note_on_full_image", "yolo_full_image"),
]


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["time"] = float(r["time"])
    return rows


def present(rows):
    """Các mệnh giá có trong tập con (Macro-F1 chỉ tính trên lớp có mặt, tránh lớp vắng bị tính F1 = 0)."""
    return sorted({r["true"] for r in rows}, key=int)


def metrics(rows):
    """Cùng định nghĩa với evaluate_sift.report()."""
    n = len(rows)
    if not n:
        return None
    classes = present(rows)
    ok = sum(r["true"] == r["pred"] for r in rows)
    ans = [r for r in rows if r["pred"] != UNKNOWN]
    f1s = []
    for c in classes:
        tp = sum(r["true"] == c and r["pred"] == c for r in rows)
        n_pred = sum(r["pred"] == c for r in rows)
        n_true = sum(r["true"] == c for r in rows)
        p = tp / n_pred if n_pred else 0.0
        rc = tp / n_true if n_true else 0.0
        f1s.append(2 * p * rc / (p + rc) if p + rc else 0.0)
    return {"n": n, "acc": ok / n, "answer": len(ans) / n,
            "acc_answered": sum(r["true"] == r["pred"] for r in ans) / len(ans) if ans else 0.0,
            "macro_f1": float(np.mean(f1s)), "wrong": sum(r["pred"] != r["true"] for r in ans)}


def multi_metrics(per_image):
    """per_image: [(số tờ thật, số khung, số khung đúng vị trí + mệnh giá, đúng tổng tiền)]."""
    if not per_image:
        return None
    n = sum(x[0] for x in per_image)
    p = sum(x[1] for x in per_image)
    tpc = sum(x[2] for x in per_image)
    r, pr = tpc / n, (tpc / p if p else 0.0)
    return {"imgs": len(per_image), "n": n, "recall": r, "precision": pr,
            "f1": 2 * r * pr / (r + pr) if r + pr else 0.0,
            "money": sum(x[3] for x in per_image) / len(per_image)}


def pct(x):
    return "–" if x is None else f"{x:.2%}".replace(".", ",")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nguong-trung", type=int, default=None,
                        help="Ngưỡng dHash coi là gần trùng valid (mặc định: ngưỡng trong test_clean.json)")
    args = parser.parse_args()
    if not LEAK_PATH.exists():
        sys.exit("Chưa có data/test_clean.json, chạy trước: python src/check_leakage.py")
    info = json.loads(LEAK_PATH.read_text(encoding="utf-8"))
    thr = info["nguong_trung"] if args.nguong_trung is None else args.nguong_trung
    leaked = {f for f, d in info["khoang_cach_toi_valid"].items() if d <= thr} | set(info["cung_anh_goc"])
    clean = set(info["khoang_cach_toi_valid"]) - leaked

    _, cat, images, anns = load_split("test")
    files_with_notes = {images[a["image_id"]]["file_name"] for a in anns}
    n_clean_notes = sum(images[a["image_id"]]["file_name"] in clean for a in anns)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = {"Toàn bộ": lambda f: True, "Sạch": lambda f: f in clean, "Rò rỉ": lambda f: f in leaked}
    lines = [f"# Kết quả trên tập test sạch (ngưỡng dHash {thr} bit)", "",
             f"Ảnh test có tờ tiền: {len(files_with_notes)} – sạch {len(files_with_notes & clean)} ảnh "
             f"({n_clean_notes} tờ), rò rỉ {len(files_with_notes & leaked)} ảnh "
             f"({len(anns) - n_clean_notes} tờ).", ""]

    # 1. Theo tờ tiền và theo ảnh một tờ
    for kind, title in (("banknote", "Theo tờ tiền (vùng cắt theo khung nhãn)"), ("image", "Theo ảnh (ảnh một tờ)")):
        lines += [f"## {title}", "",
                  "| Cấu hình | Accuracy toàn bộ | Accuracy sạch | Accuracy rò rỉ | Tỷ lệ trả lời sạch "
                  "| Acc. trên mẫu có kết quả sạch | Macro-F1 sạch | Sai mệnh giá sạch | Số mẫu sạch |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, src, out in CONFIGS:
            path = ROOT / "results" / src / f"predictions_{kind}.csv"
            if not path.exists():
                continue
            rows = read_csv(path)
            m = {k: metrics([r for r in rows if f(r["file"])]) for k, f in parts.items()}
            c = m["Sạch"]
            if c is None:
                continue
            lines.append(f"| {name} | {pct(m['Toàn bộ']['acc'])} | **{pct(c['acc'])}** | "
                         f"{pct(m['Rò rỉ']['acc'] if m['Rò rỉ'] else None)} | {pct(c['answer'])} | "
                         f"{pct(c['acc_answered'])} | {pct(c['macro_f1'])} | {c['wrong']} | {c['n']} |")
            d = OUT_DIR / out
            d.mkdir(exist_ok=True)
            label = "Banknote (cắt bbox, test sạch)" if kind == "banknote" else "Image (ảnh một tờ, test sạch)"
            sub = [r for r in rows if r["file"] in clean]
            report(label, sub, present(sub), d,
                   "YOLO" if src.startswith("yolo") else "SIFT")
        lines.append("")

    # 2. Tự tìm mọi tờ trong ảnh
    gts = defaultdict(list)
    for a in anns:
        gts[images[a["image_id"]]["file_name"]].append((cat[a["category_id"]], a["bbox"]))
    sift_rows = read_csv_multi(ROOT / "results" / "multi_note_test" / "predictions.csv")
    yolo_path = ROOT / "results" / "yolo" / "evaluation_test" / "box_predictions.json"
    yolo_rows = []
    if yolo_path.exists():
        for f, preds in json.loads(yolo_path.read_text(encoding="utf-8")).items():
            g = gts.get(f, [])
            m = match(preds, g, 0.5)
            ok = sum(preds[i][0] == g[j][0] for i, j in m)
            money = sum(int(p[0]) for p in preds) == sum(int(x[0]) for x in g)
            yolo_rows.append((f, (len(g), len(preds), ok, money)))
    lines += ["## Tự tìm mọi tờ trong ảnh (IoU ≥ 0,5)", "",
              "| Phương pháp | Phần | Số ảnh / tờ | Recall | Precision | F1 | Ảnh đúng tổng tiền |",
              "|---|---|---|---|---|---|---|"]
    for name, rows in (("SIFT + màu", sift_rows), ("YOLO11n", yolo_rows)):
        for part, f in parts.items():
            m = multi_metrics([x for file, x in rows if f(file)])
            if m:
                lines.append(f"| {name} | {part} | {m['imgs']} / {m['n']} | {pct(m['recall'])} | "
                             f"{pct(m['precision'])} | {pct(m['f1'])} | {pct(m['money'])} |")
    lines.append("")

    text = "\n".join(lines)
    (OUT_DIR / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print("Kết quả lưu tại:", OUT_DIR.relative_to(ROOT))


def read_csv_multi(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [(r["file"], (int(r["so_to_that"]), int(r["so_to_du_doan"]), int(r["dung_ca_menh_gia"]),
                             int(r["tong_tien_that"]) == int(r["tong_tien_du_doan"]))) for r in csv.DictReader(f)]


if __name__ == "__main__":
    main()
