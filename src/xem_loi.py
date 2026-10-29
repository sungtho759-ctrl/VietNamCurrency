"""Xuất ảnh các tờ tiền bị nhận diện lỗi (không xác định hoặc sai mệnh giá).

Đọc du_doan_to_tien.csv của một lần đánh giá, cắt từng tờ lỗi từ ảnh gốc và ghép thành
bảng ảnh theo mệnh giá. Kết quả lưu vào ket_qua/<thư mục>/loi/.

Chạy:  python src/xem_loi.py --ket-qua sift_test
"""
import argparse
import csv
import sys
from collections import defaultdict

import cv2
import numpy as np

from danh_gia_sift import ROOT, UNKNOWN, load_split

# In được tiếng Việt cả khi chạy trong IDE hoặc chuyển hướng đầu ra trên Windows
sys.stdout.reconfigure(encoding="utf-8")

TILE_W, TILE_H = 260, 170
COLS = 6


def tile(img, bbox, lines, color):
    """Vùng tờ tiền (đã xoay nằm ngang) kèm khung ngữ cảnh và chú thích."""
    x, y, w, h = map(int, bbox)
    pad = int(0.15 * max(w, h))
    H, W = img.shape[:2]
    crop = img[max(y - pad, 0):min(y + h + pad, H), max(x - pad, 0):min(x + w + pad, W)].copy()
    cv2.rectangle(crop, (x - max(x - pad, 0), y - max(y - pad, 0)),
                  (x - max(x - pad, 0) + w, y - max(y - pad, 0) + h), color, 2)
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    scale = min(TILE_W / crop.shape[1], (TILE_H - 40) / crop.shape[0])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out = np.full((TILE_H, TILE_W, 3), 255, np.uint8)
    out[:crop.shape[0], :crop.shape[1]] = crop
    for i, text in enumerate(lines):
        cv2.putText(out, text, (4, TILE_H - 24 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def sheet(tiles, title):
    rows = []
    for i in range(0, len(tiles), COLS):
        row = tiles[i:i + COLS]
        row += [np.full((TILE_H, TILE_W, 3), 255, np.uint8)] * (COLS - len(row))
        rows.append(np.hstack(row))
    head = np.full((36, TILE_W * COLS, 3), 255, np.uint8)
    cv2.putText(head, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([head] + rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ket-qua", default="sift_test", help="Thư mục kết quả trong ket_qua/")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    res_dir = ROOT / "ket_qua" / args.ket_qua
    out_dir = res_dir / "loi"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*"):
        old.unlink()

    split_dir, cat, images, anns = load_split(args.split)
    ann = {a["id"]: a for a in anns}
    rows = list(csv.DictReader(open(res_dir / "du_doan_to_tien.csv", encoding="utf-8")))
    errors = [r for r in rows if r["pred"] != r["true"]]

    by_class = defaultdict(list)
    wrong = []
    listing = []
    for r in sorted(errors, key=lambda r: (int(r["true"]), r["file"])):
        a = ann[int(r["ann_id"])]
        img = cv2.imread(str(split_dir / r["file"]))
        w, h = a["bbox"][2:]
        unknown = r["pred"] == UNKNOWN
        lines = [f"{r['file'][:28]}", f"that {r['true']}  ->  {'khong xd' if unknown else r['pred']}  ({int(max(w, h))}px)"]
        t = tile(img, a["bbox"], lines, (0, 165, 255) if unknown else (0, 0, 255))
        by_class[r["true"]].append(t)
        if not unknown:
            wrong.append(t)
        listing.append({"file": r["file"], "ann_id": r["ann_id"], "that": r["true"],
                        "du_doan": r["pred"], "loai_loi": "khong_xac_dinh" if unknown else "sai_menh_gia",
                        "bbox": [round(v) for v in a["bbox"]], "canh_dai_px": int(max(w, h))})

    for c, tiles in sorted(by_class.items(), key=lambda kv: int(kv[0])):
        cv2.imwrite(str(out_dir / f"loi_{int(c):06d}.jpg"),
                    sheet(tiles, f"Menh gia {c}: {len(tiles)} to loi (cam = khong xac dinh, do = sai menh gia)"))
    if wrong:
        cv2.imwrite(str(out_dir / "sai_menh_gia.jpg"), sheet(wrong, f"Nhan sai menh gia: {len(wrong)} to"))

    with open(out_dir / "danh_sach_loi.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=listing[0].keys())
        writer.writeheader()
        writer.writerows(listing)

    print(f"{len(errors)}/{len(rows)} tờ lỗi: {sum(r['pred'] == UNKNOWN for r in errors)} không xác định, "
          f"{len(wrong)} sai mệnh giá")
    for c, tiles in sorted(by_class.items(), key=lambda kv: int(kv[0])):
        print(f"  {int(c):>7,}: {len(tiles)}".replace(",", "."))
    print(f"Đã lưu vào {out_dir}")


if __name__ == "__main__":
    main()
