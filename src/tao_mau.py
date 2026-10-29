"""Tạo ảnh mẫu bổ sung và mô hình màu từ tập train.

1. Loại các ảnh train gần trùng với ảnh valid/test (Roboflow chia ngẫu nhiên các khung hình
   liên tiếp và các bản tăng cường của cùng một ảnh gốc vào nhiều tập). Hai ảnh coi là gần trùng
   nếu khoảng cách Hamming giữa mã dHash 256 bit <= --nguong-trung (mặc định 70).
2. Ảnh mẫu: với mỗi mệnh giá chọn tối đa --k tờ rõ nét, to, đa dạng (mỗi ảnh gốc tối đa một tờ,
   các tờ được chọn không gần giống nhau), lưu vào data/templates_train/<mệnh giá>_t<thứ hạng>.jpg.
   Chỉ lấy tờ nằm gần thẳng trục (tỷ lệ cạnh bbox gần tỷ lệ tờ tiền) và cắt sát bbox, để ảnh mẫu
   gần như không lẫn nền; nếu không SIFT sẽ khớp vào nền (mặt bàn, thảm) thay vì tờ tiền.
3. Mô hình màu: histogram HSV của mọi tờ train còn lại, lưu vào data/mo_hinh_mau.npz.

Chạy:  python src/tao_mau.py
"""
import argparse
import json
import sys
from collections import defaultdict

import cv2
import numpy as np

from danh_gia_sift import ROOT, crop_box, load_split
from mau_sac import MODEL_PATH, hsv_hist

# In được tiếng Việt cả khi chạy trong IDE hoặc chuyển hướng đầu ra trên Windows
sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = ROOT / "data" / "templates_train"


def dhash(img, size=16):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size + 1, size), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).ravel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8, help="Số ảnh mẫu tối đa mỗi mệnh giá")
    parser.add_argument("--nguong-trung", type=int, default=70)
    parser.add_argument("--canh-toi-thieu", type=int, default=150, help="Cạnh dài tối thiểu của tờ (px)")
    args = parser.parse_args()

    # 1. Loại ảnh train gần trùng với valid/test
    held_out = []
    for split in ("valid", "test"):
        split_dir, _, images, _ = load_split(split)
        held_out += [dhash(cv2.imread(str(split_dir / im["file_name"]))) for im in images.values()]
    held_out = np.array(held_out)

    train_dir, cat, images, anns = load_split("train")
    keep = {}
    for iid, im in images.items():
        img = cv2.imread(str(train_dir / im["file_name"]))
        if (held_out != dhash(img)).sum(1).min() > args.nguong_trung:
            keep[iid] = img
    print(f"Giữ {len(keep)}/{len(images)} ảnh train (loại ảnh gần trùng valid/test)")

    # 2. Đặc trưng màu và ứng viên ảnh mẫu
    hists, labels = [], []
    pool = defaultdict(list)
    for a in anns:
        if a["image_id"] not in keep:
            continue
        img = keep[a["image_id"]]
        label = cat[a["category_id"]]
        hists.append(hsv_hist(img, a["bbox"]))
        labels.append(label)
        if max(a["bbox"][2:]) < args.canh_toi_thieu:
            continue
        x, y, w, h = map(int, a["bbox"])
        # Loại tờ bị cắt mép ảnh: ảnh mẫu cần thấy trọn tờ tiền
        H, W = img.shape[:2]
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            continue
        # Tờ tiền Việt Nam có tỷ lệ dài/rộng khoảng 2,1–2,35; tờ xoay nghiêng cho bbox vuông hơn
        if not 1.9 <= max(w, h) / min(w, h) <= 2.6:
            continue
        crop = crop_box(img, a["bbox"], pad=0.0)
        sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        pool[label].append({"crop": crop, "image_id": a["image_id"], "hash": dhash(crop),
                            "score": max(w, h) * np.sqrt(sharp)})

    np.savez(MODEL_PATH, hists=np.array(hists, dtype=np.float32), labels=np.array(labels))
    print(f"Mô hình màu: {len(hists)} tờ -> {MODEL_PATH.relative_to(ROOT)}")

    # 3. Chọn ảnh mẫu: ưu tiên to và nét, mỗi ảnh gốc một tờ, không gần giống tờ đã chọn
    OUT_DIR.mkdir(exist_ok=True)
    for old in OUT_DIR.glob("*.jpg"):
        old.unlink()
    summary = {}
    for label in sorted(pool, key=int):
        chosen = []
        for c in sorted(pool[label], key=lambda c: -c["score"]):
            if len(chosen) == args.k:
                break
            if any(c["image_id"] == s["image_id"] or (c["hash"] != s["hash"]).sum() < 40 for s in chosen):
                continue
            chosen.append(c)
        for rank, c in enumerate(chosen):
            cv2.imwrite(str(OUT_DIR / f"{label}_t{rank}.jpg"), c["crop"])
        summary[label] = len(chosen)
    print("Ảnh mẫu mới theo mệnh giá:", summary)
    (OUT_DIR / "thong_tin.json").write_text(json.dumps(
        {"k": args.k, "nguong_trung": args.nguong_trung, "anh_train_giu_lai": len(keep), "so_mau": summary},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
