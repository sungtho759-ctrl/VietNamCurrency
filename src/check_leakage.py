"""Kiểm tra rò rỉ dữ liệu giữa các tập và tạo danh sách ảnh test "sạch".

Roboflow chia ngẫu nhiên các khung hình liên tiếp và bản tăng cường của cùng một ảnh gốc vào cả ba tập.
build_templates.py / yolo_prepare.py đã loại ảnh train gần trùng valid/test, nhưng valid và test vẫn có thể
gần trùng nhau. Vì mọi ngưỡng (và trọng số best.pt của YOLO) được chọn trên valid, ảnh test gần trùng valid
làm kết quả test lạc quan hơn thực tế.

Một ảnh test bị coi là rò rỉ nếu:
  - cùng ảnh gốc (trường extra.name của Roboflow) với một ảnh valid hoặc train, hoặc
  - mã dHash 256 bit cách một ảnh valid (kể cả bản lật ngang / dọc) <= --nguong-trung bit
    (mặc định 70, cùng ngưỡng với build_templates.py).
Ảnh train đã được lọc theo dHash so với test nên chỉ cần thêm điều kiện cùng ảnh gốc.

Kết quả: data/test_clean.json (danh sách ảnh test sạch / rò rỉ, khoảng cách tới valid, và danh sách ảnh
valid gần trùng test để dò lại tham số trên phần valid không trùng test).
Dùng với: python src/evaluate_clean.py

Chạy:  python src/check_leakage.py
"""
import argparse
import json
import sys

import cv2
import numpy as np

from build_templates import dhash
from evaluate_sift import ROOT, load_split

sys.stdout.reconfigure(encoding="utf-8")

OUT_PATH = ROOT / "data" / "test_clean.json"


def hashes(split, flips=False):
    split_dir, _, images, _ = load_split(split)
    names, src, codes = [], [], []
    for im in images.values():
        img = cv2.imread(str(split_dir / im["file_name"]))
        variants = [img, cv2.flip(img, 1), cv2.flip(img, 0)] if flips else [img]
        names.append(im["file_name"])
        src.append(im.get("extra", {}).get("name", im["file_name"]))
        codes.append([dhash(v) for v in variants])
    return names, src, np.array(codes)  # codes: (số ảnh, số biến thể, 256)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nguong-trung", type=int, default=70)
    args = parser.parse_args()

    test_names, test_src, test_codes = hashes("test", flips=True)
    valid_names, valid_src, valid_codes = hashes("valid", flips=True)
    _, train_src, _ = hashes("train")

    t = test_codes[:, 0]                                         # (n_test, 256)
    v = valid_codes.reshape(-1, valid_codes.shape[-1])           # (n_valid * 3, 256)
    dist = (t[:, None, :] != v[None, :, :]).sum(2).min(1)       # khoảng cách nhỏ nhất tới valid
    same_valid = np.array([s in set(valid_src) for s in test_src])
    same_train = np.array([s in set(train_src) for s in test_src])
    leaked = (dist <= args.nguong_trung) | same_valid | same_train

    # Chiều ngược lại: ảnh valid gần trùng test (dùng cho tune_thresholds.py --bo-trung-test)
    tv = test_codes.reshape(-1, test_codes.shape[-1])
    v_dist = (valid_codes[:, 0][:, None, :] != tv[None, :, :]).sum(2).min(1)
    v_leaked = (v_dist <= args.nguong_trung) | np.array([s in set(test_src) for s in valid_src])

    clean = [n for n, bad in zip(test_names, leaked) if not bad]
    info = {
        "nguong_trung": args.nguong_trung,
        "so_anh_test": len(test_names),
        "so_anh_sach": len(clean),
        "ly_do": {"dhash_gan_trung_valid": int((dist <= args.nguong_trung).sum()),
                  "cung_anh_goc_valid": int(same_valid.sum()),
                  "cung_anh_goc_train": int(same_train.sum())},
        "sach": sorted(clean),
        "cung_anh_goc": sorted(n for n, a, b in zip(test_names, same_valid, same_train) if a or b),
        "khoang_cach_toi_valid": {n: int(d) for n, d in zip(test_names, dist)},
        "valid_trung_test": sorted(n for n, bad in zip(valid_names, v_leaked) if bad),
    }
    OUT_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Ảnh test: {len(test_names)}, sạch: {len(clean)}, rò rỉ: {int(leaked.sum())}")
    print("  dHash gần trùng valid (<= {} bit): {}".format(args.nguong_trung, info["ly_do"]["dhash_gan_trung_valid"]))
    print("  cùng ảnh gốc với valid: {}, với train: {}".format(int(same_valid.sum()), int(same_train.sum())))
    print(f"Ảnh valid gần trùng test: {int(v_leaked.sum())}/{len(valid_names)}")
    for thr in (10, 30, 50, 70, 90):
        print(f"  khoảng cách tới valid <= {thr:>2} bit: {(dist <= thr).sum()} ảnh")
    print("Lưu tại:", OUT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
