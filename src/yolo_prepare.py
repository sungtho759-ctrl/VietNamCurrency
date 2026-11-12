"""Chuẩn bị dữ liệu cho YOLO: đổi nhãn COCO sang định dạng YOLO, tạo tệp cấu hình tiền.yaml.

- Dùng load_split() nên nhãn đã được sửa lỗi gán tên lớp (xem mục "Ghi chú về dữ liệu" trong README).
- Tập train được loại ảnh gần trùng với valid/test bằng dHash (cùng ngưỡng 70 bit với build_templates.py) để YOLO không
  được lợi từ rò rỉ dữ liệu; thêm --giu-trung để giữ toàn bộ tập train (chỉ dùng khi đo ảnh hưởng của rò rỉ).
- Ảnh được tạo liên kết cứng (hard link) nên không tốn thêm dung lượng; nhãn ghi vào data/dataset_yolo/.

Chạy:  python src/yolo_prepare.py          # thêm --zip để đóng gói colab_yolo/dataset_yolo.zip cho Colab
"""
import argparse
import os
import shutil
import sys
import zipfile
from collections import defaultdict

import cv2
import numpy as np

from evaluate_sift import ROOT, load_split
from build_templates import dhash

sys.stdout.reconfigure(encoding="utf-8")

# Lớp "500" thêm sau nên đặt cuối để giữ nguyên chỉ số của 9 lớp gốc
LOP = ["1000", "2000", "5000", "10000", "20000", "50000", "100000", "200000", "500000", "500"]


def lien_ket(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def xuat_tap(split, out_root, giu=None):
    """Ghi ảnh và nhãn YOLO (lớp, tâm x, tâm y, rộng, cao – chuẩn hóa về [0, 1]) của một tập."""
    split_dir, cat, images, anns = load_split(split)
    theo_anh = defaultdict(list)
    for a in anns:
        theo_anh[a["image_id"]].append(a)
    (out_root / split / "images").mkdir(parents=True)
    (out_root / split / "labels").mkdir(parents=True)
    so_to = 0
    for iid, im in images.items():
        if giu is not None and iid not in giu:
            continue
        W, H = im["width"], im["height"]
        dong = []
        for a in theo_anh[iid]:
            x, y, w, h = a["bbox"]
            dong.append(f"{LOP.index(cat[a['category_id']])} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                        f"{w / W:.6f} {h / H:.6f}")
        lien_ket(split_dir / im["file_name"], out_root / split / "images" / im["file_name"])
        ten_nhan = os.path.splitext(im["file_name"])[0] + ".txt"
        (out_root / split / "labels" / ten_nhan).write_text("\n".join(dong), encoding="utf-8")
        so_to += len(dong)
    so_anh = len(images) if giu is None else len(giu)
    print(f"  {split}: {so_anh} ảnh, {so_to} tờ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nguong-trung", type=int, default=70)
    parser.add_argument("--giu-trung", action="store_true", help="Giữ cả ảnh train gần trùng valid/test")
    parser.add_argument("--zip", action="store_true", help="Đóng gói thêm colab_yolo/<tên>.zip để tải lên Colab")
    args = parser.parse_args()

    ten = "dataset_yolo_full_train" if args.giu_trung else "dataset_yolo"
    out_root = ROOT / "data" / ten
    if out_root.exists():
        shutil.rmtree(out_root)

    train_dir, _, images, _ = load_split("train")
    giu = None
    if not args.giu_trung:
        held_out = []
        for split in ("valid", "test"):
            split_dir, _, ims, _ = load_split(split)
            held_out += [dhash(cv2.imread(str(split_dir / im["file_name"]))) for im in ims.values()]
        held_out = np.array(held_out)
        giu = {iid for iid, im in images.items()
               if (held_out != dhash(cv2.imread(str(train_dir / im["file_name"])))).sum(1).min() > args.nguong_trung}
        print(f"Giữ {len(giu)}/{len(images)} ảnh train (loại ảnh gần trùng valid/test)")

    xuat_tap("train", out_root, giu)
    xuat_tap("valid", out_root)
    xuat_tap("test", out_root)
    ten_lop = "\n".join(f"  {i}: '{c}'" for i, c in enumerate(LOP))
    (out_root / "currency.yaml").write_text(
        f"path: {out_root.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n{ten_lop}\n",
        encoding="utf-8")
    print("Đã ghi", out_root / "currency.yaml")
    if args.zip:
        dong_goi(out_root, ROOT / "colab_yolo" / f"{ten}.zip")


def dong_goi(out_root, dich):
    """Nén ảnh + nhãn vào thư mục dataset_yolo/ trong zip; currency.yaml trỏ tới /content/dataset_yolo trên Colab."""
    with zipfile.ZipFile(dich, "w", zipfile.ZIP_STORED) as z:
        for tep in sorted(out_root.rglob("*")):
            if tep.is_file() and tep.suffix != ".cache" and tep.name != "currency.yaml":
                z.write(tep, "dataset_yolo/" + tep.relative_to(out_root).as_posix())
        yaml = (out_root / "currency.yaml").read_text(encoding="utf-8").split("\n", 1)[1]
        z.writestr("dataset_yolo/currency.yaml", "path: /content/dataset_yolo\n" + yaml)
    print("Đã ghi", dich)


if __name__ == "__main__":
    main()
