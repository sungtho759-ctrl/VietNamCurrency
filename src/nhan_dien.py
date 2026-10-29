"""Nhận diện mệnh giá tờ tiền trong một ảnh (SIFT + RANSAC + màu HSV).

Dùng chung logic và cấu hình mặc định với danh_gia_sift.py: SIFT định vị tờ tiền bằng homography,
màu được lấy trong khung tìm được, sau đó kết hợp hai nguồn bằng fuse().

Chạy:  python src/nhan_dien.py [đường dẫn ảnh]     (mặc định data/anh_thu/query.jpg)
"""
import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from danh_gia_sift import ROOT, UNKNOWN, classify_image, load_templates
from mau_sac import MODEL_PATH, ColorModel

# In được tiếng Việt cả khi chạy trong IDE hoặc chuyển hướng đầu ra trên Windows
sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("anh", nargs="?", default=str(ROOT / "data" / "anh_thu" / "query.jpg"))
    args = parser.parse_args()

    query = cv2.imread(args.anh)
    if query is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {args.anh}")

    templates = load_templates()
    color_model = ColorModel() if MODEL_PATH.exists() else None
    if color_model is None:
        print("Chưa có mô hình màu (chạy python src/tao_mau.py); chỉ dùng SIFT.")

    label, inliers, box = classify_image(query, templates, color_model)

    result = query.copy()
    if label == UNKNOWN:
        print("Kết quả: Không xác định")
        title = "Khong xac dinh"
    else:
        value = f"{int(label):,}".replace(",", ".")
        evidence = f"{inliers} cặp điểm khớp" if inliers else "theo màu"
        print(f"Kết quả: {value} VNĐ ({evidence})")
        title = f"{value} VND"
        if box is not None:
            cv2.polylines(result, [np.int32(box)], True, (0, 255, 0), 3)

    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
