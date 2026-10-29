"""Đặc trưng màu HSV cho tờ tiền và bộ phân loại theo láng giềng gần nhất.

Histogram 2 chiều H × S (bỏ kênh V để bớt phụ thuộc độ sáng), lấy ở vùng giữa tờ tiền.
Mô hình là tập histogram của các tờ trong tập train (đã loại ảnh gần trùng với valid/test),
lưu ở data/mo_hinh_mau.npz bởi tao_mau.py.
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data" / "mo_hinh_mau.npz"
H_BINS, S_BINS = 30, 8


def hsv_hist(img, bbox=None):
    """Histogram H × S chuẩn hóa tổng bằng 1 của vùng giữa tờ tiền."""
    if bbox is not None:
        x, y, w, h = map(int, bbox)
        img = img[y:y + h, x:x + w]
    h, w = img.shape[:2]
    # Bỏ 15% mỗi phía: bbox thẳng trục nên các góc thường lẫn nền khi tờ tiền bị xoay
    core = img[int(0.15 * h):int(0.85 * h) or h, int(0.15 * w):int(0.85 * w) or w]
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS], [0, 180, 0, 256]).ravel()
    return hist / max(hist.sum(), 1e-9)


class ColorModel:
    def __init__(self, path=MODEL_PATH):
        data = np.load(path)
        self.labels = data["labels"].astype(str)
        self.sqrt_hists = np.sqrt(data["hists"])
        self.classes = sorted(set(self.labels), key=int)

    def rank(self, hist):
        """Các mệnh giá xếp theo khoảng cách Bhattacharyya tới tờ train gần nhất của lớp đó.

        Trả về [(nhãn, khoảng cách), ...] tăng dần theo khoảng cách.
        """
        sim = self.sqrt_hists @ np.sqrt(hist)
        dist = np.sqrt(np.clip(1.0 - sim, 0, None))
        per_class = [(c, float(dist[self.labels == c].min())) for c in self.classes]
        return sorted(per_class, key=lambda x: x[1])
