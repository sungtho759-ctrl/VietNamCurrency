"""Huấn luyện YOLO phát hiện và phân loại mệnh giá tiền (cần chạy yolo_chuan_bi.py trước).

Chạy:  python src/yolo_huan_luyen.py --epochs 60
Kết quả: ket_qua/yolo/<--ten>/weights/best.pt (mô hình tốt nhất theo tập valid) và các biểu đồ huấn luyện.
"""
import argparse
import sys

import torch
from ultralytics import YOLO

from danh_gia_sift import ROOT

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mo-hinh", default="yolo11n.pt", help="Trọng số khởi tạo (học chuyển giao từ COCO)")
    parser.add_argument("--du-lieu", default="dataset_yolo", help="Thư mục dữ liệu trong data/")
    parser.add_argument("--ten", default="yolo11n", help="Tên lần chạy trong ket_qua/yolo/")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15, help="Dừng sớm sau số epoch không cải thiện trên valid")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threads", type=int, default=8, help="Số luồng CPU")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model = YOLO(args.mo_hinh)
    model.train(data=str(ROOT / "data" / args.du_lieu / "tien.yaml"), epochs=args.epochs, patience=args.patience,
                batch=args.batch, imgsz=args.imgsz, device=args.device, seed=0, deterministic=True, workers=2,
                cache="ram", project=str(ROOT / "ket_qua" / "yolo"), name=args.ten, exist_ok=True, plots=True)


if __name__ == "__main__":
    main()
