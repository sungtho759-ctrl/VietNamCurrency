"""Giao diện web VND Vision AI: nhận diện mệnh giá bằng phương pháp truyền thống và YOLO11n để so sánh.

Trình duyệt gửi cùng một ảnh (ảnh tải lên hoặc khung hình webcam) tới hai API chạy độc lập:
  - POST /api/sift : SIFT + RANSAC + màu HSV, tìm mọi tờ trong ảnh (detect_all, giống "Nhận diện tất cả các tờ" của
                     gui.py và bảng "tự tìm mọi tờ" trong README).
  - POST /api/yolo : YOLO11n (results/yolo/yolo11n/weights/best.pt), ngưỡng và kiểu NMS chọn trên tập valid
                     (results/yolo/evaluation_test/results.json).
Hai API trả cùng định dạng: {"notes": [{"label", "quad", "score", "evidence"}], "total", "ms", "width", "height"}.
Mỗi phương pháp có khóa riêng: một yêu cầu SIFT chậm không chặn YOLO.

Chạy:  python src/web_app.py            rồi mở http://localhost:5000
       python src/web_app.py --port 8000
"""
import argparse
import json
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

from color_model import MODEL_PATH, ColorModel
from evaluate_sift import MULTI_UPSCALE, ROOT, detect_all, load_templates, resize_max

sys.stdout.reconfigure(encoding="utf-8")

WEB_DIR = ROOT / "web"
SAMPLE_DIR = ROOT / "data" / "test_images"
YOLO_WEIGHTS = ROOT / "results" / "yolo" / "yolo11n" / "weights" / "best.pt"
YOLO_RESULTS = ROOT / "results" / "yolo" / "evaluation_test" / "results.json"
# SIFT: ảnh lớn được thu về cạnh dài tối đa SIFT_MAX_SIDE; ảnh nhỏ cỡ bộ dữ liệu (≤ 640 px) được phóng to ×1,5 như
# khi đánh giá nhiều tờ (MULTI_UPSCALE) để tờ nhỏ có thêm điểm đặc trưng.
SIFT_MAX_SIDE = 800
MAX_UPLOAD_MB = 15

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


class Models:
    """Nạp mô hình một lần khi khởi động."""

    def __init__(self):
        print("Đang nạp ảnh mẫu SIFT và mô hình màu...", flush=True)
        self.templates = load_templates()
        self.color = ColorModel() if MODEL_PATH.exists() else None
        self.sift_lock = threading.Lock()
        self.yolo_lock = threading.Lock()
        self.yolo, self.yolo_error = None, ""
        self.yolo_conf, self.yolo_agnostic = 0.35, True
        if YOLO_RESULTS.exists():
            chon = json.loads(YOLO_RESULTS.read_text(encoding="utf-8"))["chon_tren_valid"]["nhieu_to"]
            self.yolo_conf, self.yolo_agnostic = chon["conf"], chon["agnostic"]
        try:
            from ultralytics import YOLO
            print("Đang nạp YOLO11n...", flush=True)
            self.yolo = YOLO(str(YOLO_WEIGHTS))
            self.yolo.predict(np.zeros((64, 64, 3), np.uint8), device="cpu", verbose=False)  # khởi tạo
        except Exception as exc:  # thiếu ultralytics / trọng số: vẫn dùng được phương pháp truyền thống
            self.yolo_error = str(exc).splitlines()[0]
            print("Không nạp được YOLO:", self.yolo_error, flush=True)


MODELS = None


def read_image():
    """Ảnh gửi lên dạng multipart (trường "image") hoặc thân yêu cầu nhị phân."""
    data = request.files["image"].read() if "image" in request.files else request.get_data()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None:
        abort(400, "Không đọc được ảnh")
    return img


def clip_to_image(quad, w, h):
    """Tờ bị cắt mép làm khung homography vượt ra ngoài ảnh: chỉ giữ phần nằm trong ảnh."""
    frame = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    area, inside = cv2.intersectConvexConvex(np.float32(quad), frame)
    pts = inside.reshape(-1, 2) if area > 0 and inside is not None else np.float32(quad)
    return pts.round(1).tolist()


def response(notes, img, seconds, method, **extra):
    h, w = img.shape[:2]
    return jsonify({"method": method, "notes": notes, "total": sum(int(n["label"]) for n in notes),
                    "ms": round(seconds * 1000), "width": w, "height": h, **extra})


@app.post("/api/sift")
def api_sift():
    img = read_image()
    small = resize_max(img, SIFT_MAX_SIDE)
    scale = img.shape[1] / small.shape[1]
    upscale = MULTI_UPSCALE if max(small.shape[:2]) <= 640 else 1.0
    with MODELS.sift_lock:
        t0 = time.perf_counter()
        found = detect_all(small, MODELS.templates, MODELS.color, upscale=upscale)
        seconds = time.perf_counter() - t0
    h, w = img.shape[:2]
    notes = [{"label": label, "quad": clip_to_image(np.float32(box) * scale, w, h), "score": n,
              "evidence": f"{n} cặp điểm khớp" if n else "theo màu"} for label, n, box in found]
    return response(notes, img, seconds, "sift")


@app.post("/api/yolo")
def api_yolo():
    if MODELS.yolo is None:
        return jsonify({"error": "Chưa nạp được YOLO: " + MODELS.yolo_error}), 503
    img = read_image()
    with MODELS.yolo_lock:
        t0 = time.perf_counter()
        r = MODELS.yolo.predict(img, conf=MODELS.yolo_conf, imgsz=640, agnostic_nms=MODELS.yolo_agnostic,
                                device="cpu", verbose=False)[0]
        seconds = time.perf_counter() - t0
    notes = []
    for (x1, y1, x2, y2), c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        notes.append({"label": MODELS.yolo.names[int(c)], "quad": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                      "score": round(p, 3), "evidence": f"độ tin cậy {p:.2f}"})
    return response(notes, img, seconds, "yolo", conf=MODELS.yolo_conf)


@app.get("/api/info")
def api_info():
    return jsonify({"yolo": MODELS.yolo is not None, "yolo_error": MODELS.yolo_error, "yolo_conf": MODELS.yolo_conf,
                    "yolo_classes": list(MODELS.yolo.names.values()) if MODELS.yolo else [],
                    "sift_templates": len(MODELS.templates), "color_model": MODELS.color is not None,
                    "samples": sorted(p.name for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))})


@app.get("/samples/<path:name>")
def sample(name):
    return send_from_directory(SAMPLE_DIR, name)


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/<path:name>")
def static_file(name):
    return send_from_directory(WEB_DIR, name)


def main():
    global MODELS
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    MODELS = Models()
    print(f"Mở trình duyệt tại http://localhost:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
