"""Giao diện web VND Vision AI: nhận diện mệnh giá bằng phương pháp truyền thống và YOLO11n để so sánh.

Trình duyệt gửi cùng một ảnh (ảnh tải lên hoặc khung hình webcam) tới hai API chạy độc lập:
  - POST /api/sift : SIFT + RANSAC + màu HSV, tìm mọi tờ trong ảnh (detect_all, giống "Nhận diện tất cả các tờ" của
                     gui.py và bảng "tự tìm mọi tờ" trong README).
  - POST /api/yolo : YOLO11n (results/yolo/yolo11n/weights/best.pt), ngưỡng và kiểu NMS chọn trên tập valid
                     (results/yolo/evaluation_test/results.json).
Hai API trả cùng định dạng: {"notes": [{"label", "quad", "score", "evidence"}], "total", "ms", "width", "height"}.
Mỗi phương pháp có khóa riêng: một yêu cầu SIFT chậm không chặn YOLO.

Mục "Kết hợp YOLO + SIFT" (cùng trang, tách khỏi phần so sánh hai phương pháp) dùng API thứ ba:
  - POST /api/hybrid : YOLO11n định vị từng tờ, SIFT + màu nhận diện mệnh giá trên vùng cắt – cùng hàm
                       Recognizers.classify_crop của realtime.py (vùng cắt <= 400 px, luật inlier riêng, cổng bằng
                       chứng, CLAHE dự phòng). SIFT + màu chưa kết luận thì để "?", trừ khi SIFT đã thấy hoa văn tờ
                       tiền (ứng viên >= YOLO_LABEL_MIN_PATTERN inlier, khung gần chữ nhật) nhưng chưa chắc mệnh giá:
                       khi đó dùng nhãn YOLO (mặc định; ?yolo_label=0 tắt hẳn). YOLO không thấy tờ nào thì thử ảnh
                       xoay 180°, rồi SIFT trên toàn ảnh (detect_all) với cùng cổng bằng chứng như realtime.py.
                       Không dùng nhãn YOLO vô điều kiện: YOLO chỉ học ảnh có tờ tiền nên coi thẻ ATM, thẻ bảo hiểm y tế
                       là tờ tiền, kể cả với độ tin cậy cao (docs/vercel.md, mục "Vật không phải tiền").
YOLO dùng best.pt qua ultralytics; nếu không có ultralytics (như khi triển khai trên Vercel, xem docs/vercel.md) thì
dùng best.onnx qua cv2.dnn (yolo_onnx.py). Mô hình được nạp ở yêu cầu đầu tiên, nên `app` dùng được trực tiếp
trong môi trường serverless (api/index.py).

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

from evaluate_sift import MIN_INLIERS, MULTI_UPSCALE, ROOT, UNKNOWN, detect_all, quad_angle_dev, resize_max
from realtime import EVIDENCE_ANGLE, Recognizers

sys.stdout.reconfigure(encoding="utf-8")

WEB_DIR = ROOT / "web"
SAMPLE_DIR = ROOT / "data" / "test_images"
YOLO_WEIGHTS = ROOT / "results" / "yolo" / "yolo11n" / "weights" / "best.pt"
YOLO_ONNX = YOLO_WEIGHTS.with_suffix(".onnx")
YOLO_RESULTS = ROOT / "results" / "yolo" / "evaluation_test" / "results.json"
# SIFT: ảnh lớn được thu về cạnh dài tối đa SIFT_MAX_SIDE; ảnh nhỏ cỡ bộ dữ liệu (≤ 640 px) được phóng to ×1,5 như
# khi đánh giá nhiều tờ (MULTI_UPSCALE) để tờ nhỏ có thêm điểm đặc trưng.
SIFT_MAX_SIDE = 800
# Kết hợp: chỉ dùng nhãn YOLO cho khung SIFT + màu chưa kết luận khi có ứng viên SIFT >= N inlier với khung gần chữ
# nhật (hoa văn tờ tiền, chưa chắc mệnh giá). Chọn trên valid trong các luật không để lọt vật không phải tiền (246 biến
# thể của 6 ảnh thẻ ATM / bảo hiểm y tế: 0 bị báo ra tiền): F1 valid 88,52% / test 87,14%. Luôn dùng nhãn YOLO cho F1
# 94,83% / 95,51% nhưng 132/246 biến thể thẻ bị báo ra tiền; ngưỡng độ tin cậy YOLO >= 0,8 vẫn lọt 34/246.
YOLO_LABEL_MIN_PATTERN = 8
MAX_UPLOAD_MB = 15

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


class Models:
    """Nạp mô hình một lần khi khởi động."""

    def __init__(self):
        print("Đang nạp ảnh mẫu SIFT (thường + CLAHE) và mô hình màu...", flush=True)
        # Phần SIFT + màu của realtime.py (không nạp YOLO ở đó): dùng cho /api/sift và bước nhận diện của /api/hybrid
        self.rec = Recognizers(want_yolo=False)
        self.templates, self.color = self.rec.templates, self.rec.color
        # Một khóa cho mọi việc chạy SIFT: CLAHE dự phòng của classify_crop đổi tạm biến toàn cục
        # evaluate_sift.CLAHE_CLIP, nên /api/sift và /api/hybrid không được trích SIFT cùng lúc.
        self.sift_lock = threading.Lock()
        self.yolo_lock = threading.Lock()
        self.yolo, self.yolo_error, self.yolo_names, self.yolo_backend = None, "", {}, ""
        self.yolo_conf, self.yolo_agnostic = 0.35, True
        if YOLO_RESULTS.exists():
            chon = json.loads(YOLO_RESULTS.read_text(encoding="utf-8"))["chon_tren_valid"]["nhieu_to"]
            self.yolo_conf, self.yolo_agnostic = chon["conf"], chon["agnostic"]
        try:
            try:
                from ultralytics import YOLO
            except ImportError:  # Vercel: không có torch / ultralytics -> dùng bản ONNX
                from yolo_onnx import OnnxYolo
                print("Đang nạp YOLO11n (ONNX)...", flush=True)
                self.yolo = OnnxYolo(YOLO_ONNX)
                self.yolo_backend = "onnx"
            else:
                print("Đang nạp YOLO11n...", flush=True)
                self.yolo = YOLO(str(YOLO_WEIGHTS))
                self.yolo_backend = "pt"
            self.yolo_names = self.yolo.names
            self.predict_yolo(np.zeros((64, 64, 3), np.uint8))  # khởi tạo
        except Exception as exc:  # thiếu ultralytics / trọng số: vẫn dùng được phương pháp truyền thống
            self.yolo, self.yolo_error = None, str(exc).splitlines()[0]
            print("Không nạp được YOLO:", self.yolo_error, flush=True)

    def predict_yolo(self, img):
        """[(x1, y1, x2, y2, độ tin cậy, chỉ số lớp)] với ngưỡng và kiểu NMS chọn trên valid."""
        if self.yolo_backend == "onnx":
            return self.yolo.predict(img, conf=self.yolo_conf, agnostic=self.yolo_agnostic)
        r = self.yolo.predict(img, conf=self.yolo_conf, imgsz=640, agnostic_nms=self.yolo_agnostic,
                              device="cpu", verbose=False)[0]
        return [(*xyxy, p, int(c)) for xyxy, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                                         r.boxes.conf.tolist())]


MODELS = None
_MODELS_LOCK = threading.Lock()


def models():
    """Nạp mô hình ở lần gọi đầu tiên (chạy cục bộ: nạp sẵn trong main(); serverless: ở yêu cầu đầu tiên)."""
    global MODELS
    if MODELS is None:
        with _MODELS_LOCK:
            if MODELS is None:
                MODELS = Models()
    return MODELS


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
    m = models()
    with m.sift_lock:
        t0 = time.perf_counter()
        found = detect_all(small, m.templates, m.color, upscale=upscale)
        seconds = time.perf_counter() - t0
    h, w = img.shape[:2]
    notes = [{"label": label, "quad": clip_to_image(np.float32(box) * scale, w, h), "score": n,
              "evidence": f"{n} cặp điểm khớp" if n else "theo màu"} for label, n, box in found]
    return response(notes, img, seconds, "sift")


@app.post("/api/yolo")
def api_yolo():
    m = models()
    if m.yolo is None:
        return jsonify({"error": "Chưa nạp được YOLO: " + m.yolo_error}), 503
    img = read_image()
    with m.yolo_lock:
        t0 = time.perf_counter()
        dets = m.predict_yolo(img)
        seconds = time.perf_counter() - t0
    notes = [{"label": m.yolo_names[c], "quad": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
              "score": round(p, 3), "evidence": f"độ tin cậy {p:.2f}"} for x1, y1, x2, y2, p, c in dets]
    return response(notes, img, seconds, "yolo", conf=m.yolo_conf)


def yolo_boxes(m, img):
    """Khung YOLO [(nhãn, [x, y, w, h], độ tin cậy)]; không thấy tờ nào thì thử ảnh xoay 180° (YOLO hay bỏ sót tờ lộn
    ngược, như realtime.py). Trả thêm cờ cho biết có phải dùng ảnh xoay không."""
    H, W = img.shape[:2]
    with m.yolo_lock:
        dets = m.predict_yolo(img)
        if dets:
            return [(m.yolo_names[c], [x1, y1, x2 - x1, y2 - y1], p) for x1, y1, x2, y2, p, c in dets], False
        dets = m.predict_yolo(cv2.rotate(img, cv2.ROTATE_180))
    return [(m.yolo_names[c], [W - x2, H - y2, x2 - x1, y2 - y1], p) for x1, y1, x2, y2, p, c in dets], True


@app.post("/api/hybrid")
def api_hybrid():
    m = models()
    if m.yolo is None:
        return jsonify({"error": "Phương pháp kết hợp cần YOLO: " + m.yolo_error}), 503
    use_yolo_label = request.args.get("yolo_label", "1") != "0"
    img = read_image()
    h, w = img.shape[:2]
    t0 = time.perf_counter()
    boxes, flipped = yolo_boxes(m, img)
    t1 = time.perf_counter()
    notes, fallback = [], False
    with m.sift_lock:
        for ylab, (x, y, bw, bh), p in boxes:
            pred, inliers, clahe, pattern = m.rec.classify_crop(img, [x, y, bw, bh], ylab, with_pattern=True)
            note = {"quad": [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], "yolo_label": ylab,
                    "yolo_conf": round(p, 3), "inliers": inliers, "clahe": clahe, "pattern": pattern}
            if pred != UNKNOWN:
                note.update(label=pred, source="sift", agree=pred == ylab,
                            evidence=f"{inliers} cặp điểm khớp" + (", CLAHE" if clahe else ""))
            elif use_yolo_label and pattern >= YOLO_LABEL_MIN_PATTERN:
                note.update(label=ylab, source="yolo", agree=None,
                            evidence=f"SIFT thấy hoa văn tờ tiền ({pattern} điểm) nhưng chưa chắc mệnh giá")
            else:
                note.update(label=None, source="unknown", agree=None, evidence="SIFT + màu chưa kết luận")
            notes.append(note)
        if not boxes:
            # YOLO không thấy tờ nào (kể cả ảnh xoay): SIFT tự tìm mọi tờ như /api/sift, nhưng chỉ nhận tờ có bằng
            # chứng hình học như realtime.py (Recognizers.sift): >= MIN_INLIERS cặp khớp và khung gần chữ nhật – không
            # có cổng này, vật không phải tiền (thẻ ATM) bị báo mệnh giá nhờ vài điểm khớp ngẫu nhiên hoặc chỉ theo màu.
            fallback = True
            small = resize_max(img, SIFT_MAX_SIDE)
            scale = img.shape[1] / small.shape[1]
            upscale = MULTI_UPSCALE if max(small.shape[:2]) <= 640 else 1.0
            for label, n, box in detect_all(small, m.templates, m.color, upscale=upscale):
                if n < MIN_INLIERS or quad_angle_dev(np.float32(box)) > EVIDENCE_ANGLE:
                    continue
                notes.append({"quad": clip_to_image(np.float32(box) * scale, w, h), "label": label,
                              "source": "sift_full", "agree": None, "yolo_label": None, "yolo_conf": None,
                              "inliers": n, "clahe": False, "pattern": n,
                              "evidence": f"{n} cặp điểm khớp, SIFT toàn ảnh"})
    t2 = time.perf_counter()
    known = [n for n in notes if n["label"] is not None]
    return jsonify({"method": "hybrid", "notes": notes, "total": sum(int(n["label"]) for n in known),
                    "ms": round((t2 - t0) * 1000), "ms_yolo": round((t1 - t0) * 1000),
                    "ms_sift": round((t2 - t1) * 1000), "width": w, "height": h, "flipped": flipped,
                    "fallback": fallback, "yolo_label": use_yolo_label, "yolo_conf": m.yolo_conf})


@app.get("/api/info")
def api_info():
    m = models()
    return jsonify({"yolo": m.yolo is not None, "yolo_error": m.yolo_error, "yolo_conf": m.yolo_conf,
                    "yolo_backend": m.yolo_backend if m.yolo else "",
                    "yolo_classes": list(m.yolo_names.values()),
                    "sift_templates": len(m.templates), "color_model": m.color is not None,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    models()
    print(f"Mở trình duyệt tại http://localhost:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
