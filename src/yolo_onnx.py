"""Chạy YOLO11n đã xuất sang ONNX bằng mô-đun dnn của OpenCV, không cần PyTorch / ultralytics / onnxruntime.

Dùng khi triển khai lên Vercel: torch + ultralytics (và cả onnxruntime, ~75 MB) làm gói hàm serverless vượt giới hạn
250 MB, còn OpenCV thì phương pháp truyền thống đã cần sẵn. Tiền xử lý và hậu xử lý làm giống ultralytics 8.4 (letterbox cạnh dài 640 đệm tới bội 32, NMS IoU 0,5 =
PREDICT_IOU) nên kết quả khớp best.pt: cùng khung, cùng nhãn trên 234 ảnh test + test_images.

Tạo best.onnx (một lần, trên máy có ultralytics):
    yolo export model=results/yolo/yolo11n/weights/best.pt format=onnx imgsz=640 dynamic=True opset=17
"""
import ast
import re

import cv2
import numpy as np


def onnx_metadata(path):
    """Đọc metadata_props của tệp ONNX (ultralytics ghi names, stride, ...) mà không cần gói onnx.

    Mỗi mục là StringStringEntryProto: trường 1 = khóa, trường 2 = giá trị, cùng kiểu chuỗi có độ dài varint."""
    data = path.read_bytes() if hasattr(path, "read_bytes") else open(path, "rb").read()

    def varint(i):
        n = shift = 0
        while True:
            b = data[i]
            n |= (b & 0x7F) << shift
            i += 1
            if b < 0x80:
                return n, i
            shift += 7

    meta = {}
    for key in ("names", "stride"):
        m = re.search(re.escape(bytes([0x0A, len(key)]) + key.encode() + bytes([0x12])), data)
        if m:
            n, i = varint(m.end())
            meta[key] = data[i:i + n].decode("utf-8")
    return meta


class OnnxYolo:
    def __init__(self, path, imgsz=640, iou=0.5, max_det=300):
        self.net = cv2.dnn.readNetFromONNX(str(path))
        meta = onnx_metadata(path)
        self.names = {int(k): v for k, v in ast.literal_eval(meta["names"]).items()}
        self.stride = int(meta.get("stride", 32))
        self.imgsz, self.iou, self.max_det = imgsz, iou, max_det

    def letterbox(self, img):
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = round(w * r), round(h * r)
        # Như ultralytics khi dự đoán: chỉ đệm tới bội của stride 32 (ảnh chữ nhật), không đệm thành 640 x 640
        dw, dh = ((self.imgsz - nw) % self.stride) / 2, ((self.imgsz - nh) % self.stride) / 2
        if (nw, nh) != (w, h):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, left, top

    def predict(self, img, conf=0.25, agnostic=False):
        """Trả về danh sách (x1, y1, x2, y2, độ tin cậy, chỉ số lớp) theo tọa độ ảnh gốc."""
        boxed, r, left, top = self.letterbox(img)
        blob = boxed[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        self.net.setInput(np.ascontiguousarray(blob))
        out = self.net.forward()[0].T  # (số ô, 4 + số lớp)
        scores = out[:, 4:]
        cls = scores.argmax(1)
        best = scores.max(1)
        keep = best > conf
        out, cls, best = out[keep], cls[keep], best[keep]
        if not len(out):
            return []
        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        xywh = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1)
        if agnostic:
            idx = cv2.dnn.NMSBoxes(xywh.tolist(), best.tolist(), conf, self.iou)
        else:
            idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), best.tolist(), cls.tolist(), conf, self.iou)
        idx = np.array(idx).reshape(-1)[: self.max_det]
        h, w = img.shape[:2]
        dets = []
        for i in idx:
            x, y, bw_, bh_ = xywh[i]
            x1 = float(np.clip((x - left) / r, 0, w))
            y1 = float(np.clip((y - top) / r, 0, h))
            x2 = float(np.clip((x + bw_ - left) / r, 0, w))
            y2 = float(np.clip((y + bh_ - top) / r, 0, h))
            dets.append((x1, y1, x2, y2, float(best[i]), int(cls[i])))
        return dets
