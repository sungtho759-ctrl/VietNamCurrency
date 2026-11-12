"""Nhận diện mệnh giá tiền Việt Nam theo thời gian thực từ webcam hoặc video.

Ba chế độ (đổi bằng phím 1 / 2 / 3 khi đang chạy):
  1. sift   – SIFT + RANSAC + màu trên toàn khung hình (classify_image): một tờ, khoảng 1–3 kết quả / giây.
  2. yolo   – YOLO11n tìm và nhận diện mọi tờ (khoảng 5–10 kết quả / giây trên CPU).
  3. hybrid – YOLO định vị từng tờ, SIFT + màu nhận diện mệnh giá của vùng cắt; nếu SIFT + màu không kết luận thì
              hiện "?" và thử lại sau 0,5 giây (--yolo-label: dùng nhãn YOLO như khi đánh giá). Mỗi tờ chỉ nhận
              diện lại khi mới xuất hiện hoặc sau --recheck giây. Mặc định nếu có ultralytics: trên tập valid F1
              94,5% so với 92,5% của yolo, sửa phần lớn lỗi nhầm tờ 100.000 và 500.000 đồng của YOLO
              (evaluate_hybrid.py). YOLO thử thêm khung hình xoay 180° và SIFT toàn khung hình khi không thấy tờ.

Khung hình được hiển thị ngay, việc nhận diện chạy ở luồng riêng trên khung hình mới nhất, nên hình không bị giật
dù nhận diện chậm. Kết quả được bám theo giữa các lần nhận diện (ghép khung theo IoU) và bỏ phiếu theo vài lần gần
nhất để nhãn không nhảy liên tục.

Phím: 1/2/3 đổi chế độ · Space tạm dừng · S lưu ảnh (results/realtime/) · Q hoặc Esc thoát.

Chạy:  python src/realtime.py                          # webcam 0, chế độ hybrid (hoặc sift nếu chưa có ultralytics)
       python src/realtime.py --mode yolo --camera 1
       python src/realtime.py --source video.mp4 --save out.mp4
       python src/realtime.py --source video.mp4 --no-show --sync --log log.csv   # chạy thử không cửa sổ
"""
import argparse
import csv
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from color_model import MODEL_PATH, ColorModel, hsv_hist
import evaluate_sift
from evaluate_sift import (MIN_INLIERS, RANSAC_THRESH, ROOT, UNKNOWN, candidates, classify_image, crop_box,
                           evidence_ok, fuse, load_templates, quad_angle_dev, resize_max)

sys.stdout.reconfigure(encoding="utf-8")

YOLO_WEIGHTS = ROOT / "results" / "yolo" / "yolo11n" / "weights" / "best.pt"
# Ngưỡng chọn trên tập valid khi đánh giá (results/yolo/evaluation_test/results.json, nhiều tờ: F1 cao nhất).
# Ngưỡng kép chống chập chờn: tờ mới cần >= YOLO_CONF, tờ đang bám theo chỉ cần >= YOLO_KEEP.
YOLO_CONF, YOLO_KEEP, YOLO_AGNOSTIC = 0.35, 0.25, True
# Khung YOLO yếu (YOLO_WEAK <= tin cậy < YOLO_KEEP): không hiển thị, chỉ dùng khi YOLO không có khung nào >= YOLO_KEEP –
# cắt vùng đó cho SIFT + màu trước khi phải chạy SIFT trên toàn khung hình. Tờ cầm tay sát camera, bị gập, ngược sáng
# YOLO thường chỉ cho 0,1–0,3 nhưng vẫn đúng vị trí; SIFT trên vùng cắt (độ phân giải gốc) đủ điểm hơn nhiều so với
# toàn khung hình thu về 640 px (ảnh webcam tờ 2.000: vùng cắt 45 cặp khớp 2.000 / 27 cặp 1.000, toàn khung hình 15 / 17
# -> báo nhầm 1.000). Chỉ nhận theo cổng bằng chứng chặt (>= EVIDENCE_MIN cặp khớp, không dựa vào nhãn YOLO).
YOLO_WEAK, WEAK_MAX = 0.10, 2
SIFT_MAX_SIDE = 640          # thu nhỏ khung hình trước khi trích SIFT để nhanh hơn
HYBRID_CROP_MAX = 400        # hybrid: thu vùng cắt về cạnh dài tối đa N px trước SIFT + màu (0 = giữ nguyên).
# Vùng cắt webcam lớn (~600 px) khớp nhiều chi tiết chung giữa các mệnh giá (chân dung, quốc huy) làm luật khoảng
# cách từ chối kết luận; 400 px giống cỡ vùng cắt khi dò tham số. Valid: F1 94,46% như vùng cắt gốc (YOLO 10 lớp),
# nhanh hơn ~35%; ảnh webcam tờ 100.000: đúng 7/7 thay vì 4/7 (evaluate_hybrid.py --crop-max 400).
# Luật khoảng cách theo inlier *riêng* (evaluate_sift.decide, exclusive=True): bỏ các điểm SIFT mà mẫu thắng và mẫu
# đối thủ cùng khớp vào (chân dung, quốc huy, quốc hiệu chung của mặt trước). Valid: F1 hybrid chặt 91,70% so với
# 91,65% (gần như không đổi, YOLO 10 lớp); bản ghi webcam (173 khung YOLO): đúng 69,9% so với 59,5%, "?" 20,8% so với 30,6%,
# sai 9,2% so với 9,8% (evaluate_hybrid.py --so-sanh-luat).
MARGIN_EXCLUSIVE = True
# Cổng bằng chứng (evaluate_sift.evidence_ok): camera có thể thấy vật không phải tiền – thẻ bảo hiểm y tế bị báo
# 2.000 đ, thẻ ATM bị báo 2.000 / 20.000 đ (YOLO coi thẻ là tờ tiền; SIFT khớp ngẫu nhiên 5–8 điểm; màu xanh của thẻ
# giống tờ 20.000 nên YOLO và màu cùng "đồng ý" sai). Chỉ báo mệnh giá khi có ứng viên SIFT cùng mệnh giá với khung
# homography gần chữ nhật (lệch <= EVIDENCE_ANGLE độ; khớp đúng: trung vị ~10°, khớp ngẫu nhiên trên thẻ: 64–85°) và
# >= EVIDENCE_MIN inlier, hoặc >= EVIDENCE_YOLO inlier khi YOLO cũng nói mệnh giá đó; kết quả chỉ theo màu không được
# nhận. 200 vùng cắt thẻ (5 ảnh x 40 biến thể): lọt 0 (không cổng: 39, cổng cũ ">= 8 inlier hoặc trùng YOLO": 12);
# bản ghi webcam: đúng 67,6% / sai 5,8% (cổng cũ 69,9% / 7,5%); valid F1 hybrid chặt 83,03% (cổng cũ 91,90%) vì tờ
# nhỏ chụp xa ít điểm khớp, trước đây dựa vào màu – nay hiện "?".
EVIDENCE_ANGLE, EVIDENCE_MIN, EVIDENCE_YOLO = 45.0, MIN_INLIERS, 4
# hybrid: vết mất dấu lâu hơn N giây rồi được tìm lại -> nhận diện lại ngay (có thể đã là tờ khác ở cùng chỗ)
REACQUIRE_GAP = 0.5
# CLAHE dự phòng: vùng cắt ra "?" thì trích SIFT lại sau khi tăng tương phản cục bộ (CLAHE, clipLimit này; so với bộ
# ảnh mẫu cũng đã tăng tương phản). Mặt sau tờ 20.000 dưới đèn phòng bị bạc màu, SIFT thường chỉ khớp 0–4 điểm. Luôn
# bật CLAHE làm SIFT chậm gần gấp đôi (nhiều điểm đặc trưng hơn) nên chỉ dùng khi cần. Chọn trên valid (cùng cổng bằng
# chứng ở trên, YOLO 10 lớp): F1 hybrid chặt valid / test – tắt 83,03% / 83,89%, luôn bật (clip 2) 84,73% / 84,13%,
# khi "?" 88,36% / 86,92% (evaluate_hybrid.py --clahe-du-phong 2); bản ghi webcam đúng / sai / "?" – tắt 67,6% / 5,8% / 26,6%, khi "?" 78,6% / 6,9% / 14,5%; 200 vùng
# cắt thẻ: lọt 0 ở mọi cách. 0 = tắt.
CLAHE_FALLBACK = 2.0
SIFT_FALLBACK_EVERY = 0.5    # hybrid: YOLO không thấy tờ nào -> thử SIFT toàn khung hình, tối đa mỗi N giây
SIFT_HOLD = 1.5              # hybrid: vết do SIFT toàn khung hình tìm ra được giữ N giây khi bị trượt
UNSURE_RECHECK = 0.5         # hybrid: SIFT + màu chưa kết luận -> hiện "?" và thử lại sau N giây
SIFT_PER_ROUND = 1           # hybrid: số tờ tối đa chạy SIFT + màu trong một vòng nhận diện
KNOWN_GRACE = 2.0            # hybrid: lần kiểm tra lại chưa rõ thì giữ mệnh giá nhận được gần nhất thêm N giây
UNSURE_COLOR = (150, 150, 150)
MODES = ("sift", "yolo", "hybrid")
COLORS = {"sift": (85, 157, 31), "yolo": (235, 99, 37), "hybrid": (160, 60, 200)}


def vnd(label):
    if label == UNKNOWN:
        return "?"
    return f"{int(label):,}".replace(",", ".") + " đ"


def quad_bbox(quad):
    q = np.float32(quad)
    x0, y0 = q.min(0)
    x1, y1 = q.max(0)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def overlap(a, b):
    """Diện tích giao / diện tích khung nhỏ hơn (1 = một khung nằm trọn trong khung kia)."""
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    small = min(a[2] * a[3], b[2] * b[3])
    return iw * ih / small if small > 0 else 0.0


def box_quad(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


# ------------------------------------------------------------------------------------------------ bộ nhận diện
class Recognizers:
    """Nạp mô hình một lần; process(frame, mode) trả về [{"label", "quad", "evidence"}] theo tọa độ khung hình."""

    def __init__(self, want_yolo=True, yolo_label=False):
        # yolo_label: hybrid, SIFT + màu không kết luận thì dùng nhãn YOLO (như khi đánh giá) thay vì hiện "?".
        # Mặc định tắt: khi thử bằng webcam, YOLO nhận sai tờ 100.000 ở cả 6/6 lần phải dùng nhãn YOLO.
        self.yolo_label = yolo_label
        self.templates = load_templates()
        self.templates_clahe = self._with_clahe(load_templates) if CLAHE_FALLBACK else None
        self.color = ColorModel() if MODEL_PATH.exists() else None
        self.yolo = None
        self.yolo_error = ""
        if want_yolo:
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(str(YOLO_WEIGHTS))
                self.yolo.predict(np.zeros((64, 64, 3), np.uint8), verbose=False, device="cpu")  # khởi tạo
            except Exception as exc:  # thiếu ultralytics/trọng số: vẫn chạy được chế độ sift
                self.yolo_error = str(exc).splitlines()[0]

    @property
    def modes(self):
        return MODES if self.yolo is not None else ("sift",)

    def sift(self, frame):
        small = resize_max(frame, SIFT_MAX_SIDE)
        s = frame.shape[1] / small.shape[1]
        label, inliers, quad, rect = classify_image(small, self.templates, self.color, inlier_rect=True,
                                                    exclusive=MARGIN_EXCLUSIVE)
        # Cổng bằng chứng (không có YOLO): >= EVIDENCE_MIN inlier và khung homography gần chữ nhật
        if label == UNKNOWN or quad is None or inliers < EVIDENCE_MIN or quad_angle_dev(quad) > EVIDENCE_ANGLE:
            return []
        # Vẽ khung bao các điểm khớp: khung homography ngoại suy từ ảnh mẫu vượt ra ngoài tờ bị gập / che một phần
        return [{"label": label, "quad": (np.float32(rect or quad) * s).tolist(), "conf": 1.0, "inliers": inliers,
                 "evidence": f"{inliers} khớp" if inliers else "màu"}]

    def _yolo(self, frame):
        """Khung YOLO tin cậy >= YOLO_WEAK (NMS chạy trên cả khung yếu: khung mạnh không bị khung yếu đè mất)."""
        r = self.yolo.predict(frame, conf=YOLO_WEAK, imgsz=640, agnostic_nms=YOLO_AGNOSTIC, device="cpu",
                              verbose=False)[0]
        return [(self.yolo.names[int(c)], [b[0], b[1], b[2] - b[0], b[3] - b[1]], p)
                for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]

    def yolo_boxes(self, frame, with_weak=False):
        """Khung YOLO >= YOLO_KEEP; không thấy tờ nào thì thử lại trên khung hình xoay 180° (YOLO hay bỏ sót tờ lộn
        ngược). with_weak=True: trả thêm các khung yếu (YOLO_WEAK..YOLO_KEEP) của cả hai chiều, khi không có khung mạnh."""
        H, W = frame.shape[:2]
        found = self._yolo(frame)
        strong = [x for x in found if x[2] >= YOLO_KEEP]
        weak = [x for x in found if x[2] < YOLO_KEEP]
        if not strong:
            flipped = [(lab, [W - b[0] - b[2], H - b[1] - b[3], b[2], b[3]], p)
                       for lab, b, p in self._yolo(cv2.rotate(frame, cv2.ROTATE_180))]
            strong = [x for x in flipped if x[2] >= YOLO_KEEP]
            weak += [x for x in flipped if x[2] < YOLO_KEEP]
        if not with_weak:
            return strong
        chosen = []
        for x in sorted(weak, key=lambda x: -x[2]):  # bỏ khung yếu trùng chỗ (cùng tờ thấy ở cả hai chiều)
            if all(iou(x[1], y[1]) < 0.5 for y in chosen):
                chosen.append(x)
        return strong, ([] if strong else chosen[:WEAK_MAX])

    def yolo_mode(self, frame):
        return [{"label": lab, "quad": box_quad(*box), "conf": p, "evidence": f"{p:.2f}"}
                for lab, box, p in self.yolo_boxes(frame)]

    @staticmethod
    def _with_clahe(fn, *args):
        """Chạy fn với SIFT trích trên ảnh đã tăng tương phản CLAHE (evaluate_sift.CLAHE_CLIP; chỉ luồng nhận diện
        gọi SIFT nên đổi biến toàn cục tạm thời là an toàn)."""
        evaluate_sift.CLAHE_CLIP = CLAHE_FALLBACK
        try:
            return fn(*args)
        finally:
            evaluate_sift.CLAHE_CLIP = 0.0

    def classify_crop(self, frame, box, yolo_label=None):
        """Trả về (mệnh giá hoặc UNKNOWN, số inlier, có dùng CLAHE dự phòng không)."""
        crop = crop_box(frame, box)
        if HYBRID_CROP_MAX:
            crop = resize_max(crop, HYBRID_CROP_MAX)
        rank = self.color.rank(hsv_hist(crop)) if self.color else None

        def attempt(templates):
            cands = candidates(crop, templates, (RANSAC_THRESH,))[RANSAC_THRESH]
            label, inliers = fuse(cands, rank, exclusive=MARGIN_EXCLUSIVE)
            # Cổng bằng chứng (xem EVIDENCE_ANGLE): cần ứng viên SIFT cùng mệnh giá, khung gần chữ nhật, đủ điểm khớp
            if label != UNKNOWN and not evidence_ok(cands, label, yolo_label, EVIDENCE_ANGLE, EVIDENCE_MIN,
                                                    EVIDENCE_YOLO):
                return UNKNOWN, inliers
            return label, inliers

        label, inliers = attempt(self.templates)
        if label == UNKNOWN and self.templates_clahe:
            label, inliers = self._with_clahe(attempt, self.templates_clahe)
            return label, inliers, True
        return label, inliers, False

    def hybrid(self, frame, tracker, recheck, now=None):
        """YOLO định vị; SIFT + màu nhận diện tờ mới hoặc đã lâu chưa kiểm tra lại.

        Kết quả SIFT + màu được lưu trong vết của tracker ("meta" = nhãn, bằng chứng, thời điểm kiểm tra, nhãn YOLO
        lúc đó, mệnh giá nhận được gần nhất, thời điểm nhận được). Nhãn hiển thị lấy thẳng từ kết quả SIFT gần nhất
        (không bỏ phiếu trên kết quả đã lưu); lần kiểm tra lại "chưa rõ" chỉ giữ mệnh giá cũ thêm KNOWN_GRACE giây.
        """
        now = time.perf_counter() if now is None else now
        out = []
        boxes, weak = self.yolo_boxes(frame, with_weak=True)
        if not boxes:
            return self.sift_fallback(frame, now, weak)
        weak_new, strong = [], []

        def reacquired(track):
            """Vết mất dấu lâu hơn REACQUIRE_GAP giây rồi được tìm lại: có thể đã là tờ khác ở cùng chỗ."""
            return track is not None and now - track["seen"] > REACQUIRE_GAP

        for lab, box, p in boxes:
            track = tracker.peek(box)
            if track is None and p < YOLO_CONF:
                weak_new.append((lab, box, p))  # khung yếu, tờ mới: chỉ nhận nếu SIFT xác nhận (bên dưới)
                continue
            meta = track.get("meta") if track else None
            # Cần nhận diện lại khi: tờ mới, đang "?", vết vừa mất dấu rồi được tìm lại hoặc YOLO (đủ tin cậy) đổi mệnh
            # giá so với lúc lưu (dấu hiệu tờ khác vừa thế chỗ – YOLO có thể gọi hai tờ khác nhau cùng một tên), hoặc
            # quá --recheck giây. Số thứ tự = mức ưu tiên.
            if meta is None:
                need = 0
            elif meta[0] == UNKNOWN and now - meta[2] >= recheck:
                need = 1
            elif reacquired(track) or (p >= YOLO_CONF and lab != meta[3]):
                need = 2
            elif now - meta[2] >= recheck:
                need = 3
            else:
                need = None
            strong.append((lab, box, p, track, meta, need))

        # Mỗi vòng chỉ chạy SIFT + màu cho tối đa SIFT_PER_ROUND tờ (ưu tiên tờ mới / "?"), để vòng nhận diện ngắn
        # và vị trí các tờ được YOLO cập nhật thường xuyên; tờ chưa đến lượt dùng kết quả đã lưu hoặc tạm hiện "?".
        order = sorted((i for i, x in enumerate(strong) if x[5] is not None), key=lambda i: strong[i][5])
        chosen = set(order[:SIFT_PER_ROUND])
        for i, (lab, box, p, track, meta, need) in enumerate(strong):
            reacq = reacquired(track)
            if i in chosen:
                pred, inliers, used_clahe = self.classify_crop(frame, box, lab)
                if pred == UNKNOWN and self.yolo_label:
                    pred, ev = lab, f"YOLO {p:.2f}"
                else:
                    ev = ((f"{inliers} khớp" if inliers else "màu") + (", CLAHE" if used_clahe else "")
                          if pred != UNKNOWN else "chưa rõ")
                # Mệnh giá nhận được gần nhất (giữ thêm KNOWN_GRACE giây khi lần này chưa rõ) – trừ khi vết vừa được
                # tìm lại sau khi mất dấu: mệnh giá cũ có thể thuộc tờ khác
                last = ((pred, now) if pred != UNKNOWN else (meta[4], meta[5]) if meta and not reacq
                        else (None, -1e9))
                # Chưa kết luận: thử lại sớm (sau UNSURE_RECHECK giây)
                t_check = now if pred != UNKNOWN else now - recheck + UNSURE_RECHECK
                meta = (pred, ev, t_check, lab) + last
                reacq = False
            det = {"quad": box_quad(*box), "conf": p, "vote": True, "reset": True}
            if meta is None or reacq:
                det.update(label=UNKNOWN, evidence="đang nhận diện")
            elif meta[0] != UNKNOWN:
                det.update(label=meta[0], evidence=meta[1])
            elif meta[4] is not None and now - meta[5] <= KNOWN_GRACE:
                # Lần kiểm tra gần nhất chưa rõ (nhòe, nghiêng) nhưng vừa nhận được mệnh giá cách đây chưa lâu
                det.update(label=meta[4], evidence="đang kiểm tra lại")
            else:
                det.update(label=UNKNOWN, evidence=meta[1])
            if i in chosen:
                det["meta"] = meta
            out.append(det)
        # Khung YOLO yếu (< YOLO_CONF) chưa thuộc vết nào: SIFT + màu kiểm tra (giới hạn tần suất), chỉ nhận khi
        # SIFT đủ cặp điểm khớp – YOLO hay kém tin cậy với tờ cầm nghiêng, ở gần camera.
        if weak_new and len(chosen) < SIFT_PER_ROUND and self._fallback_due(now):
            for lab, box, p in weak_new:
                pred, inliers, _ = self.classify_crop(frame, box, lab)
                if pred != UNKNOWN and inliers >= MIN_INLIERS:
                    out.append({"quad": box_quad(*box), "conf": YOLO_CONF, "vote": True, "reset": True, "label": pred,
                                "evidence": f"{inliers} khớp",
                                "meta": (pred, f"{inliers} khớp", now, lab, pred, now)})
        return out

    def _fallback_due(self, now):
        if now - getattr(self, "_last_fallback", -1e9) < SIFT_FALLBACK_EVERY:
            return False
        self._last_fallback = now
        return True

    def sift_fallback(self, frame, now, weak=()):
        """YOLO không có khung nào >= YOLO_KEEP: thử SIFT + màu trên các khung YOLO yếu (xem YOLO_WEAK), không được thì
        SIFT toàn khung hình (bất biến với góc xoay). Chỉ nhận khi đủ cặp điểm khớp (không nhận kết luận chỉ dựa vào
        màu – tránh "thấy" tờ tiền trong cảnh trống)."""
        if not self._fallback_due(now):
            return []
        dets = []
        for lab, box, p in weak:
            # Không truyền nhãn YOLO: khung yếu không đủ tin để hạ ngưỡng cổng bằng chứng xuống EVIDENCE_YOLO
            pred, inliers, used_clahe = self.classify_crop(frame, box)
            if pred != UNKNOWN and inliers >= MIN_INLIERS:
                dets.append({"label": pred, "quad": box_quad(*box), "conf": YOLO_CONF, "inliers": inliers,
                             "evidence": f"{inliers} khớp" + (", CLAHE" if used_clahe else "") + f", YOLO yếu {p:.2f}",
                             "hold": SIFT_HOLD})
        if dets:
            # Hai khung yếu (một to, một nhỏ) có thể cùng chứa một tờ: giữ khung nhiều cặp khớp hơn
            kept = []
            for d in sorted(dets, key=lambda d: -d["inliers"]):
                if all(overlap(quad_bbox(d["quad"]), quad_bbox(k["quad"])) < 0.8 for k in kept):
                    kept.append(d)
            return kept
        dets = self.sift(frame)
        # Vết do SIFT dự phòng tìm ra chỉ được xác nhận lại mỗi SIFT_FALLBACK_EVERY giây: giữ lâu hơn để không chớp
        return [dict(d, evidence=d["evidence"] + ", SIFT", hold=SIFT_HOLD) for d in dets if d["inliers"] >= MIN_INLIERS]

    def process(self, frame, mode, tracker, recheck=2.0, now=None):
        if mode == "yolo":
            return self.yolo_mode(frame)
        if mode == "hybrid":
            return self.hybrid(frame, tracker, recheck, now)
        return self.sift(frame)


# ------------------------------------------------------------------------------------------------ bám vết
class Tracker:
    """Ghép kết quả các lần nhận diện liên tiếp theo IoU, bỏ phiếu nhãn trên `votes` lần gần nhất.

    Chống chập chờn: tờ mới chỉ được tạo từ khung có độ tin cậy >= new_conf; khung yếu hơn chỉ giữ vết (cập nhật
    vị trí, không bỏ phiếu nhãn); vết chỉ bị xóa khi bị trượt >= max_missed lần nhận diện liên tiếp và quá `hold` giây.
    Trường tùy chọn của một kết quả: "vote" (có bỏ phiếu không; mặc định conf >= new_conf), "meta" (dữ liệu lưu
    kèm vết), "reset" (xóa các phiếu cũ, khi tờ ở vị trí này được nhận diện lại thành mệnh giá khác).
    """

    def __init__(self, votes=7, hold=0.6, min_iou=0.3, new_conf=YOLO_CONF, max_missed=2):
        self.votes, self.hold, self.min_iou, self.new_conf = votes, hold, min_iou, new_conf
        self.max_missed = max_missed
        self.tracks = []  # {"id", "bbox", "quad", "labels": deque, "evidence", "seen", "meta"}
        self.next_id = 0

    def peek(self, bbox):
        """Vết khớp nhất với khung (không cập nhật), None nếu là tờ mới."""
        best = max(self.tracks, key=lambda t: iou(t["bbox"], bbox), default=None)
        return best if best and iou(best["bbox"], bbox) >= self.min_iou else None

    def update(self, dets, now=None):
        now = time.perf_counter() if now is None else now
        used = set()
        pairs = sorted(((iou(t["bbox"], quad_bbox(d["quad"])), i, j) for i, t in enumerate(self.tracks)
                        for j, d in enumerate(dets)), reverse=True)
        matched_t = set()
        for v, i, j in pairs:
            if v < self.min_iou:
                break
            if i in matched_t or j in used:
                continue
            matched_t.add(i), used.add(j)
            t, d = self.tracks[i], dets[j]
            t.update(bbox=quad_bbox(d["quad"]), quad=d["quad"], seen=now, missed=0, hold=d.get("hold", self.hold))
            if d.get("meta") is not None:
                t["meta"] = d["meta"]
            if d.get("reset"):
                t["labels"].clear()
            if d.get("vote", d.get("conf", 1.0) >= self.new_conf):
                t["labels"].append(d["label"])
                t["evidence"] = d["evidence"]
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t["missed"] += 1
        # Vết bị trượt nằm chồng lên vết vừa được cập nhật (tờ xoay làm khung mới không khớp đủ IoU với khung cũ)
        # là vết trùng của cùng một tờ: bỏ.
        fresh = [self.tracks[i]["bbox"] for i in matched_t] + [
            quad_bbox(d["quad"]) for j, d in enumerate(dets) if j not in used and d.get("conf", 1.0) >= self.new_conf]
        self.tracks = [t for i, t in enumerate(self.tracks)
                       if i in matched_t or not any(overlap(t["bbox"], b) >= 0.5 for b in fresh)]
        # Xóa vết khi bộ nhận diện đã không thấy nó ít nhất max_missed vòng liên tiếp VÀ quá `hold` giây: vòng
        # nhận diện chậm (nhiều tờ, SIFT toàn khung) không làm khung tự tắt khi tờ vẫn còn trong hình.
        self.tracks = [t for t in self.tracks if t["missed"] < self.max_missed or now - t["seen"] <= t["hold"]]
        for j, d in enumerate(dets):
            if j not in used and d.get("conf", 1.0) >= self.new_conf:
                self.tracks.append({"id": self.next_id, "bbox": quad_bbox(d["quad"]), "quad": d["quad"],
                                    "labels": deque([d["label"]], maxlen=self.votes), "evidence": d["evidence"],
                                    "seen": now, "meta": d.get("meta"), "missed": 0,
                                    "hold": d.get("hold", self.hold)})
                self.next_id += 1

    def reset(self):
        self.tracks = []

    def visible(self, now=None):
        """Các vết đang hiển thị: (id, nhãn theo đa số phiếu, khung, bằng chứng)."""
        now = time.perf_counter() if now is None else now
        out = []
        for t in self.tracks:
            if not t["labels"]:
                continue
            known = [lab for lab in t["labels"] if lab != UNKNOWN]  # "?" chỉ hiện khi chưa có nhãn nào
            label = Counter(known).most_common(1)[0][0] if known else UNKNOWN
            out.append((t["id"], label, t["quad"], t["evidence"]))
        return out


# ------------------------------------------------------------------------------------------------ vẽ
def load_font(size):
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "segoeui.ttf", "arial.ttf"):
        for base in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/Library/Fonts")):
            if (base / name).exists():
                return ImageFont.truetype(str(base / name), size)
    return ImageFont.load_default()


class Painter:
    """Vẽ kết quả lên khung hình. Khung vẽ bằng OpenCV; chữ có dấu cần PIL nên mỗi dòng chữ chỉ được dựng một lần
    thành ảnh nhỏ rồi dùng lại (vẽ PIL trên cả khung hình ở mỗi khung tốn ~10 ms và tranh GIL với luồng nhận diện)."""

    def __init__(self):
        self.font = load_font(20)
        self.small = load_font(15)
        self.cache = {}

    def sprite(self, text, font, fg, bg, pad=(4, 2)):
        key = (text, id(font), fg, bg)
        img = self.cache.get(key)
        if img is None:
            x0, y0, x1, y1 = font.getbbox(text)
            im = Image.new("RGB", (x1 + 2 * pad[0], max(y1, 1) + 2 * pad[1] + 2), tuple(reversed(bg)))
            ImageDraw.Draw(im).text((pad[0], pad[1]), text, font=font, fill=tuple(reversed(fg)))
            img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            if len(self.cache) > 512:
                self.cache.clear()
            self.cache[key] = img
        return img

    @staticmethod
    def paste(dst, src, x, y):
        H, W = dst.shape[:2]
        h, w = src.shape[:2]
        x, y = int(max(0, min(x, W - 1))), int(max(0, min(y, H - 1)))
        w, h = min(w, W - x), min(h, H - y)
        dst[y:y + h, x:x + w] = src[:h, :w]

    def draw(self, frame, items, mode, info, paused=False):
        img = frame.copy()
        color = COLORS[mode]
        H = img.shape[0]
        for _, label, quad, ev in items:
            c = UNSURE_COLOR if label == UNKNOWN else color
            cv2.polylines(img, [np.int32(quad)], True, c, 3)
            x, y = np.float32(quad).min(0)
            tag = self.sprite(f"{vnd(label)}  ({ev})", self.small, (255, 255, 255), c)
            ty = int(y) - tag.shape[0] - 2
            if ty < 62:  # sát mép trên: thanh tiêu đề sẽ che nhãn, vẽ vào trong khung
                ty = max(int(y), 0) + 64 if int(y) < 62 else int(y) + 4
            self.paste(img, tag, x, ty)
        known = [label for _, label, _, _ in items if label != UNKNOWN]
        total = sum(int(label) for label in known)
        unsure = len(items) - len(known)
        head = (f"{len(known)} tờ – tổng {vnd(total)}" + (f" (+{unsure} chưa rõ)" if unsure else "")
                if items else "Chưa thấy tờ tiền")
        sub = f"Chế độ {mode} · {info}" + (" · TẠM DỪNG" if paused else "")
        a = self.sprite(head, self.font, (255, 255, 255), (20, 20, 20), (10, 4))
        b = self.sprite(sub, self.small, (200, 200, 200), (20, 20, 20), (10, 2))
        w = max(a.shape[1], b.shape[1])
        cv2.rectangle(img, (0, 0), (w, a.shape[0] + b.shape[0]), (20, 20, 20), -1)
        self.paste(img, a, 0, 0)
        self.paste(img, b, 0, a.shape[0])
        hint = self.sprite("1 sift · 2 yolo · 3 hybrid · Space dừng · S lưu · Q thoát", self.small,
                           (200, 200, 200), (20, 20, 20), (8, 3))
        self.paste(img, hint, 0, H - hint.shape[0])
        return img


# ------------------------------------------------------------------------------------------------ luồng nhận diện
class Worker(threading.Thread):
    """Luôn nhận diện khung hình mới nhất; khung đến khi đang bận thì bị bỏ qua (không dồn hàng đợi)."""

    def __init__(self, rec, tracker, mode, recheck):
        super().__init__(daemon=True)
        self.rec, self.tracker, self.mode, self.recheck = rec, tracker, mode, recheck
        self.lock = threading.Lock()
        self.frame = None
        self.new = threading.Event()
        self.stopped = False
        self.latency = deque(maxlen=10)
        self.done = deque(maxlen=100)  # thời điểm kết thúc các vòng nhận diện gần đây
        self.error = ""

    def submit(self, frame):
        with self.lock:
            self.frame = frame
        self.new.set()

    def set_mode(self, mode):
        with self.lock:
            self.mode = mode
            self.tracker.reset()

    def run(self):
        while not self.stopped:
            if not self.new.wait(0.1):
                continue
            with self.lock:
                frame, mode = self.frame, self.mode
                self.new.clear()
            start = time.perf_counter()
            try:
                dets = self.rec.process(frame, mode, self.tracker, self.recheck)
            except Exception as exc:
                self.error = str(exc)
                continue
            with self.lock:
                if mode == self.mode:  # bỏ kết quả của chế độ cũ nếu vừa đổi chế độ
                    self.tracker.update(dets)
            self.latency.append(time.perf_counter() - start)
            self.done.append(time.perf_counter())

    def snapshot(self):
        with self.lock:
            return self.tracker.visible()

    def rate(self):
        """Số vòng nhận diện mỗi giây trong 5 giây gần nhất (chạy --sync: theo thời gian xử lý từng khung)."""
        now = time.perf_counter()
        recent = [t for t in self.done if now - t <= 5.0]
        if len(recent) >= 2:
            return (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6)
        return (len(self.latency) / sum(self.latency)) if self.latency else 0.0


def open_source(args):
    src = args.camera if args.source is None else args.source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) and sys.platform == "win32" \
        else cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Không mở được nguồn video: {src}")
    return cap, isinstance(src, int)


def main():
    parser = argparse.ArgumentParser(description="Nhận diện mệnh giá tiền Việt Nam theo thời gian thực")
    parser.add_argument("--camera", type=int, default=0, help="Chỉ số webcam (mặc định 0)")
    parser.add_argument("--source", default=None, help="Tệp video thay cho webcam")
    parser.add_argument("--mode", choices=MODES, default=None, help="Mặc định: hybrid nếu có ultralytics, ngược lại sift")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--recheck", type=float, default=2.0, help="hybrid: nhận diện lại một tờ sau bao nhiêu giây")
    parser.add_argument("--votes", type=int, default=7, help="Số lần nhận diện gần nhất dùng để bỏ phiếu nhãn")
    parser.add_argument("--hold", type=float, default=0.6, help="Giữ khung thêm bao nhiêu giây khi bị trượt")
    parser.add_argument("--yolo-label", action="store_true",
                        help="hybrid: SIFT + màu không kết luận thì dùng nhãn YOLO thay vì hiện \"?\"")
    parser.add_argument("--save", default="", help="Ghi video kết quả ra tệp .mp4")
    parser.add_argument("--record", default="",
                        help="Ghi hình gốc (không vẽ kết quả) ra tệp .mp4 để chạy lại sau bằng --source")
    parser.add_argument("--no-show", action="store_true", help="Không mở cửa sổ (chạy thử / máy chủ)")
    parser.add_argument("--sync", action="store_true",
                        help="Nhận diện mọi khung hình ngay tại chỗ (không bỏ khung) – dùng để chạy thử với video")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--log", default="", help="Ghi kết quả từng khung hình ra CSV")
    args = parser.parse_args()

    print("Đang nạp ảnh mẫu, mô hình màu" + ("" if args.mode == "sift" else " và YOLO") + "...", flush=True)
    rec = Recognizers(want_yolo=args.mode != "sift", yolo_label=args.yolo_label)
    if args.mode in ("yolo", "hybrid") and rec.yolo is None:
        sys.exit(f"Không dùng được YOLO ({rec.yolo_error}). Cài: pip install -r requirements_yolo.txt")
    mode = args.mode or ("hybrid" if rec.yolo is not None else "sift")
    if rec.yolo is None and args.mode is None:
        print(f"Không có YOLO ({rec.yolo_error}) – chỉ dùng chế độ sift.")

    cap, live = open_source(args)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    tracker = Tracker(votes=args.votes, hold=args.hold)
    worker = Worker(rec, tracker, mode, args.recheck)
    if not args.sync:
        worker.start()
    painter = Painter()
    writer = None
    recorder = None
    log = None
    if args.log:
        log_f = open(args.log, "w", newline="", encoding="utf-8")
        log = csv.writer(log_f)
        log.writerow(["frame", "mode", "so_to", "nhan", "tong_tien"])
    snap_dir = ROOT / "results" / "realtime"
    win = "Nhan dien tien Viet Nam (thoi gian thuc)"
    paused, frame, shown = False, None, deque(maxlen=30)
    n = 0
    print(f"Bắt đầu ở chế độ {mode}. Phím: 1/2/3 đổi chế độ, Space tạm dừng, S lưu ảnh, Q thoát.", flush=True)

    while True:
        t0 = time.perf_counter()
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if args.record:
                if recorder is None:
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    recorder = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20 if live else fps_src,
                                               (frame.shape[1], frame.shape[0]))
                recorder.write(frame)
            if args.sync:
                start = time.perf_counter()
                tracker.update(rec.process(frame, mode, tracker, args.recheck, now=n / fps_src), now=n / fps_src)
                worker.latency.append(time.perf_counter() - start)
            else:
                worker.submit(frame)
        items = tracker.visible(now=n / fps_src) if args.sync else worker.snapshot()
        shown.append(time.perf_counter())
        disp_fps = (len(shown) - 1) / (shown[-1] - shown[0]) if len(shown) > 1 and shown[-1] > shown[0] else 0
        rate = worker.rate()
        info = f"nhận diện {rate:.1f}/s" + ("" if args.sync else f" · hiển thị {disp_fps:.0f} fps")
        if worker.error:
            info += f" · lỗi: {worker.error[:40]}"
        out = painter.draw(frame, items, mode, info, paused)

        if log is not None:
            labels = sorted((lab for _, lab, _, _ in items if lab != UNKNOWN), key=int)
            labels += ["?"] * (len(items) - len(labels))
            log.writerow([n, mode, len(items), " ".join(labels), sum(int(x) for x in labels if x != "?")])
        if args.save:
            if writer is None:
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_src,
                                         (out.shape[1], out.shape[0]))
            writer.write(out)
        if args.max_frames and n >= args.max_frames:
            break
        if args.no_show:
            if not live and not args.sync:  # video từ tệp: phát đúng tốc độ để mô phỏng webcam (bỏ khung khi bận)
                time.sleep(max(0.0, 1 / fps_src - (time.perf_counter() - t0)))
            continue

        cv2.imshow(win, out)
        # Video từ tệp: giữ đúng tốc độ phát; webcam: đọc càng nhanh càng tốt
        wait = 1 if live or args.sync else max(1, int(1000 / fps_src - 1000 * (time.perf_counter() - t0)))
        key = cv2.waitKey(wait) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("s"):
            snap_dir.mkdir(parents=True, exist_ok=True)
            path = snap_dir / time.strftime("snapshot_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(str(path), out)
            print("Đã lưu", path.relative_to(ROOT))
        elif key in (ord("1"), ord("2"), ord("3")):
            new = MODES[key - ord("1")]
            if new not in rec.modes:
                print("Chế độ này cần YOLO: pip install -r requirements_yolo.txt")
            elif new != mode:
                mode = new
                worker.set_mode(mode)
                print("Chế độ:", mode)

    worker.stopped = True
    cap.release()
    for w in (writer, recorder):
        if w is not None:
            w.release()
    if log is not None:
        log_f.close()
    cv2.destroyAllWindows()
    print(f"Đã xử lý {n} khung hình; nhận diện trung bình {worker.rate():.1f} lần / giây ở chế độ {mode}.")


if __name__ == "__main__":
    main()
