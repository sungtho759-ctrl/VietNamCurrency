"""Đánh giá phương pháp SIFT + RANSAC (như nhan_dien.py) trên tập dữ liệu có nhãn COCO.

Hai cách đánh giá:
  - Theo tờ tiền: cắt từng bounding box có nhãn rồi nhận diện vùng cắt đó.
  - Theo ảnh: chỉ dùng ảnh có đúng một tờ tiền, nhận diện trên cả ảnh.

Chạy:  python src/danh_gia_sift.py --split test
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # chỉ lưu hình ra file, tránh lỗi backend Tk khi chạy lâu
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import sys

from mau_sac import ColorModel, hsv_hist

# In được tiếng Việt cả khi chạy trong IDE hoặc chuyển hướng đầu ra trên Windows
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = "khong_xd"

# Cấu hình "cân bằng mới" (SIFT + màu): accuracy cao nhất trên tập valid với precision >= 95%
# (xem tinh_chinh_nguong.py). Ngưỡng ban đầu trước khi tinh chỉnh: RANSAC 5 px, 12 inlier,
# tỷ lệ 0.5, diện tích 3, không xét khoảng cách, không dùng màu.
RANSAC_THRESH = 12.0
MIN_INLIERS = 8
MIN_RATIO = 0.0
MAX_AREA = 10.0
MIN_MARGIN = 1.3
# Ảnh mẫu bổ sung lấy từ tập train (tao_mau.py) và kết hợp màu HSV (mau_sac.py); 0 = tắt
K_TRAIN = 0          # số ảnh mẫu train mỗi mệnh giá
COLOR_VETO = 0       # kết quả SIFT phải nằm trong top-k mệnh giá theo màu
AGREE_MIN = 5        # SIFT chưa đủ ngưỡng nhưng có >= n inlier và trùng mệnh giá màu xếp đầu -> nhận
COLOR_FALLBACK = 0.8 # SIFT không kết luận, màu đủ tách biệt (d1/d2 <= giá trị này) -> lấy theo màu
# Nhận diện nhiều tờ (detect_all): phóng to ảnh trước khi trích SIFT, chọn trên tập valid (danh_gia_nhieu_to.py)
MULTI_UPSCALE = 1.5

sift = cv2.SIFT_create()
matcher = cv2.BFMatcher(cv2.NORM_L2)


def resize_max(img, max_side):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def extract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray, None)


def load_templates(k_train=K_TRAIN):
    """Ảnh mẫu gốc trong data/templates + k_train ảnh mẫu đầu tiên mỗi mệnh giá trong data/templates_train."""
    paths = sorted((ROOT / "data" / "templates").iterdir())
    train_dir = ROOT / "data" / "templates_train"
    if k_train and train_dir.exists():
        paths += sorted(p for p in train_dir.glob("*.jpg") if int(p.stem.split("_t")[1]) < k_train)
    templates = []
    for path in paths:
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label = path.stem.split("_")[0]
        if not label.isdigit():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        # Hai ảnh mẫu rất lớn (~2000px), đưa về cùng cỡ với các mẫu còn lại
        img = resize_max(img, 600)
        kp, des = extract(img)
        if des is None:
            continue
        templates.append({"label": label, "name": path.name, "kp": kp, "des": des,
                          "shape": img.shape[:2]})
    return templates


def candidates(query, templates, ransac_thresholds=(5.0,)):
    """Đối sánh query với từng ảnh mẫu, trả về các giả thuyết hợp lệ về hình học.

    Kết quả: {ngưỡng RANSAC: [(nhãn, số inlier, tỷ lệ inlier, diện tích khung / diện tích ảnh, tên mẫu, 4 góc khung), ...]}.
    Tách riêng bước này để dò ngưỡng mà không phải chạy lại SIFT.
    """
    result = {thr: [] for thr in ransac_thresholds}
    kp_query, des_query = extract(query)
    if des_query is None or len(des_query) < 2:
        return result

    qh, qw = query.shape[:2]

    for t in templates:
        pairs = matcher.knnMatch(t["des"], des_query, k=2)

        unique = {}
        for pair in pairs:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                if m.trainIdx not in unique or m.distance < unique[m.trainIdx].distance:
                    unique[m.trainIdx] = m

        matches = list(unique.values())
        if len(matches) < 4:  # homography cần tối thiểu 4 cặp điểm
            continue

        src = np.float32([t["kp"][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_query[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        for thr in ransac_thresholds:
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thr)
            if H is None or mask is None:
                continue

            h, w = t["shape"]
            corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(corners, H)
            if not np.isfinite(projected).all():
                continue

            # Loại khung không lồi hoặc quá nhỏ; ngưỡng diện tích tối đa xét ở decide()
            polygon = projected.astype(np.float32)
            if not cv2.isContourConvex(polygon):
                continue
            area_ratio = abs(cv2.contourArea(polygon)) / (qh * qw)
            if area_ratio < 0.01:
                continue

            count = int(mask.sum())
            result[thr].append((t["label"], count, count / len(matches), area_ratio, t["name"],
                                projected.reshape(-1, 2).round(1).tolist()))

    return result


def decide(cands, min_inliers, min_ratio, max_area=3.0, min_margin=1.0):
    """Chọn mẫu có nhiều inlier nhất trong các giả thuyết vượt ngưỡng.

    min_margin: mẫu thắng phải có số inlier >= min_margin lần mẫu tốt nhất của
    mệnh giá khác; nếu không (hai mệnh giá khớp sát nút) thì trả về không xác định.
    1.0 nghĩa là tắt luật này.
    """
    valid = [c for c in cands if c[3] <= max_area]
    passed = [c for c in valid if c[1] >= min_inliers and c[2] >= min_ratio]
    if not passed:
        return UNKNOWN, 0
    label, count = max(passed, key=lambda c: (c[1], c[2]))[:2]
    rival = max((c[1] for c in valid if c[0] != label), default=0)
    if min_margin > 1.0 and count < min_margin * rival:
        return UNKNOWN, 0
    return label, count


def fuse(cands, color_rank, min_inliers=MIN_INLIERS, min_ratio=MIN_RATIO, max_area=MAX_AREA,
         min_margin=MIN_MARGIN, veto_k=COLOR_VETO, agree_min=AGREE_MIN, fallback=COLOR_FALLBACK):
    """Kết hợp quyết định SIFT với xếp hạng màu [(nhãn, khoảng cách), ...] (None = chỉ dùng SIFT)."""
    label, count = decide(cands, min_inliers, min_ratio, max_area, min_margin)
    if color_rank is None:
        return label, count
    order = [c for c, _ in color_rank]

    if label != UNKNOWN:
        # Màu phủ quyết: mệnh giá SIFT chọn có màu quá khác thì không kết luận
        if veto_k and label not in order[:veto_k]:
            return UNKNOWN, 0
        return label, count

    # SIFT chưa đủ bằng chứng: chấp nhận ứng viên SIFT tốt nhất nếu trùng mệnh giá màu xếp đầu
    if agree_min:
        weak = [c for c in cands if c[3] <= max_area and c[1] >= agree_min]
        if weak:
            top = max(weak, key=lambda c: (c[1], c[2]))
            if top[0] == order[0]:
                return top[0], top[1]

    # Không có bằng chứng SIFT: chỉ dùng màu khi mệnh giá xếp đầu tách biệt rõ
    if fallback and color_rank[0][1] <= fallback * color_rank[1][1]:
        return order[0], 0
    return UNKNOWN, 0


def classify(query, templates, min_inliers=MIN_INLIERS, min_ratio=MIN_RATIO, ransac=RANSAC_THRESH,
             max_area=MAX_AREA, min_margin=MIN_MARGIN, color_rank=None,
             veto_k=COLOR_VETO, agree_min=AGREE_MIN, fallback=COLOR_FALLBACK):
    """Chọn mẫu có nhiều cặp hợp lệ sau RANSAC nhất, kết hợp màu nếu có color_rank."""
    cands = candidates(query, templates, (ransac,))[ransac]
    return fuse(cands, color_rank, min_inliers, min_ratio, max_area, min_margin, veto_k, agree_min, fallback)


def locate(cands, max_area=MAX_AREA):
    """Giả thuyết SIFT mạnh nhất (kể cả chưa đủ ngưỡng) dùng để định vị tờ tiền trên cả ảnh."""
    valid = [c for c in cands if c[3] <= max_area]
    return max(valid, key=lambda c: (c[1], c[2]), default=None)


def classify_image(img, templates, color_model=None, ransac=RANSAC_THRESH, **params):
    """Nhận diện trên cả ảnh: SIFT định vị tờ tiền, màu lấy trong khung homography của giả thuyết mạnh nhất.

    Trả về (nhãn, số inlier, 4 góc khung hoặc None).
    """
    cands = candidates(img, templates, (ransac,))[ransac]
    top = locate(cands, params.get("max_area", MAX_AREA))
    rank = None
    if color_model is not None and top is not None:
        H, W = img.shape[:2]
        x0, y0 = np.clip(np.min(top[5], axis=0), 0, [W - 1, H - 1]).astype(int)
        x1, y1 = np.clip(np.max(top[5], axis=0), 0, [W - 1, H - 1]).astype(int)
        if x1 - x0 > 10 and y1 - y0 > 10:
            rank = color_model.rank(hsv_hist(img[y0:y1, x0:x1]))
    label, count = fuse(cands, rank, **params)
    if label == UNKNOWN:
        return label, count, None
    box = max((c for c in cands if c[0] == label and c[3] <= params.get("max_area", MAX_AREA)),
              key=lambda c: (c[1], c[2]), default=None)
    return label, count, box[5] if box else None


def quad_iou(a, b):
    """IoU của hai tứ giác lồi."""
    a, b = np.float32(a), np.float32(b)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - inter
    return inter / union if union > 0 else 0.0


def detect_all(img, templates, color_model=None, ransac=RANSAC_THRESH, min_inliers=MIN_INLIERS,
               min_ratio=MIN_RATIO, min_margin=MIN_MARGIN, veto_k=COLOR_VETO, agree_min=AGREE_MIN,
               fallback=COLOR_FALLBACK, max_notes=10, max_area_note=1.5, same_place_iou=0.3, upscale=1.0):
    """Nhận diện nhiều tờ trong một ảnh bằng cách lặp SIFT + RANSAC.

    Mỗi vòng: đối sánh các điểm SIFT còn lại với mọi ảnh mẫu, lấy giả thuyết homography mạnh nhất làm
    vị trí một tờ, quyết định mệnh giá bằng các ứng viên ở cùng vị trí (luật khoảng cách chỉ so với các
    mệnh giá khớp vào cùng chỗ) kết hợp màu trong khung, rồi loại các điểm SIFT nằm trong khung đó.
    Dừng khi giả thuyết mạnh nhất không còn đủ inlier.

    upscale: phóng to ảnh trước khi trích SIFT để tờ nhỏ có thêm điểm đặc trưng; khung trả về theo
    tọa độ ảnh gốc.

    Trả về [(nhãn, số inlier, 4 góc khung), ...].
    """
    if upscale != 1.0:
        big = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        found = detect_all(big, templates, color_model, ransac, min_inliers, min_ratio, min_margin, veto_k,
                           agree_min, fallback, max_notes, max_area_note, same_place_iou, 1.0)
        return [(lab, n, (np.float32(box) / upscale).tolist()) for lab, n, box in found]
    kp, des = extract(img)
    if des is None or len(des) < 4:
        return []
    H, W = img.shape[:2]
    pts = np.float32([k.pt for k in kp])
    active = np.ones(len(kp), bool)
    weakest = min(min_inliers, agree_min) if agree_min else min_inliers
    found = []

    # Đối sánh một lần với K láng giềng gần nhất; mỗi vòng sau chỉ cần lấy 2 láng giềng đầu còn hoạt động
    # (vòng đầu cho kết quả y hệt đối sánh k = 2 thông thường), tránh đối sánh lại toàn bộ ở mỗi vòng.
    K = 6
    neigh = []
    for t in templates:
        nn = matcher.knnMatch(t["des"], des, k=K)
        rows = [r for r in nn if len(r) == K]
        if len(rows) < 4:
            continue
        neigh.append((t, np.int32([r[0].queryIdx for r in rows]),
                      np.int32([[m.trainIdx for m in r] for r in rows]),
                      np.float32([[m.distance for m in r] for r in rows])))

    for _ in range(3 * max_notes):  # mỗi vòng loại ít nhất 4 điểm nên vòng lặp luôn dừng
        if active.sum() < 4:
            break
        hyps = []
        for t, q_idx, n_idx, n_dist in neigh:
            act = active[n_idx]
            rows = np.flatnonzero(act.sum(1) >= 2)
            if len(rows) < 4:
                continue
            a = act[rows]
            first = a.argmax(1)
            a[np.arange(len(rows)), first] = False
            second = a.argmax(1)
            d1 = n_dist[rows, first]
            d2 = n_dist[rows, second]
            good = d1 < 0.75 * d2
            rows, first, d1 = rows[good], first[good], d1[good]
            train = n_idx[rows, first]
            # Mỗi điểm của ảnh chỉ giữ cặp có khoảng cách nhỏ nhất (ràng buộc khớp một–một)
            order = np.argsort(d1)
            _, keep_first = np.unique(train[order], return_index=True)
            sel = order[keep_first]
            if len(sel) < 4:
                continue
            q_pts = np.float32([t["kp"][i].pt for i in q_idx[rows[sel]]]).reshape(-1, 1, 2)
            train_sel = train[sel]
            src, dst = q_pts, pts[train_sel].reshape(-1, 1, 2)
            Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac)
            if Hm is None or mask is None:
                continue
            h, w = t["shape"]
            quad = cv2.perspectiveTransform(np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
                                            .reshape(-1, 1, 2), Hm)
            if not np.isfinite(quad).all() or not cv2.isContourConvex(quad.astype(np.float32)):
                continue
            area = abs(cv2.contourArea(quad.astype(np.float32))) / (H * W)
            if not 0.002 <= area <= max_area_note:
                continue
            keep = mask.ravel().astype(bool)
            hyps.append((t["label"], int(keep.sum()), keep.sum() / len(sel), area, t["name"],
                         quad.reshape(-1, 2).tolist(), train_sel[keep]))
        if not hyps:
            break
        top = max(hyps, key=lambda c: (c[1], c[2]))
        if top[1] < weakest:
            break

        # Chỉ các ứng viên khớp vào cùng chỗ với giả thuyết mạnh nhất mới cạnh tranh mệnh giá
        local = [c for c in hyps if quad_iou(c[5], top[5]) >= same_place_iou]
        rank = None
        if color_model is not None:
            x0, y0 = np.clip(np.min(top[5], axis=0), 0, [W - 1, H - 1]).astype(int)
            x1, y1 = np.clip(np.max(top[5], axis=0), 0, [W - 1, H - 1]).astype(int)
            if x1 - x0 > 10 and y1 - y0 > 10:
                rank = color_model.rank(hsv_hist(img[y0:y1, x0:x1]))
        label, count = fuse(local, rank, min_inliers, min_ratio, max_area_note, min_margin, veto_k,
                            agree_min, fallback)

        if label != UNKNOWN:
            box = max((c for c in local if c[0] == label), key=lambda c: (c[1], c[2]), default=top)[5]
            if not any(quad_iou(box, f[2]) >= same_place_iou for f in found):
                found.append((label, count, box))
                if len(found) >= max_notes:
                    break

        # Loại các điểm thuộc tờ vừa xét (inlier và mọi điểm trong khung) rồi tìm tờ tiếp theo
        active &= ~MplPath(np.float32(top[5])).contains_points(pts)
        active[top[6]] = False
    return found


def crop_box(img, bbox, pad=0.05, min_side=400):
    x, y, w, h = bbox
    H, W = img.shape[:2]
    px, py = w * pad, h * pad
    x0, y0 = max(int(x - px), 0), max(int(y - py), 0)
    x1, y1 = min(int(x + w + px), W), min(int(y + h + py), H)
    crop = img[y0:y1, x0:x1]
    # Vùng cắt nhỏ (trung vị ~200px) cho ít điểm SIFT, phóng to lên
    scale = min_side / max(crop.shape[:2])
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def report(name, rows, classes, out_dir):
    y_true = [r["true"] for r in rows]
    y_pred = [r["pred"] for r in rows]
    n = len(rows)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    answered = [(t, p) for t, p in zip(y_true, y_pred) if p != UNKNOWN]
    times = [r["time"] for r in rows]

    lines = [f"===== {name} ({n} mẫu) =====",
             f"Accuracy (không xác định tính là sai): {correct / n:.2%}",
             f"Tỷ lệ đưa ra kết quả:                  {len(answered) / n:.2%}",
             f"Accuracy trên mẫu có kết quả:          "
             + (f"{sum(t == p for t, p in answered) / len(answered):.2%}" if answered else "-"),
             f"Thời gian trung bình / mẫu:            {np.mean(times) * 1000:.0f} ms",
             "",
             f"{'Lớp':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Số mẫu':>8}"]

    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        n_pred = sum(p == c for p in y_pred)
        n_true = sum(t == c for t in y_true)
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_true if n_true else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        lines.append(f"{c:>10} {prec:>10.2%} {rec:>8.2%} {f1:>8.2%} {n_true:>8}")
    lines.append(f"{'Macro F1':>10} {np.mean(f1s):>28.2%}")

    # Cột dự đoán: các lớp + nhãn mẫu không có trong dữ liệu (vd. 500) + không xác định
    extra = sorted({p for p in y_pred if p not in classes and p != UNKNOWN}, key=int)
    cols = classes + extra + [UNKNOWN]
    cm = np.zeros((len(classes), len(cols)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[classes.index(t), cols.index(p)] += 1

    pairs = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t != p:
            pairs[(t, p)] += 1
    lines += ["", "Các cặp nhầm nhiều nhất (thật -> dự đoán):"]
    for (t, p), k in sorted(pairs.items(), key=lambda x: -x[1])[:8]:
        lines.append(f"  {t:>7} -> {p:<8} {k}")

    text = "\n".join(lines)
    print(text + "\n")

    slug = name.split()[0].lower()
    (out_dir / f"bao_cao_{slug}.txt").write_text(text, encoding="utf-8")

    with open(out_dir / f"du_doan_{slug}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(cols)), cols, rotation=45)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"SIFT - {name}: accuracy {correct / n:.1%}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{slug}.png", dpi=120)
    plt.close(fig)


def load_split(split):
    """Đọc tập dữ liệu COCO; trả về (thư mục, ánh xạ lớp, ảnh, nhãn)."""
    split_dir = ROOT / "data" / "dataset" / split
    coco = json.loads((split_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    # Tên lớp trong file COCO bị gán lệch: id 1..9 thực chất ứng với tên
    # mệnh giá sắp theo thứ tự chuỗi (1000, 10000, 100000, 2000, ...).
    # Đã kiểm tra bằng mắt trên ảnh cắt của từng lớp.
    names = sorted(c["name"] for c in coco["categories"] if c["name"].isdigit())
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    cat.update({i + 1: name for i, name in enumerate(names)})
    images = {im["id"]: im for im in coco["images"]}
    anns = [a for a in coco["annotations"] if cat[a["category_id"]].isdigit()]
    return split_dir, cat, images, anns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--min-inliers", type=int, default=MIN_INLIERS)
    parser.add_argument("--min-ratio", type=float, default=MIN_RATIO)
    parser.add_argument("--ransac", type=float, default=RANSAC_THRESH, help="Ngưỡng sai số chiếu lại RANSAC (pixel)")
    parser.add_argument("--max-area", type=float, default=MAX_AREA, help="Diện tích khung tối đa / diện tích ảnh")
    parser.add_argument("--margin", type=float, default=MIN_MARGIN,
                        help="Mẫu thắng phải hơn mẫu khác mệnh giá bao nhiêu lần (1.0 = tắt)")
    parser.add_argument("--k-train", type=int, default=K_TRAIN, help="Số ảnh mẫu từ tập train mỗi mệnh giá")
    parser.add_argument("--color-veto", type=int, default=COLOR_VETO, help="Màu phủ quyết ngoài top-k (0 = tắt)")
    parser.add_argument("--agree-min", type=int, default=AGREE_MIN,
                        help="Nhận ứng viên SIFT yếu (>= n inlier) nếu trùng màu xếp đầu (0 = tắt)")
    parser.add_argument("--color-fallback", type=float, default=COLOR_FALLBACK,
                        help="Lấy theo màu khi SIFT không kết luận và d1/d2 <= giá trị này (0 = tắt)")
    parser.add_argument("--chi-mau", type=float, default=0.0,
                        help="Chỉ dùng màu, không SIFT: trả lời khi d1/d2 <= giá trị này (1.0 = luôn trả lời; "
                             "0 = tắt). Chỉ đánh giá theo tờ tiền vì màu cần biết trước vị trí tờ.")
    parser.add_argument("--out", default="", help="Tên thư mục kết quả trong ket_qua/ (mặc định sift_<split>)")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ chạy N tờ đầu (để thử nhanh)")
    args = parser.parse_args()

    split_dir, cat, images, anns = load_split(args.split)
    if args.limit:
        anns = anns[:args.limit]
    classes = sorted({cat[a["category_id"]] for a in anns}, key=int)

    out_dir = ROOT / "ket_qua" / (args.out or f"sift_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = [] if args.chi_mau else load_templates(args.k_train)
    use_color = bool(args.color_veto or args.agree_min or args.color_fallback or args.chi_mau)
    color_model = ColorModel() if use_color else None
    fuse_args = (args.color_veto, args.agree_min, args.color_fallback)
    print(f"Đã nạp {len(templates)} ảnh mẫu{' + mô hình màu' if use_color else ''}, "
          f"{len(anns)} tờ tiền trong tập {args.split}\n")

    cache = {}

    def read(image_id):
        if image_id not in cache:
            cache.clear()
            cache[image_id] = cv2.imread(str(split_dir / images[image_id]["file_name"]))
        return cache[image_id]

    # 1. Theo tờ tiền
    rows = []
    for i, a in enumerate(anns, 1):
        img = read(a["image_id"])
        crop = crop_box(img, a["bbox"])
        start = time.perf_counter()
        rank = color_model.rank(hsv_hist(crop)) if color_model else None
        if args.chi_mau:
            pred = rank[0][0] if rank[0][1] <= args.chi_mau * rank[1][1] else UNKNOWN
            inliers = 0
        else:
            pred, inliers = classify(crop, templates, args.min_inliers, args.min_ratio, args.ransac,
                                     args.max_area, args.margin, rank, *fuse_args)
        rows.append({"file": images[a["image_id"]]["file_name"], "ann_id": a["id"],
                     "true": cat[a["category_id"]], "pred": pred, "inliers": inliers,
                     "time": round(time.perf_counter() - start, 4)})
        if i % 50 == 0:
            print(f"  {i}/{len(anns)}", flush=True)
    report("To_tien (cắt bbox)", rows, classes, out_dir)

    # 2. Theo ảnh, chỉ ảnh có đúng một tờ (màu lấy trong khung homography của giả thuyết SIFT mạnh nhất)
    per_image = defaultdict(list)
    for a in anns:
        per_image[a["image_id"]].append(a)
    rows = []
    for image_id, items in per_image.items():
        if len(items) != 1 or args.chi_mau:
            continue
        img = read(image_id)
        start = time.perf_counter()
        pred, inliers, _ = classify_image(
            img, templates, color_model, args.ransac, min_inliers=args.min_inliers, min_ratio=args.min_ratio,
            max_area=args.max_area, min_margin=args.margin, veto_k=args.color_veto,
            agree_min=args.agree_min, fallback=args.color_fallback)
        rows.append({"file": images[image_id]["file_name"], "ann_id": items[0]["id"],
                     "true": cat[items[0]["category_id"]], "pred": pred, "inliers": inliers,
                     "time": round(time.perf_counter() - start, 4)})
    if rows:
        report("Anh (ảnh một tờ)", rows, classes, out_dir)

    print(f"Kết quả lưu tại: {out_dir}")


if __name__ == "__main__":
    main()
