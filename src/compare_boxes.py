"""So sánh SIFT + màu và YOLO theo từng tờ trên tập test: recall theo kích thước tờ và hình đối chiếu trên cùng ảnh.

Cần chạy yolo_evaluate.py trước. Chạy lại detect_all() của SIFT + màu (phóng to ×1,5) song song để lấy từng khung;
thời gian báo cáo của SIFT vẫn lấy từ results/multi_note_test (đo tuần tự).

Chạy:  python src/compare_boxes.py
"""
import json
import sys
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

from evaluate_multi_note import _init, _run, bbox_of, match
from evaluate_sift import MULTI_UPSCALE, ROOT, load_split
from yolo_evaluate import chi_so, theo_kich_thuoc

sys.stdout.reconfigure(encoding="utf-8")

OUT = ROOT / "results" / "yolo" / "evaluation_test"


def ve(img, gts, preds, tieu_de):
    """Khung xanh lá: đúng cả vị trí và mệnh giá; đỏ: sai; xanh dương mảnh: tờ trong nhãn."""
    img = img.copy()
    for _, b in gts:
        x, y, w, h = map(int, b)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 160, 0), 1)
    dung = {i for i, j in match(preds, gts, 0.5) if preds[i][0] == gts[j][0]}
    for i, p in enumerate(preds):
        x, y, w, h = map(int, p[1])
        mau = (0, 170, 0) if i in dung else (0, 0, 230)
        cv2.rectangle(img, (x, y), (x + w, y + h), mau, 2)
        cv2.putText(img, p[0], (max(x, 0), max(y - 4, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mau, 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (255, 255, 255), -1)
    cv2.putText(img, tieu_de, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


def main():
    split_dir, cat, images, anns = load_split("test")
    per = defaultdict(list)
    for a in anns:
        per[a["image_id"]].append(a)
    ids = sorted(per)
    jobs = [(str(split_dir / images[i]["file_name"]), MULTI_UPSCALE) for i in ids]
    with Pool(8, initializer=_init, initargs=(True,)) as pool:
        results = pool.map(_run, jobs, chunksize=2)

    gts = [[(cat[a["category_id"]], a["bbox"]) for a in per[i]] for i in ids]
    sift = [[(lab, bbox_of(q, images[i]["width"], images[i]["height"])) for lab, _, q in found]
            for i, (found, _) in zip(ids, results)]
    khung_yolo = json.loads((OUT / "box_predictions.json").read_text(encoding="utf-8"))
    yolo = [[(p[0], p[1]) for p in khung_yolo[images[i]["file_name"]]] for i in ids]

    kq = {"sift_chay_lai": chi_so(sift, gts), "sift_theo_kich_thuoc": theo_kich_thuoc(sift, gts),
          "yolo_theo_kich_thuoc": theo_kich_thuoc(yolo, gts), "yolo": chi_so(yolo, gts)}
    (OUT / "box_comparison.json").write_text(json.dumps(kq, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(kq, ensure_ascii=False, indent=1))

    # Hình đối chiếu: ảnh nhiều tờ có khác biệt rõ nhất, chọn tự động theo hiệu số tờ nhận đúng
    def dung(preds, g):
        return sum(preds[i][0] == g[j][0] for i, j in match(preds, g, 0.5))

    chenh = sorted(range(len(ids)), key=lambda k: (dung(yolo[k], gts[k]) - dung(sift[k], gts[k])), reverse=True)
    trung_vi = [k for k in chenh if len(gts[k]) > 1]
    chon = [trung_vi[0], trung_vi[-1]]  # ảnh YOLO hơn SIFT + màu nhiều nhất và ảnh YOLO kém nhất
    hang = []
    for k in chon:
        img = cv2.imread(str(split_dir / images[ids[k]]["file_name"]))
        hang.append(np.hstack([ve(img, gts[k], sift[k], f"SIFT + HSV: {dung(sift[k], gts[k])}/{len(gts[k])}"),
                               ve(img, gts[k], yolo[k], f"YOLO: {dung(yolo[k], gts[k])}/{len(gts[k])}")]))
        print(images[ids[k]]["file_name"], "SIFT", dung(sift[k], gts[k]), "YOLO", dung(yolo[k], gts[k]), "/", len(gts[k]))
    cv2.imwrite(str(ROOT / "results" / "figures" / "sift_vs_yolo.jpg"), np.vstack(hang))


if __name__ == "__main__":
    main()
