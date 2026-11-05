"""Đánh giá YOLO trên cùng dữ liệu và chỉ số với SIFT + màu để so sánh (cần results/yolo/yolo11n/weights/best.pt).

Mọi ngưỡng chọn trên tập valid, tập test chỉ dùng để báo cáo:
  - Nhiều tờ: ngưỡng tin cậy và kiểu NMS cho F1 (vị trí + mệnh giá) cao nhất.
  - Theo tờ tiền và theo ảnh: accuracy cao nhất với accuracy trên mẫu có kết quả >= 95% (cùng quy tắc với SIFT + màu).

Chạy:  python src/yolo_evaluate.py
Kết quả: results/yolo/evaluation_test/
"""
import argparse
import json
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from evaluate_multi_note import iou, match
from evaluate_sift import ROOT, UNKNOWN, crop_box, load_split, report

sys.stdout.reconfigure(encoding="utf-8")

LUOI_CONF = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
MUC_CANH = [(0, 100), (100, 150), (150, 200), (200, 300), (300, 450), (450, 10000)]


def du_doan(model, img, agnostic=False, conf=0.01):
    """Trả về [(mệnh giá, [x, y, w, h], độ tin cậy)] sắp theo độ tin cậy giảm dần."""
    r = model.predict(img, conf=conf, imgsz=640, agnostic_nms=agnostic, device="cpu", verbose=False)[0]
    out = []
    for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        out.append((model.names[int(c)], [b[0], b[1], b[2] - b[0], b[3] - b[1]], s))
    return sorted(out, key=lambda p: -p[2])


def nhom_theo_anh(anns):
    per = defaultdict(list)
    for a in anns:
        per[a["image_id"]].append(a)
    return per


def chi_so(preds_all, gts_all, thr=0.5):
    """Chỉ số nhiều tờ, cùng định nghĩa với evaluate_multi_note.py; preds_all, gts_all: danh sách theo ảnh."""
    tong = defaultdict(float)
    for preds, gts in zip(preds_all, gts_all):
        m = match(preds, gts, thr)
        tot = sum(preds[i][0] == gts[j][0] for i, j in m)
        for k in ("tat_ca", "mot_to" if len(gts) == 1 else "nhieu_to"):
            tong[k, "n"] += len(gts); tong[k, "p"] += len(preds); tong[k, "tp"] += len(m); tong[k, "tpc"] += tot
            tong[k, "img"] += 1; tong[k, "dem"] += len(preds) == len(gts)
            tong[k, "tien"] += sum(int(p[0]) for p in preds) == sum(int(g[0]) for g in gts)
    kq = {}
    for k in ("tat_ca", "mot_to", "nhieu_to"):
        n, p, tp, tpc = tong[k, "n"], tong[k, "p"], tong[k, "tp"], tong[k, "tpc"]
        r, pr = tpc / n, tpc / p if p else 0.0
        kq[k] = {"so_anh": int(tong[k, "img"]), "so_to": int(n), "so_khung": int(p), "recall_vi_tri": tp / n,
                 "precision_vi_tri": tp / p if p else 0.0, "recall": r, "precision": pr,
                 "f1": 2 * r * pr / (r + pr) if r + pr else 0.0, "dem_dung": tong[k, "dem"] / tong[k, "img"],
                 "dung_tong_tien": tong[k, "tien"] / tong[k, "img"]}
    return kq


def chon_nguong_tra_loi(mau):
    """mau: [(nhãn thật, nhãn dự đoán hoặc None, độ tin cậy)].

    Quy tắc: accuracy cao nhất với độ đúng khi trả lời >= 95%. Nếu không ngưỡng nào đạt thì lấy ngưỡng có độ đúng
    khi trả lời cao nhất và ghi dat_95 = False.
    """
    luoi = []
    for c in LUOI_CONF:
        tra_loi = [(t, p) for t, p, s in mau if p is not None and s >= c]
        if tra_loi:
            dung = sum(t == p for t, p in tra_loi)
            luoi.append({"conf": c, "acc": dung / len(mau), "ty_le_tra_loi": len(tra_loi) / len(mau),
                         "acc_tra_loi": dung / len(tra_loi)})
    dat = [x for x in luoi if x["acc_tra_loi"] >= 0.95]
    tot = max(dat, key=lambda x: x["acc"]) if dat else max(luoi, key=lambda x: x["acc_tra_loi"])
    return dict(tot, dat_95=bool(dat), luoi=luoi)


def theo_to_tren_anh(gts_anh, preds_anh, images_ten):
    """Đã biết khung tờ tiền, YOLO chạy trên ảnh đầy đủ: lấy khung tin cậy nhất có IoU >= 0,5 với khung nhãn."""
    mau = []
    for ten, gts, preds in zip(images_ten, gts_anh, preds_anh):
        for g in gts:
            khop = [p for p in preds if iou(p[1], g[1]) >= 0.5]
            best = max(khop, key=lambda p: p[2]) if khop else None
            mau.append({"file": ten, "true": g[0], "pred": best[0] if best else None, "conf": best[2] if best else 0.0})
    return mau


def thu_tap(model, split):
    """Chạy một tập, lưu dự đoán thô (conf >= 0,01) ở cả hai kiểu NMS, trên ảnh đầy đủ và vùng cắt."""
    split_dir, cat, images, anns = load_split(split)
    per = nhom_theo_anh(anns)
    ids = sorted(per)
    gts, tho, cat_tho, tg_anh, tg_cat = [], {False: [], True: []}, [], [], []
    for iid in ids:
        img = cv2.imread(str(split_dir / images[iid]["file_name"]))
        gts.append([(cat[a["category_id"]], a["bbox"]) for a in per[iid]])
        t0 = time.perf_counter()
        p = du_doan(model, img, False)
        tg_anh.append(time.perf_counter() - t0)
        tho[False].append(p)
        tho[True].append(du_doan(model, img, True))
        for a in per[iid]:
            crop = crop_box(img, a["bbox"])
            t0 = time.perf_counter()
            p = du_doan(model, crop, False)
            tg_cat.append(time.perf_counter() - t0)
            cat_tho.append({"file": images[iid]["file_name"], "ann_id": a["id"], "true": cat[a["category_id"]],
                            "pred": p[0][0] if p else None, "conf": p[0][2] if p else 0.0})
    return ids, images, gts, tho, cat_tho


def do_thoi_gian(model, split, threads, conf, agnostic):
    split_dir, cat, images, anns = load_split(split)
    per = nhom_theo_anh(anns)
    torch.set_num_threads(threads)
    anh = [cv2.imread(str(split_dir / images[i]["file_name"])) for i in sorted(per)]
    cat_anh = [crop_box(cv2.imread(str(split_dir / images[a["image_id"]]["file_name"])), a["bbox"]) for a in anns]
    du_doan(model, anh[0], agnostic, conf)  # bỏ lần chạy đầu (khởi tạo)
    t = []
    for im in anh:
        t0 = time.perf_counter(); du_doan(model, im, agnostic, conf); t.append(time.perf_counter() - t0)
    c = []
    for im in cat_anh:
        t0 = time.perf_counter(); du_doan(model, im, agnostic, conf); c.append(time.perf_counter() - t0)
    return {"so_luong": threads, "ms_moi_anh": 1000 * float(np.mean(t)), "ms_moi_to_cat": 1000 * float(np.mean(c))}


def theo_kich_thuoc(preds_all, gts_all, thr=0.5):
    """Recall (vị trí + mệnh giá) theo cạnh dài của tờ tiền."""
    dem = {m: [0, 0] for m in MUC_CANH}
    for preds, gts in zip(preds_all, gts_all):
        dung = {j for i, j in match(preds, gts, thr) if preds[i][0] == gts[j][0]}
        for j, g in enumerate(gts):
            canh = max(g[1][2:])
            for m in MUC_CANH:
                if m[0] <= canh < m[1]:
                    dem[m][0] += 1
                    dem[m][1] += j in dung
    return {f"{a}-{b}": {"so_to": n, "recall": k / n if n else 0.0} for (a, b), (n, k) in dem.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trong-so", default=str(ROOT / "results" / "yolo" / "yolo11n" / "weights" / "best.pt"))
    args = parser.parse_args()
    out_dir = ROOT / "results" / "yolo" / "evaluation_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    model = YOLO(args.trong_so)
    assert list(model.names.values()) == ["1000", "2000", "5000", "10000", "20000", "50000", "100000", "200000", "500000"]

    # 1. Chọn ngưỡng trên valid
    print("Chạy tập valid...", flush=True)
    ids_v, images_v, gts_v, tho_v, cat_v = thu_tap(model, "valid")
    luoi = []
    for ag in (False, True):
        for c in LUOI_CONF:
            preds = [[p for p in ds if p[2] >= c] for ds in tho_v[ag]]
            k = chi_so(preds, gts_v)["tat_ca"]
            luoi.append({"agnostic": ag, "conf": c, "f1": k["f1"], "recall": k["recall"], "precision": k["precision"]})
    tot = max(luoi, key=lambda x: x["f1"])
    conf, ag = tot["conf"], tot["agnostic"]
    print("Nhiều tờ - chọn trên valid:", tot, flush=True)
    nguong_cat = chon_nguong_tra_loi([(r["true"], r["pred"], r["conf"]) for r in cat_v])
    mot_to = [(g[0][0], (d[0][0] if d else None), (d[0][2] if d else 0.0))
              for g, d in zip(gts_v, tho_v[False]) if len(g) == 1]
    nguong_anh = chon_nguong_tra_loi(mot_to)
    ten_v = [images_v[i]["file_name"] for i in ids_v]
    to_anh_v = theo_to_tren_anh(gts_v, tho_v[True], ten_v)
    nguong_to_anh = chon_nguong_tra_loi([(r["true"], r["pred"], r["conf"]) for r in to_anh_v])
    tom_tat = lambda x: {k: v for k, v in x.items() if k != "luoi"}
    print("Theo tờ, vùng cắt - chọn trên valid:", tom_tat(nguong_cat), flush=True)
    print("Theo tờ, ảnh đầy đủ - chọn trên valid:", tom_tat(nguong_to_anh), flush=True)
    print("Theo ảnh - chọn trên valid:", tom_tat(nguong_anh), flush=True)
    valid = chi_so([[p for p in ds if p[2] >= conf] for ds in tho_v[ag]], gts_v)

    # 2. Báo cáo trên test
    print("Chạy tập test...", flush=True)
    ids, images, gts_t, tho_t, cat_t = thu_tap(model, "test")
    preds_t = [[p for p in ds if p[2] >= conf] for ds in tho_t[ag]]
    test = chi_so(preds_t, gts_t)
    classes = sorted({g[0] for ds in gts_t for g in ds}, key=int)

    rows = [{"file": r["file"], "ann_id": r["ann_id"], "true": r["true"],
             "pred": r["pred"] if r["pred"] is not None and r["conf"] >= nguong_cat["conf"] else UNKNOWN,
             "inliers": round(r["conf"], 3), "time": 0} for r in cat_t]
    report("Banknote (cắt bbox)", rows, classes, out_dir, "YOLO")
    rows_luon = [dict(r, pred=c["pred"] if c["pred"] is not None else UNKNOWN) for r, c in zip(rows, cat_t)]
    (out_dir / "always_answer").mkdir(exist_ok=True)
    report("Banknote (cắt bbox, luôn trả lời)", rows_luon, classes, out_dir / "always_answer", "YOLO")

    to_anh_t = theo_to_tren_anh(gts_t, tho_t[True], [images[i]["file_name"] for i in ids])
    rows_ta = [{"file": r["file"], "ann_id": k, "true": r["true"],
                "pred": r["pred"] if r["pred"] is not None and r["conf"] >= nguong_to_anh["conf"] else UNKNOWN,
                "inliers": round(r["conf"], 3), "time": 0} for k, r in enumerate(to_anh_t)]
    (out_dir / "note_on_full_image").mkdir(exist_ok=True)
    report("Banknote (khung nhãn, ảnh đầy đủ)", rows_ta, classes, out_dir / "note_on_full_image", "YOLO")

    rows = []
    for iid, g, d in zip(ids, gts_t, tho_t[False]):
        if len(g) == 1:
            ok = d and d[0][2] >= nguong_anh["conf"]
            rows.append({"file": images[iid]["file_name"], "ann_id": iid, "true": g[0][0],
                         "pred": d[0][0] if ok else UNKNOWN, "inliers": round(d[0][2], 3) if d else 0, "time": 0})
    report("Image (ảnh một tờ)", rows, classes, out_dir, "YOLO")
    rows_luon_anh = [dict(r, pred=d[0][0] if d else UNKNOWN) for r, d in
                     zip(rows, [d for g, d in zip(gts_t, tho_t[False]) if len(g) == 1])]
    report("Image (ảnh một tờ, luôn trả lời)", rows_luon_anh, classes, out_dir / "always_answer", "YOLO")

    # 3. Dự đoán từng khung để so sánh và vẽ hình; thời gian đo riêng, không có tác vụ đánh giá chen vào
    (out_dir / "box_predictions.json").write_text(json.dumps(
        {images[i]["file_name"]: [list(p) for p in ds] for i, ds in zip(ids, preds_t)}, ensure_ascii=False), encoding="utf-8")
    thoi_gian = [do_thoi_gian(model, "test", th, conf, ag) for th in (1, 8)]
    ket_qua = {"chon_tren_valid": {"nhieu_to": tot, "theo_to_vung_cat": nguong_cat,
                                   "theo_to_anh_day_du": nguong_to_anh, "theo_anh": nguong_anh},
               "nhieu_to_valid": valid, "nhieu_to_test": test, "theo_kich_thuoc": theo_kich_thuoc(preds_t, gts_t),
               "thoi_gian": thoi_gian}

    # 4. mAP chuẩn COCO trên test
    torch.set_num_threads(8)
    m = model.val(data=str(ROOT / "data" / "dataset_yolo" / "currency.yaml"), split="test", imgsz=640, device="cpu",
                  project=str(ROOT / "results" / "yolo"), name="val_test", exist_ok=True, verbose=False)
    ket_qua["map_test"] = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                           "ap50_theo_lop": {model.names[int(c)]: float(a) for c, a in zip(m.box.ap_class_index, m.box.ap50)}}
    (out_dir / "results.json").write_text(json.dumps(ket_qua, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ket_qua, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
