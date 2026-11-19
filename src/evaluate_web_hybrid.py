"""Đánh giá đúng cấu hình kết hợp của giao diện web (POST /api/hybrid, yolo_label=1) trên tập test và phần test sạch,
cùng cách tính với evaluate_clean.py (ghép một–một IoU >= 0,5, chỉ số theo ảnh). Chạy lại YOLO (cả tập) và SIFT + màu
(phần sạch) qua API để đối chiếu với số đã có (results/yolo/, results/clean_test/).

Chạy:  python src/evaluate_web_hybrid.py        (~10 phút trên CPU)
Kết quả: results/web_hybrid_test/predictions.json và summary.json
"""
import io
import json
import sys
import time

import numpy as np

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_multi_note import bbox_of, match  # noqa: E402
from evaluate_sift import ROOT, load_split  # noqa: E402
from web_app import app, models  # noqa: E402

OUT_DIR = ROOT / "results" / "web_hybrid_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = str(OUT_DIR / "predictions.json")
split_dir, cat, images, anns = load_split("test")
gts = {}
for a in anns:
    gts.setdefault(a["image_id"], []).append((cat[a["category_id"]], a["bbox"]))
clean = set(json.load(open(ROOT / "data" / "test_clean.json", encoding="utf-8"))["sach"])
models()
c = app.test_client()


def run(api, name, data, query=""):
    r = c.post(f"/api/{api}{query}", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")
    d = r.get_json()
    W, H = d["width"], d["height"]
    preds = [(n["label"], bbox_of(np.float32(n["quad"]), W, H)) for n in d["notes"] if n["label"] is not None]
    return preds, d


rows = []
t0 = time.perf_counter()
ids = sorted(i for i in images if i in gts)
for n, iid in enumerate(ids):
    name = images[iid]["file_name"]
    data = (split_dir / name).read_bytes()
    g = gts[iid]
    row = {"file": name, "clean": name in clean, "n_true": len(g)}
    apis = ["hybrid", "yolo"] + (["sift"] if name in clean else [])
    for api in apis:
        preds, d = run(api, name, data, "?yolo_label=1" if api == "hybrid" else "")
        ok = sum(preds[i][0] == g[j][0] for i, j in match(preds, g, 0.5))
        money = sum(int(p[0]) for p in preds) == sum(int(x[0]) for x in g)
        row[api] = {"n_pred": len(preds), "ok": ok, "money": money, "ms": d["ms"],
                    "ms_sift": d.get("ms_sift"), "n_boxes": len(d["notes"]),
                    "src": [x.get("source") for x in d["notes"]]}
    rows.append(row)
    if n % 25 == 0:
        print(n, round(time.perf_counter() - t0), "s", flush=True)
json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)


def metrics(rs, api):
    rs = [r for r in rs if api in r]
    T = sum(r["n_true"] for r in rs)
    Pn = sum(r[api]["n_pred"] for r in rs)
    ok = sum(r[api]["ok"] for r in rs)
    rec, pre = ok / T, (ok / Pn if Pn else 0)
    return {"anh": len(rs), "to": T, "recall": rec, "precision": pre, "f1": 2 * rec * pre / (rec + pre),
            "money": sum(r[api]["money"] for r in rs) / len(rs), "money_n": sum(r[api]["money"] for r in rs),
            "ms_tb": float(np.mean([r[api]["ms"] for r in rs]))}


res = {}
for part, sel in (("tat_ca", lambda r: True), ("sach", lambda r: r["clean"])):
    sub = [r for r in rows if sel(r)]
    res[part] = {api: metrics(sub, api) for api in ("hybrid", "yolo", "sift") if any(api in r for r in sub)}
hy = [r["hybrid"] for r in rows]
res["hybrid_khung"] = {"khung": sum(h["n_boxes"] for h in hy),
                       "theo_sift": sum(s == "sift" for h in hy for s in h["src"]),
                       "theo_yolo": sum(s == "yolo" for h in hy for s in h["src"]),
                       "sift_toan_anh": sum(s == "sift_full" for h in hy for s in h["src"]),
                       "ms_sift_moi_khung": sum(h["ms_sift"] or 0 for h in hy) / max(1, sum(h["n_boxes"] for h in hy))}
json.dump(res, open(str(OUT_DIR / "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
print("xong", round(time.perf_counter() - t0), "s")
