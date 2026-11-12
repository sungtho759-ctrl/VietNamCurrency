# Nhận diện mệnh giá tiền Việt Nam bằng SIFT + RANSAC + màu HSV

Đồ án học phần Thị giác máy tính – Nhóm 06. Hệ thống đối sánh đặc trưng SIFT giữa ảnh đầu vào và ảnh mẫu,
kiểm chứng bằng homography (RANSAC), chọn ảnh mẫu có nhiều inlier nhất và kết hợp thêm đặc trưng màu HSV khi
SIFT chưa đủ bằng chứng. Nhận diện 9 mệnh giá: 1.000, 2.000, 5.000, 10.000, 20.000, 50.000, 100.000, 200.000,
500.000 đồng. Mô hình YOLO dùng để so sánh nhận thêm tờ 500 đồng (10 lớp, xem mục **Ghi chú về dữ liệu**).

## Cấu trúc thư mục

```
VietNamCurrency/
├── README.md
├── requirements.txt, requirements_yolo.txt
├── web/                           # Trang web của web_app.py (index.html, style.css, app.js, logo.svg)
├── docs/
│   └── realtime.md                # Chi tiết thiết kế + nhật ký thử nghiệm nhận diện thời gian thực
├── src/
│   ├── evaluate_sift.py           # Logic nhận diện (SIFT, RANSAC, kết hợp màu, nhiều tờ) + đánh giá trên tập có nhãn
│   ├── color_model.py             # Đặc trưng màu HSV và bộ phân loại màu
│   ├── recognize.py               # Nhận diện mệnh giá trong một ảnh (dòng lệnh)
│   ├── gui.py                     # Giao diện đồ họa (Tkinter)
│   ├── web_app.py                 # Giao diện web VND Vision AI (Flask): so sánh SIFT + màu và YOLO11n
│   ├── realtime.py                # Nhận diện thời gian thực từ webcam / video (SIFT, YOLO, kết hợp)
│   ├── build_templates.py         # Loại ảnh train gần trùng, tạo ảnh mẫu bổ sung + mô hình màu
│   ├── tune_thresholds.py         # Dò tham số trên tập valid
│   ├── evaluate_multi_note.py     # Đánh giá nhận diện nhiều tờ trong một ảnh
│   ├── evaluate_hybrid.py         # Đánh giá chế độ kết hợp YOLO định vị + SIFT + màu nhận diện
│   ├── check_leakage.py           # Tìm ảnh test gần trùng valid/train -> data/test_clean.json
│   ├── evaluate_clean.py          # Tính lại chỉ số trên phần test sạch -> results/clean_test/
│   ├── view_errors.py             # Xuất bảng ảnh các tờ bị nhận diện lỗi
│   ├── run_comparison.sh          # Chạy lại tuần tự mọi cấu hình so sánh (đo thời gian cùng điều kiện)
│   ├── demo_sift.py               # Minh họa: điểm khóa SIFT trên một ảnh mẫu
│   ├── demo_matching.py           # Minh họa: đối sánh một ảnh mẫu với ảnh thử (ratio test, RANSAC)
│   ├── yolo_prepare.py            # Đổi nhãn COCO sang YOLO, loại ảnh train gần trùng
│   ├── yolo_train.py              # Huấn luyện YOLO11n
│   ├── yolo_evaluate.py           # Đánh giá YOLO với cùng chỉ số như SIFT + màu
│   └── compare_boxes.py           # So sánh theo kích thước tờ, hình đối chiếu
├── colab_yolo/                    # Notebook huấn luyện YOLO trên Google Colab (dữ liệu: yolo_prepare.py --zip)
├── data/
│   ├── templates/                 # 20 ảnh mẫu gốc: <mệnh giá>_<số thứ tự>
│   ├── templates_train/           # Ảnh mẫu bổ sung cắt từ tập train (build_templates.py)
│   ├── color_model.npz            # Mô hình màu: histogram HSV của 1.393 tờ train (build_templates.py)
│   ├── test_clean.json            # Danh sách ảnh test không rò rỉ (check_leakage.py)
│   ├── test_images/               # Ảnh thử cho recognize.py và gui.py
│   └── dataset/                   # Bộ dữ liệu Roboflow (+ 114 ảnh tờ 500 tự chụp): train / valid / test
│                                  #   + _annotations.coco.json
└── results/
    ├── sift_test/                 # Cấu hình mặc định trên tập test (+ errors/: ảnh các tờ lỗi)
    ├── comparison/                # Các cấu hình so sánh: baseline, tuned_thresholds, margin_rule,
    │                              #   color_only, color_only_reject, train_templates_color
    ├── clean_test/                # Chỉ số trên phần test sạch (summary.md)
    ├── parameter_search/          # Kết quả dò tham số trên tập valid
    ├── multi_note_valid/, multi_note_test/   # Nhận diện nhiều tờ (+ examples/)
    ├── hybrid_valid*/, hybrid_test*/         # Chế độ kết hợp YOLO + SIFT + màu
    ├── yolo/                      # Trọng số, biểu đồ huấn luyện và kết quả đánh giá YOLO
    └── figures/                   # Hình minh họa (pipeline, đối sánh, giao diện, kiểm tra nhãn…)
```

Mỗi thư mục kết quả đánh giá gồm: `report_banknote.txt`, `report_image.txt` (chỉ số), `predictions_*.csv`
(dự đoán từng mẫu) và `confusion_*.png` (ma trận nhầm lẫn).

## Cài đặt

```
pip install -r requirements.txt
```

Cần Python 3.8+ và OpenCV ≥ 4.4 (đã có sẵn SIFT). `data/templates_train/` và `data/color_model.npz` đã có sẵn;
nếu xóa thì tạo lại bằng `python src/build_templates.py`.

## Sử dụng

Các lệnh chạy từ thư mục dự án (script tự xác định thư mục dự án nên chạy từ đâu cũng được):

```
python src/gui.py                          # mở giao diện nhận diện
python src/web_app.py                      # giao diện web tại http://localhost:5000
python src/recognize.py                          # nhận diện data/test_images/query.jpg
python src/recognize.py duong/dan/anh.jpg        # nhận diện ảnh bất kỳ
python src/evaluate_sift.py --split test         # đánh giá trên tập test -> results/sift_test/
python src/view_errors.py                            # bảng ảnh các tờ lỗi -> results/sift_test/errors/
python src/build_templates.py                            # tạo lại ảnh mẫu train + mô hình màu
python src/tune_thresholds.py --split valid    # dò tham số trên tập valid -> results/parameter_search/
```

## Giao diện

```
python src/gui.py
```

- **Mở ảnh...** hoặc **Chụp từ webcam** để nạp ảnh (mặc định mở sẵn `data/test_images/query.jpg`).
- **Nhận diện toàn ảnh:** SIFT tự định vị tờ tiền và vẽ khung homography – dùng cho ảnh chỉ có một tờ.
- **Nhận diện vùng chọn:** kéo chuột khoanh vùng một tờ tiền rồi bấm nút – chính xác nhất (ứng với kịch bản
  "theo tờ tiền" khi đánh giá), dùng được cả với ảnh có nhiều tờ.
- **Nhận diện tất cả các tờ:** tự tìm từng tờ trong ảnh (lặp SIFT + RANSAC), vẽ khung, liệt kê và tính tổng tiền.
- Bảng kết quả hiển thị mệnh giá, số cặp điểm khớp (hoặc "theo màu"), thời gian và 3 mệnh giá xếp đầu theo màu.
- **Nhận diện thời gian thực (webcam):** mở cửa sổ của `src/realtime.py` (mục dưới).

### Giao diện web (VND Vision AI)

```
pip install -r requirements_yolo.txt     # gồm ultralytics và flask
python src/web_app.py                      # rồi mở http://localhost:5000
```

Cùng một ảnh được gửi song song tới hai phương pháp, kết quả hiện cạnh nhau để so sánh: **truyền thống** (SIFT +
RANSAC + màu HSV, tự tìm mọi tờ – như "Nhận diện tất cả các tờ" ở trên) và **học sâu** (YOLO11n, ngưỡng chọn trên
valid). Mỗi bên vẽ khung lên ảnh, liệt kê từng tờ (số cặp điểm khớp / độ tin cậy), tổng tiền và thời gian xử lý; thanh
so sánh cho biết hai kết quả có trùng khớp không và phương pháp nào nhanh hơn bao nhiêu lần.

- **Quét trực tiếp:** camera chạy trong trình duyệt, có khung căn chỉnh ở giữa. "Chụp & nhận diện" xử lý một khung hình;
  "Nhận diện liên tục" cho mỗi phương pháp lấy khung hình mới nhất ngay khi xong lần trước (YOLO cập nhật nhiều lần trong
  lúc SIFT xử lý một khung). "Chỉ xét vùng trong khung" cắt ảnh theo khung căn chỉnh trước khi gửi.
- **Tải ảnh lên:** kéo thả, chọn tệp hoặc dán (Ctrl+V); có sẵn các ảnh mẫu trong `data/test_images/`.
- Máy chủ chỉ nghe trên `127.0.0.1`; camera cần trang mở qua `http://localhost` (hoặc https). Ảnh lớn được thu về cạnh
  dài 800 px trước SIFT; ảnh ≤ 640 px được phóng to ×1,5 như khi đánh giá nhiều tờ.

## Nhận diện thời gian thực

```
python src/realtime.py                            # webcam 0; chế độ hybrid nếu đã cài ultralytics, ngược lại sift
python src/realtime.py --mode yolo --camera 1     # chọn chế độ / webcam
python src/realtime.py --source video.mp4 --save ket_qua.mp4   # chạy trên video, ghi video kết quả
```


| Chế độ (phím) | Cách làm | Số tờ | Tốc độ nhận diện (CPU i5-12450H) |
|---|---|---|---|
| `sift` (1) | SIFT + RANSAC + màu trên toàn khung hình (`classify_image`) | 1 | ~1,4 lần/s |
| `yolo` (2) | YOLO11n, ngưỡng 0,35, NMS không phân lớp (chọn trên valid) | nhiều | ~6–10 lần/s |
| **`hybrid`** (3, mặc định) | YOLO định vị từng tờ, SIFT + màu nhận diện vùng cắt; chưa kết luận thì hiện "?" (`--yolo-label`: dùng nhãn YOLO) | nhiều | ~2–4 lần/s |

**Vì sao mặc định là `hybrid`:** khi thử bằng webcam, YOLO nhận một tờ 100.000 đồng lúc thành 500.000, lúc thành 10.000
(đúng điểm yếu đã thấy trên tập test). `src/evaluate_hybrid.py` so sánh ba cách gán nhãn trên cùng khung YOLO, chỉ số
như mục nhận diện nhiều tờ (đúng vị trí IoU ≥ 0,5 và đúng mệnh giá); chế độ được chọn trên **valid**:

| Cách gán nhãn | F1 valid | F1 test | Ảnh đúng tổng tiền (test) | F1 tờ 100.000 (valid / test) | F1 tờ 500.000 (valid / test) |
|---|---|---|---|---|---|
| YOLO | 90,82% | 90,63% | 75,23% | 84,8% / 82,1% | 76,2% / 77,3% |
| **hybrid** (SIFT + màu, không kết luận thì giữ nhãn YOLO) | **93,27%** | **92,79%** | **81,19%** | 92,1% / 93,5% | 90,7% / 92,8% |
| hybrid chặt (không kết luận thì bỏ khung) | 91,13% | 91,30% | 75,23% | 99,0% / 98,0% | 98,2% / 97,7% |
| **hybrid, vùng cắt ≤ 400 px** (cỡ vùng cắt dùng trong `realtime.py`) | 92,99% | 92,97% | 81,65% | 91,2% / 93,5% | 89,7% / 94,8% |

(F1 theo mệnh giá tính trên các khung đúng vị trí. Chi tiết: `results/hybrid_valid/`, `results/hybrid_test/`.
Bảng này đo với mô hình YOLO 9 lớp trước khi thêm tờ 500, chưa chạy lại với mô hình hiện tại.)
Hybrid chặt gần như không nhận sai mệnh giá nhưng bỏ khoảng 10% khung, nên F1 chung thấp hơn.

Các cơ chế chính (chi tiết, số liệu và nhật ký các lần thử webcam: [docs/realtime.md](docs/realtime.md)):

- **Ổn định khung:** nhận diện chạy ở luồng riêng trên khung hình mới nhất, kết quả được bám vết giữa các lần nhận
  diện (ghép khung theo IoU) nên khung không chớp tắt; ở chế độ `hybrid`, mỗi vòng chỉ chạy SIFT + màu cho tối đa một
  tờ (ưu tiên tờ mới / tờ đang "?").
- **Chặn vật không phải tiền** (thẻ ATM, thẻ bảo hiểm y tế…): chỉ báo mệnh giá khi có ứng viên SIFT cùng mệnh giá với
  khung homography lệch ≤ 45° so với hình chữ nhật và ≥ 8 cặp khớp (≥ 4 nếu YOLO cũng nói mệnh giá đó); không đạt thì
  hiện khung xám "?" và không cộng vào tổng tiền. 200 vùng cắt thẻ: không vùng nào bị báo ra tiền (trước: 39).
- **CLAHE dự phòng:** vùng cắt ra "?" thì trích SIFT lại sau khi tăng tương phản cục bộ (giúp nhận mặt sau tờ 20.000
  polymer bị bạc màu dưới đèn phòng).
- **Luật khoảng cách theo inlier riêng:** khi so mệnh giá thắng với đối thủ, bỏ các điểm SIFT mà cả hai cùng khớp
  (chân dung, quốc huy, quốc hiệu chung của mọi tờ polymer).

Phím: `1`/`2`/`3` đổi chế độ, `Space` tạm dừng, `S` lưu ảnh vào `results/realtime/`, `Q` hoặc `Esc` thoát.
Tùy chọn hữu ích: `--record goc.mp4` ghi hình gốc từ webcam để chạy lại bằng `--source`; `--no-show --sync --log log.csv`
chạy không cần màn hình và ghi kết quả từng khung ra CSV; `--yolo-label` dùng nhãn YOLO khi SIFT + màu chưa kết luận.

**Lưu ý:** cấu hình của `realtime.py` (cổng bằng chứng, inlier riêng, CLAHE dự phòng, vùng cắt ≤ 400 px) khác cấu hình
mặc định dùng trong các bảng đánh giá bên dưới. Cổng bằng chứng làm F1 trên bộ dữ liệu giảm (hybrid chặt, có CLAHE dự
phòng: 87,56% valid / 86,43% test) đổi lấy việc không báo sai mệnh giá cho vật không phải tiền.

## Cấu hình mặc định (SIFT + màu)

Chọn trên tập **valid** (accuracy cao nhất với precision ≥ 95%). Hằng số ở đầu `src/evaluate_sift.py`,
dùng chung cho `recognize.py` và `evaluate_sift.py`:

| Hằng số | Giá trị | Ý nghĩa |
|---|---|---|
| `RANSAC_THRESH` | 12 | Sai số chiếu lại tối đa (pixel) |
| `MIN_INLIERS` | 8 | Số inlier tối thiểu để SIFT kết luận |
| `MIN_RATIO` | 0 | Tỷ lệ inlier / cặp khớp tối thiểu |
| `MAX_AREA` | 10 | Diện tích khung homography tối đa / diện tích ảnh |
| `MIN_MARGIN` | 1,3 | Mệnh giá thắng phải có số inlier ≥ 1,3 lần mệnh giá khác, nếu không thì "không xác định" |
| `K_TRAIN` | 0 | Số ảnh mẫu train mỗi mệnh giá (không dùng) |
| `COLOR_VETO` | 0 | Phủ quyết theo màu (tắt) |
| `AGREE_MIN` | 5 | SIFT chưa đủ ngưỡng nhưng ứng viên tốt nhất có ≥ 5 inlier và trùng mệnh giá màu xếp đầu thì nhận |
| `COLOR_FALLBACK` | 0,8 | SIFT không kết luận, màu tách biệt rõ (khoảng cách lớp 1 / lớp 2 ≤ 0,8) thì lấy theo màu |

Mọi hằng số đều có tùy chọn dòng lệnh trong `evaluate_sift.py` (`--ransac`, `--min-inliers`, `--min-ratio`,
`--max-area`, `--margin`, `--k-train`, `--color-veto`, `--agree-min`, `--color-fallback`), cùng `--chi-mau r`
(chỉ dùng màu, trả lời khi d1/d2 ≤ r), `--out` (thư mục kết quả trong `results/`) và `--limit N`.

Chạy lại **tất cả** cấu hình so sánh, tuần tự trong cùng điều kiện để thời gian so sánh được (~45–60 phút):

```
bash src/run_comparison.sh
```

## Kết quả trên tập test

**Phạm vi:** kịch bản "theo tờ tiền" nhận diện trên vùng cắt theo khung bao **có sẵn trong nhãn** (đã biết vị trí
từng tờ). Kịch bản "theo ảnh" chỉ áp dụng cho ảnh có đúng một tờ. Việc tự phát hiện nhiều tờ trong ảnh được đánh
giá riêng ở mục **Nhận diện nhiều tờ** bên dưới.

**Chỉ số:** *Accuracy* tính "không xác định" là sai; *Tỷ lệ trả lời* = số mẫu có kết quả / tổng số mẫu;
*Accuracy trên mẫu có kết quả* = số mẫu đúng / số mẫu có kết quả (khác với precision của từng lớp).
Accuracy = Tỷ lệ trả lời × Accuracy trên mẫu có kết quả.

Mọi giá trị tham số được chọn trên tập valid. Lưu ý: phân tích lỗi của cấu hình ban đầu được làm trên tập test và đã
định hướng việc thiết kế các cải tiến, nên kết quả test sau cải tiến có thể lạc quan hơn một chút. Rò rỉ giữa valid
và test đã được kiểm tra và không làm kết quả lạc quan hơn, xem mục **Kiểm tra rò rỉ giữa valid và test** (kết quả valid của
cấu hình mặc định: 87,06%).

**Theo tờ tiền** (558 tờ). Thời gian đo khi chạy tuần tự mọi cấu hình bằng `src/run_comparison.sh` trên Intel Core
i5-12450H, một tiến trình, không gồm thời gian đọc ảnh và nạp ảnh mẫu:

| Cấu hình | Accuracy | Tỷ lệ trả lời | Accuracy trên mẫu có kết quả | Macro-F1 | Sai mệnh giá | Thời gian / tờ |
|---|---|---|---|---|---|---|
| Ban đầu (SIFT, RANSAC 5 px) | 57,89% | 59,32% | 97,58% | 70,82% | 8 | 320 ms |
| Tinh chỉnh ngưỡng + nới diện tích (SIFT) | 73,48% | 77,06% | 95,35% | 81,65% | 20 | 246 ms |
| Thêm luật khoảng cách 1,1 (SIFT) | 69,89% | 71,51% | 97,74% | 79,99% | 9 | 436 ms |
| Chỉ màu, luôn trả lời | 90,68% | 100% | 90,68% | 90,66% | 52 | 1 ms |
| Chỉ màu, từ chối khi d1/d2 > 0,9 | 84,77% | 89,43% | 94,79% | 89,29% | 26 | 1 ms |
| **SIFT + màu (mặc định, RANSAC 12 px)** | **87,46%** | 90,14% | 97,02% | **91,76%** | 15 | 409 ms |
| SIFT + 8 ảnh mẫu train + màu | 86,56% | 88,53% | **97,77%** | 91,28% | 11 | 1.251 ms |

Chỉ màu, nếu luôn trả lời, có accuracy cao nhất nhưng nhận sai 52 tờ (gấp ~3,5 lần SIFT + màu). Ở cùng quy tắc chọn
ngưỡng trên valid (accuracy trên mẫu có kết quả ≥ 95%), chỉ màu (ngưỡng 0,9) thấp hơn SIFT + màu ở cả accuracy và
accuracy trên mẫu có kết quả. Chỉ màu cũng cần biết trước vùng tờ tiền nên không dùng được khi nhận diện trên toàn ảnh.

**Theo ảnh** (87 ảnh chỉ có một tờ, nhận diện trên toàn ảnh):

| Cấu hình | Accuracy | Tỷ lệ trả lời | Accuracy trên mẫu có kết quả | Sai mệnh giá |
|---|---|---|---|---|
| Ban đầu (SIFT) | 57,47% | 57,47% | 100% | 0 |
| Tinh chỉnh ngưỡng + nới diện tích (SIFT) | 82,76% | 89,66% | 92,31% | 6 |
| Thêm luật khoảng cách 1,1 (SIFT) | 79,31% | 85,06% | 93,24% | 5 |
| **SIFT + màu trong khung SIFT (mặc định)** | **88,51%** | 91,95% | 96,25% | 3 |
| SIFT + 8 ảnh mẫu train + màu | 74,71% | 74,71% | 100% | 0 |

## Nhận diện nhiều tờ trong một ảnh

`detect_all()` trong `src/evaluate_sift.py` lặp SIFT + RANSAC:

1. Đối sánh mọi điểm SIFT còn lại của ảnh với các ảnh mẫu, lấy giả thuyết homography mạnh nhất làm vị trí một tờ.
2. Quyết định mệnh giá bằng các ứng viên **ở cùng vị trí** (luật khoảng cách chỉ so với mệnh giá khớp vào cùng chỗ,
   không so với tờ khác trong ảnh) kết hợp màu trong khung của tờ đó.
3. Loại các điểm SIFT nằm trong khung vừa tìm, lặp lại đến khi giả thuyết mạnh nhất không còn đủ inlier.

Ảnh được phóng to ×1,5 trước khi trích SIFT (`MULTI_UPSCALE`) để tờ nhỏ có thêm điểm đặc trưng; mức này chọn trên
tập valid trong {1; 1,5; 2}. Đối sánh K láng giềng (K = 6) được tính một lần, các vòng sau chỉ lọc láng giềng còn
hoạt động.

```
python src/evaluate_multi_note.py --split valid --phong-to 1 1.5 2 --workers 10   # chọn mức phóng to
python src/evaluate_multi_note.py --split test --phong-to 1.5                     # đánh giá cuối -> results/multi_note_test/
```

Một khung dự đoán đúng vị trí khi khung bao của nó có IoU ≥ 0,5 với khung bao trong nhãn (ghép một–một).
**Kết quả trên tập test** (218 ảnh, 558 tờ; chạy tuần tự):

| Ảnh | Số ảnh / tờ | Recall (vị trí + mệnh giá) | Precision (vị trí + mệnh giá) | F1 | Ảnh đếm đúng số tờ | Ảnh đúng tổng tiền | Thời gian / ảnh |
|---|---|---|---|---|---|---|---|
| Tất cả | 218 / 558 | 62,37% | 93,05% | 74,68% | 58,72% | 55,50% | 2,4 s |
| Có 1 tờ | 87 / 87 | 71,26% | 81,58% | 76,07% | 82,76% | 75,86% | 3,4 s |
| Có nhiều tờ | 131 / 471 | 60,72% | 95,97% | 74,38% | 42,75% | 41,98% | 1,8 s |

Nhận xét: hệ thống hầu như **không nhận thừa** (precision ~96% ở ảnh nhiều tờ) mà chủ yếu **bỏ sót** tờ nhỏ, mờ
(cảnh chụp xa, mỗi tờ ~70 px) – ở ảnh nhiều tờ, số tờ dự đoán không bao giờ vượt số tờ thật. Thỉnh thoảng
homography suy biến cho khung méo dù mệnh giá đúng. Với ảnh chỉ có một tờ, chế độ "Nhận diện toàn ảnh"
(`classify_image`, 88,51%) vẫn tốt hơn vì không yêu cầu khung đúng vị trí và cho phép khung lớn hơn ảnh.
Hướng cải thiện: kiểm tra hình dạng khung (gần chữ nhật, đúng tỷ lệ tờ tiền), kết hợp tách vùng tờ tiền bằng xử lý
ảnh hoặc dùng bộ phát hiện học sâu (YOLO) để tìm vị trí rồi nhận diện từng vùng bằng SIFT + màu.

## So sánh với YOLO

Mô hình YOLO11n được huấn luyện trên cùng bộ dữ liệu để so sánh, thêm lớp tờ 500 đồng (10 lớp). Tập train được loại
ảnh gần trùng valid/test (cùng ngưỡng dHash 70 bit, còn 633/1.622 ảnh), mọi ngưỡng chọn trên tập valid, chỉ số tính
bằng cùng hàm với SIFT + màu. Huấn luyện trên GPU Tesla T4 (Google Colab), 60 epoch, 11,5 phút; đánh giá và đo thời
gian trên CPU i5-12450H.

```
pip install -r requirements_yolo.txt
python src/yolo_prepare.py --zip   # đổi nhãn COCO sang YOLO -> data/dataset_yolo/ (+ colab_yolo/dataset_yolo.zip)
python src/yolo_train.py           # huấn luyện (nên dùng GPU, hoặc notebook colab_yolo/train_yolo_colab.ipynb)
python src/yolo_evaluate.py        # đánh giá -> results/yolo/evaluation_test/
python src/compare_boxes.py        # recall theo kích thước tờ, hình đối chiếu
```

**Kết quả trên tập test** (219 ảnh gốc của bộ dữ liệu, giống các bảng SIFT + màu; 10 ảnh tờ 500 báo cáo riêng ở dưới):

| Kịch bản | SIFT + màu | YOLO11n |
|---|---|---|
| Tự tìm mọi tờ: F1 (vị trí + mệnh giá) | 74,68% | **93,35%** |
| Tự tìm mọi tờ: ảnh đúng tổng tiền | 55,50% | **79,82%** |
| Tự tìm mọi tờ: thời gian / ảnh | 2,4 s | **0,10 s** (8 luồng), 0,15 s (1 luồng) |
| Đã biết vị trí: accuracy (558 tờ) | 87,46% | **88,17%** (YOLO chạy trên ảnh đầy đủ) |
| Đã biết vị trí: Macro-F1 | **91,76%** | 90,76% |
| Ảnh một tờ, cùng quy tắc ngưỡng: accuracy (87 ảnh) | **88,51%** | 87,36% (89,66% nếu luôn trả lời, sai 9 tờ) |

Trên vùng cắt theo khung bao, YOLO chỉ đạt 61,11% khi luôn trả lời. YOLO kém ở tờ 100.000 và 500.000 đồng (F1 80,34% và
72,09% trên ảnh đầy đủ), SIFT + màu vẫn trên 90%. Với quy tắc ngưỡng theo ảnh, ngưỡng chọn trên valid cho YOLO trả lời
94,25% số ảnh nhưng chỉ đúng 92,68% số ảnh có trả lời (dưới mức 95% đạt được trên valid). Chi tiết:
`results/yolo/evaluation_test/`. Hình `results/figures/sift_vs_yolo.jpg` vẽ với mô hình 9 lớp cũ.

**Tờ 500 đồng** (10 ảnh test, một lượt chụp trên nền không có trong train): YOLO nhận đúng cả 10 tờ (AP50 0,995; một
ảnh thừa một khung trùng). Mô hình 9 lớp trước đó nhận 7 tờ thành 200.000, 2 tờ thành 50.000 và bỏ sót 2 tờ. Chỉ có
một tờ 500 dùng cho cả train/valid/test, nên kết quả này chưa cho biết khả năng nhận các tờ 500 khác.

So với mô hình 9 lớp trước khi thêm tờ 500 (cùng 219 ảnh test): F1 tự tìm mọi tờ 90,63% → 93,35%, ảnh đúng tổng tiền
75,23% → 79,82%; trên phần test sạch F1 gần như không đổi (97,47% → 97,05%).

## Ghi chú về dữ liệu

- **Lỗi tên lớp:** trong `_annotations.coco.json`, `category_id` 1–9 thực chất ứng với tên mệnh giá **sắp theo thứ
  tự chuỗi** (1000, 10000, 100000, 2000, 20000, 200000, 5000, 50000, 500000). `src/evaluate_sift.py` đã tự sửa ánh
  xạ này; chương trình khác dùng bộ dữ liệu cũng cần sửa tương tự (xem `results/figures/label_check.jpg`).
- **Rò rỉ dữ liệu:** Roboflow chia ngẫu nhiên các khung hình liên tiếp và bản tăng cường của cùng ảnh gốc vào cả ba
  tập. `build_templates.py` loại mọi ảnh train có mã dHash 256 bit cách ảnh valid/test bất kỳ ≤ 70 bit trước khi tạo ảnh
  mẫu và mô hình màu (còn 559/1.537 ảnh train). Rò rỉ giữa valid và test được kiểm tra riêng ở mục dưới.
- **Tờ 500 đồng (chỉ dùng cho YOLO):** bộ dữ liệu gốc không có tờ 500, nên nhóm tự chụp 114 ảnh **một tờ** trong 11
  lượt (mỗi lượt một nền / ánh sáng), một số ảnh chụp chung với tờ khác. Ảnh được thu về 640×640 như ảnh Roboflow, khung
  gán bằng CVAT (gợi ý từ mô hình cũ rồi sửa tay), lớp `500` có `category_id` 10. Chia **theo lượt chụp** để ảnh cùng
  lượt không nằm ở hai tập: train 85 ảnh (lượt 0, 1, 3, 4, 5, 6, 7, 9), valid 19 (lượt 2, 8), test 10 (lượt 10); lượt
  của từng ảnh ghi ở `extra.luot_chup`. Ảnh gốc độ phân giải đầy đủ để ở `data/raw_500/` (không đưa vào git).
- **Hạn chế:** mô hình màu học từ ảnh cùng bộ dữ liệu (cùng tờ tiền, nền, điều kiện chụp tương tự), nên với ảnh chụp
  ở môi trường khác hẳn (đèn vàng, cân bằng trắng lệch) màu có thể kém tin cậy hơn.

## Kiểm tra rò rỉ giữa valid và test

Việc loại ảnh gần trùng ở trên chỉ áp dụng cho tập train. Tập valid và test vẫn chứa nhiều khung hình liên tiếp của
cùng cảnh. Mọi ngưỡng (và trọng số `best.pt` của YOLO) đều được chọn trên valid, nên cần kiểm tra xem kết quả test có
bị thổi phồng hay không.

```
python src/check_leakage.py                                # -> data/test_clean.json
python src/evaluate_clean.py                               # -> results/clean_test/summary.md
python src/tune_thresholds.py --split valid --bo-trung-test   # dò lại tham số trên valid không trùng test (~17 phút)
```

Một ảnh test bị coi là **rò rỉ** khi cùng ảnh gốc (trường `extra.name` của Roboflow) với một ảnh valid/train, hoặc mã
dHash cách một ảnh valid (kể cả bản lật) ≤ 70 bit. Kết quả: **133/219 ảnh test rò rỉ**, chỉ **85 ảnh (237 tờ) sạch**;
theo chiều ngược lại, 209/439 ảnh valid gần trùng test. `evaluate_clean.py` lọc lại các file dự đoán có sẵn,
không chạy lại mô hình.

| Chỉ số (tập test) | Toàn bộ | Sạch | Rò rỉ |
|---|---|---|---|
| SIFT + màu, theo tờ tiền: accuracy | 87,46% | **96,62%** | 80,69% |
| SIFT + màu, ảnh một tờ: accuracy | 88,51% (87 ảnh) | **90,62%** (32 ảnh) | 87,27% (55 ảnh) |
| SIFT + màu, tự tìm mọi tờ: F1 | 74,68% | **83,10%** | 67,59% |
| YOLO11n, tự tìm mọi tờ: F1 | 93,35% | **97,05%** | 90,60% |

Kết luận:

1. **Rò rỉ không làm kết quả test lạc quan hơn.** Trên phần sạch, mọi phương pháp đều đạt kết quả cao hơn. Nhiều khả năng
   phần rò rỉ chứa phần lớn ảnh khó, ví dụ các chuỗi khung hình video (43/133 ảnh, so với 7/85 ở phần sạch) bị chia sang cả
   valid lẫn test.
2. **Việc chọn tham số không bị ảnh hưởng bởi rò rỉ.** Dò lại toàn bộ 2.592 bộ tham số chỉ trên 560 tờ valid không
   trùng test (`results/parameter_search/results_valid_no_test_dup.txt`) cho **đúng cấu hình mặc định** (RANSAC
   12 px, 8 inlier, khoảng cách 1,3, đồng thuận 5, dự phòng màu 0,8).
3. Phần sạch nhỏ (85 ảnh) và dễ hơn phần rò rỉ, nên không thay thế được kết quả trên toàn bộ tập test. Các bảng ở
   trên vẫn dùng toàn bộ test; bảng này chỉ dùng để kiểm chứng. Thứ tự giữa các phương pháp không đổi, trừ chỉ màu
   (luôn trả lời), vốn vượt nhẹ SIFT + màu trên phần sạch (97,05% so với 96,62%) nhưng nhận sai 7 tờ so với 1 tờ.
4. Chưa làm: chia lại valid/test theo nhóm cảnh và huấn luyện lại YOLO. Riêng YOLO, `best.pt` được chọn theo valid,
   nhưng YOLO cũng đạt kết quả cao hơn trên phần sạch, nên không có dấu hiệu bị thổi phồng.
