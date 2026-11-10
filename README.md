# Nhận diện mệnh giá tiền Việt Nam bằng SIFT + RANSAC + màu HSV

Đồ án học phần Thị giác máy tính – Nhóm 06. Hệ thống đối sánh đặc trưng SIFT giữa ảnh đầu vào và ảnh mẫu,
kiểm chứng bằng homography (RANSAC), chọn ảnh mẫu có nhiều inlier nhất và kết hợp thêm đặc trưng màu HSV khi
SIFT chưa đủ bằng chứng. Nhận diện 9 mệnh giá: 1.000, 2.000, 5.000, 10.000, 20.000, 50.000, 100.000, 200.000,
500.000 đồng.

## Cấu trúc thư mục

```
VietNamCurrency/
├── README.md
├── requirements.txt
├── src/                         # Mã nguồn
│   ├── gui.py             # Giao diện đồ họa nhận diện tiền (Tkinter)
│   ├── realtime.py              # Nhận diện thời gian thực từ webcam / video (SIFT, YOLO, kết hợp)
│   ├── evaluate_hybrid.py       # Đánh giá chế độ kết hợp YOLO định vị + SIFT + màu nhận diện
│   ├── recognize.py             # Nhận diện mệnh giá trong một ảnh (dòng lệnh)
│   ├── evaluate_sift.py         # Logic nhận diện (SIFT, RANSAC, kết hợp màu, nhiều tờ) + đánh giá trên tập có nhãn
│   ├── evaluate_multi_note.py     # Đánh giá nhận diện nhiều tờ trong một ảnh
│   ├── color_model.py               # Đặc trưng màu HSV và bộ phân loại màu
│   ├── build_templates.py               # Loại ảnh train gần trùng, tạo ảnh mẫu bổ sung + mô hình màu
│   ├── tune_thresholds.py     # Dò tham số trên tập valid
│   ├── check_leakage.py         # Tìm ảnh test gần trùng valid/train -> data/test_clean.json
│   ├── evaluate_clean.py        # Tính lại chỉ số trên phần test sạch -> results/clean_test/
│   ├── view_errors.py               # Xuất bảng ảnh các tờ bị nhận diện lỗi
│   ├── run_comparison.sh          # Chạy lại tuần tự mọi cấu hình so sánh (đo thời gian cùng điều kiện)
│   ├── demo_sift.py              # Minh họa: điểm khóa SIFT trên một ảnh mẫu
│   ├── demo_matching.py              # Minh họa: đối sánh một ảnh mẫu với ảnh thử (ratio test, RANSAC)
│   ├── yolo_prepare.py         # Đổi nhãn COCO sang YOLO, loại ảnh train gần trùng
│   ├── yolo_train.py       # Huấn luyện YOLO11n
│   ├── yolo_evaluate.py         # Đánh giá YOLO với cùng chỉ số như SIFT + màu
│   └── compare_boxes.py         # So sánh theo kích thước tờ, hình đối chiếu
├── colab_yolo/                  # Notebook huấn luyện trên Google Colab (+ dataset_yolo.zip để tải lên, không commit)
├── data/
│   ├── templates/               # 20 ảnh mẫu gốc: <mệnh giá>_<số thứ tự>
│   ├── templates_train/         # Ảnh mẫu bổ sung cắt từ tập train (build_templates.py)
│   ├── color_model.npz          # Mô hình màu: histogram HSV của 1.393 tờ train (build_templates.py)
│   ├── test_images/                 # Ảnh thử cho recognize.py
│   └── dataset/                 # Bộ dữ liệu Roboflow: train / valid / test + _annotations.coco.json
├── results/
│   ├── sift_test/               # Kết quả cấu hình mặc định trên tập test (+ errors/: ảnh các tờ lỗi)
│   ├── comparison/                 # Kết quả các cấu hình dùng để so sánh trong báo cáo
│   │   ├── baseline/             #   ngưỡng ban đầu, chỉ SIFT
│   │   ├── tuned_thresholds/   #   đã tinh chỉnh ngưỡng + nới diện tích, chỉ SIFT
│   │   ├── margin_rule/    #   thêm luật khoảng cách, chỉ SIFT
│   │   ├── color_only/             #   chỉ màu, luôn trả lời
│   │   ├── color_only_reject/     #   chỉ màu, từ chối khi d1/d2 > 0,9
│   │   └── train_templates_color/    #   SIFT + 8 ảnh mẫu train + màu
│   ├── parameter_search/              # Kết quả dò tham số trên tập valid
│   ├── multi_note_valid/, multi_note_test/  # Đánh giá nhận diện nhiều tờ (+ examples/)
│   ├── yolo/                    # Trọng số, biểu đồ huấn luyện và kết quả đánh giá YOLO
│   └── figures/           # Hình dùng trong báo cáo
└── report/
    ├── Report_Group06.docx / .pdf          # Báo cáo mới nhất (có phần tiền tệ và so sánh YOLO)
    ├── Currency_Recognition_Report_Previous.pdf           # Báo cáo cuối kỳ (bản trước)
    └── Currency_Recognition_Plan.docx         # Kế hoạch dự án
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

## Nhận diện thời gian thực

```
python src/realtime.py                            # webcam 0; chế độ hybrid nếu đã cài ultralytics, ngược lại sift
python src/realtime.py --mode yolo --camera 1     # chọn chế độ / webcam
python src/realtime.py --source video.mp4 --save ket_qua.mp4   # chạy trên video, ghi video kết quả
```


| Chế độ (phím) | Cách làm | Số tờ | Tốc độ nhận diện (CPU i5-12450H) |
|---|---|---|---|
| `sift` (1) | SIFT + RANSAC + màu trên toàn khung hình (`classify_image`) | 1 | ~1,4 lần/s |
| `yolo` (2) | YOLO11n, ngưỡng 0,45, NMS không phân lớp (chọn trên valid) | nhiều | ~6–10 lần/s |
| **`hybrid`** (3, mặc định) | YOLO định vị từng tờ, SIFT + màu nhận diện vùng cắt; chưa kết luận thì hiện "?" (`--yolo-label`: dùng nhãn YOLO) | nhiều | ~2–4 lần/s |

**Vì sao mặc định là `hybrid`:** khi thử bằng webcam, YOLO nhận một tờ 100.000 đồng lúc thành 500.000, lúc thành 10.000
(đúng điểm yếu đã thấy trên tập test). `src/evaluate_hybrid.py` so sánh ba cách gán nhãn trên cùng khung YOLO, chỉ số
như mục nhận diện nhiều tờ (đúng vị trí IoU ≥ 0,5 và đúng mệnh giá); chế độ được chọn trên **valid**:

| Cách gán nhãn | F1 valid | F1 test | Ảnh đúng tổng tiền (test) | F1 tờ 100.000 (valid / test) | F1 tờ 500.000 (valid / test) |
|---|---|---|---|---|---|
| YOLO | 90,82% | 90,63% | 75,23% | 84,8% / 82,1% | 76,2% / 77,3% |
| **hybrid** (SIFT + màu, không kết luận thì giữ nhãn YOLO) | **93,27%** | **92,79%** | **81,19%** | 92,1% / 93,5% | 90,7% / 92,8% |
| hybrid chặt (không kết luận thì bỏ khung) | 91,13% | 91,30% | 75,23% | 99,0% / 98,0% | 98,2% / 97,7% |
| **hybrid, vùng cắt ≤ 400 px** (dùng trong `realtime.py`) | 92,99% | 92,97% | 81,65% | 91,2% / 93,5% | 89,7% / 94,8% |

(F1 theo mệnh giá tính trên các khung đúng vị trí. Chi tiết: `results/hybrid_valid/`, `results/hybrid_test/`.)
Hybrid chặt gần như không nhận sai mệnh giá nhưng bỏ khoảng 10% khung, nên F1 chung thấp hơn.

Tóm tắt các cơ chế, chi tiết ở các mục con bên dưới:

- **Chặn vật không phải tiền** (thẻ ATM, thẻ bảo hiểm y tế…): chỉ báo mệnh giá khi SIFT xác nhận được hoa văn tờ tiền
  với khung homography gần chữ nhật; không đủ bằng chứng thì hiện khung xám "?" – xem
  [Chặn vật không phải tiền](#chặn-vật-không-phải-tiền).
- **Mặt sau bạc màu** (mặt sau tờ 20.000 dưới đèn phòng): trích SIFT lại sau khi tăng tương phản CLAHE khi lần đầu
  ra "?" – xem [Mặt sau bạc màu: CLAHE dự phòng](#mặt-sau-bạc-màu-clahe-dự-phòng).
- **Ổn định khung**: nhận diện ở luồng riêng, bám vết giữa các lần nhận diện, không chớp tắt – xem
  [Cách hoạt động](#cách-hoạt-động).

### Cách hoạt động

- Khung hình được hiển thị ngay. Việc nhận diện chạy ở luồng riêng trên khung hình mới nhất (khung đến trong lúc đang
  bận thì bỏ qua), nên hình không giật dù nhận diện chậm hơn tốc độ camera.
- **Chống chập chờn:** kết quả được bám theo giữa các lần nhận diện (ghép khung theo IoU ≥ 0,3). Tờ mới cần độ tin cậy
  YOLO ≥ 0,45; tờ đang bám theo được giữ bởi khung ≥ 0,25. Một vết chỉ bị xóa khi bộ nhận diện không thấy nó **2 vòng
  liên tiếp và** đã quá 0,6 giây (`--hold`; 1,5 giây với tờ do SIFT toàn khung hình tìm ra, vì đường này chỉ chạy mỗi
  0,5 giây) – vòng nhận diện chậm không làm khung tự tắt khi tờ vẫn còn trong hình. Vết cũ nằm chồng lên vết vừa cập
  nhật (tờ xoay làm khung mới lệch khỏi khung cũ) bị bỏ để không hiện hai khung cho một tờ.
- Ở chế độ `hybrid`, SIFT + màu (trên vùng cắt ≤ 400 px) chạy cho **tối đa một tờ mỗi vòng** (ưu tiên tờ mới, tờ đang
  "?", tờ YOLO đổi mệnh giá – dấu hiệu tờ khác vừa thế chỗ – rồi tờ quá 2 giây chưa kiểm tra, `--recheck`), để vòng
  nhận diện ngắn và vị trí các tờ được YOLO cập nhật liên tục; tờ chưa đến lượt dùng kết quả đã lưu hoặc tạm hiện "?".
  Nhãn hiển thị là kết quả SIFT + màu gần nhất (không bỏ phiếu trên kết quả đã lưu); lần kiểm tra lại "chưa rõ" chỉ giữ
  mệnh giá cũ thêm 2 giây.
- Chữ có dấu được dựng một lần thành ảnh nhỏ rồi dùng lại (vẽ: ~2 ms/khung thay vì ~11 ms), để luồng hiển thị không
  tranh tài nguyên với luồng nhận diện. `--record goc.mp4` ghi lại hình gốc từ webcam để chạy lại sau bằng `--source`
  (video từ tệp được phát đúng tốc độ, kể cả với `--no-show`, nên mô phỏng được webcam).
- Màn hình hiển thị khung, mệnh giá và bằng chứng của từng tờ (số cặp điểm khớp, "màu" hoặc độ tin cậy YOLO), tổng số
  tiền và tốc độ nhận diện. Phím: `1`/`2`/`3` đổi chế độ, `Space` tạm dừng, `S` lưu ảnh vào `results/realtime/`,
  `Q` hoặc `Esc` thoát.
- Chạy thử không cần webcam: `--no-show --sync --log log.csv` nhận diện mọi khung hình của video và ghi kết quả ra CSV.

### Chặn vật không phải tiền

Camera có thể thấy bất cứ vật gì, còn YOLO chỉ được huấn luyện trên ảnh có tờ tiền nên coi thẻ, giấy tờ hình chữ
nhật là tờ tiền. Các luật của SIFT + màu (`fuse()`) được dò trên bộ dữ liệu, nơi mọi vùng cắt chắc chắn là tờ tiền,
nên cần thêm một **cổng bằng chứng** trước khi báo mệnh giá:

- Code: `quad_angle_dev()` và `evidence_ok()` trong [`src/evaluate_sift.py`](src/evaluate_sift.py); tham số
  `EVIDENCE_ANGLE`, `EVIDENCE_MIN`, `EVIDENCE_YOLO` trong [`src/realtime.py`](src/realtime.py).
- Luật: chỉ báo mệnh giá khi có ứng viên SIFT cùng mệnh giá với khung homography lệch ≤ 45° so với hình chữ nhật và
  ≥ 8 cặp khớp (≥ 4 nếu YOLO cũng nói mệnh giá đó); kết quả chỉ dựa vào màu không được nhận. Không đạt → khung xám
  "?", không cộng vào tổng tiền.
- Kết quả: 200 vùng cắt thẻ (5 ảnh thẻ x 40 biến thể) không còn vùng nào bị báo ra tiền (trước: 39).

Hai lần sửa dẫn tới luật này:

- **Vật không phải tiền: thẻ bảo hiểm y tế bị báo là 2.000 đ.** Hai nguyên nhân cộng lại: (1) YOLO chỉ được huấn
  luyện trên ảnh có tờ tiền, chưa từng thấy vật khác, nên coi tấm thẻ hình chữ nhật có hoa văn là một tờ tiền; (2) luật
  "đồng thuận" của SIFT + màu nhận kết quả khi ứng viên SIFT tốt nhất chỉ có ≥ 5 inlier nếu trùng mệnh giá xếp đầu theo
  màu – thẻ chỉ khớp 5 điểm, màu trắng ngà gần nhất với tờ 2.000. Luật này được dò trên bộ dữ liệu, nơi mọi vùng cắt
  chắc chắn là tờ tiền (chỉ cần chọn mệnh giá); camera thì có thể thấy bất cứ vật gì. Vì vậy `realtime.py` thêm **cổng
  bằng chứng**: chỉ báo mệnh giá khi SIFT có ≥ 8 inlier **hoặc** nhãn SIFT + màu trùng nhãn YOLO (hai nguồn độc lập đồng
  ý); không đạt thì hiện "?" (chế độ `sift` không có YOLO: cần ≥ 8 inlier).
- **Thẻ ATM vẫn bị báo 2.000 / 20.000 đ với cổng trên** (lần thử sau đó, 4 ảnh): một lần SIFT khớp ngẫu nhiên đúng 8
  điểm trên chữ của thẻ; hai lần thẻ màu xanh khiến **cả YOLO lẫn màu** cùng nói 20.000 – hai nguồn "đồng ý" nhưng
  không độc lập (cùng bị màu xanh đánh lừa). Tín hiệu tách bạch là **hình dạng khung homography** (`quad_angle_dev`
  trong `evaluate_sift.py`, góc lệch 90° lớn nhất): khớp đúng tờ tiền cho khung gần chữ nhật (valid: trung vị 9,9°;
  webcam: p90 23°), khớp ngẫu nhiên trên thẻ cho khung méo dẹt (64–85° ở cả 5 ảnh thẻ). Cổng mới (`evidence_ok`):
  chỉ báo mệnh giá khi có ứng viên SIFT cùng mệnh giá với khung lệch ≤ 45° và ≥ 8 inlier, hoặc ≥ 4 inlier khi YOLO cũng
  nói mệnh giá đó; kết quả chỉ dựa vào màu không được nhận. Để có bộ âm tính ổn định (một ảnh chụp lại không tái hiện
  được lỗi, vốn chỉ xảy ra ở vài khung hình), mỗi ảnh thẻ được tạo 40 biến thể (xê dịch / co giãn khung, xoay ±8°, mờ,
  đổi sáng). So trên cùng ứng viên (`evaluate_hybrid.py --so-sanh-luat`, cột cuối của `MARGIN_RULES`):

  | Cổng | F1 hybrid chặt valid / test | Precision valid | Webcam lần 5: đúng / sai / "?" | 200 vùng cắt thẻ: bị báo ra tiền |
  |---|---|---|---|---|
  | không có | 90,96% / 91,01% | 97,45% | 69,9% / 9,2% / 20,8% | 39 |
  | ≥ 8 inlier | 80,42% / 82,46% | 98,65% | 68,8% / 6,4% / 24,9% | – |
  | ≥ 8 inlier hoặc trùng nhãn YOLO | 91,02% / 91,17% | 98,69% | 69,9% / 7,5% / 22,5% | 12 |
  | như trên, nhưng khung ≤ 45°; màu chỉ được nhận khi YOLO đồng ý | 89,15% / 89,84% | 98,98% | 68,8% / 6,9% / 24,3% | 8 (đều 20.000) |
  | **khung ≤ 45°, ≥ 8 inlier hoặc YOLO đồng ý + ≥ 4 inlier** (dùng trong `realtime.py`) | 82,26% / 84,16% | **98,95%** | 67,6% / **5,8%** / 26,6% | **0** |

  Đánh đổi: trên bộ dữ liệu (tờ nhỏ chụp xa, ít điểm khớp, trước đây dựa vào màu) mất ~9 điểm F1 – các tờ này hiện
  "?" thay vì có nhãn; trên webcam (tờ ở gần) số kết quả đúng gần như giữ nguyên. Với chương trình thời gian thực,
  báo sai mệnh giá cho vật không phải tiền tệ hơn hiện "?", nên chọn cổng chặt nhất không để lọt thẻ. Kèm theo: vết
  mất dấu > 0,5 giây rồi được tìm lại thì nhận diện lại ngay (có thể đã là tờ khác ở cùng chỗ). Video thử B (xoay,
  có khoảng trống; trung bình 3 lần): đúng đủ 51% (trước 62%), **0% nhãn sai, 0 lần tắt**.
  Hạn chế còn lại: YOLO vẫn khoanh vùng tấm thẻ (hiện khung "?"); muốn YOLO bỏ qua hẳn vật không phải tiền cần huấn
  luyện lại với ảnh âm (thẻ, giấy tờ, ví…) không gắn nhãn.

### Mặt sau bạc màu: CLAHE dự phòng

**Mặt sau tờ 20.000 không nhận được** (thẻ ATM lúc này đã ra "?" đúng như mong muốn): YOLO chắc chắn và đúng
(20.000, độ tin cậy 0,91–0,93) nhưng SIFT chỉ khớp 0–4 điểm với ảnh mẫu mặt sau – dưới đèn phòng mặt sau polymer bị
bạc màu, tương phản thấp (ảnh mẫu cắt từ ảnh chụp thật trong `data/templates_train/` cũng không khớp được). Trước
cổng bằng chứng, các tờ này được nhận nhờ màu + YOLO; cổng (cần thiết để chặn thẻ ATM xanh) chặn luôn đường đó.
Cách sửa: **tăng tương phản cục bộ CLAHE** trước khi trích SIFT (`evaluate_sift.CLAHE_CLIP`, cho cả ảnh mẫu và vùng
cắt; `evaluate_hybrid.py --clahe`). Luôn bật CLAHE làm SIFT chậm gần gấp đôi (nhiều điểm đặc trưng hơn: ~1.060 so
với ~590 điểm mỗi vùng cắt) và làm video thử B tụt còn ~34% khung đúng đủ, nên `realtime.py` chỉ dùng CLAHE **khi
lần trích thường ra "?"** (`CLAHE_FALLBACK`). Cùng cổng bằng chứng, chọn trên valid:

| CLAHE | F1 hybrid chặt valid / test | Webcam lần 5: đúng / sai / "?" | 200 vùng cắt thẻ lọt |
|---|---|---|---|
| tắt | 82,26% / 84,16% | 67,6% / 5,8% / 26,6% | 0 |
| luôn bật, clip 2 | 84,43% / 82,25% | 74,6% / 4,0% / 21,4% | 0 |
| luôn bật, clip 4 | 79,89% / 78,75% | 69,4% / 4,0% / 26,6% | 0 |
| **chỉ khi ra "?", clip 2** (dùng trong `realtime.py`) | **87,56% / 86,43%** | **78,6%** / 6,9% / **14,5%** | **0** |

Ảnh webcam: mặt sau tờ 20.000 nhận được ở 2/6 ảnh (trước 0/6; các ảnh còn lại tờ bị cong hoặc nghiêng mạnh);
38 ảnh của các lần thử trước: đúng 21 / sai 3 / "?" 14 (trước 19 / 3 / 16); video thử B (trung bình 3 lần): đúng đủ
~49%, không còn lần tắt. Vết mất dấu rồi được tìm lại không còn giữ mệnh giá cũ trong lúc chờ nhận diện lại.

### Lỗi đã biết: mặt trước tờ 20.000 / 500.000

**Mặt trước tờ 20.000 từng bị báo 500.000** (23 khớp): mặt trước 20.000 và 500.000 cùng màu xanh, cùng chân dung;
chạy lại trên ảnh chụp thì SIFT ra đúng 20.000 (24 khớp) nhưng 500.000 sát nút (22 khớp). Đã thử luật "mâu thuẫn với
YOLO" (SIFT chọn khác YOLO trong khi ứng viên của mệnh giá YOLO cũng mạnh gần ngang → "?"): trên valid gần như
không đổi, trên bản ghi webcam giảm nhãn sai (4,0% → 2,3% với ρ = 0,7) nhưng giảm mạnh số kết quả đúng (74,6% →
68,2%) vì YOLO hay sai trên webcam – chưa đưa vào.

### Nhật ký thử bằng webcam

Các lần thử webcam theo thứ tự thời gian và điều chỉnh rút ra từ mỗi lần (số liệu của các mục trên nằm ở đây).

**Điều chỉnh sau khi thử bằng webcam** (một tờ 100.000 đồng, 8 ảnh chụp lại trong lúc chạy):
- Vùng cắt từ webcam lớn (~600 px) khớp được nhiều chi tiết chung giữa các mệnh giá (chân dung, quốc huy, quốc hiệu),
  mệnh giá khác cũng đạt 40–46 cặp khớp nên luật khoảng cách 1,3 từ chối kết luận và nhãn YOLO (sai) được giữ. Thu
  vùng cắt về cạnh dài ≤ 400 px (cỡ vùng cắt khi dò tham số) giữ nguyên kết quả trên valid (bảng trên), nhanh hơn
  ~35%, và trên ảnh webcam nhận đúng 7/7 khung YOLO thay vì 4/7.
- YOLO bỏ sót tờ cầm **lộn ngược** hoặc chỉ cho độ tin cậy thấp với tờ cầm nghiêng. Khi YOLO không thấy tờ nào, thử lại
  trên khung hình xoay 180°, rồi SIFT trên toàn khung hình (bất biến với góc xoay); khung YOLO yếu (< 0,45) của tờ mới
  được SIFT + màu kiểm tra. Hai đường dự phòng này chạy tối đa mỗi 0,5 giây và chỉ nhận khi SIFT có ≥ 8 cặp khớp
  (không nhận kết luận chỉ dựa vào màu, để cảnh trống không bị "thấy" tờ tiền).
- Kết quả trên 8 ảnh: đúng 8/8 (trước: 4/8 đúng, 1 sai thành 10.000, 3 không thấy). Lưu ý các điều chỉnh được thiết kế
  dựa trên chính 8 ảnh này.
- Lần thử webcam thứ ba (tờ cầm nghiêng, gập, cong, nhòe do chuyển động; 16 ảnh chụp lại): 4 đúng, 6 không thấy (tờ
  gập đôi, cong, nhòe), **6 sai – cả 6 đều là lúc SIFT + màu chưa kết luận và nhãn YOLO được dùng thay** (có lần YOLO
  chắc 0,98 rằng tờ 100.000 là 500.000), rồi nhãn sai được giữ 2 giây. Vì vậy `realtime.py` **không dùng nhãn YOLO**:
  khi SIFT + màu chưa kết luận, tờ được vẽ khung xám "?" (không cộng vào tổng tiền), phiếu "?" không đè nhãn đã có, và
  tờ được nhận diện lại sau 0,5 giây thay vì 2 giây. Trên tập valid cách này tương ứng "hybrid chặt" (F1 91,13% so với
  93,27% khi một khung hình chỉ được nhận diện một lần), nhưng trên video mỗi tờ được thử lại nhiều lần. Tùy chọn
  `--yolo-label` trả về cách cũ.
- Lần thử webcam thứ tư (một rồi hai tờ 100.000 đồng, sau đó một tờ 500 đồng giấy; ~5,5 phút): trong các khung hình
  có nhãn mệnh giá, ~94% đúng (lần 2: ~55%, lần 3: ~65%), còn lại nhầm thành 50.000 / 10.000 / 500.000; "?" xuất hiện
  ở ~4% khung hình và thường chuyển sang nhãn đúng ngay sau đó. Tờ 500 đồng không có trong bộ dữ liệu (YOLO và mô hình
  màu không biết lớp này) nhưng vẫn được nhận nhờ ảnh mẫu SIFT trong `data/templates/`. Hạn chế còn lại: tờ thường chỉ
  được nhận từng lần rời rạc (khung hiện ~0,6 giây rồi mất), tốc độ nhận diện giảm còn ~2 lần/s khi YOLO không thấy tờ
  và phải chạy SIFT trên toàn khung hình.
- **Sửa chập chờn.** Đo trên hai video thử dựng từ tập test sạch (6 cảnh: 3 ảnh một tờ, 3 ảnh ba tờ; mỗi cảnh 4 giây
  xoay dần 0°→180° và thu nhỏ; video A chuyển cảnh đột ngột, video B có 1 giây trống giữa các cảnh như khi rút tờ ra
  rồi đưa tờ khác vào), phát đúng tốc độ thực, tính từ giây thứ hai mỗi cảnh; vì luồng nhận diện bỏ khung khác nhau
  mỗi lần chạy nên lấy trung bình 3 lần:

  | Video | Phiên bản | Khung hiện đủ và đúng mọi tờ | Khung có nhãn sai | Số lần khung tắt rồi hiện lại |
  |---|---|---|---|---|
  | A (chuyển cảnh đột ngột) | trước khi sửa (1 lần chạy) | 57,8% | 0% | 5 |
  | A | sau khi sửa | 51,0% | 7,3% | 0 |
  | B (có khoảng trống) | sau khi sửa | 62,3% | 0% | 0 |
  | A / B | chế độ `yolo` | 74,4% / 69,7% | 25,6% / 29,2% | 0 / 0 |

  Hết hẳn hiện tượng khung tắt rồi hiện lại. Đánh đổi: khi một tờ **thế chỗ tức thì** tờ khác ở cùng vị trí (chỉ có
  ở video A), vết cũ được giữ nên tờ mới có thể mang nhãn cũ đến lần kiểm tra lại (≤ 2 giây) nếu YOLO gọi hai tờ cùng
  một tên. Đã thử phát hiện việc thế chỗ bằng thay đổi màu vùng cắt: giảm nhãn sai ở video A (còn ~1–2%) nhưng
  kích hoạt nhầm khi tờ xoay nhanh, làm video B tụt còn ~34% khung đúng, nên không dùng. Phần còn thiếu chủ yếu là
  "?" ở các tờ nhỏ bị xoay nghiêng trong ảnh nhiều tờ (YOLO gọi đúng các tờ này nhưng SIFT + màu chưa kết luận).
- Lần thử webcam thứ năm (~10,5 phút, có ghi hình gốc bằng `--record`): khung không còn chớp tắt (thường hiện liền
  3–11 giây), ~90% khung có nhãn là đúng. Lỗi còn lại: ngay cả với tờ 100.000 to, phẳng, rõ nét, SIFT xếp **đúng
  100.000 đầu** nhưng mệnh giá khác cũng đạt 35–52 cặp khớp vì mặt trước mọi tờ polymer có chung chân dung, quốc huy,
  quốc hiệu, nên luật khoảng cách 1,3 từ chối kết luận (màu dưới ánh sáng webcam lại xếp 200.000 / 500.000 đầu).
- **Luật khoảng cách theo inlier riêng** (`decide(..., exclusive=True)` trong `evaluate_sift.py`): khi so mẫu thắng
  với mẫu đối thủ, bỏ các điểm SIFT của ảnh mà cả hai cùng khớp vào – điểm chung không phải bằng chứng cho mệnh giá
  nào. `candidates()` trả thêm chỉ số các điểm inlier; mặc định của dự án (bảng đánh giá ở trên) không đổi.
  `python src/evaluate_hybrid.py --crop-max 400 --so-sanh-luat` so các luật trên cùng ứng viên SIFT:

  | Luật | F1 hybrid chặt valid / test | Bản ghi webcam lần 5 (173 khung YOLO): đúng / sai / "?" | 38 ảnh chụp webcam: đúng / sai |
  |---|---|---|---|
  | tổng inlier ≥ 1,3× (mặc định dự án) | 91,20% / 91,48% | 59,5% / 9,8% / 30,6% | 16 / 2 |
  | **inlier riêng ≥ 1,3×** (dùng trong `realtime.py`) | 90,96% / 91,01% | **69,9% / 9,2% / 20,8%** | **19 / 3** |
  | inlier riêng ≥ 1,5× | 91,09% / 91,37% | 64,7% / 9,2% / 26,0% | |
  | inlier riêng ≥ 2× | 91,13% / 91,17% | 56,6% / 9,8% / 33,5% | |

  Trên bộ dữ liệu (tờ nhỏ, chụp xa) hai luật gần như ngang nhau; trên webcam (tờ ở gần, chi tiết chung chiếm nhiều
  điểm khớp) luật inlier riêng giảm mạnh số "?" mà không tăng nhãn sai. Bản ghi webcam không có nhãn: "đúng" = 100.000
  hoặc 500 đồng (hai tờ duy nhất được dùng trong buổi thử), mọi nhãn khác tính là sai. Lưu ý chưa có bản ghi webcam
  độc lập để kiểm tra lại lựa chọn này.

### Chạy thử trên video

**Chạy thử** trên video tổng hợp từ 8 ảnh test sạch (4 ảnh một tờ, 4 ảnh 3–5 tờ; mỗi ảnh 10 khung hình có trượt nhẹ),
một cảnh được tính đúng khi nhãn đa số trong 7 khung cuối khớp đúng toàn bộ các tờ: `hybrid` 6/8, `yolo` 5/8, `sift`
2/8 (`sift` chỉ nhận một tờ nên sai cả 4 cảnh nhiều tờ). Hai lỗi còn lại của `hybrid`: tờ 100.000 đồng trong một khung
hình video mờ, ám xanh (SIFT chỉ được ~10 cặp khớp, màu cũng xếp 200.000 đầu) và một tờ 500.000 đồng nhỏ mà YOLO không
tìm thấy. Đây chỉ là bài thử chức năng, không thay cho các bảng đánh giá ở trên.

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

Mô hình YOLO11n được huấn luyện trên cùng bộ dữ liệu để so sánh. Tập train được loại ảnh gần trùng valid/test (cùng
ngưỡng dHash 70 bit, còn 559 ảnh), mọi ngưỡng chọn trên tập valid, chỉ số tính bằng cùng hàm với SIFT + màu.
Huấn luyện trên GPU Tesla T4 (Google Colab), 60 epoch, 11,3 phút; đánh giá và đo thời gian trên CPU i5-12450H.

```
pip install -r requirements_yolo.txt
python src/yolo_prepare.py      # đổi nhãn COCO sang YOLO -> data/dataset_yolo/
python src/yolo_train.py    # huấn luyện (nên dùng GPU, hoặc notebook colab_yolo/train_yolo_colab.ipynb)
python src/yolo_evaluate.py      # đánh giá -> results/yolo/evaluation_test/
python src/compare_boxes.py      # recall theo kích thước tờ, hình đối chiếu
```

**Kết quả trên tập test:**

| Kịch bản | SIFT + màu | YOLO11n |
|---|---|---|
| Tự tìm mọi tờ: F1 (vị trí + mệnh giá) | 74,68% | **90,63%** |
| Tự tìm mọi tờ: ảnh đúng tổng tiền | 55,50% | **75,23%** |
| Tự tìm mọi tờ: thời gian / ảnh | 2,4 s | **0,14 s** (8 luồng), 0,41 s (1 luồng) |
| Đã biết vị trí: accuracy (558 tờ) | 87,46% | 87,46% (YOLO chạy trên ảnh đầy đủ) |
| Đã biết vị trí: Macro-F1 | **91,76%** | 90,39% |
| Ảnh một tờ, cùng quy tắc ngưỡng: accuracy (87 ảnh) | **88,51%** | 72,41% (89,66% nếu luôn trả lời, sai 8 tờ) |

Trên vùng cắt theo khung bao, YOLO chỉ đạt 68,64% khi luôn trả lời. YOLO kém ở tờ 100.000 và 500.000 đồng (F1 74,14% và
71,26%), SIFT + màu vẫn trên 90%. Chi tiết: mục 3.7 của báo cáo.

## Ghi chú về dữ liệu

- **Lỗi tên lớp:** trong `_annotations.coco.json`, `category_id` 1–9 thực chất ứng với tên mệnh giá **sắp theo thứ
  tự chuỗi** (1000, 10000, 100000, 2000, 20000, 200000, 5000, 50000, 500000). `src/evaluate_sift.py` đã tự sửa ánh
  xạ này; chương trình khác dùng bộ dữ liệu cũng cần sửa tương tự (xem `results/figures/label_check.jpg`).
- **Rò rỉ dữ liệu:** Roboflow chia ngẫu nhiên các khung hình liên tiếp và bản tăng cường của cùng ảnh gốc vào cả ba
  tập. `build_templates.py` loại mọi ảnh train có mã dHash 256 bit cách ảnh valid/test bất kỳ ≤ 70 bit trước khi tạo ảnh
  mẫu và mô hình màu (còn 559/1.537 ảnh train). Rò rỉ giữa valid và test được kiểm tra riêng ở mục dưới.
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
| YOLO11n, tự tìm mọi tờ: F1 | 90,63% | **97,47%** | 85,53% |

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

Chi tiết phương pháp, phân tích lỗi và thảo luận: xem `report/Currency_Recognition_Report_Previous.pdf`.
