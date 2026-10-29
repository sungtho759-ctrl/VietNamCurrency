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
│   ├── giao_dien.py             # Giao diện đồ họa nhận diện tiền (Tkinter)
│   ├── nhan_dien.py             # Nhận diện mệnh giá trong một ảnh (dòng lệnh)
│   ├── danh_gia_sift.py         # Logic nhận diện (SIFT, RANSAC, kết hợp màu, nhiều tờ) + đánh giá trên tập có nhãn
│   ├── danh_gia_nhieu_to.py     # Đánh giá nhận diện nhiều tờ trong một ảnh
│   ├── mau_sac.py               # Đặc trưng màu HSV và bộ phân loại màu
│   ├── tao_mau.py               # Loại ảnh train gần trùng, tạo ảnh mẫu bổ sung + mô hình màu
│   ├── tinh_chinh_nguong.py     # Dò tham số trên tập valid
│   ├── xem_loi.py               # Xuất bảng ảnh các tờ bị nhận diện lỗi
│   ├── chay_so_sanh.sh          # Chạy lại tuần tự mọi cấu hình so sánh (đo thời gian cùng điều kiện)
│   ├── thu_sift.py              # Minh họa: điểm khóa SIFT trên một ảnh mẫu
│   └── doi_sanh.py              # Minh họa: đối sánh một ảnh mẫu với ảnh thử (ratio test, RANSAC)
├── data/
│   ├── templates/               # 20 ảnh mẫu gốc: <mệnh giá>_<số thứ tự>
│   ├── templates_train/         # Ảnh mẫu bổ sung cắt từ tập train (tao_mau.py)
│   ├── mo_hinh_mau.npz          # Mô hình màu: histogram HSV của 1.393 tờ train (tao_mau.py)
│   ├── anh_thu/                 # Ảnh thử cho nhan_dien.py
│   └── dataset/                 # Bộ dữ liệu Roboflow: train / valid / test + _annotations.coco.json
├── ket_qua/
│   ├── sift_test/               # Kết quả cấu hình mặc định trên tập test (+ loi/: ảnh các tờ lỗi)
│   ├── so_sanh/                 # Kết quả các cấu hình dùng để so sánh trong báo cáo
│   │   ├── ban_dau/             #   ngưỡng ban đầu, chỉ SIFT
│   │   ├── tinh_chinh_nguong/   #   đã tinh chỉnh ngưỡng + nới diện tích, chỉ SIFT
│   │   ├── luat_khoang_cach/    #   thêm luật khoảng cách, chỉ SIFT
│   │   ├── chi_mau/             #   chỉ màu, luôn trả lời
│   │   ├── chi_mau_tu_choi/     #   chỉ màu, từ chối khi d1/d2 > 0,9
│   │   └── mau_train_va_mau/    #   SIFT + 8 ảnh mẫu train + màu
│   ├── do_tham_so/              # Kết quả dò tham số trên tập valid
│   ├── nhieu_to_valid/, nhieu_to_test/  # Đánh giá nhận diện nhiều tờ (+ minh_hoa/)
│   └── hinh_minh_hoa/           # Hình dùng trong báo cáo
└── bao_cao/
    ├── Bao_cao_Nhan_dien_tien_Viet_Nam.docx / .pdf   # Báo cáo cuối kỳ
    └── Ke_hoach_nhan_dien_tien_Viet_Nam.docx         # Kế hoạch dự án
```

Mỗi thư mục kết quả đánh giá gồm: `bao_cao_to_tien.txt`, `bao_cao_anh.txt` (chỉ số), `du_doan_*.csv`
(dự đoán từng mẫu) và `confusion_*.png` (ma trận nhầm lẫn).

## Cài đặt

```
pip install -r requirements.txt
```

Cần Python 3.8+ và OpenCV ≥ 4.4 (đã có sẵn SIFT). `data/templates_train/` và `data/mo_hinh_mau.npz` đã có sẵn;
nếu xóa thì tạo lại bằng `python src/tao_mau.py`.

## Sử dụng

Các lệnh chạy từ thư mục dự án (script tự xác định thư mục dự án nên chạy từ đâu cũng được):

```
python src/giao_dien.py                          # mở giao diện nhận diện
python src/nhan_dien.py                          # nhận diện data/anh_thu/query.jpg
python src/nhan_dien.py duong/dan/anh.jpg        # nhận diện ảnh bất kỳ
python src/danh_gia_sift.py --split test         # đánh giá trên tập test -> ket_qua/sift_test/
python src/xem_loi.py                            # bảng ảnh các tờ lỗi -> ket_qua/sift_test/loi/
python src/tao_mau.py                            # tạo lại ảnh mẫu train + mô hình màu
python src/tinh_chinh_nguong.py --split valid    # dò tham số trên tập valid -> ket_qua/do_tham_so/
```

## Giao diện

```
python src/giao_dien.py
```

- **Mở ảnh...** hoặc **Chụp từ webcam** để nạp ảnh (mặc định mở sẵn `data/anh_thu/query.jpg`).
- **Nhận diện toàn ảnh:** SIFT tự định vị tờ tiền và vẽ khung homography – dùng cho ảnh chỉ có một tờ.
- **Nhận diện vùng chọn:** kéo chuột khoanh vùng một tờ tiền rồi bấm nút – chính xác nhất (ứng với kịch bản
  "theo tờ tiền" khi đánh giá), dùng được cả với ảnh có nhiều tờ.
- **Nhận diện tất cả các tờ:** tự tìm từng tờ trong ảnh (lặp SIFT + RANSAC), vẽ khung, liệt kê và tính tổng tiền.
- Bảng kết quả hiển thị mệnh giá, số cặp điểm khớp (hoặc "theo màu"), thời gian và 3 mệnh giá xếp đầu theo màu.

## Cấu hình mặc định (SIFT + màu)

Chọn trên tập **valid** (accuracy cao nhất với precision ≥ 95%). Hằng số ở đầu `src/danh_gia_sift.py`,
dùng chung cho `nhan_dien.py` và `danh_gia_sift.py`:

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

Mọi hằng số đều có tùy chọn dòng lệnh trong `danh_gia_sift.py` (`--ransac`, `--min-inliers`, `--min-ratio`,
`--max-area`, `--margin`, `--k-train`, `--color-veto`, `--agree-min`, `--color-fallback`), cùng `--chi-mau r`
(chỉ dùng màu, trả lời khi d1/d2 ≤ r), `--out` (thư mục kết quả trong `ket_qua/`) và `--limit N`.

Chạy lại **tất cả** cấu hình so sánh, tuần tự trong cùng điều kiện để thời gian so sánh được (~45–60 phút):

```
bash src/chay_so_sanh.sh
```

## Kết quả trên tập test

**Phạm vi:** kịch bản "theo tờ tiền" nhận diện trên vùng cắt theo khung bao **có sẵn trong nhãn** (đã biết vị trí
từng tờ). Kịch bản "theo ảnh" chỉ áp dụng cho ảnh có đúng một tờ. Việc tự phát hiện nhiều tờ trong ảnh được đánh
giá riêng ở mục **Nhận diện nhiều tờ** bên dưới.

**Chỉ số:** *Accuracy* tính "không xác định" là sai; *Tỷ lệ trả lời* = số mẫu có kết quả / tổng số mẫu;
*Accuracy trên mẫu có kết quả* = số mẫu đúng / số mẫu có kết quả (khác với precision của từng lớp).
Accuracy = Tỷ lệ trả lời × Accuracy trên mẫu có kết quả.

Mọi giá trị tham số được chọn trên tập valid. Lưu ý: phân tích lỗi của cấu hình ban đầu được làm trên tập test và đã
định hướng việc thiết kế các cải tiến, nên kết quả test sau cải tiến có thể lạc quan hơn một chút (kết quả valid của
cấu hình mặc định: 87,06%).

**Theo tờ tiền** (558 tờ). Thời gian đo khi chạy tuần tự mọi cấu hình bằng `src/chay_so_sanh.sh` trên Intel Core
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

`detect_all()` trong `src/danh_gia_sift.py` lặp SIFT + RANSAC:

1. Đối sánh mọi điểm SIFT còn lại của ảnh với các ảnh mẫu, lấy giả thuyết homography mạnh nhất làm vị trí một tờ.
2. Quyết định mệnh giá bằng các ứng viên **ở cùng vị trí** (luật khoảng cách chỉ so với mệnh giá khớp vào cùng chỗ,
   không so với tờ khác trong ảnh) kết hợp màu trong khung của tờ đó.
3. Loại các điểm SIFT nằm trong khung vừa tìm, lặp lại đến khi giả thuyết mạnh nhất không còn đủ inlier.

Ảnh được phóng to ×1,5 trước khi trích SIFT (`MULTI_UPSCALE`) để tờ nhỏ có thêm điểm đặc trưng; mức này chọn trên
tập valid trong {1; 1,5; 2}. Đối sánh K láng giềng (K = 6) được tính một lần, các vòng sau chỉ lọc láng giềng còn
hoạt động.

```
python src/danh_gia_nhieu_to.py --split valid --phong-to 1 1.5 2 --workers 10   # chọn mức phóng to
python src/danh_gia_nhieu_to.py --split test --phong-to 1.5                     # đánh giá cuối -> ket_qua/nhieu_to_test/
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

## Ghi chú về dữ liệu

- **Lỗi tên lớp:** trong `_annotations.coco.json`, `category_id` 1–9 thực chất ứng với tên mệnh giá **sắp theo thứ
  tự chuỗi** (1000, 10000, 100000, 2000, 20000, 200000, 5000, 50000, 500000). `src/danh_gia_sift.py` đã tự sửa ánh
  xạ này; chương trình khác dùng bộ dữ liệu cũng cần sửa tương tự (xem `ket_qua/hinh_minh_hoa/kiem_tra_nhan_lop.jpg`).
- **Rò rỉ dữ liệu:** Roboflow chia ngẫu nhiên các khung hình liên tiếp và bản tăng cường của cùng ảnh gốc vào cả ba
  tập. `tao_mau.py` loại mọi ảnh train có mã dHash 256 bit cách ảnh valid/test bất kỳ ≤ 70 bit trước khi tạo ảnh
  mẫu và mô hình màu (còn 559/1.537 ảnh train).
- **Hạn chế:** mô hình màu học từ ảnh cùng bộ dữ liệu (cùng tờ tiền, nền, điều kiện chụp tương tự), nên với ảnh chụp
  ở môi trường khác hẳn (đèn vàng, cân bằng trắng lệch) màu có thể kém tin cậy hơn.

Chi tiết phương pháp, phân tích lỗi và thảo luận: xem `bao_cao/Bao_cao_Nhan_dien_tien_Viet_Nam.pdf`.
