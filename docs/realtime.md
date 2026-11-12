# Nhận diện thời gian thực: chi tiết thiết kế và nhật ký thử nghiệm

Tài liệu bổ sung cho mục **Nhận diện thời gian thực** trong [README](../README.md): cách `src/realtime.py` hoạt động, các cơ chế được thêm sau mỗi lần thử bằng webcam và số liệu dẫn tới từng lựa chọn.

Tóm tắt các cơ chế, chi tiết ở các mục con bên dưới:

- **Chặn vật không phải tiền** (thẻ ATM, thẻ bảo hiểm y tế…): chỉ báo mệnh giá khi SIFT xác nhận được hoa văn tờ tiền
  với khung homography gần chữ nhật; không đủ bằng chứng thì hiện khung xám "?" – xem
  [Chặn vật không phải tiền](#chặn-vật-không-phải-tiền).
- **Mặt sau bạc màu** (mặt sau tờ 20.000 dưới đèn phòng): trích SIFT lại sau khi tăng tương phản CLAHE khi lần đầu
  ra "?" – xem [Mặt sau bạc màu: CLAHE dự phòng](#mặt-sau-bạc-màu-clahe-dự-phòng).
- **Ổn định khung**: nhận diện ở luồng riêng, bám vết giữa các lần nhận diện, không chớp tắt – xem
  [Cách hoạt động](#cách-hoạt-động).

## Cách hoạt động

- Khung hình được hiển thị ngay. Việc nhận diện chạy ở luồng riêng trên khung hình mới nhất (khung đến trong lúc đang
  bận thì bỏ qua), nên hình không giật dù nhận diện chậm hơn tốc độ camera.
- **Chống chập chờn:** kết quả được bám theo giữa các lần nhận diện (ghép khung theo IoU ≥ 0,3). Tờ mới cần độ tin cậy
  YOLO ≥ 0,35; tờ đang bám theo được giữ bởi khung ≥ 0,25. Một vết chỉ bị xóa khi bộ nhận diện không thấy nó **2 vòng
  liên tiếp và** đã quá 0,6 giây (`--hold`; 1,5 giây với tờ do SIFT toàn khung hình tìm ra, vì đường này chỉ chạy mỗi
  0,5 giây) – vòng nhận diện chậm không làm khung tự tắt khi tờ vẫn còn trong hình. Vết cũ nằm chồng lên vết vừa cập
  nhật (tờ xoay làm khung mới lệch khỏi khung cũ) bị bỏ để không hiện hai khung cho một tờ.
- Ở chế độ `hybrid`, SIFT + màu (trên vùng cắt ≤ 400 px) chạy cho **tối đa một tờ mỗi vòng** (ưu tiên tờ mới, tờ đang
  "?", tờ YOLO đổi mệnh giá – dấu hiệu tờ khác vừa thế chỗ – rồi tờ quá 2 giây chưa kiểm tra, `--recheck`), để vòng
  nhận diện ngắn và vị trí các tờ được YOLO cập nhật liên tục; tờ chưa đến lượt dùng kết quả đã lưu hoặc tạm hiện "?".
  Nhãn hiển thị là kết quả SIFT + màu gần nhất (không bỏ phiếu trên kết quả đã lưu); lần kiểm tra lại "chưa rõ" chỉ giữ
  mệnh giá cũ thêm 2 giây.
- **Khi YOLO không có khung nào ≥ 0,25** (`hybrid`, tối đa mỗi 0,5 giây): thử SIFT + màu trên tối đa 2 khung YOLO
  yếu (0,10–0,25; `YOLO_WEAK`), chỉ nhận khi ≥ 8 cặp khớp; hai khung yếu cùng chứa một tờ (chồng ≥ 80%) thì giữ khung
  nhiều cặp khớp hơn. Không được mới chạy SIFT trên toàn khung hình. Khung của SIFT toàn khung hình (cả chế độ `sift`)
  là khung homography **cắt bớt phần ngoài hình chữ nhật bao các điểm khớp phóng to 1,8 lần**
  (`classify_image(..., inlier_rect=True)`): tờ bị gập / che một phần làm khung homography ngoại suy vượt xa tờ tiền.
- Chữ có dấu được dựng một lần thành ảnh nhỏ rồi dùng lại (vẽ: ~2 ms/khung thay vì ~11 ms), để luồng hiển thị không
  tranh tài nguyên với luồng nhận diện. `--record goc.mp4` ghi lại hình gốc từ webcam để chạy lại sau bằng `--source`
  (video từ tệp được phát đúng tốc độ, kể cả với `--no-show`, nên mô phỏng được webcam).
- Màn hình hiển thị khung, mệnh giá và bằng chứng của từng tờ (số cặp điểm khớp, "màu" hoặc độ tin cậy YOLO), tổng số
  tiền và tốc độ nhận diện. Phím: `1`/`2`/`3` đổi chế độ, `Space` tạm dừng, `S` lưu ảnh vào `results/realtime/`,
  `Q` hoặc `Esc` thoát.
- Chạy thử không cần webcam: `--no-show --sync --log log.csv` nhận diện mọi khung hình của video và ghi kết quả ra CSV.

## Chặn vật không phải tiền

Camera có thể thấy bất cứ vật gì, còn YOLO chỉ được huấn luyện trên ảnh có tờ tiền nên coi thẻ, giấy tờ hình chữ
nhật là tờ tiền. Các luật của SIFT + màu (`fuse()`) được dò trên bộ dữ liệu, nơi mọi vùng cắt chắc chắn là tờ tiền,
nên cần thêm một **cổng bằng chứng** trước khi báo mệnh giá:

- Code: `quad_angle_dev()` và `evidence_ok()` trong [`src/evaluate_sift.py`](../src/evaluate_sift.py); tham số
  `EVIDENCE_ANGLE`, `EVIDENCE_MIN`, `EVIDENCE_YOLO` trong [`src/realtime.py`](../src/realtime.py).
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
  | không có | 91,70% / 91,53% | 97,85% | 69,9% / 9,2% / 20,8% | 39 |
  | ≥ 8 inlier | 81,55% / 82,69% | 99,22% | 68,8% / 6,4% / 24,9% | – |
  | ≥ 8 inlier hoặc trùng nhãn YOLO | 91,90% / 92,13% | 99,37% | 69,9% / 7,5% / 22,5% | 12 |
  | như trên, nhưng khung ≤ 45°; màu chỉ được nhận khi YOLO đồng ý | 89,15% / 89,84%* | 98,98%* | 68,8% / 6,9% / 24,3% | 8 (đều 20.000) |
  | **khung ≤ 45°, ≥ 8 inlier hoặc YOLO đồng ý + ≥ 4 inlier** (dùng trong `realtime.py`) | 83,03% / 83,89% | **99,37%** | 67,6% / **5,8%** / 26,6% | **0** |

  Hai cột F1 và precision đo lại với YOLO 10 lớp (`results/hybrid_*_crop400/report_margin.txt`); dòng có dấu * không
  còn trong `MARGIN_RULES`, giữ số đo với YOLO 9 lớp lúc đó. Các cột webcam và vùng cắt thẻ đo lúc thử (YOLO 9 lớp).

  Đánh đổi: trên bộ dữ liệu (tờ nhỏ chụp xa, ít điểm khớp, trước đây dựa vào màu) mất ~9 điểm F1 – các tờ này hiện
  "?" thay vì có nhãn; trên webcam (tờ ở gần) số kết quả đúng gần như giữ nguyên. Với chương trình thời gian thực,
  báo sai mệnh giá cho vật không phải tiền tệ hơn hiện "?", nên chọn cổng chặt nhất không để lọt thẻ. Kèm theo: vết
  mất dấu > 0,5 giây rồi được tìm lại thì nhận diện lại ngay (có thể đã là tờ khác ở cùng chỗ). Video thử B (xoay,
  có khoảng trống; trung bình 3 lần): đúng đủ 51% (trước 62%), **0% nhãn sai, 0 lần tắt**.
  Hạn chế còn lại: YOLO vẫn khoanh vùng tấm thẻ (hiện khung "?"); muốn YOLO bỏ qua hẳn vật không phải tiền cần huấn
  luyện lại với ảnh âm (thẻ, giấy tờ, ví…) không gắn nhãn.

## Mặt sau bạc màu: CLAHE dự phòng

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
| tắt | 83,03% / 83,89% | 67,6% / 5,8% / 26,6% | 0 |
| luôn bật, clip 2 | 84,73% / 84,13% | 74,6% / 4,0% / 21,4% | 0 |
| luôn bật, clip 4 | 78,66% / 79,16% | 69,4% / 4,0% / 26,6% | 0 |
| **chỉ khi ra "?", clip 2** (dùng trong `realtime.py`) | **88,36% / 86,92%** | **78,6%** / 6,9% / **14,5%** | **0** |

Cột F1 đo lại với YOLO 10 lớp; dòng cuối tính bằng `python src/evaluate_hybrid.py --crop-max 400 --clahe-du-phong 2`
(ghép kết quả của lần chạy `--crop-max 400` và `--crop-max 400 --clahe 2`, ghi ra
`results/hybrid_*_crop400_clahe2/report_clahe_du_phong.txt`). Các cột webcam và vùng cắt thẻ đo lúc thử (YOLO 9 lớp).

Ảnh webcam: mặt sau tờ 20.000 nhận được ở 2/6 ảnh (trước 0/6; các ảnh còn lại tờ bị cong hoặc nghiêng mạnh);
38 ảnh của các lần thử trước: đúng 21 / sai 3 / "?" 14 (trước 19 / 3 / 16); video thử B (trung bình 3 lần): đúng đủ
~49%, không còn lần tắt. Vết mất dấu rồi được tìm lại không còn giữ mệnh giá cũ trong lúc chờ nhận diện lại.

## Lỗi đã biết: mặt trước tờ 20.000 / 500.000

**Mặt trước tờ 20.000 từng bị báo 500.000** (23 khớp): mặt trước 20.000 và 500.000 cùng màu xanh, cùng chân dung;
chạy lại trên ảnh chụp thì SIFT ra đúng 20.000 (24 khớp) nhưng 500.000 sát nút (22 khớp). Đã thử luật "mâu thuẫn với
YOLO" (SIFT chọn khác YOLO trong khi ứng viên của mệnh giá YOLO cũng mạnh gần ngang → "?"): trên valid gần như
không đổi, trên bản ghi webcam giảm nhãn sai (4,0% → 2,3% với ρ = 0,7) nhưng giảm mạnh số kết quả đúng (74,6% →
68,2%) vì YOLO hay sai trên webcam – chưa đưa vào.

**Mặt trước tờ 100.000 bị báo 500.000** (lần thử thứ sáu): mọi tờ polymer cùng chân dung, quốc huy, quốc hiệu ở mặt
trước, vùng cắt khớp 28–41 cặp với cả 500.000, 10.000, 50.000, 20.000, 200.000, 100.000; màu xếp 100.000 đầu nhưng
SIFT vượt luật khoảng cách. Chưa sửa.

## Nhật ký thử bằng webcam

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
  93,27% khi một khung hình chỉ được nhận diện một lần, đo với YOLO 9 lớp lúc đó; YOLO 10 lớp: 91,30% so với 94,46%), nhưng trên video mỗi tờ được thử lại nhiều lần. Tùy chọn
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
  | tổng inlier ≥ 1,3× (mặc định dự án) | 91,65% / 91,70% | 59,5% / 9,8% / 30,6% | 16 / 2 |
  | **inlier riêng ≥ 1,3×** (dùng trong `realtime.py`) | 91,70% / 91,53% | **69,9% / 9,2% / 20,8%** | **19 / 3** |
  | inlier riêng ≥ 1,5× | 91,74% / 91,51% | 64,7% / 9,2% / 26,0% | |
  | inlier riêng ≥ 2× | 91,22% / 91,38% | 56,6% / 9,8% / 33,5% | |

  Cột F1 đo lại với YOLO 10 lớp; các cột webcam đo lúc thử (YOLO 9 lớp).

  Trên bộ dữ liệu (tờ nhỏ, chụp xa) hai luật gần như ngang nhau; trên webcam (tờ ở gần, chi tiết chung chiếm nhiều
  điểm khớp) luật inlier riêng giảm mạnh số "?" mà không tăng nhãn sai. Bản ghi webcam không có nhãn: "đúng" = 100.000
  hoặc 500 đồng (hai tờ duy nhất được dùng trong buổi thử), mọi nhãn khác tính là sai. Lưu ý chưa có bản ghi webcam
  độc lập để kiểm tra lại lựa chọn này.
- Lần thử webcam thứ sáu (sau khi đổi sang YOLO 10 lớp, 5 ảnh chụp lỗi): ba lỗi do **YOLO không thấy tờ cầm tay sát
  camera** (cả mô hình 9 lớp cũ lẫn mô hình mới chỉ cho 0–0,27), nên phải dùng SIFT toàn khung hình: khung méo vượt ra
  ngoài tờ 100.000 bị gập, và tờ 2.000 bị báo 1.000 (khung hình thu về 640 px: 17 cặp khớp 1.000 / 15 cặp 2.000; cắt
  đúng vùng tờ thì 45 / 27, ra 2.000). Trên bản ghi webcam lần 5, YOLO mới thấy tờ (≥ 0,35) ở 397/1.269 khung hình
  lấy mẫu, mô hình cũ chỉ 83. Đã thêm bước thử khung YOLO yếu và cắt khung SIFT (mục **Cách hoạt động**). Đo trên bản
  ghi lần 5 (1/30 khung hình, 86 khung chỉ có khung YOLO yếu, kiểm tra bằng mắt): khung có nhãn tăng từ 29 lên 44
  (thêm 14 khung đúng, trong đó 8 khung tờ 500), nhưng nhãn sai mệnh giá tăng từ 2 lên 6 – vùng cắt mặt trước tờ
  100.000 bị báo 500.000 (lỗi chung của mặt trước polymer, xem mục **Lỗi đã biết**). Tắt bước này: đặt
  `YOLO_WEAK = YOLO_KEEP` trong `realtime.py`.

## Chạy thử trên video

**Chạy thử** trên video tổng hợp từ 8 ảnh test sạch (4 ảnh một tờ, 4 ảnh 3–5 tờ; mỗi ảnh 10 khung hình có trượt nhẹ),
một cảnh được tính đúng khi nhãn đa số trong 7 khung cuối khớp đúng toàn bộ các tờ: `hybrid` 6/8, `yolo` 5/8, `sift`
2/8 (`sift` chỉ nhận một tờ nên sai cả 4 cảnh nhiều tờ). Hai lỗi còn lại của `hybrid`: tờ 100.000 đồng trong một khung
hình video mờ, ám xanh (SIFT chỉ được ~10 cặp khớp, màu cũng xếp 200.000 đầu) và một tờ 500.000 đồng nhỏ mà YOLO không
tìm thấy. Đây chỉ là bài thử chức năng, không thay cho các bảng đánh giá ở trên.
