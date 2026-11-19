# Triển khai giao diện web lên Vercel

Tài liệu bổ sung cho mục **Triển khai trên Vercel** trong [README](../README.md): đưa giao diện web **VND Vision AI**
(`src/web_app.py`) lên Vercel dưới dạng một hàm Python serverless, để mở được từ điện thoại hay máy khác qua HTTPS mà
không cần cài Python.

## Kiến trúc

```
Trình duyệt ──HTTPS──> Vercel ──(mọi đường dẫn, vercel.json: rewrites)──> api/index.py (hàm Python)
                                                                           └─ Flask app của src/web_app.py
                                                                              ├─ /            web/ (index.html, app.js…)
                                                                              ├─ /api/sift    SIFT + RANSAC + màu HSV
                                                                              ├─ /api/yolo    YOLO11n: best.onnx qua cv2.dnn
                                                                              ├─ /api/hybrid  YOLO định vị + SIFT + màu nhận diện
                                                                              └─ /api/info, /samples/…
```

Mã nhận diện giữ nguyên, chỉ khác cách chạy YOLO và thời điểm nạp mô hình:

| | Chạy cục bộ (`python src/web_app.py`) | Trên Vercel (`api/index.py`) |
|---|---|---|
| YOLO11n | `best.pt` qua ultralytics + PyTorch | `best.onnx` qua `cv2.dnn` (`src/yolo_onnx.py`) |
| Thư viện | `requirements.txt` + `requirements_yolo.txt` | `api/requirements.txt`: flask, numpy, opencv-python-headless |
| Nạp mô hình | Khi khởi động (`main()`) | Ở yêu cầu đầu tiên của mỗi phiên bản hàm (`models()`) |
| Ảnh gửi lên | Thu về cạnh dài ≤ 1.600 px trong trình duyệt (`web/app.js`) | như bên trái |

`web_app.py` tự chọn: có ultralytics thì dùng `best.pt`, không có thì dùng `best.onnx`. Ô "YOLO11n" trên đầu trang
(di chuột vào) và `/api/info` (`yolo_backend`: `pt` / `onnx`) cho biết đang dùng bản nào.

### Vì sao không dùng thẳng PyTorch / ultralytics

Vercel giới hạn gói hàm Python ở **250 MB** sau khi giải nén. Riêng PyTorch bản CPU đã vượt con số này. Đo gói thư
viện Linux (Python 3.12, `pip download --platform manylinux…`):

| Bộ thư viện | Dung lượng giải nén |
|---|---|
| flask + numpy + opencv-python-headless + **onnxruntime** | ~295 MB – vượt giới hạn |
| flask + numpy + opencv-python-headless (YOLO chạy bằng `cv2.dnn`) | **~220 MB** + 15 MB dữ liệu – dưới giới hạn |

Vì vậy YOLO được xuất sang ONNX và chạy bằng mô-đun `dnn` của OpenCV, thư viện mà phương pháp truyền thống đã cần sẵn.
matplotlib và Pillow cũng được bỏ khỏi đường chạy của web (`realtime.py`, nơi `web_app.py` lấy phần SIFT + màu của
chế độ kết hợp, chỉ nạp Pillow khi vẽ chữ lên cửa sổ webcam): `evaluate_sift.py` chỉ nạp matplotlib khi vẽ ma trận nhầm lẫn, còn
phép kiểm tra điểm trong khung tờ tiền (trước đây `matplotlib.path.Path.contains_points`) được viết lại bằng numpy
(`points_in_polygon`, quy tắc chẵn–lẻ như matplotlib; cho kết quả trùng khớp trên 1,5 triệu điểm ngẫu nhiên, kể cả tứ
giác tự cắt).

### YOLO ONNX khớp với best.pt

`src/yolo_onnx.py` làm lại tiền xử lý và hậu xử lý của ultralytics 8.4: letterbox cạnh dài 640 px, chỉ đệm tới bội của
32 (ảnh chữ nhật không bị đệm thành 640 × 640), NMS với IoU **0,5** (`PREDICT_IOU` của ultralytics 8.4, không phải 0,7
như các bản cũ), cùng ngưỡng tin cậy 0,35 và NMS không phân lớp chọn trên valid. Tên lớp và stride đọc từ metadata mà
ultralytics ghi vào tệp ONNX.

So sánh trên 229 ảnh `data/dataset/test/` + 5 ảnh `data/test_images/` (234 ảnh): **234/234 ảnh cùng số khung, cùng mệnh giá, tọa độ
lệch 0,00 px** so với `best.pt`. Các chỉ số YOLO trong README vì vậy vẫn đúng cho bản trên Vercel. Đổi lại `cv2.dnn`
chậm hơn: ~0,2–0,3 s/ảnh trên CPU i5-12450H (PyTorch 0,10 s, onnxruntime 0,09 s).

Tạo lại `best.onnx` sau khi huấn luyện lại (trên máy có ultralytics):

```
yolo export model=results/yolo/yolo11n/weights/best.pt format=onnx imgsz=640 dynamic=True opset=17
```

`dynamic=True` là bắt buộc để nhận ảnh chữ nhật như `best.pt`.

## Các tệp liên quan

| Tệp | Vai trò |
|---|---|
| `vercel.json` | Chuyển mọi đường dẫn về `api/index.py`; thời gian chạy tối đa 60 s; liệt kê tệp dữ liệu đưa vào hàm (`includeFiles`) |
| `.vercelignore` | Không tải lên bộ dữ liệu, kết quả đánh giá, `*.pt`… (~450 MB); chỉ giữ ảnh mẫu, mô hình màu, `best.onnx`, `results.json` |
| `api/index.py` | Điểm vào hàm: thêm `src/` vào `sys.path` rồi dùng lại `app` của `web_app.py` |
| `api/requirements.txt` | Thư viện của hàm (Vercel dùng tệp này, không dùng `requirements.txt` ở gốc vì nó có matplotlib và opencv-python bản có GUI) |
| `src/yolo_onnx.py` | Chạy YOLO11n từ `best.onnx` bằng `cv2.dnn` |
| `results/yolo/yolo11n/weights/best.onnx` | Trọng số YOLO11n dạng ONNX (11 MB) |

## Các bước triển khai

### Cách 1: từ GitHub (khuyên dùng)

1. Đẩy mã lên GitHub (kho `VietNamCurrency`), gồm cả `best.onnx`.
2. Vào <https://vercel.com/new>, đăng nhập bằng GitHub, chọn **Import** kho này.
3. Ở màn hình cấu hình: **Framework Preset** để `Other`, **Root Directory** để `./`, không cần Build Command hay biến
   môi trường. Bấm **Deploy**.
4. Sau 1–3 phút, Vercel cho địa chỉ dạng `https://<tên-dự-án>.vercel.app`. Mỗi lần `git push` lên nhánh `main`, Vercel
   tự triển khai lại; nhánh khác được bản xem trước (Preview) với địa chỉ riêng.

### Cách 2: bằng Vercel CLI

Cần Node.js (để có `npm`):

```
npm install -g vercel
vercel login
vercel          # lần đầu: tạo dự án, triển khai bản Preview
vercel --prod   # triển khai bản chính thức
```

Chạy các lệnh trong thư mục dự án. CLI tải lên đúng những tệp không bị `.vercelignore` loại (~30 MB).

### Kiểm tra sau khi triển khai

- Mở `https://<tên-dự-án>.vercel.app/api/info`: phải có `"yolo": true`, `"yolo_backend": "onnx"`,
  `"sift_templates": 20`, `"color_model": true`.
- Mở trang chủ, chọn một ảnh mẫu: hai cột Truyền thống và YOLO11n đều hiện khung và tổng tiền.
- Nếu lỗi, xem nhật ký ở **Project → Deployments → (bản triển khai) → Functions / Logs**: lỗi thiếu tệp (đường dẫn
  `data/…`, `results/…`) nghĩa là `includeFiles` hoặc `.vercelignore` chưa đúng; lỗi "exceeds the maximum size" nghĩa
  là gói hàm vượt 250 MB (kiểm tra `api/requirements.txt` không kéo thêm torch / onnxruntime / matplotlib).

### Thử trước trên máy (giả lập môi trường Vercel)

Chặn ultralytics, torch và matplotlib rồi gọi thẳng `app` của `api/index.py`:

```
python -c "import sys, runpy; [sys.modules.__setitem__(m, None) for m in ('ultralytics','torch','matplotlib')]; app = runpy.run_path('api/index.py')['app']; print(app.test_client().get('/api/info').get_json())"
```

Kết quả mong đợi: `'yolo': True, 'yolo_backend': 'onnx'`. Khi thử như vậy, khởi động lạnh mất ~4 s (nạp ảnh mẫu SIFT
hai lần: thường và CLAHE cho chế độ kết hợp), SIFT ~0,35 s, YOLO ~0,04–0,3 s và kết hợp ~0,15–2 s cho một ảnh 640 × 640
(tùy số tờ, mỗi tờ một lần SIFT + màu).

## Giới hạn khi chạy trên Vercel

- **Khởi động lạnh:** phiên bản hàm mới phải nạp thư viện, 20 ảnh mẫu SIFT, mô hình màu và YOLO (vài giây). Yêu cầu
  đầu tiên sau một thời gian không dùng chậm hơn hẳn; các yêu cầu sau nhanh lại.
- **Kích thước ảnh:** Vercel từ chối thân yêu cầu lớn hơn 4,5 MB. `web/app.js` đã thu ảnh về cạnh dài 1.600 px (JPEG,
  thường < 1 MB) trước khi gửi; máy chủ vốn chỉ dùng tới 800 px cho SIFT và 640 px cho YOLO nên kết quả không đổi. Gọi
  API trực tiếp (không qua trang web) thì cần tự thu nhỏ ảnh.
- **Thời gian:** mỗi yêu cầu tối đa 60 s (`maxDuration`, mức tối đa của gói Hobby). SIFT tự tìm mọi tờ mất ~2 s/ảnh
  trên máy thử, CPU của Vercel có thể chậm hơn nhưng vẫn xa giới hạn này.
- **Quét trực tiếp từ camera:** chạy được vì trang có HTTPS (trình duyệt chỉ cho mở camera trên HTTPS hoặc localhost).
  Tuy nhiên mỗi khung hình là một yêu cầu qua mạng, nên "Nhận diện liên tục" chậm hơn nhiều so với chạy cục bộ và tiêu
  tốn thời gian chạy hàm (gói Hobby có hạn mức hằng tháng). Nên dùng "Chụp & nhận diện" hoặc tải ảnh lên.
- **Không có trạng thái:** mỗi yêu cầu có thể rơi vào một phiên bản hàm khác; ứng dụng không lưu gì giữa các yêu cầu
  nên không bị ảnh hưởng. Khóa `sift_lock` / `yolo_lock` chỉ có tác dụng trong một phiên bản.
- **Công khai:** địa chỉ `*.vercel.app` ai có cũng mở được. Ảnh gửi lên chỉ được xử lý trong bộ nhớ, không lưu lại.
  Muốn giới hạn người xem, bật **Vercel Authentication** trong **Project → Settings → Deployment Protection**.
- **Chỉ giao diện web:** `src/realtime.py` (cửa sổ OpenCV đọc webcam) và `src/gui.py` (Tkinter) là ứng dụng máy tính,
  không triển khai lên Vercel được.
