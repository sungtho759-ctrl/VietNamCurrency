#!/usr/bin/env bash
# Chạy lại tuần tự mọi cấu hình so sánh trên tập test trong cùng điều kiện (một tiến trình,
# không chạy việc nặng khác song song) để thời gian xử lý so sánh được với nhau.
# Chạy từ thư mục dự án:  bash src/chay_so_sanh.sh
set -e
export MPLBACKEND=Agg
run() { echo "== $1"; shift; python src/danh_gia_sift.py --split test "$@" | grep -v "^  [0-9]*/"; }

run "Ban đầu (chỉ SIFT)" --ransac 5 --min-inliers 12 --min-ratio 0.5 --max-area 3 --margin 1.0 \
    --agree-min 0 --color-fallback 0 --out so_sanh/ban_dau
run "Tinh chỉnh ngưỡng (chỉ SIFT)" --ransac 12 --min-inliers 7 --min-ratio 0.25 --margin 1.0 \
    --agree-min 0 --color-fallback 0 --out so_sanh/tinh_chinh_nguong
run "Luật khoảng cách (chỉ SIFT)" --ransac 10 --min-inliers 8 --margin 1.1 \
    --agree-min 0 --color-fallback 0 --out so_sanh/luat_khoang_cach
run "Chỉ màu, luôn trả lời" --chi-mau 1.0 --out so_sanh/chi_mau
# Ngưỡng 0,9 chọn trên valid theo cùng quy tắc với SIFT + màu (accuracy trên mẫu có kết quả >= 95%)
run "Chỉ màu, từ chối khi d1/d2 > 0,9" --chi-mau 0.9 --out so_sanh/chi_mau_tu_choi
run "SIFT + màu (mặc định)"
run "SIFT + 8 ảnh mẫu train + màu" --k-train 8 --ransac 10 --min-inliers 6 --margin 1.3 --color-veto 2 \
    --out so_sanh/mau_train_va_mau
