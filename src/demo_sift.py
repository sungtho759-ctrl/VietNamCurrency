from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import sys

# In được tiếng Việt cả khi chạy trong IDE hoặc chuyển hướng đầu ra trên Windows
sys.stdout.reconfigure(encoding="utf-8")

# Đọc ảnh mẫu
ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data" / "templates" / "500000_01.jpg"
img = cv2.imread(str(path))

if img is None:
    raise FileNotFoundError(f"Không đọc được ảnh: {path}")
# Chuyển sang ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Tìm điểm đặc trưng và mô tả chúng
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

print("Số điểm đặc trưng:", len(keypoints))
if descriptors is not None:
    print("Kích thước bộ mô tả:", descriptors.shape)

# Vẽ các điểm đặc trưng lên ảnh
result = cv2.drawKeypoints(
    img, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()