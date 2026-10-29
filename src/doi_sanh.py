from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

# In được tiếng Việt cả khi chạy trong IDE hoặc chuyển hướng đầu ra trên Windows
sys.stdout.reconfigure(encoding="utf-8")

root = Path(__file__).resolve().parents[1]

template = cv2.imread(str(root / "data" / "templates" / "50000_01.jpg"))
query = cv2.imread(str(root / "data" / "anh_thu" / "query.jpg"))

if template is None or query is None:
    raise FileNotFoundError("Kiểm tra đường dẫn ảnh mẫu và anh_thu/query.jpg")

# Tìm điểm đặc trưng và bộ mô tả
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(
    cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), None
)
kp2, des2 = sift.detectAndCompute(
    cv2.cvtColor(query, cv2.COLOR_BGR2GRAY), None
)

if des1 is None or des2 is None or len(des2) < 2:
    raise ValueError("Ảnh không có đủ đặc trưng để đối sánh")

# Với mỗi điểm ở ảnh mẫu, tìm hai điểm gần nhất ở ảnh mới
matcher = cv2.BFMatcher(cv2.NORM_L2)
pairs = matcher.knnMatch(des1, des2, k=2)

# Giữ cặp khớp có ứng viên tốt nhất khác biệt rõ với ứng viên thứ hai
good_matches = []
for pair in pairs:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

print("Số cặp khớp ban đầu:", len(good_matches))
# Mỗi điểm trên ảnh query chỉ giữ cặp có khoảng cách nhỏ nhất
unique_matches = {}

for m in good_matches:
    if (m.trainIdx not in unique_matches
            or m.distance < unique_matches[m.trainIdx].distance):
        unique_matches[m.trainIdx] = m

good_matches = list(unique_matches.values())
if len(good_matches) < 4:
    raise ValueError("Cần ít nhất 4 cặp khớp để thử homography")

# Lấy tọa độ các điểm tương ứng trên hai ảnh
src = np.float32([
    kp1[m.queryIdx].pt for m in good_matches
]).reshape(-1, 1, 2)

dst = np.float32([
    kp2[m.trainIdx].pt for m in good_matches
]).reshape(-1, 1, 2)

# Ước lượng phép biến đổi, đồng thời lọc cặp khớp sai
H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

if H is None or mask is None:
    raise ValueError("Không tìm được phép biến đổi phù hợp")

inliers = mask.ravel().tolist()
count = sum(inliers)
ratio = count / len(good_matches)

# Ngưỡng thử ban đầu, cần điều chỉnh trên tập valid
if count < 12 or ratio < 0.5:
    print(
        f"Không khớp ảnh mẫu: "
        f"{count} cặp hợp lệ, tỷ lệ {ratio:.1%}"
    )
    raise SystemExit

print("Số cặp được RANSAC giữ lại:", count)
print(f"Tỷ lệ giữ lại: {count / len(good_matches):.1%}")

# Chiếu bốn góc ảnh mẫu lên ảnh query
h, w = template.shape[:2]
corners = np.float32([
    [0, 0], [w - 1, 0],
    [w - 1, h - 1], [0, h - 1]
]).reshape(-1, 1, 2)

projected = cv2.perspectiveTransform(corners, H)

query_draw = query.copy()
if np.isfinite(projected).all():
    cv2.polylines(
        query_draw,
        [np.int32(projected)],
        True, (0, 255, 0), 3
    )

# Chỉ vẽ các cặp được RANSAC giữ lại
result = cv2.drawMatches(
    template, kp1,
    query_draw, kp2,
    good_matches, None,
    matchesMask=inliers,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(14, 7))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()