"""Điểm vào hàm Python trên Vercel: dùng lại ứng dụng Flask của src/web_app.py (xem docs/vercel.md).

vercel.json chuyển mọi đường dẫn về hàm này; mô hình được nạp ở yêu cầu đầu tiên của mỗi phiên bản hàm.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web_app import app  # noqa: E402,F401
