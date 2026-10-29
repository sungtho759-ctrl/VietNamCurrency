"""Giao diện nhận diện mệnh giá tiền Việt Nam (Tkinter).

- Mở ảnh từ máy hoặc chụp từ webcam.
- Nhận diện toàn ảnh: SIFT tự định vị tờ tiền (ảnh chỉ nên có một tờ), vẽ khung homography.
- Nhận diện vùng chọn: kéo chuột khoanh vùng một tờ tiền rồi nhận diện vùng đó
  (giống kịch bản "theo tờ tiền" khi đánh giá – chính xác nhất).
- Nhận diện tất cả các tờ: lặp SIFT + RANSAC để tìm từng tờ trong ảnh, hiển thị danh sách và tổng tiền.

Dùng chung logic và cấu hình mặc định với danh_gia_sift.py.

Chạy:  python src/giao_dien.py
"""
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from danh_gia_sift import (MULTI_UPSCALE, ROOT, UNKNOWN, classify, classify_image, crop_box, detect_all,
                           load_templates)
from mau_sac import MODEL_PATH, ColorModel, hsv_hist

# In được tiếng Việt cả khi chạy trong IDE hoặc chuyển hướng đầu ra trên Windows
sys.stdout.reconfigure(encoding="utf-8")

CANVAS_W, CANVAS_H = 860, 600
GREEN, RED, BLUE = "#1f9d55", "#d64545", "#2563eb"


def vnd(label):
    return f"{int(label):,}".replace(",", ".") + " đồng"


class App:
    def __init__(self, root):
        self.root = root
        root.title("Nhận diện mệnh giá tiền Việt Nam – SIFT + RANSAC + màu HSV")
        root.minsize(1180, 660)

        self.templates = None
        self.color_model = None
        self.image = None          # ảnh BGR gốc
        self.scale = 1.0           # tỉ lệ ảnh hiển thị / ảnh gốc
        self.offset = (0, 0)       # vị trí ảnh trên khung vẽ
        self.tk_image = None
        self.sel_start = None
        self.selection = None      # (x, y, w, h) theo tọa độ ảnh gốc
        self.busy = False

        # Khung vẽ ảnh
        self.canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H, bg="#f1f3f5", highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.create_text(CANVAS_W // 2, CANVAS_H // 2, fill="#868e96", font=("Segoe UI", 13),
                                text="Mở ảnh hoặc chụp từ webcam để bắt đầu")

        # Bảng điều khiển
        panel = tk.Frame(root, padx=8)
        panel.grid(row=0, column=1, sticky="ns", pady=12)
        btn = dict(font=("Segoe UI", 11), width=26, pady=4)
        tk.Button(panel, text="Mở ảnh...", command=self.open_image, **btn).pack(pady=3)
        tk.Button(panel, text="Chụp từ webcam", command=self.capture_webcam, **btn).pack(pady=3)
        tk.Frame(panel, height=10).pack()
        self.btn_full = tk.Button(panel, text="Nhận diện toàn ảnh", command=self.recognize_full,
                                  bg="#e7f0ff", **btn)
        self.btn_full.pack(pady=3)
        self.btn_sel = tk.Button(panel, text="Nhận diện vùng chọn", command=self.recognize_selection,
                                 bg="#e7f0ff", **btn)
        self.btn_sel.pack(pady=3)
        self.btn_all = tk.Button(panel, text="Nhận diện tất cả các tờ", command=self.recognize_all,
                                 bg="#e7f0ff", **btn)
        self.btn_all.pack(pady=3)
        tk.Label(panel, text="Kéo chuột trên ảnh để khoanh vùng một tờ tiền.\n"
                             "Vùng chọn cho kết quả chính xác nhất.",
                 font=("Segoe UI", 9), fg="#495057", justify="left").pack(pady=(4, 14), anchor="w")

        tk.Label(panel, text="Kết quả", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.result = tk.Label(panel, text="—", font=("Segoe UI", 22, "bold"), fg="#212529",
                               width=16, anchor="w")
        self.result.pack(anchor="w", pady=(2, 6))
        self.detail = tk.Label(panel, text="", font=("Consolas", 10), fg="#343a40", justify="left",
                               anchor="w", width=34)
        self.detail.pack(anchor="w")

        self.status = tk.Label(root, text="Đang nạp ảnh mẫu và mô hình màu...", anchor="w",
                               font=("Segoe UI", 9), fg="#495057", bd=1, relief="sunken", padx=6)
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.set_busy(True)
        threading.Thread(target=self.load_models, daemon=True).start()

    # ---------------------------------------------------------------- nạp mô hình
    def load_models(self):
        start = time.perf_counter()
        templates = load_templates()
        color = ColorModel() if MODEL_PATH.exists() else None
        msg = f"Sẵn sàng: {len(templates)} ảnh mẫu" + (", mô hình màu" if color else ", chưa có mô hình màu (chạy src/tao_mau.py)")
        msg += f" – nạp trong {time.perf_counter() - start:.1f} s"
        self.root.after(0, self._models_loaded, templates, color, msg)

    def _models_loaded(self, templates, color, msg):
        self.templates, self.color_model = templates, color
        self.status.config(text=msg)
        self.set_busy(False)
        sample = ROOT / "data" / "anh_thu" / "query.jpg"
        if self.image is None and sample.exists():
            self.load_image(cv2.imread(str(sample)), sample.name)

    def set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.btn_full.config(state=state)
        self.btn_sel.config(state=state)
        self.btn_all.config(state=state)
        self.root.config(cursor="watch" if busy else "")

    # ---------------------------------------------------------------- ảnh vào
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh tờ tiền", initialdir=ROOT / "data" / "anh_thu",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tất cả", "*.*")])
        if not path:
            return
        # imdecode đọc được cả đường dẫn có dấu tiếng Việt trên Windows
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Lỗi", f"Không đọc được ảnh:\n{path}")
            return
        self.load_image(img, path)

    def capture_webcam(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không mở được webcam.")
            return
        frame = None
        for _ in range(10):  # bỏ vài khung đầu để camera tự cân bằng sáng
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        if frame is None:
            messagebox.showerror("Lỗi", "Không chụp được ảnh từ webcam.")
            return
        self.load_image(frame, "ảnh chụp từ webcam")

    def load_image(self, img, name):
        self.image = img
        self.selection = None
        self.show(img)
        self.result.config(text="—", fg="#212529")
        self.detail.config(text="")
        h, w = img.shape[:2]
        self.status.config(text=f"Đã mở {name} ({w}×{h}). Bấm 'Nhận diện toàn ảnh' hoặc kéo chuột chọn vùng tờ tiền.")

    def show(self, img):
        h, w = img.shape[:2]
        self.scale = min(CANVAS_W / w, CANVAS_H / h)
        dw, dh = int(w * self.scale), int(h * self.scale)
        rgb = cv2.cvtColor(cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.offset = ((CANVAS_W - dw) // 2, (CANVAS_H - dh) // 2)
        self.canvas.delete("all")
        self.canvas.create_image(*self.offset, image=self.tk_image, anchor="nw")

    def to_canvas(self, pts):
        return [(x * self.scale + self.offset[0], y * self.scale + self.offset[1]) for x, y in pts]

    def to_image(self, cx, cy):
        h, w = self.image.shape[:2]
        x = min(max((cx - self.offset[0]) / self.scale, 0), w - 1)
        y = min(max((cy - self.offset[1]) / self.scale, 0), h - 1)
        return x, y

    # ---------------------------------------------------------------- chọn vùng
    def on_press(self, e):
        if self.image is None:
            return
        self.sel_start = (e.x, e.y)
        self.canvas.delete("sel", "result")

    def on_drag(self, e):
        if self.sel_start is None:
            return
        self.canvas.delete("sel")
        self.canvas.create_rectangle(*self.sel_start, e.x, e.y, outline=BLUE, width=2, dash=(6, 3), tags="sel")

    def on_release(self, e):
        if self.sel_start is None:
            return
        (x0, y0), (x1, y1) = self.to_image(*self.sel_start), self.to_image(e.x, e.y)
        self.sel_start = None
        x, y, w, h = min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
        if w < 20 or h < 20:
            self.canvas.delete("sel")
            self.selection = None
            return
        self.selection = (x, y, w, h)
        self.status.config(text=f"Đã chọn vùng {int(w)}×{int(h)} px. Bấm 'Nhận diện vùng chọn'.")

    # ---------------------------------------------------------------- nhận diện
    def recognize_full(self):
        if self.image is None or self.busy:
            return
        self.run_in_background(self._recognize_full)

    def recognize_selection(self):
        if self.image is None or self.busy:
            return
        if self.selection is None:
            messagebox.showinfo("Chưa chọn vùng", "Hãy kéo chuột trên ảnh để khoanh vùng một tờ tiền.")
            return
        self.run_in_background(self._recognize_selection)

    def recognize_all(self):
        if self.image is None or self.busy:
            return
        self.set_busy(True)
        self.status.config(text="Đang tìm các tờ tiền trong ảnh...")

        def task():
            start = time.perf_counter()
            try:
                found = detect_all(self.image, self.templates, self.color_model, upscale=MULTI_UPSCALE)
            except Exception as exc:
                self.root.after(0, lambda: (self.set_busy(False), messagebox.showerror("Lỗi", str(exc))))
                return
            self.root.after(0, self.show_all, found, time.perf_counter() - start)

        threading.Thread(target=task, daemon=True).start()

    def show_all(self, found, seconds):
        self.set_busy(False)
        self.canvas.delete("result", "sel")
        h, w = self.image.shape[:2]
        frame = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        for i, (label, _, box) in enumerate(found, 1):
            area, inside = cv2.intersectConvexConvex(np.float32(box), frame)
            pts = inside.reshape(-1, 2) if area > 0 else np.float32(box)
            flat = [v for p in self.to_canvas(pts) for v in p]
            self.canvas.create_polygon(*flat, outline=GREEN, fill="", width=3, tags="result")
            tx, ty = min(flat[0::2]), min(flat[1::2])
            self.canvas.create_text(tx + 4, max(ty - 4, 12), anchor="sw", tags="result", fill=GREEN,
                                    font=("Segoe UI", 12, "bold"), text=f"{i}. {vnd(label)}")
        total = sum(int(label) for label, _, _ in found)
        if found:
            self.result.config(text=f"{len(found)} tờ – {vnd(total)}" if len(found) > 1 else vnd(total), fg=GREEN)
        else:
            self.result.config(text="Không tìm thấy", fg=RED)
        lines = ["Chế độ      : tất cả các tờ", f"Thời gian   : {seconds:.2f} s", ""]
        lines += [f"  {i}. {vnd(label):<16} {n:>3} cặp khớp" if n else f"  {i}. {vnd(label):<16} theo màu"
                  for i, (label, n, _) in enumerate(found, 1)]
        if found:
            lines += ["", f"Tổng cộng   : {vnd(total)}"]
        lines += ["", "Tờ nhỏ, nhòe, chồng nhau có thể", "bị sót: khi đó khoanh từng tờ và",
                  "dùng 'Nhận diện vùng chọn'."]
        self.detail.config(text="\n".join(lines))
        self.status.config(text=f"Tìm thấy {len(found)} tờ trong {seconds:.2f} s.")

    def run_in_background(self, work):
        self.set_busy(True)
        self.status.config(text="Đang nhận diện...")

        def task():
            start = time.perf_counter()
            try:
                out = work()
            except Exception as exc:  # hiển thị lỗi thay vì làm treo giao diện
                self.root.after(0, lambda: (self.set_busy(False), messagebox.showerror("Lỗi", str(exc))))
                return
            self.root.after(0, self.show_result, *out, time.perf_counter() - start)

        threading.Thread(target=task, daemon=True).start()

    def _recognize_full(self):
        label, inliers, box = classify_image(self.image, self.templates, self.color_model)
        return "toàn ảnh", label, inliers, box, None

    def _recognize_selection(self):
        crop = crop_box(self.image, self.selection)
        rank = self.color_model.rank(hsv_hist(crop)) if self.color_model else None
        label, inliers = classify(crop, self.templates, color_rank=rank)
        x, y, w, h = self.selection
        box = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        return "vùng chọn", label, inliers, box, rank

    def show_result(self, mode, label, inliers, box, rank, seconds):
        self.set_busy(False)
        self.canvas.delete("result", "sel")
        known = label != UNKNOWN
        color = GREEN if known else RED
        if box is not None and (known or mode == "vùng chọn"):
            # Tờ bị cắt mép làm khung homography vượt ra ngoài ảnh: chỉ vẽ phần nằm trong ảnh
            h, w = self.image.shape[:2]
            frame = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            area, inside = cv2.intersectConvexConvex(np.float32(box), frame)
            pts = inside.reshape(-1, 2) if area > 0 else np.float32(box)
            flat = [v for p in self.to_canvas(pts) for v in p]
            self.canvas.create_polygon(*flat, outline=color, fill="", width=3, tags="result")
            tx, ty = min(flat[0::2]), min(flat[1::2])
            self.canvas.create_text(tx + 4, max(ty - 4, 12), anchor="sw", tags="result", fill=color,
                                    font=("Segoe UI", 13, "bold"), text=vnd(label) if known else "Không xác định")

        self.result.config(text=vnd(label) if known else "Không xác định", fg=color)
        lines = [f"Chế độ      : {mode}",
                 f"Bằng chứng  : {f'{inliers} cặp điểm khớp' if inliers else ('theo màu' if known else 'không đủ')}",
                 f"Thời gian   : {seconds:.2f} s"]
        if rank:
            lines += ["", "Xếp hạng theo màu (khoảng cách):"]
            lines += [f"  {i + 1}. {vnd(c):<16} {d:.3f}" for i, (c, d) in enumerate(rank[:3])]
        if not known:
            lines += ["", "Gợi ý: chụp gần hơn, đủ sáng, tờ tiền", "phẳng; hoặc khoanh vùng đúng một tờ."]
        self.detail.config(text="\n".join(lines))
        self.status.config(text=f"Nhận diện {mode} xong trong {seconds:.2f} s.")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
