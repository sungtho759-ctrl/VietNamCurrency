"use strict";

// Màu gần đúng của từng tờ tiền: dùng cho khung và nhãn
const NOTE_COLOR = {
  "500": "#c0396b", "1000": "#7a6a2c", "2000": "#8a4b2f", "5000": "#2b56a6", "10000": "#a8740c",
  "20000": "#1f6db3", "50000": "#b83a73", "100000": "#3f7f25", "200000": "#b0472f", "500000": "#13849a",
};
const METHODS = {
  sift: { url: "/api/sift", el: document.getElementById("m-sift"), cmp: document.getElementById("cmp-sift"), name: "Truyền thống" },
  yolo: { url: "/api/yolo", el: document.getElementById("m-yolo"), cmp: document.getElementById("cmp-yolo"), name: "YOLO11n" },
};
const $ = (id) => document.getElementById(id);
const vnd = (v) => Number(v).toLocaleString("vi-VN") + " đ";
const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms} ms`);
// Mục "Kết hợp YOLO + SIFT": tách khỏi hai thẻ so sánh, dùng API riêng
const HYBRID_EL = document.getElementById("m-hybrid");
const elOf = (key) => (key === "hybrid" ? HYBRID_EL : METHODS[key].el);

const state = {
  view: "compare",                               // mục đang mở: "compare" (SIFT + màu và YOLO) hoặc "hybrid"
  last: { sift: null, yolo: null, hybrid: null },  // kết quả gần nhất của mỗi phương pháp
  busy: { sift: false, yolo: false, hybrid: false },
  stream: null,
  live: false,
  loops: new Set(),                              // các vòng nhận diện liên tục đang chạy
  sourceLabel: "",
  input: null,                                   // ảnh gần nhất {blob, bitmap, label, id}: chạy lại khi đổi mục
  done: { compare: 0, hybrid: 0 },               // id ảnh mà mỗi mục đã nhận diện
  seq: 0,
};
// Các phương pháp thuộc mục đang mở
const activeKeys = () => (state.view === "hybrid" ? ["hybrid"] : Object.keys(METHODS));
const anyBusy = () => activeKeys().some((k) => state.busy[k]);

/* ------------------------------------------------------------------ Thông tin máy chủ */
async function loadInfo() {
  try {
    const info = await (await fetch("/api/info")).json();
    $("pill-sift").classList.add("ok");
    $("pill-sift").title = `${info.sift_templates} ảnh mẫu SIFT` + (info.color_model ? " + mô hình màu" : "");
    $("pill-yolo").classList.add(info.yolo ? "ok" : "bad");
    $("pill-yolo").title = info.yolo ? `Ngưỡng tin cậy ${info.yolo_conf} · ${info.yolo_classes.length} lớp` : info.yolo_error;
    if (info.samples.length) {
      $("samples").hidden = false;
      for (const name of info.samples) {
        const b = document.createElement("button");
        b.className = "sample";
        b.title = name;
        b.innerHTML = `<img src="/samples/${encodeURIComponent(name)}" alt="Ảnh mẫu ${name}">`;
        b.onclick = async () => {
          const blob = await (await fetch(`/samples/${encodeURIComponent(name)}`)).blob();
          useFile(new File([blob], name, { type: blob.type }));
        };
        $("sample-list").appendChild(b);
      }
    }
  } catch {
    for (const id of ["pill-sift", "pill-yolo"]) $(id).classList.add("bad");
  }
}

/* ------------------------------------------------------------------ Tabs */
function selectTab(which) {
  for (const t of ["camera", "upload"]) {
    $(`tab-${t}`).setAttribute("aria-selected", String(t === which));
    $(`pane-${t}`).hidden = t !== which;
  }
  if (which !== "camera") setLive(false);
}
$("tab-camera").onclick = () => selectTab("camera");
$("tab-upload").onclick = () => selectTab("upload");
document.querySelector(".tabs").addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const next = $("tab-camera").getAttribute("aria-selected") === "true" ? "upload" : "camera";
    selectTab(next);
    $(`tab-${next}`).focus();
  }
});

/* ------------------------------------------------------------------ Mục: so sánh / kết hợp */
function selectView(view) {
  state.view = view;
  for (const v of ["compare", "hybrid"]) {
    $(`view-tab-${v}`).setAttribute("aria-selected", String(v === view));
    $(`view-${v}`).hidden = v !== view;
  }
  if (state.live) setLive(true);  // chuyển các vòng nhận diện liên tục sang mục mới
  else if (state.input && state.done[view] !== state.input.id) rerun();
}
function rerun() {
  const i = state.input;
  if (i && !anyBusy()) recognizeBoth(i.blob, i.bitmap, i.label);
}
$("view-tab-compare").onclick = () => selectView("compare");
$("view-tab-hybrid").onclick = () => selectView("hybrid");
document.querySelector(".view-tabs").addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const next = state.view === "compare" ? "hybrid" : "compare";
    selectView(next);
    $(`view-tab-${next}`).focus();
  }
});
$("chk-yolo-label").onchange = () => { if (!state.live) rerun(); };

/* ------------------------------------------------------------------ Nhận diện */
// Gửi ảnh tới các phương pháp của mục đang mở. Mục so sánh: cùng một ảnh tới cả hai phương pháp, mỗi kết quả hiện
// ngay khi về (YOLO thường về trước).
function recognizeBoth(blob, bitmap, label) {
  state.sourceLabel = label;
  $("result-source").textContent = label;
  if (state.input?.blob !== blob) state.input = { blob, bitmap, label, id: ++state.seq };
  state.done[state.view] = state.input.id;
  return Promise.all(activeKeys().map((k) => recognize(k, blob, bitmap)));
}

async function recognize(key, blob, bitmap) {
  if (key === "hybrid") return recognizeHybrid(blob, bitmap);
  const m = METHODS[key];
  state.busy[key] = true;
  m.el.dataset.state = "loading";
  if (m.el.querySelector("canvas").width === 0 || !state.live) drawImageOnly(m.el, bitmap);
  const form = new FormData();
  form.append("image", blob, "frame.jpg");
  try {
    const res = await fetch(m.url, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `Lỗi máy chủ (${res.status})`);
    state.last[key] = data;
    render(key, data, bitmap);
  } catch (err) {
    state.last[key] = null;
    renderError(key, err.message, bitmap);
  } finally {
    state.busy[key] = false;
    if (m.el.dataset.state === "loading") m.el.dataset.state = "done";
    updateCompare();
  }
}

function drawImageOnly(el, bitmap) {
  const c = el.querySelector("canvas");
  c.width = bitmap.width;
  c.height = bitmap.height;
  c.getContext("2d").drawImage(bitmap, 0, 0);
}

// Vẽ ảnh và các khung [{quad, text, color, dashed}] (tọa độ theo ảnh máy chủ nhận: data.width x data.height)
function drawBoxes(el, bitmap, data, items) {
  const c = el.querySelector("canvas");
  c.width = bitmap.width;
  c.height = bitmap.height;
  const g = c.getContext("2d");
  g.drawImage(bitmap, 0, 0);
  const sx = bitmap.width / data.width, sy = bitmap.height / data.height;
  const lw = Math.max(2, Math.round(Math.max(c.width, c.height) / 260));
  const fs = Math.max(13, Math.round(Math.max(c.width, c.height) / 34));
  items.forEach(({ quad, text, color, dashed }) => {
    const pts = quad.map(([x, y]) => [x * sx, y * sy]);
    g.lineWidth = lw;
    g.strokeStyle = color;
    g.fillStyle = color + "22";
    g.setLineDash(dashed ? [lw * 3, lw * 2] : []);
    g.beginPath();
    pts.forEach(([x, y], j) => (j ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.closePath();
    g.fill();
    g.stroke();
    g.setLineDash([]);
    // nhãn: góc trên trái của khung, giữ trong ảnh
    g.font = `600 ${fs}px "Be Vietnam Pro", "Segoe UI", sans-serif`;
    const tw = g.measureText(text).width + fs * 0.8, th = fs * 1.5;
    let x = Math.min(...pts.map((p) => p[0])), y = Math.min(...pts.map((p) => p[1])) - th;
    x = Math.max(0, Math.min(x, c.width - tw));
    y = Math.max(0, y);
    g.fillStyle = color;
    g.beginPath();
    g.roundRect ? g.roundRect(x, y, tw, th, fs * 0.3) : g.rect(x, y, tw, th);
    g.fill();
    g.fillStyle = "#fff";
    g.textBaseline = "middle";
    g.fillText(text, x + fs * 0.4, y + th / 2);
  });
}

function render(key, data, bitmap) {
  const el = METHODS[key].el;
  el.dataset.state = "done";
  drawBoxes(el, bitmap, data, data.notes.map((n, i) => (
    { quad: n.quad, text: `${i + 1}. ${vnd(n.label)}`, color: NOTE_COLOR[n.label] || "#0f766e" })));

  setMetric(el, "count", `${data.notes.length} tờ`);
  setMetric(el, "total", vnd(data.total));
  setMetric(el, "ms", fmtMs(data.ms));
  const ol = el.querySelector(".notes");
  ol.innerHTML = "";
  if (!data.notes.length) {
    ol.innerHTML = `<li class="empty">Không tìm thấy tờ tiền nào</li>`;
    return;
  }
  data.notes.forEach((n, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="idx" style="background:${NOTE_COLOR[n.label] || "#0f766e"}">${i + 1}</span>`
      + `<span class="val">${vnd(n.label)}</span><span class="ev">${n.evidence}</span>`;
    ol.appendChild(li);
  });
}

/* ------------------------------------------------------------------ Kết hợp YOLO + SIFT */
const SOURCE = {
  sift: ["SIFT + màu", "sift"], sift_full: ["SIFT toàn ảnh", "sift_full"],
  yolo: ["theo YOLO", "yolo"], unknown: ["chưa rõ", "unknown"],
};
const UNSURE_COLOR = "#8a918e";

async function recognizeHybrid(blob, bitmap) {
  state.busy.hybrid = true;
  HYBRID_EL.dataset.state = "loading";
  if (HYBRID_EL.querySelector("canvas").width === 0 || !state.live) drawImageOnly(HYBRID_EL, bitmap);
  const form = new FormData();
  form.append("image", blob, "frame.jpg");
  try {
    const url = `/api/hybrid?yolo_label=${$("chk-yolo-label").checked ? 1 : 0}`;
    const res = await fetch(url, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `Lỗi máy chủ (${res.status})`);
    state.last.hybrid = data;
    renderHybrid(data, bitmap);
  } catch (err) {
    state.last.hybrid = null;
    renderError("hybrid", err.message, bitmap);
  } finally {
    state.busy.hybrid = false;
    if (HYBRID_EL.dataset.state === "loading") HYBRID_EL.dataset.state = "done";
  }
}

function renderHybrid(data, bitmap) {
  const el = HYBRID_EL;
  el.dataset.state = "done";
  // Khung nét đứt: mệnh giá không do SIFT + màu xác nhận (theo YOLO hoặc "?")
  drawBoxes(el, bitmap, data, data.notes.map((n, i) => ({
    quad: n.quad, text: `${i + 1}. ${n.label ? vnd(n.label) : "?"}`,
    color: n.label ? NOTE_COLOR[n.label] || "#0f766e" : UNSURE_COLOR,
    dashed: !n.source.startsWith("sift"),
  })));

  const known = data.notes.filter((n) => n.label);
  const unsure = data.notes.length - known.length;
  setMetric(el, "count", `${known.length} tờ` + (unsure ? ` + ${unsure} ?` : ""));
  setMetric(el, "total", vnd(data.total));
  setMetric(el, "ms", fmtMs(data.ms));
  el.querySelector('[data-k="ms"]').title = `YOLO ${data.ms_yolo} ms · SIFT + màu ${data.ms_sift} ms`;

  const count = (src) => data.notes.filter((n) => n.source === src).length;
  const disagree = data.notes.filter((n) => n.agree === false).length;
  let sum = "";
  if (data.fallback) {
    sum = `YOLO không thấy tờ nào (kể cả khi xoay ảnh 180°), đã dùng SIFT + màu trên toàn ảnh (${fmtMs(data.ms_sift)}).`;
  } else if (data.notes.length) {
    sum = `YOLO tìm ${data.notes.length} tờ (${fmtMs(data.ms_yolo)})${data.flipped ? " trên ảnh xoay 180°" : ""}`
      + ` · SIFT + màu xác nhận ${count("sift")} tờ (${fmtMs(data.ms_sift)})`
      + (count("yolo") ? ` · ${count("yolo")} tờ theo nhãn YOLO` : "")
      + (count("unknown") ? ` · ${count("unknown")} tờ chưa rõ` : "")
      + (disagree ? ` · <span class="warn">${disagree} tờ YOLO nhận khác SIFT + màu (dùng kết quả SIFT + màu)</span>` : "");
  }
  $("hybrid-summary").innerHTML = sum;

  const ol = el.querySelector(".notes");
  ol.innerHTML = "";
  if (!data.notes.length) {
    ol.innerHTML = `<li class="empty">Không tìm thấy tờ tiền nào</li>`;
    return;
  }
  data.notes.forEach((n, i) => {
    const [badge, cls] = SOURCE[n.source];
    const yolo = n.yolo_label ? `YOLO: ${vnd(n.yolo_label)} (${n.yolo_conf.toFixed(2)})` : "";
    let ev;
    if (n.source === "sift") {
      ev = `${n.inliers} cặp điểm khớp${n.clahe ? ", CLAHE" : ""} · `
        + (n.agree ? yolo : `<span class="warn">${yolo}</span>`);
    } else if (n.source === "sift_full") {
      ev = n.evidence;
    } else {
      // Theo YOLO: SIFT thấy hoa văn tờ tiền nhưng chưa chắc mệnh giá. "?": nhãn YOLO chỉ để tham khảo, không cộng
      ev = n.source === "yolo" ? `${n.evidence} · ${yolo}` : `SIFT + màu chưa kết luận · ${yolo.replace("YOLO:", "YOLO đoán")}`;
    }
    const color = n.label ? NOTE_COLOR[n.label] || "#0f766e" : UNSURE_COLOR;
    const li = document.createElement("li");
    if (n.agree === false) li.className = "disagree";
    li.innerHTML = `<span class="idx" style="background:${color}">${i + 1}</span>`
      + `<span class="val">${n.label ? vnd(n.label) : "?"}</span><span class="badge ${cls}">${badge}</span>`
      + `<span class="ev">${ev}</span>`;
    ol.appendChild(li);
  });
}

function renderError(key, msg, bitmap) {
  const el = elOf(key);
  el.dataset.state = "error";
  if (bitmap) drawImageOnly(el, bitmap);
  for (const k of ["count", "total", "ms"]) setMetric(el, k, "—");
  if (key === "hybrid") $("hybrid-summary").innerHTML = "";
  el.querySelector(".notes").innerHTML = `<li class="error">${msg}</li>`;
}

function setMetric(el, k, v) {
  el.querySelector(`[data-k="${k}"]`).textContent = v;
}

function summary(d) {
  if (!d) return "<b>—</b>";
  return `<b>${vnd(d.total)}</b><small>${d.notes.length} tờ</small>`;
}

function updateCompare() {
  const a = state.last.sift, b = state.last.yolo;
  METHODS.sift.cmp.innerHTML = summary(a);
  METHODS.yolo.cmp.innerHTML = summary(b);
  const v = $("cmp-verdict");
  v.className = "stat verdict";
  if (!a || !b) {
    v.innerHTML = "<span>So sánh</span><b>—</b>";
    return;
  }
  const key = (d) => d.notes.map((n) => n.label).sort().join(",");
  const same = key(a) === key(b);
  const faster = a.ms > b.ms ? ["YOLO11n", a.ms / Math.max(b.ms, 1)] : ["Truyền thống", b.ms / Math.max(a.ms, 1)];
  v.classList.add(same ? "same" : "diff");
  v.innerHTML = `<span>So sánh</span><b>${same ? "Kết quả trùng khớp" : "Kết quả khác nhau"}</b>`
    + `<small>${faster[0]} nhanh hơn ×${faster[1].toFixed(1)}</small>`;
}

/* ------------------------------------------------------------------ Tải ảnh lên */
const dz = $("dropzone");
$("file").addEventListener("change", (e) => e.target.files[0] && useFile(e.target.files[0]));
$("btn-pick").onclick = () => $("file").click();
dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
for (const ev of ["dragenter", "dragover"]) dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); });
for (const ev of ["dragleave", "drop"]) dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); });
dz.addEventListener("drop", (e) => {
  const f = [...e.dataTransfer.files].find((x) => x.type.startsWith("image/"));
  if (f) useFile(f);
});
window.addEventListener("paste", (e) => {
  const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/"));
  if (f) { selectTab("upload"); useFile(f); }
});

let currentFile = null;
async function useFile(file) {
  if (!file.type.startsWith("image/")) return;
  currentFile = file;
  $("preview").src = URL.createObjectURL(file);
  $("preview").hidden = false;
  $("dz-empty").hidden = true;
  $("file-name").textContent = file.name;
  $("btn-run").disabled = false;
  await runFile();
}
$("btn-run").onclick = runFile;

async function runFile() {
  if (!currentFile || anyBusy()) return;
  let bitmap;
  try {
    bitmap = await createImageBitmap(currentFile, { imageOrientation: "from-image" });
  } catch {
    for (const k of Object.keys(METHODS)) renderError(k, "Trình duyệt không đọc được ảnh này", null);
    return;
  }
  // Gửi đúng ảnh đã xoay theo EXIF để máy chủ và trình duyệt cùng hệ tọa độ
  const blob = await toJpeg(bitmap, 0.92);
  $("btn-run").disabled = true;
  await recognizeBoth(blob, bitmap, `Ảnh tải lên: ${currentFile.name}`);
  $("btn-run").disabled = false;
}

// Thu ảnh về cạnh dài tối đa MAX_SEND_SIDE trước khi gửi: máy chủ chỉ dùng tới 800 px (SIFT) / 640 px (YOLO), và
// Vercel từ chối thân yêu cầu lớn hơn 4,5 MB. Khung trả về được vẽ lại theo tỷ lệ data.width / bitmap.width.
const MAX_SEND_SIDE = 1600;
function toJpeg(source, quality = 0.85) {
  const k = Math.min(1, MAX_SEND_SIDE / Math.max(source.width, source.height));
  const c = document.createElement("canvas");
  c.width = Math.round(source.width * k);
  c.height = Math.round(source.height * k);
  c.getContext("2d").drawImage(source, 0, 0, c.width, c.height);
  return new Promise((r) => c.toBlob(r, "image/jpeg", quality));
}

/* ------------------------------------------------------------------ Camera */
const video = $("video");

async function startCamera(deviceId) {
  stopCamera();
  if (!navigator.mediaDevices?.getUserMedia) {
    $("cam-hint").textContent = "Trình duyệt không hỗ trợ camera (cần mở trang qua http://localhost hoặc https).";
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
      audio: false,
    });
  } catch (err) {
    $("cam-hint").textContent = "Không mở được camera: " + (err.name === "NotAllowedError" ? "chưa được cấp quyền." : err.message);
    return;
  }
  video.srcObject = state.stream;
  await video.play();
  $("cam-stage").classList.add("on");
  $("btn-snap").disabled = false;
  $("btn-cam-stop").disabled = false;
  $("cam-hint").textContent = "Đặt tờ tiền vào khung rồi bấm \"Chụp & nhận diện\", hoặc bật nhận diện liên tục.";
  await listCameras();
}

function stopCamera() {
  setLive(false);
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;
  $("cam-stage").classList.remove("on");
  $("btn-snap").disabled = true;
  $("btn-cam-stop").disabled = true;
}

async function listCameras() {
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const sel = $("sel-camera");
  sel.hidden = cams.length < 2;
  const active = state.stream?.getVideoTracks()[0]?.getSettings().deviceId;
  sel.innerHTML = cams.map((c, i) => `<option value="${c.deviceId}" ${c.deviceId === active ? "selected" : ""}>${c.label || "Camera " + (i + 1)}</option>`).join("");
}
$("sel-camera").onchange = (e) => startCamera(e.target.value);
$("btn-cam-start").onclick = () => startCamera();
$("btn-cam-stop").onclick = stopCamera;

// Khung hình hiện tại (cả khung, hoặc chỉ vùng trong khung căn chỉnh)
async function grabFrame() {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return null;
  let sx = 0, sy = 0, sw = w, sh = h;
  if ($("chk-roi").checked) {
    // Khung căn chỉnh: rộng 72% vùng hiển thị, tỷ lệ 2,25 : 1, ở giữa (xem .guide trong style.css), nới thêm 8%
    const stage = $("cam-stage").getBoundingClientRect();
    const scale = Math.min(stage.width / w, stage.height / h);
    const gw = (stage.width * 0.72) / scale * 1.08;
    const gh = (stage.width * 0.72 / 2.25) / scale * 1.08;
    sw = Math.min(w, gw); sh = Math.min(h, gh);
    sx = (w - sw) / 2; sy = (h - sh) / 2;
  }
  const c = document.createElement("canvas");
  c.width = Math.round(sw);
  c.height = Math.round(sh);
  c.getContext("2d").drawImage(video, sx, sy, sw, sh, 0, 0, c.width, c.height);
  const bitmap = await createImageBitmap(c);
  const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.9));
  return { blob, bitmap };
}

$("btn-snap").onclick = async () => {
  if (anyBusy()) return;
  const f = await grabFrame();
  if (f) recognizeBoth(f.blob, f.bitmap, "Ảnh chụp từ camera");
};

// Nhận diện liên tục: mỗi phương pháp của mục đang mở lấy khung hình mới nhất ngay khi xong lần trước, nên YOLO
// (nhanh) cập nhật nhiều lần trong lúc SIFT (chậm) xử lý một khung – thấy rõ chênh lệch tốc độ.
function setLive(on) {
  state.live = on && !!state.stream;
  $("chk-live").checked = state.live;
  $("live-badge").hidden = !state.live;
  $("btn-snap").disabled = !state.stream || state.live;
  if (state.live) {
    $("result-source").textContent = "Camera · nhận diện liên tục";
    for (const k of activeKeys()) liveLoop(k);
  }
}
$("chk-live").onchange = (e) => setLive(e.target.checked);

async function liveLoop(key) {
  if (state.loops.has(key)) return;  // vòng của phương pháp này vẫn đang chạy (đổi mục qua lại nhanh)
  state.loops.add(key);
  try {
    // Dừng khi tắt nhận diện liên tục hoặc chuyển sang mục không có phương pháp này
    while (state.live && activeKeys().includes(key)) {
      if (state.busy[key]) { await sleep(50); continue; }
      const f = await grabFrame();
      if (!f) { await sleep(100); continue; }
      await recognize(key, f.blob, f.bitmap);
      await sleep(key === "yolo" ? 60 : 150);  // nhường máy chủ / trình duyệt
    }
  } finally {
    state.loops.delete(key);
  }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

document.addEventListener("visibilitychange", () => { if (document.hidden) setLive(false); });

loadInfo();
