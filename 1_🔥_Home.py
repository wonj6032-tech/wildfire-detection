# 1_🔥_Home.py — Wildfire Detection (Image + Video, Live HUD, Streaming)
# Streamlit Cloud (CPU) 최적화: 스트리밍 추론 + 프레임 드롭 + 해상도 축소 + 스레드 튜닝
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import io
import time
import tempfile
from glob import glob
from datetime import datetime

import cv2
import requests
import streamlit as st
import numpy as np
from numpy import random
from PIL import Image
from ultralytics import YOLO

# --- CPU thread tuning (Streamlit Cloud) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
try:
    import torch
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
except Exception:
    pass


# ========================= 유틸 =========================

def _rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def _resize_keep_aspect(frame, target_w=None, max_w=None):
    h, w = frame.shape[:2]
    if target_w is not None and target_w > 0 and w != target_w:
        new_w = int(target_w)
        new_h = int(h * (new_w / w))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if max_w is not None and w > max_w:
        new_w = int(max_w)
        new_h = int(h * (new_w / w))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame

def _safe_names(model, res0=None):
    if res0 is not None and hasattr(res0, "names") and res0.names:
        return res0.names
    if hasattr(model, "names") and model.names:
        return model.names
    if hasattr(model, "model") and hasattr(model.model, "names"):
        return model.model.names
    return {0: "object"}

def _looks_like_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(256)
        return b"git-lfs" in head or b"oid sha256" in head or b"version https://git-lfs.github.com/spec" in head
    except Exception:
        return False

def _resolve_video_source(src: str) -> str:
    """
    원격 URL이면 임시파일로 받아서 OpenCV/Ultralytics가 안정적으로 열도록 변환.
    """
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        suffix = os.path.splitext(src)[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with requests.get(src, stream=True, timeout=60) as r:
            r.raise_for_status()
            size = 0
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    tmp.write(chunk)
                    size += len(chunk)
        tmp.flush()
        st.caption(f"Downloaded video → {os.path.basename(tmp.name)} ({round(size/1_048_576,2)} MB)")
        return tmp.name
    return src


# ========================= 모델 로더 =========================

@st.cache_resource
def load_model(model_path: str | None):
    """
    1) 지정 경로 .pt 로드
    2) 실패 시 저장소의 general-models/yolov8n.pt 폴백
    3) 최종 폴백: 패키지 yolov8n.pt
    """
    if model_path and os.path.exists(model_path) and os.path.getsize(model_path) > 100_000 and not _looks_like_lfs_pointer(model_path):
        try:
            return YOLO(model_path)
        except Exception as e:
            st.warning(f"가중치 로드 실패({e}). 폴백 시도…")

    fallback_local = os.path.join("general-models", "yolov8n.pt")
    if os.path.exists(fallback_local) and not _looks_like_lfs_pointer(fallback_local):
        try:
            return YOLO(fallback_local)
        except Exception as e:
            st.warning(f"로컬 폴백 로드 실패({e}). 최종 폴백 시도…")

    return YOLO("yolov8n.pt")


# ========================= 추론 (이미지) =========================

def predict_image(model, image_pil: Image.Image, conf_threshold: float, iou_threshold: float):
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    res = model.predict(img_bgr, conf=conf_threshold, iou=iou_threshold, device='cpu', verbose=False)
    res0 = res[0] if res else None
    names = _safe_names(model, res0)
    classes = res0.boxes.cls if (res0 is not None and res0.boxes is not None) else []
    class_counts = {}
    if classes is not None:
        for c in classes:
            c = int(c)
            cname = names[c] if isinstance(names, (list, tuple)) else names.get(c, str(c))
            class_counts[cname] = class_counts.get(cname, 0) + 1

    if class_counts:
        parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)]
        text = "Predicted " + ", ".join(parts)
    else:
        text = "No objects detected"

    latency_ms = sum(res0.speed.values()) if (res0 is not None and hasattr(res0, "speed")) else 0.0
    text += f" in {round(latency_ms / 1000, 2)} seconds."
    vis = res0.plot() if res0 is not None else img_bgr
    return _rgb(vis), text


# ========================= 추론 (비디오 — 스트리밍 + 라이브 HUD) =========================

def predict_video(model,
                  source,
                  conf_threshold: float,
                  iou_threshold: float,
                  frame_skip: int = 2,
                  resize_w: int | None = 960,
                  max_frames: int = 1800,
                  stop_key: str = "stop_video",
                  hud: bool = True,
                  imgsz: int = 480,        # YOLO 입력 사이즈(작을수록 빠름) 384~512 권장
                  target_fps: int = 12,    # 화면 갱신 목표 FPS (스로틀) 10~15 권장
                  preview: bool = False):
    """
    Ultralytics 스트리밍 추론 + FPS 스로틀 + 프레임 드롭.
    """
    import collections
    from time import perf_counter, sleep

    def _put_text(img, text, y, color=(0, 255, 0)):
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    path = _resolve_video_source(source)
    if preview:
        st.video(path)  # 필요 시만 디코더 체크용

    # Stop 버튼
    if stop_key not in st.session_state:
        st.session_state[stop_key] = False
    cols = st.columns([1, 4, 1])
    with cols[0]:
        if st.button("⏹ Stop"):
            st.session_state[stop_key] = True

    # YOLO 스트리밍 제너레이터 (내부에서 프레임 읽기)
    gen = model.predict(
        source=path,
        stream=True,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,            # 입력 다운스케일링 (속도↑)
        device="cpu",
        verbose=False
    )

    canvas = st.empty()
    fps_hist = collections.deque(maxlen=15)
    t_prev = perf_counter()
    processed = 0

    for res in gen:
        if st.session_state[stop_key]:
            break
        processed += 1
        if processed > max_frames:
            break

        # 프레임 스킵(추론 자체 생략)
        if frame_skip > 1 and (processed % frame_skip) != 0:
            continue

        vis = res.plot()  # 박스/라벨을 영상 안에 직접 그림 (BGR)

        # 표시 전용 리사이즈(성능)
        if resize_w and vis.shape[1] > resize_w:
            h, w = vis.shape[:2]
            new_h = int(h * (resize_w / w))
            vis = cv2.resize(vis, (resize_w, new_h), interpolation=cv2.INTER_AREA)

        # FPS 계산(스무딩)
        t_now = perf_counter()
        dt = max(t_now - t_prev, 1e-6)
        fps_hist.append(1.0 / dt)
        t_prev = t_now
        fps_smoothed = sum(fps_hist) / len(fps_hist)

        # 클래스 카운트 → HUD
        if hud:
            counts_txt = ""
            try:
                names = _safe_names(model, res)
                cls = res.boxes.cls if (res and res.boxes is not None) else []
                cc = {}
                for c in cls:
                    c = int(c)
                    name = names[c] if isinstance(names, (list, tuple)) else names.get(c, str(c))
                    cc[name] = cc.get(name, 0) + 1
                if cc:
                    parts = [f"{k}:{v}" for k, v in sorted(cc.items(), key=lambda x: x[1], reverse=True)]
                    counts_txt = " | ".join(parts)
            except Exception:
                pass
            _put_text(vis, f"FPS: {fps_smoothed:.1f}", 28, (0, 255, 0))
            if counts_txt:
                _put_text(vis, counts_txt, 58, (255, 200, 0))

        # 실시간 갱신 (자막 없이 영상만)
        canvas.image(_rgb(vis), use_column_width=True)

        # 목표 FPS로 스로틀 (표시 주기 제어)
        if target_fps > 0:
            budget = max(0.0, (1.0 / target_fps) - (perf_counter() - t_now))
            if budget > 0:
                time.sleep(budget)

    st.session_state[stop_key] = False  # 리셋


# ========================= 메인 =========================

def main():
    st.set_page_config(page_title="Wildfire Detection", page_icon="🔥", initial_sidebar_state="collapsed")

    # 디버그 배너: 빌드 확인용
    st.info(f"VIDEO BUILD ACTIVE — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # ---- Sidebar: Credits Card (optional fancy style) ----
    st.sidebar.markdown(
    """
    <style>
      .credit-card{
        padding:12px 14px; border:1px solid #e5e7eb; border-radius:12px;
        background:#fafafa; font-size:14px; line-height:1.45;
      }
      .credit-card b{font-size:15px;}
      .credit-item{ margin:6px 0; }
      .credit-card a{ text-decoration:none; }
    </style>
    <div class="credit-card">
      <div class="credit-item"><b>👤 Original author</b><br/>
        <a href="https://www.linkedin.com/in/alimtleuliyev/" target="_blank">Alim Tleuliyev</a>
      </div>
      <div class="credit-item"><b>🛠 Modified by</b><br/>
        <b>Wonjin Choi</b> (WOW Future Technology)
      </div>
      <hr/>
      <div class="credit-item">🐙 <b>GitHub</b><br/>
        <a href="https://github.com/AlimTleuliyev/wildfire-detection" target="_blank">Original repo</a><br/>
        <a href="https://github.com/wonj6032-tech/wildfire-detection" target="_blank">This fork</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
  

    # 타이틀/로고
    st.markdown("<h1 style='text-align:center;'>Wildfire Detection</h1>", unsafe_allow_html=True)
    logos = glob('dalle-logos/*.png')
    if logos:
        logo = random.choice(logos)
        st.image(logo, use_column_width=True, caption="Generated by DALL-E")
        st.sidebar.image(logo, use_column_width=True, caption="Generated by DALL-E")

    st.markdown("---")

    # 모델 선택
    colA, colB = st.columns(2)
    with colA:
        model_type = st.radio("Select Model Type", ("Fire Detection", "General"), index=0)
    models_dir = "general-models" if model_type == "General" else "fire-models"
    model_files = sorted([f[:-3] for f in os.listdir(models_dir)]) if os.path.isdir(models_dir) else []
    with colB:
        if model_files:
            default_idx = 0
            try:
                if model_type != "General":
                    if "fire_n" in model_files: default_idx = model_files.index("fire_n")
                    elif "fire_s" in model_files: default_idx = model_files.index("fire_s")
            except Exception:
                pass
            selected_model = st.selectbox("Select Model", model_files, index=min(default_idx, len(model_files)-1))
        else:
            st.warning(f"'{models_dir}' 폴더에서 .pt 파일을 찾지 못했습니다. 폴백 모델을 사용합니다.")
            selected_model = None

    model_path = os.path.join(models_dir, f"{selected_model}.pt") if selected_model else None
    with st.spinner("Loading model…"):
        model = load_model(model_path)

    st.markdown("---")

    # 공통 파라미터
    colT1, colT2 = st.columns(2)
    with colT2:
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    with colT1:
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.50, 0.05)

    st.markdown("---")

    # 탭 UI: Image | Video
    tab_img, tab_vid = st.tabs(["🖼 Image", "🎥 Video"])

    with tab_img:
        image = None
        src = st.radio("Image source", ("Enter URL", "Upload from Computer"), index=0)
        if src == "Upload from Computer":
            file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
            if file:
                image = Image.open(file).convert("RGB")
        else:
            url = st.text_input("Enter the image URL:", "")
            if url:
                try:
                    r = requests.get(url, timeout=20); r.raise_for_status()
                    image = Image.open(io.BytesIO(r.content)).convert("RGB")
                except Exception as e:
                    st.error(f"Error loading image: {e}")

        if image:
            with st.spinner("Detecting…"):
                pred_img, text = predict_image(model, image, conf_threshold, iou_threshold)
                st.image(pred_img, caption="Prediction", use_column_width=True)
                st.success(text)
            out = Image.fromarray(pred_img)
            buf = io.BytesIO(); out.save(buf, format="PNG")
            st.download_button("Download Prediction", data=buf.getvalue(), file_name="prediction.png", mime="image/png")

    with tab_vid:
        video_path = None
        v_src = st.radio("Video source", ("Enter URL", "Upload from Computer"), index=0)
        if v_src == "Upload from Computer":
            vfile = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
            if vfile:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(vfile.read()); tmp.flush()
                video_path = tmp.name
        else:
            vurl = st.text_input("Enter the video URL (mp4/mov):", "")
            if vurl:
                video_path = vurl

        c1, c2, c3 = st.columns(3)
        with c1:
            frame_skip = st.slider("Frame skip", 1, 8, 2, 1, help="큰 값일수록 덜 많은 프레임 추론 → 더 빠름")
        with c2:
            resize_w = st.slider("Resize width", 320, 1280, 800, 40, help="가로 리사이즈(표시 성능)")
        with c3:
            max_frames = st.slider("Max frames", 100, 6000, 3000, 100, help="안전 종료 상한")

        # 라이브 최적값으로 실행
        if video_path:
            if st.button("▶ Start video inference (Live)"):
                # 혹시 이전 정지 플래그가 남아있다면 초기화
                st.session_state["stop_video"] = False
                with st.spinner("Running video inference…"):
                    predict_video(
                        model,
                        source=video_path,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        frame_skip=frame_skip,   # 2~4 권장
                        resize_w=resize_w,       # 640~960 권장
                        max_frames=max_frames,
                        hud=True,
                        imgsz=480,               # 384~512 권장
                        target_fps=12,           # 10~15 권장
                        preview=False
                    )


if __name__ == "__main__":
    main()
