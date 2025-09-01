# 1_🔥_Home.py — Wildfire Detection (Image + Video, Tabs + Debug)
# Streamlit Cloud (CPU) 환경 호환 / 단일 페이지
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
    원격 URL이면 임시파일로 받아서 OpenCV가 안정적으로 열도록 변환.
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


# ========================= 추론 (비디오) =========================

def predict_video(model,
                  source,
                  conf_threshold: float,
                  iou_threshold: float,
                  frame_skip: int = 2,
                  resize_w: int | None = 960,
                  max_frames: int = 1200,
                  stop_key: str = "stop_video"):
    path = _resolve_video_source(source)
    # 미리보기(디코더 테스트)
    st.video(path)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error("Failed to open video with OpenCV.")
        return

    canvas = st.empty()
    info = st.empty()
    processed = 0
    t0 = time.time()

    if stop_key not in st.session_state:
        st.session_state[stop_key] = False

    cols = st.columns([1, 2, 2])
    with cols[0]:
        if st.button("⏹ Stop"):
            st.session_state[stop_key] = True
    with cols[1]:
        st.caption("Stop을 누르면 다음 루프에서 종료됩니다.")

    while True:
        if st.session_state[stop_key]:
            info.info("Stopped by user.")
            break

        ret, frame = cap.read()
        if not ret:
            break
        processed += 1

        if frame_skip > 1 and (processed % frame_skip) != 0:
            continue

        if resize_w and frame.shape[1] > resize_w:
            h, w = frame.shape[:2]
            new_h = int(h * (resize_w / w))
            frame = cv2.resize(frame, (resize_w, new_h), interpolation=cv2.INTER_AREA)

        res = model.predict(frame, conf=conf_threshold, iou=iou_threshold, device="cpu", verbose=False)
        vis = res[0].plot() if res else frame
        canvas.image(_rgb(vis), caption=f"Frame {processed}", use_column_width=True)

        elapsed = time.time() - t0
        fps = (processed / elapsed) if elapsed > 0 else 0.0
        info.info(f"Processed: {processed} frames  |  ~{fps:.1f} FPS (incl. skip, resized)")

        if processed >= max_frames:
            st.warning("Max frames reached; stopping.")
            break

    cap.release()
    st.session_state[stop_key] = False


# ========================= 메인 =========================

def main():
    st.set_page_config(page_title="Wildfire Detection", page_icon="🔥", initial_sidebar_state="collapsed")

    # 디버그 배너: 빌드가 새 코드인지 확인용
    st.info(f"VIDEO BUILD ACTIVE — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # 사이드바
    st.sidebar.markdown("Developed by Alim Tleuliyev")
    st.sidebar.markdown("LinkedIn: [Profile](https://www.linkedin.com/in/alimtleuliyev/)")
    st.sidebar.markdown("GitHub: [Repo](https://github.com/AlimTleuliyev/wildfire-detection)")
    st.sidebar.markdown("Email: [alim.tleuliyev@nu.edu.kz](mailto:alim.tleuliyev@nu.edu.kz)")
    st.sidebar.markdown("Telegram: [@nativealim](https://t.me/nativealim)")

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
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)
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
            resize_w = st.slider("Resize width", 320, 1280, 960, 40, help="가로 리사이즈(성능 향상)")
        with c3:
            max_frames = st.slider("Max frames", 100, 4000, 1200, 100, help="안전 종료 상한")

        if video_path:
            if st.button("▶ Start video inference"):
                with st.spinner("Running video inference…"):
                    predict_video(model,
                                  source=video_path,
                                  conf_threshold=conf_threshold,
                                  iou_threshold=iou_threshold,
                                  frame_skip=frame_skip,
                                  resize_w=resize_w,
                                  max_frames=max_frames)


if __name__ == "__main__":
    main()

