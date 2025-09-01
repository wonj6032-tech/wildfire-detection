# 1_🔥_Home.py  — Wildfire Detection (Image + Video)
# Streamlit Cloud (CPU) 기준 안정 동작용
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import io
import time
import tempfile
from glob import glob

import cv2
import requests
import streamlit as st
from numpy import random
from PIL import Image
from ultralytics import YOLO


# ========================= 공통 유틸 =========================

def _safe_names(model, res0=None):
    """
    클래스 이름 추출을 최대한 호환성 있게 처리
    """
    # 우선 결과 객체에 names가 있으면 사용
    if res0 is not None and hasattr(res0, "names") and res0.names:
        return res0.names
    # 모델에 names가 dict/list로 있을 수 있음
    if hasattr(model, "names") and model.names:
        return model.names
    if hasattr(model, "model") and hasattr(model.model, "names"):
        return model.model.names
    # 최후의 보루
    return {0: "object"}

def _rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def _resize_keep_aspect(frame, target_w=None, max_w=None):
    """
    frame을 가로 기준으로 리사이즈. target_w가 있으면 그 폭으로,
    max_w가 있으면 그 이하일 때만 축소.
    """
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


# ========================= 모델 로더 =========================

@st.cache_resource
def load_model(model_path: str | None):
    """
    1) 지정 경로의 .pt 로드 시도
    2) 실패 시 general-models/yolov8n.pt 폴백
    3) 그래도 실패 시 내장 yolov8n.pt 폴백
    """
    # 1) 직접 경로 시도
    if model_path and os.path.exists(model_path) and os.path.getsize(model_path) > 1_000_00:  # >100KB
        try:
            return YOLO(model_path)
        except Exception as e:
            st.warning(f"커스텀 가중치 로드 실패: {e}. 폴백을 시도합니다…")

    # 2) 저장소 내 기본 폴백
    fallback_local = os.path.join("general-models", "yolov8n.pt")
    if os.path.exists(fallback_local):
        try:
            return YOLO(fallback_local)
        except Exception as e:
            st.warning(f"로컬 폴백 로드 실패: {e}. 최종 폴백을 시도합니다…")

    # 3) 최종 폴백: 패키지 기본 가중치
    return YOLO("yolov8n.pt")


# ========================= 추론 함수 (이미지) =========================

def predict_image(model, image_pil: Image.Image, conf_threshold: float, iou_threshold: float):
    """
    PIL.Image 입력 → YOLO 추론 → 시각화/문구/지연시간 반환
    """
    # PIL → OpenCV(BGR)
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    res = model.predict(
        img_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',
        verbose=False,
    )

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
        parts = []
        for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
            parts.append(f"{v} {k}{'s' if v > 1 else ''}")
        prediction_text = "Predicted " + ", ".join(parts)
    else:
        prediction_text = "No objects detected"

    latency_ms = sum(res0.speed.values()) if (res0 is not None and hasattr(res0, "speed")) else 0.0
    prediction_text += f" in {round(latency_ms / 1000, 2)} seconds."

    vis = res0.plot() if res0 is not None else img_bgr
    return _rgb(vis), prediction_text


# ========================= 추론 함수 (비디오) =========================

def predict_video(model,
                  source,
                  conf_threshold: float,
                  iou_threshold: float,
                  frame_skip: int = 2,
                  resize_w: int | None = 960,
                  max_frames: int = 1200,
                  stop_flag_key: str = "stop_video"):
    """
    source: 파일 경로 또는 URL
    frame_skip: n이면 1/n 프레임만 처리
    resize_w: 가로 리사이즈 폭(너무 큰 영상 성능 보정)
    max_frames: 안전 종료용 최대 처리 프레임
    stop_flag_key: 세션 키로 Stop 버튼 연동
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Failed to open video.")
        return

    canvas = st.empty()
    info = st.empty()

    processed = 0
    t0 = time.time()

    # Stop 버튼
    if stop_flag_key not in st.session_state:
        st.session_state[stop_flag_key] = False

    stop_col1, stop_col2, stop_col3 = st.columns([1, 1, 2])
    with stop_col1:
        if st.button("⏹ Stop"):
            st.session_state[stop_flag_key] = True
    with stop_col2:
        st.caption("Stop을 누르면 다음 루프에서 종료됩니다.")

    while True:
        if st.session_state[stop_flag_key]:
            info.info("Stopped by user.")
            break

        ret, frame = cap.read()
        if not ret:
            break
        processed += 1

        # 프레임 스킵
        if frame_skip > 1 and (processed % frame_skip) != 0:
            continue

        # 리사이즈
        frame = _resize_keep_aspect(frame, target_w=resize_w if resize_w and resize_w > 0 else None)

        # 추론
        res = model.predict(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            device="cpu",
            verbose=False,
        )
        res0 = res[0] if res else None
        vis = res0.plot() if res0 is not None else frame

        canvas.image(_rgb(vis), caption=f"Frame {processed}", use_column_width=True)

        # 진행 정보
        elapsed = time.time() - t0
        fps = (processed / elapsed) if elapsed > 0 else 0.0
        info.info(f"Processed: {processed} frames  |  ~{fps:.1f} FPS (incl. skip, resized)")
        if processed >= max_frames:
            st.warning("Max frames reached; stopping.")
            break

    cap.release()
    st.session_state[stop_flag_key] = False  # 다음 실행을 위해 리셋


# ========================= 메인 앱 =========================

import numpy as np  # (여기 두는 이유: predict_image에서 np 사용)

def main():
    st.set_page_config(page_title="Wildfire Detection", page_icon="🔥", initial_sidebar_state="collapsed")

    # 사이드바 정보
    st.sidebar.markdown("Developed by Alim Tleuliyev")
    st.sidebar.markdown("LinkedIn: [Profile](https://www.linkedin.com/in/alimtleuliyev/)")
    st.sidebar.markdown("GitHub: [Repo](https://github.com/AlimTleuliyev/wildfire-detection)")
    st.sidebar.markdown("Email: [alim.tleuliyev@nu.edu.kz](mailto:alim.tleuliyev@nu.edu.kz)")
    st.sidebar.markdown("Telegram: [@nativealim](https://t.me/nativealim)")

    # 간단 스타일
    st.markdown(
        """
        <style>
        .container { max-width: 900px; }
        .title { text-align: center; font-size: 35px; font-weight: bold; margin-bottom: 10px; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div class='title'>Wildfire Detection</div>", unsafe_allow_html=True)

    # 로고
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        logos = glob('dalle-logos/*.png')
        if logos:
            logo = random.choice(logos)
            st.image(logo, use_column_width=True, caption="Generated by DALL-E")
            st.sidebar.image(logo, use_column_width=True, caption="Generated by DALL-E")

    # 설명
    st.markdown(
        """
        <div style='text-align: center;'>
            <h2>🔥 <strong>Wildfire Detection App</strong></h2>
            <p>Powered by <a href='https://github.com/ultralytics/ultralytics'>YOLOv8</a> trained on <a href='https://github.com/gaiasd/DFireDataset'>D-Fire</a>.</p>
            <p>Upload an image/video or provide a URL to test.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    # 모델 선택
    colA, colB = st.columns(2)
    with colA:
        model_type = st.radio("Select Model Type", ("Fire Detection", "General"), index=0)

    models_dir = "general-models" if model_type == "General" else "fire-models"
    model_files = []
    if os.path.isdir(models_dir):
        model_files = sorted([f[:-3] for f in os.listdir(models_dir) if f.endswith(".pt")])

    with colB:
        if model_files:
            # fire_n/fire_s 가 있으면 그것부터 선택되게
            default_idx = 0
            try:
                if model_type != "General":
                    if "fire_n" in model_files:
                        default_idx = model_files.index("fire_n")
                    elif "fire_s" in model_files:
                        default_idx = model_files.index("fire_s")
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

    # 모드: 이미지 / 비디오
    media_kind = st.radio("What to test?", ("Image", "Video"), index=0)

    # 공통 파라미터
    colT1, colT2 = st.columns(2)
    with colT2:
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)
    with colT1:
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.50, 0.05)

    st.markdown("---")

    if media_kind == "Image":
        # ===== 이미지 입력 =====
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
                    r = requests.get(url, timeout=20)
                    r.raise_for_status()
                    image = Image.open(io.BytesIO(r.content)).convert("RGB")
                except Exception as e:
                    st.error(f"Error loading image: {e}")

        if image:
            with st.spinner("Detecting…"):
                pred_img, text = predict_image(model, image, conf_threshold, iou_threshold)
                st.image(pred_img, caption="Prediction", use_column_width=True)
                st.success(text)

            # 다운로드 버튼
            out = Image.fromarray(pred_img)
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            st.download_button("Download Prediction", data=buf.getvalue(), file_name="prediction.png", mime="image/png")

    else:
        # ===== 비디오 입력 =====
        video_path = None
        v_src = st.radio("Video source", ("Enter URL", "Upload from Computer"), index=0)

        if v_src == "Upload from Computer":
            vfile = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
            if vfile:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(vfile.read())
                tmp.flush()
                video_path = tmp.name
        else:
            vurl = st.text_input("Enter the video URL (mp4/mov):", "")
            if vurl:
                video_path = vurl

        # 비디오 처리 파라미터
        c1, c2, c3 = st.columns(3)
        with c1:
            frame_skip = st.slider("Frame skip", 1, 8, 2, 1, help="큰 값일수록 덜 많은 프레임을 처리 → 더 빠름")
        with c2:
            resize_w = st.slider("Resize width", 320, 1280, 960, 40, help="가로 폭 리사이즈(성능 향상용)")
        with c3:
            max_frames = st.slider("Max frames", 100, 4000, 1200, 100, help="안전 종료용 최대 처리 프레임")

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

    # 하단 안내
    st.markdown("---")
    st.caption("Tip: Cloud는 CPU 환경입니다. 프레임 스킵/리사이즈를 조절하면 더 매끄럽게 동작합니다.")


if __name__ == "__main__":
    main()
