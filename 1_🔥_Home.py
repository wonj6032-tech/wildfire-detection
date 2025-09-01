import streamlit as st  # type: ignore
import cv2
from ultralytics import YOLO
import requests  # type: ignore
from PIL import Image
import os
from glob import glob
from numpy import random
import io

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ===================== [ADD] 안전한 가중치 로더 =====================

# ▶ Secrets에 넣어두면 코드에 노출되지 않습니다.
WEIGHT_URL = st.secrets.get(
    "WEIGHT_URL",
    ""  # 예: "https://github.com/<you>/<repo>/releases/download/<tag>/yolov8n_fire.pt"
)
LOCAL_WEIGHT = "/tmp/yolov8n_fire.pt"  # Streamlit Cloud는 /tmp 권장
MIN_VALID_SIZE = 1_000_000  # 1MB 미만이면 비정상으로 판단(포인터/손상 가능)

def _is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(256)
        return b"git-lfs" in head or b"oid sha256" in head or b"version https://git-lfs.github.com/spec" in head
    except Exception:
        return False

def _download_weight(url: str, path: str) -> None:
    st.info("Downloading model weights…")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)

def ensure_weight(path: str = LOCAL_WEIGHT, url: str = WEIGHT_URL) -> str:
    """
    - 경로에 파일이 없거나 너무 작거나(LFS 포인터 의심), 실제 LFS 포인터면 URL에서 재다운.
    - 최종적으로 유효한 로컬 경로를 반환.
    """
    need_download = False
    if not os.path.exists(path) or os.path.getsize(path) < MIN_VALID_SIZE:
        need_download = True
    elif _is_lfs_pointer(path):
        need_download = True

    if need_download:
        if not url:
            raise RuntimeError("WEIGHT_URL이 설정되지 않았습니다. Streamlit Secrets 또는 코드 상단 WEIGHT_URL을 설정하세요.")
        _download_weight(url, path)

    if _is_lfs_pointer(path) or os.path.getsize(path) < MIN_VALID_SIZE:
        raise RuntimeError("가중치 파일이 여전히 LFS 포인터/손상 상태로 보입니다.")
    return path

# ===================== [MOD] 모델 로더 (런타임 다운로드 + 폴백) =====================

@st.cache_resource
def load_model(model_path: str | None):
    """
    1) 사용자가 고른 경로가 정상 .pt면 그대로 로드
    2) 비정상이면 런타임 다운로드(ensure_weight)로 우회
    3) 그래도 실패하면 yolov8n.pt로 폴백
    """
    # 1) 사용자가 선택한 경로 우선 시도
    if model_path and os.path.exists(model_path) and os.path.getsize(model_path) >= MIN_VALID_SIZE and not _is_lfs_pointer(model_path):
        try:
            return YOLO(model_path)
        except Exception as e:
            st.warning(f"커스텀 가중치 로드 실패: {e}. 런타임 다운로드로 재시도합니다…")

    # 2) 런타임 다운로드 우회
    try:
        wp = ensure_weight()
        return YOLO(wp)
    except Exception as e:
        st.warning(f"런타임 다운로드 방식 실패: {e}. 기본 yolov8n.pt로 폴백합니다…")

    # 3) 최후의 폴백
    return YOLO("yolov8n.pt")

# ===================== 추론 함수 =====================

def predict_image(model, image, conf_threshold, iou_threshold):
    res = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',
    )

    class_name = model.model.names
    classes = res[0].boxes.cls if res and res[0].boxes is not None else []
    class_counts = {}

    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

    prediction_text = 'Predicted '
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}'
        if v > 1:
            prediction_text += 's'
        prediction_text += ', '
    prediction_text = prediction_text[:-2] if class_counts else "No objects detected"

    latency_ms = sum(res[0].speed.values()) if res and len(res) > 0 else 0.0
    latency = round(latency_ms / 1000, 2)
    prediction_text += f' in {latency} seconds.'

    res_image = res[0].plot() if res and len(res) > 0 else image
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)

    return res_image, prediction_text

# ===================== 메인 =====================

def main():
    st.set_page_config(
        page_title="Wildfire Detection",
        page_icon="🔥",
        initial_sidebar_state="collapsed",
    )

    st.sidebar.markdown("Developed by Alim Tleuliyev")
    st.sidebar.markdown("LinkedIn: [Profile](https://www.linkedin.com/in/alimtleuliyev/)")
    st.sidebar.markdown("GitHub: [Repo](https://github.com/AlimTleuliyev/wildfire-detection)")
    st.sidebar.markdown("Email: [alim.tleuliyev@nu.edu.kz](mailto:alim.tleuliyev@nu.edu.kz)")
    st.sidebar.markdown("Telegram: [@nativealim](https://t.me/nativealim)")

    st.markdown(
        """
        <style>
        .container { max-width: 800px; }
        .title { text-align: center; font-size: 35px; font-weight: bold; margin-bottom: 10px; }
        .description { margin-bottom: 30px; }
        .instructions { margin-bottom: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div class='title'>Wildfire Detection</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write("")
    with col2:
        logos = glob('dalle-logos/*.png')
        if logos:
            logo = random.choice(logos)
            st.image(logo, use_column_width=True, caption="Generated by DALL-E")
            st.sidebar.image(logo, use_column_width=True, caption="Generated by DALL-E")
        else:
            st.caption("No logo images found.")
            logo = None
    with col3:
        st.write("")

    st.markdown(
        """
        <div style='text-align: center;'>
            <h2>🔥 <strong>Wildfire Detection App</strong></h2>
            <p>Welcome! Powered by <a href='https://github.com/ultralytics/ultralytics'>YOLOv8</a> trained on <a href='https://github.com/gaiasd/DFireDataset'>D-Fire</a>.</p>
            <h3>🌍 <strong>Preventing Wildfires with Computer Vision</strong></h3>
            <p>Detect fire and smoke in images with high accuracy and speed.</p>
            <h3>📸 <strong>Try It Out!</strong></h3>
            <p>Upload an image or provide a URL.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    # 모델 선택
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.radio("Select Model Type", ("Fire Detection", "General"), index=0)

    models_dir = "general-models" if model_type == "General" else "fire-models"
    model_files = []
    if os.path.isdir(models_dir):
        model_files = [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]
    if not model_files:
        st.warning(f"'{models_dir}' 폴더에서 .pt 파일을 찾지 못했습니다. 런타임 다운로드/폴백을 사용합니다.")
        selected_model = None
    else:
        with col2:
            # 인덱스 보호
            default_idx = min(2, max(0, len(sorted(model_files)) - 1))
            selected_model = st.selectbox("Select Model Size", sorted(model_files), index=default_idx)

    # 모델 로드
    model_path = os.path.join(models_dir, selected_model + ".pt") if selected_model else None
    with st.spinner("Loading model…"):
        model = load_model(model_path)

    st.markdown("---")

    # Thresholds
    col1, col2 = st.columns(2)
    with col2:
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)
        with st.expander("What is Confidence Threshold?"):
            st.caption("Minimum confidence for a detection to be kept.")
    with col1:
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)
        with st.expander("What is IOU Threshold?"):
            st.caption("Overlap threshold for NMS.")

    st.markdown("---")

    # 이미지 입력
    image = None
    image_source = st.radio("Select image source:", ("Enter URL", "Upload from Computer"))

    if image_source == "Upload from Computer":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        url = st.text_input("Enter the image URL:")
        if url:
            try:
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                image = Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception as e:
                st.error(f"Error loading image from URL: {e}")
                image = None

    if image:
        with st.spinner("Detecting…"):
            prediction, text = predict_image(model, image, conf_threshold, iou_threshold)
            st.image(prediction, caption="Prediction", use_column_width=True)
            st.success(text)

        prediction_pil = Image.fromarray(prediction)
        buf = io.BytesIO()
        prediction_pil.save(buf, format='PNG')
        st.download_button(
            label='Download Prediction',
            data=buf.getvalue(),
            file_name='prediction.png',
            mime='image/png'
        )

if __name__ == "__main__":
    main()
