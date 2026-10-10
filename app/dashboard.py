"""
Streamlit Web Dashboard for Smart Industrial Safety Monitoring System.
Ultra-responsive, real-time visualization of PPE compliance, violation counters,
model selection (CUDA GPU vs OpenVINO INT8/FP16 CPU), and audit logs.
Optimized for high-throughput video processing and zero UI lag.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import streamlit as st
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.detector import PPEDetector
from app.compliance import PPEComplianceEngine
from app.logger import SafetyEventLogger
from app.main import draw_hud, ThreadedVideoStream

# Page Configuration
st.set_page_config(
    page_title="Smart Industrial Safety Monitor",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🦺 Smart Industrial Safety Monitoring System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Research-Grade Real-Time PPE Compliance & Hardware Acceleration Platform</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
st.sidebar.header("⚙️ Performance & Model Controls")

has_cuda = torch.cuda.is_available()
cuda_name = torch.cuda.get_device_name(0) if has_cuda else "None"

# Model Selection with explicit Hardware Routing
MODEL_OPTIONS = {}

if has_cuda:
    MODEL_OPTIONS["🚀 YOLOv8s (NVIDIA GPU - CUDA FP16 Balanced)"] = {
        "path": PROJECT_ROOT / "models" / "yolov8s_sh17_best.pt",
        "precision": 0.7668,
        "map50": 0.6214,
        "runtime": "PyTorch GPU",
        "device": "0"
    }
    MODEL_OPTIONS["⚡ YOLOv8n (NVIDIA GPU - CUDA FP16 Ultra-Fast)"] = {
        "path": PROJECT_ROOT / "models" / "yolov8n_sh17_best.pt",
        "precision": 0.6701,
        "map50": 0.5473,
        "runtime": "PyTorch GPU",
        "device": "0"
    }

MODEL_OPTIONS["⚡ OpenVINO INT8 (Intel CPU - PTQ High-Speed)"] = {
    "path": PROJECT_ROOT / "models" / "yolov8s_int8_openvino_model",
    "precision": 0.7039,
    "map50": 0.6178,
    "runtime": "OpenVINO INT8",
    "device": "cpu"
}

yolov8n_ov = PROJECT_ROOT / "models" / "yolov8n_sh17_best_openvino_model"
if yolov8n_ov.exists():
    MODEL_OPTIONS["⚡ OpenVINO FP16 YOLOv8n (Intel CPU Lightweight)"] = {
        "path": yolov8n_ov,
        "precision": 0.6701,
        "map50": 0.5473,
        "runtime": "OpenVINO FP16",
        "device": "cpu"
    }

MODEL_OPTIONS["🔧 OpenVINO FP16 YOLOv8s (Intel CPU)"] = {
    "path": PROJECT_ROOT / "models" / "yolov8s_fp16_openvino_model",
    "precision": 0.7062,
    "map50": 0.6184,
    "runtime": "OpenVINO FP16",
    "device": "cpu"
}

MODEL_OPTIONS["🔧 OpenVINO FP32 YOLOv8s (Intel CPU)"] = {
    "path": PROJECT_ROOT / "models" / "yolov8s_fp32_openvino_model",
    "precision": 0.7062,
    "map50": 0.6184,
    "runtime": "OpenVINO FP32",
    "device": "cpu"
}

MODEL_OPTIONS["🐢 PyTorch FP32 YOLOv8s (CPU Fallback)"] = {
    "path": PROJECT_ROOT / "models" / "yolov8s_sh17_best.pt",
    "precision": 0.7668,
    "map50": 0.6214,
    "runtime": "PyTorch CPU",
    "device": "cpu"
}

selected_model_name = st.sidebar.selectbox("Model Architecture & Hardware", list(MODEL_OPTIONS.keys()))
model_meta = MODEL_OPTIONS[selected_model_name]

# Resolution & Frame Stride Speed Controls
st.sidebar.subheader("⚡ Speed & Responsiveness Tuning")
col_res, col_stride = st.sidebar.columns(2)
with col_res:
    inference_imgsz = st.selectbox("Resolution", [480, 512, 640], index=0, help="480p gives maximum response speed; 640p gives full detail.")
with col_stride:
    frame_stride = st.selectbox("Frame Stride", [1, 2, 3], index=1, help="Process every Nth frame (Stride 2 doubles playback throughput).")

# Confidence & Policy Controls
conf_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.15, 0.90, 0.35, 0.05)

st.sidebar.subheader("Mandatory PPE Policy")
PPE_CHOICES = ["helmet", "safety-vest", "gloves", "glasses", "face-mask"]
required_ppe = st.sidebar.multiselect(
    "Select Required Safety Gear",
    PPE_CHOICES,
    default=["helmet", "safety-vest"]
)

save_violations = st.sidebar.checkbox("Save Violation Snapshots", value=True)

# Input Source
st.sidebar.subheader("Video / Stream Source")
input_source_type = st.sidebar.radio(
    "Choose Input Stream",
    ["Sample Industrial Image", "Image Upload", "Video File Upload", "Live Webcam Feed (Device 0)"]
)

# -------------------------------------------------------------
# Detector & Engine Cache
# -------------------------------------------------------------
@st.cache_resource
def get_detector(model_path_str: str, device: str, imgsz: int):
    return PPEDetector(model_path=model_path_str, device=device, imgsz=imgsz)

try:
    detector = get_detector(str(model_meta["path"]), model_meta["device"], inference_imgsz)
except Exception as e:
    st.error(f"Failed to load model from {model_meta['path']}: {e}")
    st.stop()

compliance_engine = PPEComplianceEngine(required_ppe=required_ppe)
logger = SafetyEventLogger(
    log_file=PROJECT_ROOT / "results" / "logs" / "safety_events.csv",
    violation_dir=PROJECT_ROOT / "results" / "logs" / "violations",
    save_screenshots=save_violations
)

# -------------------------------------------------------------
# Main Display Layout
# -------------------------------------------------------------
# Top Metrics Row
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
metric_workers = kpi1.empty()
metric_compliant = kpi2.empty()
metric_violations = kpi3.empty()
metric_fps = kpi4.empty()
metric_latency = kpi5.empty()
metric_precision = kpi6.empty()

video_placeholder = st.empty()
details_expander = st.expander("📋 Live Worker Compliance Breakdown", expanded=True)
details_placeholder = details_expander.empty()

# Disclaimer Box
st.info(
    "⚠️ **Compliance Reasoning Disclaimer:** Missing detections are designated as **'NOT DETECTED (UNVERIFIED)'**. "
    "Under physical occlusions, extreme viewpoints, or sensor limitations, an unobserved detection does not constitute "
    "definitive proof of physical absence."
)

# -------------------------------------------------------------
# High-Speed Frame Processing
# -------------------------------------------------------------
def process_single_frame(frame_bgr: np.ndarray, frame_id: int = 1, update_ui_metrics: bool = True):
    detections, latency_ms = detector.detect(
        frame_bgr,
        conf_thresh=conf_threshold,
        imgsz=inference_imgsz
    )
    compliance_data = compliance_engine.evaluate_compliance(detections)

    # Log events asynchronously
    for w_info in compliance_data["worker_assessments"]:
        if not w_info["is_compliant"] or frame_id % 30 == 0:
            logger.log_compliance_event(frame_id, w_info, frame_bgr)

    fps_val = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    # Draw HUD
    annotated = draw_hud(
        frame_bgr,
        compliance_data,
        fps=fps_val,
        latency_ms=latency_ms,
        model_name=model_meta["runtime"],
        runtime_name=detector.runtime_name
    )

    # Render RGB frame
    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, width='stretch')

    # Throttled KPI & Table Updates (only every N frames to eliminate Streamlit WebSocket lag)
    if update_ui_metrics:
        metric_workers.metric("Workers Detected", compliance_data["total_workers"])
        metric_compliant.metric("Compliant", compliance_data["compliant_workers"])
        metric_violations.metric("Violations", compliance_data["violating_workers"])
        metric_fps.metric("Throughput (FPS)", f"{fps_val:.1f}")
        metric_latency.metric("Latency", f"{latency_ms:.1f} ms")
        metric_precision.metric("Model Precision", f"{model_meta['precision']*100:.1f}%")

        if compliance_data["worker_assessments"]:
            table_rows = []
            for w in compliance_data["worker_assessments"]:
                row = {"Worker ID": w["worker_id"], "Overall Status": w["status"]}
                for req in required_ppe:
                    check = w["checklist"].get(req, {"status": "N/A"})
                    row[req.replace("-", " ").title()] = check["status"]
                table_rows.append(row)
            details_placeholder.dataframe(pd.DataFrame(table_rows), width='stretch')
        else:
            details_placeholder.write("No active workers detected in frame.")

if input_source_type == "Sample Industrial Image":
    sample_img_path = PROJECT_ROOT / "datasets" / "SH17" / "images" / "val" / "pexels-photo-10246146.jpeg"
    if sample_img_path.exists():
        frame = cv2.imread(str(sample_img_path))
        process_single_frame(frame, frame_id=1, update_ui_metrics=True)
    else:
        st.warning("Sample image not found.")

elif input_source_type == "Image Upload":
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        process_single_frame(frame, frame_id=1, update_ui_metrics=True)
    else:
        st.info("Please upload an image from the sidebar.")

elif input_source_type == "Video File Upload":
    uploaded_video = st.sidebar.file_uploader("Upload MP4 Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        temp_video_path = PROJECT_ROOT / "runs" / "temp_upload.mp4"
        temp_video_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(str(temp_video_path))
        stop_btn = st.sidebar.button("Stop Video Playback")
        frame_idx = 0
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_idx += 1
            if frame_idx % frame_stride == 0:
                should_update_table = (frame_idx % (frame_stride * 4) == 0)
                process_single_frame(frame, frame_id=frame_idx, update_ui_metrics=should_update_table)
        cap.release()

elif input_source_type == "Live Webcam Feed (Device 0)":
    run_webcam = st.sidebar.checkbox("Start Live Webcam Stream", value=False)
    if run_webcam:
        try:
            stream = ThreadedVideoStream(0)
            frame_idx = 0
            while run_webcam and not stream.stopped:
                ret, frame = stream.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                frame_idx += 1
                if frame_idx % frame_stride == 0:
                    should_update_table = (frame_idx % (frame_stride * 4) == 0)
                    process_single_frame(frame, frame_id=frame_idx, update_ui_metrics=should_update_table)
            stream.stop()
        except Exception as e:
            st.error(f"Cannot initialize webcam stream: {e}")
    else:
        st.info("Check 'Start Live Webcam Stream' in the sidebar to initiate real-time video.")

# -------------------------------------------------------------
# Bottom Section: Audit Log Viewer
# -------------------------------------------------------------
st.subheader("📜 Recent Safety Violation Audit Trail")
log_path = PROJECT_ROOT / "results" / "logs" / "safety_events.csv"
if log_path.exists():
    try:
        log_df = pd.read_csv(log_path)
        if not log_df.empty:
            st.dataframe(log_df.tail(15).iloc[::-1], width='stretch')
        else:
            st.write("No audit events recorded yet.")
    except Exception as e:
        st.write(f"Audit log status: {e}")
else:
    st.write("Audit log initialized; awaiting events.")
