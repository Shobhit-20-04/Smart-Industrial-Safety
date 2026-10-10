"""
Main Real-Time Application for Smart Industrial Safety Monitoring System.
Handles webcam, video file, and image inputs.
Executes OpenVINO/PyTorch detection, evaluates compliance policies, renders HUD,
and records audit logs.
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from app.detector import PPEDetector
from app.compliance import PPEComplianceEngine
from app.logger import SafetyEventLogger

# Color Palette (BGR)
COLOR_COMPLIANT = (0, 200, 0)      # Green
COLOR_VIOLATION = (0, 0, 220)      # Red
COLOR_HELMET = (0, 215, 255)       # Yellow/Gold
COLOR_VEST = (255, 180, 0)         # Cyan/Blue
COLOR_NEUTRAL = (180, 180, 180)    # Gray
COLOR_PANEL_BG = (25, 25, 25)      # Dark Gray

def draw_hud(
    frame: np.ndarray,
    compliance_data: dict,
    fps: float,
    latency_ms: float,
    model_name: str,
    runtime_name: str
) -> np.ndarray:
    """
    Renders research-grade industrial safety HUD with statistics panel and worker cards.
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # 1. Top Status Banner (Translucent Dark Bar)
    banner_height = 55
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Top Banner Text
    title_text = f"SMART INDUSTRIAL SAFETY MONITOR | {runtime_name.upper()} ({model_name})"
    cv2.putText(annotated, title_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    stat_text = (
        f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | "
        f"Workers: {compliance_data['total_workers']} "
        f"(Compliant: {compliance_data['compliant_workers']} | Violations: {compliance_data['violating_workers']}) | "
        f"Total Violations: {compliance_data['cumulative_violations']}"
    )
    cv2.putText(annotated, stat_text, (15, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 240, 255), 1)

    # 2. Draw PPE Detections (Thin bounding boxes)
    for ppe in compliance_data.get("all_ppe_detections", []):
        bx = [int(v) for v in ppe["box"]]
        cname = ppe["class_name"]
        color = COLOR_HELMET if "helmet" in cname else (COLOR_VEST if "vest" in cname else COLOR_NEUTRAL)
        cv2.rectangle(annotated, (bx[0], bx[1]), (bx[2], bx[3]), color, 1)
        label = f"{cname} {ppe['conf']:.2f}"
        cv2.putText(annotated, label, (bx[0], max(12, bx[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # 3. Draw Worker Detections & Compliance Status
    card_y_offset = banner_height + 20
    for w_info in compliance_data["worker_assessments"]:
        w_box = [int(v) for v in w_info["worker_box"]]
        is_compliant = w_info["is_compliant"]
        box_color = COLOR_COMPLIANT if is_compliant else COLOR_VIOLATION

        # Thick worker bounding box
        cv2.rectangle(annotated, (w_box[0], w_box[1]), (w_box[2], w_box[3]), box_color, 2)

        # Worker Header Tag
        tag_text = f"{w_info['worker_id']}: {w_info['status']}"
        tag_size, _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (w_box[0], max(0, w_box[1] - 24)), (w_box[0] + tag_size[0] + 10, w_box[1]), box_color, -1)
        cv2.putText(annotated, tag_text, (w_box[0] + 5, max(16, w_box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 4. Side Compliance Card (Rendered for up to 3 workers on screen)
        if card_y_offset < h - 120:
            card_w = 260
            card_h = 28 + len(w_info["checklist"]) * 20 + 22
            card_x1 = max(10, w - card_w - 15)
            card_x2 = card_x1 + card_w
            card_y1 = card_y_offset
            card_y2 = card_y1 + card_h

            # Translucent card background
            card_bg = annotated.copy()
            cv2.rectangle(card_bg, (card_x1, card_y1), (card_x2, card_y2), (20, 20, 20), -1)
            cv2.rectangle(card_bg, (card_x1, card_y1), (card_x2, card_y2), box_color, 1)
            cv2.addWeighted(card_bg, 0.85, annotated, 0.15, 0, annotated)

            # Card Title
            cv2.putText(annotated, f"[{w_info['worker_id']}] Status: {w_info['status']}",
                        (card_x1 + 8, card_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 2)

            line_y = card_y1 + 36
            for ppe_name, check_data in w_info["checklist"].items():
                status_val = check_data["status"]
                val_color = (0, 230, 0) if status_val == "YES" else (0, 80, 255)
                conf_str = f"({check_data['confidence']:.2f})" if status_val == "YES" else ""
                check_str = f"  {ppe_name.replace('-', ' ').title()}: {status_val} {conf_str}"
                cv2.putText(annotated, check_str, (card_x1 + 6, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, val_color, 1)
                line_y += 18

            # Disclaimer line
            cv2.putText(annotated, "* Missing != Definite Absence", (card_x1 + 8, card_y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 150), 1)

            card_y_offset += card_h + 10

    return annotated

def run_safety_monitor(
    source: str,
    model_path: Path,
    device: str = "cpu",
    conf_thresh: float = 0.35,
    required_ppe: list = None,
    save_violations: bool = True,
    log_file: Path = Path("results/logs/safety_events.csv"),
    output_video: Path = None,
    display: bool = True
):
    print("=" * 70)
    print("LAUNCHING SMART INDUSTRIAL SAFETY MONITORING SYSTEM")
    print("=" * 70)
    print(f"Input Source:       {source}")
    print(f"Model Architecture: {model_path}")
    print(f"Execution Device:   {device.upper()}")
    print(f"Confidence Thresh:  {conf_thresh}")
    print(f"Required PPE:       {required_ppe}")
    print(f"Audit Log File:     {log_file}")
    print("=" * 70)

    # Initialize components
    detector = PPEDetector(model_path=model_path, device=device)
    compliance_engine = PPEComplianceEngine(required_ppe=required_ppe)
    logger = SafetyEventLogger(log_file=log_file, save_screenshots=save_violations)

    # Check if source is image file
    source_path = Path(source) if not source.isdigit() else None
    is_image = source_path and source_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]

    if is_image:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise FileNotFoundError(f"Could not load image file: {source_path}")

        detections, latency_ms = detector.detect(frame, conf_thresh=conf_thresh)
        compliance_data = compliance_engine.evaluate_compliance(detections)

        # Log events
        for w_info in compliance_data["worker_assessments"]:
            logger.log_compliance_event(1, w_info, frame)

        annotated = draw_hud(
            frame,
            compliance_data,
            fps=1000.0 / latency_ms if latency_ms > 0 else 0.0,
            latency_ms=latency_ms,
            model_name=detector.model_path.name,
            runtime_name=detector.runtime_name
        )

        out_img_path = Path("results/demo_image_result.jpg")
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img_path), annotated)
        print(f"[monitor] Processed image saved to: {out_img_path}")

        # Print console checklist
        print("\n" + "=" * 50)
        print("WORKER SAFETY CHECKLIST SUMMARY")
        print("=" * 50)
        for w_info in compliance_data["worker_assessments"]:
            print(f"{w_info['worker_id']}")
            for item, data in w_info["checklist"].items():
                print(f"  {item.replace('-', ' ').title()}: {data['status']}")
            print(f"  Status: {w_info['status']}\n")
        print("=" * 50)
        return

    # Video stream / Webcam processing
    cap_src = int(source) if source.isdigit() else str(source)
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video stream/webcam source: {source}")

    video_writer = None
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if output_video:
        output_video = Path(output_video).resolve()
        output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_video), fourcc, fps_in, (frame_width, frame_height))
        print(f"[monitor] Recording video output to: {output_video}")

    frame_id = 0
    fps_history = []

    print("[monitor] Commencing real-time monitoring loop (press 'q' or ESC to stop)...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_id += 1
            t_start = time.perf_counter()

            # 1. Detection
            detections, latency_ms = detector.detect(frame, conf_thresh=conf_thresh)

            # 2. Compliance Evaluation
            compliance_data = compliance_engine.evaluate_compliance(detections)

            # 3. Log Audit Events (log every 15 frames or when violations occur)
            for w_info in compliance_data["worker_assessments"]:
                if not w_info["is_compliant"] or frame_id % 30 == 0:
                    logger.log_compliance_event(frame_id, w_info, frame)

            t_end = time.perf_counter()
            instant_fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0.0
            fps_history.append(instant_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = float(np.mean(fps_history))

            # 4. Render HUD
            annotated_frame = draw_hud(
                frame,
                compliance_data,
                fps=avg_fps,
                latency_ms=latency_ms,
                model_name=detector.model_path.name,
                runtime_name=detector.runtime_name
            )

            if video_writer:
                video_writer.write(annotated_frame)

            if display:
                cv2.imshow("Smart Industrial Safety Monitor", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), 27]: # 'q' or ESC
                    print("\n[monitor] User interrupted video feed.")
                    break

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()
        print(f"[monitor] Session concluded. Total frames processed: {frame_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Smart Industrial Safety Monitoring System.")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: '0' for webcam, path to video (.mp4), or path to image (.jpg)")
    parser.add_argument("--model", type=Path, default=Path("models/yolov8s_int8_openvino_model"),
                        help="Path to OpenVINO model folder or PyTorch .pt model file")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu", "0"],
                        help="Hardware execution device")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--required-ppe", type=str, default="helmet,safety-vest",
                        help="Comma-separated required PPE list (e.g. helmet,safety-vest,gloves,glasses)")
    parser.add_argument("--save-violations", action="store_true", default=True,
                        help="Save visual snapshot on safety violation")
    parser.add_argument("--log-file", type=Path, default=Path("results/logs/safety_events.csv"),
                        help="CSV event log destination")
    parser.add_argument("--output-video", type=Path, default=None,
                        help="Optional video output recording path")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless without opening OpenCV window")
    args = parser.parse_args()

    ppe_list = [p.strip() for p in args.required_ppe.split(",") if p.strip()]

    run_safety_monitor(
        source=args.source,
        model_path=args.model,
        device=args.device,
        conf_thresh=args.conf,
        required_ppe=ppe_list,
        save_violations=args.save_violations,
        log_file=args.log_file,
        output_video=args.output_video,
        display=not args.no_display
    )
