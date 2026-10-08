"""
Real-Time Industrial Safety & PPE Compliance Monitoring Application.
Processes live webcam streams, video files, or still images.
Associates detected PPE (helmet, vest, glasses, gloves) with detected workers
and flags compliance violations according to configs/compliance.yaml.
"""

import time
import argparse
from pathlib import Path
import cv2
import yaml
import numpy as np

def load_compliance_config(cfg_path: Path):
    if not cfg_path.exists():
        return {
            "mandatory_ppe": ["helmet", "safety-vest"],
            "recommended_ppe": ["glasses", "gloves"],
            "association": {"head_region_fraction": 0.35, "torso_region_fraction": 0.60}
        }
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("compliance_rules", {})

def run_monitor(
    model_path: Path,
    source: str = "0",
    device: str = "auto",
    conf_thresh: float = 0.35,
    imgsz: int = 640,
    compliance_cfg_path: Path = None
):
    from ultralytics import YOLO

    if compliance_cfg_path is None:
        compliance_cfg_path = Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/configs/compliance.yaml")
    rules = load_compliance_config(compliance_cfg_path)
    mandatory = rules.get("mandatory_ppe", ["helmet", "safety-vest"])
    recommended = rules.get("recommended_ppe", ["glasses", "gloves"])

    print("=" * 70)
    print("STARTING REAL-TIME PPE SAFETY MONITOR")
    print(f"Model:           {model_path}")
    print(f"Source:          {source}")
    print(f"Mandatory PPE:   {mandatory}")
    print(f"Recommended PPE: {recommended}")
    print("=" * 70)

    model = YOLO(str(model_path))

    # Parse video capture source
    is_cam = source.isdigit()
    cap_src = int(source) if is_cam else source
    cap = cv2.VideoCapture(cap_src)

    if not cap.isOpened():
        print(f"[Error] Failed to open video source: {source}")
        return

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Info] End of video stream or input source.")
            break

        cur_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(cur_time - prev_time, 1e-5))
        prev_time = cur_time

        # Run inference
        results = model.predict(frame, conf=conf_thresh, imgsz=imgsz, device=device if device != "auto" else None, verbose=False)[0]

        # Extract person and PPE detections
        persons = []
        ppe_items = []

        names = results.names
        for box in results.boxes:
            cls_id = int(box.cls.item())
            cname = names.get(cls_id, str(cls_id))
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf.item())

            if cname.lower() == "person":
                persons.append({"box": xyxy, "conf": conf, "ppe": set()})
            else:
                ppe_items.append({"name": cname.lower(), "box": xyxy, "conf": conf})

        # Associate PPE to persons
        for person in persons:
            px1, py1, px2, py2 = person["box"]
            for item in ppe_items:
                ix1, iy1, ix2, iy2 = item["box"]
                icx = (ix1 + ix2) / 2
                icy = (iy1 + iy2) / 2
                # Check if center of PPE box is inside the person bounding box
                if px1 <= icx <= px2 and py1 <= icy <= py2:
                    person["ppe"].add(item["name"])

        # Draw overlays
        # 1. PPE boxes
        for item in ppe_items:
            ix1, iy1, ix2, iy2 = item["box"]
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (255, 200, 0), 2)
            cv2.putText(frame, f"{item['name']} {item['conf']:.2f}", (ix1, max(iy1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        # 2. Worker boxes and compliance banner
        for w_idx, person in enumerate(persons):
            px1, py1, px2, py2 = person["box"]
            detected_ppe = person["ppe"]
            missing_mandatory = [m for m in mandatory if m not in detected_ppe]
            is_compliant = len(missing_mandatory) == 0

            box_color = (0, 220, 0) if is_compliant else (0, 0, 230)
            status_text = "COMPLIANT" if is_compliant else f"VIOLATION: Missing {', '.join(missing_mandatory)}"

            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            # Worker ID & Status Banner
            cv2.putText(frame, f"Worker #{w_idx+1}: {status_text}", (px1, max(py1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        # Draw system HUD
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Workers: {len(persons)}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Smart Industrial Safety PPE Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time PPE compliance monitoring.")
    parser.add_argument("--model", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/models/best.pt"),
                        help="Trained YOLO weights or OpenVINO model")
    parser.add_argument("--source", type=str, default="0", help="Webcam (0), video path, or image path")
    parser.add_argument("--device", type=str, default="auto", help="Inference device (0, cpu)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    args = parser.parse_args()

    run_monitor(
        model_path=args.model,
        source=args.source,
        device=args.device,
        conf_thresh=args.conf,
        imgsz=args.imgsz
    )
