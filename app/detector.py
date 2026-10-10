"""
Object Detection Engine for Smart Industrial Safety Monitoring System.
Supports OpenVINO IR models (FP32, FP16, INT8) and PyTorch (.pt) weights.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2

class PPEDetector:
    """
    YOLO-based PPE and Worker Detector supporting PyTorch and OpenVINO runtimes.
    """
    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        imgsz: int = 640
    ):
        from ultralytics import YOLO

        self.model_path = Path(model_path).resolve()
        self.device = device
        self.imgsz = imgsz

        # Detect runtime format
        self.is_openvino = "_openvino_model" in self.model_path.name or self.model_path.suffix == ".xml"
        self.runtime_name = "OpenVINO" if self.is_openvino else "PyTorch"

        print(f"[detector] Initializing {self.runtime_name} model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))

        # Determine precision tag
        m_name = self.model_path.name.lower()
        if "int8" in m_name:
            self.precision = "INT8 (PTQ)"
        elif "fp16" in m_name:
            self.precision = "FP16"
        else:
            self.precision = "FP32"

        # Load class names from model
        self.class_names = self.model.names if hasattr(self.model, "names") else {}
        print(f"[detector] Loaded {len(self.class_names)} classes. Runtime: {self.runtime_name} ({self.precision}) on {self.device.upper()}")

    def detect(
        self,
        image: np.ndarray,
        conf_thresh: float = 0.30,
        iou_thresh: float = 0.45
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Runs object detection on input frame.
        Returns:
            detections: List of dicts with {'box': [x1, y1, x2, y2], 'conf': float, 'class_id': int, 'class_name': str}
            latency_ms: Inference execution time in milliseconds
        """
        t0 = time.perf_counter()
        results = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=conf_thresh,
            iou=iou_thresh,
            device=self.device,
            verbose=False
        )[0]
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cid in zip(boxes, confs, clss):
                cname = self.class_names.get(cid, str(cid))
                detections.append({
                    "box": [float(b) for b in box],
                    "conf": float(conf),
                    "class_id": int(cid),
                    "class_name": cname
                })

        return detections, latency_ms
