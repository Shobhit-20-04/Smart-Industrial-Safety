"""
High-Performance Object Detection Engine for Smart Industrial Safety Monitoring System.
Supports OpenVINO IR models (FP32, FP16, INT8) and GPU-accelerated PyTorch (.pt) weights.
Includes automatic hardware acceleration, model warm-up, and optimized batching.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2

class PPEDetector:
    """
    Optimized YOLO-based PPE and Worker Detector supporting PyTorch (GPU/CPU)
    and Intel OpenVINO runtimes with dynamic hardware routing.
    """
    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        imgsz: int = 640
    ):
        import torch
        from ultralytics import YOLO

        self.model_path = Path(model_path).resolve()
        self.imgsz = imgsz

        # Detect runtime format
        self.is_openvino = "_openvino_model" in self.model_path.name or self.model_path.suffix == ".xml"
        self.runtime_name = "OpenVINO" if self.is_openvino else "PyTorch"

        # Hardware acceleration resolution
        if device == "auto":
            if self.is_openvino:
                self.device = "cpu" # Default OpenVINO device
            else:
                self.device = "0" if torch.cuda.is_available() else "cpu"
        elif device in ["gpu", "cuda"]:
            self.device = "0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[detector] Initializing {self.runtime_name} engine on target: {self.device.upper()}")
        self.model = YOLO(str(self.model_path))

        # Precision classification
        m_name = self.model_path.name.lower()
        if "int8" in m_name:
            self.precision = "INT8 (PTQ)"
        elif "fp16" in m_name:
            self.precision = "FP16"
        else:
            self.precision = "FP32"

        # Load class mappings
        self.class_names = self.model.names if hasattr(self.model, "names") else {}

        # Determine FP16 / Tensor Core acceleration
        self.use_half = False
        if not self.is_openvino and self.device in ["0", "cuda", "cuda:0"] and torch.cuda.is_available():
            self.use_half = True
            print(f"[detector] Enabled Tensor Core FP16 half-precision on {self.device.upper()}")

        # Pre-warm model with dummy inference to avoid initial frame latency spikes
        dummy_frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            warm_kwargs = {
                "imgsz": self.imgsz,
                "device": self.device,
                "verbose": False
            }
            if self.use_half:
                warm_kwargs["half"] = True
            _ = self.model.predict(dummy_frame, **warm_kwargs)
            print(f"[detector] Engine pre-warmed. Active classes: {len(self.class_names)} ({self.runtime_name} {self.precision})")
        except Exception as e:
            print(f"[detector] Note: warm-up skipped: {e}")

    def detect(
        self,
        image: np.ndarray,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        imgsz: int = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes high-speed inference on the input image.
        Returns:
            detections: List of parsed detection dictionaries
            latency_ms: Exact measured inference latency in milliseconds
        """
        target_imgsz = imgsz or self.imgsz
        pred_kwargs = {
            "imgsz": target_imgsz,
            "conf": conf_thresh,
            "iou": iou_thresh,
            "device": self.device,
            "verbose": False
        }
        if self.use_half:
            pred_kwargs["half"] = True

        t0 = time.perf_counter()
        results = self.model.predict(image, **pred_kwargs)[0]
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
