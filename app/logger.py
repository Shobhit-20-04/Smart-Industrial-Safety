"""
Event Logging and Violation Snapshot Recorder.
Maintains structured CSV event audit trails and optional visual violation archives.
"""

import os
import csv
import time
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np

class SafetyEventLogger:
    """
    Records compliance audit trails and violation image snapshots.
    """
    def __init__(
        self,
        log_file: str | Path = "results/logs/safety_events.csv",
        violation_dir: str | Path = "results/logs/violations",
        save_screenshots: bool = True
    ):
        self.log_file = Path(log_file).resolve()
        self.violation_dir = Path(violation_dir).resolve()
        self.save_screenshots = save_screenshots

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.save_screenshots:
            self.violation_dir.mkdir(parents=True, exist_ok=True)

        self.fieldnames = [
            "timestamp",
            "frame_id",
            "worker_id",
            "status",
            "detected_ppe",
            "missing_ppe",
            "worker_confidence",
            "screenshot_path"
        ]

        if not self.log_file.exists():
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_compliance_event(
        self,
        frame_id: int,
        worker_info: Dict[str, Any],
        raw_frame: Optional[np.ndarray] = None
    ) -> Optional[str]:
        """
        Logs an individual worker compliance event to the CSV log file.
        If the event is a violation and save_screenshots is enabled, writes an image snapshot.
        """
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        status = worker_info["status"]
        screenshot_path = ""

        # Save violation snapshot if requested
        if not worker_info["is_compliant"] and self.save_screenshots and raw_frame is not None:
            filename = f"violation_f{frame_id:06d}_{worker_info['worker_id'].replace(' ', '_')}_{int(time.time())}.jpg"
            save_path = self.violation_dir / filename
            
            # Create a localized crop or annotated full frame
            annotated_frame = raw_frame.copy()
            w_box = [int(v) for v in worker_info["worker_box"]]
            cv2.rectangle(annotated_frame, (w_box[0], w_box[1]), (w_box[2], w_box[3]), (0, 0, 255), 2)
            cv2.putText(
                annotated_frame,
                f"PPE VIOLATION: Missing {', '.join(worker_info['missing_ppe'])}",
                (w_box[0], max(20, w_box[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
            cv2.imwrite(str(save_path), annotated_frame)
            screenshot_path = str(save_path)

        # Write CSV row
        row = {
            "timestamp": timestamp_str,
            "frame_id": frame_id,
            "worker_id": worker_info["worker_id"],
            "status": status,
            "detected_ppe": "; ".join(worker_info["detected_ppe"]) if worker_info["detected_ppe"] else "NONE",
            "missing_ppe": "; ".join(worker_info["missing_ppe"]) if worker_info["missing_ppe"] else "NONE",
            "worker_confidence": worker_info["worker_conf"],
            "screenshot_path": screenshot_path
        }

        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

        return screenshot_path
