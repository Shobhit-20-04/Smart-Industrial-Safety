"""
High-Performance Asynchronous Event Logging and Violation Snapshot Recorder.
Offloads all disk I/O (CSV writing and image encoding) to a background thread
so inference and video display loops maintain maximum frame rate without stutter.
"""

import os
import csv
import time
import queue
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np

class SafetyEventLogger:
    """
    Asynchronous event logger that records compliance audit trails and violation image snapshots
    without blocking the video inference loop.
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

        # Asynchronous Queue & Background Worker Thread
        self._queue = queue.Queue(maxsize=500)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def _process_queue(self):
        """Background worker consuming disk I/O tasks."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                task_type = task["type"]
                if task_type == "log":
                    # 1. Optionally save image
                    screenshot_path = ""
                    if task.get("frame_to_save") is not None:
                        img_path = task["screenshot_target_path"]
                        cv2.imwrite(str(img_path), task["frame_to_save"])
                        screenshot_path = str(img_path)

                    # 2. Append CSV
                    row = task["row_data"]
                    if screenshot_path:
                        row["screenshot_path"] = screenshot_path

                    with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                        writer.writerow(row)

            except Exception as e:
                print(f"[logger] Error writing event in background thread: {e}")
            finally:
                self._queue.task_done()

    def log_compliance_event(
        self,
        frame_id: int,
        worker_info: Dict[str, Any],
        raw_frame: Optional[np.ndarray] = None
    ) -> Optional[str]:
        """
        Asynchronously enqueues an event for zero-latency execution.
        Returns immediately without stalling the video inference loop.
        """
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        status = worker_info["status"]
        screenshot_path = ""
        frame_copy = None

        if not worker_info["is_compliant"] and self.save_screenshots and raw_frame is not None:
            filename = f"violation_f{frame_id:06d}_{worker_info['worker_id'].replace(' ', '_')}_{int(time.time() * 1000)}.jpg"
            screenshot_target = self.violation_dir / filename
            screenshot_path = str(screenshot_target)

            # Draw annotation on a fast copy
            frame_copy = raw_frame.copy()
            w_box = [int(v) for v in worker_info["worker_box"]]
            cv2.rectangle(frame_copy, (w_box[0], w_box[1]), (w_box[2], w_box[3]), (0, 0, 255), 2)
            cv2.putText(
                frame_copy,
                f"PPE VIOLATION: Missing {', '.join(worker_info['missing_ppe'])}",
                (w_box[0], max(20, w_box[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

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

        task = {
            "type": "log",
            "row_data": row,
            "screenshot_target_path": screenshot_path if frame_copy is not None else None,
            "frame_to_save": frame_copy
        }

        try:
            self._queue.put_nowait(task)
        except queue.Full:
            pass # Prevent unbounded memory usage if disk write stalls

        return screenshot_path

    def delete_oldest_records(self, n: int, delete_screenshots: bool = True) -> int:
        """
        Deletes the oldest n records from the CSV audit trail and optionally deletes associated screenshots.
        Returns the number of records actually deleted.
        """
        # Ensure any in-flight asynchronous write tasks are flushed
        try:
            self._queue.join()
        except Exception:
            pass

        if not self.log_file.exists():
            return 0

        # Read all rows
        rows = []
        with open(self.log_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or self.fieldnames
            rows = list(reader)

        total_rows = len(rows)
        if total_rows == 0 or n <= 0:
            return 0

        num_to_delete = min(n, total_rows)
        rows_to_delete = rows[:num_to_delete]
        rows_to_keep = rows[num_to_delete:]

        # Delete corresponding screenshot files if requested
        if delete_screenshots:
            for r in rows_to_delete:
                s_path = r.get("screenshot_path", "").strip()
                if s_path:
                    try:
                        p = Path(s_path)
                        if p.exists() and p.is_file():
                            p.unlink(missing_ok=True)
                    except Exception as e:
                        print(f"[logger] Note: could not delete screenshot {s_path}: {e}")

        # Rewrite remaining rows back to CSV
        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_keep)

        return num_to_delete

    def close(self):
        """Flush queue and cleanly stop background thread."""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
