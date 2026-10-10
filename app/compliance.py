"""
PPE Compliance Reasoning Engine for Smart Industrial Safety Monitoring System.
Maintains a strict architectural separation between raw object detection and domain compliance reasoning.

Scientific Principle:
A missing detection does NOT prove absence of physical PPE. Unobserved items are explicitly
classified as 'NOT DETECTED (UNVERIFIED)' to avoid false biological or factual claims.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

class PPEComplianceEngine:
    """
    Evaluates spatial association between workers and PPE items,
    determines compliance status, and tracks violation statistics.
    """
    # Mapping of PPE concepts to compatible model class names
    # High-visibility body gear satisfies mandatory upper-body protection (vests, jackets, and safety-suits)
    PPE_SYNONYMS = {
        "helmet": ["helmet", "blue helmet", "red helmet", "white helmet", "yellow helmet"],
        "safety-vest": ["safety-vest", "vest", "safety-suit"],
        "gloves": ["gloves"],
        "glasses": ["glasses", "glass"],
        "face-mask": ["face-mask-medical"],
        "face-guard": ["face-guard"],
        "safety-suit": ["safety-suit", "medical-suit", "safety-vest"]
    }

    def __init__(
        self,
        required_ppe: List[str] = None
    ):
        # Default mandatory industrial PPE
        self.required_ppe = required_ppe or ["helmet", "safety-vest"]
        self.cumulative_violations = 0
        self.total_frames_processed = 0

    def set_required_ppe(self, required_ppe: List[str]):
        """Dynamically update required PPE checklist."""
        self.required_ppe = required_ppe

    @staticmethod
    def _is_box_associated(worker_box: List[float], ppe_box: List[float], tolerance: float = 0.15) -> bool:
        """
        Determines if a PPE item's bounding box belongs to a worker bounding box
        using spatial overlap and center containment with boundary margin.
        """
        wx1, wy1, wx2, wy2 = worker_box
        w_width = wx2 - wx1
        w_height = wy2 - wy1

        # Expand worker box slightly by tolerance to catch helmets above head or shoes at ground
        expanded_wx1 = wx1 - w_width * tolerance
        expanded_wy1 = wy1 - w_height * tolerance
        expanded_wx2 = wx2 + w_width * tolerance
        expanded_wy2 = wy2 + w_height * tolerance

        # PPE center point
        pxc = (ppe_box[0] + ppe_box[2]) / 2.0
        pyc = (ppe_box[1] + ppe_box[3]) / 2.0

        return (expanded_wx1 <= pxc <= expanded_wx2) and (expanded_wy1 <= pyc <= expanded_wy2)

    def evaluate_compliance(
        self,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates raw detections against safety compliance policies.
        Returns detailed compliance payload with per-worker checklists.
        """
        self.total_frames_processed += 1

        # 1. Separate workers from PPE detections
        workers = []
        ppe_items = []

        for d in detections:
            cname = d["class_name"].lower()
            if cname in ["person", "worker"]:
                workers.append(d)
            else:
                ppe_items.append(d)

        # 2. Evaluate each worker individually
        worker_assessments = []
        frame_violations = 0

        for idx, worker in enumerate(workers, start=1):
            w_box = worker["box"]
            checklist = {}
            detected_ppe_list = []
            missing_ppe_list = []

            # Check each required PPE policy item
            for req_item in self.required_ppe:
                synonyms = self.PPE_SYNONYMS.get(req_item.lower(), [req_item.lower()])
                found = False
                matched_conf = 0.0
                matched_box = None

                for item in ppe_items:
                    if item["class_name"].lower() in synonyms:
                        if self._is_box_associated(w_box, item["box"]):
                            found = True
                            if item["conf"] > matched_conf:
                                matched_conf = item["conf"]
                                matched_box = item["box"]

                if found:
                    checklist[req_item] = {
                        "status": "YES",
                        "detected": True,
                        "confidence": round(matched_conf, 3),
                        "box": matched_box
                    }
                    detected_ppe_list.append(req_item)
                else:
                    checklist[req_item] = {
                        "status": "NOT DETECTED",
                        "detected": False,
                        "confidence": 0.0,
                        "note": "Unobserved in current frame (does not conclusively prove absence)"
                    }
                    missing_ppe_list.append(req_item)

            # Determine overall worker status
            is_compliant = len(missing_ppe_list) == 0
            if not is_compliant:
                frame_violations += 1

            status_text = "COMPLIANT" if is_compliant else "PPE VIOLATION"

            worker_assessments.append({
                "worker_id": f"Worker {idx}",
                "worker_box": w_box,
                "worker_conf": round(worker["conf"], 3),
                "checklist": checklist,
                "status": status_text,
                "is_compliant": is_compliant,
                "detected_ppe": detected_ppe_list,
                "missing_ppe": missing_ppe_list
            })

        self.cumulative_violations += frame_violations

        return {
            "total_workers": len(workers),
            "compliant_workers": len(workers) - frame_violations,
            "violating_workers": frame_violations,
            "cumulative_violations": self.cumulative_violations,
            "required_ppe_policies": self.required_ppe,
            "worker_assessments": worker_assessments,
            "all_ppe_detections": ppe_items,
            "disclaimer": "Note: 'NOT DETECTED' signifies an unverified sensor observation and must not be interpreted as definitive physical absence."
        }
