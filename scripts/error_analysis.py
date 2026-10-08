"""
Research-Grade Error Analysis Script.
Identifies, categorizes, and visually saves:
- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)
- Low-Confidence Detections
Saves visual overlays in results/evaluation/errors/ and metadata in error_summary.csv.
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict
import cv2
import yaml
import numpy as np

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def run_error_analysis(
    weights_path: Path,
    data_yaml: Path,
    output_dir: Path,
    split: str = "val",
    max_samples_per_category: int = 15,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50
):
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    tp_dir = output_dir / "true_positives"
    fp_dir = output_dir / "false_positives"
    fn_dir = output_dir / "false_negatives"
    low_conf_dir = output_dir / "low_confidence"

    for d in [tp_dir, fp_dir, fn_dir, low_conf_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    class_names = [cfg["names"][i] for i in sorted(cfg["names"].keys())]
    ds_base = Path(cfg["path"])
    img_dir = ds_base / cfg.get(split, "images/val")
    lbl_dir = ds_base / "labels" / Path(cfg.get(split, "images/val")).name
    if not lbl_dir.exists():
        lbl_dir = ds_base / "labels"

    print("=" * 70)
    print(f"RUNNING ERROR ANALYSIS FOR: {weights_path.name}")
    print(f"Images Dir: {img_dir}")
    print(f"Labels Dir: {lbl_dir}")
    print("=" * 70)

    model = YOLO(str(weights_path))

    saved_counts = {"tp": 0, "fp": 0, "fn": 0, "low_conf": 0}
    summary_rows = []

    img_paths = sorted(list(img_dir.glob("*.[jJ][pP][gG]")) + list(img_dir.glob("*.[pP][nN][gG]")))

    for img_p in img_paths:
        if all(cnt >= max_samples_per_category for cnt in saved_counts.values()):
            break

        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        orig_img = cv2.imread(str(img_p))
        if orig_img is None:
            continue
        h_img, w_img = orig_img.shape[:2]

        # Load GT boxes
        gt_boxes = []
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    xc, yc, w, h = [float(x) for x in parts[1:]]
                    x1 = int((xc - w / 2.0) * w_img)
                    y1 = int((yc - h / 2.0) * h_img)
                    x2 = int((xc + w / 2.0) * w_img)
                    y2 = int((yc + h / 2.0) * h_img)
                    gt_boxes.append({"cid": cid, "cname": class_names[cid], "box": [x1, y1, x2, y2], "matched": False})

        # Predict
        res = model.predict(img_p, conf=conf_thresh, verbose=False)[0]
        preds = []
        for b in res.boxes:
            cid = int(b.cls.item())
            cname = class_names[cid] if cid < len(class_names) else str(cid)
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            conf = float(b.conf.item())
            preds.append({"cid": cid, "cname": cname, "box": xyxy, "conf": conf, "matched": False})

        # Match preds to GTs
        has_tp, has_fp, has_fn, has_low = False, False, False, False

        for p in preds:
            best_iou = 0.0
            best_gt = -1
            for g_idx, g in enumerate(gt_boxes):
                if g["cid"] == p["cid"] and not g["matched"]:
                    iou = compute_iou(p["box"], g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g_idx

            if best_iou >= iou_thresh and best_gt >= 0:
                p["matched"] = True
                gt_boxes[best_gt]["matched"] = True
                has_tp = True
            else:
                has_fp = True

            if p["conf"] < 0.45:
                has_low = True

        if any(not g["matched"] for g in gt_boxes):
            has_fn = True

        # Render visualizations for representative cases
        def draw_annotated(vis_img, gts, predictions):
            out = vis_img.copy()
            # Draw GT in blue
            for g in gts:
                x1, y1, x2, y2 = g["box"]
                cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(out, f"GT: {g['cname']}", (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            # Draw Preds in green (TP) or red (FP)
            for p in predictions:
                x1, y1, x2, y2 = p["box"]
                col = (0, 220, 0) if p["matched"] else (0, 0, 255)
                cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
                cv2.putText(out, f"{p['cname']} {p['conf']:.2f}", (x1, max(y2 + 12, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            return out

        if has_tp and saved_counts["tp"] < max_samples_per_category:
            rendered = draw_annotated(orig_img, gt_boxes, preds)
            save_p = tp_dir / f"tp_{saved_counts['tp']+1:02d}_{img_p.name}"
            cv2.imwrite(str(save_p), rendered)
            saved_counts["tp"] += 1
            summary_rows.append({"category": "True Positive", "image": img_p.name, "path": str(save_p)})

        if has_fp and saved_counts["fp"] < max_samples_per_category:
            rendered = draw_annotated(orig_img, gt_boxes, preds)
            save_p = fp_dir / f"fp_{saved_counts['fp']+1:02d}_{img_p.name}"
            cv2.imwrite(str(save_p), rendered)
            saved_counts["fp"] += 1
            summary_rows.append({"category": "False Positive", "image": img_p.name, "path": str(save_p)})

        if has_fn and saved_counts["fn"] < max_samples_per_category:
            rendered = draw_annotated(orig_img, gt_boxes, preds)
            save_p = fn_dir / f"fn_{saved_counts['fn']+1:02d}_{img_p.name}"
            cv2.imwrite(str(save_p), rendered)
            saved_counts["fn"] += 1
            summary_rows.append({"category": "False Negative", "image": img_p.name, "path": str(save_p)})

        if has_low and saved_counts["low_conf"] < max_samples_per_category:
            rendered = draw_annotated(orig_img, gt_boxes, preds)
            save_p = low_conf_dir / f"low_{saved_counts['low_conf']+1:02d}_{img_p.name}"
            cv2.imwrite(str(save_p), rendered)
            saved_counts["low_conf"] += 1
            summary_rows.append({"category": "Low Confidence", "image": img_p.name, "path": str(save_p)})

    csv_out = output_dir / "error_summary.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "image", "path"])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS VISUALIZATIONS GENERATED")
    print(f"True Positives Saved:     {saved_counts['tp']}")
    print(f"False Positives Saved:    {saved_counts['fp']}")
    print(f"False Negatives Saved:    {saved_counts['fn']}")
    print(f"Low Confidence Saved:     {saved_counts['low_conf']}")
    print(f"Index Summary:            {csv_out}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Error Analysis.")
    parser.add_argument("--weights", type=Path, required=True, help="Model weights path")
    parser.add_argument("--data", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/datasets/SH17/data.yaml"),
                        help="Dataset data.yaml")
    parser.add_argument("--output", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/results/evaluation/errors"),
                        help="Output directory")
    parser.add_argument("--split", type=str, default="val", help="Evaluation split")
    args = parser.parse_args()

    run_error_analysis(
        weights_path=args.weights,
        data_yaml=args.data,
        output_dir=args.output,
        split=args.split
    )
