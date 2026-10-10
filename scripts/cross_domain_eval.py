"""
Research-Grade Cross-Domain Evaluation & Generalization Analysis Script.
Evaluates an SH17-trained YOLO model on CHV and CHVG benchmarks without retraining.
Uses explicit semantic class mappings for common PPE concepts.
Computes:
- Precision, Recall, F1-score
- COCO/VOC 101-point Interpolated AP@50 and mAP@50
- Inference Latency (ms) and FPS
- Class-wise performance breakdowns
- Visual error overlays (FP, FN, TP) for empirical degradation analysis
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
import yaml
import cv2
import numpy as np

def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Format: [x1, y1, x2, y2] in normalized coordinates [0, 1].
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0.0 else 0.0

def compute_ap50(tp_list, conf_list, total_gts):
    """
    Computes 101-point interpolated Average Precision at IoU=0.50 (AP@50).
    tp_list: list of 1s (TP) and 0s (FP) sorted by descending confidence.
    conf_list: list of prediction confidence scores.
    total_gts: total number of ground truth instances for this class.
    """
    if total_gts == 0 or len(tp_list) == 0:
        return 0.0

    # Sort by descending confidence
    indices = np.argsort(-np.array(conf_list))
    tp_sorted = np.array(tp_list)[indices]

    tp_cumsum = np.cumsum(tp_sorted)
    fp_cumsum = np.cumsum(1 - tp_sorted)

    recalls = tp_cumsum / total_gts
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # 101-point interpolation (COCO standard)
    rec_thresholds = np.linspace(0.0, 1.0, 101)
    prec_interpolated = []
    for r in rec_thresholds:
        p_sub = precisions[recalls >= r]
        p_interp = np.max(p_sub) if len(p_sub) > 0 else 0.0
        prec_interpolated.append(p_interp)

    return float(np.mean(prec_interpolated))

def evaluate_cross_domain(
    weights_path: Path,
    target_dataset: str,
    target_images_dir: Path,
    target_labels_dir: Path,
    mapping_yaml: Path,
    output_dir: Path,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
    save_visual_errors: bool = True,
    max_error_images: int = 25
):
    from ultralytics import YOLO

    target_dataset = target_dataset.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    error_dir = output_dir / f"errors_{target_dataset.lower()}"
    if save_visual_errors:
        error_dir.mkdir(parents=True, exist_ok=True)

    with open(mapping_yaml, "r", encoding="utf-8") as f:
        map_cfg = yaml.safe_load(f)

    # Load mappings
    sh17_classes = map_cfg.get("sh17_classes", {})
    mapping_key = f"sh17_to_{target_dataset.lower()}"
    if mapping_key not in map_cfg:
        raise ValueError(f"Mapping '{mapping_key}' not found in {mapping_yaml}")

    pred_map = map_cfg[mapping_key]["predictions"]
    gt_map = map_cfg[mapping_key]["ground_truth"]
    target_class_dict = map_cfg.get(f"{target_dataset.lower()}_classes", {})

    common_concepts = sorted(list(set(pred_map.values())))
    print("=" * 70)
    print(f"CROSS-DOMAIN EVALUATION: SH17 Model -> {target_dataset}")
    print(f"Weights Path:    {weights_path}")
    print(f"Target Images:   {target_images_dir}")
    print(f"Target Labels:   {target_labels_dir}")
    print(f"Common Concepts: {common_concepts}")
    print("=" * 70)

    model = YOLO(str(weights_path))

    # Collect matched image-label pairs
    img_files = sorted(list(target_images_dir.glob("*.[jJ][pP][gG]")) + list(target_images_dir.glob("*.[pP][nN][gG]")))
    print(f"[cross_domain] Found {len(img_files):,} candidate images in {target_images_dir}")

    all_preds_per_concept = defaultdict(list) # concept: [(tp_bool, conf)]
    gt_counts = defaultdict(int)
    tp_counts = defaultdict(int)
    fp_counts = defaultdict(int)
    fn_counts = defaultdict(int)

    latencies_ms = []
    saved_error_count = 0

    for idx, img_p in enumerate(img_files):
        lbl_p = target_labels_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        # 1. Parse Ground Truth annotations
        gt_boxes = []
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    xc, yc, w, h = [float(x) for x in parts[1:5]]
                    raw_cname = target_class_dict.get(cid, str(cid))
                    unified_concept = gt_map.get(raw_cname, None)
                    if unified_concept in common_concepts:
                        x1 = max(0.0, xc - w / 2.0)
                        y1 = max(0.0, yc - h / 2.0)
                        x2 = min(1.0, xc + w / 2.0)
                        y2 = min(1.0, yc + h / 2.0)
                        gt_boxes.append({
                            "concept": unified_concept,
                            "raw_class": raw_cname,
                            "box": [x1, y1, x2, y2],
                            "matched": False
                        })
                        gt_counts[unified_concept] += 1

        # 2. Run model prediction with latency profiling
        t0 = time.perf_counter()
        res = model.predict(img_p, imgsz=640, conf=conf_thresh, verbose=False)[0]
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        # 3. Parse predicted boxes and project into unified concept space
        pred_boxes = []
        for box in res.boxes:
            cid = int(box.cls.item())
            sh17_name = sh17_classes.get(cid, str(cid))
            unified_concept = pred_map.get(sh17_name, None)
            if unified_concept in common_concepts:
                xywhn = box.xywhn[0].cpu().numpy()
                xc, yc, w, h = xywhn
                x1 = max(0.0, xc - w / 2.0)
                y1 = max(0.0, yc - h / 2.0)
                x2 = min(1.0, xc + w / 2.0)
                y2 = min(1.0, yc + h / 2.0)
                conf = float(box.conf.item())
                pred_boxes.append({
                    "concept": unified_concept,
                    "sh17_class": sh17_name,
                    "box": [x1, y1, x2, y2],
                    "conf": conf,
                    "matched": False
                })

        # 4. Greedy bipartite matching (IoU >= 0.50)
        # Sort predictions by descending confidence
        pred_boxes.sort(key=lambda x: x["conf"], reverse=True)

        for p in pred_boxes:
            p_concept = p["concept"]
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, g in enumerate(gt_boxes):
                if g["concept"] == p_concept and not g["matched"]:
                    iou = compute_iou(p["box"], g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                gt_boxes[best_gt_idx]["matched"] = True
                p["matched"] = True
                tp_counts[p_concept] += 1
                all_preds_per_concept[p_concept].append((1, p["conf"]))
            else:
                fp_counts[p_concept] += 1
                all_preds_per_concept[p_concept].append((0, p["conf"]))

        for g in gt_boxes:
            if not g["matched"]:
                fn_counts[g["concept"]] += 1

        # 5. Visual Error Overlays for empirical degradation analysis
        has_error = any(not g["matched"] for g in gt_boxes) or any(not p["matched"] for p in pred_boxes)
        if save_visual_errors and has_error and saved_error_count < max_error_images:
            img_bgr = cv2.imread(str(img_p))
            if img_bgr is not None:
                h_px, w_px = img_bgr.shape[:2]
                # Draw Ground Truths (Green)
                for g in gt_boxes:
                    bx = [int(g["box"][0] * w_px), int(g["box"][1] * h_px), int(g["box"][2] * w_px), int(g["box"][3] * h_px)]
                    color = (0, 255, 0) if g["matched"] else (0, 0, 255) # Green if matched, Red if FN
                    cv2.rectangle(img_bgr, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
                    label = f"GT:{g['concept']}" if g['matched'] else f"FN:{g['concept']}"
                    cv2.putText(img_bgr, label, (bx[0], max(15, bx[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Draw Predictions (Blue for TP, Orange for FP)
                for p in pred_boxes:
                    bx = [int(p["box"][0] * w_px), int(p["box"][1] * h_px), int(p["box"][2] * w_px), int(p["box"][3] * h_px)]
                    color = (255, 200, 0) if p["matched"] else (0, 165, 255) # Blue/Cyan for TP, Orange for FP
                    cv2.rectangle(img_bgr, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
                    label = f"TP:{p['concept']} {p['conf']:.2f}" if p["matched"] else f"FP:{p['concept']} {p['conf']:.2f}"
                    cv2.putText(img_bgr, label, (bx[0], min(h_px - 5, bx[3] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                overlay_path = error_dir / f"err_{img_p.stem}.jpg"
                cv2.imwrite(str(overlay_path), img_bgr)
                saved_error_count += 1

    # Speed metrics
    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    # 6. Compute Class-Wise and Overall Metrics
    class_results = []
    ap_list = []
    prec_list = []
    rec_list = []
    f1_list = []

    total_tp = sum(tp_counts.values())
    total_fp = sum(fp_counts.values())
    total_fn = sum(fn_counts.values())
    total_gt = sum(gt_counts.values())

    for concept in common_concepts:
        tp = tp_counts[concept]
        fp = fp_counts[concept]
        fn = fn_counts[concept]
        gt = gt_counts[concept]

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        preds_info = all_preds_per_concept[concept]
        tps = [x[0] for x in preds_info]
        confs = [x[1] for x in preds_info]
        ap50 = compute_ap50(tps, confs, gt)

        class_results.append({
            "target_dataset": target_dataset,
            "concept": concept,
            "ground_truth_instances": gt,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "ap50": round(ap50, 4)
        })

        if gt > 0:
            prec_list.append(p)
            rec_list.append(r)
            f1_list.append(f1)
            ap_list.append(ap50)

    # Macro averages
    macro_p = float(np.mean(prec_list)) if prec_list else 0.0
    macro_r = float(np.mean(rec_list)) if rec_list else 0.0
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    map50 = float(np.mean(ap_list)) if ap_list else 0.0

    overall_summary = {
        "source_dataset": "SH17",
        "target_dataset": target_dataset,
        "evaluated_images": len(img_files),
        "total_ground_truth": total_gt,
        "common_concepts": ", ".join(common_concepts),
        "precision": round(macro_p, 4),
        "recall": round(macro_r, 4),
        "f1_score": round(macro_f1, 4),
        "map50": round(map50, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 2)
    }

    # Save detailed CSV
    csv_file = output_dir / f"SH17_to_{target_dataset}_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(class_results[0].keys()))
        writer.writeheader()
        for row in class_results:
            writer.writerow(row)

    print("\n" + "=" * 70)
    print(f"CROSS-DOMAIN RESULTS: SH17 -> {target_dataset}")
    print("=" * 70)
    for k, v in overall_summary.items():
        print(f"  {k:<24}: {v}")
    print("-" * 70)
    print(f"{'Concept':<15} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'AP@50':<7} | {'GT':<6}")
    print("-" * 70)
    for cr in class_results:
        print(f"{cr['concept']:<15} | {cr['precision']:<7.4f} | {cr['recall']:<7.4f} | {cr['f1_score']:<7.4f} | {cr['ap50']:<7.4f} | {cr['ground_truth_instances']:<6,}")
    print("=" * 70)
    print(f"Results saved to: {csv_file}")
    if save_visual_errors:
        print(f"Visual error overlays saved to: {error_dir} ({saved_error_count} images)")

    return overall_summary, class_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-domain YOLO evaluation.")
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8s_sh17_best.pt"),
                        help="Trained SH17 model weights")
    parser.add_argument("--target", type=str, required=True, choices=["CHV", "CHVG", "chv", "chvg"],
                        help="Target benchmark")
    parser.add_argument("--images", type=Path, required=True, help="Target images directory")
    parser.add_argument("--labels", type=Path, required=True, help="Target labels directory")
    parser.add_argument("--mapping", type=Path, default=Path("configs/class_mapping.yaml"),
                        help="Path to class mapping YAML")
    parser.add_argument("--output", type=Path, default=Path("results/cross_dataset"),
                        help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    parser.add_argument("--max-errors", type=int, default=25, help="Max visual error images to save")
    args = parser.parse_args()

    evaluate_cross_domain(
        weights_path=args.weights,
        target_dataset=args.target,
        target_images_dir=args.images,
        target_labels_dir=args.labels,
        mapping_yaml=args.mapping,
        output_dir=args.output,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        max_error_images=args.max_errors
    )
