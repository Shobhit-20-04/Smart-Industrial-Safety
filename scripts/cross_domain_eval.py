"""
Cross-Domain Evaluation Script.
Evaluates an SH17-trained YOLO model on CHV and CHVG benchmarks without retraining.
Maps distinct class taxonomies into unified semantic PPE concepts (person, vest, helmet, head, glasses)
and computes research-grade precision, recall, and mAP metrics.
Saves results under results/cross_dataset/.
"""

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
import yaml
import numpy as np

def compute_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea > 0 else 0.0

def evaluate_cross_domain(
    weights_path: Path,
    target_dataset: str,
    target_images_dir: Path,
    target_labels_dir: Path,
    mapping_yaml: Path,
    output_dir: Path,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50
):
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(mapping_yaml, "r", encoding="utf-8") as f:
        map_cfg = yaml.safe_load(f)

    # Determine mapping rule
    if target_dataset.upper() == "CHV":
        pred_map = map_cfg.get("sh17_to_chv_mapping", {})
        gt_map = map_cfg.get("chv_to_unified", {})
        gt_classes = map_cfg.get("chv_classes", {})
    elif target_dataset.upper() == "CHVG":
        pred_map = map_cfg.get("sh17_to_chvg_mapping", {})
        gt_map = map_cfg.get("chvg_to_unified", {})
        gt_classes = map_cfg.get("chvg_classes", {})
    else:
        raise ValueError(f"Unsupported target dataset: {target_dataset}")

    sh17_classes = map_cfg.get("sh17_classes", {})
    common_concepts = sorted(list(set(pred_map.values())))
    print("=" * 70)
    print(f"CROSS-DOMAIN EVALUATION: SH17 Model -> {target_dataset.upper()}")
    print(f"Common Concepts: {common_concepts}")
    print("=" * 70)

    model = YOLO(str(weights_path))

    # Collect matched image-label pairs
    img_files = list(target_images_dir.glob("*.[jJ][pP][gG]")) + list(target_images_dir.glob("*.[pP][nN][gG]"))

    tp_counts = defaultdict(int)
    fp_counts = defaultdict(int)
    fn_counts = defaultdict(int)
    gt_counts = defaultdict(int)

    for img_p in img_files:
        lbl_p = target_labels_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        # Load GT boxes
        gt_boxes = []
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    xc, yc, w, h = [float(x) for x in parts[1:]]
                    cname = gt_classes.get(cid, str(cid))
                    unified_name = gt_map.get(cname, None)
                    if unified_name in common_concepts:
                        x1 = xc - w / 2.0
                        y1 = yc - h / 2.0
                        x2 = xc + w / 2.0
                        y2 = yc + h / 2.0
                        gt_boxes.append({"concept": unified_name, "box": [x1, y1, x2, y2], "matched": False})
                        gt_counts[unified_name] += 1

        # Run prediction
        res = model.predict(img_p, conf=conf_thresh, verbose=False)[0]
        pred_boxes = []
        for box in res.boxes:
            cid = int(box.cls.item())
            cname = sh17_classes.get(cid, str(cid))
            unified_name = pred_map.get(cname, None)
            if unified_name in common_concepts:
                # Bbox xywhn
                xywhn = box.xywhn[0].cpu().numpy()
                xc, yc, w, h = xywhn
                x1 = xc - w / 2.0
                y1 = yc - h / 2.0
                x2 = xc + w / 2.0
                y2 = yc + h / 2.0
                conf = float(box.conf.item())
                pred_boxes.append({"concept": unified_name, "box": [x1, y1, x2, y2], "conf": conf})

        # Match predictions to GTs
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
                tp_counts[p_concept] += 1
            else:
                fp_counts[p_concept] += 1

        for g in gt_boxes:
            if not g["matched"]:
                fn_counts[g["concept"]] += 1

    # Compute metrics per concept
    results_rows = []
    for concept in common_concepts:
        tp = tp_counts[concept]
        fp = fp_counts[concept]
        fn = fn_counts[concept]
        gt = gt_counts[concept]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        results_rows.append({
            "target_dataset": target_dataset.upper(),
            "concept": concept,
            "ground_truth_count": gt,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4)
        })

    # Save to CSV
    csv_file = output_dir / f"SH17_to_{target_dataset.upper()}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        writer.writeheader()
        for r in results_rows:
            writer.writerow(r)

    print(f"\n[cross_domain_eval] Saved cross-domain results to: {csv_file}")
    for r in results_rows:
        print(f"  {r['concept']:<15} | Prec: {r['precision']:.4f} | Rec: {r['recall']:.4f} | F1: {r['f1_score']:.4f} (GT: {r['ground_truth_count']:,})")
    return results_rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-domain YOLO evaluation.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained SH17 model weights")
    parser.add_argument("--target", type=str, required=True, choices=["CHV", "CHVG"], help="Target benchmark")
    parser.add_argument("--images", type=Path, required=True, help="Target images directory")
    parser.add_argument("--labels", type=Path, required=True, help="Target labels directory")
    parser.add_argument("--mapping", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/configs/class_mapping.yaml"),
                        help="Path to class mapping YAML")
    parser.add_argument("--output", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/results/cross_dataset"),
                        help="Output directory")
    args = parser.parse_args()

    evaluate_cross_domain(
        weights_path=args.weights,
        target_dataset=args.target,
        target_images_dir=args.images,
        target_labels_dir=args.labels,
        mapping_yaml=args.mapping,
        output_dir=args.output
    )
