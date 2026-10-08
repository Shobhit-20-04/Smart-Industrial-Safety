"""
Research-Grade Model Evaluation Script.
Evaluates trained YOLOv8 models on validation or test sets.
Computes overall metrics (Precision, Recall, mAP50, mAP50-95) and class-wise metrics.
Saves results in CSV and JSON formats under results/evaluation/.
"""

import json
import csv
import argparse
from pathlib import Path
import yaml

def evaluate_yolo(
    weights_path: Path,
    data_yaml: Path,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    output_dir: Path = None,
    name: str = None
):
    import torch
    from ultralytics import YOLO

    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    if output_dir is None:
        output_dir = Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/results/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_name = name or f"{weights_path.stem}_{split}"
    print("=" * 70)
    print(f"EVALUATING MODEL: {weights_path.name}")
    print(f"Dataset Config: {data_yaml}")
    print(f"Split:          {split}")
    print(f"Device:         {device}")
    print("=" * 70)

    model = YOLO(str(weights_path))
    val_results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=True,
        save_json=True
    )

    # Read class names
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    class_names = [cfg["names"][i] for i in sorted(cfg["names"].keys())]

    # Overall metrics
    box_metrics = val_results.box
    overall = {
        "model": weights_path.name,
        "dataset": data_yaml.stem,
        "split": split,
        "precision": round(float(box_metrics.p.mean()), 4),
        "recall": round(float(box_metrics.r.mean()), 4),
        "map50": round(float(box_metrics.map50), 4),
        "map50_95": round(float(box_metrics.map), 4),
        "speed_preprocess_ms": round(float(val_results.speed.get("preprocess", 0.0)), 2),
        "speed_inference_ms": round(float(val_results.speed.get("inference", 0.0)), 2),
        "speed_postprocess_ms": round(float(val_results.speed.get("postprocess", 0.0)), 2)
    }

    # Class-wise metrics
    class_wise = []
    for cid, cname in enumerate(class_names):
        cp = float(box_metrics.p[cid]) if cid < len(box_metrics.p) else 0.0
        cr = float(box_metrics.r[cid]) if cid < len(box_metrics.r) else 0.0
        cmap50 = float(box_metrics.ap50[cid]) if cid < len(box_metrics.ap50) else 0.0
        cmap = float(box_metrics.ap[cid]) if cid < len(box_metrics.ap) else 0.0

        class_wise.append({
            "class_id": cid,
            "class_name": cname,
            "precision": round(cp, 4),
            "recall": round(cr, 4),
            "ap50": round(cmap50, 4),
            "ap50_95": round(cmap, 4)
        })

    eval_data = {
        "overall": overall,
        "class_wise": class_wise
    }

    # Export JSON
    json_path = output_dir / f"{exp_name}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)

    # Export overall CSV
    overall_csv = output_dir / "evaluation_summary.csv"
    file_exists = overall_csv.exists()
    with open(overall_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(overall.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(overall)

    # Export class-wise CSV
    class_csv = output_dir / f"{exp_name}_class_wise_metrics.csv"
    with open(class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "precision", "recall", "ap50", "ap50_95"])
        writer.writeheader()
        for row in class_wise:
            writer.writerow(row)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    for k, v in overall.items():
        print(f"  {k:<22}: {v}")
    print(f"Results saved to: {json_path}")
    print(f"Class-wise metrics: {class_csv}")
    print("=" * 70)

    return eval_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model .pt file")
    parser.add_argument("--data", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/datasets/SH17/data.yaml"),
                        help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate on (val, test)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device (0, cpu, auto)")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    args = parser.parse_args()

    evaluate_yolo(
        weights_path=args.weights,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        output_dir=args.output,
        name=args.name
    )
