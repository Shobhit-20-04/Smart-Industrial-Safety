"""
Research-Grade YOLO Training Script for Ultralytics YOLOv8.
Supports YOLOv8n and YOLOv8s training on SH17 dataset.
Records all metrics, training configurations, and summaries into results/training/experiment_summary.csv.
"""

import os
import sys
import time
import argparse
import csv
from pathlib import Path
import yaml

def train_yolo(
    model_name: str,
    data_yaml: Path,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    project: Path,
    name: str,
    workers: int = 4,
    patience: int = 20,
    seed: int = 42
):
    import torch
    from ultralytics import YOLO

    start_time = time.time()
    print("=" * 70)
    print("STARTING YOLOv8 TRAINING EXPERIMENT")
    print("=" * 70)
    print(f"Model Architecture:    {model_name}")
    print(f"Dataset Configuration: {data_yaml}")
    print(f"Epochs:                {epochs}")
    print(f"Batch Size:            {batch}")
    print(f"Image Resolution:      {imgsz}")
    print(f"Hardware Device:       {device}")
    print(f"Workers:               {workers}")
    print(f"Patience:              {patience}")
    print(f"Output Project Dir:    {project}")
    print(f"Experiment Run Name:   {name}")
    print("=" * 70)

    # Check CUDA device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device != 'cpu' and torch.cuda.is_available() else 'CPU'})")

    project = project.resolve()
    project.mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(model_name)

    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(project),
        name=name,
        workers=workers,
        patience=patience,
        seed=seed,
        plots=True,
        save=True,
        val=True
    )

    elapsed_sec = time.time() - start_time
    elapsed_min = elapsed_sec / 60.0

    # Locate output directory
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else project / name
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    # Extract metrics from validation or results dict
    metrics_dict = {}
    if hasattr(results, "results_dict"):
        metrics_dict = results.results_dict
    
    # Try reading from results.csv for exact last/best epoch values
    res_csv = save_dir / "results.csv"
    p = metrics_dict.get("metrics/precision(B)", 0.0)
    r = metrics_dict.get("metrics/recall(B)", 0.0)
    map50 = metrics_dict.get("metrics/mAP50(B)", 0.0)
    map50_95 = metrics_dict.get("metrics/mAP50-95(B)", 0.0)
    best_epoch = epochs

    if res_csv.exists():
        try:
            with open(res_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    # Clean strip keys
                    cleaned_row = {k.strip(): v.strip() for k, v in last_row.items()}
                    p = float(cleaned_row.get("metrics/precision(B)", p))
                    r = float(cleaned_row.get("metrics/recall(B)", r))
                    map50 = float(cleaned_row.get("metrics/mAP50(B)", map50))
                    map50_95 = float(cleaned_row.get("metrics/mAP50-95(B)", map50_95))
                    best_epoch = int(cleaned_row.get("epoch", epochs))
        except Exception as e:
            print(f"[train] Note: could not parse results.csv: {e}")

    # Model file size
    model_size_mb = 0.0
    if best_pt.exists():
        model_size_mb = round(best_pt.stat().st_size / (1024 * 1024), 2)
    elif last_pt.exists():
        model_size_mb = round(last_pt.stat().st_size / (1024 * 1024), 2)

    # Number of parameters
    num_params = sum(p_param.numel() for p_param in model.model.parameters()) if hasattr(model, "model") else 0

    print("\n" + "=" * 70)
    print("TRAINING EXPERIMENT FINISHED")
    print("=" * 70)
    print(f"Elapsed Time:       {elapsed_min:.2f} minutes ({elapsed_sec:.1f} s)")
    print(f"Best Weights:       {best_pt if best_pt.exists() else 'N/A'}")
    print(f"Last Weights:       {last_pt if last_pt.exists() else 'N/A'}")
    print(f"Model Size:         {model_size_mb} MB")
    print(f"Parameters:         {num_params:,}")
    print(f"Precision:          {p:.4f}")
    print(f"Recall:             {r:.4f}")
    print(f"mAP@50:             {map50:.4f}")
    print(f"mAP@50-95:          {map50_95:.4f}")
    print(f"Best Epoch:         {best_epoch}")
    print("=" * 70)

    # Save to results/training/experiment_summary.csv
    exp_summary_csv = project.parent.parent / "results" / "training" / "experiment_summary.csv"
    exp_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = exp_summary_csv.exists()

    fieldnames = [
        "experiment",
        "model",
        "dataset",
        "epochs",
        "imgsz",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "best_epoch",
        "model_size_mb",
        "parameters",
        "elapsed_minutes"
    ]

    with open(exp_summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "experiment": name,
            "model": model_name,
            "dataset": data_yaml.stem,
            "epochs": epochs,
            "imgsz": imgsz,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "best_epoch": best_epoch,
            "model_size_mb": model_size_mb,
            "parameters": num_params,
            "elapsed_minutes": round(elapsed_min, 2)
        })

    print(f"Summary appended to: {exp_summary_csv}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Industrial Safety PPE dataset.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights or architecture")
    parser.add_argument("--data", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/datasets/SH17/data.yaml"),
                        help="Path to dataset data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (e.g. 16 or 32)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument("--device", type=str, default="auto", help="Device (0, cpu, or auto)")
    parser.add_argument("--project", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/runs/training"),
                        help="Output project directory")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    train_yolo(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed
    )
