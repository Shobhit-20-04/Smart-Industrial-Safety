"""
Compare YOLOv8n and YOLOv8s Models.
Computes multi-dimensional trade-offs across:
- Precision, Recall, mAP50, mAP50-95
- Parameter count, Model size (MB)
- Inference latency (ms), Throughput (FPS)
Generates:
- results/evaluation/model_comparison.csv
- results/graphs/model_comparison.png
"""

import json
import csv
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def compare_models(
    model_eval_files: list,
    output_dir: Path,
    graphs_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for f in model_eval_files:
        p = Path(f)
        if p.exists():
            with open(p, "r", encoding="utf-8") as json_f:
                d = json.load(json_f)
                records.append(d.get("overall", d))

    if not records:
        print("[compare_models] No evaluation records found.")
        return

    csv_path = output_dir / "model_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"[compare_models] Model comparison CSV saved to: {csv_path}")

    # Plot multi-metric comparison if records contain >= 2 models
    if len(records) >= 2:
        model_names = [r["model"] for r in records]
        map50_vals = [r.get("map50", 0.0) for r in records]
        map50_95_vals = [r.get("map50_95", 0.0) for r in records]
        latency_vals = [r.get("speed_inference_ms", 0.0) for r in records]
        fps_vals = [1000.0 / max(r.get("speed_inference_ms", 1.0), 0.1) for r in records]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy plot
        x = np.arange(len(model_names))
        w = 0.35
        ax1.bar(x - w/2, map50_vals, w, label="mAP@50", color="#1f77b4", alpha=0.85)
        ax1.bar(x + w/2, map50_95_vals, w, label="mAP@50-95", color="#ff7f0e", alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names)
        ax1.set_ylabel("Detection Accuracy")
        ax1.set_title("Detection Accuracy Comparison", weight="bold")
        ax1.set_ylim(0, 1.0)
        ax1.legend()
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # Efficiency plot
        ax2.bar(x - w/2, latency_vals, w, label="Inference Latency (ms)", color="#2ca02c", alpha=0.85)
        ax2_fps = ax2.twinx()
        ax2_fps.plot(x, fps_vals, color="#d62728", marker="o", linewidth=2.5, label="Throughput (FPS)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names)
        ax2.set_ylabel("Latency (ms)")
        ax2_fps.set_ylabel("Throughput (FPS)")
        ax2.set_title("Computational Efficiency Trade-off", weight="bold")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_path = graphs_dir / "model_comparison.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[compare_models] Model comparison plot saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare YOLO models.")
    parser.add_argument("--files", nargs="+", required=True, help="List of evaluation JSON metric files")
    parser.add_argument("--output", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/results/evaluation"),
                        help="Output directory for CSV")
    parser.add_argument("--graphs", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/results/graphs"),
                        help="Output directory for graphs")
    args = parser.parse_args()

    compare_models(args.files, args.output, args.graphs)
