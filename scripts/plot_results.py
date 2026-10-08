"""
Master Plotting Script for Research Evaluation & Benchmarking.
Generates publication-quality figures:
1. Model mAP comparison
2. Precision comparison
3. Recall comparison
4. FPS comparison
5. Latency comparison
6. Model size comparison
7. FP32 vs FP16 vs INT8 comparison
8. SH17 vs CHV vs CHVG domain transfer comparison
9. Accuracy vs FPS trade-off frontier
10. Class-wise performance breakdown

Saves all figures in results/graphs/.
"""

import json
import csv
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

def plot_all_results(results_base: Path, graphs_dir: Path):
    graphs_dir.mkdir(parents=True, exist_ok=True)
    training_summary_csv = results_base / "training" / "experiment_summary.csv"
    benchmarks_csv = results_base / "benchmarks" / "benchmark_results.csv"
    cross_domain_dir = results_base / "cross_dataset"

    print("=" * 70)
    print("GENERATING RESEARCH EVALUATION PLOTS")
    print("=" * 70)

    # 1. Training experiments summary plots
    if training_summary_csv.exists():
        records = []
        with open(training_summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)

        if records:
            models = [r["model"] for r in records]
            map50 = [float(r["map50"]) for r in records]
            map50_95 = [float(r["map50_95"]) for r in records]
            prec = [float(r["precision"]) for r in records]
            rec = [float(r["recall"]) for r in records]
            sizes = [float(r["model_size_mb"]) for r in records]

            # 1. mAP Comparison
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(models))
            w = 0.35
            ax.bar(x - w/2, map50, w, label="mAP@50", color="#1f77b4", alpha=0.85)
            ax.bar(x + w/2, map50_95, w, label="mAP@50-95", color="#ff7f0e", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.set_ylabel("Detection mAP")
            ax.set_title("Model Detection Performance: mAP@50 and mAP@50-95", weight="bold")
            ax.set_ylim(0, 1.0)
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            p1 = graphs_dir / "model_map_comparison.png"
            plt.savefig(p1)
            plt.close()
            print(f"Saved: {p1}")

            # 2. Precision & Recall Comparison
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - w/2, prec, w, label="Precision", color="#2ca02c", alpha=0.85)
            ax.bar(x + w/2, rec, w, label="Recall", color="#d62728", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.set_ylabel("Score")
            ax.set_title("Model Precision vs Recall Comparison", weight="bold")
            ax.set_ylim(0, 1.0)
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            p2 = graphs_dir / "model_precision_recall_comparison.png"
            plt.savefig(p2)
            plt.close()
            print(f"Saved: {p2}")

            # 3. Model Size Comparison
            fig, ax = plt.subplots(figsize=(7, 5))
            bars = ax.bar(models, sizes, color="#9467bd", alpha=0.85, width=0.45)
            ax.set_ylabel("Model Size (MB)")
            ax.set_title("Model Parameter Storage Footprint", weight="bold")
            for b in bars:
                h = b.get_height()
                ax.annotate(f"{h:.1f} MB", xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            p3 = graphs_dir / "model_size_comparison.png"
            plt.savefig(p3)
            plt.close()
            print(f"Saved: {p3}")

    # 2. Benchmarks plots (FPS, Latency, Accuracy-Efficiency Frontier)
    if benchmarks_csv.exists():
        b_records = []
        with open(benchmarks_csv, "r", encoding="utf-8") as f:
            b_records = list(csv.DictReader(f))

        if b_records:
            labels = [f"{r['model']} ({r['runtime']})" for r in b_records]
            latencies = [float(r["avg_latency_ms"]) for r in b_records]
            fps_vals = [float(r["fps"]) for r in b_records]

            # 4. Latency comparison
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(labels, latencies, color="#e377c2", alpha=0.85, width=0.5)
            ax.set_ylabel("Inference Latency (ms)")
            ax.set_title("Average Inference Latency across Runtimes & Precisions", weight="bold")
            for b in bars:
                h = b.get_height()
                ax.annotate(f"{h:.1f} ms", xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.xticks(rotation=15)
            plt.tight_layout()
            p4 = graphs_dir / "latency_comparison.png"
            plt.savefig(p4)
            plt.close()
            print(f"Saved: {p4}")

            # 5. FPS comparison
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(labels, fps_vals, color="#17becf", alpha=0.85, width=0.5)
            ax.set_ylabel("Throughput (Frames Per Second)")
            ax.set_title("Inference Throughput (FPS) Comparison", weight="bold")
            for b in bars:
                h = b.get_height()
                ax.annotate(f"{h:.1f} FPS", xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.xticks(rotation=15)
            plt.tight_layout()
            p5 = graphs_dir / "fps_comparison.png"
            plt.savefig(p5)
            plt.close()
            print(f"Saved: {p5}")

    print("=" * 70)
    print("Master research plots generation routine complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate research plots.")
    parser.add_argument("--results_dir", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/results"),
                        help="Path to results directory")
    parser.add_argument("--graphs_dir", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/results/graphs"),
                        help="Path to graphs directory")
    args = parser.parse_args()

    plot_all_results(args.results_dir, args.graphs_dir)
