"""
Empirical Response Time & Throughput Benchmark.
Measures latency (ms) and throughput (FPS) across architectures, runtimes,
hardware accelerators (RTX 3050 GPU vs Intel CPU), and input resolutions (640, 512, 480).
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def benchmark_config(model_path, device, imgsz, num_warmup=15, num_runs=50):
    model = YOLO(str(model_path))
    dummy_input = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    # Warm-up phase
    for _ in range(num_warmup):
        _ = model.predict(dummy_input, imgsz=imgsz, device=device, verbose=False)

    if device in ["0", "cuda"] and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measurement phase
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model.predict(dummy_input, imgsz=imgsz, device=device, verbose=False)
        if device in ["0", "cuda"] and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "fps": round(fps, 1)
    }

def main():
    print("=" * 75)
    print("EMPIRICAL LATENCY & THROUGHPUT PROFILING SUITE")
    print("=" * 75)

    experiments = [
        # 1. PyTorch GPU Experiments (NVIDIA RTX 3050)
        {
            "name": "YOLOv8n PyTorch (CUDA GPU)",
            "model_path": PROJECT_ROOT / "models" / "yolov8n_sh17_best.pt",
            "runtime": "PyTorch",
            "device": "0",
            "imgsz": 640
        },
        {
            "name": "YOLOv8n PyTorch (CUDA GPU - 480p)",
            "model_path": PROJECT_ROOT / "models" / "yolov8n_sh17_best.pt",
            "runtime": "PyTorch",
            "device": "0",
            "imgsz": 480
        },
        {
            "name": "YOLOv8s PyTorch (CUDA GPU)",
            "model_path": PROJECT_ROOT / "models" / "yolov8s_sh17_best.pt",
            "runtime": "PyTorch",
            "device": "0",
            "imgsz": 640
        },
        {
            "name": "YOLOv8s PyTorch (CUDA GPU - 480p)",
            "model_path": PROJECT_ROOT / "models" / "yolov8s_sh17_best.pt",
            "runtime": "PyTorch",
            "device": "0",
            "imgsz": 480
        },

        # 2. OpenVINO Intel CPU Experiments
        {
            "name": "YOLOv8n OpenVINO FP16 (Intel CPU)",
            "model_path": PROJECT_ROOT / "models" / "yolov8n_sh17_best_openvino_model",
            "runtime": "OpenVINO FP16",
            "device": "cpu",
            "imgsz": 640
        },
        {
            "name": "YOLOv8n OpenVINO FP16 (Intel CPU - 480p)",
            "model_path": PROJECT_ROOT / "models" / "yolov8n_sh17_best_openvino_model",
            "runtime": "OpenVINO FP16",
            "device": "cpu",
            "imgsz": 480
        },
        {
            "name": "YOLOv8s OpenVINO INT8 (Intel CPU - 640p)",
            "model_path": PROJECT_ROOT / "models" / "yolov8s_int8_openvino_model",
            "runtime": "OpenVINO INT8",
            "device": "cpu",
            "imgsz": 640
        },
        {
            "name": "YOLOv8s OpenVINO INT8 (Intel CPU - 512p)",
            "model_path": PROJECT_ROOT / "models" / "yolov8s_int8_openvino_model",
            "runtime": "OpenVINO INT8",
            "device": "cpu",
            "imgsz": 512
        },
        {
            "name": "YOLOv8s OpenVINO INT8 (Intel CPU - 480p)",
            "model_path": PROJECT_ROOT / "models" / "yolov8s_int8_openvino_model",
            "runtime": "OpenVINO INT8",
            "device": "cpu",
            "imgsz": 480
        },
        {
            "name": "YOLOv8s OpenVINO FP16 (Intel CPU - 640p)",
            "model_path": PROJECT_ROOT / "models" / "yolov8s_fp16_openvino_model",
            "runtime": "OpenVINO FP16",
            "device": "cpu",
            "imgsz": 640
        },

        # 3. CPU Baseline (Slow baseline)
        {
            "name": "YOLOv8s PyTorch CPU Baseline",
            "model_path": PROJECT_ROOT / "models" / "yolov8s_sh17_best.pt",
            "runtime": "PyTorch",
            "device": "cpu",
            "imgsz": 640
        }
    ]

    records = []
    for exp in experiments:
        print(f"Benchmarking: {exp['name']} @ {exp['imgsz']}x{exp['imgsz']} on {exp['device'].upper()}...")
        metrics = benchmark_config(exp["model_path"], exp["device"], exp["imgsz"])
        rec = {
            "Configuration": exp["name"],
            "Runtime": exp["runtime"],
            "Hardware": "NVIDIA RTX 3050 Laptop GPU" if exp["device"] == "0" else "Intel Core i5-13450HX CPU",
            "Resolution": f"{exp['imgsz']}x{exp['imgsz']}",
            "Mean Latency (ms)": metrics["mean_latency_ms"],
            "Median Latency (ms)": metrics["median_latency_ms"],
            "P95 Latency (ms)": metrics["p95_latency_ms"],
            "Throughput (FPS)": metrics["fps"]
        }
        records.append(rec)
        print(f"  -> Latency: {metrics['mean_latency_ms']} ms | Throughput: {metrics['fps']} FPS\n")

    df = pd.DataFrame(records)
    out_csv = PROJECT_ROOT / "results" / "benchmarks" / "response_time_optimization.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Benchmark results successfully saved to: {out_csv}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
