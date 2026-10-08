"""
Inference Latency & Throughput Benchmarking Script.
Supports PyTorch (.pt) and OpenVINO IR (.xml) runtime models.
Measures:
- Warm-up iterations
- Multi-run Average latency, Median latency, 95th-percentile latency
- Frame-rate Throughput (FPS)
- Model size (MB)
- RAM / VRAM memory usage
Saves results to results/benchmarks/benchmark_results.csv.
"""

import time
import argparse
import csv
from pathlib import Path
import numpy as np
import psutil

def benchmark_model(
    model_path: Path,
    imgsz: int = 640,
    device: str = "auto",
    num_runs: int = 100,
    warmup: int = 20,
    output_dir: Path = None
):
    import torch

    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    if output_dir is None:
        output_dir = Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/results/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"BENCHMARKING MODEL: {model_path.name}")
    print(f"Device:             {device}")
    print(f"Image Resolution:   {imgsz}x{imgsz}")
    print(f"Warm-up Runs:       {warmup}")
    print(f"Measured Runs:      {num_runs}")
    print("=" * 70)

    # Determine runtime type
    is_openvino = model_path.suffix.lower() == ".xml" or model_path.is_dir()
    runtime = "OpenVINO" if is_openvino else "PyTorch"

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    # Create dummy batch of images
    dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warm-up phase
    print(f"[benchmark] Executing {warmup} warm-up iterations...")
    for _ in range(warmup):
        _ = model.predict(dummy_input, imgsz=imgsz, device=device, verbose=False)

    # Synchronize CUDA if applicable
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measured runs
    latencies = []
    print(f"[benchmark] Measuring {num_runs} timed iterations...")
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model.predict(dummy_input, imgsz=imgsz, device=device, verbose=False)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # in ms

    avg_lat = float(np.mean(latencies))
    med_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg_lat

    # Model size
    if model_path.is_file():
        model_size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)
    elif model_path.is_dir():
        model_size_mb = round(sum(f.stat().st_size for f in model_path.glob("**/*") if f.is_file()) / (1024 * 1024), 2)
    else:
        model_size_mb = 0.0

    # Memory usage
    ram_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)

    result_row = {
        "model": model_path.name,
        "runtime": runtime,
        "device": device,
        "imgsz": imgsz,
        "measured_runs": num_runs,
        "avg_latency_ms": round(avg_lat, 2),
        "median_latency_ms": round(med_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "fps": round(fps, 2),
        "model_size_mb": model_size_mb,
        "ram_usage_mb": ram_mb
    }

    print("\n" + "=" * 70)
    print("BENCHMARK METRICS SUMMARY")
    print("=" * 70)
    for k, v in result_row.items():
        print(f"  {k:<20}: {v}")
    print("=" * 70)

    # Save to CSV
    csv_path = output_dir / "benchmark_results.csv"
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_row)
    print(f"[benchmark] Results appended to: {csv_path}")

    return result_row

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark YOLO model inference latency.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model file or directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="auto", help="Device (0, cpu, auto)")
    parser.add_argument("--runs", type=int, default=100, help="Number of measured runs")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup runs")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    benchmark_model(
        model_path=args.model,
        imgsz=args.imgsz,
        device=args.device,
        num_runs=args.runs,
        warmup=args.warmup,
        output_dir=args.output
    )
