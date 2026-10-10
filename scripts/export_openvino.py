"""
Research-Grade Intel OpenVINO Export & Optimization Script.
Exports trained PyTorch YOLO models to:
1. OpenVINO FP32
2. OpenVINO FP16 (half-precision)
3. OpenVINO INT8 (Post-Training Quantization with representative SH17 calibration)

Saves models to dedicated directories and records calibration configuration.
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path
import yaml

def export_to_openvino(
    weights_path: Path,
    precision: str,
    imgsz: int = 640,
    data_yaml: Path = None,
    split: str = "train",
    fraction: float = 0.05,
    output_dir: Path = None
):
    from ultralytics import YOLO

    precision = precision.lower()
    if precision not in ["fp32", "fp16", "int8"]:
        raise ValueError(f"Unsupported precision: {precision}. Must be fp32, fp16, or int8.")

    if output_dir is None:
        output_dir = weights_path.parent / f"openvino_{precision}"
    output_dir = Path(output_dir).resolve()

    print("=" * 70)
    print(f"EXPORTING MODEL TO OPENVINO: {precision.upper()}")
    print("=" * 70)
    print(f"Source Weights:       {weights_path}")
    print(f"Target Precision:     {precision.upper()}")
    print(f"Image Resolution:     {imgsz}x{imgsz}")
    print(f"Target Output Dir:    {output_dir}")

    # Remove target directory if it already exists to avoid collisions
    if output_dir.exists():
        print(f"[export] Cleaning existing target directory: {output_dir}")
        shutil.rmtree(output_dir)

    model = YOLO(str(weights_path))
    start_time = time.time()

    export_args = {
        "format": "openvino",
        "imgsz": imgsz,
        "batch": 1
    }

    if precision == "fp16":
        export_args["half"] = True
    elif precision == "int8":
        export_args["quantize"] = 8

    if precision == "int8":
        if not data_yaml or not Path(data_yaml).exists():
            raise ValueError("INT8 quantization requires representative dataset via --data YAML")
        export_args["data"] = str(data_yaml)
        export_args["fraction"] = fraction
        export_args["split"] = split

        print(f"Representative Data:  {data_yaml}")
        print(f"Calibration Split:    {split}")
        print(f"Calibration Fraction: {fraction * 100:.1f}%")

    # Run Ultralytics export
    raw_exported_path = model.export(**export_args)
    raw_exported_path = Path(raw_exported_path).resolve()

    # Move/Rename exported model to standardized output_dir
    if raw_exported_path != output_dir:
        print(f"[export] Moving exported model from {raw_exported_path} to {output_dir}")
        shutil.move(str(raw_exported_path), str(output_dir))

    elapsed = time.time() - start_time

    # Calculate model size
    total_size_bytes = sum(f.stat().st_size for f in output_dir.glob("*") if f.is_file())
    model_size_mb = round(total_size_bytes / (1024 * 1024), 2)

    # Save calibration config for INT8 provenance
    if precision == "int8":
        calib_cfg = {
            "model_source": str(weights_path),
            "precision": "INT8",
            "optimization_method": "Post-Training Quantization (PTQ)",
            "framework": "Intel OpenVINO + NNCF",
            "dataset_config": str(data_yaml),
            "calibration_split": split,
            "calibration_fraction": fraction,
            "image_size": imgsz,
            "export_duration_seconds": round(elapsed, 2),
            "model_size_mb": model_size_mb,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        calib_file = output_dir / "calibration_config.yaml"
        with open(calib_file, "w", encoding="utf-8") as f:
            yaml.dump(calib_cfg, f, default_flow_style=False)
        print(f"[export] Saved calibration configuration to: {calib_file}")

    print("\n" + "=" * 70)
    print(f"OPENVINO {precision.upper()} EXPORT COMPLETED")
    print(f"Final Model Directory: {output_dir}")
    print(f"Model Size:            {model_size_mb} MB")
    print(f"Export Duration:       {elapsed:.2f} seconds")
    print("=" * 70)

    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to OpenVINO format.")
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8s_sh17_best.pt"),
                        help="Path to best.pt weights")
    parser.add_argument("--precision", type=str, required=True, choices=["fp32", "fp16", "int8"],
                        help="Target precision: fp32, fp16, int8")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--data", type=Path, default=Path("datasets/SH17/data.yaml"),
                        help="Path to data.yaml (required for INT8 calibration)")
    parser.add_argument("--split", type=str, default="train",
                        help="Calibration split for INT8 (default: train)")
    parser.add_argument("--fraction", type=float, default=0.05,
                        help="Calibration dataset fraction for INT8 (e.g. 0.05 = 5%%)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Custom output directory")
    args = parser.parse_args()

    export_to_openvino(
        weights_path=args.weights,
        precision=args.precision,
        imgsz=args.imgsz,
        data_yaml=args.data,
        split=args.split,
        fraction=args.fraction,
        output_dir=args.output
    )
