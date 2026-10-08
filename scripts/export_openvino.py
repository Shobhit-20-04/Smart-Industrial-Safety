"""
Intel OpenVINO Export & Optimization Script.
Exports trained PyTorch YOLO models to:
1. OpenVINO FP32
2. OpenVINO FP16 (half-precision)
3. OpenVINO INT8 (Post-Training Quantization with representative SH17 calibration)

Outputs stored under models/ or user-specified directory.
"""

import argparse
import time
from pathlib import Path
import yaml

def export_to_openvino(
    weights_path: Path,
    precision: str,
    imgsz: int = 640,
    data_yaml: Path = None,
    fraction: float = 0.10,
    output_dir: Path = None
):
    from ultralytics import YOLO

    print("=" * 70)
    print(f"EXPORTING MODEL TO OPENVINO: {precision.upper()}")
    print("=" * 70)
    print(f"Source Weights:       {weights_path}")
    print(f"Target Precision:     {precision.upper()}")
    print(f"Image Resolution:     {imgsz}x{imgsz}")

    model = YOLO(str(weights_path))
    start_time = time.time()

    half = False
    int8 = False

    if precision.lower() == "fp16":
        half = True
    elif precision.lower() == "int8":
        int8 = True
        if not data_yaml or not data_yaml.exists():
            raise ValueError("INT8 quantization requires representative dataset via --data YAML")
        print(f"Representative Data:  {data_yaml}")
        print(f"Calibration Fraction: {fraction * 100:.1f}%")

    export_args = {
        "format": "openvino",
        "imgsz": imgsz,
        "half": half,
        "int8": int8
    }

    if int8:
        export_args["data"] = str(data_yaml)
        export_args["fraction"] = fraction

    export_path = model.export(**export_args)
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"OPENVINO {precision.upper()} EXPORT COMPLETED")
    print(f"Export Output Path: {export_path}")
    print(f"Export Duration:    {elapsed:.2f} seconds")
    print("=" * 70)

    return export_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to OpenVINO format.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt weights")
    parser.add_argument("--precision", type=str, required=True, choices=["fp32", "fp16", "int8"],
                        help="Target precision: fp32, fp16, int8")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--data", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/datasets/SH17/data.yaml"),
                        help="Path to data.yaml (required for INT8 calibration)")
    parser.add_argument("--fraction", type=float, default=0.10, help="Calibration dataset fraction for INT8 (e.g. 0.10 or 0.20)")
    args = parser.parse_args()

    export_to_openvino(
        weights_path=args.weights,
        precision=args.precision,
        imgsz=args.imgsz,
        data_yaml=args.data,
        fraction=args.fraction
    )
