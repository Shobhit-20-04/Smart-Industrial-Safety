"""
Environment Verification Script.
Checks and prints:
- Python version
- PyTorch version
- CUDA availability
- GPU name (if available)
- Ultralytics version
- OpenVINO version
- OpenCV version
Saves report to results/environment_report.txt.
"""

import sys
import platform
import psutil
from pathlib import Path

def check_env(output_file: Path = None):
    report = []
    report.append("=" * 60)
    report.append("SYSTEM & ML ENVIRONMENT VERIFICATION")
    report.append("=" * 60)

    # OS & Python
    py_ver = sys.version.replace("\n", " ")
    os_name = f"{platform.system()} {platform.release()} ({platform.version()})"
    cpu_info = platform.processor() or "Unknown CPU"
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    report.append(f"Operating System: {os_name}")
    report.append(f"CPU: {cpu_info}")
    report.append(f"RAM: {ram_gb:.2f} GB")
    report.append(f"Python version: {py_ver}")

    # PyTorch & CUDA
    try:
        import torch
        report.append(f"PyTorch version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        report.append(f"CUDA availability: {cuda_avail}")
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            report.append(f"GPU name: {gpu_name}")
            report.append(f"GPU VRAM: {vram_gb:.2f} GB")
        else:
            report.append("GPU name: None (CPU mode only)")
    except ImportError:
        report.append("PyTorch version: NOT INSTALLED")
        report.append("CUDA availability: False (PyTorch missing)")
        report.append("GPU name: N/A")

    # Ultralytics
    try:
        import ultralytics
        report.append(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        report.append("Ultralytics version: NOT INSTALLED")

    # OpenVINO
    try:
        import openvino as ov
        report.append(f"OpenVINO version: {ov.__version__}")
    except ImportError:
        report.append("OpenVINO version: NOT INSTALLED")

    # OpenCV
    try:
        import cv2
        report.append(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        report.append("OpenCV version: NOT INSTALLED")

    report.append("=" * 60)
    text = "\n".join(report)
    print(text)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n[check_environment] Saved environment report to: {output_file}")

if __name__ == "__main__":
    out_p = Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/results/environment_report.txt")
    check_env(out_p)
