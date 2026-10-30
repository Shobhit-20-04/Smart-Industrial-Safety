# Smart Industrial Safety Monitoring System: Real-Time Edge PPE Compliance via OpenVINO INT8 Quantization and FPGA Implementation Roadmap

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![OpenVINO 2024+](https://img.shields.io/badge/OpenVINO-2024%2B-0072C6.svg)](https://docs.openvino.ai/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-green.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Executive Summary & Overview

In heavy manufacturing, chemical refining, and construction environments, consistent compliance with **Personal Protective Equipment (PPE)** standards (safety helmets, high-visibility vests, protective eyewear, and gloves) is essential to prevent severe and fatal workplace accidents. Traditional manual inspection relies on human safety auditors, which is labor-intensive, fatigue-prone, and cannot sustain continuous 24/7 coverage across multi-shift production lines.

This repository presents the **Smart Industrial Safety Monitoring System**, an edge-optimized computer vision system that:
1. **Detects Workers & PPE Artifacts**: Utilizes single-stage YOLOv8 architectures trained on the 17-class manufacturing benchmark **SH17** (8,099 images, 75,994 bounding boxes).
2. **Evaluates Compliance in Real-Time**: Implements an ultra-fast **centroid-in-bounding-box spatial containment** matching algorithm that associates detected helmets, safety vests, glasses, and gloves with individual worker boundaries.
3. **Breaks the Edge Memory Wall**: Employs **Intel OpenVINO Intermediate Representation (IR)** compilation and **Post-Training INT8 Quantization (NNCF calibration)**, delivering a **5.42–6.22× speedup** and **73.8% model compression** on commodity x86 CPUs with only a negligible **0.4% mAP loss**.
4. **Provides an Intel FPGA AI Suite Roadmap**: Formalizes an 8-stage hardware deployment methodology, `.arch` architecture parameters, offline `dla_compiler` graph mapping, Quartus Prime Pro bitstream compilation, and hardware roofline modeling targeting **Intel Arria 10 GX, Stratix 10 GX, and Agilex 7 FPGAs** in OpenVINO `HETERO:FPGA,CPU` mode.
5. **Evaluates Zero-Shot Generalization**: Benchmarks cross-domain transferability on two external construction datasets (**CHV** and **CHVG**) without retraining.

---

## 2. Key Highlights & Technical Contributions

- **YOLOv8s Selection for Critical Small Objects**: YOLOv8s achieves **0.6214 mAP@50** on SH17 (vs. 0.5473 for YOLOv8n, $+0.0741$ absolute gain), with major accuracy improvements on small PPE items: **$+35.51\%$** on face-guards, **$+18.59\%$** on earmuffs, **$+10.17\%$** on glasses, and **$+6.53\%$** on gloves.
- **Hardware-Aware Kernel Fusion**: OpenVINO graph compilation folds Batch Normalization and SiLU activation directly into Conv2D kernels, eliminating intermediate DRAM round-trips and vectorizing execution across 256-bit AVX2 SIMD registers.
- **Aggressive INT8 Quantization via NNCF**: Symmetric signed 8-bit integer quantization shrinks YOLOv8s from **42.79 MB to 11.23 MB**, elevating CPU throughput from **8.8–10.8 FPS up to 47.6–63.2 FPS** (mean $21.27\text{ ms}$, $47.02\text{ FPS}$) at $640\times640$ resolution.
- **Integrated GPU Acceleration**: Benchmarked on Intel Iris Xe graphics (Raptor Lake-S iGPU, 32 EUs) achieving $21.80\text{ FPS}$ in INT8 mode with shared DDR5 system memory via OpenCL Shared Virtual Memory (SVM).
- **Sub-Millisecond Compliance Logic**: The spatial containment engine runs in **$0.01\text{ ms}$** per frame ($<0.1\%$ of pipeline latency), enabling end-to-end sustained system throughput of **$30+\text{ FPS}$** ($33.38\text{ ms}$ total latency including decode, pre-processing, inference, NMS, compliance logic, and HUD rendering).
- **Production-Grade Delivery**: Fully featured threaded streaming monitor (`app/main.py`) with zero camera I/O blocking, and an interactive Streamlit analytics dashboard (`app/dashboard.py`).

---

## 3. End-to-End System Architecture

The following diagram illustrates the 14-stage execution flow of the system from video ingestion to compliance logging:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 1: INPUT ACQUISITION & PREPROCESSING           │
│                                                                        │
│   [ RTSP Stream / USB Camera / MP4 Video ]                             │
│                      │                                                 │
│                      ▼                                                 │
│   Threaded Video Ingestion (Zero-blocking ring buffer)                 │
│                      │                                                 │
│                      ▼                                                 │
│   Letterbox Preprocessing: 640x640x3, BGR -> RGB, Normalize [0, 1]     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 2: HARDWARE-ACCELERATED INFERENCE              │
│                                                                        │
│          ┌──────────────────────────────────────────────────┐          │
│          │             Target Execution Engine              │          │
│          ├─────────────────────────┬────────────────────────┤          │
│          │ Intel CPU (AVX2 INT8)   │ 15.81 - 21.01 ms       │          │
│          │ Intel iGPU (Iris Xe)    │ 43.59 - 44.34 ms       │          │
│          │ Intel FPGA (HETERO DLA) │ 16.9 - 56.6 FPS (Sim)  │          │
│          │ NVIDIA dGPU (CUDA FP32) │ 6.25 - 11.88 ms        │          │
│          └──────────────────────────────────────────────────┘          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 3: POST-PROCESSING & COMPLIANCE LOGIC          │
│                                                                        │
│   Anchor Decoding + Non-Maximum Suppression (NMS, IoU = 0.45)          │
│                      │                                                 │
│                      ▼                                                 │
│   Class Partitioning: Workers (W) vs. Protective Equipment (P)         │
│                      │                                                 │
│                      ▼                                                 │
│   Spatial Centroid Containment: a(p, w) = 1 [ c(p) in B_w ]            │
│   Compliance Verification:      C_w = PROD 1 [ exists p in R : a=1 ]   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 4: AUDIT LOGGING & VISUALIZATION               │
│                                                                        │
│   ┌──────────────────────────────┐    ┌─────────────────────────────┐  │
│   │ Visual Overlay Display       │    │ Audit Trail & Violation Log │  │
│   │ Green: Compliant Worker      │    │ Event CSV: results/logs/    │  │
│   │ Red:   Safety Violation Flag │    │ JPEG Snapshots: violations/ │  │
│   └──────────────────────────────┘    └─────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Benchmark Datasets

The system is evaluated across three industrial and construction datasets:

| Benchmark | Domain | Images | Annotations | Classes | Primary Target Artifacts |
|---|---|---|---|---|---|
| **SH17** | Manufacturing & Industrial | 8,099 | 75,994 | 17 | `person`, `helmet`, `safety-vest`, `glasses`, `gloves`, `face-guard`, `ear-mufs`, etc. |
| **CHV** | Real-World Construction Sites | 1,330 | 9,209 | 6 | `person`, `vest`, `blue/red/white/yellow helmet` |
| **CHVG** | Construction (Hardhat / Vest / Glass) | 1,699 | 11,604 | 8 | `person`, `vest`, `head`, `glass`, `blue/yellow/red helmet` |

### SH17 Class Taxonomy (17 Classes):
`person`, `ear`, `ear-mufs`, `face`, `face-guard`, `face-mask-medical`, `foot`, `tools`, `glasses`, `gloves`, `helmet`, `hands`, `head`, `medical-suit`, `shoes`, `safety-suit`, `safety-vest`.

---

## 5. Comprehensive Experimental Results

All experiments were benchmarked on physical edge hardware:
- **Host CPU**: Intel Core i5-13450HX (10 Cores, 16 Threads, up to 4.60 GHz, 20 MB L3 Cache)
- **Host RAM**: 16 GB DDR5-4800 MHz (76.8 GB/s peak bandwidth)
- **Integrated GPU**: Intel Iris Xe Graphics (32 Execution Units, 256 ALUs, Raptor Lake-S iGPU)
- **Discrete GPU**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.1)
- **OS**: Windows 11 Enterprise (64-bit)

### Table 1: In-Domain Detection Accuracy on SH17 Benchmark (100 Epochs)

| Model Architecture | Parameters | GFLOPs | Weight Size (.pt) | Precision | Recall | F1-Score | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|---|---|
| **YOLOv8n** | 3.01M | 8.2 | 5.96 MB | 0.5750 | 0.5360 | 0.5548 | 0.5473 | 0.3403 |
| **YOLOv8s (Selected)** | **11.14M** | **28.7** | **21.48 MB** | **0.6380** | **0.5750** | **0.6048** | **0.6214** | **0.3846** |
| **Performance Gain ($\Delta$)** | $+8.13\text{M}$ | $+20.5$ | $+15.52\text{ MB}$ | **$+0.0630$** | **$+0.0390$** | **$+0.0500$** | **$+0.0741$** | **$+0.0443$** |

### Table 2: Small-Object PPE Detection Performance (AP@50)

| PPE Category | YOLOv8n AP@50 | YOLOv8s AP@50 | Relative Gain | Significance |
|---|---|---|---|---|
| **Face-Guard** | 0.3120 | **0.4228** | **$+35.51\%$** | Critical eye/face splash protection |
| **Ear-Muffs** | 0.2850 | **0.3380** | **$+18.59\%$** | High-decibel noise environment PPE |
| **Safety Glasses** | 0.3540 | **0.3900** | **$+10.17\%$** | Small ocular projectile shield |
| **Safety Gloves** | 0.4610 | **0.4911** | **$+6.53\%$** | Chemical and mechanical hand hazard |
| **Safety Helmet** | 0.8120 | **0.8410** | $+3.57\%$ | Primary mandatory safety artifact |
| **Safety Vest** | 0.7980 | **0.8240** | $+3.26\%$ | High-visibility torso identification |

### Table 3: Hardware Inference Latency & Throughput Benchmark Across Devices

| Model | Runtime Engine | Target Device | Precision | Mean Latency | Throughput (FPS) | Model Size | mAP@50 | Speedup vs CPU Baseline |
|---|---|---|---|---|---|---|---|---|
| YOLOv8n | PyTorch (Baseline) | Intel CPU | FP32 | 57.61 ms | 17.36 FPS | 5.96 MB | 0.5473 | 1.00× (Baseline) |
| YOLOv8n | PyTorch (Baseline) | NVIDIA dGPU | FP32 | 6.25 ms | 160.01 FPS | 5.96 MB | 0.5473 | 9.22× |
| YOLOv8n | OpenVINO | Intel CPU | FP32 | 28.56 ms | 35.01 FPS | 11.92 MB | 0.5473 | **1.77×** |
| YOLOv8n | OpenVINO | Intel iGPU | FP16 | 22.73 ms | 43.99 FPS | 5.98 MB | 0.5473 | **2.22×** |
| YOLOv8s | PyTorch (Baseline) | Intel CPU | FP32 | 92.82–132.37 ms | 8.8–10.8 FPS | 21.48 MB | 0.6214 | 1.00× (Baseline) |
| YOLOv8s | PyTorch (Baseline) | NVIDIA dGPU | FP32 | 11.88 ms | 84.62 FPS | 21.48 MB | 0.6214 | 9.59× |
| YOLOv8s | OpenVINO | Intel CPU | FP32 | 58.23 ms | 17.17 FPS | 42.79 MB | 0.6214 | **1.96×** |
| YOLOv8s | OpenVINO | Intel CPU | FP16 | 57.89 ms | 17.27 FPS | 21.58 MB | 0.6184 | 1.97× |
| YOLOv8s | OpenVINO | Intel iGPU | FP16 | 66.76 ms | 14.98 FPS | 21.58 MB | 0.6184 | 1.71× |
| **YOLOv8s** | **OpenVINO (NNCF)** | **Intel CPU** | **INT8** | **15.81–21.01 ms** | **47.6–63.2 FPS** | **11.23 MB** | **0.6174** | **5.42–6.22×** |
| **YOLOv8s** | **OpenVINO (NNCF)** | **Intel iGPU** | **INT8** | **43.59–44.34 ms** | **22.6–22.9 FPS** | **11.23 MB** | **0.6174** | **2.57×** |
| **YOLOv8s** | **OpenVINO (NNCF)** | **OpenVINO AUTO**| **INT8** | **43.20–43.88 ms** | **22.8–23.2 FPS** | **11.23 MB** | **0.6174** | **2.64×** |

> **Key Architectural Takeaway**: Post-training INT8 quantization yields a **5.42–6.22× speedup** on commodity CPU cores, cuts model storage by **73.8%**, and sustains **47+ FPS**, while mAP@50 drops by only **0.0040 (0.4%)**.

### Table 4: Stage-by-Stage Latency Breakdown (End-to-End Pipeline on CPU)

| Pipeline Stage | Mean Latency | 95th Percentile Latency | Latency Share (%) |
|---|---|---|---|
| 1. Frame Acquisition & Decode | 0.18 ms | 0.26 ms | 0.6% |
| 2. Preprocessing (Letterbox 640×640) | 1.66 ms | 2.83 ms | 5.0% |
| **3. Inference (YOLOv8s INT8 OpenVINO)** | **30.23 ms** | **43.08 ms** | **90.5%** |
| 4. Post-processing (NMS, IoU 0.45) | 0.17 ms | 0.12 ms | 0.5% |
| **5. Spatial Compliance Logic** | **0.01 ms** | **0.01 ms** | **<0.1%** |
| 6. HUD Rendering & Overlay Display | 1.13 ms | 0.58 ms | 3.4% |
| **Total End-to-End Latency** | **33.38 ms** | **46.82 ms** | **100.0% (~30 FPS)** |

### Table 5: Zero-Shot Cross-Domain Generalization

| Training Benchmark | Evaluation Benchmark | Evaluated Domain | Common Classes | mAP@50 | Accuracy Transfer Retention |
|---|---|---|---|---|---|
| SH17 (In-Domain) | SH17 Val Split | Manufacturing | 17 Classes | **0.6214** | 100.0% (In-domain Baseline) |
| SH17 (Trained) | **CHV** | Construction Site | `person`, `vest`, `helmet` | **0.5750** | **92.5%** (Drop: only 0.0464) |
| SH17 (Trained) | **CHVG** | Construction Site | `person`, `vest`, `helmet`, `glass`, `head` | 0.2870 | 46.2% (Annotation convention shift) |

### Table 6: Intel FPGA AI Suite Hardware Roofline Analysis (`HETERO:FPGA,CPU`)

*Note: As formally documented in our project submission, academic licensing constraints for Quartus Prime Pro and DLA IP generators prevented physical FPGA bitstream flashing. Performance is modeled analytically via the Intel FPGA AI Suite roofline model.*

| FPGA Target Platform | Model | DSP Blocks | Peak INT8 Compute | Modeled Latency | Throughput (FPS) | Operational Viability |
|---|---|---|---|---|---|---|
| Cyclone V SoC (5CSXFC6) | YOLOv8n | 112 | 33.6 GOPs | 376.5 ms | 2.7 FPS | Compute-bound (Infeasible) |
| Cyclone V SoC (5CSXFC6) | YOLOv8s | 112 | 33.6 GOPs | 1315.2 ms | 0.8 FPS | Compute-bound (Infeasible) |
| **Intel Arria 10 GX 1150** | **YOLOv8n** | **1,518** | **759.0 GOPs** | **17.7 ms** | **56.6 FPS** | **Edge Real-Time Feasible** |
| **Intel Arria 10 GX 1150** | **YOLOv8s** | **1,518** | **759.0 GOPs** | **59.3 ms** | **16.9 FPS** | **Edge Industrial Feasible** |
| Stratix 10 GX 2800 | YOLOv8n | 5,760 | 4,032 GOPs | 4.2 ms | 239.3 FPS | Multi-stream Edge Server |
| Stratix 10 GX 2800 | YOLOv8s | 5,760 | 4,032 GOPs | 12.0 ms | 83.3 FPS | Multi-stream Edge Server |
| Agilex 7 (AGI 027) | YOLOv8n | 8,500 | 6,800 GOPs | 2.9 ms | 344.2 FPS | High-density Server Array |
| Agilex 7 (AGI 027) | YOLOv8s | 8,500 | 6,800 GOPs | 7.5 ms | 132.6 FPS | High-density Server Array |

---

## 6. Repository File Structure

```
Smart-Industrial-Safety/
│
├── app/                                    # Real-time production application
│   ├── __init__.py
│   ├── main.py                             # Ultra-fast threaded video monitor
│   ├── dashboard.py                        # Streamlit web analytics dashboard
│   ├── detector.py                         # OpenVINO & PyTorch inference wrapper
│   ├── compliance.py                       # Spatial PPE containment logic
│   └── logger.py                           # Asynchronous CSV audit & snapshot logger
│
├── configs/                                # Configuration files
│   ├── compliance.yaml                     # Mandatory & recommended PPE rules
│   └── sh17_data.yaml                      # Dataset YAML configuration
│
├── datasets/                               # Benchmark dataset directory
│   ├── SH17/                               # Primary training & validation benchmark
│   ├── CHV/                                # Cross-domain construction benchmark
│   └── CHVG/                               # Cross-domain hardhat/vest benchmark
│
├── demo/                                   # Demonstration materials
│   ├── realtime_monitor.py                 # Standalone lightweight monitor
│   └── sample_industrial_test.mp4          # Industrial test video stream
│
├── models/                                 # Trained checkpoints & OpenVINO IR models
│   ├── yolov8n_sh17_best.pt                # Trained PyTorch YOLOv8n weights
│   ├── yolov8s_sh17_best.pt                # Trained PyTorch YOLOv8s weights
│   ├── yolov8n_sh17_best_openvino_model/   # OpenVINO FP32/FP16 IR (.xml, .bin)
│   ├── yolov8s_fp32_openvino_model/        # OpenVINO FP32 IR (.xml, .bin)
│   ├── yolov8s_fp16_openvino_model/        # OpenVINO FP16 IR (.xml, .bin)
│   └── yolov8s_int8_openvino_model/        # NNCF Quantized INT8 IR (.xml, .bin)
│
├── results/                                # Output evaluation artifacts
│   ├── benchmarks/                         # Latency & throughput CSV logs
│   ├── cross_dataset/                      # Zero-shot cross-domain evaluation logs
│   ├── graphs/                             # Training curves, PR curves, CM plots
│   ├── logs/                               # Real-time safety violation event logs
│   │   └── violations/                     # Violation image captures
│   └── environment_report.txt              # Host hardware & environment log
│
├── scripts/                                # Pipeline automation scripts
│   ├── check_environment.py                # Environment verification utility
│   ├── prepare_sh17.py                     # SH17 dataset formatting & verification
│   ├── prepare_chv.py                      # CHV dataset formatting
│   ├── prepare_chvg.py                     # CHVG Pascal VOC to YOLO converter
│   ├── train.py                            # YOLOv8 training script
│   ├── export_openvino.py                  # OpenVINO FP32 / FP16 / INT8 exporter
│   ├── benchmark.py                        # Latency & FPS benchmark suite
│   ├── benchmark_response_time.py          # Multi-frame latency breakdown profiler
│   └── cross_domain_eval.py                # Zero-shot cross-domain evaluation suite
│
├── .gitignore                              # Git exclusion rules (protects paper files)
├── README.md                               # Project documentation (this file)
└── requirements.txt                        # Core Python dependencies
```

---

## 7. Installation & Quickstart Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/Shobhit-20-04/Smart-Industrial-Safety.git
cd Smart-Industrial-Safety
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
# Using conda
conda create -n safety-edge python=3.10 -y
conda activate safety-edge

# Or using standard venv
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify System Environment
```bash
python scripts/check_environment.py
```
*Expected Output*: Displays OS, CPU, RAM, PyTorch, CUDA, Ultralytics, OpenVINO, and OpenCV versions, confirming hardware availability.

---

## 8. Running the Real-Time Safety Application

### Option A: Threaded Video Stream Monitor (Highest Performance)
Runs an asynchronous, zero-blocking frame capture engine with full HUD overlay, dynamic worker-PPE association, and violation logging:

```bash
# 1. Run with quantized OpenVINO INT8 model on CPU (Default / Recommended)
python app/main.py --model models/yolov8s_int8_openvino_model --source 0 --device cpu

# 2. Run on a recorded video file
python app/main.py --model models/yolov8s_int8_openvino_model --source demo/sample_industrial_test.mp4 --device cpu

# 3. Run with PyTorch CUDA acceleration on discrete GPU
python app/main.py --model models/yolov8s_sh17_best.pt --source 0 --device 0

# 4. Run headless (no window, saving audit events to CSV)
python app/main.py --model models/yolov8s_int8_openvino_model --source demo/sample_industrial_test.mp4 --no-display
```

**Key CLI Arguments for `app/main.py`**:
- `--model`: Path to OpenVINO model directory or PyTorch `.pt` file.
- `--source`: Webcam index (`0`), video file path (`.mp4`), or image path (`.jpg`).
- `--device`: Target hardware (`cpu`, `gpu`, `0`, or `auto`).
- `--imgsz`: Input image resolution (`640`, `512`, or `480`).
- `--stride`: Frame processing stride (`1` = every frame, `2` = alternate frames).
- `--conf`: Detection confidence threshold (default `0.25`).
- `--required-ppe`: Comma-separated mandatory PPE list (default `"helmet,safety-vest"`).

### Option B: Interactive Streamlit Web Dashboard
Launches a web browser interface with live camera feed, real-time violation metrics, compliance percentages, model switching, and audit log inspection:

```bash
streamlit run app/dashboard.py
```
Navigate to `http://localhost:8501` to view the live dashboard.

### Option C: Standalone Lightweight Demo
```bash
python demo/realtime_monitor.py --model models/yolov8s_sh17_best.pt --source demo/sample_industrial_test.mp4
```

---

## 9. Reproducing Experiments & Pipelines

### 1. Dataset Preparation
```bash
# Inspect raw data formats
python scripts/inspect_all_datasets.py

# Format and validate SH17 manufacturing dataset
python scripts/prepare_sh17.py --mode hardlink

# Format cross-domain evaluation benchmarks
python scripts/prepare_chv.py --mode hardlink
python scripts/prepare_chvg.py --mode hardlink

# Validate integrity of all splits
python scripts/validate_dataset.py --all
```

### 2. Model Training (SH17 Benchmark)
```bash
# Train baseline YOLOv8n (100 epochs)
python scripts/train.py --model weights/yolov8n.pt --data datasets/SH17/data.yaml --epochs 100 --batch 16 --name train_yolov8n

# Train primary YOLOv8s (100 epochs)
python scripts/train.py --model weights/yolov8s.pt --data datasets/SH17/data.yaml --epochs 100 --batch 16 --name train_yolov8s
```

### 3. OpenVINO IR Export & INT8 Quantization
```bash
# Export YOLOv8s to OpenVINO FP32
python scripts/export_openvino.py --weights models/yolov8s_sh17_best.pt --precision fp32

# Export YOLOv8s to OpenVINO FP16
python scripts/export_openvino.py --weights models/yolov8s_sh17_best.pt --precision fp16

# Quantize YOLOv8s to INT8 with NNCF calibration (uses 300 representative SH17 images)
python scripts/export_openvino.py --weights models/yolov8s_sh17_best.pt --precision int8 --data datasets/SH17/data.yaml --fraction 0.05
```

### 4. Hardware Benchmarking
```bash
# Benchmark PyTorch FP32 on CPU & GPU
python scripts/benchmark.py --model models/yolov8s_sh17_best.pt --device cpu --num-runs 100
python scripts/benchmark.py --model models/yolov8s_sh17_best.pt --device 0 --num-runs 100

# Benchmark OpenVINO INT8 on CPU
python scripts/benchmark.py --model models/yolov8s_int8_openvino_model --device cpu --num-runs 100

# Profile multi-stage pipeline latency breakdown
python scripts/benchmark_response_time.py --model models/yolov8s_int8_openvino_model --num-frames 100
```

### 5. Zero-Shot Cross-Domain Evaluation
```bash
# Evaluate SH17-trained model on CHV construction benchmark
python scripts/cross_domain_eval.py --model models/yolov8s_sh17_best.pt --target-dataset chv

# Evaluate SH17-trained model on CHVG benchmark
python scripts/cross_domain_eval.py --model models/yolov8s_sh17_best.pt --target-dataset chvg
```

---

## 10. Research Team & Academic Information

This project was conducted as part of research in **Computer Architecture and Organization (MACSE517)** at the **School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, Tamil Nadu, India**.

### Research Team:
- **Shobhit Agrawal** (Reg. No.: `26MCS0057`) – *Literature survey, research gap formulation, YOLO training, OpenVINO IR compilation, NNCF INT8 post-training quantization, multi-device hardware benchmarking, Intel FPGA AI Suite roadmap, and roofline modeling.*
- **Mayuri Rajendra Pawar** (Reg. No.: `26MCS0063`) – *State-of-the-art comparative review, system workflow engineering, spatial worker-PPE compliance association engine, and real-time application integration.*
- **Aaryan Dawalkar** (Reg. No.: `26MCS0078`) – *Dataset profiling, SH17/CHV/CHVG class taxonomy alignment, zero-shot cross-domain evaluation, and validation split consistency verification.*

---

## 11. Citation & Reference

If you use this project, codebase, or findings in your academic work or industrial research, please cite:

### Standard Reference:
> **Shobhit Agrawal, Mayuri Rajendra Pawar, and Aaryan Dawalkar**, *"Smart Industrial Safety Monitoring System: Real-Time Edge PPE Compliance via OpenVINO INT8 Quantization and FPGA Implementation Roadmap"*, School of Computer Science and Engineering, Vellore Institute of Technology, 2026. Available: https://github.com/Shobhit-20-04/Smart-Industrial-Safety

### BibTeX:
```bibtex
@misc{smart_industrial_safety_2026,
  title={Smart Industrial Safety Monitoring System: Real-Time Edge PPE Compliance via OpenVINO INT8 Quantization and FPGA Implementation Roadmap},
  author={Shobhit Agrawal and Mayuri Rajendra Pawar and Aaryan Dawalkar},
  year={2026},
  howpublished={\url{https://github.com/Shobhit-20-04/Smart-Industrial-Safety}},
  note={School of Computer Science and Engineering, Vellore Institute of Technology}
}
```

---

## 12. License

This repository is distributed under the [MIT License](LICENSE).
You are free to use, modify, and distribute this software for educational, research, and commercial purposes with appropriate attribution.
