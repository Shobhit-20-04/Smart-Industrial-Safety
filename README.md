# Smart Industrial Safety Monitoring System
### Research-Grade PPE Detection & Intel OpenVINO Edge Optimization

---

## 1. Project Title
**Smart Industrial Safety Monitoring System: Lightweight YOLO-based PPE Detection and Edge Optimization with Intel OpenVINO**

---

## 2. Problem Statement
In high-risk industrial, manufacturing, and construction environments, adherence to Personal Protective Equipment (PPE) regulations is crucial for preventing life-threatening injuries and occupational hazards. However, manual safety auditing is labor-intensive, error-prone, and incapable of providing continuous oversight. 

Deploying automated deep learning vision models directly on edge computing devices presents steep trade-offs: full-precision models frequently experience severe frame rate degradation and high latency, while overly quantized models can suffer catastrophic accuracy drops on small, critical PPE items (e.g., safety glasses, gloves, ear muffs). 

---

## 3. Research Objectives
1. **Curate & Standardize Datasets**: Validate and structure SH17, CHV, and CHVG industrial safety benchmarks with zero raw data alteration.
2. **Train Deep Detectors**: Train lightweight YOLOv8 models (YOLOv8n and YOLOv8s) on the comprehensive 17-class SH17 dataset.
3. **Compare Architectures**: Quantitatively assess the accuracy-complexity-latency frontier between YOLOv8n and YOLOv8s.
4. **Quantize & Optimize**: Export the primary model to Intel OpenVINO FP32, FP16, and post-training INT8 using representative calibration subsets.
5. **Cross-Domain Generalization**: Evaluate cross-dataset transferability on CHV and CHVG benchmarks without retraining.
6. **Edge Safety Application**: Develop a modular real-time monitoring engine with worker-level PPE compliance assessment.

---

## 4. Research Questions
- **RQ1**: How do lightweight detector architectures (YOLOv8n vs YOLOv8s) trade off mean Average Precision (mAP@50 and mAP@50-95) against computational efficiency on small industrial PPE artifacts?
- **RQ2**: What is the impact of Intel OpenVINO precision reduction (PyTorch FP32 $\rightarrow$ OpenVINO FP32 $\rightarrow$ FP16 $\rightarrow$ INT8) on inference latency, memory footprint, and detection degradation?
- **RQ3**: How well do models trained exclusively on large-scale web/industrial data (SH17) generalize to distinct construction video domains (CHV, CHVG) across shared PPE concepts?

---

## 5. System Architecture
```
[ Video / Camera Stream ]
           │
           ▼
[ Frame Preprocessing (640x640) ]
           │
           ▼
[ Optimized Detection Engine ] ──► (PyTorch FP32 / OpenVINO FP32 / FP16 / INT8)
           │
           ▼
[ Bounding Box & Class Filter ]
           │
           ▼
[ Worker-PPE Spatial Association ]
           │
           ▼
[ Safety Compliance Rules Engine ] ──► (Helmet, Vest, Eyewear, Gloves)
           │
           ▼
[ Real-Time Display & Alert Dashboard ]
```

---

## 6. Dataset Information
| Dataset | Purpose | Images | Annotations | Classes | Primary Format |
|---|---|---|---|---|---|
| **SH17** | Primary Training & In-Domain Validation | 8,099 | 75,994 | 17 | YOLO TXT / Pascal VOC |
| **CHV** | Cross-Domain Evaluation (Construction) | 1,330 | 9,209 | 6 | YOLO TXT |
| **CHVG** | Cross-Domain Evaluation (Hardhat/Vest/Glass) | 1,699 | 11,604 | 8 | Pascal VOC XML |

### Discovered Class Taxonomies:
- **SH17**: `person`, `ear`, `ear-mufs`, `face`, `face-guard`, `face-mask-medical`, `foot`, `tools`, `glasses`, `gloves`, `helmet`, `hands`, `head`, `medical-suit`, `shoes`, `safety-suit`, `safety-vest`
- **CHV**: `person`, `vest`, `blue helmet`, `red helmet`, `white helmet`, `yellow helmet`
- **CHVG**: `person`, `vest`, `white`, `yellow`, `head`, `blue`, `glass`, `red`

---

## 7. Installation & Environment Setup
```bash
# Clone or navigate to the repository
cd "D:/VIT/CAO Project Code/Smart-Industrial-Safety"

# Install dependencies
pip install -r requirements.txt

# Verify environment
python scripts/check_environment.py
```

---

## 8. Dataset Preparation & Validation
```bash
# 1. Run raw inspection
python scripts/inspect_all_datasets.py

# 2. Prepare SH17
python scripts/prepare_sh17.py --mode hardlink

# 3. Prepare CHV
python scripts/prepare_chv.py --mode hardlink

# 4. Prepare CHVG
python scripts/prepare_chvg.py --mode hardlink

# 5. Validate prepared datasets
python scripts/validate_dataset.py --all
```

---

## 9. Model Training
```bash
# Smoke test (3 epochs)
python scripts/train.py --model yolov8n.pt --data datasets/SH17/data.yaml --epochs 3 --batch 16 --name smoke_yolov8n

# Full training (YOLOv8n)
python scripts/train.py --model yolov8n.pt --data datasets/SH17/data.yaml --epochs 100 --batch 16 --name train_yolov8n

# Full training (YOLOv8s)
python scripts/train.py --model yolov8s.pt --data datasets/SH17/data.yaml --epochs 100 --batch 16 --name train_yolov8s
```

---

## 10. Experimental Results (To Be Populated Post-Execution)

### Model Comparison
| Model | Params (M) | FLOPs (G) | Size (MB) | Precision | Recall | mAP@50 | mAP@50-95 | Latency (ms) | FPS |
|---|---|---|---|---|---|---|---|---|---|
| **YOLOv8n** | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| **YOLOv8s** | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |

### OpenVINO Optimization Trade-off
| Runtime | Precision | Latency (ms) | FPS | Model Size (MB) | mAP@50 | Accuracy Delta (%) |
|---|---|---|---|---|---|---|
| PyTorch | FP32 | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | Baseline |
| OpenVINO | FP32 | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| OpenVINO | FP16 | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| OpenVINO | INT8 | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |

### Cross-Dataset Generalization
| Train Domain | Eval Domain | Common Classes | mAP@50 | mAP@50-95 | Generalization Drop (%) |
|---|---|---|---|---|---|
| SH17 | SH17 (In-domain) | 17 classes | [TO BE FILLED] | [TO BE FILLED] | Baseline |
| SH17 | CHV (Cross-domain) | person, vest, helmet | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| SH17 | CHVG (Cross-domain) | person, vest, helmet, glass, head | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |

---

## 11. Real-Time Application & Dashboard
```bash
# Real-time PPE monitor (webcam)
python demo/realtime_monitor.py --model models/best.pt --source 0

# Streamlit interactive dashboard
streamlit run app/streamlit_app.py
```

---

## 12. Authors & Academic Affiliation
- **Author**: M.Tech CSE Candidate
- **Project**: M.Tech Research Project - Computer Science & Engineering
- **Institution**: Vellore Institute of Technology (VIT)
