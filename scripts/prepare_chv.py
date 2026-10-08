"""
Prepare CHV dataset for YOLO evaluation and cross-domain testing.
Preserves original raw dataset.
Splits images and labels according to data split files (train.txt, valid.txt, test.txt).
Generates data.yaml and preparation report.
"""

import os
import shutil
import argparse
from pathlib import Path
import yaml

CHV_CLASSES = [
    "person",        # 0
    "vest",          # 1
    "blue helmet",   # 2
    "red helmet",    # 3
    "white helmet",  # 4
    "yellow helmet"  # 5
]

def place_file(src: Path, dst: Path, mode: str = "hardlink"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)

def prepare_chv(
    raw_dir: Path,
    dest_dir: Path,
    mode: str = "hardlink",
    report_file: Path = None
):
    print(f"[prepare_chv] Starting CHV preparation...")
    print(f"  Raw directory: {raw_dir}")
    print(f"  Destination: {dest_dir}")
    print(f"  Placement mode: {mode}")

    raw_images_dir = raw_dir / "images"
    raw_annotations_dir = raw_dir / "annotations"
    split_dir = raw_dir / "data split"

    split_configs = {
        "train": split_dir / "train.txt",
        "valid": split_dir / "valid.txt",
        "test": split_dir / "test.txt"
    }

    report_lines = [
        "============================================================",
        "CHV DATASET PREPARATION REPORT",
        "============================================================",
        f"Raw Directory: {raw_dir}",
        f"Prepared Directory: {dest_dir}",
        f"Placement Mode: {mode}",
        ""
    ]

    total_stats = {}

    for split_key, split_path in split_configs.items():
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        with open(split_path, "r", encoding="utf-8") as f:
            rel_lines = [l.strip() for l in f if l.strip()]

        out_img_dir = dest_dir / "images" / split_key
        out_lbl_dir = dest_dir / "labels" / split_key
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        copied_img = 0
        copied_lbl = 0
        missing_img = 0
        missing_lbl = 0

        for line in rel_lines:
            # line is e.g. "CHV_dataset/images/ppe_1106.jpg" or "ppe_1106.jpg"
            fname = Path(line).name
            stem = Path(line).stem

            src_img = raw_images_dir / fname
            dst_img = out_img_dir / fname
            if src_img.exists():
                place_file(src_img, dst_img, mode)
                copied_img += 1
            else:
                missing_img += 1

            src_lbl = raw_annotations_dir / f"{stem}.txt"
            dst_lbl = out_lbl_dir / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
                copied_lbl += 1
            else:
                missing_lbl += 1

        total_stats[split_key] = {
            "total": len(rel_lines),
            "images": copied_img,
            "labels": copied_lbl,
            "missing_images": missing_img,
            "missing_labels": missing_lbl
        }

        report_lines.append(f"Split '{split_key}':")
        report_lines.append(f"  Target Samples: {len(rel_lines)}")
        report_lines.append(f"  Images Placed: {copied_img}")
        report_lines.append(f"  Labels Placed: {copied_lbl}")
        report_lines.append(f"  Missing Images: {missing_img}")
        report_lines.append(f"  Missing Labels: {missing_lbl}")
        report_lines.append("")

    # Create data.yaml
    data_yaml_path = dest_dir / "data.yaml"
    yaml_content = {
        "path": str(dest_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CHV_CLASSES)}
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    report_lines.append(f"Data YAML: {data_yaml_path}")
    report_lines.append(f"Total Classes ({len(CHV_CLASSES)}): {CHV_CLASSES}")
    report_lines.append("============================================================")

    report_text = "\n".join(report_lines)
    print(report_text)

    if report_file:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[prepare_chv] Report written to: {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CHV dataset for YOLO.")
    parser.add_argument("--raw_dir", type=Path, default=Path("D:/VIT/CAO Project Code/CHV_dataset/CHV_dataset"),
                        help="Path to raw CHV_dataset/CHV_dataset directory")
    parser.add_argument("--dest_dir", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/datasets/CHV"),
                        help="Path to destination CHV directory")
    parser.add_argument("--mode", type=str, default="hardlink", choices=["hardlink", "copy"],
                        help="File placement mode (hardlink or copy)")
    parser.add_argument("--report", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/results/dataset/chv_preparation_report.txt"),
                        help="Path to save preparation report")
    args = parser.parse_args()

    prepare_chv(args.raw_dir, args.dest_dir, args.mode, args.report)
