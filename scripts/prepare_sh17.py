"""
Prepare SH17 dataset for YOLOv8 training.
Splits images and labels into train/val according to train_files.txt and val_files.txt.
Maintains data integrity without altering the raw dataset.
Generates data.yaml and preparation report.
"""

import os
import shutil
import argparse
from pathlib import Path
import yaml

SH17_CLASSES = [
    "person",            # 0
    "ear",               # 1
    "ear-mufs",          # 2
    "face",              # 3
    "face-guard",        # 4
    "face-mask-medical", # 5
    "foot",              # 6
    "tools",             # 7
    "glasses",           # 8
    "gloves",            # 9
    "helmet",            # 10
    "hands",             # 11
    "head",              # 12
    "medical-suit",      # 13
    "shoes",             # 14
    "safety-suit",       # 15
    "safety-vest"        # 16
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
            # Fallback to copy if hardlink fails
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)

def prepare_sh17(
    raw_dir: Path,
    dest_dir: Path,
    mode: str = "hardlink",
    report_file: Path = None
):
    print(f"[prepare_sh17] Starting SH17 preparation...")
    print(f"  Raw directory: {raw_dir}")
    print(f"  Destination: {dest_dir}")
    print(f"  File placement mode: {mode}")

    raw_images_dir = raw_dir / "images"
    raw_labels_dir = raw_dir / "labels"
    train_txt_file = raw_dir / "train_files.txt"
    val_txt_file = raw_dir / "val_files.txt"

    if not train_txt_file.exists() or not val_txt_file.exists():
        raise FileNotFoundError("train_files.txt or val_files.txt not found in raw_dir")

    with open(train_txt_file, "r", encoding="utf-8") as f:
        train_filenames = [l.strip() for l in f if l.strip()]
    with open(val_txt_file, "r", encoding="utf-8") as f:
        val_filenames = [l.strip() for l in f if l.strip()]

    # Output directories
    img_train_dir = dest_dir / "images" / "train"
    img_val_dir = dest_dir / "images" / "val"
    lbl_train_dir = dest_dir / "labels" / "train"
    lbl_val_dir = dest_dir / "labels" / "val"

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_sh17] Processing {len(train_filenames)} train samples...")
    train_copied_img = 0
    train_copied_lbl = 0
    train_missing_img = 0
    train_missing_lbl = 0

    for fname in train_filenames:
        src_img = raw_images_dir / fname
        dst_img = img_train_dir / fname
        if src_img.exists():
            place_file(src_img, dst_img, mode)
            train_copied_img += 1
        else:
            train_missing_img += 1

        stem = Path(fname).stem
        src_lbl = raw_labels_dir / f"{stem}.txt"
        dst_lbl = lbl_train_dir / f"{stem}.txt"
        if src_lbl.exists():
            # Labels are always copied to ensure complete independence
            shutil.copy2(src_lbl, dst_lbl)
            train_copied_lbl += 1
        else:
            train_missing_lbl += 1

    print(f"[prepare_sh17] Processing {len(val_filenames)} validation samples...")
    val_copied_img = 0
    val_copied_lbl = 0
    val_missing_img = 0
    val_missing_lbl = 0

    for fname in val_filenames:
        src_img = raw_images_dir / fname
        dst_img = img_val_dir / fname
        if src_img.exists():
            place_file(src_img, dst_img, mode)
            val_copied_img += 1
        else:
            val_missing_img += 1

        stem = Path(fname).stem
        src_lbl = raw_labels_dir / f"{stem}.txt"
        dst_lbl = lbl_val_dir / f"{stem}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
            val_copied_lbl += 1
        else:
            val_missing_lbl += 1

    # Create data.yaml
    data_yaml_path = dest_dir / "data.yaml"
    yaml_content = {
        "path": str(dest_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(SH17_CLASSES)}
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    report_lines = [
        "============================================================",
        "SH17 DATASET PREPARATION REPORT",
        "============================================================",
        f"Raw Directory: {raw_dir}",
        f"Prepared Directory: {dest_dir}",
        f"Placement Mode: {mode}",
        f"Data YAML: {data_yaml_path}",
        "",
        "Train Set:",
        f"  Target Image Count: {len(train_filenames)}",
        f"  Images Placed: {train_copied_img}",
        f"  Labels Placed: {train_copied_lbl}",
        f"  Missing Images: {train_missing_img}",
        f"  Missing Labels: {train_missing_lbl}",
        "",
        "Validation Set:",
        f"  Target Image Count: {len(val_filenames)}",
        f"  Images Placed: {val_copied_img}",
        f"  Labels Placed: {val_copied_lbl}",
        f"  Missing Images: {val_missing_img}",
        f"  Missing Labels: {val_missing_lbl}",
        "",
        f"Total Classes: {len(SH17_CLASSES)}",
        "Class Hierarchy / List:",
    ]
    for cid, cname in enumerate(SH17_CLASSES):
        report_lines.append(f"  {cid:2d}: {cname}")
    report_lines.append("============================================================")

    report_text = "\n".join(report_lines)
    print(report_text)

    if report_file:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[prepare_sh17] Report written to: {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SH17 dataset for YOLOv8.")
    parser.add_argument("--raw_dir", type=Path, default=Path("D:/VIT/CAO Project Code/SH_dataset"),
                        help="Path to raw SH_dataset directory")
    parser.add_argument("--dest_dir", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/datasets/SH17"),
                        help="Path to destination SH17 directory")
    parser.add_argument("--mode", type=str, default="hardlink", choices=["hardlink", "copy"],
                        help="File placement mode (hardlink or copy)")
    parser.add_argument("--report", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/results/dataset/sh17_preparation_report.txt"),
                        help="Path to save preparation report")
    args = parser.parse_args()

    prepare_sh17(args.raw_dir, args.dest_dir, args.mode, args.report)
