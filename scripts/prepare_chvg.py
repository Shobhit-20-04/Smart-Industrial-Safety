"""
Prepare CHVG dataset for YOLO evaluation and cross-domain testing.
Parses Pascal VOC XML annotations, converts bounding boxes to normalized YOLO TXT format.
Places images and labels without modifying the raw dataset.
Generates data.yaml and chvg_conversion_report.txt.
"""

import os
import shutil
import argparse
from pathlib import Path
from collections import Counter
import yaml
from convert_voc_to_yolo import convert_voc_xml_to_yolo

# Discovered CHVG classes from XML annotations
CHVG_CLASSES = [
    "person",   # 0
    "vest",     # 1
    "blue",     # 2 (blue helmet)
    "red",      # 3 (red helmet)
    "white",    # 4 (white helmet)
    "yellow",   # 5 (yellow helmet)
    "head",     # 6
    "glass"     # 7 (protective glasses/eyewear)
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

def prepare_chvg(
    raw_dir: Path,
    dest_dir: Path,
    mode: str = "hardlink",
    report_file: Path = None
):
    print(f"[prepare_chvg] Starting CHVG preparation...")
    print(f"  Raw directory: {raw_dir}")
    print(f"  Destination: {dest_dir}")
    print(f"  Placement mode: {mode}")

    class_to_id = {cname: i for i, cname in enumerate(CHVG_CLASSES)}

    dest_images_dir = dest_dir / "images"
    dest_labels_dir = dest_dir / "labels"
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(list(raw_dir.glob("*.xml")))
    total_xml = len(xml_files)
    successful_conversions = 0
    missing_images = 0
    invalid_xml = 0
    empty_annotations = 0
    unknown_classes = Counter()
    objects_per_class = Counter()

    for xml_p in xml_files:
        stem = xml_p.stem
        img_candidates = [
            raw_dir / f"{stem}.jpg",
            raw_dir / f"{stem}.jpeg",
            raw_dir / f"{stem}.png"
        ]
        src_img = next((c for c in img_candidates if c.exists()), None)
        if not src_img:
            missing_images += 1
            continue

        # Place image
        dst_img = dest_images_dir / src_img.name
        place_file(src_img, dst_img, mode)

        # Convert annotation
        try:
            yolo_lines, ok, err_msg = convert_voc_xml_to_yolo(xml_p, class_to_id, image_dir=raw_dir, allow_unknown=True)
            if not ok:
                invalid_xml += 1
                continue

            if not yolo_lines:
                empty_annotations += 1

            for line in yolo_lines:
                cid = int(line.split()[0])
                objects_per_class[CHVG_CLASSES[cid]] += 1

            dst_lbl = dest_labels_dir / f"{stem}.txt"
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            successful_conversions += 1
        except Exception:
            invalid_xml += 1

    # Create data.yaml
    data_yaml_path = dest_dir / "data.yaml"
    yaml_content = {
        "path": str(dest_dir.resolve()).replace("\\", "/"),
        "train": "images",
        "val": "images",
        "names": {i: name for i, name in enumerate(CHVG_CLASSES)}
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    report_lines = [
        "============================================================",
        "CHVG DATASET CONVERSION REPORT",
        "============================================================",
        f"Raw Directory: {raw_dir}",
        f"Prepared Directory: {dest_dir}",
        f"Placement Mode: {mode}",
        f"Data YAML: {data_yaml_path}",
        "",
        f"total XML files: {total_xml}",
        f"successful conversions: {successful_conversions}",
        f"missing images: {missing_images}",
        f"invalid XML: {invalid_xml}",
        f"unknown classes: {sum(unknown_classes.values())}",
        f"empty annotations: {empty_annotations}",
        "",
        "total objects per class:",
    ]
    for cname in CHVG_CLASSES:
        cnt = objects_per_class.get(cname, 0)
        report_lines.append(f"  {cname:<12}: {cnt:,}")
    report_lines.append("============================================================")

    report_text = "\n".join(report_lines)
    print(report_text)

    if report_file:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[prepare_chvg] Report written to: {report_file}")

    return {
        "total_xml": total_xml,
        "successful_conversions": successful_conversions,
        "missing_images": missing_images,
        "invalid_xml": invalid_xml,
        "empty_annotations": empty_annotations,
        "objects_per_class": dict(objects_per_class)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CHVG dataset for YOLO.")
    parser.add_argument("--raw_dir", type=Path, default=Path("D:/VIT/CAO Project Code/CHVG-Dataset"),
                        help="Path to raw CHVG-Dataset directory")
    parser.add_argument("--dest_dir", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/datasets/CHVG"),
                        help="Path to destination CHVG directory")
    parser.add_argument("--mode", type=str, default="hardlink", choices=["hardlink", "copy"],
                        help="File placement mode (hardlink or copy)")
    parser.add_argument("--report", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/results/dataset/chvg_conversion_report.txt"),
                        help="Path to save conversion report")
    args = parser.parse_args()

    prepare_chvg(args.raw_dir, args.dest_dir, args.mode, args.report)
