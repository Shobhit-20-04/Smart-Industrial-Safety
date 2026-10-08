"""
Inspect SH17 dataset.
Dataset location: D:/VIT/CAO Project Code/SH_dataset
Verifies images, YOLO labels, VOC labels, meta-data, and train/val splits.
Extracts class IDs and names, counts objects, and reports statistics.
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET

def inspect_sh17(raw_dir: Path, output_file: Path = None):
    print(f"[SH17] Inspecting raw directory: {raw_dir}")
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"
    voc_dir = raw_dir / "voc_labels"
    meta_dir = raw_dir / "meta-data"
    train_file = raw_dir / "train_files.txt"
    val_file = raw_dir / "val_files.txt"

    # Verify directories
    for d, name in [(images_dir, "images"), (labels_dir, "labels"), 
                    (voc_dir, "voc_labels"), (meta_dir, "meta-data")]:
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(f"Missing expected directory: {d}")

    # Read files
    image_files = {p.name: p for p in images_dir.iterdir() if p.is_file()}
    image_stems = {p.stem: p for p in image_files.values()}
    label_files = {p.name: p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}
    label_stems = {p.stem: p for p in label_files.values()}
    voc_files = {p.name: p for p in voc_dir.iterdir() if p.is_file() and p.suffix.lower() == ".xml"}
    voc_stems = {p.stem: p for p in voc_files.values()}
    meta_files = {p.name: p for p in meta_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"}

    # Train and val splits
    train_images = set()
    val_images = set()
    if train_file.exists():
        with open(train_file, "r", encoding="utf-8") as f:
            train_images = {line.strip() for line in f if line.strip()}
    if val_file.exists():
        with open(val_file, "r", encoding="utf-8") as f:
            val_images = {line.strip() for line in f if line.strip()}

    # Image-label matching
    missing_labels = set(image_stems.keys()) - set(label_stems.keys())
    missing_images = set(label_stems.keys()) - set(image_stems.keys())
    missing_voc = set(image_stems.keys()) - set(voc_stems.keys())

    # Map class IDs to class names by cross-referencing VOC xml and YOLO labels
    id_to_name = {}
    class_id_counts = Counter()
    total_annotations = 0
    invalid_labels = 0

    # Read class mappings from first set of matched files
    for stem in list(voc_stems.keys())[:500]:
        xml_p = voc_stems[stem]
        txt_p = label_stems.get(stem)
        if not txt_p:
            continue
        try:
            tree = ET.parse(xml_p)
            root = tree.getroot()
            xml_names = [obj.find("name").text.strip() for obj in root.findall("object") if obj.find("name") is not None]
            with open(txt_p, "r", encoding="utf-8") as f:
                txt_lines = [l.strip().split() for l in f if l.strip()]
            txt_ids = [int(l[0]) for l in txt_lines if len(l) >= 5]
            if len(xml_names) == len(txt_ids):
                for cid, cname in zip(txt_ids, xml_names):
                    if cid not in id_to_name:
                        id_to_name[cid] = cname
            if len(id_to_name) >= 17:
                break
        except Exception:
            continue

    # Count all annotations and check validity across all labels
    for txt_p in label_files.values():
        try:
            with open(txt_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        invalid_labels += 1
                        continue
                    cid = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    # check normalized coords [0, 1]
                    if any(c < 0.0 or c > 1.0 for c in coords):
                        invalid_labels += 1
                        continue
                    class_id_counts[cid] += 1
                    total_annotations += 1
        except Exception:
            invalid_labels += 1

    # Standard class list ordered by ID
    discovered_classes = [id_to_name.get(i, f"class_{i}") for i in range(len(id_to_name))]

    summary = {
        "dataset": "SH17",
        "raw_directory": str(raw_dir),
        "total_images": len(image_files),
        "total_labels": len(label_files),
        "total_voc_labels": len(voc_files),
        "total_metadata": len(meta_files),
        "train_split_count": len(train_images),
        "val_split_count": len(val_images),
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
        "missing_voc": len(missing_voc),
        "invalid_labels": invalid_labels,
        "total_annotations": total_annotations,
        "num_classes": len(discovered_classes),
        "classes": discovered_classes,
        "class_id_mapping": id_to_name,
        "class_counts": {id_to_name.get(k, f"class_{k}"): v for k, v in sorted(class_id_counts.items())}
    }

    report_lines = [
        "============================================================",
        "DATASET INSPECTION REPORT: SH17",
        "============================================================",
        f"Raw Directory: {raw_dir}",
        f"Total Images: {summary['total_images']}",
        f"Total YOLO Labels: {summary['total_labels']}",
        f"Total VOC XMLs: {summary['total_voc_labels']}",
        f"Total Metadata JSONs: {summary['total_metadata']}",
        f"Train Split Files: {summary['train_split_count']}",
        f"Validation Split Files: {summary['val_split_count']}",
        f"Missing Labels for Images: {summary['missing_labels']}",
        f"Missing Images for Labels: {summary['missing_images']}",
        f"Invalid Annotation Lines: {summary['invalid_labels']}",
        f"Total Object Annotations: {summary['total_annotations']}",
        f"Number of Discovered Classes: {summary['num_classes']}",
        "",
        "Discovered Class Index to Name Mapping:",
    ]
    for cid in range(len(discovered_classes)):
        cname = discovered_classes[cid]
        cnt = summary["class_counts"].get(cname, 0)
        report_lines.append(f"  ID {cid:2d}: {cname:<20} (Instance count: {cnt:,})")
    report_lines.append("============================================================")

    report_text = "\n".join(report_lines)
    print(report_text)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[SH17] Report saved to: {output_file}")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect raw SH17 dataset.")
    parser.add_argument("--raw_dir", type=Path, default=Path("D:/VIT/CAO Project Code/SH_dataset"),
                        help="Path to raw SH_dataset directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to save report text file")
    args = parser.parse_args()
    inspect_sh17(args.raw_dir, args.output)
