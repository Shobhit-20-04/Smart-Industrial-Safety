"""
Inspect CHVG dataset.
Dataset location: D:/VIT/CAO Project Code/CHVG-Dataset
Verifies JPG images, Pascal VOC XML annotations.
Extracts all unique <name> class tags, checks image dimensions and coordinates (xmin, ymin, xmax, ymax).
Identifies empty annotations, missing pairs, and bounding box validity.
"""

import argparse
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

def inspect_chvg(raw_dir: Path, output_file: Path = None):
    print(f"[CHVG] Inspecting raw directory: {raw_dir}")
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing expected directory: {raw_dir}")

    image_files = {}
    xml_files = {}
    other_files = []

    for p in raw_dir.iterdir():
        if p.is_file():
            suffix = p.suffix.lower()
            if suffix in [".jpg", ".jpeg", ".png"]:
                image_files[p.stem] = p
            elif suffix == ".xml":
                xml_files[p.stem] = p
            else:
                other_files.append(p.name)

    # Missing checks
    missing_xml = set(image_files.keys()) - set(xml_files.keys())
    missing_images = set(xml_files.keys()) - set(image_files.keys())

    class_counts = Counter()
    invalid_xml = 0
    invalid_boxes = 0
    empty_annotations = 0
    total_objects = 0
    image_sizes = Counter()

    for stem, xml_path in xml_files.items():
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Image size
            size_el = root.find("size")
            if size_el is not None:
                w = int(size_el.find("width").text)
                h = int(size_el.find("height").text)
                image_sizes[(w, h)] += 1

            objs = root.findall("object")
            if not objs:
                empty_annotations += 1

            for obj in objs:
                total_objects += 1
                name_el = obj.find("name")
                cname = name_el.text.strip() if (name_el is not None and name_el.text) else "UNKNOWN"
                class_counts[cname] += 1

                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0:
                        invalid_boxes += 1
                else:
                    invalid_boxes += 1
        except Exception:
            invalid_xml += 1

    # Sort classes by count descending
    discovered_classes = [c for c, _ in sorted(class_counts.items(), key=lambda x: -x[1])]

    summary = {
        "dataset": "CHVG",
        "raw_directory": str(raw_dir),
        "total_images": len(image_files),
        "total_xml_annotations": len(xml_files),
        "other_files": other_files,
        "missing_xml": len(missing_xml),
        "missing_images": len(missing_images),
        "invalid_xml": invalid_xml,
        "invalid_boxes": invalid_boxes,
        "empty_annotations": empty_annotations,
        "total_objects": total_objects,
        "num_classes": len(discovered_classes),
        "classes": discovered_classes,
        "class_counts": dict(class_counts),
        "image_sizes": {f"{w}x{h}": cnt for (w, h), cnt in image_sizes.items()}
    }

    report_lines = [
        "============================================================",
        "DATASET INSPECTION REPORT: CHVG",
        "============================================================",
        f"Raw Directory: {raw_dir}",
        f"Total Images: {summary['total_images']}",
        f"Total Pascal VOC XML Annotations: {summary['total_xml_annotations']}",
        f"Missing XML for Images: {summary['missing_xml']}",
        f"Missing Images for XML: {summary['missing_images']}",
        f"Invalid XML Files: {summary['invalid_xml']}",
        f"Invalid Bounding Boxes: {summary['invalid_boxes']}",
        f"Empty Annotation XMLs (Background Images): {summary['empty_annotations']}",
        f"Total Object Annotations: {summary['total_objects']}",
        f"Image Resolutions: {summary['image_sizes']}",
        f"Number of Discovered Classes: {summary['num_classes']}",
        "",
        "Discovered Class Names from XML <name> tags (ranked by frequency):",
    ]
    for cid, cname in enumerate(discovered_classes):
        cnt = summary["class_counts"].get(cname, 0)
        report_lines.append(f"  [{cid}] {cname:<15} : {cnt:,} instances")
    report_lines.append("============================================================")

    report_text = "\n".join(report_lines)
    print(report_text)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[CHVG] Report saved to: {output_file}")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect raw CHVG dataset.")
    parser.add_argument("--raw_dir", type=Path, default=Path("D:/VIT/CAO Project Code/CHVG-Dataset"),
                        help="Path to raw CHVG-Dataset directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to save report text file")
    args = parser.parse_args()
    inspect_chvg(args.raw_dir, args.output)
