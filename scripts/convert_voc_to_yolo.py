"""
Convert Pascal VOC XML annotations to YOLO TXT format.
Normalizes coordinates to [0, 1] as (class_id, x_center, y_center, width, height).
Preserves data integrity and generates conversion statistics report.
"""

import argparse
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

def convert_voc_xml_to_yolo(
    xml_path: Path,
    class_to_id: dict,
    image_dir: Path = None,
    allow_unknown: bool = False
):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Image dimensions
    size_el = root.find("size")
    if size_el is None:
        raise ValueError(f"XML missing <size> tag: {xml_path}")
    
    width = float(size_el.find("width").text)
    height = float(size_el.find("height").text)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions (width={width}, height={height}) in {xml_path}")

    # Verify matching image exists if image_dir is provided
    if image_dir:
        filename_el = root.find("filename")
        fname = filename_el.text if filename_el is not None else f"{xml_path.stem}.jpg"
        img_candidates = [
            image_dir / fname,
            image_dir / f"{xml_path.stem}.jpg",
            image_dir / f"{xml_path.stem}.jpeg",
            image_dir / f"{xml_path.stem}.png"
        ]
        if not any(cand.exists() for cand in img_candidates):
            return [], False, f"Image file not found for {xml_path.name}"

    yolo_lines = []
    unknown_classes = []
    invalid_boxes = 0

    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        cname = name_el.text.strip()
        if cname not in class_to_id:
            if allow_unknown:
                unknown_classes.append(cname)
                continue
            else:
                raise ValueError(f"Unknown class '{cname}' in {xml_path.name}")

        cid = class_to_id[cname]
        bndbox = obj.find("bndbox")
        if bndbox is None:
            invalid_boxes += 1
            continue

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Clip slightly out-of-bound coords to image boundaries
        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))

        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 0 or bh <= 0:
            invalid_boxes += 1
            continue

        xc = (xmin + xmax) / 2.0 / width
        yc = (ymin + ymax) / 2.0 / height
        norm_w = bw / width
        norm_h = bh / height

        # Ensure normalized in [0, 1]
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        norm_w = max(0.0, min(1.0, norm_w))
        norm_h = max(0.0, min(1.0, norm_h))

        yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")

    return yolo_lines, True, None

def convert_voc_dataset(
    xml_dir: Path,
    output_dir: Path,
    classes: list,
    image_dir: Path = None,
    report_file: Path = None
):
    output_dir.mkdir(parents=True, exist_ok=True)
    class_to_id = {name: i for i, name in enumerate(classes)}

    xml_files = sorted(list(xml_dir.glob("*.xml")))
    total_xml = len(xml_files)
    successful = 0
    missing_images = 0
    invalid_xml = 0
    empty_annotations = 0
    unknown_classes = Counter()
    objects_per_class = Counter()

    for xml_p in xml_files:
        try:
            lines, ok, err_msg = convert_voc_xml_to_yolo(xml_p, class_to_id, image_dir=image_dir, allow_unknown=True)
            if not ok:
                if "Image file not found" in err_msg:
                    missing_images += 1
                else:
                    invalid_xml += 1
                continue

            if not lines:
                empty_annotations += 1

            for line in lines:
                cid = int(line.split()[0])
                cname = classes[cid]
                objects_per_class[cname] += 1

            out_path = output_dir / f"{xml_p.stem}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

            successful += 1
        except Exception as e:
            invalid_xml += 1

    report_lines = [
        "============================================================",
        "PASCAL VOC TO YOLO CONVERSION REPORT",
        "============================================================",
        f"Source XML Directory: {xml_dir}",
        f"Output YOLO Directory: {output_dir}",
        f"Total XML Files: {total_xml}",
        f"Successful Conversions: {successful}",
        f"Missing Images: {missing_images}",
        f"Invalid XML Files: {invalid_xml}",
        f"Empty Annotations (Background Images): {empty_annotations}",
        f"Total Classes: {len(classes)}",
        f"Class List: {classes}",
        "",
        "Objects Per Class Converted:",
    ]
    for cname in classes:
        cnt = objects_per_class.get(cname, 0)
        report_lines.append(f"  {cname:<15}: {cnt:,}")
    report_lines.append("============================================================")

    report_text = "\n".join(report_lines)
    print(report_text)

    if report_file:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[convert_voc_to_yolo] Report saved to: {report_file}")

    return {
        "total_xml": total_xml,
        "successful": successful,
        "missing_images": missing_images,
        "invalid_xml": invalid_xml,
        "empty_annotations": empty_annotations,
        "objects_per_class": dict(objects_per_class)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XMLs to YOLO TXT.")
    parser.add_argument("--xml_dir", type=Path, required=True, help="Directory containing XML files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for YOLO TXT files")
    parser.add_argument("--classes", nargs="+", required=True, help="List of class names in exact index order")
    parser.add_argument("--image_dir", type=Path, default=None, help="Directory containing images to verify presence")
    parser.add_argument("--report", type=Path, default=None, help="Path to save report file")
    args = parser.parse_args()

    convert_voc_dataset(
        args.xml_dir,
        args.output_dir,
        args.classes,
        image_dir=args.image_dir,
        report_file=args.report
    )
