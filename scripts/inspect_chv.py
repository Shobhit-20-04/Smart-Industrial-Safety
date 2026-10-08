"""
Inspect CHV dataset.
Dataset location: D:/VIT/CAO Project Code/CHV_dataset/CHV_dataset
Verifies images, YOLO annotations, README class specification, and train/valid/test splits.
Extracts class IDs and names, counts objects, and reports statistics.
"""

import argparse
from pathlib import Path
from collections import Counter

def inspect_chv(raw_dir: Path, output_file: Path = None):
    print(f"[CHV] Inspecting raw directory: {raw_dir}")
    images_dir = raw_dir / "images"
    annotations_dir = raw_dir / "annotations"
    split_dir = raw_dir / "data split"
    readme_path = annotations_dir / "README.md"

    # Verify directories
    for d, name in [(images_dir, "images"), (annotations_dir, "annotations"), (split_dir, "data split")]:
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(f"Missing expected directory: {d}")

    # Read class definitions from README.md if present
    classes_from_readme = {}
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    parts = line.split(":")
                    if parts[0].strip().isdigit():
                        cid = int(parts[0].strip())
                        cname = parts[1].strip()
                        classes_from_readme[cid] = cname

    image_files = {p.name: p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]}
    image_stems = {p.stem: p for p in image_files.values()}
    
    annotation_files = {p.name: p for p in annotations_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}
    annotation_stems = {p.stem: p for p in annotation_files.values()}

    # Check splits
    splits = {}
    for split_name in ["train.txt", "valid.txt", "test.txt"]:
        sp_path = split_dir / split_name
        if sp_path.exists():
            with open(sp_path, "r", encoding="utf-8") as f:
                splits[split_name] = [l.strip() for l in f if l.strip()]
        else:
            splits[split_name] = []

    # Image-annotation matching
    missing_annotations = set(image_stems.keys()) - set(annotation_stems.keys())
    missing_images = set(annotation_stems.keys()) - set(image_stems.keys())

    # Count object instances and check annotation validity
    class_id_counts = Counter()
    total_annotations = 0
    invalid_annotations = 0

    for txt_p in annotation_files.values():
        try:
            with open(txt_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        invalid_annotations += 1
                        continue
                    cid = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    if any(c < 0.0 or c > 1.0 for c in coords):
                        invalid_annotations += 1
                        continue
                    class_id_counts[cid] += 1
                    total_annotations += 1
        except Exception:
            invalid_annotations += 1

    discovered_classes = [classes_from_readme.get(i, f"class_{i}") for i in range(len(classes_from_readme))]

    summary = {
        "dataset": "CHV",
        "raw_directory": str(raw_dir),
        "total_images": len(image_files),
        "total_annotations": len(annotation_files),
        "train_split_count": len(splits.get("train.txt", [])),
        "valid_split_count": len(splits.get("valid.txt", [])),
        "test_split_count": len(splits.get("test.txt", [])),
        "missing_annotations": len(missing_annotations),
        "missing_images": len(missing_images),
        "invalid_annotations": invalid_annotations,
        "total_objects": total_annotations,
        "num_classes": len(discovered_classes),
        "classes": discovered_classes,
        "class_id_mapping": classes_from_readme,
        "class_counts": {classes_from_readme.get(k, f"class_{k}"): v for k, v in sorted(class_id_counts.items())}
    }

    report_lines = [
        "============================================================",
        "DATASET INSPECTION REPORT: CHV",
        "============================================================",
        f"Raw Directory: {raw_dir}",
        f"Total Images: {summary['total_images']}",
        f"Total YOLO Annotation TXT: {summary['total_annotations']}",
        f"Train Split Files: {summary['train_split_count']}",
        f"Valid Split Files: {summary['valid_split_count']}",
        f"Test Split Files: {summary['test_split_count']}",
        f"Missing Annotations for Images: {summary['missing_annotations']}",
        f"Missing Images for Annotations: {summary['missing_images']}",
        f"Invalid Annotation Lines: {summary['invalid_annotations']}",
        f"Total Object Annotations: {summary['total_objects']}",
        f"Number of Discovered Classes: {summary['num_classes']}",
        "",
        "Discovered Class Index to Name Mapping (from annotations/README.md):",
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
        print(f"[CHV] Report saved to: {output_file}")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect raw CHV dataset.")
    parser.add_argument("--raw_dir", type=Path, default=Path("D:/VIT/CAO Project Code/CHV_dataset/CHV_dataset"),
                        help="Path to raw CHV_dataset/CHV_dataset directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to save report text file")
    args = parser.parse_args()
    inspect_chv(args.raw_dir, args.output)
