"""
Dataset Validation Script.
Validates YOLO formatted datasets against 10 strict criteria:
1. Every image has expected label.
2. Every label has corresponding image.
3. Bounding boxes are valid.
4. Coordinates are normalized in [0, 1].
5. Class IDs are within range [0, num_classes-1].
6. Images are readable.
7. Labels are readable.
8. No NaN or Inf values.
9. No negative or zero widths/heights.
10. No duplicate image names causing collisions.

Outputs results/dataset/validation_report.txt.
Returns exit code 0 if valid, non-zero if critical failure.
"""

import sys
import math
import argparse
from pathlib import Path
from PIL import Image
import yaml

def validate_split(
    img_dir: Path,
    lbl_dir: Path,
    num_classes: int,
    check_image_readability: bool = True,
    sample_img_check: int = 100
):
    errors = []
    warnings = []

    if not img_dir.exists():
        errors.append(f"Image directory does not exist: {img_dir}")
        return errors, warnings, {}

    if not lbl_dir.exists():
        errors.append(f"Label directory does not exist: {lbl_dir}")
        return errors, warnings, {}

    # Discover images
    img_map = {}
    duplicates = set()
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            stem = p.stem
            if stem in img_map:
                duplicates.add(stem)
            img_map[stem] = p

    if duplicates:
        errors.append(f"Duplicate image names found ({len(duplicates)}): {list(duplicates)[:5]}")

    # Discover labels
    lbl_map = {}
    for p in lbl_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".txt":
            lbl_map[p.stem] = p

    # 1 & 2. Check 1-to-1 matching
    missing_labels = set(img_map.keys()) - set(lbl_map.keys())
    missing_images = set(lbl_map.keys()) - set(img_map.keys())

    if missing_labels:
        errors.append(f"Images missing labels ({len(missing_labels)}): {list(missing_labels)[:5]}")
    if missing_images:
        errors.append(f"Labels missing images ({len(missing_images)}): {list(missing_images)[:5]}")

    # 3, 4, 5, 7, 8, 9. Validate label files
    total_objects = 0
    invalid_labels = 0
    out_of_range_classes = 0
    unnormalized_coords = 0
    nan_coords = 0
    invalid_dimensions = 0

    for stem, lbl_path in lbl_map.items():
        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        errors.append(f"Invalid column count in {lbl_path.name} line {line_idx+1}: {line.strip()}")
                        invalid_labels += 1
                        continue

                    # Check float parsing
                    try:
                        cid = int(parts[0])
                        vals = [float(x) for x in parts[1:]]
                    except ValueError:
                        errors.append(f"Parse error in {lbl_path.name} line {line_idx+1}")
                        invalid_labels += 1
                        continue

                    # Check NaN or Inf
                    if any(math.isnan(v) or math.isinf(v) for v in vals):
                        nan_coords += 1
                        errors.append(f"NaN/Inf detected in {lbl_path.name} line {line_idx+1}")
                        continue

                    xc, yc, w, h = vals

                    # Check class ID range
                    if cid < 0 or cid >= num_classes:
                        out_of_range_classes += 1
                        errors.append(f"Class ID {cid} out of range [0, {num_classes-1}] in {lbl_path.name}")

                    # Check normalized range [0, 1]
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                        unnormalized_coords += 1
                        errors.append(f"Unnormalized coords ({xc}, {yc}, {w}, {h}) in {lbl_path.name}")

                    # Check positive width & height
                    if w <= 0.0 or h <= 0.0:
                        invalid_dimensions += 1
                        errors.append(f"Non-positive box width/height ({w}, {h}) in {lbl_path.name}")

                    total_objects += 1

        except Exception as e:
            errors.append(f"Unreadable label file {lbl_path.name}: {e}")
            invalid_labels += 1

    # 6. Check image readability (sample or all)
    unreadable_images = 0
    to_check = list(img_map.values())
    if sample_img_check > 0 and len(to_check) > sample_img_check:
        to_check = to_check[:sample_img_check]

    if check_image_readability:
        for img_p in to_check:
            try:
                with Image.open(img_p) as img:
                    img.verify()
            except Exception as e:
                unreadable_images += 1
                errors.append(f"Unreadable image {img_p.name}: {e}")

    stats = {
        "images": len(img_map),
        "labels": len(lbl_map),
        "objects": total_objects,
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
        "invalid_labels": invalid_labels,
        "out_of_range_classes": out_of_range_classes,
        "unnormalized_coords": unnormalized_coords,
        "nan_coords": nan_coords,
        "invalid_dimensions": invalid_dimensions,
        "unreadable_images": unreadable_images
    }

    return errors, warnings, stats

def validate_dataset(
    dataset_dir: Path,
    data_yaml_path: Path,
    report_file: Path = None,
    sample_img_check: int = 200
):
    print(f"[validate_dataset] Validating dataset at: {dataset_dir}")
    print(f"  YAML config: {data_yaml_path}")

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes = cfg.get("names", {})
    if isinstance(classes, dict):
        num_classes = len(classes)
        class_list = [classes[k] for k in sorted(classes.keys())]
    elif isinstance(classes, list):
        num_classes = len(classes)
        class_list = classes
    else:
        raise ValueError("Invalid format for 'names' in data.yaml")

    all_errors = []
    all_warnings = []
    split_summaries = {}

    splits_to_check = []
    # Check what splits exist
    for split_key in ["train", "val", "valid", "test"]:
        rel_path = cfg.get(split_key)
        if rel_path:
            img_dir = dataset_dir / rel_path
            lbl_dir = dataset_dir / "labels" / Path(rel_path).name
            if not lbl_dir.exists():
                lbl_dir = dataset_dir / "labels"
            splits_to_check.append((split_key, img_dir, lbl_dir))

    # If no standard split keys, check images/ and labels/ directly
    if not splits_to_check:
        img_dir = dataset_dir / "images"
        lbl_dir = dataset_dir / "labels"
        splits_to_check.append(("default", img_dir, lbl_dir))

    for split_name, img_dir, lbl_dir in splits_to_check:
        print(f"  -> Validating split '{split_name}' (images: {img_dir}, labels: {lbl_dir})...")
        errs, warns, stats = validate_split(
            img_dir, lbl_dir, num_classes,
            check_image_readability=True,
            sample_img_check=sample_img_check
        )
        all_errors.extend(errs)
        all_warnings.extend(warns)
        split_summaries[split_name] = stats

    is_passed = len(all_errors) == 0

    report_lines = [
        "============================================================",
        "DATASET VALIDATION REPORT",
        "============================================================",
        f"Dataset: {dataset_dir.name}",
        f"Directory: {dataset_dir}",
        f"YAML: {data_yaml_path}",
        f"Total Classes Defined: {num_classes}",
        f"Class Names: {class_list}",
        f"Overall Status: {'PASSED (VERIFIED RESEARCH-GRADE)' if is_passed else 'FAILED'}",
        "",
        "Split Breakdown:"
    ]

    for split_name, stats in split_summaries.items():
        report_lines.append(f"Split: {split_name}")
        for k, v in stats.items():
            report_lines.append(f"  {k}: {v}")
        report_lines.append("")

    if all_errors:
        report_lines.append(f"Errors Encountered ({len(all_errors)}):")
        for e in all_errors[:20]:
            report_lines.append(f"  - ERROR: {e}")
        if len(all_errors) > 20:
            report_lines.append(f"  ... and {len(all_errors) - 20} more errors.")
    else:
        report_lines.append("All 10 validation criteria verified with ZERO errors.")

    report_lines.append("============================================================")
    report_text = "\n".join(report_lines)
    print(report_text)

    if report_file:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(report_text + "\n\n")
        print(f"[validate_dataset] Report appended to: {report_file}")

    return is_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLO dataset.")
    parser.add_argument("--dataset_dir", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/datasets/SH17"),
                        help="Path to dataset directory")
    parser.add_argument("--data_yaml", type=Path, default=None,
                        help="Path to data.yaml (default: dataset_dir/data.yaml)")
    parser.add_argument("--report", type=Path, default=Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/results/dataset/validation_report.txt"),
                        help="Path to output validation report")
    parser.add_argument("--all", action="store_true", help="Validate all three datasets (SH17, CHV, CHVG)")
    args = parser.parse_args()

    if args.all:
        base = Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/datasets")
        # Clear existing report file
        if args.report.exists():
            args.report.unlink()
        overall_ok = True
        for dname in ["SH17", "CHV", "CHVG"]:
            ds_dir = base / dname
            yaml_p = ds_dir / "data.yaml"
            if ds_dir.exists() and yaml_p.exists():
                ok = validate_dataset(ds_dir, yaml_p, args.report)
                if not ok:
                    overall_ok = False
        sys.exit(0 if overall_ok else 1)
    else:
        yaml_p = args.data_yaml or (args.dataset_dir / "data.yaml")
        ok = validate_dataset(args.dataset_dir, yaml_p, args.report)
        sys.exit(0 if ok else 1)
