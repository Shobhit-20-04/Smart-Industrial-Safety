"""
Generate Comprehensive Dataset Statistics.
Computes class frequencies, split counts, objects per image, and bounding box scale distributions
for SH17, CHV, and CHVG.
Outputs results to CSV and JSON formats for research analysis.
"""

import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
import yaml

def analyze_dataset_labels(label_dir: Path, class_names: list):
    class_counts = Counter()
    objects_per_img = []
    box_scales = {"small": 0, "medium": 0, "large": 0}
    total_boxes = 0

    txt_files = list(label_dir.glob("*.txt"))
    for txt_path in txt_files:
        count_in_file = 0
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    xc, yc, w, h = [float(x) for x in parts[1:]]
                    if 0 <= cid < len(class_names):
                        class_counts[class_names[cid]] += 1
                    else:
                        class_counts[f"class_{cid}"] += 1

                    # Area calculation (normalized)
                    norm_area = w * h
                    if norm_area < 0.005:
                        box_scales["small"] += 1
                    elif norm_area < 0.05:
                        box_scales["medium"] += 1
                    else:
                        box_scales["large"] += 1

                    count_in_file += 1
                    total_boxes += 1
        objects_per_img.append(count_in_file)

    mean_obj = sum(objects_per_img) / max(len(objects_per_img), 1)
    min_obj = min(objects_per_img) if objects_per_img else 0
    max_obj = max(objects_per_img) if objects_per_img else 0

    return {
        "num_images": len(txt_files),
        "total_objects": total_boxes,
        "mean_objects_per_image": round(mean_obj, 2),
        "min_objects_per_image": min_obj,
        "max_objects_per_image": max_obj,
        "class_counts": dict(class_counts),
        "scale_distribution": box_scales
    }

def generate_all_stats(base_dir: Path, results_dir: Path):
    datasets_dir = base_dir / "datasets"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    csv_rows = []

    # 1. SH17
    sh17_yaml = datasets_dir / "SH17" / "data.yaml"
    with open(sh17_yaml, "r", encoding="utf-8") as f:
        sh17_cfg = yaml.safe_load(f)
    sh17_classes = [sh17_cfg["names"][i] for i in sorted(sh17_cfg["names"].keys())]

    sh17_train_stats = analyze_dataset_labels(datasets_dir / "SH17" / "labels" / "train", sh17_classes)
    sh17_val_stats = analyze_dataset_labels(datasets_dir / "SH17" / "labels" / "val", sh17_classes)
    all_stats["SH17"] = {
        "classes": sh17_classes,
        "train": sh17_train_stats,
        "val": sh17_val_stats,
        "total_images": sh17_train_stats["num_images"] + sh17_val_stats["num_images"],
        "total_objects": sh17_train_stats["total_objects"] + sh17_val_stats["total_objects"]
    }

    # 2. CHV
    chv_yaml = datasets_dir / "CHV" / "data.yaml"
    with open(chv_yaml, "r", encoding="utf-8") as f:
        chv_cfg = yaml.safe_load(f)
    chv_classes = [chv_cfg["names"][i] for i in sorted(chv_cfg["names"].keys())]

    chv_train_stats = analyze_dataset_labels(datasets_dir / "CHV" / "labels" / "train", chv_classes)
    chv_val_stats = analyze_dataset_labels(datasets_dir / "CHV" / "labels" / "valid", chv_classes)
    chv_test_stats = analyze_dataset_labels(datasets_dir / "CHV" / "labels" / "test", chv_classes)
    all_stats["CHV"] = {
        "classes": chv_classes,
        "train": chv_train_stats,
        "valid": chv_val_stats,
        "test": chv_test_stats,
        "total_images": chv_train_stats["num_images"] + chv_val_stats["num_images"] + chv_test_stats["num_images"],
        "total_objects": chv_train_stats["total_objects"] + chv_val_stats["total_objects"] + chv_test_stats["total_objects"]
    }

    # 3. CHVG
    chvg_yaml = datasets_dir / "CHVG" / "data.yaml"
    with open(chvg_yaml, "r", encoding="utf-8") as f:
        chvg_cfg = yaml.safe_load(f)
    chvg_classes = [chvg_cfg["names"][i] for i in sorted(chvg_cfg["names"].keys())]

    chvg_stats = analyze_dataset_labels(datasets_dir / "CHVG" / "labels", chvg_classes)
    all_stats["CHVG"] = {
        "classes": chvg_classes,
        "all": chvg_stats,
        "total_images": chvg_stats["num_images"],
        "total_objects": chvg_stats["total_objects"]
    }

    # Export to JSON
    json_path = results_dir / "dataset_detailed_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"[generate_dataset_stats] Detailed JSON written to: {json_path}")

    # Export class frequencies to CSV
    csv_path = results_dir / "class_distribution_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "split", "class_id", "class_name", "instance_count", "percentage_of_split"])

        # SH17
        for split, sdata in [("train", sh17_train_stats), ("val", sh17_val_stats)]:
            tot = max(sdata["total_objects"], 1)
            for cid, cname in enumerate(sh17_classes):
                cnt = sdata["class_counts"].get(cname, 0)
                pct = round(100.0 * cnt / tot, 2)
                writer.writerow(["SH17", split, cid, cname, cnt, pct])

        # CHV
        for split, sdata in [("train", chv_train_stats), ("valid", chv_val_stats), ("test", chv_test_stats)]:
            tot = max(sdata["total_objects"], 1)
            for cid, cname in enumerate(chv_classes):
                cnt = sdata["class_counts"].get(cname, 0)
                pct = round(100.0 * cnt / tot, 2)
                writer.writerow(["CHV", split, cid, cname, cnt, pct])

        # CHVG
        tot = max(chvg_stats["total_objects"], 1)
        for cid, cname in enumerate(chvg_classes):
            cnt = chvg_stats["class_counts"].get(cname, 0)
            pct = round(100.0 * cnt / tot, 2)
            writer.writerow(["CHVG", "all", cid, cname, cnt, pct])

    print(f"[generate_dataset_stats] Class distribution CSV written to: {csv_path}")

if __name__ == "__main__":
    base = Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety")
    res = base / "results" / "dataset"
    generate_all_stats(base, res)
