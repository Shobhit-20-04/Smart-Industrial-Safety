"""
Master Dataset Inspection Runner.
Runs inspections on SH17, CHV, and CHVG raw datasets.
Generates:
  - results/dataset_inspection_report.txt
  - results/dataset_summary.csv
"""

import csv
from pathlib import Path
from inspect_sh17 import inspect_sh17
from inspect_chv import inspect_chv
from inspect_chvg import inspect_chvg

def run_all_inspections(
    sh17_raw: Path,
    chv_raw: Path,
    chvg_raw: Path,
    results_dir: Path
):
    print("=" * 70)
    print("RUNNING COMPREHENSIVE DATASET INSPECTION FOR ALL THREE DATASETS")
    print("=" * 70)

    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = results_dir / "dataset_inspection_report.txt"
    summary_csv = results_dir / "dataset_summary.csv"

    # Run inspections
    sh_sum = inspect_sh17(sh17_raw)
    chv_sum = inspect_chv(chv_raw)
    chvg_sum = inspect_chvg(chvg_raw)

    all_summaries = [sh_sum, chv_sum, chvg_sum]

    # Combine text reports
    full_report = []
    full_report.append("=" * 70)
    full_report.append("RESEARCH-GRADE DATASET INSPECTION REPORT")
    full_report.append("Project: Smart Industrial Safety Monitoring System")
    full_report.append("=" * 70)
    full_report.append("")

    for s in all_summaries:
        full_report.append(f"DATASET: {s['dataset']}")
        full_report.append(f"  Raw Path: {s['raw_directory']}")
        full_report.append(f"  Images: {s['total_images']}")
        ann_cnt = s.get("total_labels") or s.get("total_annotations") or s.get("total_xml_annotations")
        full_report.append(f"  Annotations: {ann_cnt}")
        full_report.append(f"  Missing Images: {s.get('missing_images', 0)}")
        full_report.append(f"  Missing Annotations: {s.get('missing_labels') or s.get('missing_annotations') or s.get('missing_xml', 0)}")
        full_report.append(f"  Invalid Annotations: {s.get('invalid_labels', 0) + s.get('invalid_annotations', 0) + s.get('invalid_xml', 0) + s.get('invalid_boxes', 0)}")
        full_report.append(f"  Class Count: {s['num_classes']}")
        full_report.append(f"  Classes: {', '.join(s['classes'])}")
        full_report.append("  Class Instance Breakdown:")
        for cname, cnt in s["class_counts"].items():
            full_report.append(f"    - {cname}: {cnt:,}")
        full_report.append("-" * 70)

    report_text = "\n".join(full_report)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Master Inspection] Full report written to: {report_file}")

    # Write summary CSV
    # Columns required:
    # dataset,image_count,annotation_count,class_count,classes,missing_images,missing_annotations,invalid_annotations
    csv_rows = []
    for s in all_summaries:
        ann_cnt = s.get("total_labels") or s.get("total_annotations") or s.get("total_xml_annotations")
        miss_ann = s.get("missing_labels") or s.get("missing_annotations") or s.get("missing_xml", 0)
        inval = s.get("invalid_labels", 0) + s.get("invalid_annotations", 0) + s.get("invalid_xml", 0) + s.get("invalid_boxes", 0)
        classes_str = ";".join(s["classes"])
        csv_rows.append({
            "dataset": s["dataset"],
            "image_count": s["total_images"],
            "annotation_count": ann_cnt,
            "class_count": s["num_classes"],
            "classes": classes_str,
            "missing_images": s.get("missing_images", 0),
            "missing_annotations": miss_ann,
            "invalid_annotations": inval
        })

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset",
            "image_count",
            "annotation_count",
            "class_count",
            "classes",
            "missing_images",
            "missing_annotations",
            "invalid_annotations"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    print(f"[Master Inspection] Summary CSV written to: {summary_csv}")

if __name__ == "__main__":
    sh17 = Path("D:/VIT/CAO Project Code/SH_dataset")
    chv = Path("D:/VIT/CAO Project Code/CHV_dataset/CHV_dataset")
    chvg = Path("D:/VIT/CAO Project Code/CHVG-Dataset")
    res = Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety/results")
    run_all_inspections(sh17, chv, chvg, res)
