"""
Generate YOLO data.yaml configuration file.
Compatible with Ultralytics YOLOv8.
"""

import argparse
from pathlib import Path
import yaml

def create_dataset_yaml(
    output_path: Path,
    dataset_path: Path,
    train_rel: str,
    val_rel: str,
    class_names: list,
    test_rel: str = None
):
    data_dict = {
        "path": str(dataset_path.resolve()).replace("\\", "/"),
        "train": train_rel,
        "val": val_rel,
    }
    if test_rel:
        data_dict["test"] = test_rel

    data_dict["names"] = {i: name for i, name in enumerate(class_names)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)

    print(f"[create_yaml] Created YAML config at: {output_path}")
    print(f"  Path: {data_dict['path']}")
    print(f"  Train: {train_rel}")
    print(f"  Val: {val_rel}")
    print(f"  Classes ({len(class_names)}): {class_names}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create YOLO data.yaml file.")
    parser.add_argument("--output", type=Path, required=True, help="Output data.yaml path")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Base dataset path")
    parser.add_argument("--train", type=str, required=True, help="Relative train images path")
    parser.add_argument("--val", type=str, required=True, help="Relative val images path")
    parser.add_argument("--test", type=str, default=None, help="Relative test images path")
    parser.add_argument("--classes", nargs="+", required=True, help="List of class names")
    args = parser.parse_args()

    create_dataset_yaml(
        args.output,
        args.dataset_path,
        args.train,
        args.val,
        args.classes,
        args.test
    )
