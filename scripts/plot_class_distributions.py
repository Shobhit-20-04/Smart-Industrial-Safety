"""
Generate Publication-Quality Class Distribution and Dataset Exploration Plots.
Generates:
1. results/graphs/sh17_class_distribution.png
2. results/graphs/chv_class_distribution.png
3. results/graphs/chvg_class_distribution.png
4. results/graphs/cross_dataset_common_ppe.png
5. results/graphs/bounding_box_scale_distribution.png
"""

import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Set clean scientific plotting style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

def plot_distributions(stats_json: Path, graphs_dir: Path):
    graphs_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. SH17 Class Distribution
    sh17 = data["SH17"]
    classes = sh17["classes"]
    train_counts = [sh17["train"]["class_counts"].get(c, 0) for c in classes]
    val_counts = [sh17["val"]["class_counts"].get(c, 0) for c in classes]

    y_pos = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(10, 8))
    width = 0.45

    ax.barh(y_pos + width/2, train_counts, width, label=f"Train Split ({sh17['train']['total_objects']:,} objs)", color="#1f77b4", alpha=0.85)
    ax.barh(y_pos - width/2, val_counts, width, label=f"Val Split ({sh17['val']['total_objects']:,} objs)", color="#ff7f0e", alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel("Instance Count")
    ax.set_title("SH17 Dataset: Class Frequency by Split (17 Classes)", weight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    # Annotate total counts
    for i, (tc, vc) in enumerate(zip(train_counts, val_counts)):
        tot = tc + vc
        ax.text(tc + vc + 150, i, f"{tot:,}", va="center", fontsize=9, color="#333333")

    plt.tight_layout()
    out1 = graphs_dir / "sh17_class_distribution.png"
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"[plot_class_distributions] Saved: {out1}")

    # 2. CHV Class Distribution
    chv = data["CHV"]
    chv_classes = chv["classes"]
    chv_train = [chv["train"]["class_counts"].get(c, 0) for c in chv_classes]
    chv_val = [chv["valid"]["class_counts"].get(c, 0) for c in chv_classes]
    chv_test = [chv["test"]["class_counts"].get(c, 0) for c in chv_classes]

    y_pos = np.arange(len(chv_classes))
    fig, ax = plt.subplots(figsize=(9, 6))
    bar_w = 0.25

    ax.barh(y_pos - bar_w, chv_train, bar_w, label=f"Train ({chv['train']['total_objects']:,})", color="#2ca02c", alpha=0.85)
    ax.barh(y_pos, chv_val, bar_w, label=f"Valid ({chv['valid']['total_objects']:,})", color="#1f77b4", alpha=0.85)
    ax.barh(y_pos + bar_w, chv_test, bar_w, label=f"Test ({chv['test']['total_objects']:,})", color="#d62728", alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(chv_classes)
    ax.invert_yaxis()
    ax.set_xlabel("Instance Count")
    ax.set_title("CHV Dataset: Class Frequency Across Splits (6 Classes)", weight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    for i, (t, v, te) in enumerate(zip(chv_train, chv_val, chv_test)):
        tot = t + v + te
        ax.text(tot + 60, i, f"{tot:,}", va="center", fontsize=9, color="#333333")

    plt.tight_layout()
    out2 = graphs_dir / "chv_class_distribution.png"
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"[plot_class_distributions] Saved: {out2}")

    # 3. CHVG Class Distribution
    chvg = data["CHVG"]
    chvg_classes = chvg["classes"]
    chvg_counts = [chvg["all"]["class_counts"].get(c, 0) for c in chvg_classes]

    # Sort descending
    sorted_pairs = sorted(zip(chvg_classes, chvg_counts), key=lambda x: -x[1])
    s_classes, s_counts = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(s_classes, s_counts, color="#9467bd", alpha=0.85, width=0.6)
    ax.set_ylabel("Instance Count")
    ax.set_title("CHVG Dataset: Class Frequency Distribution (8 Classes)", weight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:,}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out3 = graphs_dir / "chvg_class_distribution.png"
    plt.savefig(out3, dpi=300)
    plt.close()
    print(f"[plot_class_distributions] Saved: {out3}")

    # 4. Cross-Dataset Common PPE Concept Comparison
    # Common concepts: Person, Vest, Helmet, Head, Glasses
    # For Helmet: CHV aggregates (blue, red, white, yellow), CHVG aggregates (blue, red, white, yellow)
    # SH17 counts
    sh_person = sh17["train"]["class_counts"].get("person", 0) + sh17["val"]["class_counts"].get("person", 0)
    sh_vest = sh17["train"]["class_counts"].get("safety-vest", 0) + sh17["val"]["class_counts"].get("safety-vest", 0)
    sh_helmet = sh17["train"]["class_counts"].get("helmet", 0) + sh17["val"]["class_counts"].get("helmet", 0)
    sh_head = sh17["train"]["class_counts"].get("head", 0) + sh17["val"]["class_counts"].get("head", 0)
    sh_glass = sh17["train"]["class_counts"].get("glasses", 0) + sh17["val"]["class_counts"].get("glasses", 0)

    # CHV counts
    chv_all_counts = Counter()
    for s in ["train", "valid", "test"]:
        for c, cnt in chv[s]["class_counts"].items():
            chv_all_counts[c] += cnt
    chv_person = chv_all_counts.get("person", 0)
    chv_vest = chv_all_counts.get("vest", 0)
    chv_helmet = chv_all_counts.get("blue helmet", 0) + chv_all_counts.get("red helmet", 0) + \
                 chv_all_counts.get("white helmet", 0) + chv_all_counts.get("yellow helmet", 0)
    chv_head = 0 # Not labeled in CHV
    chv_glass = 0 # Not labeled in CHV

    # CHVG counts
    chvg_c = chvg["all"]["class_counts"]
    chvg_person = chvg_c.get("person", 0)
    chvg_vest = chvg_c.get("vest", 0)
    chvg_helmet = chvg_c.get("blue", 0) + chvg_c.get("red", 0) + chvg_c.get("white", 0) + chvg_c.get("yellow", 0)
    chvg_head = chvg_c.get("head", 0)
    chvg_glass = chvg_c.get("glass", 0)

    concepts = ["Person", "Safety Vest", "Helmet (All Colors)", "Head", "Eye Protection"]
    sh_vals = [sh_person, sh_vest, sh_helmet, sh_head, sh_glass]
    chv_vals = [chv_person, chv_vest, chv_helmet, chv_head, chv_glass]
    chvg_vals = [chvg_person, chvg_vest, chvg_helmet, chvg_head, chvg_glass]

    x = np.arange(len(concepts))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w, sh_vals, w, label=f"SH17 (Source Domain: {sh17['total_images']:,} imgs)", color="#1f77b4", alpha=0.85)
    ax.bar(x, chv_vals, w, label=f"CHV (Cross Domain 1: {chv['total_images']:,} imgs)", color="#2ca02c", alpha=0.85)
    ax.bar(x + w, chvg_vals, w, label=f"CHVG (Cross Domain 2: {chvg['total_images']:,} imgs)", color="#ff7f0e", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(concepts)
    ax.set_ylabel("Instance Count (Log Scale)")
    ax.set_yscale("log")
    ax.set_title("Cross-Dataset Common Safety Concepts Instance Comparison", weight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out4 = graphs_dir / "cross_dataset_common_ppe.png"
    plt.savefig(out4, dpi=300)
    plt.close()
    print(f"[plot_class_distributions] Saved: {out4}")

    # 5. Object Scale (Small vs Medium vs Large) Distribution
    datasets = ["SH17", "CHV", "CHVG"]
    scales = ["small", "medium", "large"]

    # Sum scales
    sh_scales = [sh17["train"]["scale_distribution"][s] + sh17["val"]["scale_distribution"][s] for s in scales]
    chv_scales = [sum(chv[sp]["scale_distribution"][s] for sp in ["train", "valid", "test"]) for s in scales]
    chvg_scales = [chvg["all"]["scale_distribution"][s] for s in scales]

    # Convert to percentages
    sh_pct = [100.0 * v / sum(sh_scales) for v in sh_scales]
    chv_pct = [100.0 * v / sum(chv_scales) for v in chv_scales]
    chvg_pct = [100.0 * v / sum(chvg_scales) for v in chvg_scales]

    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(8, 5))

    bottoms = np.zeros(len(datasets))
    colors = ["#17becf", "#bcbd22", "#7f7f7f"]
    labels = ["Small (Area < 0.5%)", "Medium (0.5% - 5%)", "Large (Area > 5%)"]

    all_pcts = np.array([sh_pct, chv_pct, chvg_pct]).T
    for i in range(3):
        ax.bar(datasets, all_pcts[i], bottom=bottoms, label=labels[i], color=colors[i], alpha=0.85, width=0.5)
        for j, val in enumerate(all_pcts[i]):
            ax.text(j, bottoms[j] + val/2, f"{val:.1f}%", ha="center", va="center", fontsize=9, color="black", weight="bold")
        bottoms += all_pcts[i]

    ax.set_ylabel("Percentage of Total Objects (%)")
    ax.set_title("Bounding Box Scale Distribution Across Datasets", weight="bold", pad=12)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out5 = graphs_dir / "bounding_box_scale_distribution.png"
    plt.savefig(out5, dpi=300)
    plt.close()
    print(f"[plot_class_distributions] Saved: {out5}")

if __name__ == "__main__":
    base = Path("D:/VIT/CAO Project Code/Smart-Industrial-Safety")
    stats_p = base / "results" / "dataset" / "dataset_detailed_stats.json"
    g_dir = base / "results" / "graphs"
    plot_distributions(stats_p, g_dir)
