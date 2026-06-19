import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def analyze_dataset(label_dir: str):
    print(PROJECT_ROOT)
    class_counts = Counter()
    rotations = []
    aspect_ratios = []

    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(label_dir, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                cls = int(parts[0])
                class_counts[cls] += 1
                coords = np.array([float(x) for x in parts[1:]]).reshape(4, 2)
                # Compute angle from first edge
                dx = coords[1, 0] - coords[0, 0]
                dy = coords[1, 1] - coords[0, 1]
                angle = np.degrees(np.arctan2(dy, dx))
                rotations.append(angle)
                # Compute side lengths for aspect ratio
                w = np.linalg.norm(coords[1] - coords[0])
                h = np.linalg.norm(coords[3] - coords[0])
                if h > 0:
                    aspect_ratios.append(w / h)

    print("Class distribution:", dict(class_counts))
    print(f"Angle range: {min(rotations):.1f}° to {max(rotations):.1f}°")
    print(f"Median aspect ratio: {np.median(aspect_ratios):.2f}")

    # Plot class balance
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    classes = ['cargo', 'military', 'carrier', 'cruise', 'tanker', 'ferry']
    axes[0].bar(classes, [class_counts[i] for i in range(6)])
    axes[0].set_title("Class distribution")
    axes[0].tick_params(axis='x', rotation=30)
    axes[1].hist(rotations, bins=36, range=(-180, 180))
    axes[1].set_title("Rotation distribution (degrees)")
    plt.tight_layout()
    output_path = os.path.expanduser("~/workspace/projects/amvit/data/dataset_analysis.png")
    plt.savefig(output_path, dpi=120)
    return class_counts

if __name__ == "__main__":
    label_path = os.path.expanduser("~/workspace/projects/amvit/data/raw/train/labels")
    analyze_dataset(label_path)
