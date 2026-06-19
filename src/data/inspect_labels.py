import os
import numpy as np

# 1. Expand the tilde (~) so Python recognizes the home directory
label_path = os.path.expanduser("~/workspace/projects/amvit/data/raw/train/labels")

def main():
    # Check if the directory actually exists
    if not os.path.exists(label_path):
        print(f"Error: The directory '{label_path}' does not exist.")
        return

    files = os.listdir(label_path)
    if not files:
        print("Error: No files found in the directory.")
        return

    # Filter for .txt files to avoid hidden system files
    txt_files = [f for f in files if f.endswith('.txt')]
    if not txt_files:
        print("No .txt label files found.")
        return

    sample = txt_files[0]
    print(f"Inspecting file: {sample}")

    with open(os.path.join(label_path, sample), 'r') as f:
        lines = f.readlines()

    for line in lines[:3]:
        parts = line.strip().split()
        if not parts:
            continue
            
        class_id = int(parts[0])
        # YOLOv8-OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
        coords = [float(x) for x in parts[1:]]
        
        print(f"Class {class_id}: corners = {coords}")
        # Note: Values are normalized (0-1) relative to image size

if __name__ == "__main__":
    main()

