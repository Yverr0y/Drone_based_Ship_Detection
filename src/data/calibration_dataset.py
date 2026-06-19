import os
import shutil
import random
import cv2
import numpy as np

img_path = os.path.expanduser("~/workspace/projects/amvit/data/raw/train/images")
cal_path = os.path.expanduser("~/workspace/projects/amvit/data/calibration")

def prepare_calibration_set(

    source_dir: str = img_path,
    output_dir: str = cal_path,
    n_images: int = 1000,
    target_size: tuple = (640, 640)
):
    """
    Select a representative subset for INT8 calibration.
    Stratified by lighting, scale, and density if possible.
    For simplicity here: random sample covering diversity.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_images = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png'))]
    
    # Sample without replacement
    selected = random.sample(all_images, min(n_images, len(all_images)))
    
    for fname in selected:
        src = os.path.join(source_dir, fname)
        img = cv2.imread(src)
        img = cv2.resize(img, target_size)
        cv2.imwrite(os.path.join(output_dir, fname), img)
    
    print(f"Prepared {len(selected)} calibration images in {output_dir}")

if __name__ == "__main__":
    prepare_calibration_set()
