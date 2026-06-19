from roboflow import Roboflow
import yaml, os

def download_dataset(output_dir: str = "data/raw"):
    rf = Roboflow(api_key="e1f5705PG8tX1h9YI9z3") 
    project = rf.workspace("b-rubi").project("vesselimg")
    dataset = project.version(1).download("yolov8-obb", location=output_dir)

    print(f"Downloaded {count_images(output_dir)} images")
    return dataset

def count_images(root: str) -> dict:
    counts = {}
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(root, split, "images")
        counts[split] = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
    return counts

if __name__ == "__main__":
    download_dataset()
