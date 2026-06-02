from ultralytics import YOLO
from pathlib import Path
from train import _write_resolved_dataset_config

ROOT_DIR = Path(__file__).resolve().parents[2]

config_dataset_path = ROOT_DIR / "configs" / "vessel.yaml"
model_path = ROOT_DIR / "models" / "train" / "vesselimg_nano_obb7" / "weights" / "best.pt"
# /home/acer/workspace/projects/amvit/models/train/vesselimg_nano_obb7/weights/
#print(config_dataset_path)
#print(model_path)

resolved_dataset_path = _write_resolved_dataset_config(config_dataset_path)
try:
    model = YOLO(model_path)
    results = model.val(data=str(resolved_dataset_path), split="test")
finally:
    resolved_dataset_path.unlink(missing_ok=True)

classes = ['cargo', 'military', 'carrier', 'cruise', 'tanker', 'ferry']

for i, cls in enumerate(classes):
    ap = results.box.ap[i] if hasattr(results.box, 'ap') else 0
    print(f"{cls:12s}: AP50-95 = {ap:.4f}")
