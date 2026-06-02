from ultralytics import YOLO
from ultralytics.utils import SETTINGS
import mlflow
import yaml
import time
import os
import re
from numbers import Real
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent

config_train_path = ROOT_DIR / "configs" / "train.yaml"
config_dataset_path = ROOT_DIR / "configs" / "vessel.yaml"
model_weights_path = MODEL_DIR / "yolov8n-obb.pt"
MLFLOW_NAME_RE = re.compile(r"[^A-Za-z0-9_\-./ :]")


def _resolve_path(path: str | os.PathLike, base_dir: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_yaml(path: str | os.PathLike) -> tuple[Path, dict]:
    path = _resolve_path(path, ROOT_DIR)
    with path.open() as f:
        return path, yaml.safe_load(f) or {}


def _resolve_existing_relative_path(path: str | os.PathLike, *base_dirs: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()

    for base_dir in base_dirs:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate

    return (base_dirs[0] / path).resolve()


def _prepare_training_config(config_path: str | os.PathLike) -> dict:
    _, cfg = _load_yaml(config_path)

    if cfg.get("project"):
        cfg["project"] = str(_resolve_existing_relative_path(cfg["project"], ROOT_DIR))

    return cfg


def _write_resolved_dataset_config(dataset_path: str | os.PathLike) -> Path:
    dataset_path, dataset_cfg = _load_yaml(dataset_path)

    if dataset_cfg.get("path"):
        dataset_cfg["path"] = str(
            _resolve_existing_relative_path(dataset_cfg["path"], ROOT_DIR, dataset_path.parent)
        )
    else:
        dataset_cfg["path"] = str(dataset_path.parent)

    with NamedTemporaryFile("w", suffix=".yaml", prefix="vessel_dataset_", delete=False) as f:
        yaml.safe_dump(dataset_cfg, f, sort_keys=False)
        return Path(f.name)


def _sanitize_mlflow_name(name: str) -> str:
    return MLFLOW_NAME_RE.sub("", name) or "metric"


def train_vessel_detector(
    config_path: str | os.PathLike = config_train_path,
    dataset_path: str | os.PathLike = config_dataset_path,
    resume: bool = False
):
    cfg = _prepare_training_config(config_path)
    resolved_dataset_path = _write_resolved_dataset_config(dataset_path)

    # Keep MLflow logging owned by this script; Ultralytics also has an MLflow callback.
    SETTINGS["mlflow"] = False

    # Load OBB model with ImageNet/COCO pretrained nano weights.
    model_path = model_weights_path if model_weights_path.exists() else "yolov8n-obb.pt"
    model = YOLO(str(model_path))

    mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
    mlflow.set_experiment("vessel-tracking")
    try:
        with mlflow.start_run(run_name=f"yolov8n-obb-{int(time.time())}"):
            mlflow.log_params(cfg)
            mlflow.log_param("dataset_config", str(_resolve_path(dataset_path, ROOT_DIR)))

            results = model.train(
                data=str(resolved_dataset_path),
                **cfg,
                resume=resume
            )

            # Log final metrics.
            metrics = results.results_dict if hasattr(results, "results_dict") else {}
            for k, v in metrics.items():
                if isinstance(v, Real):
                    mlflow.log_metric(_sanitize_mlflow_name(k), float(v))

            trainer = getattr(model, "trainer", None)
            save_dir = getattr(trainer, "save_dir", None)
            if save_dir is None:
                save_dir = getattr(results, "save_dir", None)
            best_weights = getattr(trainer, "best", None)
            if best_weights is None or not Path(best_weights).exists():
                best_weights = getattr(trainer, "last", None)
            if save_dir:
                mlflow.log_param("save_dir", str(save_dir))

            print(f"\n--- Training Complete ---")
            print(f"mAP50:    {metrics.get('metrics/mAP50(B)', 0):.4f}")
            print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
            if best_weights and Path(best_weights).exists():
                print(f"Best weights: {best_weights}")
    finally:
        resolved_dataset_path.unlink(missing_ok=True)

    return results

if __name__ == "__main__":
    train_vessel_detector()
