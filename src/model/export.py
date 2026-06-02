from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_WEIGHTS_PATH = (
    ROOT_DIR / "models" / "train" / "vesselimg_nano_obb7" / "weights" / "best.pt"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "models" / "onnx"
DEFAULT_TRAIN_CONFIG_PATH = ROOT_DIR / "configs" / "train.yaml"

# Avoid Ultralytics/Matplotlib warnings when home config dirs are not writable.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("/tmp") / "ultralytics"))


def _resolve_path(path: str | os.PathLike[str], base_dir: Path = ROOT_DIR) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved_path = _resolve_path(path)
    with resolved_path.open() as f:
        return yaml.safe_load(f) or {}


def _default_imgsz(config_path: str | os.PathLike[str] = DEFAULT_TRAIN_CONFIG_PATH) -> int:
    cfg = _load_yaml(config_path)
    imgsz = cfg.get("imgsz", 640)
    if isinstance(imgsz, (list, tuple)):
        imgsz = imgsz[0]
    return int(imgsz)


def _validate_onnx(onnx_path: Path) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "The ONNX export finished, but validation requires the 'onnx' package. "
            "Install it in the vessel environment with: pip install onnx"
        ) from exc

    model_onnx = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_onnx)

    input_names = [input_tensor.name for input_tensor in model_onnx.graph.input]
    output_names = [output_tensor.name for output_tensor in model_onnx.graph.output]
    print(f"ONNX model validated: {onnx_path}")
    print(f"Inputs: {input_names}")
    print(f"Outputs: {output_names}")
    print(f"Graph nodes: {len(model_onnx.graph.node)}")


def _can_simplify_onnx() -> bool:
    return importlib.util.find_spec("onnxsim") is not None


def export_to_onnx(
    weights_path: str | os.PathLike[str] = DEFAULT_WEIGHTS_PATH,
    output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
    img_size: int | None = None,
    opset: int = 17,
    dynamic: bool = False,
    simplify: bool = True,
    half: bool = False,
    device: str | int = 0,
    batch: int = 1,
    validate: bool = True,
) -> Path:
    weights_path = _resolve_path(weights_path)
    output_dir = _resolve_path(output_dir)
    img_size = img_size if img_size is not None else _default_imgsz()

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if simplify and not _can_simplify_onnx():
        print(
            "onnxsim is not installed; exporting without graph simplification. "
            "Install it with 'pip install onnxsim' to enable simplify=True."
        )
        simplify = False

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=img_size,
            opset=opset,
            dynamic=dynamic,
            simplify=simplify,
            half=half,
            device=device,
            batch=batch,
        )
    ).resolve()

    destination_path = output_dir / exported_path.name
    if exported_path != destination_path:
        if destination_path.exists():
            destination_path.unlink()
        shutil.move(str(exported_path), str(destination_path))
    else:
        destination_path = exported_path

    if validate:
        _validate_onnx(destination_path)

    print(f"Exported ONNX model: {destination_path}")
    return destination_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the trained VESSELimg YOLOv8n OBB model to ONNX."
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Export image size. Defaults to configs/train.yaml imgsz.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_to_onnx(
        weights_path=args.weights,
        output_dir=args.output_dir,
        img_size=args.img_size,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=not args.no_simplify,
        half=args.half,
        device=args.device,
        batch=args.batch,
        validate=not args.no_validate,
    )
