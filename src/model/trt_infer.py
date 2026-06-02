from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_ENGINE_PATH = ROOT_DIR / "models" / "trt" / "vessel_int8.engine"
DEFAULT_DATA_CONFIG_PATH = ROOT_DIR / "configs" / "vessel.yaml"
DEFAULT_CLASS_NAMES = ["cargo", "military", "carrier", "cruise", "tanker", "ferry"]


def _resolve_path(path: str | os.PathLike[str], base_dir: Path = ROOT_DIR) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _import_cuda_and_tensorrt(verbose: bool = False) -> tuple[Any, Any, Any]:
    """
    Import PyCUDA before TensorRT.

    TensorRT builders/runtimes and PyCUDA must share the same CUDA context. The
    matching builder script follows the same order for INT8 calibration.
    """
    try:
        import pycuda.autoinit  # noqa: F401  # creates the CUDA context
        import pycuda.driver as cuda
    except ImportError as exc:
        raise RuntimeError(
            "PyCUDA is required for TensorRT inference. Activate the vessel "
            "environment and ensure pycuda==2023.1 is installed."
        ) from exc

    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT Python bindings are required for inference. Activate the "
            "vessel environment and ensure tensorrt==8.6.1 is installed."
        ) from exc

    severity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    return cuda, trt, trt.Logger(severity)


def load_class_names(config_path: str | os.PathLike[str] = DEFAULT_DATA_CONFIG_PATH) -> list[str]:
    config_path = _resolve_path(config_path)
    if not config_path.exists():
        return DEFAULT_CLASS_NAMES.copy()

    with config_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    names = cfg.get("names")
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return DEFAULT_CLASS_NAMES.copy()


@dataclass
class HostDeviceBuffer:
    name: str
    index: int
    host: np.ndarray
    device: Any
    shape: tuple[int, ...]
    dtype: np.dtype
    is_input: bool


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    pad_left: int
    pad_top: int
    input_hw: tuple[int, int]
    original_hw: tuple[int, int]


def _letterbox_bgr(frame: np.ndarray, target_hw: tuple[int, int]) -> tuple[np.ndarray, LetterboxMeta]:
    target_h, target_w = target_hw
    orig_h, orig_w = frame.shape[:2]
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError("Invalid input frame dimensions")

    scale = min(target_w / orig_w, target_h / orig_h)
    resized_w = int(round(orig_w * scale))
    resized_h = int(round(orig_h * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_left = (target_w - resized_w) // 2
    pad_top = (target_h - resized_h) // 2
    canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized

    return canvas, LetterboxMeta(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        input_hw=(target_h, target_w),
        original_hw=(orig_h, orig_w),
    )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * percentile))
    return float(sorted_values[index])


class TRTVesselDetector:
    """
    TensorRT inference wrapper for the YOLOv8n OBB vessel detector.

    The exported OBB head produces `[x, y, w, h, cls0..clsN, angle]`, so the
    current 6-class model has output shape `[1, 11, 8400]`.
    """

    def __init__(
        self,
        engine_path: str | os.PathLike[str] = DEFAULT_ENGINE_PATH,
        conf_thresh: float = 0.4,
        iou_thresh: float = 0.5,
        class_names: Sequence[str] | None = None,
        max_det: int = 300,
        verbose: bool = False,
    ) -> None:
        self.engine_path = _resolve_path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.class_names = list(class_names) if class_names is not None else DEFAULT_CLASS_NAMES.copy()
        self.max_det = int(max_det)

        self.cuda, self.trt, logger = _import_cuda_and_tensorrt(verbose=verbose)
        self.runtime = self.trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.stream = self.cuda.Stream()
        self.inputs: list[HostDeviceBuffer] = []
        self.outputs: list[HostDeviceBuffer] = []
        self.bindings: list[int] = []
        self._allocate_bindings()

        if len(self.inputs) != 1:
            raise RuntimeError(f"Expected one input tensor, found {len(self.inputs)}")
        if len(self.outputs) != 1:
            raise RuntimeError(f"Expected one output tensor, found {len(self.outputs)}")

        self.input = self.inputs[0]
        self.output = self.outputs[0]
        self.input_shape = self.input.shape
        self.output_shape = self.output.shape
        self.batch, self.channels, self.input_h, self.input_w = self._validate_input_shape(self.input_shape)

        output_channels = self.output_shape[1] if len(self.output_shape) == 3 else None
        expected_channels = 4 + len(self.class_names) + 1
        if output_channels is not None and output_channels != expected_channels:
            inferred_classes = output_channels - 5
            if inferred_classes <= 0:
                raise RuntimeError(f"Unexpected OBB output shape: {self.output_shape}")
            self.class_names = [str(i) for i in range(inferred_classes)]

        print(
            "TRT engine loaded: "
            f"input={self.input_shape}, output={self.output_shape}, "
            f"classes={len(self.class_names)}"
        )

    def _allocate_bindings(self) -> None:
        if not hasattr(self.engine, "num_bindings"):
            raise RuntimeError(
                "This wrapper currently expects TensorRT binding APIs available in TensorRT 8.x."
            )

        self.bindings = [0] * int(self.engine.num_bindings)
        if hasattr(self.engine, "num_io_tensors"):
            tensor_items = [
                (i, self.engine.get_tensor_name(i)) for i in range(int(self.engine.num_io_tensors))
            ]
        else:
            tensor_items = [
                (i, self.engine.get_binding_name(i)) for i in range(self.engine.num_bindings)
            ]

        for index, name in tensor_items:
            if hasattr(self.engine, "get_tensor_mode"):
                is_input = self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT
                shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
                dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
            else:
                is_input = bool(self.engine.binding_is_input(index))
                shape = tuple(int(dim) for dim in self.engine.get_binding_shape(index))
                dtype = np.dtype(self.trt.nptype(self.engine.get_binding_dtype(index)))

            if any(dim <= 0 for dim in shape):
                raise RuntimeError(
                    f"Dynamic tensor shape is not supported by this wrapper: {name} {shape}"
                )

            size = int(np.prod(shape))
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            self.bindings[index] = int(device_mem)

            buffer = HostDeviceBuffer(
                name=name,
                index=index,
                host=host_mem,
                device=device_mem,
                shape=shape,
                dtype=dtype,
                is_input=is_input,
            )
            if is_input:
                self.inputs.append(buffer)
            else:
                self.outputs.append(buffer)

    @staticmethod
    def _validate_input_shape(shape: Sequence[int]) -> tuple[int, int, int, int]:
        if len(shape) != 4:
            raise RuntimeError(f"Expected NCHW input shape, got {tuple(shape)}")
        batch, channels, height, width = (int(dim) for dim in shape)
        if batch != 1:
            raise RuntimeError(f"This wrapper expects batch size 1, got {batch}")
        if channels != 3:
            raise RuntimeError(f"This wrapper expects 3-channel input, got {channels}")
        return batch, channels, height, width

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, LetterboxMeta]:
        if frame is None:
            raise ValueError("Input frame is None")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected BGR image with shape HxWx3, got {frame.shape}")

        padded, meta = _letterbox_bgr(frame, (self.input_h, self.input_w))
        image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)[np.newaxis, ...]
        return np.ascontiguousarray(image, dtype=self.input.dtype), meta

    def infer(
        self,
        frame: np.ndarray,
        return_raw: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], np.ndarray]:
        blob, meta = self.preprocess(frame)
        np.copyto(self.input.host, blob.ravel())

        for inp in self.inputs:
            self.cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        ok = self.context.execute_async_v2(self.bindings, self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed")

        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()

        raw = self.output.host.reshape(self.output.shape)
        detections = self.postprocess(raw, meta)
        if return_raw:
            return detections, raw.copy()
        return detections

    def infer_path(self, image_path: str | os.PathLike[str]) -> list[dict[str, Any]]:
        image_path = _resolve_path(image_path)
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Failed to read image: {image_path}")
        return self.infer(frame)

    def postprocess(self, raw: np.ndarray, meta: LetterboxMeta) -> list[dict[str, Any]]:
        if raw.ndim != 3 or raw.shape[0] != 1:
            raise RuntimeError(f"Expected raw output shape [1, C, N], got {raw.shape}")

        predictions = raw[0].T.astype(np.float32, copy=False)
        num_classes = len(self.class_names)
        expected_channels = 4 + num_classes + 1
        if predictions.shape[1] != expected_channels:
            num_classes = predictions.shape[1] - 5
            if num_classes <= 0:
                raise RuntimeError(f"Unexpected OBB output channel count: {predictions.shape[1]}")

        boxes = predictions[:, :4].copy()
        class_scores = predictions[:, 4 : 4 + num_classes]
        angles = predictions[:, 4 + num_classes]

        confidences = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1).astype(np.int32)
        keep_mask = (confidences >= self.conf_thresh) & (boxes[:, 2] > 0.0) & (boxes[:, 3] > 0.0)

        if not np.any(keep_mask):
            return []

        boxes = boxes[keep_mask]
        angles = angles[keep_mask]
        confidences = confidences[keep_mask]
        class_ids = class_ids[keep_mask]

        keep = self._rotated_nms(boxes, angles, confidences, class_ids)
        if not keep:
            return []

        detections: list[dict[str, Any]] = []
        orig_h, orig_w = meta.original_hw
        for index in keep:
            x, y, width, height = boxes[index]
            angle = angles[index]

            x = (x - meta.pad_left) / meta.scale
            y = (y - meta.pad_top) / meta.scale
            width = width / meta.scale
            height = height / meta.scale

            xywhr = (
                float(x),
                float(y),
                float(width),
                float(height),
                float(angle),
            )
            points = self.xywhr_to_points(xywhr)
            points[:, 0] = np.clip(points[:, 0], 0, orig_w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, orig_h - 1)

            cls_id = int(class_ids[index])
            detections.append(
                {
                    "xywhr": xywhr,
                    "points": points.astype(float).round(2).tolist(),
                    "conf": float(confidences[index]),
                    "cls": cls_id,
                    "name": self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id),
                }
            )

        return detections

    def _rotated_nms(
        self,
        boxes_xywh: np.ndarray,
        angles_rad: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> list[int]:
        selected: list[int] = []

        for cls_id in np.unique(class_ids):
            class_indices = np.where(class_ids == cls_id)[0]
            rotated_rects = [
                (
                    (float(boxes_xywh[i, 0]), float(boxes_xywh[i, 1])),
                    (max(float(boxes_xywh[i, 2]), 1e-3), max(float(boxes_xywh[i, 3]), 1e-3)),
                    float(np.degrees(angles_rad[i])),
                )
                for i in class_indices
            ]
            class_scores = [float(scores[i]) for i in class_indices]
            kept = cv2.dnn.NMSBoxesRotated(
                rotated_rects,
                class_scores,
                self.conf_thresh,
                self.iou_thresh,
            )
            if len(kept) == 0:
                continue
            kept = np.array(kept).reshape(-1)
            selected.extend(int(class_indices[i]) for i in kept)

        selected.sort(key=lambda i: float(scores[i]), reverse=True)
        return selected[: self.max_det]

    @staticmethod
    def xywhr_to_points(xywhr: Sequence[float]) -> np.ndarray:
        x, y, width, height, angle = xywhr
        rect = ((float(x), float(y)), (float(width), float(height)), float(np.degrees(angle)))
        return cv2.boxPoints(rect)

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: Sequence[dict[str, Any]],
    ) -> np.ndarray:
        annotated = frame.copy()
        for det in detections:
            cls_id = int(det["cls"])
            color = (
                int((37 * cls_id + 80) % 255),
                int((17 * cls_id + 180) % 255),
                int((97 * cls_id + 40) % 255),
            )
            points = np.asarray(det["points"], dtype=np.int32)
            cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=2)

            x, y = points[0]
            label = f'{det["name"]} {det["conf"]:.2f}'
            cv2.putText(
                annotated,
                label,
                (int(x), max(15, int(y) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return annotated

    def benchmark(
        self,
        n_runs: int = 200,
        warmup: int = 20,
        frame: np.ndarray | None = None,
        dummy_shape: tuple[int, int] = (720, 1280),
    ) -> dict[str, float]:
        if n_runs <= 0:
            raise ValueError("n_runs must be greater than zero")
        if warmup < 0:
            raise ValueError("warmup must be non-negative")

        if frame is None:
            rng = np.random.default_rng(0)
            frame = rng.integers(0, 256, (*dummy_shape, 3), dtype=np.uint8)

        for _ in range(warmup):
            self.infer(frame)

        latencies: list[float] = []
        for _ in range(n_runs):
            start = time.perf_counter()
            self.infer(frame)
            latencies.append((time.perf_counter() - start) * 1000.0)

        sorted_latencies = sorted(latencies)
        mean_ms = float(sum(latencies) / len(latencies))
        return {
            "runs": float(n_runs),
            "warmup": float(warmup),
            "min_ms": float(sorted_latencies[0]),
            "mean_ms": mean_ms,
            "p50_ms": _percentile(sorted_latencies, 0.50),
            "p90_ms": _percentile(sorted_latencies, 0.90),
            "p95_ms": _percentile(sorted_latencies, 0.95),
            "p99_ms": _percentile(sorted_latencies, 0.99),
            "fps": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
        }


def _load_image(path: str | os.PathLike[str]) -> np.ndarray:
    image_path = _resolve_path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TensorRT inference and benchmarking for the VESSELimg YOLOv8n OBB engine."
    )
    parser.add_argument("--engine", default=str(DEFAULT_ENGINE_PATH), help="TensorRT engine path.")
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG_PATH), help="Dataset YAML for class names.")
    parser.add_argument("--image", default=None, help="Optional image path for inference or image-based benchmark.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="Rotated NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections after NMS.")
    parser.add_argument("--save-vis", default=None, help="Optional path to save annotated image output.")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark.")
    parser.add_argument("--runs", type=int, default=200, help="Benchmark measured runs.")
    parser.add_argument("--warmup", type=int, default=20, help="Benchmark warmup runs.")
    parser.add_argument(
        "--dummy-shape",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=(720, 1280),
        help="Dummy benchmark frame shape when --image is not provided.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose TensorRT logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = load_class_names(args.data_config)
    detector = TRTVesselDetector(
        engine_path=args.engine,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        class_names=class_names,
        max_det=args.max_det,
        verbose=args.verbose,
    )

    frame = _load_image(args.image) if args.image else None

    if frame is not None:
        detections = detector.infer(frame)
        print(json.dumps({"detections": detections}, indent=2))

        if args.save_vis:
            output_path = _resolve_path(args.save_vis)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            annotated = detector.draw_detections(frame, detections)
            cv2.imwrite(str(output_path), annotated)
            print(f"Annotated image saved: {output_path}")

    if args.benchmark or frame is None:
        stats = detector.benchmark(
            n_runs=args.runs,
            warmup=args.warmup,
            frame=frame,
            dummy_shape=tuple(args.dummy_shape),
        )
        print(json.dumps({"benchmark": stats}, indent=2))


if __name__ == "__main__":
    main()
