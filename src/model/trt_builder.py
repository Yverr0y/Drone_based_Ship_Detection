from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_ONNX_PATH = ROOT_DIR / "models" / "onnx" / "best.onnx"
DEFAULT_ENGINE_PATH = ROOT_DIR / "models" / "trt" / "vessel_int8.engine"
DEFAULT_CALIBRATION_DIR = ROOT_DIR / "data" / "calibration"
DEFAULT_CALIBRATION_CACHE = ROOT_DIR / "models" / "trt" / "calibration.cache"
DEFAULT_INPUT_SHAPE = (1, 3, 640, 640)

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def _resolve_path(path: str | os.PathLike[str], base_dir: Path = ROOT_DIR) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _import_tensorrt(verbose: bool = False) -> tuple[Any, Any]:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT Python bindings are not installed in this environment. "
            "Run this from the Jetson/TensorRT environment that provides "
            "'import tensorrt'."
        ) from exc

    severity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    return trt, trt.Logger(severity)


def _import_pycuda() -> Any:
    try:
        import pycuda.autoinit  # noqa: F401  # creates the CUDA context
        import pycuda.driver as cuda
    except ImportError as exc:
        raise RuntimeError(
            "PyCUDA is required for TensorRT INT8 entropy calibration. "
            "Install pycuda in the vessel environment before building INT8."
        ) from exc

    return cuda


def _iter_image_paths(
    calibration_dir: Path,
    max_images: int | None,
    recursive: bool,
) -> list[Path]:
    if not calibration_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {calibration_dir}")
    if not calibration_dir.is_dir():
        raise NotADirectoryError(f"Calibration path is not a directory: {calibration_dir}")

    iterator = calibration_dir.rglob("*") if recursive else calibration_dir.iterdir()
    image_paths = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]

    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in {calibration_dir}")

    return image_paths


def _letterbox_bgr(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Invalid image dimensions")

    scale = min(target_w / width, target_h / height)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    top = (target_h - resized_h) // 2
    left = (target_w - resized_w) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def preprocess_image(
    image_path: Path,
    input_hw: tuple[int, int],
    resize_mode: str = "letterbox",
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    input_h, input_w = input_hw
    if resize_mode == "letterbox":
        image = _letterbox_bgr(image, (input_h, input_w))
    elif resize_mode == "stretch":
        image = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(f"Unsupported resize mode: {resize_mode}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    return np.ascontiguousarray(image, dtype=np.float32)


def _validate_nchw_shape(shape: Sequence[int], label: str) -> tuple[int, int, int, int]:
    if len(shape) != 4:
        raise ValueError(f"{label} must be NCHW with four dimensions, got {tuple(shape)}")

    resolved = tuple(int(dim) for dim in shape)
    if any(dim <= 0 for dim in resolved):
        raise ValueError(f"{label} dimensions must be positive, got {resolved}")
    if resolved[1] != 3:
        raise ValueError(f"{label} must have 3 input channels, got {resolved[1]}")

    return resolved


def _resolve_network_input_shape(
    network_shape: Sequence[int],
    requested_shape: Sequence[int] | None,
) -> tuple[int, int, int, int]:
    network_shape = tuple(int(dim) for dim in network_shape)
    if len(network_shape) != 4:
        raise ValueError(f"Expected a 4D NCHW ONNX input, got {network_shape}")

    requested = (
        _validate_nchw_shape(requested_shape, "requested input shape")
        if requested_shape is not None
        else DEFAULT_INPUT_SHAPE
    )

    resolved: list[int] = []
    for index, network_dim in enumerate(network_shape):
        if network_dim > 0:
            if requested_shape is not None and requested[index] != network_dim:
                raise ValueError(
                    "Requested input shape does not match the static ONNX input: "
                    f"requested={requested}, onnx={network_shape}"
                )
            resolved.append(network_dim)
        else:
            resolved.append(requested[index])

    return _validate_nchw_shape(resolved, "resolved input shape")


def _set_workspace_limit(config: Any, trt: Any, workspace_gb: float) -> None:
    workspace_bytes = int(workspace_gb * (1 << 30))
    if workspace_bytes <= 0:
        raise ValueError("workspace_gb must be greater than zero")

    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes


def _build_serialized_engine(builder: Any, network: Any, config: Any) -> Any:
    if hasattr(builder, "build_serialized_network"):
        return builder.build_serialized_network(network, config)

    engine = builder.build_engine(network, config)
    if engine is None:
        return None
    return engine.serialize()


def _is_shape_dynamic(shape: Sequence[int]) -> bool:
    return any(int(dim) < 0 for dim in shape)


def _shape_with_batch(shape: Sequence[int], batch_size: int) -> tuple[int, int, int, int]:
    _, channels, height, width = _validate_nchw_shape(shape, "input shape")
    return _validate_nchw_shape((batch_size, channels, height, width), "batched input shape")


def _validate_profile_shapes(
    min_shape: Sequence[int],
    opt_shape: Sequence[int],
    max_shape: Sequence[int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    min_shape = _validate_nchw_shape(min_shape, "min profile shape")
    opt_shape = _validate_nchw_shape(opt_shape, "opt profile shape")
    max_shape = _validate_nchw_shape(max_shape, "max profile shape")

    for axis, (min_dim, opt_dim, max_dim) in enumerate(zip(min_shape, opt_shape, max_shape)):
        if not (min_dim <= opt_dim <= max_dim):
            raise ValueError(
                "Optimization profile shapes must satisfy min <= opt <= max on each axis; "
                f"axis={axis}, min={min_shape}, opt={opt_shape}, max={max_shape}"
            )

    return min_shape, opt_shape, max_shape


def _create_entropy_calibrator_class(trt: Any) -> type:
    class VesselEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        """
        Entropy calibrator for YOLOv8 OBB inputs.

        The calibrator streams images from disk to avoid keeping all calibration
        tensors in host memory. Preprocessing mirrors Ultralytics inference:
        BGR image read, letterbox or stretch resize to the exported input size,
        RGB conversion, float32 normalization to [0, 1], and NCHW layout.
        """

        def __init__(
            self,
            image_paths: Sequence[Path],
            cache_file: Path,
            batch_size: int,
            input_chw: tuple[int, int, int],
            cuda: Any,
            resize_mode: str,
            use_cache: bool,
        ) -> None:
            super().__init__()

            if batch_size <= 0:
                raise ValueError("Calibration batch size must be greater than zero")
            if len(image_paths) < batch_size:
                raise ValueError(
                    "Not enough calibration images for one full batch: "
                    f"images={len(image_paths)}, batch_size={batch_size}"
                )

            channels, height, width = input_chw
            if channels != 3:
                raise ValueError(f"Expected 3-channel input, got {channels}")

            self.image_paths = list(image_paths)
            self.cache_file = cache_file
            self.batch_size = int(batch_size)
            self.input_chw = (int(channels), int(height), int(width))
            self.cuda = cuda
            self.resize_mode = resize_mode
            self.use_cache = use_cache
            self.current_index = 0
            self.total_batches = len(self.image_paths) // self.batch_size

            input_bytes = (
                self.batch_size
                * int(np.prod(self.input_chw))
                * np.dtype(np.float32).itemsize
            )
            self.device_input = self.cuda.mem_alloc(input_bytes)

            print(
                "Calibrator: "
                f"{len(self.image_paths)} images, "
                f"batch={self.batch_size}, "
                f"full_batches={self.total_batches}, "
                f"input=({self.batch_size}, {channels}, {height}, {width}), "
                f"resize={self.resize_mode}"
            )

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names: Sequence[str]) -> list[int] | None:
            del names

            batch = np.empty(
                (self.batch_size, *self.input_chw),
                dtype=np.float32,
            )
            filled = 0

            while filled < self.batch_size and self.current_index < len(self.image_paths):
                image_path = self.image_paths[self.current_index]
                self.current_index += 1
                try:
                    batch[filled] = preprocess_image(
                        image_path=image_path,
                        input_hw=(self.input_chw[1], self.input_chw[2]),
                        resize_mode=self.resize_mode,
                    )
                    filled += 1
                except ValueError as exc:
                    print(f"Skipping calibration image: {exc}")

            if filled < self.batch_size:
                return None

            batch = np.ascontiguousarray(batch)
            self.cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]

        def read_calibration_cache(self) -> bytes | None:
            if self.use_cache and self.cache_file.exists():
                print(f"Using calibration cache: {self.cache_file}")
                return self.cache_file.read_bytes()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_bytes(cache)
            print(f"Calibration cache written: {self.cache_file}")

    return VesselEntropyCalibrator


def build_trt_engine(
    onnx_path: str | os.PathLike[str] = DEFAULT_ONNX_PATH,
    engine_path: str | os.PathLike[str] = DEFAULT_ENGINE_PATH,
    calibration_dir: str | os.PathLike[str] = DEFAULT_CALIBRATION_DIR,
    precision: str = "int8",
    workspace_gb: float = 4.0,
    input_shape: Sequence[int] | None = None,
    calibration_batch_size: int | None = None,
    calibration_cache: str | os.PathLike[str] = DEFAULT_CALIBRATION_CACHE,
    max_calibration_images: int | None = 1000,
    resize_mode: str = "letterbox",
    recursive_calibration: bool = False,
    use_calibration_cache: bool = True,
    fp16_fallback: bool = True,
    min_shape: Sequence[int] | None = None,
    opt_shape: Sequence[int] | None = None,
    max_shape: Sequence[int] | None = None,
    verbose: bool = False,
) -> Path:
    """
    Build a TensorRT engine from the exported YOLOv8 OBB ONNX model.

    The current project export is static NCHW [1, 3, 640, 640], so INT8
    calibration defaults to batch 1. For future dynamic-batch exports, pass
    --input-shape and optional profile shapes.
    """
    onnx_path = _resolve_path(onnx_path)
    engine_path = _resolve_path(engine_path)
    calibration_dir = _resolve_path(calibration_dir)
    calibration_cache = _resolve_path(calibration_cache)

    precision = precision.lower()
    if precision not in {"fp32", "fp16", "int8"}:
        raise ValueError("precision must be one of: fp32, fp16, int8")
    if resize_mode not in {"letterbox", "stretch"}:
        raise ValueError("resize_mode must be either 'letterbox' or 'stretch'")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    # TensorRT builders hold CUDA resources. For INT8 calibration, create the
    # PyCUDA context before importing/building TensorRT so both use one context.
    cuda = _import_pycuda() if precision == "int8" else None
    trt, logger = _import_tensorrt(verbose=verbose)

    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    _set_workspace_limit(config=config, trt=trt, workspace_gb=workspace_gb)

    print(f"TensorRT version: {getattr(trt, '__version__', 'unknown')}")
    print(f"Parsing ONNX: {onnx_path}")
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parsing failed:\n{errors}")

    if network.num_inputs != 1:
        raise RuntimeError(f"Expected one ONNX input, found {network.num_inputs}")

    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    network_shape = tuple(int(dim) for dim in input_tensor.shape)
    resolved_shape = _resolve_network_input_shape(network_shape, input_shape)
    dynamic_shape = _is_shape_dynamic(network_shape)

    print(f"Network input:  {input_name} {network_shape}")
    print(
        "Network outputs: "
        + ", ".join(
            f"{network.get_output(i).name} {tuple(int(dim) for dim in network.get_output(i).shape)}"
            for i in range(network.num_outputs)
        )
    )
    print(f"Resolved input shape: {resolved_shape}")

    min_profile_shape = opt_profile_shape = max_profile_shape = None
    if dynamic_shape:
        min_profile_shape, opt_profile_shape, max_profile_shape = _validate_profile_shapes(
            min_shape or _shape_with_batch(resolved_shape, 1),
            opt_shape or resolved_shape,
            max_shape or resolved_shape,
        )
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, min_profile_shape, opt_profile_shape, max_profile_shape)
        config.add_optimization_profile(profile)
        print(
            "Optimization profile: "
            f"min={min_profile_shape}, opt={opt_profile_shape}, max={max_profile_shape}"
        )

    if precision == "fp16":
        if getattr(builder, "platform_has_fast_fp16", False):
            config.set_flag(trt.BuilderFlag.FP16)
            print("Precision: FP16")
        else:
            print("WARNING: FP16 is not fast on this platform; building FP32.")
    elif precision == "int8":
        if not getattr(builder, "platform_has_fast_int8", False):
            raise RuntimeError("INT8 is not supported efficiently on this TensorRT platform.")

        config.set_flag(trt.BuilderFlag.INT8)
        print("Precision: INT8 entropy calibration")

        if fp16_fallback:
            if getattr(builder, "platform_has_fast_fp16", False):
                config.set_flag(trt.BuilderFlag.FP16)
                print("FP16 fallback: enabled")
            else:
                print("FP16 fallback requested but not fast on this platform; using FP32 fallback.")

        calibration_batch_size = calibration_batch_size or resolved_shape[0]
        calibration_shape = _shape_with_batch(resolved_shape, calibration_batch_size)
        if not dynamic_shape and calibration_shape != resolved_shape:
            raise ValueError(
                "The ONNX input has a static batch dimension. "
                f"Calibration batch must be {resolved_shape[0]}, got {calibration_batch_size}. "
                "Re-export ONNX with a larger or dynamic batch to calibrate with larger batches."
            )
        if dynamic_shape and not all(
            min_dim <= cal_dim <= max_dim
            for min_dim, cal_dim, max_dim in zip(
                min_profile_shape,
                calibration_shape,
                max_profile_shape,
            )
        ):
            raise ValueError(
                "Calibration shape must fit inside the optimization profile: "
                f"calibration={calibration_shape}, "
                f"min={min_profile_shape}, max={max_profile_shape}"
            )

        image_paths = _iter_image_paths(
            calibration_dir=calibration_dir,
            max_images=max_calibration_images,
            recursive=recursive_calibration,
        )
        calibrator_cls = _create_entropy_calibrator_class(trt)
        calibrator = calibrator_cls(
            image_paths=image_paths,
            cache_file=calibration_cache,
            batch_size=calibration_batch_size,
            input_chw=calibration_shape[1:],
            cuda=cuda,
            resize_mode=resize_mode,
            use_cache=use_calibration_cache,
        )
        config.int8_calibrator = calibrator

        if dynamic_shape and hasattr(config, "set_calibration_profile"):
            calibration_profile = builder.create_optimization_profile()
            calibration_profile.set_shape(
                input_name,
                calibration_shape,
                calibration_shape,
                calibration_shape,
            )
            config.set_calibration_profile(calibration_profile)
            print(f"Calibration profile: {calibration_shape}")
    else:
        print("Precision: FP32")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building TensorRT engine: {engine_path}")
    serialized_engine = _build_serialized_engine(builder, network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed.")

    engine_path.write_bytes(bytes(serialized_engine))
    size_mb = engine_path.stat().st_size / (1 << 20)
    print(f"Engine saved: {engine_path} ({size_mb:.1f} MB)")
    return engine_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine for the VESSELimg YOLOv8n OBB ONNX model."
    )
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX_PATH), help="Input ONNX model path.")
    parser.add_argument(
        "--engine",
        default=str(DEFAULT_ENGINE_PATH),
        help="Output TensorRT engine path.",
    )
    parser.add_argument(
        "--calibration-dir",
        default=str(DEFAULT_CALIBRATION_DIR),
        help="Directory containing representative calibration images.",
    )
    parser.add_argument(
        "--calibration-cache",
        default=str(DEFAULT_CALIBRATION_CACHE),
        help="Path to the TensorRT INT8 calibration cache.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "int8"),
        default="int8",
        help="Engine precision mode.",
    )
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument(
        "--input-shape",
        nargs=4,
        type=int,
        metavar=("N", "C", "H", "W"),
        default=None,
        help="NCHW shape used to resolve dynamic ONNX dimensions.",
    )
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=None,
        help="INT8 calibration batch size. Defaults to the resolved ONNX batch.",
    )
    parser.add_argument(
        "--max-calibration-images",
        type=int,
        default=1000,
        help="Maximum number of calibration images to use; pass 0 to use all.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=("letterbox", "stretch"),
        default="letterbox",
        help="Calibration preprocessing resize mode.",
    )
    parser.add_argument(
        "--recursive-calibration",
        action="store_true",
        help="Search calibration-dir recursively.",
    )
    parser.add_argument(
        "--force-calibration",
        action="store_true",
        help="Ignore an existing calibration cache and recalibrate.",
    )
    parser.add_argument(
        "--no-fp16-fallback",
        action="store_true",
        help="Do not enable FP16 fallback while building INT8.",
    )
    parser.add_argument(
        "--min-shape",
        nargs=4,
        type=int,
        metavar=("N", "C", "H", "W"),
        default=None,
        help="Dynamic ONNX optimization profile min shape.",
    )
    parser.add_argument(
        "--opt-shape",
        nargs=4,
        type=int,
        metavar=("N", "C", "H", "W"),
        default=None,
        help="Dynamic ONNX optimization profile opt shape.",
    )
    parser.add_argument(
        "--max-shape",
        nargs=4,
        type=int,
        metavar=("N", "C", "H", "W"),
        default=None,
        help="Dynamic ONNX optimization profile max shape.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose TensorRT logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_calibration_images = (
        None if args.max_calibration_images == 0 else args.max_calibration_images
    )

    build_trt_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        calibration_dir=args.calibration_dir,
        precision=args.precision,
        workspace_gb=args.workspace_gb,
        input_shape=args.input_shape,
        calibration_batch_size=args.calibration_batch_size,
        calibration_cache=args.calibration_cache,
        max_calibration_images=max_calibration_images,
        resize_mode=args.resize_mode,
        recursive_calibration=args.recursive_calibration,
        use_calibration_cache=not args.force_calibration,
        fp16_fallback=not args.no_fp16_fallback,
        min_shape=args.min_shape,
        opt_shape=args.opt_shape,
        max_shape=args.max_shape,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
