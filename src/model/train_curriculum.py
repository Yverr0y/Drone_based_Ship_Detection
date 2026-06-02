from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import floor
from numbers import Real
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "curriculum.yaml"
DEFAULT_DATASET_CONFIG_PATH = ROOT_DIR / "configs" / "vessel.yaml"
DEFAULT_NANO_WEIGHTS = "yolov8n-obb.pt"
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

MLFLOW_NAME_RE = re.compile(r"[^A-Za-z0-9_./ :\\-]")
INTERPOLATED_KEYS = {
    "imgsz",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "mosaic",
    "mixup",
    "copy_paste",
    "box",
    "cls",
    "dfl",
    "freeze",
}
INTEGER_KEYS = {"imgsz", "freeze", "workers", "batch", "seed", "patience", "save_period"}

# Keep Ultralytics/Matplotlib from writing to unexpected home config locations.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("/tmp") / "ultralytics"))


@dataclass(frozen=True)
class CurriculumSegment:
    index: int
    phase_index: int
    phase_key: str
    phase_title: str
    phase_rationale: str
    start_epoch: int
    end_epoch: int
    epochs: int
    progress: float
    train_args: dict[str, Any] = field(default_factory=dict)


def _resolve_path(path: str | os.PathLike[str], base_dir: Path = ROOT_DIR) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _load_yaml(path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    resolved = _resolve_path(path)
    with resolved.open(encoding="utf-8") as f:
        return resolved, yaml.safe_load(f) or {}


def _write_yaml(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False)


def _resolve_existing_relative_path(path: str | os.PathLike[str], *base_dirs: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()

    for base_dir in base_dirs:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate

    return (base_dirs[0] / path).resolve()


def _sanitize_mlflow_name(name: str) -> str:
    return MLFLOW_NAME_RE.sub("", name) or "metric"


def _sanitize_run_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "curriculum_segment"


def _select_model_weights(cfg: Mapping[str, Any], override: str | os.PathLike[str] | None = None) -> str:
    if override:
        resolved = _resolve_existing_relative_path(override, ROOT_DIR, MODEL_DIR)
        return str(resolved) if resolved.exists() else str(override)

    model_cfg = cfg.get("model", {}) or {}
    for candidate in model_cfg.get("local_candidates", []) or []:
        resolved = _resolve_existing_relative_path(candidate, ROOT_DIR, MODEL_DIR)
        if resolved.exists():
            return str(resolved)

    fallback = model_cfg.get("weights", DEFAULT_NANO_WEIGHTS)
    resolved = _resolve_existing_relative_path(fallback, ROOT_DIR, MODEL_DIR)
    return str(resolved) if resolved.exists() else str(fallback)


def _write_resolved_dataset_config(
    dataset_path: str | os.PathLike[str],
    cfg: Mapping[str, Any],
) -> tuple[Path, list[Path], dict[str, Any] | None]:
    dataset_path, dataset_cfg = _load_yaml(dataset_path)
    dataset_root = dataset_cfg.get("path")
    if dataset_root:
        root = _resolve_existing_relative_path(dataset_root, ROOT_DIR, dataset_path.parent)
        dataset_cfg["path"] = str(root)
    else:
        root = dataset_path.parent.resolve()
        dataset_cfg["path"] = str(root)

    temp_files: list[Path] = []
    balance_info = None
    balanced_cfg = ((cfg.get("imbalance", {}) or {}).get("balanced_sampling", {}) or {})
    if balanced_cfg.get("enabled", False):
        train_list, balance_info = _write_balanced_train_list(dataset_cfg, root, balanced_cfg)
        dataset_cfg["train"] = str(train_list)
        temp_files.append(train_list)

    with NamedTemporaryFile("w", suffix=".yaml", prefix="vessel_curriculum_dataset_", delete=False) as f:
        yaml.safe_dump(dataset_cfg, f, sort_keys=False)
        dataset_yaml = Path(f.name)

    return dataset_yaml, temp_files, balance_info


def _iter_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Training image directory not found: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _label_classes(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []
    classes: list[int] = []
    with label_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                classes.append(int(float(parts[0])))
            except ValueError:
                continue
    return classes


def _write_balanced_train_list(
    dataset_cfg: Mapping[str, Any],
    root: Path,
    balanced_cfg: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    train_entry = Path(dataset_cfg["train"])
    image_dir = train_entry if train_entry.is_absolute() else root / train_entry
    label_dir = image_dir.parent / "labels"
    images = _iter_images(image_dir)

    image_classes: dict[Path, set[int]] = {}
    class_counts: Counter[int] = Counter()
    for image_path in images:
        classes = set(_label_classes(label_dir / f"{image_path.stem}.txt"))
        image_classes[image_path] = classes
        class_counts.update(classes)

    positive_counts = [count for count in class_counts.values() if count > 0]
    max_count = max(positive_counts) if positive_counts else 1
    exponent = float(balanced_cfg.get("exponent", 0.5))
    max_repeat = max(1, int(balanced_cfg.get("max_repeat", 6)))
    include_empty = bool(balanced_cfg.get("include_empty", True))

    expanded: list[Path] = []
    repeat_histogram: Counter[int] = Counter()
    class_repeat_totals: defaultdict[int, int] = defaultdict(int)
    for image_path in images:
        classes = image_classes[image_path]
        if not classes and not include_empty:
            continue
        if classes:
            repeat_score = max((max_count / max(class_counts[class_id], 1)) ** exponent for class_id in classes)
        else:
            repeat_score = 1.0
        repeat = min(max_repeat, max(1, int(round(repeat_score))))
        repeat_histogram[repeat] += 1
        for class_id in classes:
            class_repeat_totals[class_id] += repeat
        expanded.extend([image_path.resolve()] * repeat)

    with NamedTemporaryFile("w", suffix=".txt", prefix="vessel_curriculum_balanced_train_", delete=False) as f:
        for image_path in expanded:
            f.write(f"{image_path}\n")
        train_list = Path(f.name)

    return train_list, {
        "enabled": True,
        "source_train_images": len(images),
        "expanded_train_images": len(expanded),
        "exponent": exponent,
        "max_repeat": max_repeat,
        "repeat_histogram": {str(key): value for key, value in sorted(repeat_histogram.items())},
        "class_image_counts": {str(key): value for key, value in sorted(class_counts.items())},
        "class_repeat_totals": {str(key): value for key, value in sorted(class_repeat_totals.items())},
        "train_list": str(train_list),
    }


def _preview_balanced_sampling(
    dataset_path: str | os.PathLike[str],
    cfg: Mapping[str, Any],
) -> dict[str, Any] | None:
    balanced_cfg = ((cfg.get("imbalance", {}) or {}).get("balanced_sampling", {}) or {})
    if not balanced_cfg.get("enabled", False):
        return None

    dataset_path, dataset_cfg = _load_yaml(dataset_path)
    dataset_root = dataset_cfg.get("path")
    root = (
        _resolve_existing_relative_path(dataset_root, ROOT_DIR, dataset_path.parent)
        if dataset_root
        else dataset_path.parent.resolve()
    )
    train_list, balance_info = _write_balanced_train_list(dataset_cfg, root, balanced_cfg)
    train_list.unlink(missing_ok=True)
    balance_info.pop("train_list", None)
    return balance_info


def _validate_dataset_config(dataset_path: str | os.PathLike[str]) -> dict[str, Any]:
    dataset_path, dataset_cfg = _load_yaml(dataset_path)
    root = _resolve_existing_relative_path(dataset_cfg.get("path", dataset_path.parent), ROOT_DIR, dataset_path.parent)

    for key in ("train", "val"):
        if key not in dataset_cfg:
            raise ValueError(f"Dataset config missing required '{key}' entry: {dataset_path}")
        split_path = Path(dataset_cfg[key])
        split_path = split_path if split_path.is_absolute() else root / split_path
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset {key} path not found: {split_path}")

    names = dataset_cfg.get("names")
    nc = dataset_cfg.get("nc")
    if names is None or nc is None:
        raise ValueError(f"Dataset config must define both nc and names: {dataset_path}")

    return {"path": str(root), "nc": int(nc), "names": names}


def _count_labels(dataset_path: str | os.PathLike[str]) -> dict[str, Any]:
    dataset_path, dataset_cfg = _load_yaml(dataset_path)
    root = _resolve_existing_relative_path(dataset_cfg.get("path", dataset_path.parent), ROOT_DIR, dataset_path.parent)
    names = dataset_cfg.get("names", {})
    if isinstance(names, list):
        class_names = {index: str(name) for index, name in enumerate(names)}
    else:
        class_names = {int(index): str(name) for index, name in names.items()}

    counts = {class_id: 0 for class_id in class_names}
    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        image_rel = dataset_cfg.get(split)
        if not image_rel:
            continue
        image_dir = Path(image_rel)
        image_dir = image_dir if image_dir.is_absolute() else root / image_dir
        label_dir = image_dir.parent / "labels"
        split_total = 0
        if label_dir.exists():
            for label_path in label_dir.glob("*.txt"):
                with label_path.open(encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            class_id = int(float(parts[0]))
                        except ValueError:
                            continue
                        counts[class_id] = counts.get(class_id, 0) + 1
                        split_total += 1
        split_counts[split] = split_total

    return {
        "class_counts": {class_names.get(class_id, str(class_id)): count for class_id, count in sorted(counts.items())},
        "split_label_counts": split_counts,
    }


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        return float(value)
    return None


def _interpolate_value(start: Any, end: Any, progress: float, key: str) -> Any:
    start_num = _as_number(start)
    end_num = _as_number(end)
    if start_num is None or end_num is None:
        return end if progress >= 0.5 else start

    value = start_num + (end_num - start_num) * progress
    if key == "imgsz":
        return int(max(32, round(value / 32) * 32))
    if key in INTEGER_KEYS:
        return int(round(value))
    return float(value)


def _segment_count(phase_epochs: int, segment_epochs: int) -> int:
    return max(1, int((phase_epochs + segment_epochs - 1) // segment_epochs))


def _segment_lengths(phase_epochs: int, count: int) -> list[int]:
    base = phase_epochs // count
    remainder = phase_epochs % count
    return [base + (1 if index < remainder else 0) for index in range(count)]


def _allocate_phase_epochs(phases: Sequence[Mapping[str, Any]], configured_total: int) -> list[int]:
    original = [int(phase.get("epochs", 0)) for phase in phases]
    phase_total = sum(original)
    if phase_total <= 0:
        raise ValueError("The sum of curriculum phase epochs must be positive")
    if configured_total < len(phases):
        raise ValueError(
            f"total_epochs must be at least the number of curriculum phases "
            f"({len(phases)}), got {configured_total}"
        )
    if configured_total == phase_total:
        return original

    scale = configured_total / phase_total
    raw = [epochs * scale for epochs in original]
    allocated = [max(1, int(floor(value))) for value in raw]

    while sum(allocated) < configured_total:
        candidates = sorted(
            range(len(raw)),
            key=lambda index: (raw[index] - floor(raw[index]), raw[index]),
            reverse=True,
        )
        for index in candidates:
            allocated[index] += 1
            if sum(allocated) == configured_total:
                break

    while sum(allocated) > configured_total:
        candidates = sorted(
            [index for index, value in enumerate(allocated) if value > 1],
            key=lambda index: (raw[index] - floor(raw[index]), raw[index]),
        )
        if not candidates:
            raise ValueError("Could not allocate curriculum phase epochs")
        for index in candidates:
            allocated[index] -= 1
            if sum(allocated) == configured_total:
                break

    return allocated


def _build_segments(cfg: Mapping[str, Any], total_epochs_override: int | None = None) -> list[CurriculumSegment]:
    training_cfg = cfg.get("training", {}) or {}
    curriculum_cfg = cfg.get("curriculum", {}) or {}
    phases = curriculum_cfg.get("phases", []) or []
    if not phases:
        raise ValueError("curriculum.phases must contain at least one phase")

    configured_total = int(total_epochs_override or training_cfg.get("total_epochs", 120))
    phase_epochs = _allocate_phase_epochs(phases, configured_total)

    segment_epochs = int(training_cfg.get("segment_epochs", 5))
    if segment_epochs <= 0:
        raise ValueError("training.segment_epochs must be positive")

    base_train_args = _base_train_args(training_cfg)
    segments: list[CurriculumSegment] = []
    absolute_epoch = 1

    for phase_index, (phase, epochs_for_phase) in enumerate(zip(phases, phase_epochs), start=1):
        start = dict(phase.get("start", {}) or {})
        end = dict(phase.get("end", start) or {})
        fixed = dict(phase.get("fixed", {}) or {})

        keys = sorted((set(start) | set(end)) & INTERPOLATED_KEYS)
        count = _segment_count(epochs_for_phase, segment_epochs)
        lengths = _segment_lengths(epochs_for_phase, count)

        for segment_in_phase, epochs_for_segment in enumerate(lengths, start=1):
            progress = 1.0 if count == 1 else (segment_in_phase - 1) / (count - 1)
            scheduled = {}
            for key in keys:
                scheduled[key] = _interpolate_value(start.get(key, end.get(key)), end.get(key, start.get(key)), progress, key)

            train_args = dict(base_train_args)
            train_args.update(scheduled)
            train_args.update(fixed)
            train_args["epochs"] = epochs_for_segment

            start_epoch = absolute_epoch
            end_epoch = absolute_epoch + epochs_for_segment - 1
            segments.append(
                CurriculumSegment(
                    index=len(segments) + 1,
                    phase_index=phase_index,
                    phase_key=str(phase.get("key", f"phase_{phase_index}")),
                    phase_title=str(phase.get("title", phase.get("key", f"phase_{phase_index}"))),
                    phase_rationale=str(phase.get("rationale", "")),
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    epochs=epochs_for_segment,
                    progress=progress,
                    train_args=train_args,
                )
            )
            absolute_epoch = end_epoch + 1

    return segments


def _base_train_args(training_cfg: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"total_epochs", "segment_epochs", "carry_weights"}
    args = {key: value for key, value in training_cfg.items() if key not in excluded}
    for key in INTEGER_KEYS & set(args):
        if args[key] is not None:
            args[key] = int(args[key])
    return args


def _format_segment_row(segment: CurriculumSegment) -> dict[str, Any]:
    keys = [
        "imgsz",
        "degrees",
        "translate",
        "scale",
        "perspective",
        "flipud",
        "mosaic",
        "mixup",
        "box",
        "cls",
        "dfl",
        "freeze",
    ]
    row = {
        "seg": segment.index,
        "phase": segment.phase_key,
        "epochs": f"{segment.start_epoch}-{segment.end_epoch}",
        "n": segment.epochs,
    }
    row.update({key: segment.train_args.get(key) for key in keys})
    return row


def _print_schedule(segments: Sequence[CurriculumSegment]) -> None:
    print("\nCurriculum schedule:")
    for segment in segments:
        row = _format_segment_row(segment)
        print(
            "  "
            f"{row['seg']:02d} {row['phase']:<15} ep {row['epochs']:<7} "
            f"img={row['imgsz']} deg={float(row['degrees']):6.1f} "
            f"scale={float(row['scale']):.2f} mosaic={float(row['mosaic']):.2f} "
            f"mixup={float(row['mixup']):.2f} box={float(row['box']):.2f} "
            f"cls={float(row['cls']):.2f}"
        )


def _import_ultralytics() -> tuple[Any, Any]:
    try:
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for curriculum training. Activate the vessel "
            "environment and ensure ultralytics is installed."
        ) from exc

    return YOLO, SETTINGS


def _import_mlflow() -> Any | None:
    try:
        import mlflow
    except ImportError:
        return None
    return mlflow


def _extract_metrics(results: Any) -> dict[str, float]:
    metrics = getattr(results, "results_dict", None)
    if not isinstance(metrics, Mapping):
        return {}
    return {str(key): float(value) for key, value in metrics.items() if isinstance(value, Real)}


def _metric_value(metrics: Mapping[str, float], suffixes: Sequence[str]) -> float | None:
    for key, value in metrics.items():
        normalized = key.lower()
        if any(normalized.endswith(suffix.lower()) or suffix.lower() in normalized for suffix in suffixes):
            return float(value)
    return None


def _find_run_paths(model: Any, results: Any) -> tuple[Path | None, Path | None, Path | None]:
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None) or getattr(results, "save_dir", None)
    if save_dir is not None:
        save_dir = Path(save_dir)

    best = getattr(trainer, "best", None)
    last = getattr(trainer, "last", None)
    if best is not None:
        best = Path(best)
    if last is not None:
        last = Path(last)

    if save_dir is not None:
        best_candidate = save_dir / "weights" / "best.pt"
        last_candidate = save_dir / "weights" / "last.pt"
        if best is None and best_candidate.exists():
            best = best_candidate
        if last is None and last_candidate.exists():
            last = last_candidate

    return save_dir, best if best and best.exists() else None, last if last and last.exists() else None


def _segment_run_name(base_name: str, segment: CurriculumSegment) -> str:
    return _sanitize_run_name(
        f"{base_name}_s{segment.index:02d}_e{segment.start_epoch:03d}-{segment.end_epoch:03d}_{segment.phase_key}"
    )


def _train_segment(
    *,
    YOLO: Any,
    weights: str | os.PathLike[str],
    dataset_config_path: Path,
    project: Path,
    base_name: str,
    segment: CurriculumSegment,
    exist_ok: bool,
) -> dict[str, Any]:
    model = YOLO(str(weights))
    run_name = _segment_run_name(base_name, segment)

    train_args = dict(segment.train_args)
    train_args["project"] = str(project)
    train_args["name"] = run_name
    train_args["exist_ok"] = bool(exist_ok)

    print(
        f"\n=== Segment {segment.index:02d}: {segment.phase_title} "
        f"(epochs {segment.start_epoch}-{segment.end_epoch}) ==="
    )
    print(f"Weights: {weights}")
    print(
        "Schedule: "
        f"imgsz={train_args.get('imgsz')} degrees={train_args.get('degrees')} "
        f"scale={train_args.get('scale')} mosaic={train_args.get('mosaic')} "
        f"mixup={train_args.get('mixup')} box={train_args.get('box')} cls={train_args.get('cls')}"
    )

    results = model.train(data=str(dataset_config_path), **train_args)
    metrics = _extract_metrics(results)
    save_dir, best_weights, last_weights = _find_run_paths(model, results)

    return {
        "segment": segment.index,
        "phase": segment.phase_key,
        "phase_title": segment.phase_title,
        "start_epoch": segment.start_epoch,
        "end_epoch": segment.end_epoch,
        "epochs": segment.epochs,
        "weights_in": str(weights),
        "save_dir": str(save_dir) if save_dir else None,
        "best_weights": str(best_weights) if best_weights else None,
        "last_weights": str(last_weights) if last_weights else None,
        "metrics": metrics,
        "train_args": train_args,
    }


def _validate_final_model(
    *,
    YOLO: Any,
    weights: str | os.PathLike[str],
    dataset_config_path: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: Any,
    project: Path,
    name: str,
) -> dict[str, Any]:
    print(f"\n=== Final {split} evaluation: {weights} ===")
    model = YOLO(str(weights))
    results = model.val(
        data=str(dataset_config_path),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        plots=True,
    )
    return {
        "split": split,
        "weights": str(weights),
        "metrics": _extract_metrics(results),
        "save_dir": str(getattr(results, "save_dir", "")),
    }


def _write_summary(summary: Mapping[str, Any], summary_dir: Path, base_name: str) -> tuple[Path, Path]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / f"{base_name}_{timestamp}.json"
    csv_path = summary_dir / f"{base_name}_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    segment_records = summary.get("segments", [])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "segment",
                "phase",
                "start_epoch",
                "end_epoch",
                "epochs",
                "imgsz",
                "degrees",
                "scale",
                "mosaic",
                "mixup",
                "box",
                "cls",
                "dfl",
                "map50",
                "map50_95",
                "save_dir",
                "best_weights",
                "last_weights",
            ],
        )
        writer.writeheader()
        for record in segment_records:
            train_args = record.get("train_args", {})
            metrics = record.get("metrics", {})
            writer.writerow(
                {
                    "segment": record.get("segment"),
                    "phase": record.get("phase"),
                    "start_epoch": record.get("start_epoch"),
                    "end_epoch": record.get("end_epoch"),
                    "epochs": record.get("epochs"),
                    "imgsz": train_args.get("imgsz"),
                    "degrees": train_args.get("degrees"),
                    "scale": train_args.get("scale"),
                    "mosaic": train_args.get("mosaic"),
                    "mixup": train_args.get("mixup"),
                    "box": train_args.get("box"),
                    "cls": train_args.get("cls"),
                    "dfl": train_args.get("dfl"),
                    "map50": _metric_value(metrics, ("map50", "metrics/mAP50(B)")),
                    "map50_95": _metric_value(metrics, ("map50-95", "map50_95", "metrics/mAP50-95(B)")),
                    "save_dir": record.get("save_dir"),
                    "best_weights": record.get("best_weights"),
                    "last_weights": record.get("last_weights"),
                }
            )

    return json_path, csv_path


def train_curriculum(
    config_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
    *,
    dataset_config: str | os.PathLike[str] | None = None,
    weights: str | os.PathLike[str] | None = None,
    total_epochs: int | None = None,
    dry_run: bool = False,
    no_mlflow: bool = False,
) -> dict[str, Any]:
    config_path, cfg = _load_yaml(config_path)
    dataset_config_path = dataset_config or cfg.get("dataset_config", DEFAULT_DATASET_CONFIG_PATH)
    dataset_info = _validate_dataset_config(dataset_config_path)
    label_info = _count_labels(dataset_config_path)
    segments = _build_segments(cfg, total_epochs_override=total_epochs)
    balance_preview = _preview_balanced_sampling(dataset_config_path, cfg)

    _print_schedule(segments)
    print("\nDataset:")
    dataset_summary = {**dataset_info, **label_info}
    if balance_preview is not None:
        dataset_summary["balanced_sampling"] = {
            key: value for key, value in balance_preview.items() if key != "train_list"
        }
    print(json.dumps(dataset_summary, indent=2))

    selected_weights = _select_model_weights(cfg, override=weights)
    output_cfg = cfg.get("output", {}) or {}
    training_cfg = cfg.get("training", {}) or {}
    validation_cfg = cfg.get("validation", {}) or {}

    project = _resolve_path(output_cfg.get("project", ROOT_DIR / "models" / "train"))
    base_name = _sanitize_run_name(str(output_cfg.get("name", "vesselimg_nano_obb_curriculum")))
    exist_ok = bool(output_cfg.get("exist_ok", False))
    summary_dir = _resolve_path(output_cfg.get("summary_dir", ROOT_DIR / "models" / "train" / "curriculum_summaries"))

    if dry_run:
        return {
            "config": str(config_path),
            "dataset_config": str(_resolve_path(dataset_config_path)),
            "weights": selected_weights,
            "segments": [_format_segment_row(segment) for segment in segments],
            "dataset": dataset_summary,
        }

    YOLO, SETTINGS = _import_ultralytics()
    SETTINGS["mlflow"] = False
    mlflow = None if no_mlflow else _import_mlflow()

    resolved_dataset_path, temp_dataset_files, balance_info = _write_resolved_dataset_config(dataset_config_path, cfg)
    segment_records: list[dict[str, Any]] = []
    current_weights: str | os.PathLike[str] = selected_weights
    carry_weights = str(training_cfg.get("carry_weights", "best")).lower()

    run_context = None
    if mlflow is not None:
        mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
        mlflow.set_experiment("vessel-tracking-curriculum")
        run_context = mlflow.start_run(run_name=f"{base_name}-{int(time.time())}")

    try:
        if run_context is not None:
            run_context.__enter__()
            mlflow.log_param("config", str(config_path))
            mlflow.log_param("dataset_config", str(_resolve_path(dataset_config_path)))
            mlflow.log_param("initial_weights", str(selected_weights))
            mlflow.log_param("total_epochs", sum(segment.epochs for segment in segments))
            mlflow.log_dict({**dataset_info, **label_info}, "dataset_summary.json")
            if balance_info is not None:
                mlflow.log_dict(balance_info, "balanced_sampling.json")

        for segment in segments:
            record = _train_segment(
                YOLO=YOLO,
                weights=current_weights,
                dataset_config_path=resolved_dataset_path,
                project=project,
                base_name=base_name,
                segment=segment,
                exist_ok=exist_ok,
            )
            segment_records.append(record)

            if mlflow is not None:
                mlflow.log_params(
                    {
                        f"seg{segment.index:02d}_phase": segment.phase_key,
                        f"seg{segment.index:02d}_imgsz": record["train_args"].get("imgsz"),
                        f"seg{segment.index:02d}_degrees": record["train_args"].get("degrees"),
                        f"seg{segment.index:02d}_mosaic": record["train_args"].get("mosaic"),
                    }
                )
                for key, value in record.get("metrics", {}).items():
                    mlflow.log_metric(f"seg{segment.index:02d}/{_sanitize_mlflow_name(key)}", value)

            next_weights = record.get("last_weights") if carry_weights == "last" else record.get("best_weights")
            if not next_weights:
                next_weights = record.get("best_weights") or record.get("last_weights")
            if not next_weights:
                raise RuntimeError(f"Segment {segment.index} completed without usable weights")
            current_weights = next_weights

        final_weights = str(current_weights)
        final_evals = []
        last_train_args = segments[-1].train_args
        eval_imgsz = int(last_train_args.get("imgsz", training_cfg.get("imgsz", 768)))
        eval_batch = int(training_cfg.get("batch", 8))
        eval_device = training_cfg.get("device", 0)

        if validation_cfg.get("run_final_val", True):
            final_evals.append(
                _validate_final_model(
                    YOLO=YOLO,
                    weights=final_weights,
                    dataset_config_path=resolved_dataset_path,
                    split="val",
                    imgsz=eval_imgsz,
                    batch=eval_batch,
                    device=eval_device,
                    project=project,
                    name=f"{base_name}_final_val",
                )
            )
        if validation_cfg.get("run_final_test", True):
            final_evals.append(
                _validate_final_model(
                    YOLO=YOLO,
                    weights=final_weights,
                    dataset_config_path=resolved_dataset_path,
                    split="test",
                    imgsz=eval_imgsz,
                    batch=eval_batch,
                    device=eval_device,
                    project=project,
                    name=f"{base_name}_final_test",
                )
            )

        summary = {
            "config": str(config_path),
            "dataset_config": str(_resolve_path(dataset_config_path)),
            "dataset": {**dataset_info, **label_info},
            "balanced_sampling": balance_info,
            "initial_weights": str(selected_weights),
            "final_weights": final_weights,
            "total_epochs": sum(segment.epochs for segment in segments),
            "segments": segment_records,
            "final_evaluations": final_evals,
        }
        json_path, csv_path = _write_summary(summary, summary_dir, base_name)
        summary["summary_json"] = str(json_path)
        summary["summary_csv"] = str(csv_path)

        if mlflow is not None:
            mlflow.log_param("final_weights", final_weights)
            mlflow.log_artifact(str(json_path))
            mlflow.log_artifact(str(csv_path))
            for evaluation in final_evals:
                split = evaluation["split"]
                for key, value in evaluation.get("metrics", {}).items():
                    mlflow.log_metric(f"final_{split}/{_sanitize_mlflow_name(key)}", value)

        print("\n--- Curriculum Training Complete ---")
        print(f"Final weights: {final_weights}")
        print(f"Summary JSON:  {json_path}")
        print(f"Summary CSV:   {csv_path}")
        return summary
    finally:
        resolved_dataset_path.unlink(missing_ok=True)
        for temp_file in temp_dataset_files:
            temp_file.unlink(missing_ok=True)
        if run_context is not None:
            run_context.__exit__(*sys.exc_info())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train YOLOv8n-OBB on VESSELimg with a curriculum that ramps "
            "difficulty over 120 epochs by default."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Curriculum YAML config.")
    parser.add_argument("--dataset-config", default=None, help="Override dataset YAML path.")
    parser.add_argument("--weights", default=None, help="Override initial YOLOv8n OBB weights path/name.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total curriculum epochs.")
    parser.add_argument("--dry-run", action="store_true", help="Print/return the expanded curriculum without training.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_curriculum(
        config_path=args.config,
        dataset_config=args.dataset_config,
        weights=args.weights,
        total_epochs=args.epochs,
        dry_run=args.dry_run,
        no_mlflow=args.no_mlflow,
    )
    if args.dry_run:
        print("\nDry-run summary:")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
