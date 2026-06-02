"""
Controlled ablation study runner for the VESSELimg YOLOv8n OBB model.

This file intentionally keeps the user's existing run as the reference point:

    models/train/vesselimg_nano_obb7/weights/best.pt

The baseline training setup in configs/train.yaml is already strong for aerial
oriented boxes: full rotation, vertical and horizontal flips, mosaic, mixup,
HSV jitter, slight perspective distortion, and heavier box loss. The ablation
plan changes one factor at a time so the final table can answer which parts of
that setup are carrying the model and which parts can be simplified.

Study design
------------
1. Use the same dataset split from configs/vessel.yaml for every run.
2. Use the same pretrained YOLOv8 nano OBB checkpoint for every training run.
3. Keep epochs, batch size, optimizer, seed, and image size fixed unless a
   study explicitly changes the factor being tested.
4. Evaluate every completed model on val and test splits, then write JSON, CSV,
   and Markdown summaries under models/ablation_studies/.
5. Treat the existing best.pt run as an anchor, not as a replacement for a
   controlled retrain. The "baseline_retrain" study is included to measure run
   variance under the same config.

Typical usage
-------------
Run these from the repository root after activating the conda environment:

    conda activate vessel
    python src/model/abalation_studies.py --list
    python src/model/abalation_studies.py --dry-run --studies core
    python src/model/abalation_studies.py --studies core
    python src/model/abalation_studies.py --summarize

For a cheaper smoke pass:

    python src/model/abalation_studies.py --studies smoke --epochs 5 --fraction 0.1

The filename preserves the requested spelling ("abalation"), but the code and
outputs use the standard term "ablation".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent

CONFIG_TRAIN_PATH = ROOT_DIR / "configs" / "train.yaml"
CONFIG_DATASET_PATH = ROOT_DIR / "configs" / "vessel.yaml"
DEFAULT_TRAIN_WEIGHTS = MODEL_DIR / "yolov8n-obb.pt"
FALLBACK_TRAIN_WEIGHTS = ROOT_DIR / "yolov8n-obb.pt"

BASELINE_RUN_DIR = ROOT_DIR / "models" / "train" / "vesselimg_nano_obb7"
BASELINE_WEIGHTS = BASELINE_RUN_DIR / "weights" / "best.pt"
DEFAULT_STUDY_PROJECT = ROOT_DIR / "models" / "ablation_studies"

MLFLOW_NAME_RE = re.compile(r"[^A-Za-z0-9_./ :\\-]")
RESULT_FILE = "ablation_result.json"

# Keep Ultralytics and Matplotlib from trying to write under a locked home dir.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("/tmp") / "ultralytics"))


@dataclass(frozen=True)
class AblationStudy:
    key: str
    title: str
    action: str
    group: str
    rationale: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    weights: str | os.PathLike[str] | None = None


STUDIES: tuple[AblationStudy, ...] = (
    AblationStudy(
        key="baseline_best_eval",
        title="Evaluate existing best.pt",
        action="eval",
        group="reference",
        weights="baseline",
        rationale=(
            "Anchors the study to the already trained vesselimg_nano_obb7 "
            "checkpoint before spending time on controlled retraining."
        ),
    ),
    AblationStudy(
        key="baseline_retrain",
        title="Retrain unchanged baseline",
        action="train",
        group="reference",
        rationale=(
            "Measures normal run-to-run variance with the same config, seed, "
            "dataset, and pretrained nano OBB checkpoint."
        ),
    ),
    AblationStudy(
        key="no_full_rotation",
        title="Remove arbitrary rotation",
        action="train",
        group="geometry",
        overrides={"degrees": 0.0},
        rationale=(
            "Aerial vessels have no fixed image orientation. Dropping full "
            "rotation tests whether this augmentation is essential for OBB "
            "generalization."
        ),
    ),
    AblationStudy(
        key="limited_rotation_45",
        title="Limit rotation to 45 degrees",
        action="train",
        group="geometry",
        overrides={"degrees": 45.0},
        rationale=(
            "Tests whether full 180 degree rotation is necessary, or whether "
            "a smaller rotation envelope is enough."
        ),
    ),
    AblationStudy(
        key="no_vertical_flip",
        title="Remove vertical flip",
        action="train",
        group="geometry",
        overrides={"flipud": 0.0},
        rationale=(
            "Aerial imagery usually permits vertical flips, but this checks "
            "whether flipud harms class cues such as ship superstructure."
        ),
    ),
    AblationStudy(
        key="no_mosaic",
        title="Remove mosaic",
        action="train",
        group="composition",
        overrides={"mosaic": 0.0},
        rationale=(
            "Mosaic can help dense scenes and scale diversity, but it may also "
            "create unrealistic coastline or wake compositions."
        ),
    ),
    AblationStudy(
        key="no_mixup",
        title="Remove mixup",
        action="train",
        group="composition",
        overrides={"mixup": 0.0},
        rationale=(
            "Mixup regularizes small datasets, but transparent vessels can "
            "confuse OBB boundaries. This isolates that tradeoff."
        ),
    ),
    AblationStudy(
        key="no_color_jitter",
        title="Remove HSV jitter",
        action="train",
        group="photometric",
        overrides={"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0},
        rationale=(
            "Tests whether color and brightness variation are carrying weather, "
            "water color, and illumination robustness."
        ),
    ),
    AblationStudy(
        key="no_perspective",
        title="Remove perspective distortion",
        action="train",
        group="geometry",
        overrides={"perspective": 0.0},
        rationale=(
            "The configured perspective is small. This checks whether camera "
            "tilt simulation has measurable value or just adds noise."
        ),
    ),
    AblationStudy(
        key="mild_scale_translate",
        title="Reduce scale and translation",
        action="train",
        group="geometry",
        overrides={"scale": 0.25, "translate": 0.05},
        rationale=(
            "A lower geometric perturbation budget tests whether the baseline "
            "is over-regularizing object size and placement."
        ),
    ),
    AblationStudy(
        key="box_loss_lower",
        title="Lower box loss weight",
        action="train",
        group="loss",
        overrides={"box": 5.0},
        rationale=(
            "The baseline emphasizes localization. Lowering box loss tests "
            "whether tight OBB regression is the limiting factor."
        ),
    ),
    AblationStudy(
        key="class_loss_higher",
        title="Raise class loss weight",
        action="train",
        group="loss",
        overrides={"cls": 1.0},
        rationale=(
            "VESSELimg has visually similar classes. Raising cls tests whether "
            "class separation benefits from more classification pressure."
        ),
    ),
    AblationStudy(
        key="imgsz_1024",
        title="Increase image size to 1024",
        action="train",
        group="resolution",
        overrides={"imgsz": 1024, "batch": 8},
        rationale=(
            "Small and elongated vessels may benefit from more pixels. Batch is "
            "reduced to fit common GPU memory limits."
        ),
    ),
)

PROFILES: dict[str, tuple[str, ...]] = {
    "smoke": (
        "baseline_best_eval",
        "baseline_retrain",
        "no_full_rotation",
        "no_mosaic",
    ),
    "core": (
        "baseline_best_eval",
        "baseline_retrain",
        "no_full_rotation",
        "limited_rotation_45",
        "no_vertical_flip",
        "no_mosaic",
        "no_mixup",
        "no_color_jitter",
        "no_perspective",
    ),
    "extended": tuple(study.key for study in STUDIES),
    "all": tuple(study.key for study in STUDIES),
}


def _resolve_path(path: str | os.PathLike[str], base_dir: Path = ROOT_DIR) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _load_yaml(path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    resolved = _resolve_path(path)
    with resolved.open() as f:
        return resolved, yaml.safe_load(f) or {}


def _resolve_existing_relative_path(
    path: str | os.PathLike[str], *base_dirs: Path
) -> Path:
    candidate_path = Path(path).expanduser()
    if candidate_path.is_absolute():
        return candidate_path.resolve()

    for base_dir in base_dirs:
        candidate = (base_dir / candidate_path).resolve()
        if candidate.exists():
            return candidate

    return (base_dirs[0] / candidate_path).resolve()


def _write_resolved_dataset_config(dataset_path: str | os.PathLike[str]) -> Path:
    resolved_path, dataset_cfg = _load_yaml(dataset_path)

    if dataset_cfg.get("path"):
        dataset_cfg["path"] = str(
            _resolve_existing_relative_path(
                dataset_cfg["path"], ROOT_DIR, resolved_path.parent
            )
        )
    else:
        dataset_cfg["path"] = str(resolved_path.parent)

    with NamedTemporaryFile(
        "w", suffix=".yaml", prefix="vessel_dataset_", delete=False
    ) as f:
        yaml.safe_dump(dataset_cfg, f, sort_keys=False)
        return Path(f.name)


def _dataset_names(dataset_path: str | os.PathLike[str]) -> dict[int, str]:
    _, cfg = _load_yaml(dataset_path)
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    if isinstance(names, Mapping):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def _study_map() -> dict[str, AblationStudy]:
    return {study.key: study for study in STUDIES}


def _resolve_study_selection(tokens: list[str]) -> list[AblationStudy]:
    studies_by_key = _study_map()
    selected: list[AblationStudy] = []
    seen: set[str] = set()

    for token in tokens:
        keys = PROFILES.get(token, (token,))
        for key in keys:
            if key not in studies_by_key:
                valid = ", ".join(sorted([*PROFILES.keys(), *studies_by_key.keys()]))
                raise SystemExit(f"Unknown study or profile '{key}'. Valid values: {valid}")
            if key not in seen:
                selected.append(studies_by_key[key])
                seen.add(key)

    return selected


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        return value
    return str(value)


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_mlflow_name(name: str) -> str:
    return MLFLOW_NAME_RE.sub("", name) or "metric"


def _import_yolo():
    try:
        from ultralytics import YOLO

        try:
            from ultralytics.utils import SETTINGS

            SETTINGS["mlflow"] = False
        except Exception:
            pass

        return YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required for training/evaluation. Activate the "
            "conda environment first, for example: conda activate vessel"
        ) from exc


@contextmanager
def _mlflow_run(args: argparse.Namespace, run_name: str, params: Mapping[str, Any]):
    if args.no_mlflow:
        yield None
        return

    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed; continuing without MLflow logging.")
        yield None
        return

    mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name=run_name):
        for key, value in params.items():
            mlflow.log_param(key, json.dumps(_jsonable(value), sort_keys=True))
        yield mlflow


def _log_mlflow_metrics(mlflow_module: Any, metrics: Mapping[str, float]) -> None:
    if mlflow_module is None:
        return

    for key, value in metrics.items():
        if isinstance(value, Real) and not isinstance(value, bool):
            mlflow_module.log_metric(_sanitize_mlflow_name(key), float(value))


def _training_weights(args: argparse.Namespace) -> str | Path:
    if args.weights:
        return _resolve_path(args.weights)
    if DEFAULT_TRAIN_WEIGHTS.exists():
        return DEFAULT_TRAIN_WEIGHTS
    if FALLBACK_TRAIN_WEIGHTS.exists():
        return FALLBACK_TRAIN_WEIGHTS
    return "yolov8n-obb.pt"


def _baseline_weights(args: argparse.Namespace) -> Path:
    if args.baseline_weights:
        weights = _resolve_path(args.baseline_weights)
    else:
        weights = _resolve_path(args.baseline_run_dir) / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(
            f"Baseline weights not found at {weights}. "
            "Pass --baseline-weights to evaluate a different checkpoint."
        )
    return weights


def _runtime_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in ("epochs", "batch", "device", "workers", "fraction", "patience"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

    if args.cache is not None:
        overrides["cache"] = False if args.cache == "false" else args.cache
    if args.seed is not None:
        overrides["seed"] = args.seed
        overrides["deterministic"] = True
    if args.no_plots:
        overrides["plots"] = False

    return overrides


def _base_train_config(args: argparse.Namespace) -> dict[str, Any]:
    _, cfg = _load_yaml(args.config)
    cfg = dict(cfg)
    cfg["project"] = str(_resolve_path(args.project))
    cfg["exist_ok"] = args.exist_ok
    cfg.setdefault("seed", args.seed)
    cfg.setdefault("deterministic", True)
    return cfg


def _train_config_for(study: AblationStudy, args: argparse.Namespace) -> dict[str, Any]:
    cfg = _base_train_config(args)
    cfg.update(dict(study.overrides))
    cfg.update(_runtime_overrides(args))
    cfg["name"] = f"{args.name_prefix}_{study.key}"
    cfg["val"] = True
    return cfg


def _eval_kwargs(args: argparse.Namespace, train_cfg: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "plots": False,
        "save_json": False,
    }

    for key in ("device", "workers", "batch", "imgsz"):
        value = getattr(args, f"eval_{key}", None)
        if value is None:
            value = train_cfg.get(key)
        if value is not None:
            kwargs[key] = value

    return kwargs


def _metric_dict(results: Any, class_names: Mapping[int, str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    results_dict = getattr(results, "results_dict", None)

    if isinstance(results_dict, Mapping):
        for key, value in results_dict.items():
            if isinstance(value, Real) and not isinstance(value, bool):
                metrics[str(key)] = float(value)

    box = getattr(results, "box", None)
    if box is not None:
        for attr, prefix in (("ap", "class_ap50_95"), ("ap50", "class_ap50")):
            values = getattr(box, attr, None)
            if values is None:
                continue
            for index, value in enumerate(list(values)):
                if isinstance(value, Real) and not isinstance(value, bool):
                    class_name = class_names.get(index, str(index))
                    metrics[f"{prefix}/{class_name}"] = float(value)

    speed = getattr(results, "speed", None)
    if isinstance(speed, Mapping):
        for key, value in speed.items():
            if isinstance(value, Real) and not isinstance(value, bool):
                metrics[f"speed/{key}"] = float(value)

    return metrics


def _prefix_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def _read_results_csv(run_dir: str | os.PathLike[str]) -> dict[str, float]:
    results_path = Path(run_dir) / "results.csv"
    if not results_path.exists():
        return {}

    with results_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [{key.strip(): value for key, value in row.items()} for row in reader]

    if not rows:
        return {}

    def row_metric(row: Mapping[str, Any]) -> float:
        return (
            _float_or_none(row.get("metrics/mAP50-95(B)"))
            or _float_or_none(row.get("metrics/mAP50-95"))
            or -1.0
        )

    last_row = rows[-1]
    best_row = max(rows, key=row_metric)
    metrics: dict[str, float] = {}

    for prefix, row in (("train_last", last_row), ("train_best", best_row)):
        for key, value in row.items():
            numeric = _float_or_none(value)
            if numeric is not None:
                metrics[f"{prefix}/{key}"] = numeric

    return metrics


def _record_path(run_dir: Path) -> Path:
    return run_dir / RESULT_FILE


def _load_record(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_record(run_dir: Path, record: Mapping[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = _record_path(run_dir)
    with path.open("w") as f:
        json.dump(_jsonable(record), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _record_from_existing_run(
    study: AblationStudy, run_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        weights = run_dir / "weights" / "last.pt"

    record = {
        "study_key": study.key,
        "title": study.title,
        "action": study.action,
        "group": study.group,
        "rationale": study.rationale,
        "status": "existing_run_summarized",
        "run_dir": str(run_dir),
        "weights": str(weights) if weights.exists() else None,
        "overrides": dict(study.overrides),
        "metrics": _read_results_csv(run_dir),
    }
    _write_record(run_dir, record)
    print(f"Summarized existing run for {study.key}: {run_dir}")
    return record


def _evaluate_weights(
    weights: str | os.PathLike[str],
    split: str,
    args: argparse.Namespace,
    dataset_cfg_path: Path,
    train_cfg: Mapping[str, Any],
    class_names: Mapping[int, str],
    name: str,
) -> dict[str, float]:
    YOLO = _import_yolo()
    model = YOLO(str(weights))
    eval_project = _resolve_path(args.project) / "eval"
    eval_project.mkdir(parents=True, exist_ok=True)

    results = model.val(
        data=str(dataset_cfg_path),
        split=split,
        project=str(eval_project),
        name=f"{name}_{split}",
        exist_ok=True,
        **_eval_kwargs(args, train_cfg),
    )
    return _metric_dict(results, class_names)


def run_eval_study(study: AblationStudy, args: argparse.Namespace) -> dict[str, Any]:
    project = _resolve_path(args.project)
    run_dir = project / f"{args.name_prefix}_{study.key}"
    existing_record = _record_path(run_dir)
    if existing_record.exists() and not args.rerun:
        print(f"Using existing record for {study.key}: {existing_record}")
        return _load_record(existing_record)

    dataset_cfg_path = _write_resolved_dataset_config(args.data)
    class_names = _dataset_names(args.data)
    train_cfg = _base_train_config(args)
    started = time.time()

    try:
        weights = _baseline_weights(args) if study.weights == "baseline" else study.weights
        if weights is None:
            raise ValueError(f"Evaluation study {study.key} has no weights configured.")

        record = {
            "study_key": study.key,
            "title": study.title,
            "action": study.action,
            "group": study.group,
            "rationale": study.rationale,
            "status": "completed",
            "run_dir": str(run_dir),
            "weights": str(weights),
            "overrides": dict(study.overrides),
            "metrics": {},
            "duration_sec": None,
        }

        with _mlflow_run(args, study.key, record) as mlflow_module:
            for split in args.eval_splits:
                print(f"Evaluating {study.key} on {split}: {weights}")
                metrics = _evaluate_weights(
                    weights=weights,
                    split=split,
                    args=args,
                    dataset_cfg_path=dataset_cfg_path,
                    train_cfg=train_cfg,
                    class_names=class_names,
                    name=f"{args.name_prefix}_{study.key}",
                )
                prefixed = _prefix_metrics(split, metrics)
                record["metrics"].update(prefixed)
                _log_mlflow_metrics(mlflow_module, prefixed)

        record["duration_sec"] = round(time.time() - started, 3)
        _write_record(run_dir, record)
        return record
    finally:
        dataset_cfg_path.unlink(missing_ok=True)


def run_train_study(study: AblationStudy, args: argparse.Namespace) -> dict[str, Any]:
    project = _resolve_path(args.project)
    expected_run_dir = project / f"{args.name_prefix}_{study.key}"
    existing_record = _record_path(expected_run_dir)
    if existing_record.exists() and not args.rerun:
        print(f"Using existing record for {study.key}: {existing_record}")
        return _load_record(existing_record)
    if (expected_run_dir / "results.csv").exists() and not args.rerun:
        return _record_from_existing_run(study, expected_run_dir, args)

    cfg = _train_config_for(study, args)
    dataset_cfg_path = _write_resolved_dataset_config(args.data)
    class_names = _dataset_names(args.data)
    weights = _training_weights(args)
    started = time.time()

    record: dict[str, Any] = {
        "study_key": study.key,
        "title": study.title,
        "action": study.action,
        "group": study.group,
        "rationale": study.rationale,
        "status": "started",
        "run_dir": None,
        "weights": None,
        "start_weights": str(weights),
        "overrides": dict(study.overrides),
        "train_config": cfg,
        "metrics": {},
        "duration_sec": None,
    }

    try:
        YOLO = _import_yolo()
        print(f"Training {study.key} from {weights}")

        with _mlflow_run(args, study.key, record) as mlflow_module:
            model = YOLO(str(weights))
            results = model.train(data=str(dataset_cfg_path), **cfg, resume=args.resume)

            trainer = getattr(model, "trainer", None)
            save_dir = getattr(trainer, "save_dir", None)
            if save_dir is None:
                save_dir = getattr(results, "save_dir", None)
            save_dir = Path(save_dir) if save_dir else expected_run_dir

            best_weights = getattr(trainer, "best", None)
            if best_weights is None or not Path(best_weights).exists():
                best_weights = getattr(trainer, "last", None)
            best_weights = Path(best_weights) if best_weights else save_dir / "weights" / "best.pt"

            train_metrics = _prefix_metrics("train_final", _metric_dict(results, class_names))
            train_metrics.update(_read_results_csv(save_dir))
            record["metrics"].update(train_metrics)
            _log_mlflow_metrics(mlflow_module, train_metrics)

            if not args.no_eval_after_train and best_weights.exists():
                for split in args.eval_splits:
                    print(f"Evaluating {study.key} best weights on {split}: {best_weights}")
                    metrics = _evaluate_weights(
                        weights=best_weights,
                        split=split,
                        args=args,
                        dataset_cfg_path=dataset_cfg_path,
                        train_cfg=cfg,
                        class_names=class_names,
                        name=f"{args.name_prefix}_{study.key}",
                    )
                    prefixed = _prefix_metrics(split, metrics)
                    record["metrics"].update(prefixed)
                    _log_mlflow_metrics(mlflow_module, prefixed)

            record["status"] = "completed"
            record["run_dir"] = str(save_dir)
            record["weights"] = str(best_weights) if best_weights.exists() else None
            record["duration_sec"] = round(time.time() - started, 3)
            _write_record(save_dir, record)
            return record
    finally:
        dataset_cfg_path.unlink(missing_ok=True)


def _primary_metric(record: Mapping[str, Any]) -> float | None:
    metrics = record.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return None

    priority = (
        "test/metrics/mAP50-95(B)",
        "val/metrics/mAP50-95(B)",
        "train_best/metrics/mAP50-95(B)",
        "train_last/metrics/mAP50-95(B)",
        "train_final/metrics/mAP50-95(B)",
    )
    for key in priority:
        value = _float_or_none(metrics.get(key))
        if value is not None:
            return value
    return None


def _baseline_primary(records: list[Mapping[str, Any]]) -> float | None:
    for key in ("baseline_best_eval", "baseline_retrain"):
        for record in records:
            if record.get("study_key") == key:
                value = _primary_metric(record)
                if value is not None:
                    return value
    return None


def _discover_records(project: str | os.PathLike[str]) -> list[dict[str, Any]]:
    project_path = _resolve_path(project)
    if not project_path.exists():
        return []

    records = []
    for path in sorted(project_path.glob(f"**/{RESULT_FILE}")):
        try:
            records.append(_load_record(path))
        except json.JSONDecodeError:
            print(f"Skipping invalid result file: {path}")
    return records


def summarize_records(
    args: argparse.Namespace, records: list[dict[str, Any]] | None = None
) -> None:
    discovered = _discover_records(args.project)
    if records is not None:
        merged: dict[tuple[str | None, str | None], dict[str, Any]] = {}
        for record in [*discovered, *records]:
            key = (record.get("study_key"), record.get("run_dir"))
            merged[key] = record
        records = list(merged.values())
    else:
        records = discovered
    project = _resolve_path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    if not records:
        print(f"No {RESULT_FILE} files found under {project}; nothing to summarize yet.")
        return

    baseline = _baseline_primary(records)
    rows: list[dict[str, Any]] = []
    metric_keys: set[str] = set()

    for record in records:
        metrics = record.get("metrics", {})
        if isinstance(metrics, Mapping):
            metric_keys.update(str(key) for key in metrics.keys())

    for record in records:
        primary = _primary_metric(record)
        row: dict[str, Any] = {
            "study_key": record.get("study_key"),
            "title": record.get("title"),
            "group": record.get("group"),
            "status": record.get("status"),
            "action": record.get("action"),
            "run_dir": record.get("run_dir"),
            "weights": record.get("weights"),
            "primary_mAP50_95": primary,
            "delta_primary_vs_baseline": (
                None if primary is None or baseline is None else primary - baseline
            ),
            "duration_sec": record.get("duration_sec"),
        }
        metrics = record.get("metrics", {})
        if isinstance(metrics, Mapping):
            for key in metric_keys:
                row[key] = metrics.get(key)
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["primary_mAP50_95"] is None,
            -(row["primary_mAP50_95"] or -1.0),
            str(row["study_key"]),
        )
    )

    summary_json = project / "ablation_summary.json"
    with summary_json.open("w") as f:
        json.dump(_jsonable(records), f, indent=2, sort_keys=True)
        f.write("\n")

    summary_csv = project / "ablation_summary.csv"
    base_fields = [
        "study_key",
        "title",
        "group",
        "status",
        "action",
        "primary_mAP50_95",
        "delta_primary_vs_baseline",
        "duration_sec",
        "run_dir",
        "weights",
    ]
    fields = base_fields + sorted(metric_keys)
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_md = project / "ablation_summary.md"
    with summary_md.open("w") as f:
        f.write("# VESSELimg YOLOv8n OBB Ablation Summary\n\n")
        f.write(
            "Primary metric priority: test mAP50-95, then val mAP50-95, "
            "then best training-run val mAP50-95.\n\n"
        )
        f.write("| Rank | Study | Group | Primary mAP50-95 | Delta | Status |\n")
        f.write("| ---: | --- | --- | ---: | ---: | --- |\n")
        for index, row in enumerate(rows, start=1):
            primary = row["primary_mAP50_95"]
            delta = row["delta_primary_vs_baseline"]
            primary_text = "" if primary is None else f"{primary:.5f}"
            delta_text = "" if delta is None else f"{delta:+.5f}"
            f.write(
                f"| {index} | {row['study_key']} | {row['group']} | "
                f"{primary_text} | {delta_text} | {row['status']} |\n"
            )

    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote summary CSV:  {summary_csv}")
    print(f"Wrote summary MD:   {summary_md}")


def print_study_list() -> None:
    print("Available profiles:")
    for name, keys in PROFILES.items():
        print(f"  {name}: {', '.join(keys)}")

    print("\nAvailable studies:")
    for study in STUDIES:
        overrides = json.dumps(dict(study.overrides), sort_keys=True)
        print(f"  {study.key}")
        print(f"    action:    {study.action}")
        print(f"    group:     {study.group}")
        print(f"    title:     {study.title}")
        print(f"    overrides: {overrides}")
        print(f"    why:       {study.rationale}")


def print_dry_run(studies: list[AblationStudy], args: argparse.Namespace) -> None:
    print("Ablation dry run")
    print(f"  config:           {_resolve_path(args.config)}")
    print(f"  data:             {_resolve_path(args.data)}")
    print(f"  project:          {_resolve_path(args.project)}")
    print(f"  start weights:    {_training_weights(args)}")
    print(f"  baseline weights: {_baseline_weights(args)}")
    print(f"  eval splits:      {', '.join(args.eval_splits)}")
    print()

    for index, study in enumerate(studies, start=1):
        cfg = _train_config_for(study, args) if study.action == "train" else _base_train_config(args)
        print(f"{index}. {study.key} ({study.action}, {study.group})")
        print(f"   {study.rationale}")
        if study.overrides:
            print(f"   changed params: {json.dumps(dict(study.overrides), sort_keys=True)}")
        if study.action == "train":
            print(f"   run name:       {cfg['name']}")
            print(f"   epochs/imgsz:   {cfg.get('epochs')} / {cfg.get('imgsz')}")
            print(f"   batch/device:   {cfg.get('batch')} / {cfg.get('device')}")
        else:
            print(f"   weights:        {_baseline_weights(args)}")
        print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled ablation studies for YOLOv8n OBB on VESSELimg."
    )
    parser.add_argument("--list", action="store_true", help="List profiles and studies.")
    parser.add_argument(
        "--studies",
        nargs="+",
        default=["core"],
        help="Study keys or profiles to run. Built-in profiles: smoke, core, extended, all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved study plan without training or evaluating.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Only summarize existing ablation_result.json files and exit.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run only evaluation studies from the selected profile.",
    )
    parser.add_argument("--config", default=str(CONFIG_TRAIN_PATH), help="Training YAML.")
    parser.add_argument("--data", default=str(CONFIG_DATASET_PATH), help="Dataset YAML.")
    parser.add_argument("--project", default=str(DEFAULT_STUDY_PROJECT), help="Output dir.")
    parser.add_argument("--name-prefix", default="vesselimg_nano_obb_ablation")
    parser.add_argument("--weights", default=None, help="Training start weights.")
    parser.add_argument("--baseline-weights", default=None)
    parser.add_argument("--baseline-run-dir", default=str(BASELINE_RUN_DIR))
    parser.add_argument("--mlflow-experiment", default="vessel-ablation-studies")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Ignore existing records.")
    parser.add_argument("--resume", action="store_true", help="Pass resume=True to train.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow YOLO to reuse run dirs.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-eval-after-train", action="store_true")
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["val", "test"],
        choices=["val", "test"],
        help="Splits to evaluate after each study.",
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", choices=["ram", "disk", "false"], default=None)

    parser.add_argument("--eval-imgsz", type=int, default=None)
    parser.add_argument("--eval-batch", type=int, default=None)
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--eval-workers", type=int, default=None)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list:
        print_study_list()
        return

    if args.summarize:
        summarize_records(args)
        return

    studies = _resolve_study_selection(args.studies)
    if args.eval_only:
        studies = [study for study in studies if study.action == "eval"]

    if not studies:
        raise SystemExit("No studies selected.")

    if args.dry_run:
        print_dry_run(studies, args)
        return

    records = []
    for study in studies:
        if study.action == "eval":
            records.append(run_eval_study(study, args))
        elif study.action == "train":
            records.append(run_train_study(study, args))
        else:
            raise ValueError(f"Unsupported study action: {study.action}")

    summarize_records(args, records)


if __name__ == "__main__":
    main(sys.argv[1:])
