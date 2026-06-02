# Aerial Maritime Vessel Intelligence and Tracking

This repository contains an end-to-end computer vision workflow for aerial maritime surveillance. The project is focused on detecting vessels from drone or elevated camera footage, classifying them by type, and maintaining persistent tracks across video frames.

The current codebase is organized around a modern oriented-bounding-box pipeline:

- YOLOv8 OBB training for rotated vessel detection
- curriculum-based training for staged optimization and imbalance handling
- TensorRT export and low-latency inference for deployment
- DeepSORT-style multi-object tracking adapted for oriented bounding boxes
- video pipeline tooling for annotated output and per-track event logs

The older `Ship_Detection/` and `Object_Tracking/` directories are preserved because they already existed on the remote branch. The actively maintained project code introduced in this repository lives under `src/` and `configs/`.

## Project Goals

The system is intended for maritime monitoring scenarios where a fixed-axis detector is not sufficient:

- vessels appear at arbitrary headings and aspect ratios
- drone footage contains camera motion, scale changes, and long-range targets
- downstream tracking benefits from rotation-aware geometry instead of axis-aligned boxes
- deployment requires a path from training to optimized inference on GPU hardware

The result is a pipeline that can train a vessel detector, export it to TensorRT, run real-time inference on video, and associate detections into stable track identities.

## Core Capabilities

### 1. Oriented vessel detection

The detector is trained as a YOLOv8 oriented bounding box model. Instead of standard axis-aligned boxes, each detection includes rotated geometry, which is better suited for ships viewed from aerial imagery where heading matters.

The dataset configuration in [`configs/vessel.yaml`](/home/acer/workspace/projects/amvit/configs/vessel.yaml:1) defines six vessel classes:

- `cargo`
- `military`
- `carrier`
- `cruise`
- `tanker`
- `ferry`

### 2. Standard and curriculum training

The training code in [`src/model/train.py`](/home/acer/workspace/projects/amvit/src/model/train.py:1) handles conventional YOLOv8 OBB training with MLflow logging. It resolves dataset paths, loads training hyperparameters from YAML, and logs metrics and run artifacts for experiment tracking.

The curriculum pipeline in [`src/model/train_curriculum.py`](/home/acer/workspace/projects/amvit/src/model/train_curriculum.py:1) extends this with:

- staged training segments
- interpolated hyperparameters across phases
- optional balanced sampling for class imbalance
- structured summaries for curriculum segments and dataset composition

This makes it easier to train progressively across easier and harder regimes instead of relying on a single static training schedule.

### 3. TensorRT deployment path

The inference wrapper in [`src/model/trt_infer.py`](/home/acer/workspace/projects/amvit/src/model/trt_infer.py:1) loads a serialized TensorRT engine and runs low-latency OBB inference on GPU. It includes:

- TensorRT engine loading
- PyCUDA-backed buffer management
- letterbox preprocessing
- OBB output decoding
- class-name loading from dataset config

This is the runtime path used for deployment-oriented detection rather than research-only evaluation.

### 4. Oriented multi-object tracking

Tracking is implemented in [`src/tracking/deepsort.py`](/home/acer/workspace/projects/amvit/src/tracking/deepsort.py:1) and supporting geometry/filter modules under `src/tracking/`. The tracker combines:

- rotation-aware Kalman filtering
- OBB IoU-based association
- optional appearance embeddings
- a deterministic color-histogram fallback when deep ReID features are unavailable

This allows track management that respects vessel orientation and visual similarity, improving stability compared with axis-aligned tracking logic.

### 5. Video pipeline and logging

The main demo/integration script is [`src/pipeline_1.py`](/home/acer/workspace/projects/amvit/src/pipeline_1.py:1). It ties together:

- TensorRT detector loading
- OBB DeepSORT tracking
- frame-by-frame annotation
- track trails and per-object overlays
- JSONL track logging
- output video generation

By default it reads a test video from `data/video/`, writes annotated video to `outputs/`, and can save serialized tracking records for later analysis.

## Repository Structure

```text
.
├── configs/
│   ├── vessel.yaml          # dataset and class configuration
│   ├── train.yaml           # base training configuration
│   └── curriculum.yaml      # staged curriculum training configuration
├── src/
│   ├── pipeline_1.py        # end-to-end detection + tracking video pipeline
│   ├── data/                # dataset utilities, augmentation, inspection
│   ├── model/               # training, export, TensorRT inference, analysis
│   └── tracking/            # OBB geometry, Kalman filter, DeepSORT tracker
├── data/                    # local datasets and videos, ignored by git
├── models/                  # local weights, engines, exports, ignored by git
├── outputs/                 # runtime outputs, ignored by git
├── Ship_Detection/          # legacy remote branch content
└── Object_Tracking/         # legacy remote branch content
```

## Configuration and Data Layout

The active dataset configuration expects a YOLO-style layout rooted at `data/raw`:

```text
data/raw/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The codebase intentionally keeps large data, trained weights, TensorRT engines, videos, and generated outputs outside version control. Those paths are covered in [`.gitignore`](/home/acer/workspace/projects/amvit/.gitignore:1).

## Typical Workflow

### Train a detector

Use the training scripts under `src/model/` to fit an oriented vessel detector using the dataset described in `configs/vessel.yaml` and the hyperparameters in `configs/train.yaml`.

### Run curriculum experiments

Use `src/model/train_curriculum.py` when you want phase-based training, interpolation of augmentation settings, or class-balancing experiments.

### Export and optimize

Use the model export and TensorRT tooling under `src/model/` to produce optimized inference assets for deployment on NVIDIA hardware.

### Run video tracking

Use `src/pipeline_1.py` to execute the full detection-and-tracking pipeline on a video source and generate:

- annotated output video
- per-frame tracking logs
- basic runtime statistics such as detection count, track count, and processing FPS

## Dependencies

The code references the following major libraries and runtimes:

- Python 3
- OpenCV
- NumPy
- Ultralytics YOLO
- MLflow
- PyYAML
- SciPy
- PyCUDA
- TensorRT

Some deployment features, especially TensorRT inference, require a correctly configured NVIDIA CUDA environment.

## Notes

- The repository currently mixes preserved legacy content with the newer `src/` pipeline. The new code should be treated as the active path for further development.
- Large assets such as datasets, `.pt` weights, `.onnx` exports, `.engine` files, MLflow state, and runtime outputs are intentionally not committed.
- The project is structured to support both experimentation and deployment, not just offline notebook-style training.

## Status

This repository is actively evolving from earlier ship detection and tracking experiments into a cleaner, deployment-oriented aerial maritime surveillance stack built around oriented detection and tracking.
