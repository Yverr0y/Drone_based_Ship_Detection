from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model.trt_infer import (  # noqa: E402
    DEFAULT_DATA_CONFIG_PATH,
    DEFAULT_ENGINE_PATH,
    TRTVesselDetector,
    load_class_names,
)
from src.tracking.deepsort import (  # noqa: E402
    DEFAULT_REID_WEIGHTS_PATH,
    OBBDeepSORT,
)
from src.tracking.obb_utils import obb_to_corners  # noqa: E402


DEFAULT_INPUT_VIDEO = ROOT_DIR / "data" / "video" / "test_vid.mp4"
DEFAULT_OUTPUT_VIDEO = ROOT_DIR / "outputs" / "pipeline_1_tracked.mp4"
DEFAULT_TRACK_LOG = ROOT_DIR / "outputs" / "pipeline_1_tracks.jsonl"


def _resolve_path(path: str | Path, base_dir: Path = ROOT_DIR) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _safe_fps(fps: float, default: float = 30.0) -> float:
    if not np.isfinite(fps) or fps <= 1e-3:
        return default
    return float(fps)


def _open_video(input_path: Path) -> cv2.VideoCapture:
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")
    return capture


def _make_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    codecs = ["mp4v", "avc1", "H264"] if suffix == ".mp4" else ["XVID", "MJPG", "mp4v"]
    for codec in codecs:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(f"Failed to create video writer for: {output_path}")


def _track_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(track_id) * 9973)
    color = rng.integers(64, 256, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def _draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(int(x), frame.shape[1] - text_w - 4))
    y = max(text_h + 4, min(int(y), frame.shape[0] - baseline - 4))
    cv2.rectangle(
        frame,
        (x, y - text_h - baseline - 4),
        (x + text_w + 4, y + baseline + 2),
        bg_color,
        thickness=-1,
    )
    cv2.putText(frame, text, (x + 2, y - 2), font, scale, color, thickness, cv2.LINE_AA)


def _points_from_track(track: dict[str, Any]) -> np.ndarray:
    points = track.get("points")
    if points is None:
        points = obb_to_corners(track["xywhr"])
    return np.asarray(points, dtype=np.int32).reshape(-1, 2)


def _draw_tracks(
    frame: np.ndarray,
    tracks: Sequence[dict[str, Any]],
    trails: dict[int, deque[tuple[int, int]]],
    *,
    draw_trails: bool = True,
) -> np.ndarray:
    annotated = frame.copy()

    for track in tracks:
        track_id = int(track["track_id"])
        points = _points_from_track(track)
        color = _track_color(track_id)
        xywhr = track["xywhr"]
        center = (int(round(float(xywhr[0]))), int(round(float(xywhr[1]))))

        cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=2)
        cv2.circle(annotated, center, 3, color, thickness=-1)

        trails[track_id].append(center)
        if draw_trails and len(trails[track_id]) > 1:
            trail_points = np.asarray(trails[track_id], dtype=np.int32)
            cv2.polylines(annotated, [trail_points], isClosed=False, color=color, thickness=2)

        name = str(track.get("name", track.get("cls", "vessel")))
        conf = float(track.get("conf", 0.0))
        label = f"ID {track_id} {name} {conf:.2f}"
        text_origin = (int(points[:, 0].min()), int(points[:, 1].min()) - 4)
        _draw_text(annotated, label, text_origin, color=(255, 255, 255), bg_color=color)

    return annotated


def _draw_status(
    frame: np.ndarray,
    *,
    frame_index: int,
    total_frames: int,
    detections_count: int,
    tracks_count: int,
    fps: float,
    processing_fps: float,
) -> None:
    if total_frames > 0:
        frame_text = f"Frame {frame_index}/{total_frames}"
    else:
        frame_text = f"Frame {frame_index}"
    status = (
        f"{frame_text} | det {detections_count} | tracks {tracks_count} | "
        f"video {fps:.1f} FPS | proc {processing_fps:.1f} FPS"
    )
    _draw_text(
        frame,
        status,
        (10, 26),
        color=(255, 255, 255),
        bg_color=(30, 30, 30),
        scale=0.6,
        thickness=1,
    )


def _serialize_track(track: dict[str, Any]) -> dict[str, Any]:
    return {
        "track_id": int(track["track_id"]),
        "xywhr": [float(value) for value in track["xywhr"]],
        "cls": int(track.get("cls", -1)),
        "name": track.get("name"),
        "conf": float(track.get("conf", 0.0)),
        "state": track.get("state"),
        "age": int(track.get("age", 0)),
        "hits": int(track.get("hits", 0)),
        "time_since_update": int(track.get("time_since_update", 0)),
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    log_path = (
        None
        if args.save_jsonl is None or str(args.save_jsonl).lower() in {"", "none", "null", "false"}
        else _resolve_path(args.save_jsonl)
    )

    class_names = load_class_names(args.data_config)
    detector = TRTVesselDetector(
        engine_path=args.engine,
        conf_thresh=args.conf,
        iou_thresh=args.det_iou,
        class_names=class_names,
        max_det=args.max_det,
        verbose=args.verbose_trt,
    )

    tracker = OBBDeepSORT(
        max_age=args.max_age,
        n_init=args.n_init,
        iou_threshold=args.track_iou,
        max_cosine_distance=args.max_cosine_distance,
        appearance_weight=args.appearance_weight,
        min_confidence=args.min_track_conf,
        class_aware=not args.class_agnostic,
        appearance_model=args.appearance_model,
        reid_weights_path=args.reid_weights,
        device=args.reid_device,
        return_stale_tracks=args.draw_stale,
    )

    capture = _open_video(input_path)
    input_fps = _safe_fps(capture.get(cv2.CAP_PROP_FPS), default=args.fallback_fps)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid input video dimensions: {width}x{height}")

    output_fps = input_fps if args.output_fps <= 0 else float(args.output_fps)
    writer = _make_video_writer(output_path, output_fps, (width, height))

    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")

    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=args.trail_length))
    frame_index = 0
    written_frames = 0
    total_detections = 0
    total_tracks = 0
    start_time = time.perf_counter()
    last_report = start_time

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            if args.max_frames > 0 and frame_index > args.max_frames:
                frame_index -= 1
                break

            if args.frame_stride > 1 and (frame_index - 1) % args.frame_stride != 0:
                continue

            frame_start = time.perf_counter()
            detections = detector.infer(frame)
            tracks = tracker.update(
                detections,
                frame=frame,
                include_unconfirmed=args.draw_unconfirmed,
                include_stale=args.draw_stale,
            )
            processing_fps = 1.0 / max(time.perf_counter() - frame_start, 1e-9)

            annotated = _draw_tracks(
                frame,
                tracks,
                trails,
                draw_trails=not args.no_trails,
            )
            _draw_status(
                annotated,
                frame_index=frame_index,
                total_frames=total_frames,
                detections_count=len(detections),
                tracks_count=len(tracks),
                fps=input_fps,
                processing_fps=processing_fps,
            )

            writer.write(annotated)
            written_frames += 1
            total_detections += len(detections)
            total_tracks += len(tracks)

            if log_file is not None:
                log_file.write(
                    json.dumps(
                        {
                            "frame": frame_index,
                            "time_sec": frame_index / input_fps,
                            "detections": len(detections),
                            "tracks": [_serialize_track(track) for track in tracks],
                        }
                    )
                    + "\n"
                )

            if args.display:
                cv2.imshow("Vessel Identification + Tracking", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            now = time.perf_counter()
            if now - last_report >= args.report_interval:
                elapsed = now - start_time
                avg_fps = written_frames / max(elapsed, 1e-9)
                print(
                    f"frame={frame_index} written={written_frames} "
                    f"det={len(detections)} tracks={len(tracks)} avg_fps={avg_fps:.2f}",
                    flush=True,
                )
                last_report = now
    finally:
        capture.release()
        writer.release()
        if log_file is not None:
            log_file.close()
        if args.display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "track_log": str(log_path) if log_path is not None else None,
        "frames_read": frame_index,
        "frames_written": written_frames,
        "input_fps": input_fps,
        "output_fps": output_fps,
        "width": width,
        "height": height,
        "avg_processing_fps": written_frames / max(elapsed, 1e-9),
        "avg_detections_per_frame": total_detections / max(written_frames, 1),
        "avg_tracks_per_frame": total_tracks / max(written_frames, 1),
        "appearance_extractor": type(tracker.feature_extractor).__name__
        if tracker.feature_extractor is not None
        else None,
    }
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end vessel identification and tracking on an MP4 video "
            "using the YOLOv8n OBB TensorRT engine plus OBB DeepSORT."
        )
    )

    parser.add_argument("--input", default=str(DEFAULT_INPUT_VIDEO), help="Input MP4 video path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_VIDEO), help="Output annotated video path.")
    parser.add_argument(
        "--save-jsonl",
        default=str(DEFAULT_TRACK_LOG),
        help="Optional JSONL track log path. Use 'none' to disable.",
    )

    parser.add_argument("--engine", default=str(DEFAULT_ENGINE_PATH), help="TensorRT engine path.")
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG_PATH), help="Dataset YAML with class names.")
    parser.add_argument("--conf", type=float, default=0.4, help="Detector confidence threshold.")
    parser.add_argument("--det-iou", type=float, default=0.5, help="Detector rotated NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per frame.")

    parser.add_argument("--max-age", type=int, default=30, help="Tracker max missed frames before deletion.")
    parser.add_argument("--n-init", type=int, default=3, help="Hits required before track confirmation.")
    parser.add_argument("--track-iou", type=float, default=0.3, help="Tracker rotated IoU fallback threshold.")
    parser.add_argument("--max-cosine-distance", type=float, default=0.35, help="Appearance matching threshold.")
    parser.add_argument("--appearance-weight", type=float, default=0.7, help="Appearance-vs-IoU blended cost weight.")
    parser.add_argument("--min-track-conf", type=float, default=0.0, help="Tracker-side detection confidence filter.")
    parser.add_argument("--class-agnostic", action="store_true", help="Allow cross-class track matching.")

    parser.add_argument(
        "--appearance-model",
        default="auto",
        choices=["auto", "mobilenet", "histogram", "none"],
        help="Appearance model for DeepSORT ReID.",
    )
    parser.add_argument(
        "--reid-weights",
        default=None,
        help=f"MobileNetV3 ReID checkpoint path. Default uses {DEFAULT_REID_WEIGHTS_PATH} when present.",
    )
    parser.add_argument("--reid-device", default="auto", help="ReID device: auto, cuda:0, or cpu.")

    parser.add_argument("--output-fps", type=float, default=0.0, help="Override output FPS; <=0 keeps input FPS.")
    parser.add_argument("--fallback-fps", type=float, default=30.0, help="FPS used when the input metadata is invalid.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many input frames; 0 means all.")
    parser.add_argument("--trail-length", type=int, default=30, help="Track center trail length.")
    parser.add_argument("--no-trails", action="store_true", help="Disable track center trails.")
    parser.add_argument("--draw-unconfirmed", action="store_true", help="Draw tentative tracks too.")
    parser.add_argument("--draw-stale", action="store_true", help="Draw predicted tracks during missed frames.")
    parser.add_argument("--display", action="store_true", help="Show the annotated video while processing.")
    parser.add_argument("--report-interval", type=float, default=2.0, help="Progress print interval in seconds.")
    parser.add_argument("--verbose-trt", action="store_true", help="Enable verbose TensorRT logs.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.trail_length < 1:
        raise ValueError("--trail-length must be >= 1")
    run_pipeline(args)


if __name__ == "__main__":
    main()
