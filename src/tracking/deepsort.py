from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import numpy as np

from .obb_kalman import create_obb_kalman_filter, gating_distance
from .obb_utils import (
    EPSILON,
    as_xywhr,
    obb_iou,
    obb_to_corners,
    pairwise_obb_iou,
)


TENTATIVE = 0
CONFIRMED = 1
DELETED = 2

STATE_NAMES = {
    TENTATIVE: "tentative",
    CONFIRMED: "confirmed",
    DELETED: "deleted",
}

INFTY_COST = 1e5
CHI2INV95_5D = 11.070497693516351
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REID_WEIGHTS_PATH = ROOT_DIR / "models" / "reid" / "mobilenet_v3_small_imagenet.pth"


class AppearanceExtractor(Protocol):
    """Interface used by OBBDeepSORT for ReID embeddings."""

    def extract(self, frame: np.ndarray, detections: Sequence[dict[str, Any]]) -> np.ndarray:
        ...


def _l2_normalize(features: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=axis, keepdims=True)
    return features / np.maximum(norms, eps)


def _normalize_feature(feature: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if feature is None:
        return None
    values = np.asarray(feature, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    norm = float(np.linalg.norm(values))
    if norm <= EPSILON:
        return None
    return values / norm


def _linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost_matrix)
    except Exception:
        return _greedy_linear_assignment(cost_matrix)


def _greedy_linear_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cost_matrix.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    pairs = [
        (float(cost_matrix[row, col]), row, col)
        for row in range(cost_matrix.shape[0])
        for col in range(cost_matrix.shape[1])
    ]
    pairs.sort(key=lambda item: item[0])

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    rows: list[int] = []
    cols: list[int] = []
    for _, row, col in pairs:
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        rows.append(row)
        cols.append(col)
        if len(used_rows) == cost_matrix.shape[0] or len(used_cols) == cost_matrix.shape[1]:
            break
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def _torchvision_checkpoint_exists(url: str) -> bool:
    filename = Path(url).name
    candidates = []

    torch_home = Path.home() / ".cache" / "torch"
    candidates.append(torch_home / "hub" / "checkpoints" / filename)

    env_torch_home = None
    try:
        import os

        env_torch_home = os.environ.get("TORCH_HOME")
    except Exception:
        env_torch_home = None

    if env_torch_home:
        candidates.append(Path(env_torch_home).expanduser() / "hub" / "checkpoints" / filename)

    return any(path.exists() for path in candidates)


def extract_rotated_crop(
    frame: np.ndarray,
    xywhr: Sequence[float],
    *,
    padding: float = 0.12,
    output_size: tuple[int, int] | None = None,
    min_size: int = 8,
) -> np.ndarray | None:
    """
    Extract an orientation-aligned crop from a BGR frame.

    The crop is built with a perspective transform from the OBB corners, so the
    vessel's long axis is consistently aligned for appearance comparison.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected BGR frame with shape HxWx3, got {None if frame is None else frame.shape}")

    x, y, width, height, theta = as_xywhr(xywhr, min_box_size=EPSILON)
    width = max(width * (1.0 + float(padding)), float(min_size))
    height = max(height * (1.0 + float(padding)), float(min_size))

    corners = obb_to_corners((x, y, width, height, theta)).astype(np.float32)
    crop_w = max(int(round(width)), min_size)
    crop_h = max(int(round(height)), min_size)
    destination = np.array(
        [
            [0.0, 0.0],
            [crop_w - 1.0, 0.0],
            [crop_w - 1.0, crop_h - 1.0],
            [0.0, crop_h - 1.0],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(corners, destination)
    crop = cv2.warpPerspective(
        frame,
        transform,
        (crop_w, crop_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if crop.size == 0:
        return None
    if output_size is not None:
        out_w, out_h = output_size
        crop = cv2.resize(crop, (int(out_w), int(out_h)), interpolation=cv2.INTER_LINEAR)
    return crop


class ColorHistAppearanceExtractor:
    """
    Deterministic appearance fallback for environments without ReID weights.

    It is not a deep network, but it gives the tracker useful color, texture,
    and shape cues while the MobileNet ReID backbone is optional.
    """

    def __init__(
        self,
        crop_size: tuple[int, int] = (96, 96),
        padding: float = 0.12,
        hist_bins: tuple[int, int, int] = (16, 8, 8),
    ) -> None:
        self.crop_size = crop_size
        self.padding = float(padding)
        self.hist_bins = hist_bins
        self.feature_dim = sum(hist_bins) + 64 + 8 + 3

    def extract(self, frame: np.ndarray, detections: Sequence[dict[str, Any]]) -> np.ndarray:
        if not detections:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        features = []
        for detection in detections:
            crop = extract_rotated_crop(
                frame,
                detection["xywhr"],
                padding=self.padding,
                output_size=self.crop_size,
            )
            features.append(self._describe_crop(crop, detection["xywhr"]))
        return _l2_normalize(np.asarray(features, dtype=np.float32))

    def _describe_crop(self, crop: np.ndarray | None, xywhr: Sequence[float]) -> np.ndarray:
        if crop is None or crop.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [self.hist_bins[0]], [0, 180]).reshape(-1)
        s_hist = cv2.calcHist([hsv], [1], None, [self.hist_bins[1]], [0, 256]).reshape(-1)
        v_hist = cv2.calcHist([hsv], [2], None, [self.hist_bins[2]], [0, 256]).reshape(-1)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        gray_small = (gray_small - gray_small.mean()) / (gray_small.std() + 1e-6)

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=False)
        bins = np.floor((angle % np.pi) / np.pi * 8).astype(np.int32)
        grad_hist = np.bincount(
            bins.reshape(-1),
            weights=magnitude.reshape(-1),
            minlength=8,
        ).astype(np.float32)

        _, _, width, height, theta = as_xywhr(xywhr, min_box_size=EPSILON)
        shape = np.asarray(
            [
                np.log(width / max(height, EPSILON)),
                np.log(width * height + 1.0),
                theta,
            ],
            dtype=np.float32,
        )

        feature = np.concatenate(
            [
                h_hist.astype(np.float32),
                s_hist.astype(np.float32),
                v_hist.astype(np.float32),
                gray_small.reshape(-1).astype(np.float32),
                grad_hist.astype(np.float32),
                shape,
            ]
        )
        return feature


class MobileNetV3ReIDExtractor:
    """
    Lightweight CNN ReID backbone for vessel crops.

    Use weights_path for a vessel-specific ReID checkpoint when available. With
    imagenet_pretrained=True, torchvision will load cached ImageNet weights or
    download them according to the local torch/torchvision behavior.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        device: str = "auto",
        input_size: tuple[int, int] = (128, 128),
        padding: float = 0.12,
        imagenet_pretrained: bool = False,
        batch_size: int = 32,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except ImportError as exc:
            raise RuntimeError(
                "MobileNetV3ReIDExtractor requires torch and torchvision. "
                "Use appearance_model='histogram' or install the PyTorch stack."
            ) from exc

        self.torch = torch
        self.input_size = input_size
        self.padding = float(padding)
        self.batch_size = int(batch_size)

        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        weights = MobileNet_V3_Small_Weights.DEFAULT if imagenet_pretrained else None
        base_model = mobilenet_v3_small(weights=weights)
        if weights_path is not None:
            self._load_weights(base_model, Path(weights_path).expanduser())

        self.model = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.model.to(self.device)
        self.model.eval()
        self.feature_dim = 576
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _load_weights(self, model: Any, weights_path: Path) -> None:
        if not weights_path.exists():
            raise FileNotFoundError(f"ReID weights not found: {weights_path}")

        checkpoint = self.torch.load(str(weights_path), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        cleaned = {}
        for key, value in state_dict.items():
            for prefix in ("module.", "model.", "backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            cleaned[key] = value

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if len(cleaned) == 0 or (missing and len(missing) == len(model.state_dict())):
            raise RuntimeError(
                f"Could not load usable ReID weights from {weights_path}; "
                f"unexpected keys={len(unexpected)}"
            )

    def extract(self, frame: np.ndarray, detections: Sequence[dict[str, Any]]) -> np.ndarray:
        if not detections:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        tensors = []
        for detection in detections:
            crop = extract_rotated_crop(
                frame,
                detection["xywhr"],
                padding=self.padding,
                output_size=self.input_size,
            )
            if crop is None:
                crop = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensors.append(tensor)

        features = []
        with self.torch.no_grad():
            for start in range(0, len(tensors), self.batch_size):
                batch = self.torch.stack(tensors[start : start + self.batch_size]).to(self.device)
                batch = (batch - self.mean) / self.std
                output = self.model(batch)
                output = self.torch.nn.functional.normalize(output, p=2, dim=1)
                features.append(output.detach().cpu().numpy())

        return np.concatenate(features, axis=0).astype(np.float32)


def build_appearance_extractor(
    appearance_model: str | AppearanceExtractor | None = "auto",
    *,
    reid_weights_path: str | Path | None = None,
    device: str = "auto",
    imagenet_pretrained: bool = False,
) -> AppearanceExtractor | None:
    if appearance_model is None or appearance_model == "none":
        return None
    if hasattr(appearance_model, "extract"):
        return appearance_model

    model_name = str(appearance_model).lower()
    if model_name in {"hist", "histogram", "color", "fallback"}:
        return ColorHistAppearanceExtractor()

    resolved_weights_path = Path(reid_weights_path).expanduser() if reid_weights_path is not None else None
    if resolved_weights_path is None and DEFAULT_REID_WEIGHTS_PATH.exists():
        resolved_weights_path = DEFAULT_REID_WEIGHTS_PATH

    if model_name == "mobilenet":
        return MobileNetV3ReIDExtractor(
            weights_path=resolved_weights_path,
            device=device,
            imagenet_pretrained=imagenet_pretrained and resolved_weights_path is None,
        )

    if model_name != "auto":
        raise ValueError(f"Unsupported appearance_model: {appearance_model}")

    if resolved_weights_path is not None:
        return MobileNetV3ReIDExtractor(weights_path=resolved_weights_path, device=device)

    try:
        from torchvision.models import MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.DEFAULT
        if imagenet_pretrained or _torchvision_checkpoint_exists(weights.url):
            return MobileNetV3ReIDExtractor(device=device, imagenet_pretrained=True)
    except Exception:
        pass

    return ColorHistAppearanceExtractor()


@dataclass
class MatchResult:
    matches: list[tuple[int, int]]
    unmatched_tracks: list[int]
    unmatched_detections: list[int]


class OBBTrack:
    """Single DeepSORT track with OBB Kalman state and appearance gallery."""

    def __init__(
        self,
        detection: dict[str, Any],
        track_id: int,
        *,
        n_init: int = 3,
        max_feature_history: int = 100,
        feature_alpha: float = 0.9,
        dt: float = 1.0,
    ) -> None:
        self.track_id = int(track_id)
        self.n_init = int(n_init)
        self.state = CONFIRMED if self.n_init <= 1 else TENTATIVE

        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        self.cls = int(detection.get("cls", -1))
        self.conf = float(detection.get("conf", 1.0))
        self.name = detection.get("name")
        self.last_detection = dict(detection)

        self.kf = create_obb_kalman_filter(detection["xywhr"], dt=dt)
        self.features: deque[np.ndarray] = deque(maxlen=int(max_feature_history))
        self.smooth_feature: np.ndarray | None = None
        self.feature_alpha = float(feature_alpha)
        self.update_feature(detection.get("feature"))

    def predict(self) -> None:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: dict[str, Any]) -> None:
        self.kf.update(detection["xywhr"])
        self.hits += 1
        self.time_since_update = 0

        self.cls = int(detection.get("cls", self.cls))
        self.conf = float(detection.get("conf", self.conf))
        self.name = detection.get("name", self.name)
        self.last_detection = dict(detection)
        self.update_feature(detection.get("feature"))

        if self.state == TENTATIVE and self.hits >= self.n_init:
            self.state = CONFIRMED

    def update_feature(self, feature: Sequence[float] | np.ndarray | None) -> None:
        normalized = _normalize_feature(feature)
        if normalized is None:
            return

        self.features.append(normalized)
        if self.smooth_feature is None:
            self.smooth_feature = normalized
        else:
            alpha = self.feature_alpha
            self.smooth_feature = _normalize_feature(alpha * self.smooth_feature + (1.0 - alpha) * normalized)

    def appearance_distance(self, feature: Sequence[float] | np.ndarray | None) -> float | None:
        normalized = _normalize_feature(feature)
        if normalized is None:
            return None

        candidates = list(self.features)
        if self.smooth_feature is not None:
            candidates.append(self.smooth_feature)
        if not candidates:
            return None

        gallery = np.stack(candidates, axis=0)
        similarities = gallery @ normalized
        return float(1.0 - np.clip(np.max(similarities), -1.0, 1.0))

    def mark_missed(self, max_age: int) -> None:
        if self.state == TENTATIVE:
            self.mark_deleted()
        elif self.time_since_update > max_age:
            self.mark_deleted()

    def mark_deleted(self) -> None:
        self.state = DELETED

    def is_tentative(self) -> bool:
        return self.state == TENTATIVE

    def is_confirmed(self) -> bool:
        return self.state == CONFIRMED

    def is_deleted(self) -> bool:
        return self.state == DELETED

    def has_appearance(self) -> bool:
        return bool(self.features) or self.smooth_feature is not None

    def get_xywhr(self) -> tuple[float, float, float, float, float]:
        return self.kf.to_xywhr()

    def to_dict(self, *, include_feature: bool = False) -> dict[str, Any]:
        xywhr = self.get_xywhr()
        output: dict[str, Any] = {
            "track_id": self.track_id,
            "xywhr": xywhr,
            "points": obb_to_corners(xywhr).astype(float).round(2).tolist(),
            "cls": self.cls,
            "conf": self.conf,
            "state": STATE_NAMES[self.state],
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
        }
        if self.name is not None:
            output["name"] = self.name
        if include_feature and self.smooth_feature is not None:
            output["feature"] = self.smooth_feature.copy()
        return output


class OBBDeepSORT:
    """
    DeepSORT tracker adapted for YOLOv8 OBB vessel detections.

    Inputs are detection dictionaries from src/model/trt_infer.py:
        {"xywhr": (x, y, w, h, theta_rad), "conf": float, "cls": int}

    When update(..., frame=...) receives a frame, ReID appearance features are
    extracted from orientation-aligned vessel crops. Matching uses appearance
    distance plus Kalman gating for confirmed tracks and rotated IoU fallback
    for tentative or recently unmatched tracks.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        iou_threshold: float = 0.3,
        max_cosine_distance: float = 0.35,
        appearance_weight: float = 0.7,
        max_gating_distance: float = CHI2INV95_5D,
        nn_budget: int = 100,
        min_confidence: float = 0.0,
        class_aware: bool = True,
        dt: float = 1.0,
        appearance_model: str | AppearanceExtractor | None = "auto",
        reid_weights_path: str | Path | None = None,
        imagenet_pretrained: bool = False,
        device: str = "auto",
        return_stale_tracks: bool = False,
    ) -> None:
        if max_age < 1:
            raise ValueError("max_age must be >= 1")
        if n_init < 1:
            raise ValueError("n_init must be >= 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if not 0.0 <= max_cosine_distance <= 2.0:
            raise ValueError("max_cosine_distance must be in [0, 2]")
        if not 0.0 <= appearance_weight <= 1.0:
            raise ValueError("appearance_weight must be in [0, 1]")

        self.max_age = int(max_age)
        self.n_init = int(n_init)
        self.iou_threshold = float(iou_threshold)
        self.max_cosine_distance = float(max_cosine_distance)
        self.appearance_weight = float(appearance_weight)
        self.max_gating_distance = float(max_gating_distance)
        self.nn_budget = int(nn_budget)
        self.min_confidence = float(min_confidence)
        self.class_aware = bool(class_aware)
        self.dt = float(dt)
        self.return_stale_tracks = bool(return_stale_tracks)

        self.feature_extractor = build_appearance_extractor(
            appearance_model,
            reid_weights_path=reid_weights_path,
            device=device,
            imagenet_pretrained=imagenet_pretrained,
        )
        self.tracks: list[OBBTrack] = []
        self._next_track_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_track_id = 1

    def predict(self) -> None:
        for track in self.tracks:
            track.predict()

    def update(
        self,
        detections: Sequence[dict[str, Any]],
        frame: np.ndarray | None = None,
        *,
        include_unconfirmed: bool = False,
        include_stale: bool | None = None,
        include_feature: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Advance the tracker by one frame and return active tracks.

        Pass the BGR frame to enable appearance/ReID matching. Without a frame,
        the tracker falls back to OBB IoU + Kalman motion.
        """
        detections = self._prepare_detections(detections, frame)

        self.predict()

        if detections:
            matches, unmatched_tracks, unmatched_detections = self._match(detections)
            for track_index, detection_index in matches:
                self.tracks[track_index].update(detections[detection_index])

            for track_index in unmatched_tracks:
                self.tracks[track_index].mark_missed(self.max_age)

            for detection_index in unmatched_detections:
                self._initiate_track(detections[detection_index])
        else:
            for track in self.tracks:
                track.mark_missed(self.max_age)

        self.tracks = [track for track in self.tracks if not track.is_deleted()]
        return self.active_tracks(
            include_unconfirmed=include_unconfirmed,
            include_stale=self.return_stale_tracks if include_stale is None else include_stale,
            include_feature=include_feature,
        )

    def active_tracks(
        self,
        *,
        include_unconfirmed: bool = False,
        include_stale: bool = False,
        include_feature: bool = False,
    ) -> list[dict[str, Any]]:
        outputs = []
        for track in self.tracks:
            if track.is_deleted():
                continue
            if not include_unconfirmed and not track.is_confirmed():
                continue
            if not include_stale and track.time_since_update > 0:
                continue
            outputs.append(track.to_dict(include_feature=include_feature))
        return outputs

    def _prepare_detections(
        self,
        detections: Sequence[dict[str, Any]],
        frame: np.ndarray | None,
    ) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for detection in detections:
            if "xywhr" not in detection:
                raise KeyError("Each detection must contain an 'xywhr' field")
            conf = float(detection.get("conf", 1.0))
            if conf < self.min_confidence:
                continue

            item = dict(detection)
            item["xywhr"] = tuple(float(value) for value in as_xywhr(item["xywhr"], min_box_size=EPSILON))
            item["conf"] = conf
            item["cls"] = int(item.get("cls", -1))
            item["feature"] = _normalize_feature(item.get("feature"))
            prepared.append(item)

        missing_indices = [index for index, detection in enumerate(prepared) if detection["feature"] is None]
        if missing_indices and frame is not None and self.feature_extractor is not None:
            missing_detections = [prepared[index] for index in missing_indices]
            extracted = self.feature_extractor.extract(frame, missing_detections)
            if len(extracted) != len(missing_indices):
                raise RuntimeError(
                    "Feature extractor returned the wrong number of embeddings: "
                    f"expected={len(missing_indices)}, got={len(extracted)}"
                )
            for index, feature in zip(missing_indices, extracted):
                prepared[index]["feature"] = _normalize_feature(feature)

        return prepared

    def _match(self, detections: list[dict[str, Any]]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        confirmed = [index for index, track in enumerate(self.tracks) if track.is_confirmed()]
        unconfirmed = [index for index, track in enumerate(self.tracks) if not track.is_confirmed()]
        detection_indices = list(range(len(detections)))

        appearance_available = self._appearance_available(confirmed, detections)
        if appearance_available:
            matches_a, unmatched_confirmed, unmatched_detections = self._matching_cascade(
                confirmed,
                detection_indices,
                detections,
            )
            iou_candidates = unconfirmed + [
                index
                for index in unmatched_confirmed
                if self.tracks[index].time_since_update == 1
            ]
        else:
            matches_a = []
            unmatched_detections = detection_indices
            iou_candidates = unconfirmed + confirmed

        matches_b, unmatched_iou_tracks, unmatched_detections = self._min_cost_matching(
            self._iou_cost,
            iou_candidates,
            unmatched_detections,
            detections,
        )

        matched_tracks = {track_index for track_index, _ in matches_a + matches_b}
        all_track_indices = set(range(len(self.tracks)))
        unmatched_tracks = sorted(all_track_indices - matched_tracks)
        return matches_a + matches_b, unmatched_tracks, unmatched_detections

    def _matching_cascade(
        self,
        track_indices: list[int],
        detection_indices: list[int],
        detections: list[dict[str, Any]],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        unmatched_detections = list(detection_indices)
        unmatched_tracks: list[int] = []
        matches: list[tuple[int, int]] = []
        evaluated_tracks: set[int] = set()

        for level in range(self.max_age):
            if not unmatched_detections:
                break
            track_indices_l = [
                index
                for index in track_indices
                if self.tracks[index].time_since_update == 1 + level
            ]
            if not track_indices_l:
                continue

            evaluated_tracks.update(track_indices_l)
            result = self._min_cost_matching(
                self._appearance_cost,
                track_indices_l,
                unmatched_detections,
                detections,
            )
            matches.extend(result[0])
            unmatched_tracks.extend(result[1])
            unmatched_detections = result[2]

        not_evaluated = [index for index in track_indices if index not in evaluated_tracks]
        unmatched_tracks.extend(not_evaluated)
        return matches, sorted(set(unmatched_tracks)), unmatched_detections

    def _min_cost_matching(
        self,
        distance_metric,
        track_indices: list[int],
        detection_indices: list[int],
        detections: list[dict[str, Any]],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not track_indices or not detection_indices:
            return [], list(track_indices), list(detection_indices)

        cost_matrix = distance_metric(track_indices, detection_indices, detections)
        row_indices, col_indices = _linear_sum_assignment(cost_matrix)

        unmatched_tracks = set(track_indices)
        unmatched_detections = set(detection_indices)
        matches: list[tuple[int, int]] = []

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] >= INFTY_COST:
                continue
            track_index = track_indices[int(row)]
            detection_index = detection_indices[int(col)]
            matches.append((track_index, detection_index))
            unmatched_tracks.discard(track_index)
            unmatched_detections.discard(detection_index)

        return matches, sorted(unmatched_tracks), sorted(unmatched_detections)

    def _appearance_cost(
        self,
        track_indices: list[int],
        detection_indices: list[int],
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        cost = np.full((len(track_indices), len(detection_indices)), INFTY_COST, dtype=np.float32)
        detection_boxes = [detections[index]["xywhr"] for index in detection_indices]

        for row, track_index in enumerate(track_indices):
            track = self.tracks[track_index]
            try:
                motion_distances = gating_distance(track.kf, detection_boxes)
            except Exception:
                motion_distances = np.full(len(detection_indices), INFTY_COST, dtype=np.float64)

            for col, detection_index in enumerate(detection_indices):
                detection = detections[detection_index]
                if self._class_mismatch(track, detection):
                    continue
                if motion_distances[col] > self.max_gating_distance:
                    continue

                iou = obb_iou(track.get_xywhr(), detection["xywhr"])
                appearance = track.appearance_distance(detection.get("feature"))
                if appearance is None:
                    if iou < self.iou_threshold:
                        continue
                    cost[row, col] = 1.0 - iou
                    continue
                if appearance > self.max_cosine_distance:
                    continue

                iou_cost = 1.0 - iou
                cost[row, col] = (
                    self.appearance_weight * appearance
                    + (1.0 - self.appearance_weight) * iou_cost
                )
        return cost

    def _iou_cost(
        self,
        track_indices: list[int],
        detection_indices: list[int],
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        cost = np.full((len(track_indices), len(detection_indices)), INFTY_COST, dtype=np.float32)
        if not track_indices or not detection_indices:
            return cost

        track_boxes = [self.tracks[index].get_xywhr() for index in track_indices]
        detection_boxes = [detections[index]["xywhr"] for index in detection_indices]
        ious = pairwise_obb_iou(track_boxes, detection_boxes)

        for row, track_index in enumerate(track_indices):
            track = self.tracks[track_index]
            for col, detection_index in enumerate(detection_indices):
                detection = detections[detection_index]
                if self._class_mismatch(track, detection):
                    continue
                iou = float(ious[row, col])
                if iou < self.iou_threshold:
                    continue
                cost[row, col] = 1.0 - iou
        return cost

    def _appearance_available(
        self,
        track_indices: Sequence[int],
        detections: Sequence[dict[str, Any]],
    ) -> bool:
        if not track_indices or not detections:
            return False
        has_track_features = any(self.tracks[index].has_appearance() for index in track_indices)
        has_detection_features = any(detection.get("feature") is not None for detection in detections)
        return has_track_features and has_detection_features

    def _class_mismatch(self, track: OBBTrack, detection: dict[str, Any]) -> bool:
        if not self.class_aware:
            return False
        det_cls = int(detection.get("cls", -1))
        return track.cls >= 0 and det_cls >= 0 and track.cls != det_cls

    def _initiate_track(self, detection: dict[str, Any]) -> None:
        track = OBBTrack(
            detection,
            self._next_track_id,
            n_init=self.n_init,
            max_feature_history=self.nn_budget,
            dt=self.dt,
        )
        self.tracks.append(track)
        self._next_track_id += 1


__all__ = [
    "CONFIRMED",
    "DELETED",
    "DEFAULT_REID_WEIGHTS_PATH",
    "TENTATIVE",
    "AppearanceExtractor",
    "ColorHistAppearanceExtractor",
    "MobileNetV3ReIDExtractor",
    "OBBDeepSORT",
    "OBBTrack",
    "build_appearance_extractor",
    "extract_rotated_crop",
]
