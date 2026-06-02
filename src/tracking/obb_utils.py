from __future__ import annotations

from typing import Any, Iterable, Sequence

import cv2
import numpy as np


ANGLE_PERIOD = np.pi
ANGLE_LIMIT = np.pi / 2.0
EPSILON = 1e-7

XYWHR_SIZE = 5
CornerArray = np.ndarray
RotatedRect = tuple[tuple[float, float], tuple[float, float], float]


def normalize_angle(theta: float | np.ndarray) -> float | np.ndarray:
    """
    Normalize OBB angles to [-pi/2, pi/2).

    OBB orientations are pi-periodic: theta and theta + pi describe the same
    rotated rectangle.
    """
    normalized = (np.asarray(theta, dtype=np.float64) + ANGLE_LIMIT) % ANGLE_PERIOD - ANGLE_LIMIT
    if normalized.ndim == 0:
        return float(normalized)
    return normalized


def angle_difference(theta: float, reference: float) -> float:
    """Return the shortest pi-periodic angular residual theta - reference."""
    return float(normalize_angle(float(theta) - float(reference)))


def radians_to_degrees(theta_rad: float | np.ndarray) -> float | np.ndarray:
    degrees = np.degrees(theta_rad)
    if np.asarray(degrees).ndim == 0:
        return float(degrees)
    return degrees


def degrees_to_radians(theta_deg: float | np.ndarray) -> float | np.ndarray:
    radians = np.radians(theta_deg)
    if np.asarray(radians).ndim == 0:
        return float(radians)
    return radians


def as_xywhr(
    xywhr: Sequence[float] | np.ndarray,
    *,
    angle_in_degrees: bool = False,
    normalize: bool = True,
    min_box_size: float = 0.0,
) -> np.ndarray:
    """
    Validate an OBB as [x, y, w, h, theta] and return float64 radians.

    The TensorRT detector and tracker use theta in radians. Set
    angle_in_degrees=True only for external boxes that are already in OpenCV's
    degree convention.
    """
    values = np.asarray(xywhr, dtype=np.float64).reshape(-1)
    if values.size != XYWHR_SIZE:
        raise ValueError(f"Expected xywhr with 5 values, got shape {np.asarray(xywhr).shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"xywhr contains non-finite values: {values}")
    if values[2] < min_box_size or values[3] < min_box_size:
        raise ValueError(
            f"OBB width and height must be >= {min_box_size}, got w={values[2]}, h={values[3]}"
        )

    values = values.copy()
    if angle_in_degrees:
        values[4] = np.radians(values[4])
    if normalize:
        values[4] = normalize_angle(values[4])
    return values


def as_xywhr_array(
    boxes_xywhr: Sequence[Sequence[float]] | np.ndarray,
    *,
    angle_in_degrees: bool = False,
    normalize: bool = True,
    min_box_size: float = 0.0,
) -> np.ndarray:
    """Validate an array of OBBs and return shape [N, 5] in radians."""
    boxes = np.asarray(boxes_xywhr, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, XYWHR_SIZE), dtype=np.float64)
    if boxes.ndim == 1:
        if boxes.size != XYWHR_SIZE:
            raise ValueError(f"Expected one xywhr box with 5 values, got shape {boxes.shape}")
        boxes = boxes.reshape(1, XYWHR_SIZE)
    elif boxes.ndim != 2 or boxes.shape[1] != XYWHR_SIZE:
        raise ValueError(f"Expected boxes with shape [N, 5], got {boxes.shape}")
    if not np.all(np.isfinite(boxes)):
        raise ValueError("boxes_xywhr contains non-finite values")
    if np.any(boxes[:, 2] < min_box_size) or np.any(boxes[:, 3] < min_box_size):
        raise ValueError(f"All OBB widths and heights must be >= {min_box_size}")

    boxes = boxes.copy()
    if angle_in_degrees:
        boxes[:, 4] = np.radians(boxes[:, 4])
    if normalize:
        boxes[:, 4] = normalize_angle(boxes[:, 4])
    return boxes


def canonicalize_xywhr(
    xywhr: Sequence[float] | np.ndarray,
    *,
    prefer_long_width: bool = False,
    angle_in_degrees: bool = False,
) -> tuple[float, float, float, float, float]:
    """
    Return a canonical radians OBB tuple.

    prefer_long_width=True swaps width/height so width is the longer side. This
    is useful for geometry normalization, but the tracker should normally keep
    detector dimensions unchanged to avoid unnecessary angle jumps.
    """
    x, y, width, height, theta = as_xywhr(xywhr, angle_in_degrees=angle_in_degrees)
    if prefer_long_width and width < height:
        width, height = height, width
        theta = normalize_angle(theta + ANGLE_LIMIT)
    return float(x), float(y), float(width), float(height), float(theta)


def xywhr_to_corners(x: float, y: float, w: float, h: float, theta_rad: float) -> CornerArray:
    """
    Convert OBB center/size/rotation to four corner points.

    Args:
        x, y: box center in pixels.
        w, h: box width and height in pixels.
        theta_rad: rotation angle in radians.

    Returns:
        Float32 array with shape [4, 2], ordered around the rectangle.
    """
    box = as_xywhr((x, y, w, h, theta_rad))
    x, y, w, h, theta_rad = box
    cos_a = np.cos(theta_rad)
    sin_a = np.sin(theta_rad)
    half_w = w / 2.0
    half_h = h / 2.0

    corners = np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float64,
    )
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    corners = corners @ rotation.T
    corners += np.array([x, y], dtype=np.float64)
    return corners.astype(np.float32)


def obb_to_corners(
    xywhr: Sequence[float] | np.ndarray,
    *,
    angle_in_degrees: bool = False,
) -> CornerArray:
    """Convert an [x, y, w, h, theta] OBB to four corner points."""
    x, y, w, h, theta = as_xywhr(xywhr, angle_in_degrees=angle_in_degrees)
    return xywhr_to_corners(float(x), float(y), float(w), float(h), float(theta))


def xywhr_to_rotated_rect(
    xywhr: Sequence[float] | np.ndarray,
    *,
    angle_in_degrees: bool = False,
) -> RotatedRect:
    """
    Convert an OBB to OpenCV RotatedRect format.

    OpenCV expects ((center_x, center_y), (width, height), angle_degrees).
    """
    x, y, w, h, theta = as_xywhr(xywhr, angle_in_degrees=angle_in_degrees)
    return ((float(x), float(y)), (float(w), float(h)), float(np.degrees(theta)))


def rotated_rect_to_xywhr(
    rotated_rect: RotatedRect,
    *,
    output_degrees: bool = False,
    normalize: bool = True,
) -> tuple[float, float, float, float, float]:
    """Convert an OpenCV RotatedRect to project [x, y, w, h, theta]."""
    (x, y), (w, h), angle_deg = rotated_rect
    theta = float(np.radians(angle_deg))
    if normalize:
        theta = float(normalize_angle(theta))
    if output_degrees:
        theta = float(np.degrees(theta))
    return float(x), float(y), float(w), float(h), theta


def corners_to_xywhr(
    corners: Sequence[Sequence[float]] | np.ndarray,
    *,
    output_degrees: bool = False,
) -> tuple[float, float, float, float, float]:
    """Fit the minimum-area OBB around four or more points."""
    points = as_corners(corners)
    rect = cv2.minAreaRect(points.astype(np.float32))
    return rotated_rect_to_xywhr(rect, output_degrees=output_degrees)


def as_corners(corners: Sequence[Sequence[float]] | np.ndarray) -> CornerArray:
    """Validate corner/point arrays as float32 shape [N, 2]."""
    points = np.asarray(corners, dtype=np.float32)
    if points.ndim == 3 and points.shape[1] == 1 and points.shape[2] == 2:
        points = points.reshape(-1, 2)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError(f"Expected points with shape [N, 2], got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("corners contains non-finite values")
    return points


def order_corners_clockwise(corners: Sequence[Sequence[float]] | np.ndarray) -> CornerArray:
    """Return points ordered clockwise around their centroid."""
    points = as_corners(corners).astype(np.float64)
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    ordered = points[np.argsort(angles)]
    return ordered.astype(np.float32)


def polygon_area(points: Sequence[Sequence[float]] | np.ndarray) -> float:
    """Compute polygon area with the shoelace formula."""
    points = order_corners_clockwise(points).astype(np.float64)
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def obb_area(xywhr: Sequence[float] | np.ndarray) -> float:
    """Return the area of an OBB."""
    _, _, w, h, _ = as_xywhr(xywhr, min_box_size=0.0)
    return float(max(w, 0.0) * max(h, 0.0))


def obb_intersection_area(
    det1: Sequence[float] | np.ndarray,
    det2: Sequence[float] | np.ndarray,
    *,
    angle_in_degrees: bool = False,
) -> float:
    """Compute exact intersection area between two rotated rectangles."""
    box1 = as_xywhr(det1, angle_in_degrees=angle_in_degrees, min_box_size=0.0)
    box2 = as_xywhr(det2, angle_in_degrees=angle_in_degrees, min_box_size=0.0)
    area1 = float(box1[2] * box1[3])
    area2 = float(box2[2] * box2[3])
    if area1 <= EPSILON or area2 <= EPSILON:
        return 0.0

    rect1 = xywhr_to_rotated_rect(box1)
    rect2 = xywhr_to_rotated_rect(box2)
    status, intersection = cv2.rotatedRectangleIntersection(rect1, rect2)

    if status == cv2.INTERSECT_NONE:
        return 0.0
    if intersection is None:
        return min(area1, area2) if status == cv2.INTERSECT_FULL else 0.0

    hull = cv2.convexHull(intersection.astype(np.float32))
    return float(max(cv2.contourArea(hull), 0.0))


def obb_iou(
    det1: Sequence[float] | np.ndarray,
    det2: Sequence[float] | np.ndarray,
    *,
    angle_in_degrees: bool = False,
) -> float:
    """
    Compute exact IoU between two OBBs.

    Boxes use project convention [x, y, w, h, theta_rad] by default. Set
    angle_in_degrees=True for external/OpenCV-style degree inputs.
    """
    box1 = as_xywhr(det1, angle_in_degrees=angle_in_degrees, min_box_size=0.0)
    box2 = as_xywhr(det2, angle_in_degrees=angle_in_degrees, min_box_size=0.0)
    area1 = float(box1[2] * box1[3])
    area2 = float(box2[2] * box2[3])
    if area1 <= EPSILON or area2 <= EPSILON:
        return 0.0

    inter_area = obb_intersection_area(box1, box2)
    union_area = area1 + area2 - inter_area
    if union_area <= EPSILON:
        return 0.0
    return float(np.clip(inter_area / union_area, 0.0, 1.0))


def pairwise_obb_iou(
    boxes1: Sequence[Sequence[float]] | np.ndarray,
    boxes2: Sequence[Sequence[float]] | np.ndarray,
    *,
    angle_in_degrees: bool = False,
) -> np.ndarray:
    """Return an [N, M] IoU matrix for two OBB arrays."""
    boxes1 = as_xywhr_array(boxes1, angle_in_degrees=angle_in_degrees, min_box_size=0.0)
    boxes2 = as_xywhr_array(boxes2, angle_in_degrees=angle_in_degrees, min_box_size=0.0)
    ious = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            ious[i, j] = obb_iou(box1, box2)
    return ious


def get_obb_centroid(xywhr: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Return the (x, y) center of an OBB."""
    x, y, _, _, _ = as_xywhr(xywhr, min_box_size=0.0)
    return float(x), float(y)


def center_distance(
    det1: Sequence[float] | np.ndarray,
    det2: Sequence[float] | np.ndarray,
) -> float:
    """Return Euclidean distance between two OBB centers."""
    x1, y1 = get_obb_centroid(det1)
    x2, y2 = get_obb_centroid(det2)
    return float(np.hypot(x1 - x2, y1 - y2))


def pairwise_center_distance(
    boxes1: Sequence[Sequence[float]] | np.ndarray,
    boxes2: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return an [N, M] matrix of center distances."""
    boxes1 = as_xywhr_array(boxes1, min_box_size=0.0)
    boxes2 = as_xywhr_array(boxes2, min_box_size=0.0)
    centers1 = boxes1[:, :2]
    centers2 = boxes2[:, :2]
    diff = centers1[:, None, :] - centers2[None, :, :]
    return np.linalg.norm(diff, axis=2).astype(np.float32)


def obb_to_aabb(
    xywhr: Sequence[float] | np.ndarray,
    *,
    angle_in_degrees: bool = False,
) -> tuple[float, float, float, float]:
    """Convert an OBB to enclosing axis-aligned [x1, y1, x2, y2]."""
    corners = obb_to_corners(xywhr, angle_in_degrees=angle_in_degrees)
    x1 = float(np.min(corners[:, 0]))
    y1 = float(np.min(corners[:, 1]))
    x2 = float(np.max(corners[:, 0]))
    y2 = float(np.max(corners[:, 1]))
    return x1, y1, x2, y2


def clip_corners_to_image(
    corners: Sequence[Sequence[float]] | np.ndarray,
    image_shape: Sequence[int],
) -> CornerArray:
    """Clip corner coordinates to image bounds."""
    points = as_corners(corners).copy()
    if len(image_shape) < 2:
        raise ValueError(f"Expected image_shape with at least H,W, got {image_shape}")
    height = int(image_shape[0])
    width = int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image_shape: {image_shape}")
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points.astype(np.float32)


def rotated_nms(
    boxes_xywhr: Sequence[Sequence[float]] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    class_ids: Sequence[int] | np.ndarray | None = None,
    class_aware: bool = True,
    max_detections: int | None = None,
    angle_in_degrees: bool = False,
) -> list[int]:
    """
    Run OpenCV rotated NMS and return kept indices sorted by score descending.

    When class_ids are provided and class_aware=True, suppression is applied
    independently per class.
    """
    boxes = as_xywhr_array(boxes_xywhr, angle_in_degrees=angle_in_degrees, min_box_size=EPSILON)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores) != len(boxes):
        raise ValueError(f"scores length {len(scores)} does not match boxes length {len(boxes)}")
    if len(boxes) == 0:
        return []
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")

    if class_ids is None or not class_aware:
        groups: Iterable[np.ndarray] = [np.arange(len(boxes), dtype=np.int64)]
    else:
        classes = np.asarray(class_ids).reshape(-1)
        if len(classes) != len(boxes):
            raise ValueError(f"class_ids length {len(classes)} does not match boxes length {len(boxes)}")
        groups = (np.where(classes == cls_id)[0] for cls_id in np.unique(classes))

    selected: list[int] = []
    for group_indices in groups:
        rects = [xywhr_to_rotated_rect(boxes[index]) for index in group_indices]
        group_scores = [float(scores[index]) for index in group_indices]
        kept = cv2.dnn.NMSBoxesRotated(
            rects,
            group_scores,
            float(score_threshold),
            float(iou_threshold),
        )
        if len(kept) == 0:
            continue
        kept = np.asarray(kept).reshape(-1)
        selected.extend(int(group_indices[index]) for index in kept)

    selected.sort(key=lambda index: float(scores[index]), reverse=True)
    if max_detections is not None:
        selected = selected[: int(max_detections)]
    return selected


def detections_to_xywhr_array(detections: Sequence[dict[str, Any]]) -> np.ndarray:
    """Extract detector/tracker dictionaries into an [N, 5] OBB array."""
    if not detections:
        return np.empty((0, XYWHR_SIZE), dtype=np.float32)
    return as_xywhr_array([det["xywhr"] for det in detections]).astype(np.float32)


def detections_to_scores(detections: Sequence[dict[str, Any]]) -> np.ndarray:
    """Extract confidence scores from detector/tracker dictionaries."""
    return np.asarray([float(det.get("conf", 1.0)) for det in detections], dtype=np.float32)


def detections_to_classes(detections: Sequence[dict[str, Any]]) -> np.ndarray:
    """Extract class ids from detector/tracker dictionaries."""
    return np.asarray([int(det.get("cls", -1)) for det in detections], dtype=np.int32)


def draw_obb(
    image: np.ndarray,
    xywhr: Sequence[float] | np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: str | None = None,
) -> np.ndarray:
    """Draw one OBB on an image and return the same image object."""
    corners = obb_to_corners(xywhr).astype(np.int32)
    cv2.polylines(image, [corners], isClosed=True, color=color, thickness=thickness)
    if label:
        x, y = corners[0]
        cv2.putText(
            image,
            label,
            (int(x), max(15, int(y) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            max(1, thickness // 2),
            cv2.LINE_AA,
        )
    return image


__all__ = [
    "ANGLE_LIMIT",
    "ANGLE_PERIOD",
    "EPSILON",
    "XYWHR_SIZE",
    "angle_difference",
    "as_corners",
    "as_xywhr",
    "as_xywhr_array",
    "canonicalize_xywhr",
    "center_distance",
    "clip_corners_to_image",
    "corners_to_xywhr",
    "degrees_to_radians",
    "detections_to_classes",
    "detections_to_scores",
    "detections_to_xywhr_array",
    "draw_obb",
    "get_obb_centroid",
    "normalize_angle",
    "obb_area",
    "obb_intersection_area",
    "obb_iou",
    "obb_to_aabb",
    "obb_to_corners",
    "order_corners_clockwise",
    "pairwise_center_distance",
    "pairwise_obb_iou",
    "polygon_area",
    "radians_to_degrees",
    "rotated_nms",
    "rotated_rect_to_xywhr",
    "xywhr_to_corners",
    "xywhr_to_rotated_rect",
]
