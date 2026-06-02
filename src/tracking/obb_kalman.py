from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from filterpy.kalman import KalmanFilter as _FilterPyKalmanFilter
except ImportError:  # pragma: no cover - exercised only when filterpy is absent
    _FilterPyKalmanFilter = None


STATE_DIM = 10
MEASUREMENT_DIM = 5
ANGLE_PERIOD = np.pi
ANGLE_LIMIT = np.pi / 2.0
MIN_BOX_SIZE = 1e-3

DEFAULT_MEASUREMENT_NOISE = (2.0, 2.0, 4.0, 4.0, 0.1)
DEFAULT_POSITION_PROCESS_NOISE = 0.1
DEFAULT_VELOCITY_PROCESS_NOISE = 0.001
DEFAULT_POSITION_COVARIANCE = 10.0
DEFAULT_VELOCITY_COVARIANCE = 1000.0


class _NumpyKalmanFilter:
    """
    Minimal FilterPy-compatible linear Kalman filter fallback.

    The project guide installs filterpy on the deployment environment, but this
    fallback keeps the tracking module importable in lighter development shells.
    """

    def __init__(self, dim_x: int, dim_z: int) -> None:
        self.dim_x = int(dim_x)
        self.dim_z = int(dim_z)

        self.x = np.zeros((self.dim_x, 1), dtype=np.float64)
        self.P = np.eye(self.dim_x, dtype=np.float64)
        self.Q = np.eye(self.dim_x, dtype=np.float64)
        self.F = np.eye(self.dim_x, dtype=np.float64)
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float64)
        self.R = np.eye(self.dim_z, dtype=np.float64)
        self.B = None

        self.y = np.zeros((self.dim_z, 1), dtype=np.float64)
        self.S = np.zeros((self.dim_z, self.dim_z), dtype=np.float64)
        self.K = np.zeros((self.dim_x, self.dim_z), dtype=np.float64)
        self.z = None

    def predict(
        self,
        u: np.ndarray | float = 0.0,
        B: np.ndarray | None = None,
        F: np.ndarray | None = None,
        Q: np.ndarray | None = None,
    ) -> np.ndarray:
        F = self.F if F is None else np.asarray(F, dtype=np.float64)
        Q = self.Q if Q is None else np.asarray(Q, dtype=np.float64)
        B = self.B if B is None else B

        self.x = F @ self.x
        if B is not None:
            self.x = self.x + np.asarray(B, dtype=np.float64) @ np.asarray(u, dtype=np.float64)

        self.P = F @ self.P @ F.T + Q
        return self.x

    def update(
        self,
        z: Sequence[float] | np.ndarray,
        R: np.ndarray | None = None,
        H: np.ndarray | None = None,
    ) -> np.ndarray:
        z = _as_column(z, self.dim_z)
        H = self.H if H is None else np.asarray(H, dtype=np.float64)
        R = self.R if R is None else np.asarray(R, dtype=np.float64)

        self.y = z - H @ self.x
        pht = self.P @ H.T
        self.S = H @ pht + R
        self.K = pht @ np.linalg.pinv(self.S)
        self.x = self.x + self.K @ self.y

        identity = np.eye(self.dim_x, dtype=np.float64)
        ikh = identity - self.K @ H
        self.P = ikh @ self.P @ ikh.T + self.K @ R @ self.K.T
        self.z = z
        return self.x


_BaseKalmanFilter = _FilterPyKalmanFilter or _NumpyKalmanFilter


def normalize_angle(theta: float | np.ndarray) -> float | np.ndarray:
    """
    Normalize OBB angle to the half-open interval [-pi/2, pi/2).

    Rotated rectangles are pi-periodic: an angle shifted by pi describes the
    same physical orientation. Tracking in this canonical range avoids
    accumulating equivalent rotations.
    """
    normalized = (np.asarray(theta, dtype=np.float64) + ANGLE_LIMIT) % ANGLE_PERIOD - ANGLE_LIMIT
    if normalized.ndim == 0:
        return float(normalized)
    return normalized


def angle_difference(theta: float, reference: float) -> float:
    """Return the shortest pi-periodic angular residual theta - reference."""
    return float(normalize_angle(float(theta) - float(reference)))


def align_angle_to_reference(theta: float, reference: float) -> float:
    """
    Express theta in the branch closest to reference.

    Example: a prediction near +pi/2 and a measurement near -pi/2 are adjacent
    OBB orientations, not a 180 degree turn.
    """
    return float(reference) + angle_difference(theta, reference)


def as_xywhr(
    xywhr: Sequence[float] | np.ndarray,
    reference_angle: float | None = None,
    min_box_size: float = MIN_BOX_SIZE,
) -> np.ndarray:
    """
    Validate and normalize an OBB measurement as [x, y, w, h, theta].

    Width and height are expected in pixels, theta in radians.
    """
    values = np.asarray(xywhr, dtype=np.float64).reshape(-1)
    if values.size != MEASUREMENT_DIM:
        raise ValueError(f"Expected xywhr with 5 values, got shape {np.asarray(xywhr).shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"xywhr contains non-finite values: {values}")
    if values[2] < min_box_size or values[3] < min_box_size:
        raise ValueError(f"OBB width and height must be positive, got w={values[2]}, h={values[3]}")

    values = values.copy()
    if reference_angle is None:
        values[4] = normalize_angle(values[4])
    else:
        values[4] = align_angle_to_reference(values[4], reference_angle)
    return values


def state_to_xywhr(state: Sequence[float] | np.ndarray) -> tuple[float, float, float, float, float]:
    """Extract canonical [x, y, w, h, theta] from a Kalman state vector."""
    values = np.asarray(state, dtype=np.float64).reshape(-1)
    if values.size < MEASUREMENT_DIM:
        raise ValueError(f"State must contain at least 5 values, got shape {np.asarray(state).shape}")
    return (
        float(values[0]),
        float(values[1]),
        max(float(values[2]), MIN_BOX_SIZE),
        max(float(values[3]), MIN_BOX_SIZE),
        float(normalize_angle(values[4])),
    )


def _as_column(values: Sequence[float] | np.ndarray, length: int) -> np.ndarray:
    column = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if column.shape != (length, 1):
        raise ValueError(f"Expected column vector with length {length}, got shape {column.shape}")
    return column


def _diag(values: Sequence[float], length: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != length:
        raise ValueError(f"{name} must contain {length} values, got {array.size}")
    if np.any(array <= 0) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain positive finite values, got {array}")
    return np.diag(array)


class OBBKalmanFilter(_BaseKalmanFilter):
    """
    Constant-velocity Kalman filter for oriented bounding boxes.

    State:
        [x, y, w, h, theta, dx, dy, dw, dh, dtheta]

    Measurement:
        [x, y, w, h, theta]
    """

    def __init__(self, dt: float = 1.0, min_box_size: float = MIN_BOX_SIZE) -> None:
        super().__init__(dim_x=STATE_DIM, dim_z=MEASUREMENT_DIM)
        self.min_box_size = float(min_box_size)
        self.dt = float(dt)
        self._configure_transition(self.dt)

    def _configure_transition(self, dt: float) -> None:
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError(f"dt must be a positive finite value, got {dt}")

        self.dt = float(dt)
        self.F = np.eye(STATE_DIM, dtype=np.float64)
        for measurement_index in range(MEASUREMENT_DIM):
            self.F[measurement_index, measurement_index + MEASUREMENT_DIM] = self.dt

    def predict(self, *args, **kwargs) -> np.ndarray:
        result = super().predict(*args, **kwargs)
        self._sanitize_state()
        return result

    def update(
        self,
        z: Sequence[float] | np.ndarray,
        R: np.ndarray | None = None,
        H: np.ndarray | None = None,
    ) -> np.ndarray:
        if H is not None:
            result = super().update(z, R=R, H=H)
            self._sanitize_state()
            return result

        measurement = as_xywhr(
            z,
            reference_angle=float(np.asarray(self.x).reshape(-1)[4]),
            min_box_size=self.min_box_size,
        )
        result = super().update(measurement.reshape(MEASUREMENT_DIM, 1), R=R, H=self.H)
        self._sanitize_state()
        return result

    def to_xywhr(self) -> tuple[float, float, float, float, float]:
        return state_to_xywhr(self.x)

    def _sanitize_state(self) -> None:
        self.x = _as_column(self.x, STATE_DIM)
        self.x[2, 0] = max(float(self.x[2, 0]), self.min_box_size)
        self.x[3, 0] = max(float(self.x[3, 0]), self.min_box_size)
        self.x[4, 0] = normalize_angle(float(self.x[4, 0]))


def create_obb_kalman_filter(
    initial_xywhr: Sequence[float] | np.ndarray,
    dt: float = 1.0,
    measurement_noise: Sequence[float] = DEFAULT_MEASUREMENT_NOISE,
    position_process_noise: float = DEFAULT_POSITION_PROCESS_NOISE,
    velocity_process_noise: float = DEFAULT_VELOCITY_PROCESS_NOISE,
    position_covariance: float = DEFAULT_POSITION_COVARIANCE,
    velocity_covariance: float = DEFAULT_VELOCITY_COVARIANCE,
    min_box_size: float = MIN_BOX_SIZE,
) -> OBBKalmanFilter:
    """
    Create an OBB Kalman filter initialized from [x, y, w, h, theta].

    Noise values are covariances, not standard deviations. Defaults mirror the
    project guide sample: detector position is trusted more than dimensions,
    initial velocities are uncertain, and velocity process noise is small.
    """
    initial = as_xywhr(initial_xywhr, min_box_size=min_box_size)

    kf = OBBKalmanFilter(dt=dt, min_box_size=min_box_size)

    kf.H = np.zeros((MEASUREMENT_DIM, STATE_DIM), dtype=np.float64)
    kf.H[:MEASUREMENT_DIM, :MEASUREMENT_DIM] = np.eye(MEASUREMENT_DIM, dtype=np.float64)

    kf.R = _diag(measurement_noise, MEASUREMENT_DIM, "measurement_noise")

    if position_process_noise <= 0 or velocity_process_noise <= 0:
        raise ValueError("Process noise values must be positive")
    kf.Q = np.eye(STATE_DIM, dtype=np.float64) * float(position_process_noise)
    kf.Q[MEASUREMENT_DIM:, MEASUREMENT_DIM:] = (
        np.eye(MEASUREMENT_DIM, dtype=np.float64) * float(velocity_process_noise)
    )

    if position_covariance <= 0 or velocity_covariance <= 0:
        raise ValueError("Initial covariance values must be positive")
    kf.P = np.eye(STATE_DIM, dtype=np.float64) * float(position_covariance)
    kf.P[MEASUREMENT_DIM:, MEASUREMENT_DIM:] = (
        np.eye(MEASUREMENT_DIM, dtype=np.float64) * float(velocity_covariance)
    )

    kf.x = np.zeros((STATE_DIM, 1), dtype=np.float64)
    kf.x[:MEASUREMENT_DIM, 0] = initial
    return kf


def predict_obb(kf: OBBKalmanFilter) -> tuple[float, float, float, float, float]:
    """Run prediction and return the predicted OBB measurement state."""
    kf.predict()
    return kf.to_xywhr()


def update_obb(
    kf: OBBKalmanFilter,
    measurement_xywhr: Sequence[float] | np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Run an angle-aware measurement update and return the updated OBB state."""
    kf.update(measurement_xywhr)
    return kf.to_xywhr()


def project_obb(kf: OBBKalmanFilter) -> tuple[np.ndarray, np.ndarray]:
    """Project state distribution into measurement space."""
    mean = kf.H @ _as_column(kf.x, STATE_DIM)
    covariance = kf.H @ np.asarray(kf.P, dtype=np.float64) @ kf.H.T + np.asarray(kf.R, dtype=np.float64)
    mean[4, 0] = normalize_angle(float(mean[4, 0]))
    return mean.reshape(-1), covariance


def gating_distance(
    kf: OBBKalmanFilter,
    measurements_xywhr: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """
    Return squared Mahalanobis distances from this track to candidate OBBs.

    DeepSORT can use this for motion gating before Hungarian assignment.
    """
    measurements = np.asarray(measurements_xywhr, dtype=np.float64)
    if measurements.ndim == 1:
        measurements = measurements.reshape(1, -1)
    if measurements.shape[1] != MEASUREMENT_DIM:
        raise ValueError(f"Expected measurements with shape Nx5, got {measurements.shape}")

    mean, covariance = project_obb(kf)
    covariance_inv = np.linalg.pinv(covariance)
    distances = []
    for measurement in measurements:
        aligned = as_xywhr(measurement, reference_angle=float(mean[4]), min_box_size=kf.min_box_size)
        residual = aligned - mean
        residual[4] = angle_difference(aligned[4], mean[4])
        distances.append(float(residual.T @ covariance_inv @ residual))
    return np.asarray(distances, dtype=np.float64)
