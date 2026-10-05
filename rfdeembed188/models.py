from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TouchstoneData:
    filename: str
    frequencies_hz: np.ndarray
    s_parameters: np.ndarray
    reference_impedance: float
    sha256: str


@dataclass(frozen=True)
class DeembeddingResult:
    frequencies_hz: np.ndarray
    device_s: np.ndarray
    reference_impedance: float
    max_reembed_error: float
    measurement_point_count: int
    excluded_measurement_points: int
