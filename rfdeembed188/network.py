from __future__ import annotations

import numpy as np

from .errors import AlignmentError, NetworkComputationError
from .models import DeembeddingResult, TouchstoneData


TRANSMISSION_THRESHOLD = 1.0e-12
CONDITION_LIMIT = 1.0e10


def s_to_wave_transfer(s: np.ndarray) -> np.ndarray:
    s11, s12 = s[0]
    s21, s22 = s[1]
    return np.array(
        [
            [1.0 / s21, -s22 / s21],
            [s11 / s21, s12 - (s11 * s22 / s21)],
        ],
        dtype=np.complex128,
    )


def wave_transfer_to_s(t: np.ndarray) -> np.ndarray:
    t11, t12 = t[0]
    t21, t22 = t[1]
    return np.array(
        [
            [t21 / t11, np.linalg.det(t) / t11],
            [1.0 / t11, -t12 / t11],
        ],
        dtype=np.complex128,
    )


def cascade_s(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return wave_transfer_to_s(s_to_wave_transfer(left) @ s_to_wave_transfer(right))


def _validate_transmission(name: str, network: np.ndarray, frequency: float) -> None:
    if abs(network[1, 0]) <= TRANSMISSION_THRESHOLD:
        raise NetworkComputationError(
            f"{name} S21 magnitude is {abs(network[1, 0]):.3g}, which is not greater than 1e-12",
            frequency,
        )


def deembed(
    measurement: TouchstoneData,
    left_fixture: TouchstoneData,
    right_fixture: TouchstoneData,
    left_s: np.ndarray,
    right_s: np.ndarray,
    aligned_grid: np.ndarray,
) -> DeembeddingResult:
    device_s = np.empty_like(measurement.s_parameters)
    excluded = measurement.frequencies_hz.size - aligned_grid.size

    for index, frequency in enumerate(aligned_grid):
        measured_s = measurement.s_parameters[index]
        left = left_s[index]
        right = right_s[index]
        _validate_transmission("measurement", measured_s, frequency)
        _validate_transmission("left fixture", left, frequency)
        _validate_transmission("right fixture", right, frequency)

        try:
            measured_t = s_to_wave_transfer(measured_s)
            left_t = s_to_wave_transfer(left)
            right_t = s_to_wave_transfer(right)
            if not (np.all(np.isfinite(measured_t)) and np.all(np.isfinite(left_t)) and np.all(np.isfinite(right_t))):
                raise NetworkComputationError("S-to-transfer conversion produced non-finite values", frequency)

            left_condition = np.linalg.cond(left_t)
            right_condition = np.linalg.cond(right_t)
            if not (np.isfinite(left_condition) and np.isfinite(right_condition)):
                raise NetworkComputationError("fixture transfer matrix condition number is non-finite", frequency)
            if left_condition > CONDITION_LIMIT:
                raise NetworkComputationError(f"left fixture transfer matrix condition number {left_condition:.6g} exceeds 1e10", frequency)
            if right_condition > CONDITION_LIMIT:
                raise NetworkComputationError(f"right fixture transfer matrix condition number {right_condition:.6g} exceeds 1e10", frequency)

            right_inverse_device = np.linalg.solve(left_t, measured_t)
            device_t = np.linalg.solve(right_t.T, right_inverse_device.T).T
            if not np.all(np.isfinite(device_t)):
                raise NetworkComputationError("device transfer matrix contains non-finite values", frequency)
            if device_t[0, 0] == 0.0:
                raise NetworkComputationError("device transfer matrix cannot be converted to S-parameters", frequency)

            calculated_s = wave_transfer_to_s(device_t)
            if not np.all(np.isfinite(calculated_s)):
                raise NetworkComputationError("device S-parameters contain non-finite values", frequency)
            if abs(calculated_s[1, 0]) <= TRANSMISSION_THRESHOLD:
                raise NetworkComputationError(
                    f"device S21 magnitude is {abs(calculated_s[1, 0]):.3g}, which is not greater than 1e-12",
                    frequency,
                )
            device_s[index] = calculated_s
        except np.linalg.LinAlgError as exc:
            raise NetworkComputationError("matrix computation failed", frequency) from exc

    max_error = 0.0
    for index, frequency in enumerate(aligned_grid):
        reembedded = cascade_s(cascade_s(left_s[index], device_s[index]), right_s[index])
        if not np.all(np.isfinite(reembedded)):
            raise NetworkComputationError("re-embedding produced non-finite values", frequency)
        max_error = max(max_error, float(np.max(np.abs(reembedded - measurement.s_parameters[index]))))
    if not np.isfinite(max_error):
        raise NetworkComputationError("re-embedding error is non-finite", float(aligned_grid[0]))

    return DeembeddingResult(
        frequencies_hz=aligned_grid,
        device_s=device_s,
        reference_impedance=measurement.reference_impedance,
        max_reembed_error=max_error,
        measurement_point_count=measurement.frequencies_hz.size,
        excluded_measurement_points=excluded,
    )
