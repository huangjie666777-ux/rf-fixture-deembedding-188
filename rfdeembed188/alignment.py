from __future__ import annotations

import numpy as np

from .errors import AlignmentError
from .models import TouchstoneData


def swap_ports(s_parameters: np.ndarray) -> np.ndarray:
    return s_parameters[:, ::-1, ::-1]


def align_on_measurement_grid(
    measurement: TouchstoneData,
    fixture: TouchstoneData,
    swap: bool,
) -> tuple[np.ndarray, np.ndarray]:
    grid = measurement.frequencies_hz
    fixture_grid = fixture.frequencies_hz
    coverage = (grid >= fixture_grid[0]) & (grid <= fixture_grid[-1])
    if not np.any(coverage):
        raise AlignmentError(
            f"fixture '{fixture.filename}' has no frequency point in common with the measurement",
            float(grid[0]),
        )

    aligned_grid = grid[coverage]
    fixture_s = swap_ports(fixture.s_parameters) if swap else fixture.s_parameters
    aligned = np.empty((aligned_grid.size, 2, 2), dtype=np.complex128)
    for row in range(2):
        for column in range(2):
            values = fixture_s[:, row, column]
            aligned[:, row, column] = np.interp(aligned_grid, fixture_grid, values.real) + (
                1j * np.interp(aligned_grid, fixture_grid, values.imag)
            )
    return aligned_grid, aligned
