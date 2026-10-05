from __future__ import annotations

import io
import json
import zipfile

import numpy as np

from .alignment import align_on_measurement_grid
from .errors import AlignmentError, RFDeembedError
from .models import DeembeddingResult, TouchstoneData
from .network import deembed
from .touchstone import parse_s2p, write_s2p


def _require_common_impedance(networks: list[TouchstoneData]) -> float:
    reference = networks[0].reference_impedance
    for network in networks[1:]:
        if not np.isclose(network.reference_impedance, reference, rtol=0.0, atol=0.0):
            raise RFDeembedError(
                f"reference impedances differ: {networks[0].filename} has {reference:g}, "
                f"{network.filename} has {network.reference_impedance:g}"
            )
    return reference


def run_deembedding(
    measurement_raw: bytes,
    left_raw: bytes,
    right_raw: bytes,
    measurement_name: str,
    left_name: str,
    right_name: str,
    left_swapped: bool,
    right_swapped: bool,
) -> tuple[DeembeddingResult, dict[str, object]]:
    measurement = parse_s2p(measurement_name, measurement_raw)
    left_fixture = parse_s2p(left_name, left_raw)
    right_fixture = parse_s2p(right_name, right_raw)
    reference_impedance = _require_common_impedance([measurement, left_fixture, right_fixture])

    grid = measurement.frequencies_hz
    joint_coverage = (
        (grid >= left_fixture.frequencies_hz[0])
        & (grid <= left_fixture.frequencies_hz[-1])
        & (grid >= right_fixture.frequencies_hz[0])
        & (grid <= right_fixture.frequencies_hz[-1])
    )
    if not np.all(joint_coverage):
        first_missing = int(np.flatnonzero(~joint_coverage)[0])
        raise AlignmentError(
            "fixtures do not jointly cover every measurement frequency; extrapolation is not allowed",
            float(grid[first_missing]),
        )

    aligned_grid, left_s = align_on_measurement_grid(measurement, left_fixture, left_swapped)
    _, right_s = align_on_measurement_grid(measurement, right_fixture, right_swapped)
    result = deembed(measurement, left_fixture, right_fixture, left_s, right_s, aligned_grid)

    diagnostics = {
        "project": "rfdeembed188",
        "reference_impedance_ohm": reference_impedance,
        "frequency": {
            "unit": "Hz",
            "point_count": int(result.frequencies_hz.size),
            "minimum_hz": float(result.frequencies_hz[0]),
            "maximum_hz": float(result.frequencies_hz[-1]),
        },
        "port_orientation": {
            "left_fixture_swapped": bool(left_swapped),
            "right_fixture_swapped": bool(right_swapped),
        },
        "input_sha256": {
            "measurement": {"filename": measurement.filename, "sha256": measurement.sha256},
            "left_fixture": {"filename": left_fixture.filename, "sha256": left_fixture.sha256},
            "right_fixture": {"filename": right_fixture.filename, "sha256": right_fixture.sha256},
        },
        "max_reembed_complex_absolute_error": result.max_reembed_error,
    }
    return result, diagnostics


def build_result_zip(result: DeembeddingResult, diagnostics: dict[str, object]) -> bytes:
    s2p_bytes = write_s2p(
        "device.s2p",
        result.frequencies_hz,
        result.device_s,
        result.reference_impedance,
    )
    json_bytes = (
        json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("device.s2p", s2p_bytes)
        archive.writestr("diagnostics.json", json_bytes)
    return buffer.getvalue()
