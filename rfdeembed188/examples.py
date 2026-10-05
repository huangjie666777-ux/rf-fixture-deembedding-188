from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .network import cascade_s


def _network(prefix: str, frequency: np.ndarray) -> np.ndarray:
    normalized = (frequency - 1.0e9) / 1.0e9
    if prefix == "device":
        s11 = 0.10 * np.exp(1j * (0.20 + 0.30 * normalized))
        s22 = 0.22 * np.exp(1j * (-0.50 - 0.15 * normalized))
        s21 = 0.72 * np.exp(1j * (-1.10 - 2.40 * normalized))
        s12 = 0.64 * np.exp(1j * (0.80 + 1.70 * normalized))
    elif prefix == "left":
        s11 = 0.03 * np.exp(1j * (0.40 - normalized))
        s22 = 0.07 * np.exp(1j * (-0.80 + 0.50 * normalized))
        s21 = 0.90 * np.exp(1j * (-0.70 - 3.20 * normalized))
        s12 = 0.88 * np.exp(1j * (0.60 + 2.30 * normalized))
    else:
        s11 = 0.06 * np.exp(1j * (-0.20 + 0.70 * normalized))
        s22 = 0.02 * np.exp(1j * (0.90 - 0.40 * normalized))
        s21 = 0.86 * np.exp(1j * (-1.40 - 4.10 * normalized))
        s12 = 0.84 * np.exp(1j * (1.10 + 3.10 * normalized))
    return np.stack(
        [np.stack([s11, s12], axis=1), np.stack([s21, s22], axis=1)],
        axis=1,
    )


def _line(freq_value: float, matrix: np.ndarray, fmt: str) -> str:
    values = [matrix[0, 0], matrix[1, 0], matrix[0, 1], matrix[1, 1]]
    parts = [f"{freq_value:.12g}"]
    for value in values:
        if fmt == "RI":
            parts.extend([f"{value.real:.15g}", f"{value.imag:.15g}"])
        elif fmt == "MA":
            parts.extend([f"{abs(value):.15g}", f"{np.angle(value, deg=True):.15g}"])
        else:
            parts.extend([f"{20.0 * np.log10(abs(value)):.15g}", f"{np.angle(value, deg=True):.15g}"])
    return " ".join(parts)


def _write(path: Path, frequencies: np.ndarray, matrices: np.ndarray, unit: str, fmt: str) -> None:
    scale = {"Hz": 1.0, "MHz": 1.0e6, "GHz": 1.0e9}[unit]
    lines = [f"! rfdeembed188 example: {path.stem}", f"# {unit} S {fmt} R 50"]
    lines.extend(_line(frequency / scale, matrix, fmt) for frequency, matrix in zip(frequencies, matrices))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_examples(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    measurement_grid = np.linspace(1.0e9, 2.0e9, 12)
    left_grid = np.linspace(0.90e9, 2.10e9, 11)
    right_grid = np.linspace(0.80e9, 2.20e9, 13)
    device = _network("device", measurement_grid)
    left = _network("left", measurement_grid)
    right = _network("right", measurement_grid)
    measurement = np.stack(
        [cascade_s(cascade_s(left[index], device[index]), right[index]) for index in range(measurement_grid.size)]
    )
    _write(output_dir / "measurement.s2p", measurement_grid, measurement, "Hz", "RI")
    _write(output_dir / "device_expected.s2p", measurement_grid, device, "GHz", "RI")
    _write(output_dir / "left_fixture.s2p", left_grid, _network("left", left_grid), "MHz", "MA")
    _write(output_dir / "right_fixture.s2p", right_grid, _network("right", right_grid), "GHz", "DB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create rfdeembed188 example Touchstone files.")
    parser.add_argument("output_dir", nargs="?", default="examples", type=Path)
    args = parser.parse_args()
    create_examples(args.output_dir)


if __name__ == "__main__":
    main()
