from __future__ import annotations

import hashlib
import re

import numpy as np

from .errors import TouchstoneParseError
from .models import TouchstoneData


FREQUENCY_SCALES = {"HZ": 1.0, "KHZ": 1.0e3, "MHZ": 1.0e6, "GHZ": 1.0e9}
OPTION_RE = re.compile(r"^#\s+(\S+)\s+S\s+(\S+)\s+R\s+(\S+)\s*$", re.IGNORECASE)


def _line_number_for_byte_offset(data: bytes, offset: int) -> int:
    return data.count(b"\n", 0, max(0, offset)) + 1


def _parse_complex(tokens: list[str], component_start: int, fmt: str, filename: str, line_number: int) -> complex:
    try:
        first = float(tokens[component_start])
        second = float(tokens[component_start + 1])
    except (ValueError, IndexError) as exc:
        raise TouchstoneParseError(filename, line_number, "invalid numeric value") from exc

    if not (np.isfinite(first) and np.isfinite(second)):
        raise TouchstoneParseError(filename, line_number, "numeric values must be finite")

    if fmt == "RI":
        return complex(first, second)
    if fmt == "MA":
        magnitude, angle_degrees = first, second
    else:
        magnitude = 10.0 ** (first / 20.0)
        angle_degrees = second
    if not np.isfinite(magnitude):
        raise TouchstoneParseError(filename, line_number, "decibel magnitude produced a non-finite value")
    return complex(magnitude * np.exp(1j * np.deg2rad(angle_degrees)))


def parse_s2p(filename: str, raw: bytes) -> TouchstoneData:
    if not filename.lower().endswith(".s2p"):
        raise TouchstoneParseError(filename, None, "file name must use the .s2p extension")

    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        line = _line_number_for_byte_offset(raw, exc.start)
        raise TouchstoneParseError(filename, line, "file is not valid UTF-8 text") from exc

    scale: float | None = None
    fmt: str | None = None
    reference_impedance: float | None = None
    option_seen = False
    frequencies: list[float] = []
    matrices: list[np.ndarray] = []

    for line_number, original_line in enumerate(text.splitlines(), start=1):
        line = original_line.split("!", 1)[0].strip()
        if not line:
            continue

        if line.upper().startswith("#"):
            if option_seen:
                raise TouchstoneParseError(filename, line_number, "duplicate option line")
            if frequencies:
                raise TouchstoneParseError(filename, line_number, "option line must precede data")
            match = OPTION_RE.match(line)
            if not match:
                raise TouchstoneParseError(
                    filename,
                    line_number,
                    "complete option line required: # <Hz/kHz/MHz/GHz> S <RI/MA/DB> R <impedance>",
                )
            unit, parsed_fmt, impedance_text = (value.upper() for value in match.groups())
            if unit not in FREQUENCY_SCALES:
                raise TouchstoneParseError(filename, line_number, "unsupported frequency unit")
            if parsed_fmt not in {"RI", "MA", "DB"}:
                raise TouchstoneParseError(filename, line_number, "unsupported S-parameter format")
            try:
                impedance = float(impedance_text)
            except ValueError as exc:
                raise TouchstoneParseError(filename, line_number, "invalid reference impedance") from exc
            if not np.isfinite(impedance) or impedance <= 0.0:
                raise TouchstoneParseError(filename, line_number, "reference impedance must be positive and finite")
            scale = FREQUENCY_SCALES[unit]
            fmt = parsed_fmt
            reference_impedance = impedance
            option_seen = True
            continue

        if not option_seen:
            raise TouchstoneParseError(filename, line_number, "option line is required before data")

        tokens = line.split()
        if len(tokens) != 9:
            raise TouchstoneParseError(
                filename,
                line_number,
                "expected frequency followed by S11, S21, S12 and S22 complex pairs",
            )
        try:
            frequency = float(tokens[0]) * scale
        except ValueError as exc:
            raise TouchstoneParseError(filename, line_number, "invalid frequency value") from exc
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise TouchstoneParseError(filename, line_number, "frequency must be positive and finite")
        if frequencies and frequency <= frequencies[-1]:
            raise TouchstoneParseError(filename, line_number, "frequencies must be strictly increasing")

        values = [_parse_complex(tokens, index, fmt, filename, line_number) for index in (1, 3, 5, 7)]
        matrix = np.array([[values[0], values[2]], [values[1], values[3]]], dtype=np.complex128)
        frequencies.append(float(frequency))
        matrices.append(matrix)

    if not option_seen:
        raise TouchstoneParseError(filename, None, "missing option line")
    if not 2 <= len(frequencies) <= 20000:
        raise TouchstoneParseError(filename, None, "each file must contain 2 to 20000 frequency points")

    s_array = np.asarray(matrices, dtype=np.complex128)
    if not np.all(np.isfinite(s_array)):
        raise TouchstoneParseError(filename, None, "S-parameters must be finite")

    return TouchstoneData(
        filename=filename,
        frequencies_hz=np.asarray(frequencies, dtype=np.float64),
        s_parameters=s_array,
        reference_impedance=float(reference_impedance),
        sha256=digest,
    )


def write_s2p(filename: str, frequencies_hz: np.ndarray, s_parameters: np.ndarray, reference_impedance: float) -> bytes:
    lines = ["! Generated by rfdeembed188", f"# Hz S RI R {reference_impedance:g}"]
    for frequency, matrix in zip(frequencies_hz, s_parameters):
        s11, s12 = matrix[0]
        s21, s22 = matrix[1]
        lines.append(
            f"{frequency:.12g} {s11.real:.15g} {s11.imag:.15g} "
            f"{s21.real:.15g} {s21.imag:.15g} "
            f"{s12.real:.15g} {s12.imag:.15g} "
            f"{s22.real:.15g} {s22.imag:.15g}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")
