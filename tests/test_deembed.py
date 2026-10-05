from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from rfdeembed188.alignment import align_on_measurement_grid
from rfdeembed188.errors import AlignmentError, NetworkComputationError
from rfdeembed188.examples import create_examples
from rfdeembed188.main import app
from rfdeembed188.network import cascade_s
from rfdeembed188.pipeline import run_deembedding
from rfdeembed188.touchstone import parse_s2p, write_s2p


def test_parse_formats_and_interpolation(tmp_path: Path) -> None:
    frequencies = np.array([1.0e9, 2.0e9])
    s = np.broadcast_to(
        np.array([[0.1 + 0.2j, 0.3 + 0.4j], [0.5 + 0.6j, 0.7 + 0.8j]], dtype=np.complex128),
        (2, 2, 2),
    ).copy()
    for filename, frequency_unit, fmt in [
        ("ri.s2p", "GHz", "RI"),
        ("ma.s2p", "MHz", "MA"),
        ("db.s2p", "Hz", "DB"),
    ]:
        path = tmp_path / filename
        path.write_bytes(write_s2p(filename, frequencies, s, 50.0))
        lines = path.read_text().splitlines()
        lines[1] = f"# {frequency_unit} S {fmt} R 50"
        path.write_text("\n".join(lines) + "\n")
        parsed = parse_s2p(filename, path.read_bytes())
        if fmt == "RI":
            assert np.allclose(parsed.s_parameters, s, rtol=1e-10, atol=1e-10)
        else:
            expected = np.empty_like(s)
            for point in range(2):
                for row in range(2):
                    for column in range(2):
                        first = s[point, row, column].real
                        angle = np.deg2rad(s[point, row, column].imag)
                        magnitude = first if fmt == "MA" else 10.0 ** (first / 20.0)
                        expected[point, row, column] = magnitude * np.exp(1j * angle)
            assert np.allclose(parsed.s_parameters, expected, rtol=1e-10, atol=1e-10)


def test_rejects_invalid_file_with_file_and_line() -> None:
    content = "# Hz S RI R 50\n1e9 bad 0 0.9 0 0 0 0\n2e9 0 0 0.9 0 0 0 0\n"
    try:
        parse_s2p("bad.s2p", content.encode())
    except Exception as exc:
        assert exc.extra["file"] == "bad.s2p"
        assert exc.extra["line"] == 2
    else:
        raise AssertionError("invalid file was accepted")


def test_example_deembedding_and_reembed_error(tmp_path: Path) -> None:
    create_examples(tmp_path)
    result, diagnostics = run_deembedding(
        (tmp_path / "measurement.s2p").read_bytes(),
        (tmp_path / "left_fixture.s2p").read_bytes(),
        (tmp_path / "right_fixture.s2p").read_bytes(),
        "measurement.s2p",
        "left_fixture.s2p",
        "right_fixture.s2p",
        False,
        False,
    )
    assert result.frequencies_hz.size == 12
    assert result.max_reembed_error < 1e-12
    assert diagnostics["reference_impedance_ohm"] == 50.0


def test_port_swap_exchanges_rows_and_columns(tmp_path: Path) -> None:
    create_examples(tmp_path)
    fixture = parse_s2p("left_fixture.s2p", (tmp_path / "left_fixture.s2p").read_bytes())
    measurement = parse_s2p("measurement.s2p", (tmp_path / "measurement.s2p").read_bytes())
    _, aligned = align_on_measurement_grid(measurement, fixture, True)
    _, not_swapped = align_on_measurement_grid(measurement, fixture, False)
    assert np.allclose(aligned, not_swapped[:, ::-1, ::-1])


def test_http_returns_zip(tmp_path: Path) -> None:
    create_examples(tmp_path)
    client = TestClient(app)
    with (tmp_path / "measurement.s2p").open("rb") as measurement, (
        tmp_path / "left_fixture.s2p"
    ).open("rb") as left, (tmp_path / "right_fixture.s2p").open("rb") as right:
        response = client.post(
            "/deembed",
            files={
                "measurement": ("measurement.s2p", measurement, "text/plain"),
                "left_fixture": ("left_fixture.s2p", left, "text/plain"),
                "right_fixture": ("right_fixture.s2p", right, "text/plain"),
            },
            data={"left_swapped": "false", "right_swapped": "false"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.content[:2] == b"PK"


def test_rejects_extrapolation_with_frequency(tmp_path: Path) -> None:
    create_examples(tmp_path)
    measurement_bytes = (tmp_path / "measurement.s2p").read_bytes()
    right_bytes = (tmp_path / "right_fixture.s2p").read_bytes()
    shifted_grid = np.array([3.0e9, 4.0e9])
    zero_s = np.zeros((2, 2, 2), dtype=np.complex128)
    left_path = tmp_path / "shifted_left.s2p"
    left_path.write_bytes(write_s2p("shifted_left.s2p", shifted_grid, zero_s, 50.0))
    try:
        run_deembedding(
            measurement_bytes,
            left_path.read_bytes(),
            right_bytes,
            "measurement.s2p",
            "shifted_left.s2p",
            "right_fixture.s2p",
            False,
            False,
        )
    except AlignmentError as exc:
        assert exc.extra["frequency_hz"] == 1.0e9
    else:
        raise AssertionError("out-of-range fixture was accepted")


def test_rejects_non_transmitting_fixture_with_frequency(tmp_path: Path) -> None:
    create_examples(tmp_path)
    grid = np.linspace(1.0e9, 2.0e9, 12)
    zero_s = np.zeros((12, 2, 2), dtype=np.complex128)
    left_path = tmp_path / "zero_left.s2p"
    left_path.write_bytes(write_s2p("zero_left.s2p", grid, zero_s, 50.0))
    try:
        run_deembedding(
            (tmp_path / "measurement.s2p").read_bytes(),
            left_path.read_bytes(),
            (tmp_path / "right_fixture.s2p").read_bytes(),
            "measurement.s2p",
            "zero_left.s2p",
            "right_fixture.s2p",
            False,
            False,
        )
    except NetworkComputationError as exc:
        assert exc.extra["frequency_hz"] == 1.0e9
    else:
        raise AssertionError("non-transmitting fixture was accepted")
