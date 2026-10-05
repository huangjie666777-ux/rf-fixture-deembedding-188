# rfdeembed188

Two-port RF fixture de-embedding backend. It accepts three Touchstone 1.0 `.s2p` files (combined measurement, left fixture, right fixture), aligns fixtures to the measurement frequency grid, solves the wave-transfer matrix cascade, and returns the device network plus diagnostics.

## Requirements

Use the project virtual environment:

```bash
.venv/bin/python --version
```

The locked stack is Python 3.10, FastAPI 0.115.12, NumPy 2.2.6, Uvicorn, python-multipart, pytest, and httpx.

## Run

```bash
.venv/bin/uvicorn rfdeembed188.main:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

## API

`POST /deembed` is a `multipart/form-data` request:

- `measurement`: combined measured two-port `.s2p`
- `left_fixture`: left-fixture `.s2p`
- `right_fixture`: right-fixture `.s2p`
- `left_swapped`: `true` or `false`; swaps fixture rows and columns before alignment
- `right_swapped`: `true` or `false`; swaps fixture rows and columns before alignment

The response is a ZIP containing `device.s2p` and `diagnostics.json`. The output Touchstone uses the original retained measurement grid, Hz units, RI format, and the common positive reference impedance.

Example download:

```bash
curl -sS -o result.zip \
  -F measurement=@examples/measurement.s2p \
  -F left_fixture=@examples/left_fixture.s2p \
  -F right_fixture=@examples/right_fixture.s2p \
  -F left_swapped=false \
  -F right_swapped=false \
  http://127.0.0.1:8000/deembed
.venv/bin/python -m zipfile -l result.zip
```

## Examples

The example DUT is asymmetric (`S21 != S12` and `S11 != S22`), and the left and right fixtures differ. Measurement, fixture, and expected DUT files can be regenerated with:

```bash
.venv/bin/python -m rfdeembed188.examples examples
```

## Validation

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q rfdeembed188 tests
```

The parser supports Touchstone option-line frequency units Hz/kHz/MHz/GHz and data formats RI/MA/DB, plus `!` comments. It requires S-parameters, one complete option line, positive strictly increasing frequencies, positive finite reference impedance, finite values, identical reference impedances, 2-20000 points per file, and no extrapolation. Errors return HTTP 400 with the offending file, line when available, and frequency for alignment or network-matrix failures.
