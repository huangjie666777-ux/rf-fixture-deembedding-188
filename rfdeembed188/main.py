from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from .errors import RFDeembedError
from .pipeline import build_result_zip, run_deembedding


app = FastAPI(
    title="rfdeembed188",
    version="0.1.0",
    description="Two-port Touchstone fixture de-embedding backend.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "project": "rfdeembed188"}


@app.post("/deembed")
async def deembed_s2p(
    measurement: UploadFile = File(...),
    left_fixture: UploadFile = File(...),
    right_fixture: UploadFile = File(...),
    left_swapped: bool = Form(False),
    right_swapped: bool = Form(False),
) -> Response:
    try:
        measurement_raw = await measurement.read()
        left_raw = await left_fixture.read()
        right_raw = await right_fixture.read()
        result, diagnostics = run_deembedding(
            measurement_raw=measurement_raw,
            left_raw=left_raw,
            right_raw=right_raw,
            measurement_name=measurement.filename or "measurement.s2p",
            left_name=left_fixture.filename or "left_fixture.s2p",
            right_name=right_fixture.filename or "right_fixture.s2p",
            left_swapped=left_swapped,
            right_swapped=right_swapped,
        )
        payload = build_result_zip(result, diagnostics)
        return Response(
            content=payload,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="rfdeembed188-device.zip"'},
        )
    except RFDeembedError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.to_detail()})
