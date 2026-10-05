from __future__ import annotations


class RFDeembedError(Exception):
    status_code = 400

    def to_detail(self) -> dict[str, object]:
        detail: dict[str, object] = {"message": str(self)}
        detail.update(self.extra)
        return detail

    @property
    def extra(self) -> dict[str, object]:
        return {}


class TouchstoneParseError(RFDeembedError):
    def __init__(self, filename: str, line: int | None, message: str) -> None:
        self.filename = filename
        self.line = line
        location = filename if line is None else f"{filename}:{line}"
        super().__init__(f"{location}: {message}")

    @property
    def extra(self) -> dict[str, object]:
        result: dict[str, object] = {"file": self.filename, "error": "invalid_touchstone"}
        if self.line is not None:
            result["line"] = self.line
        return result


class AlignmentError(RFDeembedError):
    def __init__(self, message: str, frequency_hz: float | None = None) -> None:
        self.frequency_hz = frequency_hz
        super().__init__(message)

    @property
    def extra(self) -> dict[str, object]:
        result = {"error": "frequency_alignment"}
        if self.frequency_hz is not None:
            result["frequency_hz"] = self.frequency_hz
        return result


class NetworkComputationError(RFDeembedError):
    def __init__(self, message: str, frequency_hz: float) -> None:
        self.frequency_hz = float(frequency_hz)
        super().__init__(f"{message} at {self.frequency_hz:.12g} Hz")

    @property
    def extra(self) -> dict[str, object]:
        return {"error": "network_computation", "frequency_hz": self.frequency_hz}
