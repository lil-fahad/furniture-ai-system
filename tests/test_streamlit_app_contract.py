from __future__ import annotations

import runpy
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "apps" / "streamlit_app.py"


class StopCalled(RuntimeError):
    pass


class UploadedFile:
    def __init__(self, data: bytes, *, name: str = "floor.png", media_type: str = "image/png"):
        self._data = data
        self.name = name
        self.type = media_type

    def getvalue(self) -> bytes:
        return self._data


class Response:
    def __init__(
        self,
        *,
        ok: bool,
        payload: Any = None,
        text: str = "",
        status_code: int = 200,
    ) -> None:
        self.ok = ok
        self._payload = payload
        self.text = text
        self.status_code = status_code

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeStreamlit(ModuleType):
    def __init__(
        self,
        *,
        uploaded: UploadedFile | None,
        pixels_per_cm: float = 4.0,
        use_openai: bool = True,
        preferences: str = "Warm modern",
        submit: bool = True,
    ) -> None:
        super().__init__("streamlit")
        self.uploaded = uploaded
        self.pixels_per_cm = pixels_per_cm
        self.use_openai = use_openai
        self.preferences = preferences
        self.submit = submit
        self.errors: list[str] = []
        self.successes: list[str] = []
        self.json_payloads: list[Any] = []

    def set_page_config(self, **_: Any) -> None:
        return None

    def title(self, _: str) -> None:
        return None

    def caption(self, _: str) -> None:
        return None

    def file_uploader(self, _: str, **__: Any) -> UploadedFile | None:
        return self.uploaded

    def number_input(self, _: str, **__: Any) -> float:
        return self.pixels_per_cm

    def checkbox(self, _: str, **__: Any) -> bool:
        return self.use_openai

    def text_area(self, _: str, **__: Any) -> str:
        return self.preferences

    def button(self, _: str, **__: Any) -> bool:
        return self.submit

    def spinner(self, _: str):
        return nullcontext()

    def error(self, message: str) -> None:
        self.errors.append(message)

    def success(self, message: str) -> None:
        self.successes.append(message)

    def json(self, payload: Any) -> None:
        self.json_payloads.append(payload)

    def stop(self) -> None:
        raise StopCalled


class TimeoutErrorForTest(Exception):
    pass


class RequestErrorForTest(Exception):
    pass


def _requests_module(post: Callable[..., Any]) -> ModuleType:
    module = ModuleType("requests")
    module.Timeout = TimeoutErrorForTest
    module.RequestException = RequestErrorForTest
    module.post = post
    return module


def _run_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    streamlit: FakeStreamlit,
    post: Callable[..., Any],
    api_url: str = "https://furniture.test",
    service_key: str = "service-secret",
    max_upload_bytes: int | str = 1024,
) -> None:
    monkeypatch.setitem(sys.modules, "streamlit", streamlit)
    monkeypatch.setitem(sys.modules, "requests", _requests_module(post))
    monkeypatch.setenv("FURNITURE_API_URL", api_url)
    monkeypatch.setenv("FURNITURE_MAX_UPLOAD_BYTES", str(max_upload_bytes))
    if service_key:
        monkeypatch.setenv("SERVICE_API_KEY", service_key)
    else:
        monkeypatch.delenv("SERVICE_API_KEY", raising=False)
    runpy.run_path(str(APP_PATH), run_name="__main__")


def test_streamlit_success_request_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    streamlit = FakeStreamlit(uploaded=UploadedFile(b"png-bytes"))
    calls: list[dict[str, Any]] = []
    response = Response(ok=True, payload={"analysis_id": "demo"})

    def post(url: str, **kwargs: Any) -> Response:
        calls.append({"url": url, **kwargs})
        return response

    _run_app(monkeypatch, streamlit=streamlit, post=post)

    assert calls == [
        {
            "url": "https://furniture.test/api/v1/analyze",
            "files": {"image": ("floor.png", b"png-bytes", "image/png")},
            "data": {
                "pixels_per_cm": "4.0",
                "use_openai": "true",
                "preferences": "Warm modern",
            },
            "headers": {"X-API-Key": "service-secret"},
            "timeout": 180,
        }
    ]
    assert streamlit.successes == ["Analysis complete"]
    assert streamlit.json_payloads == [{"analysis_id": "demo"}]
    assert streamlit.errors == []


def test_streamlit_normalizes_trailing_api_url_slashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = FakeStreamlit(uploaded=UploadedFile(b"png-bytes"))
    urls: list[str] = []

    def post(url: str, **_: Any) -> Response:
        urls.append(url)
        return Response(ok=True, payload={"analysis_id": "demo"})

    _run_app(
        monkeypatch,
        streamlit=streamlit,
        post=post,
        api_url=" https://furniture.test/// ",
    )

    assert urls == ["https://furniture.test/api/v1/analyze"]


@pytest.mark.parametrize("invalid_limit", ["not-a-number", 0, -1])
def test_streamlit_invalid_upload_limit_falls_back_to_safe_default(
    monkeypatch: pytest.MonkeyPatch,
    invalid_limit: int | str,
) -> None:
    streamlit = FakeStreamlit(uploaded=UploadedFile(b"png-bytes"))
    called = False

    def post(*_: Any, **__: Any) -> Response:
        nonlocal called
        called = True
        return Response(ok=True, payload={"analysis_id": "demo"})

    _run_app(
        monkeypatch,
        streamlit=streamlit,
        post=post,
        max_upload_bytes=invalid_limit,
    )

    assert called is True
    assert streamlit.errors == []
    assert streamlit.successes == ["Analysis complete"]


def test_streamlit_rejects_oversized_upload_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = FakeStreamlit(uploaded=UploadedFile(b"x" * 11))
    called = False

    def post(*_: Any, **__: Any) -> Response:
        nonlocal called
        called = True
        return Response(ok=True, payload={})

    with pytest.raises(StopCalled):
        _run_app(
            monkeypatch,
            streamlit=streamlit,
            post=post,
            max_upload_bytes=10,
        )

    assert called is False
    assert len(streamlit.errors) == 1
    assert "Please upload a smaller image" in streamlit.errors[0]


def test_streamlit_timeout_is_reported_without_leaking_request_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = FakeStreamlit(uploaded=UploadedFile(b"png-bytes"))

    def post(*_: Any, **__: Any) -> Response:
        raise TimeoutErrorForTest("internal timeout detail")

    with pytest.raises(StopCalled):
        _run_app(monkeypatch, streamlit=streamlit, post=post)

    assert streamlit.errors == [
        "The API at https://furniture.test did not respond within 180 seconds. Try again later."
    ]


def test_streamlit_request_error_does_not_expose_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = FakeStreamlit(uploaded=UploadedFile(b"png-bytes"))

    def post(*_: Any, **__: Any) -> Response:
        raise RequestErrorForTest("internal host=10.0.0.7 token=private-value")

    with pytest.raises(StopCalled):
        _run_app(monkeypatch, streamlit=streamlit, post=post)

    assert streamlit.errors == [
        "The API at https://furniture.test is unreachable. Check the service and try again."
    ]
    assert "10.0.0.7" not in streamlit.errors[0]
    assert "private-value" not in streamlit.errors[0]


def test_streamlit_invalid_success_json_is_reported_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = FakeStreamlit(uploaded=UploadedFile(b"png-bytes"))
    response = Response(ok=True, payload=ValueError("invalid JSON body"))

    def post(*_: Any, **__: Any) -> Response:
        return response

    with pytest.raises(StopCalled):
        _run_app(monkeypatch, streamlit=streamlit, post=post)

    assert streamlit.errors == [
        "The API returned an invalid success response. Try again later."
    ]
    assert streamlit.successes == []
    assert streamlit.json_payloads == []


def test_streamlit_non_object_error_json_falls_back_to_response_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = FakeStreamlit(uploaded=UploadedFile(b"png-bytes"))
    response = Response(
        ok=False,
        payload=["unexpected", "shape"],
        text="upstream returned an invalid error envelope",
        status_code=502,
    )

    def post(*_: Any, **__: Any) -> Response:
        return response

    _run_app(monkeypatch, streamlit=streamlit, post=post)

    assert streamlit.errors == [
        "API error 502: upstream returned an invalid error envelope"
    ]
