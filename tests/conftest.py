from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from webcite import Webcite

FIXTURES = Path(__file__).parent / "fixtures"
ROUTES = {
    "/api/v1/verify": "claim-great-wall.json",
    "/api/v1/sources/search": "search-great-wall.json",
    "/api/v1/extract": "doc-irs-w9.json",
}


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


class RecordedApi:
    """Answers Webcite API calls from the saved responses and records each request body."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        self.bodies.append(json.loads(request.content or b"{}"))
        return httpx.Response(200, json=load(ROUTES[request.url.path]))


def recorded_client(api: RecordedApi) -> Webcite:
    return Webcite("test-key", http_client=httpx.Client(transport=httpx.MockTransport(api)))


@pytest.fixture
def api() -> RecordedApi:
    return RecordedApi()


@pytest.fixture
def client(api: RecordedApi) -> Webcite:
    return recorded_client(api)
