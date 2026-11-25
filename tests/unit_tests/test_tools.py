from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests
from langchain_tests.unit_tests import ToolsUnitTests
from webcite import Webcite

from langchain_webcite import (
    WebciteExtractDocumentTool,
    WebciteSearchSourcesTool,
    WebciteToolkit,
    WebciteVerifyClaimTool,
)
from tests.conftest import RecordedApi, recorded_client

W9 = "https://www.irs.gov/pub/irs-pdf/fw9.pdf"


def test_verify_returns_verdict_and_bound_passages(client: Webcite, api: RecordedApi) -> None:
    tool = WebciteVerifyClaimTool(client=client, max_citations=2)
    message = tool.invoke(
        {
            "type": "tool_call",
            "id": "call-1",
            "name": tool.name,
            "args": {"claim": "The Great Wall of China is visible from the Moon."},
        }
    )
    assert isinstance(message, ToolMessage)
    body = json.loads(message.content)  # type: ignore[arg-type]
    assert body["verdict"] == "contradicted"
    assert body["credits"] == 4
    first = body["citations"][0]
    assert first["url"] == "https://www.nasa.gov/image-article/great-wall/"
    assert first["stance"] == "contradicts"
    assert "isn't visible from the moon" in first["passage"]
    assert len(body["citations"]) == 2
    assert message.artifact["claim_groups"][0]["verdict"]["result"] == "contradicted"
    assert api.paths == ["/api/v1/verify"]
    assert "source_urls" not in api.bodies[0]


def test_verify_passes_source_urls(client: Webcite, api: RecordedApi) -> None:
    WebciteVerifyClaimTool(client=client).invoke(
        {"claim": "x", "source_urls": ["https://www.nasa.gov/image-article/great-wall/"]}
    )
    assert api.bodies[0]["source_urls"] == ["https://www.nasa.gov/image-article/great-wall/"]


def test_search_limits_results(client: Webcite, api: RecordedApi) -> None:
    text = WebciteSearchSourcesTool(client=client).invoke({"query": "great wall", "limit": 3})
    results = json.loads(text)
    assert len(results) == 3
    assert {"url", "title", "passage"} <= set(results[0])
    assert api.paths == ["/api/v1/sources/search"]
    assert api.bodies[0] == {"query": "great wall", "limit": 3}


def test_extract_returns_state_header_then_markdown(client: Webcite, api: RecordedApi) -> None:
    text = WebciteExtractDocumentTool(client=client).invoke({"url": W9})
    header, markdown = text.split("\n\n", 1)
    assert json.loads(header) == {"state": "complete", "pages": 1, "credits": 0.4966}
    assert markdown.startswith("Form **W-9**")
    assert api.bodies[0] == {"asset_url": W9}


def test_extract_truncates_and_says_so(client: Webcite) -> None:
    tool = WebciteExtractDocumentTool(client=client, max_chars=20, mode="lite")
    header, markdown = tool.invoke({"url": W9}).split("\n\n", 1)
    assert json.loads(header)["truncated_to_chars"] == 20
    assert len(markdown) == 20


def test_toolkit_shares_one_client(client: Webcite) -> None:
    tools = WebciteToolkit(client=client).get_tools()
    assert [t.name for t in tools] == [
        "webcite_verify_claim",
        "webcite_search_sources",
        "webcite_extract_document",
    ]
    assert all(t.client is client for t in tools)


def test_missing_key_fails_with_the_action_to_take(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEBCITE_API_KEY", raising=False)
    with pytest.raises(Exception, match="WEBCITE_API_KEY"):
        WebciteVerifyClaimTool()


def test_reads_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBCITE_API_KEY", "env-key")
    assert WebciteVerifyClaimTool().client is not None


EXAMPLES: dict[type[BaseTool], dict[str, Any]] = {
    WebciteVerifyClaimTool: {"claim": "The Great Wall of China is visible from the Moon."},
    WebciteSearchSourcesTool: {"query": "Great Wall visible from the Moon", "limit": 2},
    WebciteExtractDocumentTool: {"url": W9},
}


class _Standard:
    tool_class: type[BaseTool]

    @property
    def tool_constructor(self) -> type[BaseTool]:
        return self.tool_class

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"client": recorded_client(RecordedApi())}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return EXAMPLES[self.tool_class]


class TestVerifyUnit(_Standard, ToolsUnitTests):
    tool_class = WebciteVerifyClaimTool


class TestSearchUnit(_Standard, ToolsUnitTests):
    tool_class = WebciteSearchSourcesTool


class TestExtractUnit(_Standard, ToolsUnitTests):
    tool_class = WebciteExtractDocumentTool


class TestVerifyRecorded(_Standard, ToolsIntegrationTests):
    """The standard invoke tests, served from recorded responses instead of the live API."""

    tool_class = WebciteVerifyClaimTool


class TestSearchRecorded(_Standard, ToolsIntegrationTests):
    tool_class = WebciteSearchSourcesTool


class TestExtractRecorded(_Standard, ToolsIntegrationTests):
    tool_class = WebciteExtractDocumentTool
