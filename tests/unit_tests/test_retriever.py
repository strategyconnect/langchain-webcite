from __future__ import annotations

from typing import Any

from langchain_tests.integration_tests import RetrieversIntegrationTests
from webcite import Webcite

from langchain_webcite import WebciteRetriever
from tests.conftest import RecordedApi, recorded_client


def test_documents_carry_passage_and_source(client: Webcite, api: RecordedApi) -> None:
    docs = WebciteRetriever(client=client, k=2).invoke("great wall from the moon")
    assert len(docs) == 2
    assert "isn't visible from the moon" in docs[0].page_content
    assert docs[0].metadata["url"] == "https://www.nasa.gov/image-article/great-wall/"
    assert docs[0].metadata["stance"] == "contradicts"
    assert api.bodies[0] == {"query": "great wall from the moon", "limit": 2}


def test_k_at_call_time_overrides(client: Webcite, api: RecordedApi) -> None:
    docs = WebciteRetriever(client=client, k=5).invoke("q", k=1)
    assert len(docs) == 1
    assert api.bodies[0]["limit"] == 1


class TestRetrieverRecorded(RetrieversIntegrationTests):
    """The standard retriever tests, served from recorded responses instead of the live API."""

    @property
    def retriever_constructor(self) -> type[WebciteRetriever]:
        return WebciteRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return {"client": recorded_client(RecordedApi())}

    @property
    def retriever_query_example(self) -> str:
        return "Is the Great Wall of China visible from the Moon?"


def test_source_without_passage_keeps_its_title() -> None:
    import httpx

    def answer(_: httpx.Request) -> httpx.Response:
        cited = {"url": "https://example.org/a", "title": "Example A"}
        return httpx.Response(200, json={"claim_groups": [{"citations": [cited]}]})

    client = Webcite("test-key", http_client=httpx.Client(transport=httpx.MockTransport(answer)))
    docs = WebciteRetriever(client=client).invoke("q")
    assert docs[0].page_content == "Example A"
    assert docs[0].metadata["url"] == "https://example.org/a"
