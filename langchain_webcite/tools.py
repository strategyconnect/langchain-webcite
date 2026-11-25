"""LangChain tools for the Webcite API: claim verification, source search, document parsing."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import BaseModel, ConfigDict, Field
from webcite import Webcite

from langchain_webcite._format import (
    compact_list,
    extraction_text,
    search_text,
    verification_text,
)


class _WebciteTool(BaseTool):
    """Shared client handling. ``client`` defaults to ``Webcite()``, which reads WEBCITE_API_KEY."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Webcite = Field(default_factory=Webcite, exclude=True)
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"


class VerifyClaimInput(BaseModel):
    claim: str = Field(description="One factual statement to check, in plain language.")
    source_urls: list[str] | None = Field(
        default=None,
        description="Optional URLs to check the claim against instead of searching the web.",
    )


class WebciteVerifyClaimTool(_WebciteTool):
    """Check a claim against sources and return a verdict with citations bound to passages.

    A verification with stance and verdict uses 4 credits (https://webcite.co/pricing).
    The full API response is returned as the tool message artifact.
    """

    name: str = "webcite_verify_claim"
    description: str = (
        "Check whether a factual claim is supported or contradicted by sources. Returns a "
        "verdict, a short summary and citations, each with its URL, stance and the source "
        "passage it rests on. Input is one claim; optionally the URLs to check it against."
    )
    args_schema: type[BaseModel] = VerifyClaimInput
    max_citations: int = 5

    def _run(
        self,
        claim: str,
        source_urls: list[str] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        response = self.client.verify(claim, source_urls=compact_list(source_urls))
        return verification_text(response, self.max_citations), dict(response)


class SearchSourcesInput(BaseModel):
    query: str = Field(description="What to find sources for.")
    limit: int | None = Field(default=None, description="Maximum number of sources.")


class WebciteSearchSourcesTool(_WebciteTool):
    """Find sources for a query, without stance or verdict. Uses 2 credits per search."""

    name: str = "webcite_search_sources"
    description: str = (
        "Search for sources relevant to a query. Returns a list of sources with URL, title "
        "and the matching passage. Use it to find evidence before writing or citing."
    )
    args_schema: type[BaseModel] = SearchSourcesInput

    def _run(
        self,
        query: str,
        limit: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        response = self.client.search(query, limit=limit)
        return search_text(response, limit), dict(response)


class ExtractDocumentInput(BaseModel):
    url: str = Field(description="Direct HTTPS URL of a PDF, image, spreadsheet or office file.")


class WebciteExtractDocumentTool(_WebciteTool):
    """Read a document at a URL into Markdown, with every page read before it returns.

    Extraction is metered per page at cost (https://webcite.co/pricing). Check ``state`` in
    the first line of the output: anything other than ``complete`` names what was not read.
    """

    name: str = "webcite_extract_document"
    description: str = (
        "Read a document (PDF, scan, image, spreadsheet, DOCX, PPTX) from a direct URL and "
        "return its text as Markdown, with tables kept as tables. The first line is a JSON "
        "header with the extraction state, page count and credits used."
    )
    args_schema: type[BaseModel] = ExtractDocumentInput
    mode: Literal["standard", "lite"] | None = None
    max_chars: int | None = 50_000

    def _run(
        self,
        url: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        response = self.client.extract(asset_url=url, mode=self.mode)
        return extraction_text(response, self.max_chars), dict(response)


class WebciteToolkit(BaseToolkit):
    """The three Webcite tools sharing one client."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Webcite = Field(default_factory=Webcite, exclude=True)

    def get_tools(self) -> list[BaseTool]:
        return [
            WebciteVerifyClaimTool(client=self.client),
            WebciteSearchSourcesTool(client=self.client),
            WebciteExtractDocumentTool(client=self.client),
        ]
