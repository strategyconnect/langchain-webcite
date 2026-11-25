"""A LangChain retriever over Webcite source search."""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from webcite import Webcite

from langchain_webcite._format import search_results


class WebciteRetriever(BaseRetriever):
    """Return sources for a query as Documents: the passage as content, URL and title as metadata.

    Each query is one source search (2 credits, https://webcite.co/pricing).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Webcite = Field(default_factory=Webcite, exclude=True)
    k: int | None = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k: int | None = None,
    ) -> list[Document]:
        limit = k if k is not None else self.k
        response = self.client.search(query, limit=limit)
        return [
            Document(
                page_content=str(item.get("passage") or ""),
                metadata={key: val for key, val in item.items() if key != "passage"},
            )
            for item in search_results(response, limit)
        ]
