# langchain-webcite

LangChain tools and a retriever for [Webcite](https://webcite.co), one evidence suite for AI
apps and agents: document parsing and OCR to Markdown, claim verification with verdicts,
citations bound to source passages, and source search.

| Class | Kind | What it does | Credits |
|---|---|---|---|
| `WebciteVerifyClaimTool` | Tool | Checks a claim and returns a verdict, a summary and citations with URL, stance and the source passage | 4 per claim |
| `WebciteSearchSourcesTool` | Tool | Finds sources for a query, with the matching passage | 2 per search |
| `WebciteExtractDocumentTool` | Tool | Reads a PDF, scan, image, spreadsheet or office file at a URL into Markdown | Per page, at cost |
| `WebciteRetriever` | Retriever | Source search as LangChain `Document`s (passage as content, URL, title and stance as metadata) | 2 per query |
| `WebciteToolkit` | Toolkit | The three tools sharing one client | |

Prices are on [webcite.co/pricing](https://webcite.co/pricing). The Free plan includes 100
credits, renewing every 30 days while credits remain.

## Install

<!-- Switch back to "pip install langchain-webcite" once it is published on PyPI. -->
```bash
pip install "git+https://github.com/strategyconnect/webcite-python.git@v0.1.0" "git+https://github.com/strategyconnect/langchain-webcite.git@v0.1.0"
export WEBCITE_API_KEY=...   # create one at https://webcite.co/api-keys
```

Python 3.10 or newer. Depends on `langchain-core` and the official
[`webcite`](https://github.com/strategyconnect/webcite-python) Python SDK.

## Use the tools in an agent

```python
from langchain.agents import create_agent
from langchain_webcite import WebciteToolkit

agent = create_agent(
    "anthropic:claude-sonnet-5",
    tools=WebciteToolkit().get_tools(),
    system_prompt="Check each claim with webcite_verify_claim and cite the URLs it returns.",
)
agent.invoke({"messages": [{"role": "user", "content": "Can you see the Great Wall from the Moon?"}]})
```

## Verify a claim

```python
from langchain_webcite import WebciteVerifyClaimTool

tool = WebciteVerifyClaimTool(max_citations=2)
print(tool.invoke({"claim": "The Great Wall of China is visible from the Moon with the naked eye."}))
```

Output, rendered from the response recorded from the API on 24 November 2026
([webcite.co/samples/claim-great-wall.json](https://webcite.co/samples/claim-great-wall.json)),
shortened here:

```json
{"claim": "The Great Wall of China is visible from the Moon with the naked eye.",
 "verdict": "contradicted",
 "summary": "The claim that the Great Wall of China is visible from the Moon with the naked eye is false ...",
 "citations": [{"id": "1", "url": "https://www.nasa.gov/image-article/great-wall/", "title": "nasa.gov",
   "stance": "contradicts",
   "passage": "... Despite myths to the contrary, the wall isn't visible from the moon, ..."}, ...],
 "credits": 4}
```

Pass `source_urls` to check the claim against specific pages instead of searching the web.
When the tool is called by a model (with a `ToolCall`), the full API response is attached as the
`ToolMessage` artifact, so nothing is lost when the model only sees the compact text.

## Parse a document to Markdown

```python
from langchain_webcite import WebciteExtractDocumentTool

tool = WebciteExtractDocumentTool()          # mode="lite" for the faster, cheaper read
print(tool.invoke({"url": "https://www.irs.gov/pub/irs-pdf/fw9.pdf"}))
```

The first line is a JSON header; the Markdown follows. Recorded output for page 1 of the form
([webcite.co/samples/doc-irs-w9.json](https://webcite.co/samples/doc-irs-w9.json)):

```text
{"state": "complete", "pages": 1, "credits": 0.4966}

Form **W-9**
(Rev. March 2024)
...
```

Check `state`: anything other than `complete` comes with a `reason` naming what was not read.
Long documents are cut at `max_chars` (50,000 by default) and the header says so.

## Retrieve sources

```python
from langchain_webcite import WebciteRetriever

retriever = WebciteRetriever(k=3)
for doc in retriever.invoke("Apple total net sales fiscal 2024"):
    print(doc.metadata["url"], doc.page_content[:80])
```

## Configuration

Every class takes an optional `client`, a `webcite.Webcite` instance. Without one it creates
`Webcite()`, which reads `WEBCITE_API_KEY` (and `WEBCITE_API_URL` if set). Pass your own client to
set a timeout, retries or a custom `httpx` client:

```python
from webcite import Webcite
from langchain_webcite import WebciteToolkit

tools = WebciteToolkit(client=Webcite(timeout=120, max_retries=2)).get_tools()
```

The same capabilities are also available to any MCP client from the hosted server at
`https://api.webcite.co/mcp` (OAuth or API key).

## Development

```bash
make install   # uv sync --all-groups
make check     # ruff, mypy, pytest
make build     # wheel and sdist in dist/
```

Tests run the LangChain standard tool and retriever tests (`langchain-tests`) against recorded
API responses in `tests/fixtures`; sockets are disabled, so no test reaches the network or spends
credits. Examples are in [`examples/`](examples).

## Links

- [Webcite API reference](https://webcite.co/api-docs)
- [Webcite Python SDK](https://github.com/strategyconnect/webcite-python)
- Support: support@webcite.co

MIT licensed.
