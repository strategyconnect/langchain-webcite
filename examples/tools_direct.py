"""Call the Webcite tools directly, without a model.

Install: pip install "git+https://github.com/strategyconnect/webcite-python.git@v0.1.0" "git+https://github.com/strategyconnect/langchain-webcite.git@v0.1.0"
Run:     WEBCITE_API_KEY=... python examples/tools_direct.py
"""

from langchain_webcite import WebciteExtractDocumentTool, WebciteRetriever, WebciteVerifyClaimTool

verify = WebciteVerifyClaimTool()
print(
    verify.invoke({"claim": "The Great Wall of China is visible from the Moon with the naked eye."})
)

extract = WebciteExtractDocumentTool(mode="lite")
print(extract.invoke({"url": "https://www.irs.gov/pub/irs-pdf/fw9.pdf"})[:400])

for doc in WebciteRetriever(k=3).invoke("Apple total net sales fiscal 2024"):
    print(doc.metadata["url"], doc.page_content[:80])
