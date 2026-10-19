"""In-memory stand-ins for the store, extractor and pipeline (no Neo4j, models or network)."""

from __future__ import annotations

from graphrag.models import Entity, ExtractionResult, QuotaExhaustedError, Relationship
from graphrag.retrieval import Answer, GraphFact, RetrievalContext, RetrievedChunk

WS_A = "a" * 32
WS_B = "b" * 32


class FakeStore:
    def __init__(self, chunk_budget: int = 1000, ask_budget: int = 1000) -> None:
        self.docs: dict[str, dict] = {}
        self.budget = {"chunks": chunk_budget, "asks": ask_budget}
        self.released: list[tuple[str, int]] = []
        self.fail_upsert = False
        self.extractions: dict = {}

    def list_documents(self, ws: str) -> list[dict]:
        return [
            {"doc_id": k, "title": v["title"], "created": 0, "chunks": v["chunks"]}
            for k, v in self.docs.items()
            if v["workspace"] == ws
        ]

    def reserve_budget(self, kind: str, amount: int, limit: int) -> bool:
        if self.budget[kind] >= amount:
            self.budget[kind] -= amount
            return True
        return False

    def release_budget(self, kind: str, amount: int) -> None:
        self.budget[kind] += amount
        self.released.append((kind, amount))

    def budget_remaining(self, kind: str, limit: int) -> int:
        return self.budget[kind]

    def upsert_chunks(self, chunks, workspace=None, title=None) -> None:
        if self.fail_upsert:
            raise RuntimeError("neo4j down")
        for c in chunks:
            doc = self.docs.setdefault(c.doc_id, {"workspace": workspace, "title": title, "chunks": 0})
            doc["chunks"] += 1

    def upsert_extractions(self, extractions, workspace=None) -> None:
        self.extractions.update(extractions)

    def delete_document(self, ws: str, doc_id: str) -> bool:
        if self.docs.get(doc_id, {}).get("workspace") != ws:
            return False
        del self.docs[doc_id]
        return True

    def delete_workspace(self, ws: str) -> None:
        self.docs = {k: v for k, v in self.docs.items() if v["workspace"] != ws}


class FakeExtractor:
    def __init__(self, quota_after: int | None = None, fail_on: set[int] | None = None) -> None:
        self.calls = 0
        self.quota_after = quota_after
        self.fail_on = fail_on or set()

    def extract(self, chunk) -> ExtractionResult:
        self.calls += 1
        if self.quota_after is not None and self.calls > self.quota_after:
            raise QuotaExhaustedError("all models exhausted")
        if self.calls in self.fail_on:
            raise ValueError("bad model output")
        return ExtractionResult(
            entities=[Entity(name="Alpha", type="Concept"), Entity(name="Beta", type="Concept")],
            relationships=[Relationship(source="Alpha", target="Beta", type="uses")],
        )


class FakeRag:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, list[str] | None]] = []

    def ask(self, question: str, scopes: list[str] | None = None) -> Answer:
        self.calls.append((question, scopes))
        if self.error:
            raise self.error
        ctx = RetrievalContext(
            chunks=[
                RetrievedChunk(chunk_id="D::0", doc_id="D", text="x" * 1000, score=0.9),
                RetrievedChunk(chunk_id="E::0", doc_id="E", text="y", via="graph"),
            ],
            facts=[GraphFact(source="A", relation="USES", target="B")],
        )
        return Answer(question=question, answer="the answer [D::0]", context=ctx)


def long_text(n_sentences: int = 40) -> bytes:
    """Plain text long enough for several chunks."""
    return (b"Neural networks learn useful representations from data. " * n_sentences)


def minimal_pdf(text: str) -> bytes:
    """A tiny valid one-page PDF containing `text` (no external libraries needed)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref)
    return out
