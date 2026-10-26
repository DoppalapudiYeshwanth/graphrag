"""Turn an uploaded file into a private, searchable, graph-linked document.

The flow is: validate -> read text -> chunk -> reserve shared budget -> extract entities
chunk by chunk (reporting progress) -> embed and store in the visitor's workspace.
Nothing is written to Neo4j until every chunk has been processed, so a failed upload
never leaves a half-ingested document behind.
"""

from __future__ import annotations

import io
import logging
import re
import uuid
from collections.abc import Iterator
from pathlib import PurePosixPath
from typing import Any

import pandas as pd
from pypdf import PdfReader

from graphrag.config import get_settings
from graphrag.data import Chunk, chunk_documents
from graphrag.graph import GraphStore
from graphrag.models import ExtractionResult, GraphExtractor, QuotaExhaustedError

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = (".txt", ".md", ".pdf")
MAX_PDF_PAGES = 40
MIN_TEXT_CHARS = 50
UPLOAD_CATEGORY = "Uploaded"

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class IngestError(ValueError):
    """A problem the uploader can fix. The message is written for them and safe to show."""


def safe_title(filename: str) -> str:
    """A display name: base name only, no control characters, capped length."""
    name = PurePosixPath(filename.replace("\\", "/")).name
    name = _CONTROL_CHARS.sub("", name).strip()
    return (name or "document")[:80]


def _clean(text: str) -> str:
    text = _CONTROL_CHARS.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(filename: str, data: bytes) -> str:
    """Read the text of a .txt, .md or .pdf upload, or raise IngestError."""
    s = get_settings()
    suffix = PurePosixPath(filename.lower()).suffix
    if suffix not in ALLOWED_EXTENSIONS:
        raise IngestError(f"Unsupported file type. Upload one of: {', '.join(ALLOWED_EXTENSIONS)}.")
    if len(data) > s.max_upload_bytes:
        raise IngestError(f"File is larger than {s.max_upload_bytes // 1_000_000} MB.")

    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise IngestError("This PDF is password-protected.")
            if len(reader.pages) > MAX_PDF_PAGES:
                raise IngestError(f"PDF has more than {MAX_PDF_PAGES} pages; upload a shorter one.")
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except IngestError:
            raise
        except Exception as exc:  # pypdf raises many error types for malformed files
            logger.info("PDF parse failure: %s", exc)
            raise IngestError("Could not read this PDF.") from exc
    else:
        text = data.decode("utf-8-sig", errors="replace")
        if text.count("�") > max(10, len(text) // 10):
            raise IngestError("This file does not look like readable text.")

    text = _clean(text)
    if len(text) < MIN_TEXT_CHARS:
        raise IngestError("Not enough readable text found (scanned PDFs and images are not supported).")
    if len(text) > s.max_doc_chars:
        raise IngestError(
            f"Document has {len(text):,} characters; the limit is {s.max_doc_chars:,}. "
            "Upload a shorter document or an excerpt."
        )
    return text


def split_into_chunks(text: str, doc_id: str, title: str) -> list[Chunk]:
    frame = pd.DataFrame([{"doc_id": doc_id, "text": text, "category": UPLOAD_CATEGORY, "source": title}])
    return chunk_documents(frame)


def ingest_stream(
    store: GraphStore,
    extractor: GraphExtractor,
    workspace: str,
    filename: str,
    data: bytes,
) -> Iterator[dict[str, Any]]:
    """Process one upload, yielding progress events (dicts) for the web page.

    Events: start, progress, done, or error. An `error` event is always the last one.
    """
    s = get_settings()
    reserved = 0
    try:
        if len(store.list_documents(workspace)) >= s.max_docs_per_workspace:
            raise IngestError(f"You can keep up to {s.max_docs_per_workspace} documents. Delete one first.")
        title = safe_title(filename)
        text = extract_text(filename, data)
        doc_id = f"UP-{uuid.uuid4().hex[:12]}"
        chunks = split_into_chunks(text, doc_id, title)

        if not store.reserve_budget("chunks", len(chunks), s.daily_chunk_budget):
            raise IngestError("Today's shared processing capacity is used up. Please try again tomorrow.")
        reserved = len(chunks)
        yield {"event": "start", "doc_id": doc_id, "title": title, "chunks": len(chunks)}

        results: dict[str, ExtractionResult] = {}
        failed = 0
        for done, chunk in enumerate(chunks, start=1):
            try:
                results[chunk.chunk_id] = extractor.extract(chunk)
            except QuotaExhaustedError as exc:
                store.release_budget("chunks", reserved - (done - 1))
                reserved = 0
                raise IngestError("The language model's quota is used up. Please try again later.") from exc
            except Exception:
                logger.exception("Extraction failed for %s", chunk.chunk_id)
                results[chunk.chunk_id] = ExtractionResult()  # still searchable by similarity
                failed += 1
            yield {"event": "progress", "done": done, "total": len(chunks)}

        if failed == len(chunks):
            raise IngestError("Could not analyse this document. Please try again later.")

        try:
            store.upsert_chunks(chunks, workspace=workspace, title=title)
            store.upsert_extractions(results, workspace=workspace)
        except Exception as exc:
            logger.exception("Storing %s failed; rolling back", doc_id)
            store.delete_document(workspace, doc_id)
            raise IngestError("Could not save the document. Please try again.") from exc

        yield {
            "event": "done",
            "doc_id": doc_id,
            "title": title,
            "chunks": len(chunks),
            "entities": len({e.key for r in results.values() for e in r.entities}),
            "relationships": sum(len(r.relationships) for r in results.values()),
            "failed_chunks": failed,
        }
    except IngestError as exc:
        if reserved:  # work that did not complete does not count against the shared budget
            store.release_budget("chunks", reserved)
        yield {"event": "error", "detail": str(exc)}
    except Exception:
        logger.exception("Unexpected ingestion failure")
        if reserved:
            store.release_budget("chunks", reserved)
        yield {"event": "error", "detail": "Something went wrong. Please try again."}
