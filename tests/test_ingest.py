"""Upload ingestion: parsing, limits, progress events, budget handling and rollback."""

from __future__ import annotations

import pytest

from graphrag.ingest import IngestError, extract_text, ingest_stream, safe_title

from .fakes import WS_A, FakeExtractor, FakeStore, long_text, minimal_pdf


# ── safe_title ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\notes.txt", "notes.txt"),
        ("bad\x00name\x07.txt", "badname.txt"),
        ("", "document"),
        ("x" * 200 + ".txt", ("x" * 200 + ".txt")[:80]),
    ],
)
def test_safe_title(raw: str, expected: str) -> None:
    assert safe_title(raw) == expected


# ── extract_text ─────────────────────────────────────────────────────────────
def test_reads_plain_text_and_normalises_whitespace() -> None:
    text = extract_text("a.txt", b"Hello   world.\r\n\r\n\r\n\r\nSecond paragraph that is long enough to pass.")
    assert text.startswith("Hello world.") and "\n\n\n" not in text and "\r" not in text


def test_reads_markdown() -> None:
    assert "Heading" in extract_text("a.md", b"# Heading\n\nSome body text that is certainly long enough to keep.")


def test_reads_a_pdf() -> None:
    pdf = minimal_pdf("The quick brown fox jumps over the lazy dog many many times today.")
    assert "quick brown fox" in extract_text("doc.pdf", pdf)


def test_rejects_corrupt_pdf() -> None:
    with pytest.raises(IngestError, match="Could not read this PDF"):
        extract_text("doc.pdf", b"%PDF-1.4 this is not really a pdf")


@pytest.mark.parametrize("name", ["a.exe", "a.docx", "noextension", "a.txt.exe"])
def test_rejects_unsupported_types(name: str) -> None:
    with pytest.raises(IngestError, match="Unsupported file type"):
        extract_text(name, b"x" * 100)


def test_extension_check_is_case_insensitive() -> None:
    assert extract_text("NOTES.TXT", long_text(3))


def test_rejects_too_little_text() -> None:
    with pytest.raises(IngestError, match="Not enough readable text"):
        extract_text("a.txt", b"too short")


def test_rejects_binary_garbage_as_text() -> None:
    with pytest.raises(IngestError, match="readable text"):
        extract_text("a.txt", bytes(range(128, 256)) * 50)


def test_rejects_documents_over_the_character_limit(settings_env) -> None:
    settings_env(MAX_DOC_CHARS="500")
    with pytest.raises(IngestError, match="limit is 500"):
        extract_text("a.txt", long_text(40))


def test_rejects_files_over_the_size_limit(settings_env) -> None:
    settings_env(MAX_UPLOAD_BYTES="1000")
    with pytest.raises(IngestError, match="larger than"):
        extract_text("a.txt", b"x" * 1001)


# ── ingest_stream ────────────────────────────────────────────────────────────
def run(store, extractor, name="paper.txt", data=None):
    return list(ingest_stream(store, extractor, WS_A, name, data if data is not None else long_text(40)))


def test_successful_upload_emits_start_progress_done(settings_env) -> None:
    settings_env()
    store, extractor = FakeStore(), FakeExtractor()
    events = run(store, extractor)

    kinds = [e["event"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert set(kinds[1:-1]) == {"progress"}
    total = events[0]["chunks"]
    assert total > 1 and [e["done"] for e in events if e["event"] == "progress"] == list(range(1, total + 1))
    assert events[-1]["entities"] == 2 and events[-1]["failed_chunks"] == 0
    assert extractor.calls == total
    assert store.budget["chunks"] == 1000 - total  # the budget was used
    assert len(store.list_documents(WS_A)) == 1


def test_document_ids_are_unique_per_upload(settings_env) -> None:
    settings_env()
    store = FakeStore()
    a = run(store, FakeExtractor())[0]["doc_id"]
    b = run(store, FakeExtractor())[0]["doc_id"]
    assert a != b and a.startswith("UP-")


def test_user_errors_become_error_events_not_exceptions(settings_env) -> None:
    settings_env()
    events = run(FakeStore(), FakeExtractor(), name="x.exe")
    assert events == [{"event": "error", "detail": "Unsupported file type. Upload one of: .txt, .md, .pdf."}]


def test_document_count_limit(settings_env) -> None:
    settings_env(MAX_DOCS_PER_WORKSPACE="1")
    store = FakeStore()
    assert run(store, FakeExtractor())[-1]["event"] == "done"
    events = run(store, FakeExtractor())
    assert events[-1]["event"] == "error" and "up to 1 documents" in events[-1]["detail"]


def test_exhausted_daily_budget_is_refused_before_any_llm_call(settings_env) -> None:
    settings_env()
    store, extractor = FakeStore(chunk_budget=1), FakeExtractor()
    events = run(store, extractor)
    assert events[-1]["event"] == "error" and "capacity is used up" in events[-1]["detail"]
    assert extractor.calls == 0 and store.docs == {}


def test_quota_exhaustion_midway_stores_nothing_and_refunds_only_unprocessed_chunks(settings_env) -> None:
    settings_env()
    store, extractor = FakeStore(), FakeExtractor(quota_after=1)
    events = run(store, extractor)
    assert events[-1]["event"] == "error" and "quota" in events[-1]["detail"]
    assert store.docs == {}
    # The one chunk that really was extracted used tokens, so it stays counted;
    # every chunk that never ran is given back.
    assert store.budget["chunks"] == 1000 - 1


def test_a_failing_chunk_is_tolerated_but_reported(settings_env) -> None:
    settings_env()
    store = FakeStore()
    events = run(store, FakeExtractor(fail_on={2}))
    done = events[-1]
    assert done["event"] == "done" and done["failed_chunks"] == 1
    assert len(store.list_documents(WS_A)) == 1  # still stored, still searchable by similarity


def test_failure_of_every_chunk_is_an_error_and_refunds(settings_env) -> None:
    settings_env()
    store = FakeStore()
    n = run(FakeStore(), FakeExtractor())[0]["chunks"]
    events = run(store, FakeExtractor(fail_on=set(range(1, n + 1))))
    assert events[-1]["event"] == "error" and store.budget["chunks"] == 1000


def test_storage_failure_rolls_back_and_refunds(settings_env) -> None:
    settings_env()
    store = FakeStore()
    store.fail_upsert = True
    events = run(store, FakeExtractor())
    assert events[-1] == {"event": "error", "detail": "Could not save the document. Please try again."}
    assert store.docs == {} and store.budget["chunks"] == 1000
