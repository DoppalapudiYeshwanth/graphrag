"""Schema behaviour: relationship typing, name normalisation, pruning, cache re-validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphrag.models import (
    ALLOWED_RELATIONS,
    Entity,
    ExtractionResult,
    Relationship,
    canonical_relation,
    normalize_name,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("uses", "USES"),
        ("is evaluated on", "EVALUATED_ON"),
        ("Utilizes", "USES"),
        ("enhances", "IMPROVES"),
        ("is based on", "BASED_ON"),
        ("builds on", "EXTENDS"),
        ("outperforms", "OUTPERFORMS"),
        ("used by", "RELATED_TO"),  # reversed verb: never flipped silently
        ("includes", "RELATED_TO"),
        ("totally made up", "RELATED_TO"),
        ("SUPPORTS", "RELATED_TO"),
    ],
)
def test_canonical_relation(raw: str, expected: str) -> None:
    assert canonical_relation(raw) == expected


def test_canonical_relation_output_is_always_allowed() -> None:
    for raw in ["x", "a-b-c", "USES!!", "  spaced out  ", "Cause_Of"]:
        assert canonical_relation(raw) in ALLOWED_RELATIONS


def test_empty_relation_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_relation("___")


def test_normalize_name_treats_underscores_and_case_alike() -> None:
    assert normalize_name("Smart_card") == normalize_name("  smart   CARD ") == "smart card"


def test_entity_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        Entity(name="X", type="Banana")  # type: ignore[arg-type]


def test_entity_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        Entity(name="   ", type="Concept")


def test_entity_name_is_cleaned() -> None:
    assert Entity(name="Smart_card", type="Technology").name == "Smart card"


def test_relationship_type_is_canonicalised_on_construction() -> None:
    assert Relationship(source="A", target="B", type="tested on").type == "EVALUATED_ON"


def test_extraction_result_dedupes_entities_and_prunes_edges() -> None:
    result = ExtractionResult(
        entities=[
            Entity(name="Smart_card", type="Technology"),
            Entity(name="smart card", type="Technology"),  # duplicate after normalisation
            Entity(name="Scheme", type="Algorithm"),
        ],
        relationships=[
            Relationship(source="Scheme", target="Smart card", type="uses"),  # valid
            Relationship(source="Scheme", target="Missing", type="uses"),  # dangling
            Relationship(source="Scheme", target="scheme", type="uses"),  # self loop
        ],
    )
    assert [e.name for e in result.entities] == ["Smart card", "Scheme"]
    assert len(result.relationships) == 1
    assert result.relationships[0].type == "USES"


def test_old_cached_data_is_cleaned_on_revalidation() -> None:
    cached = {
        "entities": [{"name": "A", "type": "Concept"}, {"name": "B", "type": "Concept"}],
        "relationships": [{"source": "A", "target": "B", "type": "SUPPORTS"}],
    }
    assert ExtractionResult.model_validate(cached).relationships[0].type == "RELATED_TO"


def test_model_dump_roundtrip() -> None:
    result = ExtractionResult(
        entities=[Entity(name="A", type="Concept"), Entity(name="B", type="Task")],
        relationships=[Relationship(source="A", target="B", type="applied to")],
    )
    assert ExtractionResult.model_validate(result.model_dump()) == result
