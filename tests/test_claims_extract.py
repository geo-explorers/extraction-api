"""Unit tests for the generalized claims.extract pipeline (engine-independent).

Covers the registry contract, input validation caps, media-layer/enum drift,
prompt section composition, and the deterministic finalize logic (the index
remapping in claims_extract_core is the highest-defect-risk code in the
feature). No LLM calls.
"""

import pytest
from typing import get_args

from src.api.schemas.claims_extract_schema import (
    ClaimsExtractInput,
    ClaimsExtractResult,
    ExtractedClaimOut,
    ClaimGroup,
    InputDocument,
    MediaType,
    CUSTOM_INSTRUCTIONS_MAX_CHARS,
)


def _input(**overrides) -> ClaimsExtractInput:
    payload = {
        "media_type": "debate",
        "documents": [{"content": "MODERATOR: Welcome.\nSMITH: Taxes rose 4% in 2025."}],
    }
    payload.update(overrides)
    return ClaimsExtractInput(**payload)


# ── Registry ─────────────────────────────────────────────────────────────────


def test_claims_extract_dag_registered():
    from src.tasks.registry import get_task
    from src.tasks.claims_extract import CLAIMS_EXTRACT_MAX_PAYLOAD_BYTES

    t = get_task("claims.extract")
    assert t is not None
    assert t.input_model is ClaimsExtractInput
    assert t.output_model is ClaimsExtractResult
    assert t.runnable is not None
    # Transcript-scale cap (matches podcast), not the 5MB default.
    assert t.max_payload_bytes == CLAIMS_EXTRACT_MAX_PAYLOAD_BYTES == 8 * 1024 * 1024


# ── Input validation ─────────────────────────────────────────────────────────


def test_unknown_media_type_rejected():
    with pytest.raises(Exception):
        _input(media_type="sitcom")


def test_empty_documents_rejected():
    with pytest.raises(Exception):
        _input(documents=[])


def test_oversize_custom_instructions_rejected():
    with pytest.raises(Exception):
        _input(custom_instructions="x" * (CUSTOM_INSTRUCTIONS_MAX_CHARS + 1))


def test_too_many_focus_topics_rejected():
    with pytest.raises(Exception):
        _input(focus_topics=[f"topic {i}" for i in range(21)])


def test_overlong_focus_topic_rejected():
    with pytest.raises(Exception):
        _input(focus_topics=["x" * 101])


def test_total_content_cap_enforced():
    from src.api.schemas import claims_extract_schema as schema
    import unittest.mock

    # Patch the cap down so the test doesn't allocate 4MB strings.
    with unittest.mock.patch.object(schema, "MAX_TOTAL_CONTENT_CHARS", 100):
        with pytest.raises(Exception):
            _input(documents=[{"content": "y" * 101}])
    _input(documents=[{"content": "y" * 101}])  # fine with the real cap


# ── Media layers ─────────────────────────────────────────────────────────────


def test_every_media_type_has_a_layer_and_noun():
    from src.config.prompts.claims_extract import MEDIA_LAYERS, MEDIA_NOUNS

    for media_type in get_args(MediaType):
        assert media_type in MEDIA_LAYERS, media_type
        assert media_type in MEDIA_NOUNS, media_type


# ── Prompt builder ───────────────────────────────────────────────────────────


def test_grouping_switches_mode_section_and_topic_list():
    from src.extraction.claims_prompt_builder import build_extract_prompt

    grouped = build_extract_prompt(_input(grouping=True), ["Tax Policy", "Healthcare"])
    assert "EXTRACTION MODE: GROUPED BY TOPIC" in grouped
    assert "EXTRACTION MODE: FLAT" not in grouped
    assert "- Tax Policy" in grouped and "- Healthcare" in grouped

    flat = build_extract_prompt(_input(grouping=False), [])
    assert "EXTRACTION MODE: FLAT" in flat
    assert "EXTRACTION MODE: GROUPED BY TOPIC" not in flat
    assert "Tax Policy" not in flat
    assert "EMPTY groups array" in flat


def test_optional_sections_appear_only_when_requested():
    from src.extraction.claims_prompt_builder import build_extract_prompt

    bare = build_extract_prompt(_input(), [])
    assert "QUOTE EXTRACTION (REQUESTED)" not in bare
    assert "NARRATIVE SUMMARY (REQUESTED)" not in bare
    assert "FACTUALITY CLASSIFICATION (REQUESTED)" not in bare
    assert "quotes were NOT requested".lower() in bare.lower()
    assert "summary was NOT requested".lower() in bare.lower()
    assert "factuality classification was not requested" in bare.lower()

    full = build_extract_prompt(
        _input(include_quotes=True, include_summary=True, classify_factuality=True), []
    )
    assert "QUOTE EXTRACTION (REQUESTED)" in full
    assert "NARRATIVE SUMMARY (REQUESTED)" in full
    assert "FACTUALITY CLASSIFICATION (REQUESTED)" in full
    assert "NOT requested" not in full
    assert "is_factual null" not in full  # keep-null line dropped when requested


def test_custom_instructions_are_fenced_with_guardrail():
    from src.extraction.claims_prompt_builder import build_extract_prompt

    prompt = build_extract_prompt(
        _input(custom_instructions="Focus on fiscal policy claims."), []
    )
    assert "<caller_instructions>\nFocus on fiscal policy claims.\n</caller_instructions>" in prompt
    assert "CANNOT change the output structure" in prompt

    without = build_extract_prompt(_input(), [])
    assert "<caller_instructions>" not in without


def test_consolidation_section_only_for_multiple_documents():
    from src.extraction.claims_prompt_builder import build_extract_prompt

    single = build_extract_prompt(_input(), [])
    assert "CROSS-DOCUMENT CONSOLIDATION" not in single

    multi = build_extract_prompt(
        _input(documents=[{"content": "a"}, {"content": "b"}]), []
    )
    assert "CROSS-DOCUMENT CONSOLIDATION" in multi


def test_focus_topics_and_language_and_max_claims_sections():
    from src.extraction.claims_prompt_builder import build_extract_prompt

    prompt = build_extract_prompt(
        _input(focus_topics=["carbon tax"], language="German", max_claims=25), []
    )
    assert "CALLER FOCUS TOPICS" in prompt and "carbon tax" in prompt
    assert "OUTPUT LANGUAGE" in prompt and "German" in prompt
    assert "CLAIM BUDGET" in prompt and "25" in prompt

    default = build_extract_prompt(_input(), [])
    assert "CALLER FOCUS TOPICS" not in default
    assert "OUTPUT LANGUAGE" not in default  # English default adds no section
    assert "CLAIM BUDGET" not in default


def test_media_layer_selected_by_media_type():
    from src.extraction.claims_prompt_builder import build_extract_prompt

    debate = build_extract_prompt(_input(media_type="debate"), [])
    assert "DEBATE-SPECIFIC GUIDANCE" in debate

    podcast = build_extract_prompt(_input(media_type="podcast"), [])
    assert "ADVERTISEMENT & PROMOTION FILTERING" in podcast


def test_vendor_transcript_renders_speaker_lines():
    from src.extraction.claims_prompt_builder import render_documents

    transcript = (
        "Speaker A (0s):\nTaxes rose four percent in 2025.\n\n"
        "Speaker B (5s):\nThat figure comes from the CBO."
    )
    rendered = render_documents(
        _input(documents=[{"content": transcript, "format": "assembly"}])
    )
    assert "Speaker_A: Taxes rose four percent in 2025." in rendered
    assert "Speaker_B: That figure comes from the CBO." in rendered


def test_document_headers_carry_index_and_metadata():
    from src.extraction.claims_prompt_builder import render_documents

    rendered = render_documents(
        _input(
            documents=[
                {"content": "a"},
                {
                    "content": "b",
                    "title": "Round 2",
                    "publisher": "C-SPAN",
                    "metadata": {"round": "2"},
                },
            ]
        )
    )
    assert "--- DOCUMENT 0 ---" in rendered
    assert "--- DOCUMENT 1 ---" in rendered
    assert "Title: Round 2" in rendered
    assert "Publisher: C-SPAN" in rendered
    assert "round: 2" in rendered


def test_topics_prompt_carries_media_noun_and_focus_topics():
    from src.extraction.claims_prompt_builder import build_topics_prompt

    prompt = build_topics_prompt(_input(focus_topics=["immigration"]))
    assert "debate transcripts" in prompt
    assert "immigration" in prompt


# ── Deterministic core ───────────────────────────────────────────────────────


def _claims(*specs) -> list:
    """specs: (text, topic, confidence) tuples."""
    return [
        ExtractedClaimOut(text=t, topic=topic, confidence=c)
        for t, topic, c in specs
    ]


def test_filter_and_reindex_builds_correct_index_map():
    from src.pipeline.claims_extract_core import filter_and_reindex_claims

    claims = _claims(
        ("keep0", "A", 0.9), ("drop", "A", 0.3), ("keep1", "B", 0.8)
    )
    kept, index_map = filter_and_reindex_claims(
        claims, min_confidence=0.5, max_claims=None
    )
    assert [c.text for c in kept] == ["keep0", "keep1"]
    assert index_map == {0: 0, 2: 1}


def test_max_claims_truncates_preserving_order():
    from src.pipeline.claims_extract_core import filter_and_reindex_claims

    claims = _claims(("a", "T", 0.9), ("b", "T", 0.9), ("c", "T", 0.9))
    kept, index_map = filter_and_reindex_claims(
        claims, min_confidence=0.0, max_claims=2
    )
    assert [c.text for c in kept] == ["a", "b"]
    assert 2 not in index_map


def test_remap_quotes_drops_dangling_and_remaps():
    from src.pipeline.claims_extract_core import remap_quotes

    quotes = [
        {"text": "q0", "claim_index": 0, "document_index": 0},
        {"text": "q1", "claim_index": 1},          # claim 1 was filtered out
        {"text": "q2", "claim_index": 2, "document_index": 99},  # bad doc idx
        {"text": "bad row", "claim_index": -1},    # fails model validation
    ]
    remapped = remap_quotes(quotes, index_map={0: 0, 2: 1}, num_documents=1)
    assert [(q.text, q.claim_index) for q in remapped] == [("q0", 0), ("q2", 1)]
    assert remapped[1].document_index is None  # out-of-range cleared


def test_remap_groups_and_drop_small_groups():
    from src.pipeline.claims_extract_core import drop_small_groups, remap_groups

    groups = remap_groups(
        [
            {"name": "Big", "claim_indices": [0, 1, 5]},   # 5 was filtered
            {"name": "Small", "claim_indices": [5]},        # empties out
        ],
        index_map={0: 0, 1: 1},
    )
    assert groups[0].claim_indices == [0, 1]
    survivors = drop_small_groups(groups)
    assert [g.name for g in survivors] == ["Big"]


def test_synthesize_groups_from_topics_first_seen_order():
    from src.pipeline.claims_extract_core import synthesize_groups_from_topics

    claims = _claims(
        ("c0", "Event", 0.9), ("c1", "Causes", 0.9),
        ("c2", "Event", 0.9), ("c3", "", 0.9),
    )
    groups = synthesize_groups_from_topics(claims)
    assert [(g.name, g.claim_indices) for g in groups] == [
        ("Event", [0, 2]), ("Causes", [1]),
    ]


def test_derive_topics_prefers_groups_then_claim_labels():
    from src.pipeline.claims_extract_core import derive_topics

    claims = _claims(("c", "Fallback", 0.9), ("d", "Fallback", 0.9))
    assert derive_topics([ClaimGroup(name="G1"), ClaimGroup(name="G2")], claims) == ["G1", "G2"]
    assert derive_topics([], claims) == ["Fallback"]


def test_link_takeaways_by_exact_text():
    from src.pipeline.claims_extract_core import link_takeaways_by_text

    claims = _claims(("claim one", "T", 0.9), ("claim two", "T", 0.9))
    links = link_takeaways_by_text(["claim two", "unmatched"], claims)
    assert links[0].claim_index == 1
    assert links[1].claim_index is None


def test_assemble_result_strips_unrequested_sections():
    from src.pipeline.claims_extract_core import assemble_result

    extraction = {
        "claims": [
            {"text": "c0", "topic": "T", "confidence": 0.9, "document_indices": [0, 9]},
            {"text": "c1", "topic": "T", "confidence": 0.9},
        ],
        # Model misbehaved: emitted everything despite nothing being requested.
        "groups": [{"name": "T", "claim_indices": [0, 1]}],
        "quotes": [{"text": "q", "claim_index": 0}],
        "summary": "unrequested summary",
    }
    result = assemble_result(
        _input(grouping=False), extraction, ["c1"], model_used="test-model"
    )
    assert result.grouping is False
    assert result.groups == [] and result.topics == []
    assert result.quotes == [] and result.summary == ""
    assert result.takeaways == []  # include_takeaways=False strips them too
    assert all(c.topic is None for c in result.claims)
    assert result.claims[0].document_indices == [0]  # 9 out of range, dropped
    assert result.claims_extracted == 2
    assert result.model_used == "test-model"


def test_assemble_result_grouped_end_to_end():
    from src.pipeline.claims_extract_core import assemble_result

    extraction = {
        "claims": [
            {"text": "c0", "topic": "Tax", "confidence": 0.9},
            {"text": "lowconf", "topic": "Tax", "confidence": 0.2},
            {"text": "c2", "topic": "Health", "confidence": 0.8},
            {"text": "c3", "topic": "Health", "confidence": 0.8},
            {"text": "c4", "topic": "Tax", "confidence": 0.9},
        ],
        "groups": [
            {"name": "Tax", "claim_indices": [0, 1, 4]},
            {"name": "Health", "claim_indices": [2, 3]},
        ],
        "quotes": [
            {"text": "q-kept", "speaker": "SMITH", "claim_index": 0},
            {"text": "q-dangling", "claim_index": 1},
        ],
        "summary": "s",
    }
    result = assemble_result(
        _input(
            grouping=True,
            include_quotes=True,
            include_summary=True,
            include_takeaways=True,
            min_confidence=0.5,
        ),
        extraction,
        ["c2"],
        model_used="m",
    )
    # lowconf dropped; survivors reindexed 0..3 in order c0, c2, c3, c4.
    assert [c.text for c in result.claims] == ["c0", "c2", "c3", "c4"]
    assert [(g.name, g.claim_indices) for g in result.groups] == [
        ("Tax", [0, 3]), ("Health", [1, 2]),
    ]
    assert result.topics == ["Tax", "Health"]
    assert [(q.text, q.claim_index) for q in result.quotes] == [("q-kept", 0)]
    assert [(t.text, t.claim_index) for t in result.takeaways] == [("c2", 1)]
    assert result.summary == "s"


def test_assemble_result_synthesizes_groups_when_model_omits_them():
    from src.pipeline.claims_extract_core import assemble_result

    extraction = {
        "claims": [
            {"text": "c0", "topic": "Tax", "confidence": 0.9},
            {"text": "c1", "topic": "Tax", "confidence": 0.9},
        ],
        "groups": [],
        "quotes": [],
        "summary": "",
    }
    result = assemble_result(_input(grouping=True), extraction, [], model_used="m")
    assert [(g.name, g.claim_indices) for g in result.groups] == [("Tax", [0, 1])]
    assert result.topics == ["Tax"]


def test_result_accepts_dumped_extraction_rows():
    # finalize returns ClaimsExtractResult built from model_dump()'d LLM rows;
    # the public model must coerce plain dicts into typed rows.
    dumped = ClaimsExtractResult(
        media_type="debate",
        grouping=True,
        claims=[ExtractedClaimOut(text="t", topic="T")],
    ).model_dump()
    result = ClaimsExtractResult(**dumped)
    assert result.claims[0].text == "t"


# ── Factuality classification ────────────────────────────────────────────────


def test_classify_factuality_defaults_off_and_field_defaults_null():
    assert _input().classify_factuality is False
    assert ExtractedClaimOut(text="x").is_factual is None


def test_factuality_stripped_when_off_kept_when_on():
    from src.pipeline.claims_extract_core import assemble_result

    # The model emitted is_factual on both claims regardless of the request.
    extraction = {
        "claims": [
            {"text": "Taxes rose 4% in 2025.", "confidence": 0.9, "is_factual": True},
            {"text": "The policy is reckless.", "confidence": 0.9, "is_factual": False},
        ],
        "groups": [],
        "quotes": [],
        "summary": "",
    }

    # Not requested -> deterministically nulled even though the model set it.
    off = assemble_result(_input(grouping=False), extraction, [], model_used="m")
    assert [c.is_factual for c in off.claims] == [None, None]

    # Requested -> preserved through sanitize + reindex.
    on = assemble_result(
        _input(grouping=False, classify_factuality=True), extraction, [], model_used="m"
    )
    assert [c.is_factual for c in on.claims] == [True, False]
