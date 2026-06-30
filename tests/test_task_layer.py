"""Unit tests for the generic task layer (engine-independent parts).

These exercise the registry wiring, the spend circuit breaker, and the task
specs without connecting to a Hatchet engine.
"""

import pytest

from src.infrastructure.spend_guard import SpendGuard, SpendLimitExceeded


def test_registry_builds_expected_tasks():
    from src.tasks.registry import get_task, all_tasks

    assert get_task("ping") is not None
    assert get_task("news.extract_claims") is not None
    assert get_task("news.extract_claims_claude") is not None
    assert get_task("podcast.extract_claims") is not None
    assert get_task("does.not.exist") is None
    assert len(all_tasks()) >= 4


def test_registry_entries_carry_contract():
    from src.tasks.registry import get_task
    from src.api.schemas.news_claim_extract_schema import (
        NewsClaimExtractRequest,
        NewsClaimExtractResponse,
    )
    from src.tasks.podcast_extract_claims import (
        PodcastExtractInput,
        PodcastExtractResult,
    )

    news = get_task("news.extract_claims")
    assert news.input_model is NewsClaimExtractRequest
    assert news.output_model is NewsClaimExtractResponse
    assert news.runnable is not None

    podcast = get_task("podcast.extract_claims")
    assert podcast.input_model is PodcastExtractInput
    assert podcast.output_model is PodcastExtractResult


def test_news_specs_use_distinct_rate_limit_keys():
    from src.tasks.news_extract_claims import (
        NEWS_EXTRACT_CLAIMS_SPEC,
        NEWS_EXTRACT_CLAIMS_CLAUDE_SPEC,
    )

    assert NEWS_EXTRACT_CLAIMS_SPEC.rate_limit_key == "gemini_global"
    assert NEWS_EXTRACT_CLAIMS_CLAUDE_SPEC.rate_limit_key == "claude_global"


def test_spend_guard_disabled_is_noop():
    guard = SpendGuard(max_calls_per_hour=0)
    for _ in range(1000):
        guard.check_and_record("gemini")  # never raises when disabled


def test_spend_guard_trips_after_budget():
    guard = SpendGuard(max_calls_per_hour=3)
    for _ in range(3):
        guard.check_and_record("gemini")
    with pytest.raises(SpendLimitExceeded):
        guard.check_and_record("gemini")
    # Other providers have independent budgets.
    guard.check_and_record("claude")


def test_payload_cap_is_present():
    from src.tasks.registry import get_task

    assert get_task("news.extract_claims").max_payload_bytes > 0
    assert get_task("podcast.extract_claims").max_payload_bytes > 0


def test_build_claim_topics_filters_and_orders():
    from src.pipeline.premium_extraction_core import build_claim_topics, MIN_CLAIMS_PER_TOPIC

    cwt = {
        "Topic A": ["a1", "a2", "a3"],   # kept (>= MIN)
        "Topic B": ["b1"],               # dropped (sparse)
    }
    ordered_topics, filtered, claim_topics, count = build_claim_topics(
        ["Topic A", "Topic B"], cwt, episode_id=42
    )
    assert MIN_CLAIMS_PER_TOPIC == 3
    assert ordered_topics == ["Topic A"]
    assert "Topic B" not in filtered
    assert count == 3
    # claim_order is sequential starting at 1; episode_id stamped through.
    assert [c.claim_order for c in claim_topics] == [1, 2, 3]
    assert all(c.episode_id == 42 for c in claim_topics)


def test_link_takeaways_resolves_claim_order():
    from src.pipeline.premium_extraction_core import build_claim_topics, link_takeaways_to_claims

    _, _, claim_topics, _ = build_claim_topics(
        ["T"], {"T": ["claim one", "claim two", "claim three"]}, episode_id=1
    )
    links = link_takeaways_to_claims(["claim two", "unmatched takeaway"], claim_topics)
    assert links[0].text == "claim two" and links[0].claim_order == 2
    assert links[1].text == "unmatched takeaway" and links[1].claim_order is None


def test_news_topics_and_claims_dag_registered():
    from src.tasks.registry import get_task
    from src.api.schemas.news_topics_and_claims_schema import (
        NewsTopicsAndClaimsRequest,
        NewsTopicsAndClaimsResponse,
    )
    from src.tasks.base import DEFAULT_MAX_PAYLOAD_BYTES

    t = get_task("news.extract_topics_and_claims")
    assert t is not None
    assert t.input_model is NewsTopicsAndClaimsRequest
    assert t.output_model is NewsTopicsAndClaimsResponse
    assert t.runnable is not None
    # The request carries NO pre-extracted topics — they are derived in-task.
    assert "topics" not in NewsTopicsAndClaimsRequest.model_fields
    # Uses the 5MB default cap, NOT podcast's larger transcript-specific cap.
    assert t.max_payload_bytes == DEFAULT_MAX_PAYLOAD_BYTES


def test_derive_topics_uses_distinct_claim_topics_in_first_seen_order():
    from src.tasks.news_extract_topics_and_claims import _derive_topics

    claims = [
        {"topic": "Event"},
        {"topic": "Causes"},
        {"topic": "Event"},     # dup collapsed
        {"topic": ""},          # blank skipped
        {"topic": "Context"},
    ]
    # Step-1 labels are ignored when claims carry topics (the claim pass may have
    # relabeled/added topics) — order follows first appearance in the claims.
    assert _derive_topics(["StepOne"], claims) == ["Event", "Causes", "Context"]


def test_derive_topics_falls_back_to_step1_when_no_claims():
    from src.tasks.news_extract_topics_and_claims import _derive_topics

    assert _derive_topics(["A", "B"], []) == ["A", "B"]


def test_podcast_export_registered_with_contract():
    from src.tasks.registry import get_task
    from src.api.schemas.podcast_export_schema import (
        PodcastExportRequest,
        PodcastExportResult,
    )

    t = get_task("podcast.export")
    assert t is not None
    assert t.input_model is PodcastExportRequest
    assert t.output_model is PodcastExportResult
    assert t.runnable is not None


def test_podcast_export_spec_is_single_consumer_no_retry():
    from datetime import timedelta
    from src.tasks.podcast_export import PODCAST_EXPORT_SPEC

    # concurrency=1 makes it a single-consumer queue; retries=0 because the
    # downstream publish is non-idempotent and must never auto-retry.
    assert PODCAST_EXPORT_SPEC.concurrency == 1
    assert PODCAST_EXPORT_SPEC.retries == 0
    assert PODCAST_EXPORT_SPEC.rate_limit_key is None
    assert PODCAST_EXPORT_SPEC.execution_timeout == timedelta(minutes=40)


def test_concurrency_threads_into_build_task_kwargs(monkeypatch):
    # build_task must forward `concurrency` to the Hatchet decorator only when set.
    import src.tasks.base as base

    captured = {}

    def fake_task(**kwargs):
        captured.update(kwargs)
        def _decorator(fn):
            return fn
        return _decorator

    monkeypatch.setattr(base.hatchet, "task", fake_task)

    async def _h(inp, ctx):  # pragma: no cover - never called
        return inp

    from pydantic import BaseModel

    class _In(BaseModel):
        x: int = 0

    base.build_task(base.TaskSpec(
        name="t.with_conc", input_model=_In, output_model=_In, handler=_h, concurrency=1
    ))
    assert captured.get("concurrency") == 1

    captured.clear()
    base.build_task(base.TaskSpec(
        name="t.without_conc", input_model=_In, output_model=_In, handler=_h
    ))
    assert "concurrency" not in captured  # omitted when None


def test_podcast_export_handler_forwards_and_maps_response(monkeypatch):
    import asyncio
    import src.tasks.podcast_export as pe
    from src.api.schemas.podcast_export_schema import PodcastExportRequest

    monkeypatch.setattr(pe.settings, "postgrestogeo_url", "http://stub:3000")
    monkeypatch.setattr(pe.settings, "postgrestogeo_api_key", "secret-key")

    calls = {}

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "success": True,
                "message": "ok",
                "data": {"episodes_processed": 3, "ops_created": 7, "duration_ms": 1234},
            }

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.update(url=url, json=json, headers=headers, timeout=timeout)
        return _FakeResp()

    monkeypatch.setattr(pe.requests, "post", fake_post)

    req = PodcastExportRequest(
        podcast_name=["Bankless"], limit=5, num_episodes=10, date_filter="2020-01-01"
    )
    result = asyncio.run(pe._handle(req, ctx=None))

    # Forwarded to the private /api/export with auth + verbatim body.
    assert calls["url"] == "http://stub:3000/api/export"
    assert calls["headers"]["X-API-Key"] == "secret-key"
    assert calls["json"] == {
        "podcast_name": ["Bankless"],
        "limit": 5,
        "num_episodes": 10,
        "date_filter": "2020-01-01",
    }
    # Response `data` block mapped onto the flat result.
    assert result.success is True
    assert result.episodes_processed == 3
    assert result.ops_created == 7
    assert result.duration_ms == 1234
    assert result.message == "ok"


def test_response_accepts_dumped_claim_result():
    # finalize builds the response from extract_news_claims(...).model_dump(), so
    # the response model must coerce those plain dicts back into typed rows.
    from src.api.schemas.news_topics_and_claims_schema import NewsTopicsAndClaimsResponse
    from src.api.schemas.news_claim_extract_schema import (
        NewsClaimExtractResponse,
        ExtractedClaim,
    )

    dumped = NewsClaimExtractResponse(
        claims=[ExtractedClaim(text="t", topic="Event")],
        summary="s",
    ).model_dump()
    resp = NewsTopicsAndClaimsResponse(topics=["Event"], **dumped)
    assert resp.claims[0].topic == "Event"
    assert resp.topics == ["Event"]
    assert resp.summary == "s"
