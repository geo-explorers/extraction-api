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
    assert get_task("does.not.exist") is None
    # All specs built into task objects.
    assert len(all_tasks()) >= 3


def test_registry_specs_carry_contract():
    from src.tasks.registry import get_task
    from src.api.schemas.news_claim_extract_schema import (
        NewsClaimExtractRequest,
        NewsClaimExtractResponse,
    )

    entry = get_task("news.extract_claims")
    assert entry.spec.input_model is NewsClaimExtractRequest
    assert entry.spec.output_model is NewsClaimExtractResponse
    assert entry.spec.rate_limit_key == "gemini_global"
    assert get_task("news.extract_claims_claude").spec.rate_limit_key == "claude_global"


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


def test_payload_cap_is_enforced_by_spec():
    from src.tasks.registry import get_task

    entry = get_task("news.extract_claims")
    assert entry.spec.max_payload_bytes > 0
