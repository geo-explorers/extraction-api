"""claims.judge_duplicates without geo-lens or Gemini: candidate merging, the judgement
mapping, the prompt contract, and registry wiring."""

import pytest

from src.api.schemas.claims_judge_duplicates_schema import ClaimIn, ClaimsJudgeDuplicatesInput
from src.api.services.geo_lens_client import LensHit
from src.extraction.claim_dedup_judge import LLMVerdict, build_prompt
from src.tasks.claims_judge_duplicates import gather_candidates, judge_claim, merge_candidates


def hit(id_: str, name: str, score: float) -> LensHit:
    return LensHit(id=id_, name=name, score=score, payload={})


def test_merge_candidates_unions_by_id_keeps_best_score_and_excludes_self():
    merged = merge_candidates(
        {
            "vector": [hit("a", "A text", 0.91), hit("b", "B text", 0.80), hit("me", "the claim", 1.0)],
            "text": [hit("b", "B text", 3.2), hit("c", "C text", 2.1)],
        },
        exclude_id="me",
    )
    assert [c.id for c in merged] == ["b", "c", "a"]  # best score first (BM25 3.2 > cosine 0.91)
    b = merged[0]
    assert b.score == 3.2 and b.sources == ["text", "vector"] and b.text == "B text"
    assert merged[2].sources == ["vector"]


def test_prompt_carries_the_rubric_and_indexes_candidates():
    p = build_prompt("Sunlight exposure can cause skin cancer.", ["Sun exposure may cause skin cancer.", "Sunlight is good for you."])
    assert "EXACTLY the same" in p and "polarity" in p and "hedging" in p and "generalizes" in p
    assert "[0] Sun exposure may cause skin cancer." in p and "[1] Sunlight is good for you." in p
    assert "NEW CLAIM:\nSunlight exposure can cause skin cancer." in p


class FakeLens:
    def __init__(self, by_strategy):
        self.by_strategy = by_strategy
        self.calls = []

    async def query(self, cache, strategy, input, *, k, min_score=None, filters=None):
        self.calls.append((cache, strategy, input, k, min_score, filters))
        return self.by_strategy.get(strategy, [])


class FakeJudge:
    model_name = "fake-judge"

    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.calls = 0

    async def judge(self, claim_text, candidates):
        self.calls += 1
        return [LLMVerdict(candidate_index=i, verdict=v, rationale=f"because {i}") for i, v in enumerate(self.verdicts)]


@pytest.mark.asyncio
async def test_gather_uses_vector_floor_only_and_passes_space_filter():
    lens = FakeLens({"vector": [hit("a", "A", 0.9)], "text": [hit("a", "A", 2.0), hit("b", "B", 1.5)]})
    claim = ClaimIn(id="me", text="X")
    inp = ClaimsJudgeDuplicatesInput(claims=[claim], k=7, min_score=0.8, space_ids=["s1"])
    cands = await gather_candidates(lens, claim, inp, "claims")
    assert [c.id for c in cands] == ["a", "b"]
    by_strategy = {c[1]: c for c in lens.calls}
    assert by_strategy["vector"][4] == 0.8 and by_strategy["text"][4] is None  # cosine floor is vector-only
    assert all(c[3] == 7 and c[5] == {"spaceIds": ["s1"]} and c[0] == "claims" for c in lens.calls)


@pytest.mark.asyncio
async def test_judge_claim_maps_verdicts_and_skips_llm_without_candidates(monkeypatch):
    from src.tasks import claims_judge_duplicates as mod

    monkeypatch.setattr(mod.spend_guard, "check_and_record", lambda provider: None)
    judge = FakeJudge(["same", "different", "unsure"])
    claim = ClaimIn(text="X")
    cands = merge_candidates({"vector": [hit("a", "A", 0.95), hit("b", "B", 0.9), hit("c", "C", 0.85)]}, None)
    result = await judge_claim(judge, claim, cands)
    assert result.candidates_considered == 3 and judge.calls == 1
    assert result.same == ["a"] and result.unsure == ["c"]
    assert [m.verdict for m in result.matches] == ["same", "different", "unsure"]
    assert result.matches[0].rationale == "because 0"

    empty = await judge_claim(judge, claim, [])
    assert empty.matches == [] and empty.same == [] and judge.calls == 1  # no LLM call without candidates


def test_registry_has_the_task_with_its_contract():
    from src.api.schemas.claims_judge_duplicates_schema import ClaimsJudgeDuplicatesResult
    from src.tasks.registry import get_task

    entry = get_task("claims.judge_duplicates")
    assert entry is not None
    assert entry.input_model is ClaimsJudgeDuplicatesInput
    assert entry.output_model is ClaimsJudgeDuplicatesResult
    from src.tasks.claims_judge_duplicates import CLAIMS_JUDGE_DUPLICATES_SPEC

    assert CLAIMS_JUDGE_DUPLICATES_SPEC.rate_limit_key == "gemini_global"


def test_input_validation_bounds():
    with pytest.raises(ValueError):
        ClaimsJudgeDuplicatesInput(claims=[])
    with pytest.raises(ValueError):
        ClaimsJudgeDuplicatesInput(claims=[ClaimIn(text="x")], k=0)
    with pytest.raises(ValueError):
        ClaimsJudgeDuplicatesInput(claims=[ClaimIn(text="x")], strategies=["exact"])  # not a candidate strategy
