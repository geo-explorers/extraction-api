"""claims.judge_equivalence without Gemini: the prompt contract, verdict assembly, the
missing-verdict fallback, input bounds, and registry wiring."""

import pytest

from src.api.schemas.claims_judge_equivalence_schema import ClaimsJudgeEquivalenceInput, ClaimText
from src.extraction.claim_equivalence_judge import LLMVerdict, build_prompt
from src.tasks.claims_judge_equivalence import assemble


def test_prompt_carries_the_equivalence_rubric_and_indexes_candidates():
    p = build_prompt(
        "Sunlight exposure can cause skin cancer.",
        ["Sun exposure may cause skin cancer.", "Sunlight is good for you."],
    )
    assert "LOGICALLY EQUIVALENT" in p and "truth conditions" in p and "both directions" in p
    assert "similar or very close" in p and "entails the other but not the reverse" in p
    assert "[0] Sun exposure may cause skin cancer." in p and "[1] Sunlight is good for you." in p
    assert "CLAIM:\nSunlight exposure can cause skin cancer." in p


def test_assemble_surfaces_equivalent_and_unsure_and_keeps_every_verdict_in_order():
    inp = ClaimsJudgeEquivalenceInput(
        claim=ClaimText(id="new", text="X"),
        candidates=[ClaimText(id="a", text="A"), ClaimText(id="b", text="B"), ClaimText(text="C")],
    )
    verdicts = [
        LLMVerdict(candidate_index=0, verdict="equivalent", rationale="same proposition"),
        LLMVerdict(candidate_index=1, verdict="not_equivalent", rationale="adds a quantity"),
        LLMVerdict(candidate_index=2, verdict="unsure", rationale="ambiguous referent"),
    ]
    out = assemble(inp, verdicts, "fake-model")
    assert [j.verdict for j in out.judged] == ["equivalent", "not_equivalent", "unsure"]
    assert [j.index for j in out.judged] == [0, 1, 2]
    assert [j.id for j in out.equivalent] == ["a"] and out.equivalent[0].rationale == "same proposition"
    assert [j.text for j in out.unsure] == ["C"] and out.unsure[0].id is None
    assert out.claim.id == "new" and out.model_used == "fake-model"


@pytest.mark.asyncio
async def test_judge_fills_skipped_indices_with_unsure(monkeypatch):
    from src.extraction import claim_equivalence_judge as mod

    class Fake(mod.ClaimEquivalenceJudge):
        def __init__(self):  # no client, no key
            self.model_name = "fake"

        async def _call_gemini(self, prompt: str) -> str:
            # the model answers only for index 1 and invents an out-of-range index
            return '{"verdicts": [{"candidate_index": 1, "verdict": "equivalent", "rationale": "ok"}, {"candidate_index": 7, "verdict": "equivalent", "rationale": "bogus"}]}'

    verdicts = await Fake().judge("X", ["A", "B", "C"])
    assert [v.verdict for v in verdicts] == ["unsure", "equivalent", "unsure"]
    assert verdicts[0].rationale == "no verdict returned"
    assert await Fake().judge("X", []) == []


def test_registry_has_the_task_with_its_contract():
    from src.api.schemas.claims_judge_equivalence_schema import ClaimsJudgeEquivalenceResult
    from src.tasks.claims_judge_equivalence import CLAIMS_JUDGE_EQUIVALENCE_SPEC
    from src.tasks.registry import get_task

    entry = get_task("claims.judge_equivalence")
    assert entry is not None
    assert entry.input_model is ClaimsJudgeEquivalenceInput
    assert entry.output_model is ClaimsJudgeEquivalenceResult
    assert CLAIMS_JUDGE_EQUIVALENCE_SPEC.rate_limit_key == "gemini_global"
    assert get_task("claims.judge_duplicates") is None  # the retrieval-coupled name is gone


def test_input_bounds():
    with pytest.raises(ValueError):
        ClaimsJudgeEquivalenceInput(claim=ClaimText(text="x"), candidates=[])
    with pytest.raises(ValueError):
        ClaimsJudgeEquivalenceInput(claim=ClaimText(text=""), candidates=[ClaimText(text="y")])
    with pytest.raises(ValueError):
        ClaimsJudgeEquivalenceInput(claim=ClaimText(text="x"), candidates=[ClaimText(text="y")] * 51)
