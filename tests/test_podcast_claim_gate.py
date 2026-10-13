"""Podcast claim gate: rewrite acceptance, gate application, and the DAG step."""

import asyncio

import pytest

from src.extraction import podcast_claim_gate as pcg
from src.extraction.podcast_claim_gate import (
    GateEpisodeMeta,
    GateReport,
    PodcastClaimGate,
    _DetectVerdict,
    _RepairDecision,
    accept_rewrite,
    material_names,
)

TRANSCRIPT = (
    "Speaker A: Donald Trump announced a deal to import Russian diesel in October. "
    "Speaker B: Chris Camillo put ten million dollars into the Amazon options trade. "
    "Saudi Arabia has fought the Houthis since 2015."
)
META = GateEpisodeMeta(
    podcast_name="The Dispatch Podcast",
    title="Why Trump Wants to Lift Russian Sanctions",
    air_date="2026-10-13T00:00:00.000Z",
    people=["Steve Hayes (host)", "Jonah Goldberg (guest)"],
)
MATERIAL = (META.render() + "\n" + TRANSCRIPT)
LOWER = MATERIAL.lower()
NAMES = material_names(MATERIAL)


# ── accept_rewrite ──────────────────────────────────────────────────────────


def test_accept_rewrite_takes_a_minimal_grounded_fix():
    ok, why = accept_rewrite(
        "The deal was presented as a measure to address high diesel prices.",
        "Donald Trump's deal to import Russian diesel was presented as a measure to address high diesel prices.",
        "context_dependent", LOWER, NAMES,
    )
    assert ok, why


def test_accept_rewrite_rejects_unchanged_low_overlap_and_lost_tokens():
    orig = "Saudi Arabia has been at war with the Houthis for 11 years."
    assert accept_rewrite(orig, orig, "relative_time", LOWER, NAMES) == (False, "unchanged")
    assert accept_rewrite(orig, "Diesel prices are high.", "relative_time", LOWER, NAMES)[1] == "low overlap"
    ok, why = accept_rewrite(orig, "Saudi Arabia has been at war for 11 years.", "relative_time", LOWER, NAMES)
    assert not ok and why.startswith("lost Houthis")
    ok, why = accept_rewrite(orig, "Saudi Arabia has been at war with the Houthis for years.", "relative_time", LOWER, NAMES)
    assert not ok and why.startswith("lost 11")


def test_accept_rewrite_rejects_invented_anchors_and_long_rewrites():
    orig = "The trade represented 10% of Chris Camillo's liquid net worth."
    ok, why = accept_rewrite(
        orig, "The Tesla trade represented 10% of Chris Camillo's liquid net worth.", "context_dependent", LOWER, NAMES
    )
    assert not ok and why == "not in material: Tesla"
    ok, why = accept_rewrite(
        orig, "The trade in 2019 represented 10% of Chris Camillo's liquid net worth.", "context_dependent", LOWER, NAMES
    )
    assert not ok and why == "not in material: 2019"
    padded = orig[:-1] + " " + " ".join(["indeed"] * (pcg.MAX_ADDED_WORDS + 1)) + "."
    ok, why = accept_rewrite(orig, padded, "context_dependent", LOWER, NAMES)
    assert not ok and why.startswith("adds")


def test_accept_rewrite_issue_specific_allowances():
    # An attribution repair may drop the speaker's name when it is a known person.
    ok, _ = accept_rewrite(
        "Jonah Goldberg notes that Saudi Arabia has fought the Houthis since 2015.",
        "Saudi Arabia has fought the Houthis since 2015.",
        "attribution_frame", LOWER, NAMES,
    )
    assert ok
    # A relative-time repair may replace a weekday with the absolute date.
    ok, _ = accept_rewrite(
        "Federal agents raided Ilhan Omar's home on Monday.",
        "Federal agents raided Ilhan Omar's home on October 12, 2026.",
        "relative_time", LOWER + " ilhan omar monday october 12 2026", NAMES,
    )
    assert ok
    # ...but not when the issue is something else.
    ok, why = accept_rewrite(
        "Jonah Goldberg notes that Saudi Arabia has fought the Houthis since 2015.",
        "Saudi Arabia has fought the Houthis since 2015.",
        "context_dependent", LOWER, NAMES,
    )
    assert not ok and why.startswith("lost Goldberg")


def test_meta_render_handles_missing_air_date():
    text = GateEpisodeMeta(title="t").render()
    assert "AIR DATE: unknown" in text and "PEOPLE: unknown" in text
    assert "AIR DATE: October 13, 2026 (2026-10-13)" in META.render()


# ── apply ───────────────────────────────────────────────────────────────────


class _FakeExtractor:
    def __init__(self):
        self.calls = []

    async def _call_gemini(self, prompt, config, step_name, model=None):
        raise AssertionError("tests stub detect/repair, the client must not be called")


def _gate_with(monkeypatch, verdicts, decisions, failed_batches=0):
    gate = PodcastClaimGate(_FakeExtractor())
    monkeypatch.setattr(pcg.llm, "get", lambda name: {"gemini_gate_detect_model": "flash", "gemini_gate_repair_model": "pro", "gemini_gate_temperature": 0.0}[name])

    async def detect(texts):
        return [verdicts.get(i) for i in range(len(texts))], failed_batches

    captured = {}

    async def repair(flagged, transcript, meta):
        captured["flagged"] = [i for i, _, _ in flagged]
        return decisions

    monkeypatch.setattr(gate, "detect", detect)
    monkeypatch.setattr(gate, "repair", repair)
    return gate, captured


CWT = {
    "Diesel": [
        "The deal was presented as a measure to address high diesel prices.",
        "Donald Trump announced a deal to import Russian diesel.",
    ],
    "Markets": [
        "The trade represented 10% of Chris Camillo's liquid net worth.",
        "Isaac Saul predicts Democrats will win the House.",
        "Chris Camillo monitors Google Trends.",
    ],
}


def test_apply_rewrites_withholds_and_keeps_in_order(monkeypatch):
    fires = lambda i, issue: _DetectVerdict(index=i, fires=True, issue=issue, rationale="r")
    verdicts = {0: fires(0, "context_dependent"), 2: fires(2, "context_dependent"), 3: fires(3, "attribution_frame"),
                1: _DetectVerdict(index=1, fires=False), 4: _DetectVerdict(index=4, fires=False)}
    decisions = {
        0: _RepairDecision(index=0, action="rewrite", issue="context_dependent",
                           text="Donald Trump's deal to import Russian diesel was presented as a measure to address high diesel prices."),
        2: _RepairDecision(index=2, action="rewrite", issue="context_dependent",
                           text="The Tesla trade represented 10% of Chris Camillo's liquid net worth."),  # invented -> rejected
        3: _RepairDecision(index=3, action="withhold", issue="attribution_frame", reason="prediction needs its speaker"),
    }
    gate, captured = _gate_with(monkeypatch, verdicts, decisions)
    out, report = asyncio.run(gate.apply(CWT, TRANSCRIPT, META))

    assert captured["flagged"] == [0, 2, 3]
    assert list(out) == ["Diesel", "Markets"]
    assert out["Diesel"][0].startswith("Donald Trump's deal") and out["Diesel"][1] == CWT["Diesel"][1]
    assert out["Markets"] == [CWT["Markets"][0], CWT["Markets"][2]]  # rejected rewrite keeps original; withheld gone
    assert (report.claims_in, report.flagged, report.repaired, report.withheld) == (5, 3, 1, 1)
    assert report.kept_flagged == 1 and report.rewrites_rejected == 1 and report.errors == 0
    assert report.withheld_claims[0].text == CWT["Markets"][1] and report.withheld_claims[0].topic == "Markets"
    assert (report.detect_model, report.repair_model) == ("flash", "pro")


def test_apply_is_fail_open(monkeypatch):
    # Detect batch failed (None verdicts) and repair returned nothing: every claim survives unchanged.
    verdicts = {0: _DetectVerdict(index=0, fires=True, issue="context_dependent")}
    gate, _ = _gate_with(monkeypatch, verdicts, {}, failed_batches=1)
    out, report = asyncio.run(gate.apply(CWT, TRANSCRIPT, META))
    assert out == CWT
    assert report.errors == 2 and report.kept_flagged == 1 and report.withheld == 0


def test_apply_drops_emptied_topics_and_handles_empty_input(monkeypatch):
    verdicts = {i: _DetectVerdict(index=i, fires=(i < 2), issue="impersonal_frame") for i in range(5)}
    decisions = {i: _RepairDecision(index=i, action="withhold", issue="impersonal_frame", reason="opinion") for i in (0, 1)}
    gate, _ = _gate_with(monkeypatch, verdicts, decisions)
    out, report = asyncio.run(gate.apply(CWT, TRANSCRIPT, META))
    assert list(out) == ["Markets"] and report.withheld == 2
    out, report = asyncio.run(gate.apply({}, TRANSCRIPT, META))
    assert out == {} and report.claims_in == 0


# ── DAG step ────────────────────────────────────────────────────────────────


def test_gate_step_passes_through_when_disabled_and_runs_when_enabled(monkeypatch):
    import src.tasks.podcast_extract_claims as pod

    inp = pod.PodcastExtractInput(
        episode_id=1, title="t", transcript="Speaker A (0s):\nhello", transcript_format="assembly",
        air_date="2026-10-13", podcast_name="P", credits=["A (host)"],
    )

    class Ctx:
        def task_output(self, task):
            return {pod.claims: {"claims_with_topics": CWT}}[task]

    spend = {"n": 0}
    monkeypatch.setattr(pod.spend_guard, "check_and_record", lambda provider: spend.__setitem__("n", spend["n"] + 1))

    monkeypatch.setattr(pod.settings, "podcast_gate_enabled", False)
    out = asyncio.run(pod.gate.fn(inp, Ctx()))
    assert out == {"claims_with_topics": CWT, "gate": None} and spend["n"] == 0

    seen = {}

    class FakeGate:
        async def apply(self, cwt, transcript, meta):
            seen["meta"] = meta
            seen["transcript"] = transcript
            return {"Diesel": cwt["Diesel"]}, GateReport(claims_in=5, flagged=1, withheld=1)

    monkeypatch.setattr(pod.settings, "podcast_gate_enabled", True)
    monkeypatch.setattr(pod, "_gate", lambda: FakeGate())
    out = asyncio.run(pod.gate.fn(inp, Ctx()))
    assert out["claims_with_topics"] == {"Diesel": CWT["Diesel"]}
    assert out["gate"]["withheld"] == 1 and spend["n"] == 2
    assert seen["meta"].air_date == "2026-10-13" and seen["meta"].people == ["A (host)"] and seen["meta"].podcast_name == "P"
    assert "hello" in seen["transcript"]


def test_input_defaults_keep_old_callers_working():
    import src.tasks.podcast_extract_claims as pod

    inp = pod.PodcastExtractInput(episode_id=1, title="t", transcript="x", transcript_format="podscribe")
    assert (inp.air_date, inp.podcast_name, inp.credits) == (None, "", [])
    assert pod.PodcastExtractResult(episode_id=1).gate is None


def test_gate_prompts_are_registered_with_their_slots():
    from src.config.prompt_registry import PROMPTS

    assert PROMPTS["podcast_gate.detect"].slots == ["claims"]
    assert PROMPTS["podcast_gate.repair"].slots == ["max_added_words", "max_claim_words", "metadata", "claims", "transcript"]
