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
    material_words,
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
WORDS = material_words(MATERIAL)


# ── accept_rewrite ──────────────────────────────────────────────────────────


def test_accept_rewrite_takes_a_minimal_grounded_fix():
    ok, why = accept_rewrite(
        "The deal was presented as a measure to address high diesel prices.",
        "Donald Trump's deal to import Russian diesel was presented as a measure to address high diesel prices.",
        "context_dependent", WORDS,
    )
    assert ok, why


def test_accept_rewrite_rejects_unchanged_low_overlap_and_lost_tokens():
    orig = "Saudi Arabia has been at war with the Houthis for 11 years."
    assert accept_rewrite(orig, orig, "relative_time", WORDS) == (False, "unchanged")
    assert accept_rewrite(orig, "Diesel prices are high.", "relative_time", WORDS)[1] == "low overlap"
    ok, why = accept_rewrite(orig, "Saudi Arabia has been at war for 11 years.", "relative_time", WORDS)
    assert not ok and why.startswith("lost Houthis")
    ok, why = accept_rewrite(orig, "Saudi Arabia has been at war with the Houthis for years.", "relative_time", WORDS)
    assert not ok and why.startswith("lost 11")


def test_accept_rewrite_rejects_invented_anchors_and_long_rewrites():
    orig = "The trade represented 10% of Chris Camillo's liquid net worth."
    ok, why = accept_rewrite(
        orig, "The Tesla trade represented 10% of Chris Camillo's liquid net worth.", "context_dependent", WORDS
    )
    assert not ok and why == "not in material: Tesla"
    ok, why = accept_rewrite(
        orig, "The trade in 2019 represented 10% of Chris Camillo's liquid net worth.", "context_dependent", WORDS
    )
    assert not ok and why == "not in material: 2019"
    padded = orig[:-1] + " " + " ".join(["indeed"] * (pcg.MAX_ADDED_WORDS + 1)) + "."
    ok, why = accept_rewrite(orig, padded, "context_dependent", WORDS)
    assert not ok and why.startswith("adds")


def test_accept_rewrite_issue_specific_allowances():
    # An attribution repair may drop the speaker's name when it is a known person.
    ok, _ = accept_rewrite(
        "Jonah Goldberg notes that Saudi Arabia has fought the Houthis since 2015.",
        "Saudi Arabia has fought the Houthis since 2015.",
        "attribution_frame", WORDS,
    )
    assert ok
    # A relative-time repair may replace a weekday with the absolute date.
    ok, _ = accept_rewrite(
        "Federal agents raided Ilhan Omar's home on Monday.",
        "Federal agents raided Ilhan Omar's home on October 12, 2026.",
        "relative_time", WORDS | {"ilhan", "omar", "october", "12", "2026"},
    )
    assert ok
    # ...but not when the issue is something else.
    ok, why = accept_rewrite(
        "Jonah Goldberg notes that Saudi Arabia has fought the Houthis since 2015.",
        "Saudi Arabia has fought the Houthis since 2015.",
        "context_dependent", WORDS,
    )
    assert not ok and why.startswith("lost Jonah")


def test_accept_rewrite_grounds_a_sentence_initial_added_name():
    ok, why = accept_rewrite(
        "The company reported five billion in revenue.",
        "Tesla reported five billion in revenue.",
        "context_dependent", WORDS,
    )
    assert not ok and why == "not in material: Tesla"
    ok, why = accept_rewrite(
        "The deal was presented as a measure to address high diesel prices.",
        "Donald Trump's deal to import Russian diesel was presented as a measure to address high diesel prices.",
        "context_dependent", WORDS,
    )
    assert ok, why


def test_accept_rewrite_uses_whole_words_not_substrings():
    # "Mars" is not grounded by "markets", "Ross" not by "across".
    words = material_words("Donald Trump spoke about markets across the board.")
    ok, why = accept_rewrite(
        "The deal was announced by Donald Trump.", "The Mars deal was announced by Donald Trump.", "context_dependent", words
    )
    assert not ok and why == "not in material: Mars"
    ok, why = accept_rewrite(
        "The deal was announced by Donald Trump.", "Ross's deal was announced by Donald Trump.", "context_dependent", words
    )
    assert not ok and why == "not in material: Ross"


def test_accept_rewrite_tolerates_possessives_apostrophes_and_moved_names():
    # Curly vs straight apostrophe and possessive -> plain name are not "lost".
    ok, why = accept_rewrite(
        "Trump\u2019s deal lowered diesel prices.", "Donald Trump announced a deal that lowered diesel prices.",
        "context_dependent", WORDS,
    )
    assert ok, why
    # A name the extractor spelled its own way stays accepted when it moves mid-sentence.
    ok, why = accept_rewrite(
        "Zelenskyy met the leader in Paris.", "In Paris, Zelenskyy met Macron.",
        "context_dependent", WORDS | {"paris", "macron", "zelenskyy"},
    )
    assert ok, why
    # Long originals keep a budget of +10 words instead of a hard 40.
    long = " ".join(f"w{i}" for i in range(38)) + " Donald Trump."
    ok, why = accept_rewrite(long, long[:-1] + " in Saudi Arabia.", "context_dependent", WORDS)
    assert ok, why


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
    assert out["claims_with_topics"] == CWT and out["gate"]["enabled"] is False and spend["n"] == 0

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
    assert out["gate"]["withheld"] == 1 and spend["n"] == 0  # spend is recorded per model call inside the gate
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


def test_by_index_shifts_one_based_replies_and_warns_otherwise(caplog):
    from src.extraction.podcast_claim_gate import PodcastClaimGate

    one_based = [(1, "a"), (2, "b"), (3, "c")]
    assert PodcastClaimGate._by_index(one_based, {0, 1, 2}, "detect") == {0: "a", 1: "b", 2: "c"}
    with caplog.at_level("WARNING"):
        out = PodcastClaimGate._by_index([(0, "a"), (5, "x")], {0, 1}, "detect")
    assert out == {0: "a", 5: "x"} and "expected indices" in caplog.text


def test_detect_batch_maps_indices_and_fails_open_per_batch(monkeypatch):
    import json

    calls = []

    class Ext:
        async def _call_gemini(self, prompt, config, step_name, model=None):
            calls.append(prompt)
            if "boom" in prompt:
                raise RuntimeError("gemini down")
            # 1-based reply on purpose; only the second claim fires
            return json.dumps({"verdicts": [
                {"index": 1, "fires": False, "issue": "none", "rationale": ""},
                {"index": 2, "fires": True, "issue": "relative_time", "rationale": "currently"},
            ]})

    gate = PodcastClaimGate(Ext())
    monkeypatch.setattr(pcg.llm, "get", lambda name: {"gemini_gate_detect_model": "flash", "gemini_gate_repair_model": "pro", "gemini_gate_temperature": 0.0}[name])
    monkeypatch.setattr(pcg, "DETECT_BATCH_SIZE", 2)
    verdicts, failed = asyncio.run(gate.detect(["fine", "currently bad", "boom", "boom too"]))
    assert failed == 1 and len(verdicts) == 4
    assert verdicts[0].fires is False and verdicts[1].fires is True and verdicts[1].index == 1
    assert verdicts[2] is None and verdicts[3] is None
    assert len(calls) == 2


def test_gate_step_passes_through_on_timeout_and_error(monkeypatch):
    import src.tasks.podcast_extract_claims as pod

    inp = pod.PodcastExtractInput(episode_id=1, title="t", transcript="Speaker A (0s):\nhello", transcript_format="assembly")

    class Ctx:
        def task_output(self, task):
            return {pod.claims: {"claims_with_topics": CWT}}[task]

    monkeypatch.setattr(pod.settings, "podcast_gate_enabled", True)
    monkeypatch.setattr(pod, "GATE_BUDGET_SECONDS", 0.01)

    class Slow:
        async def apply(self, cwt, transcript, meta):
            await asyncio.sleep(1)
            return {}, GateReport()

    monkeypatch.setattr(pod, "_gate", lambda: Slow())
    out = asyncio.run(pod.gate.fn(inp, Ctx()))
    assert out["claims_with_topics"] == CWT and out["gate"]["errors"] == 1 and out["gate"]["claims_in"] == 5

    class Broken:
        async def apply(self, cwt, transcript, meta):
            raise RuntimeError("unexpected")

    monkeypatch.setattr(pod, "GATE_BUDGET_SECONDS", 5)
    monkeypatch.setattr(pod, "_gate", lambda: Broken())
    out = asyncio.run(pod.gate.fn(inp, Ctx()))
    assert out["claims_with_topics"] == CWT and out["gate"]["errors"] == 1

    monkeypatch.setattr(pod.settings, "podcast_gate_enabled", False)
    out = asyncio.run(pod.gate.fn(inp, Ctx()))
    assert out["gate"]["enabled"] is False and out["gate"]["claims_in"] == 5


def test_finalize_reads_gate_output_and_counts_topic_filter_loss(monkeypatch):
    import src.tasks.podcast_extract_claims as pod

    inp = pod.PodcastExtractInput(episode_id=1, title="t", transcript="x", transcript_format="podscribe")
    # Gate withheld one claim of a 3-claim topic: the other two fall to the topic filter.
    gated = {"Diesel": ["a", "b", "c"], "Markets": ["d", "e"]}
    report = GateReport(claims_in=6, flagged=1, withheld=1).model_dump()

    class Ctx:
        def task_output(self, task):
            return {
                pod.topics: {"topics": ["Diesel", "Markets"]},
                pod.gate: {"claims_with_topics": gated, "gate": report},
                pod.takeaways: {"key_takeaways": ["a"]},
            }[task]

    monkeypatch.setattr(pod, "_extractor", lambda: type("E", (), {"model_name": "m"})())
    out = asyncio.run(pod.finalize.fn(inp, Ctx()))
    assert [c.claim_text for c in out.claims] == ["a", "b", "c"]
    assert out.gate.withheld == 1 and out.gate.dropped_by_topic_filter == 2
    assert out.key_takeaways[0].claim_order == 1

    class OldRun:  # created before the gate step existed: no gate output
        def task_output(self, task):
            if task is pod.gate:
                raise KeyError("no such step")
            return {pod.topics: {"topics": ["Diesel"]}, pod.claims: {"claims_with_topics": {"Diesel": ["a", "b", "c"]}}, pod.takeaways: {"key_takeaways": []}}[task]

    out = asyncio.run(pod.finalize.fn(inp, OldRun()))
    assert len(out.claims) == 3 and out.gate is None
