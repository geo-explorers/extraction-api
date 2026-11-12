"""debates.pick_moments under its run deadline (GEO-3697), on a fake clock: a slow pick returns
the best reel it holds before geo-chat stops waiting, and a normal-speed pick is unchanged."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from hatchet_sdk.exceptions import NonRetryableException

from src.pipeline import claude_json
from src.pipeline import debates_pick_moments_core as pick
from src.pipeline import deadline as deadline_mod
from src.pipeline.claude_json import REQUEST_TIMEOUT_S, FinalDecisionError
from src.pipeline.deadline import ENQUEUED_AT, MIN_REQUEST_S, Deadline, enqueued_at
from tests.test_moment_reel_tasks import BROKEN, GOOD, OTHER, _debate, _FakeClaude, _last_resort_debate


class _Clock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class _SlowClaude(_FakeClaude):
    """A _FakeClaude whose every answer takes `took` seconds of the fake clock."""

    def __init__(self, clock: _Clock, took: float, *replies):
        super().__init__(*replies)
        self.clock = clock
        self.took = took

    async def create(self, **kwargs):
        self.clock.now += self.took
        return await super().create(**kwargs)


def _deadline(clock: _Clock, seconds: float) -> Deadline:
    return Deadline.after(seconds, clock=clock)


def _warnings(monkeypatch):
    lines = []
    monkeypatch.setattr(claude_json.logger, "warning", lines.append)
    monkeypatch.setattr(pick.logger, "warning", lines.append)
    return lines


def _soft_debate():
    debate = _debate()
    debate.sentences[3].text = "But it is a federal problem."  # GOOD's second clip: a soft problem
    return debate


# --- send-backs ---------------------------------------------------------------------------


def test_the_deadline_stops_send_backs_and_keeps_the_best_answer(monkeypatch):
    lines = _warnings(monkeypatch)
    clock = _Clock()
    # Each answer takes 230 s of 500: after two, 40 s are left, too few for a third.
    client = _SlowClaude(clock, 230, {"reels": [GOOD]}, {"reels": [BROKEN]}, {"reels": [GOOD]})
    result = asyncio.run(pick.pick_moments(_soft_debate(), client=client, deadline=_deadline(clock, 500)))
    # The first answer (soft problems only) beats the broken second; no third was asked for.
    assert len(client.calls) == 2 and result.attempts == 1 and len(result.moments) == 4
    assert any("shortened by the deadline" in line and "answer 3 was not asked for" in line for line in lines), lines


def test_a_last_resort_is_kept_when_the_deadline_stops_send_backs(monkeypatch):
    clock = _Clock()
    client = _SlowClaude(clock, 450, {"reels": [GOOD]}, {"reels": [GOOD]})
    result = asyncio.run(pick.pick_moments(_last_resort_debate(), client=client, deadline=_deadline(clock, 500)))
    assert len(client.calls) == 1 and result.attempts == 1 and result.first_clip_problems


# --- the judge ----------------------------------------------------------------------------


def test_the_judge_is_skipped_near_the_deadline_for_the_top_ranked_reel(monkeypatch):
    lines = _warnings(monkeypatch)
    clock = _Clock()
    client = _SlowClaude(clock, 450, {"reels": [GOOD, OTHER]}, {"best": 1, "why": "never asked"})
    result = asyncio.run(pick.pick_moments(_debate(), client=client, deadline=_deadline(clock, 500)))
    assert len(client.calls) == 1
    assert [m.speaker for m in result.moments] == [1, 2, 1, 2]
    assert result.candidates == 2 and result.judge_why is None
    assert any("shortened by the deadline" in line and "judge was skipped" in line for line in lines), lines


def test_the_top_ranked_reel_has_the_fewest_problems_the_first_among_equals():
    assert pick.top_ranked([("a", ["x", "y"]), ("b", ["x"]), ("c", ["z"])]) == 1
    assert pick.top_ranked([("a", []), ("b", [])]) == 0


def test_the_judge_runs_with_time_left_and_its_request_is_capped_too():
    clock = _Clock()
    client = _SlowClaude(clock, 100, {"reels": [GOOD, OTHER]}, {"best": 1, "why": "Nico opens stronger."})
    result = asyncio.run(pick.pick_moments(_debate(), client=client, deadline=_deadline(clock, 250)))
    assert len(client.calls) == 2 and result.judge_why == "Nico opens stronger."
    assert [call["timeout"] for call in client.calls] == [250, 150]


def _reel_0_worse(monkeypatch):
    """A debate where both GOOD and OTHER pass with a soft problem (clips opening on sentence 9's
    "But"), and reel 0 is given one more, so the top-ranked reel is reel 1 (OTHER, Nico first)."""
    real = pick.check_reels

    def check(input, answer):
        passed, hard, soft = real(input, answer)
        if passed:
            passed = [(reel, problems + ["one more"] if i == 0 else problems) for i, (reel, problems) in enumerate(passed)]
        return passed, hard, soft

    monkeypatch.setattr(pick, "check_reels", check)
    debate = _debate()
    debate.sentences[9].text = "But it is a federal problem."
    return debate


def test_a_judge_cut_off_at_the_deadline_keeps_the_top_ranked_reel(monkeypatch):
    debate = _reel_0_worse(monkeypatch)
    clock = _Clock()
    reels = {"reels": [GOOD, OTHER]}

    class _JudgeRunsOut(_SlowClaude):
        async def create(self, **kwargs):
            if "best" in str(self.replies[0]):  # the judge: takes all its time and is cut off
                self.calls.append(kwargs)
                self.clock.now += kwargs["timeout"]
                raise TimeoutError()
            return await super().create(**kwargs)

    # Three soft answers leave 90 s: the judge starts, then runs out at the deadline.
    client = _JudgeRunsOut(clock, 270, reels, reels, reels, {"best": 0, "why": "never read"})
    deadline = _deadline(clock, 900)
    result = asyncio.run(pick.pick_moments(debate, client=client, deadline=deadline))
    assert len(client.calls) == 4 and client.calls[-1]["timeout"] == 90 and deadline.left() == 0
    # The reel a skipped judge keeps, not reel 0 with its extra problem.
    assert [m.speaker for m in result.moments] == [2, 1, 2, 1] and result.judge_why is None


def test_a_judge_skipped_at_the_deadline_keeps_the_same_reel(monkeypatch):
    debate = _reel_0_worse(monkeypatch)
    clock = _Clock()
    reels = {"reels": [GOOD, OTHER]}
    client = _SlowClaude(clock, 290, reels, reels, reels, {"best": 0, "why": "never asked"})
    result = asyncio.run(pick.pick_moments(debate, client=client, deadline=_deadline(clock, 900)))
    assert len(client.calls) == 3 and [m.speaker for m in result.moments] == [2, 1, 2, 1]


# --- request timeouts ---------------------------------------------------------------------


def test_each_request_timeout_is_capped_by_the_time_left():
    clock = _Clock()
    client = _SlowClaude(clock, 200, {"reels": [BROKEN]}, {"reels": [BROKEN]}, {"reels": [GOOD]})
    asyncio.run(pick.pick_moments(_debate(), client=client, deadline=_deadline(clock, 600)))
    # The usual ceiling while there is time for it, then only what is left.
    assert [call["timeout"] for call in client.calls] == [REQUEST_TIMEOUT_S, REQUEST_TIMEOUT_S, 200]


def test_a_request_that_would_run_past_the_deadline_is_cut_off_at_it(monkeypatch):
    # On the real clock, scaled down: 0.1 s to the deadline, a request that would take 5 s.
    monkeypatch.setattr(deadline_mod, "MIN_REQUEST_S", 0.05)

    class _Hangs(_FakeClaude):
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            await asyncio.sleep(5)

    started = time.monotonic()
    with pytest.raises(FinalDecisionError, match="shortened by the deadline"):
        asyncio.run(pick.pick_moments(_debate(), client=_Hangs(), deadline=Deadline.after(0.1)))
    assert time.monotonic() - started < 1


# --- no usable reel -----------------------------------------------------------------------


def test_no_usable_reel_by_the_deadline_fails_fast_and_is_not_rerun(monkeypatch):
    clock = _Clock()
    client = _SlowClaude(clock, 230, {"reels": [BROKEN]}, {"reels": [BROKEN]}, {"reels": [GOOD]})
    with pytest.raises(FinalDecisionError, match="shortened by the deadline") as e:
        asyncio.run(pick.pick_moments(_debate(), client=client, deadline=_deadline(clock, 500)))
    assert len(client.calls) == 2 and isinstance(e.value, NonRetryableException)


def test_a_run_that_starts_past_its_deadline_asks_for_nothing():
    clock = _Clock()
    client = _SlowClaude(clock, 1, {"reels": [GOOD]})
    with pytest.raises(FinalDecisionError, match="shortened by the deadline"):
        asyncio.run(pick.pick_moments(_debate(), client=client, deadline=_deadline(clock, MIN_REQUEST_S - 1)))
    assert client.calls == []


def test_a_request_failing_with_too_little_time_left_is_final_not_rerun():
    clock = _Clock()
    client = _SlowClaude(clock, 50, RuntimeError("overloaded"))
    with pytest.raises(FinalDecisionError, match="shortened by the deadline"):
        asyncio.run(pick.pick_moments(_debate(), client=client, deadline=_deadline(clock, 100)))


def test_a_request_failing_with_time_left_stays_retryable():
    clock = _Clock()
    client = _SlowClaude(clock, 50, RuntimeError("overloaded"))
    with pytest.raises(Exception) as e:
        asyncio.run(pick.pick_moments(_debate(), client=client, deadline=_deadline(clock, 900)))
    assert not isinstance(e.value, NonRetryableException) and e.value.retryable


# --- normal speed -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make, replies",
    [
        # Judged between two reels; a broken answer sent back; soft answers sent back to the last.
        (_debate, ({"reels": [GOOD, OTHER]}, {"best": 1, "why": "Nico opens stronger."})),
        (_debate, ({"reels": [BROKEN]}, {"reels": [GOOD]})),
        (_soft_debate, ({"reels": [GOOD]}, {"reels": [GOOD]}, {"reels": [GOOD]})),
    ],
)
def test_a_normal_speed_pick_is_the_same_with_the_deadline(make, replies):
    debate = make()
    without = _FakeClaude(*json.loads(json.dumps(replies)))
    expected = asyncio.run(pick.pick_moments(debate, client=without))
    clock = _Clock()
    within = _SlowClaude(clock, 60, *json.loads(json.dumps(replies)))
    result = asyncio.run(pick.pick_moments(debate, client=within, deadline=_deadline(clock, 15 * 60)))
    assert result == expected and len(within.calls) == len(without.calls)
    # Without a deadline the client keeps its own timeout.
    assert all("timeout" not in call for call in without.calls)


# --- the task: when its clock starts, and the engine's limits -----------------------------


def test_the_run_clock_starts_when_the_facade_enqueued_it():
    clock = _Clock()
    assert enqueued_at({ENQUEUED_AT: str(clock.now - 300)}, clock) == clock.now - 300
    # Missing, unreadable or ahead of this host's clock: now.
    for metadata in (None, {}, {ENQUEUED_AT: "soon"}, {ENQUEUED_AT: "nan"}, {ENQUEUED_AT: str(clock.now + 5)}):
        assert enqueued_at(metadata, clock) == clock.now


def test_the_task_counts_its_deadline_from_the_enqueue_stamp(monkeypatch):
    from src.tasks import debates_pick_moments as task

    seen = {}

    async def fake_pick(input, *, deadline):
        seen["deadline"] = deadline
        return SimpleNamespace(moments=[], attempts=1, candidates=1, model_used="m")

    monkeypatch.setattr(task, "pick_moments", fake_pick)
    stamp = time.time() - 600  # queued, or retried, for 10 minutes
    ctx = SimpleNamespace(additional_metadata={ENQUEUED_AT: str(stamp)})
    asyncio.run(task._handle(_debate(), ctx))
    assert seen["deadline"].at == stamp + task.PICK_DEADLINE_S


def test_the_deadline_and_engine_timeout_stay_under_geo_chats_wait():
    from src.tasks.debates_pick_moments import (
        DEBATES_PICK_MOMENTS_SPEC,
        EXECUTION_TIMEOUT_S,
        GEO_CHAT_WAIT_S,
        PICK_DEADLINE_S,
    )

    # Two minutes for the result to reach geo-chat; the engine's backstop also ends before it.
    assert PICK_DEADLINE_S <= GEO_CHAT_WAIT_S - 120
    assert DEBATES_PICK_MOMENTS_SPEC.execution_timeout.total_seconds() == EXECUTION_TIMEOUT_S < GEO_CHAT_WAIT_S
    # Before this deadline, one attempt could take every call at its ceiling.
    assert pick.CALLS * REQUEST_TIMEOUT_S > GEO_CHAT_WAIT_S


def test_the_facade_stamps_when_it_enqueued_a_run(monkeypatch):
    import src.tasks.registry as registry
    from src.api.routers.tasks import EnqueueRequest, enqueue_task

    entry = registry.get_task("ping")
    runs = []

    def run(input, **kwargs):
        runs.append(kwargs)
        return SimpleNamespace(workflow_run_id="run-1")

    monkeypatch.setattr(entry, "runnable", SimpleNamespace(run=run))
    before = time.time()
    enqueue_task(EnqueueRequest(type="ping", payload={"message": "hi"}), SimpleNamespace(state=SimpleNamespace()))
    stamped = float(runs[0]["additional_metadata"][ENQUEUED_AT])
    assert before <= stamped <= time.time() and runs[0]["wait_for_result"] is False
