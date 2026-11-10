# claims.score_clips

Judges a debate's candidate clips as a viewer hears them. Built for geo-chat's moment reels (GEO-3614), beside `claims.score_highlights`.

`claims.score_highlights` scores an extracted **claim**, a sentence the extractor wrote. A reel plays **clips**: the words a speaker says between two cut points. A clip can belong to a claim that scores well and still:
- misrepresent its speaker,
- miss its own punchline,
- or depend on a sentence the viewer never hears.

A review of 65 debates' reels on 10 Nov 2026 found these defects in most reels. This task asks about the clip itself.

## Input

```json
{
  "media_type": "debate",
  "title": "It's fine to ghost someone after a few dates",
  "context": "Urban argues Agree; Susan argues Disagree.",
  "documents": [{"speaker": "Susan", "content": "..."}, {"speaker": "Urban", "content": "..."}],
  "clips": [
    {"id": "12", "speaker": "Urban", "side": "Agree", "text": "So is it closure or is it entitlement? ...", "document_index": 1, "duration_ms": 17200}
  ]
}
```

- `documents` is the transcript in spoken order, as for `claims.score_highlights`.
- `clips` holds at most 24 candidates. Each is the clip's words exactly as transcribed between its cut points.
- `side` is the side the caller knows the speaker argues. Faithfulness is judged against it, not guessed.
- `document_index` and `duration_ms` are optional.

## What is asked

Each clip gets one request, with the whole transcript and every candidate clip in view:

| Decision | Kind | Catches |
|---|---|---|
| `own_side` | yes/no | Motion read aloud, the other side's view, conceding and stopping, and misheard words that reverse the meaning. The words are read as transcribed. |
| `standalone` | yes/no | A clip that opens on "that", "they" or "the first one", starts mid-thought, or answers something unheard. |
| `complete` | yes/no | The punchline or conclusion comes right after the clip ends. |
| `responds_to` | choice | Which of the other speaker's clips this one answers, or `none`, or `elsewhere`. |

`opener` is then asked over all the clips at once: which clip would make a stranger keep watching as the reel's opener? It is asked three times in seeded shuffled orders, and the answers are averaged.

## Output, per clip, in input order

- `hook` (0–1): the clip's percentile on the averaged opener probability among this run's clips. It can be compared within a debate, not across debates.
- `decisions`:
  - `own_side`, `standalone`, `complete`: probabilities of yes.
  - `opener`: the averaged opener probability.
  - `answers_elsewhere`: the probability that the clip replies to a point no listed clip contains.
- `responds_to`: the input index of the clip it answers, or null.

How to weigh these is the caller's choice. geo-chat treats them as soft penalties: a strong clip with a small flaw can still beat a weak clean one.

## Model, limits, failure

- It uses the same decision provider and model setting as `claims.score_highlights` (`claims_highlights_provider`, `claims_highlights_model`), so the two move together.
- It also shares that task's `decisions_global` rate limit and its bound on requests in flight per worker.
- A run of N clips costs N + 3 requests.
- A run is all or nothing. One failed request, or the provider's identical-distribution glitch (the same answers for every clip, or a flat opener round), fails the run, and the engine retries it.
- Every question text is a prompt key under `claims_score_clips.debate.*`, so it can be overridden per run (`GET /prompts`).

## Not yet measured

The rubric is written from the review's defect tags but has not been run against them on the provider. That evaluation is part of GEO-3614.
