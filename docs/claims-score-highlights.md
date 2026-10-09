# `claims.score_highlights`

A discussion and the claims extracted from it in; a **highlight score** per claim out, with three
**axis scores** beside it. The highlight score says how much a claim carries the discussion, so a
consumer can show the few claims that matter instead of all of them. The axes — relevance, quality,
controversy — describe the claim on three planes the highlight score does not: how directly it
bears on the main claim, whether it stands as a clean statement, and how likely an audience is to
split on it.

The task is a pure scorer. It does not extract, drop, or reorder claims, and it does not pick a
cut-off: what to do with a score (a threshold, the top N) is the consumer's decision. A consumer
composes two calls:

1. **Extract** with `claims.extract`.
2. **Score** here: enqueue `claims.score_highlights` with the same documents and the extracted
   claims.

Scoring is separate from extraction because a claim's role can only be judged with the whole
discussion in view, and because claims can then be re-scored without being re-extracted.

## Enqueue

```bash
curl -X POST "$EXTRACTION_API/tasks" -H "X-API-Key: $API_KEY" -H 'content-type: application/json' -d '{
  "type": "claims.score_highlights",
  "payload": {
    "media_type": "debate",
    "title": "Illegal immigration is a national security risk",
    "context": "David Lindstrom argues Agree; Adam Fischer argues Disagree.",
    "documents": [
      {"speaker": "David Lindstrom", "content": "OK, so illegal immigration is a national security risk. I think the vast majority…"},
      {"speaker": "Adam Fischer", "content": "Breaking immigration rules does not by itself…"}
    ],
    "claims": [
      {"id": null, "text": "The vast majority of undocumented immigrants do not present a security risk", "document_indices": [0]},
      {"id": null, "text": "Borders play a role in stopping trafficking", "document_indices": [1]}
    ]
  }
}'
```

- `media_type` selects the rubric. Only `debate` exists today; another value is a 422.
- `documents` are the passages in spoken order, each with its `speaker`. The speaker label is how
  the model attributes a claim and judges who answered whom.
- `claims[].document_indices` are the documents a claim was drawn from, as `claims.extract`
  returns them. An index outside `documents` is a 422.
- Limits: 200 documents, 100 claims, 400,000 characters of document content.

Poll `GET /tasks/{id}`. The result:

```json
{
  "claims": [
    {
      "index": 0, "id": null, "text": "The vast majority of undocumented immigrants…",
      "score": 0.29,
      "decisions": {"essential": 0.29, "relevance": 0.94, "quality": 0.97, "controversy": 0.88}
    }
  ],
  "claims_scored": 2,
  "provider": "perplexity",
  "model_used": "pplx-decider-v1.1-27b"
}
```

(The numbers are from the 2026-10-08 evaluation run, debate 12, claim 1.)

`score` is the highlight score; `decisions` holds every decision asked — `essential`, which is the
score, and the three axes described below — each 0 to 1.

`claims` carries every input claim in input order, each with a score. **A run scores every claim
or fails**: the first claim whose request fails stops the others and fails the run, so Hatchet
retries it (`retries=2`). A consumer stores what it gets and does not ask again, so a half-scored
list would stay half-scored for good; a failed run is the honest alternative. A refused key or an
unknown model fails on the first request rather than once per claim.

## How the score is reached

For each claim the model is shown the title, the context, the whole transcript with speaker labels,
every extracted claim, and the claim under test with its attribution. It answers one yes/no
question, and the probability of yes is the score:

> If the claim under test were removed from the list of extracted claims, would the list
> misrepresent the debate transcript? Most claims carry some information; answer yes only when
> the loss is major — a central point, decisive evidence or a key step of the exchange that no
> other claim in the list carries.

with "yes" defined as *without it the list misrepresents the debate: a reader would miss one of its
central points, a piece of evidence the debate turned on, or a step without which the exchange does
not make sense, and no other extracted claim carries it*, and "no" as *the list still represents the
debate without it: another extracted claim carries the same point, or what is lost is a detail, an
example, a side point or a minor piece of context*.

The question is counterfactual on purpose: it makes the model weigh the claim against the rest of
the list, which is how a reader choosing highlights works. It is strict on purpose: almost every
claim carries some information, so a wording that asked about "important information or context"
said yes to most claims (median 0.83); with the bar at a major loss the median is 0.38 and the
scores spread.

The wording lives in `src/config/prompts/claims_score_highlights_prompt.py`; the instructions and
both criteria are prompt keys (`claims_score_highlights.debate.essential`, `.yes`, `.no`, listed by
`GET /prompts`) and can be replaced for one run with `prompt_overrides`. `decisions` in the result
holds every decision asked, so a second question can be added and compared without changing the
contract.

Not covered: two claims that say nearly the same thing both lose some score (each makes the other
less essential), but nothing says which copy to keep.

## The three axes

The same request asks three more questions, each a **four-level scale**: the model gets the level
descriptions in order from the lowest to the highest and returns a probability per level; the
axis score is the probability-weighted position, 0 meaning surely the lowest level and 1 surely the
highest. They are reported in `decisions` beside the highlight score and are not folded into it.
The content is shared, so the three add only their own wording to each request: about half again
the input tokens of the highlight question alone (the rubrics are long; see the evaluation for the
figure), and no extra requests.

| Axis | Question | Levels, lowest first |
|---|---|---|
| `relevance` | How directly does the claim bear on the main claim (the title)? Content only — a reason for, a reason against and a relevant fact count equally. | unrelated · tangential (same topic, bearing unclear) · connected (one explicit step away) · direct (a position on the main claim, or a reason that bears on it with no further step) |
| `quality` | Compared with the passage it came from, what is wrong with the claim as a standalone statement? The top level only when nothing an editor would change is found. | misleading (not what the speaker said, or not a claim) · unusable (needs the transcript, or bundles several assertions) · flawed (one fix left: a vague subject, a missing scope) · clean |
| `controversy` | How likely is a general audience to split on it? Not whether the debaters disagreed, not whether it is true, not how boldly it is put. | none (a plain fact, a truism, an anecdote) · mild · debatable (most lean one way) · divisive (reasonable people split) |

The full wording is in the rubric module, under the keys `claims_score_highlights.debate.<axis>`
(instructions) and `.<axis>.<level>` (one text per level), all overridable per run. The level
names only key the registry; the model never sees them.

What a consumer should expect (from the evaluation below): relevance and controversy spread across
their scales and rank the references they were checked against well; quality is compressed near
the top — the extractor's claims mostly *are* clean — and what it ranks low is the bundled,
list-like claim and the one that sharpened what the speaker said. Because the model behind the
scores can change (see the evaluation), use them to rank claims within a debate rather than
against a fixed threshold.

## The decision model

The scorer talks to a **decision model**: a model that answers named questions about content with
probabilities instead of text. The interface is `src/decisions/base.py`:

```python
answers = await decider.decide(content, questions, model=model)
# content:   {"title": …, "transcript": …, …}        named sections, in reading order
# questions: {"essential": YesNo(…), "relevance": Scale(…)}   or a Choice(…) between named options
# answers:   {"essential": YesNoAnswer(probability),          or a ChoiceAnswer(choice, probabilities)
#             "relevance": ScaleAnswer(score, probabilities)}
```

A `Scale` is an ordered rubric (two or more level descriptions, lowest first); its answer is the
probability-weighted position on it, 0 to 1, with a probability per level. Perplexity answers it
natively (its `score` question type); a provider without one can ask a choice between the levels
and weight the result the same way.

The task knows only this interface. To use another model, implement `DecisionModel`, add it to
`_PROVIDERS` in `src/decisions/__init__.py`, and set the two variables below.

The first provider is Perplexity's [Decisions API](https://docs.perplexity.ai/docs/decisions/quickstart)
(`src/decisions/perplexity.py`).

## Configuration

| Variable | Meaning |
|---|---|
| `CLAIMS_HIGHLIGHTS_PROVIDER` | decision-model provider (default `perplexity`) |
| `CLAIMS_HIGHLIGHTS_MODEL` | that provider's model name (default `pplx-decider-v1.1-27b`); overridable per run with `llm_overrides` |
| `PERPLEXITY_API_KEY` | required for the `perplexity` provider |
| `DECISIONS_GLOBAL_RATE_PER_MIN` | task runs per minute across all workers (default `20`) |

Cost: one request per claim, each carrying the whole discussion. Three things keep requests under
Perplexity's 10 per second per organisation:

- a **per-worker bound of 6 requests in flight** across all runs the worker is executing (a
  per-run bound would multiply by the worker's slots);
- the task's **engine-level `concurrency=2`**: at most two runs at once across all workers, the
  rest queue;
- the `decisions_global` rate limit, which counts **runs**, not requests, and only smooths how
  fast runs start.

A `429` is retried up to 8 times, waiting the server's `Retry-After` capped at 30 s; an outage
(5xx, timeout, dropped connection) up to 3 times with backoff. Each is its own budget. A request
that outlasts its budget fails the run as retryable. Every request is counted by the spend guard
under the provider's name; a tripped guard fails the run.

## Evaluation

`tests/fixtures/claim_highlights_eval.json` holds the 30 debates with a label per claim: `selected`
(with its job) or the main reason it was left out. The labels are one reader's and **preliminary**.

```sh
uv run python scripts/eval_claim_highlights.py --dry-run           # print the first request; sends nothing
PERPLEXITY_API_KEY=… uv run python scripts/eval_claim_highlights.py [--debates N] [--out results.json]
```

It runs the task's own code over each debate and reports: AUC of the score for selected claims
against the rest, precision at k per debate (k = the reader's number of picks), the mean score per
label, and the axes against the references the set carries (relevance against the `peripheral`
and `background` labels, quality against `faithfulness_concern`, controversy against the
extractor's own `is_contestable` flag, kept on each claim for this). A full run is 444 requests,
about 1,170,000 input tokens with the four questions (760,000 with the highlight question alone —
the three rubrics are what the other 54% is).

### Results — `perplexity` / `pplx-decider-v1-27b`

Seven rubric versions, each a full run over the 444 claims (about 70 s; a request that hits the
10/s limit is retried and succeeds). Within a run scores are deterministic: repeat runs return
identical numbers, and a decision's answer does not change when other questions are added to or
removed from the request — the axes below were added without moving the highlight score.

**The model behind an id moves.** The v7 run of 2026-10-05 could not be reproduced on 2026-10-07
with the same prompt, content and model id: the median highlight score fell from 0.38 to 0.18,
claims at or above 0.5 from 117 to 45, with rank correlation 0.76 between the two runs and AUC
0.800 → 0.779. On 2026-10-08 `pplx-decider-v1-27b` returned the same numbers as the newly listed
`pplx-decider-v1.1-27b`, so the unversioned id had become an alias of the new model; the default
now names the versioned id. The lesson for a consumer: the ranking survives a model change far
better than the absolute values, so rank within a debate rather than cut at a fixed score.

| Version | Questions | AUC, selected vs rest | Precision at k | …counting runner-ups |
|---|---|---|---|---|
| v1 (2026-10-02) | `job` choice over 8 options × 3 yes/no checks (faithful, specific, stands alone) | 0.780 | 0.558 | 0.706 |
| v2 | three independent yes/no decisions, one per job; score = P(at least one) | 0.750 | 0.484 | 0.662 |
| v3 | one choice: `position` / `clash` / `turn` / `none`; score = 1 − P(none) | 0.774 | 0.541 | 0.728 |
| v4 | v2 + holistic `pivotal` (alone: 0.752 / 0.529 / 0.674) | 0.750 | 0.484 | 0.662 |
| v5 | v4 + `essential`, loose wording (alone: 0.580; a duplicate detector, restatement vs rest 0.865) | 0.750 | 0.484 | 0.662 |
| v6 | v5 with `essential` strict (alone: 0.800 / 0.596 / 0.738) | 0.750 | 0.484 | 0.662 |
| **v7 (2026-10-05, current)** | `essential` strict, on its own | **0.800** | **0.596** | **0.738** |

Chance precision at k is 0.270; the existing `is_contestable` flag reaches AUC 0.549.

What the runs showed:

- **The counterfactual question beat every rubric about the claim's role.** Three role decisions
  (position, clash, turn) reached 0.75 however they were combined; a holistic "is this pivotal?"
  0.75; the strict "would the list misrepresent the debate without it?" 0.80 alone, and no
  combination with the others improved on it by more than 0.016 AUC while costing precision.
- **The bar matters more than the question.** The loose wording of the same question (important
  information or context) said yes to most claims and ranked at 0.58; raising the bar to a major
  loss gave 0.80.
- **Per label (v7):** selected claims average 0.54 (median 0.55), runner-ups 0.44, supporting and
  restatements 0.36–0.37, background 0.26, peripheral 0.18, anecdotes 0.17. Among the reader's
  picks, positions score 0.60, turns 0.51, clash claims 0.46 — evidence the other speaker took up
  is the kind of highlight it credits least.
- **Per debate (v7):** of the reader's picks, the top k by score recovers all of them in 1 debate,
  all but one in 15, and fewer in 14; the weakest is 1 of 3.

### Results — the axes, `pplx-decider-v1.1-27b`, 2026-10-08

One full run of the shipped questions (the highlight question plus the three scales, 444 claims in
99 s). The axes have no labels of their own, so each is checked against what the set does carry.

| Axis | Median (p10–p90) | Check | AUC |
|---|---|---|---|
| `relevance` | 0.76 (0.57–0.89) | other claims vs the reader's `peripheral` / `background` | 0.828 / 0.740 |
| `quality` | 0.88 (0.73–0.94) | other claims vs the reader's `faithfulness_concern` | 0.774 |
| `controversy` | 0.75 (0.44–0.91) | the extractor's `is_contestable` true vs false | 0.892 |

- The axes are different instruments from the highlight score: as rankers of the reader's picks
  they reach 0.67 (relevance), 0.55 (quality) and 0.70 (controversy) against 0.78 for the
  highlight question; relevance and controversy share about half their ranking (Spearman 0.53),
  quality correlates with nothing.
- **Relevance** follows the stance pattern: claims the stance backfill called `addresses` sit at
  *connected*, directional ones at *direct*; the reader's peripheral claims are the lowest. No
  claim reached *unrelated* — the extractor already filters those.
- **Controversy** puts background facts at 0.34 and anecdotes at 0.44 against 0.80 for the
  reader's picks; the lowest are definitions and dates ("the NATO alliance was created during the
  Cold War"), the highest policy positions.
- **Quality** is the strict wording of a first version that rated 427 of 444 claims at the top
  level; it still sits near the top (191 claims in 0.8–0.9, 176 above) because most extracted
  claims pass its tests, and what it ranks low is the right thing: bundled lists of assertions
  and the claims that sharpened the speaker's words. Treat it as a flag for the bottom of a
  debate's list rather than a spread to rank the top by.
- The same questions asked as a `choice` between named levels (the 2026-10-07 trial) rank the
  claims identically (Spearman 0.99 per axis); the level names add nothing.

Per-claim results for every run are in `~/Documents/debate-highlight-eval/results/` (not in the
repo). To print one debate as a table from a saved run:

```sh
uv run python scripts/eval_claim_highlights.py --table 12 --from results.json
```
