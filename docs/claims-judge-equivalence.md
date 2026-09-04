# `claims.judge_equivalence`

One claim and a group of candidate claims in; the candidates that are **logically equivalent**
to the claim out. Equivalent means the same truth conditions: accepting one commits a reader
to the other, in both directions. Similar, very close, or one-way entailment is **not**
equivalent.

The task is a pure judge. It does not fetch candidates. Whoever needs "the published claims
that mean exactly this" composes two calls:

1. **Retrieve** candidates from [geo-lens](https://github.com/geobrowser/geo-lens):
   `POST /caches/{handle}/query` with the `vector` strategy (cosine over embeddings) and, if
   wanted, the `text` strategy (BM25). Merge by id.
2. **Judge** them here: enqueue `claims.judge_equivalence` with the claim and the candidates.

Keeping retrieval out of the task means geo-lens and extraction-api never need to know about
each other; the consumer owns the composition.

## Enqueue

```bash
curl -X POST "$EXTRACTION_API/tasks" -H "X-API-Key: $API_KEY" -H 'content-type: application/json' -d '{
  "type": "claims.judge_equivalence",
  "payload": {
    "claim": {"id": null, "text": "Sunlight exposure can cause skin cancer."},
    "candidates": [
      {"id": "bd25feeb…", "text": "Sunlight exposure can cause skin cancer"},
      {"id": "d544f8c0…", "text": "Waking up early provides more access to natural sunlight"}
    ]
  }
}'
```

Poll `GET /tasks/{id}`. The result:

```json
{
  "claim": {"id": null, "text": "Sunlight exposure can cause skin cancer."},
  "equivalent": [{"index": 0, "id": "bd25feeb…", "text": "…", "verdict": "equivalent", "rationale": "…"}],
  "unsure": [],
  "judged": [ {"index": 0, …, "verdict": "equivalent"}, {"index": 1, …, "verdict": "not_equivalent"} ],
  "model_used": "gemini-2.5-flash"
}
```

`judged` carries every candidate with its verdict and a one-sentence rationale, in input order;
an index the model skips becomes `unsure` rather than disappearing. Ids are caller-opaque and
echoed back, so the consumer can pass Geo entity ids straight through.

## Configuration

| Variable | Meaning |
|---|---|
| `CLAIMS_EQUIVALENCE_MODEL` | judge model (default `gemini-2.5-flash`) |
| `CLAIMS_EQUIVALENCE_TEMPERATURE` | default `0.0` |

Cost: exactly one Gemini call per task run, rate-limited under `gemini_global` and counted by the
spend guard. Up to 50 candidates per call.
