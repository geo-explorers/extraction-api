# `claims.judge_duplicates`

Given claims, find already-published Geo claims that mean **exactly** the same thing.

1. **Candidates** come from [geo-lens](https://github.com/geobrowser/geo-lens), the local cache
   of Geo subgraphs: for each claim, the `vector` (cosine over embeddings) and `text` (BM25)
   strategies are queried on its claims cache, merged by id with the best score kept, and the
   claim's own id excluded. No Geo API call is made.
2. **Judgement** is one Gemini call per claim over all its candidates, with a strict rubric:
   same referents, quantities, time, polarity and hedging; paraphrase and word order do not
   matter; a claim that entails, contains, specifies or generalizes another is *different*.
3. **Output** per claim: every judged candidate with `same | different | unsure`, plus the
   `same` and `unsure` id lists. The service persists nothing; the caller decides what to do.

## Enqueue

```bash
curl -X POST "$EXTRACTION_API/tasks" -H "X-API-Key: $API_KEY" -H 'content-type: application/json' -d '{
  "type": "claims.judge_duplicates",
  "payload": {
    "claims": [{"id": null, "text": "Sunlight exposure can cause skin cancer."}],
    "k": 10, "min_score": 0.75, "strategies": ["vector", "text"], "space_ids": []
  }
}'
```

Poll `GET /tasks/{id}`; the result carries `results[].matches[]`, `results[].same`, `model_used`,
`llm_calls`.

## Configuration

| Variable | Meaning |
|---|---|
| `GEO_LENS_URL`, `GEO_LENS_API_KEY` | geo-lens endpoint and this service's consumer key |
| `GEO_LENS_CLAIMS_CACHE` | handle of the claims cache (default `claims`) |
| `CLAIMS_DEDUP_MODEL`, `CLAIMS_DEDUP_TEMPERATURE` | the judge model (default `gemini-2.5-flash`, 0.0) |

Cost: one Gemini call per claim that has candidates (rate-limited under `gemini_global`, counted by
the spend guard); two geo-lens queries per claim, each tens of milliseconds.
