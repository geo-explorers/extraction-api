STANDALONE_CLAIM_KEYWORDS_PROMPT = """You are given a list of claims. Extract {min_keywords} to {max_keywords} keywords for each claim.

Rules:
- Keywords should capture the main concepts/entities in each claim
- Use sentence case (e.g., "Bitcoin", "Federal Reserve", not "BITCOIN")
- Prefer canonical names (Wikipedia-style)
- Each keyword should be 1-3 words max

Claims:
{claims_json}

For each claim, extract the keywords based on the rules above."""
