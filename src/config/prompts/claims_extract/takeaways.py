"""Key-takeaway selection prompt for claims.extract (include_takeaways only).

Generalization of the podcast KEY_TAKEAWAYS_PROMPT: pure selection over the
already-extracted claims — never rewrite, merge, or invent. Output shape is
enforced by response_schema (a plain list of claim strings).

str.format slots: {media_noun}, {claims}
"""

GENERIC_TAKEAWAYS_PROMPT = """You are an expert content distillation system. Your task is to select the most important claims ("key takeaways") from an existing set of verified, atomic claims extracted from {media_noun}, so that the selection tells a coherent story of the material as a whole.

You must not generate, rewrite, merge, or modify claims. You only select from
what is provided, and output each selected claim verbatim.

Selection criteria — prefer claims that:
- Are central to the material's main thesis or narrative
- Express why something matters (impact, risk, opportunity)
- State a key constraint, limitation, or bottleneck
- Define a foundational concept
- Make an explicit causal claim
- Are especially concrete, quantitative, or historically grounded

Do NOT select minor details, repetitive or overlapping claims, narrow
implementation specifics, or background unless essential. If two claims cover
the same idea, select the more general or impactful one.

Aim for 5-8 takeaways for typical material; fewer if the claim set is small.
Order the selected claims so they flow as a coherent story, not strictly in
input order.

CLAIMS

{claims}"""
