"""Rubric for claims.score_highlights, per media type.

A decision model answers one yes/no question about one extracted claim at a
time, with the whole discussion and the other extracted claims in view: if
this claim were removed from the list, would the list misrepresent the
discussion? The probability of yes is the claim's score
(src/pipeline/claims_score_highlights_core.py). Plain text, no format slots;
every text is a prompt key, so a run can override any of them through
`prompt_overrides`.

How this question was chosen (2026-10-02 → 2026-10-05, measured against one
reader's picks over 444 claims from 30 Geo testnet debates): three role
decisions (position / clash / turn) ranked the picks at AUC 0.75, a holistic
"is this pivotal?" at 0.75, and this counterfactual one at 0.80 — the best of
anything tried, alone or combined. It is strict on purpose: almost every claim
carries some information, so a wording that asked about "important
information or context" said yes to most claims (median 0.83); the bar is a
major loss, and the median is 0.38.
"""

DEBATE_ESSENTIAL_INSTRUCTIONS = (
    "If the claim under test were removed from the list of extracted claims, would the list "
    "misrepresent the debate transcript? Most claims carry some information; answer yes only when "
    "the loss is major — a central point, decisive evidence or a key step of the exchange that no "
    "other claim in the list carries."
)
DEBATE_ESSENTIAL_YES = (
    "Without it the list misrepresents the debate: a reader would miss one of its central points, "
    "a piece of evidence the debate turned on, or a step without which the exchange does not make "
    "sense, and no other extracted claim carries it."
)
DEBATE_ESSENTIAL_NO = (
    "The list still represents the debate without it: another extracted claim carries the same "
    "point, or what is lost is a detail, an example, a side point or a minor piece of context."
)
