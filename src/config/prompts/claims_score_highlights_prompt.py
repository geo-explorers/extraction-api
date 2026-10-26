"""Rubric for claims.score_highlights, per media type.

A decision model answers questions about one extracted claim at a time, with
the whole discussion and the other extracted claims in view. Plain text, no
format slots; every text is a prompt key, so a run can override any of them
through `prompt_overrides`.

The highlight score is one yes/no question: if this claim were removed from
the list, would the list misrepresent the discussion? The probability of yes
is the score (src/pipeline/claims_score_highlights_core.py).

How that question was chosen (2026-10-02 → 2026-10-05, measured against one
reader's picks over 444 claims from 30 Geo testnet debates): three role
decisions (position / clash / turn) ranked the picks at AUC 0.75, a holistic
"is this pivotal?" at 0.75, and this counterfactual one at 0.80 — the best of
anything tried, alone or combined. It is strict on purpose: almost every claim
carries some information, so a wording that asked about "important
information or context" said yes to most claims (median 0.83); the bar is a
major loss, and the median is 0.38.

Three further scores ride in the same request, each a four-level scale whose
levels are listed from the lowest to the highest (the model sees the
descriptions in that order, never the level names, which only key the prompt
registry): relevance — how directly the claim bears on the main claim;
quality — whether the claim stands as a faithful, single, self-contained
statement; controversy — how likely a general audience is to split on it.
They are reported beside the highlight score, not folded into it.

Two more questions read the claim list instead of the transcript: which claim
a viewer would most want to watch (a choice over all the claims, asked three
times in shuffled order, once per debate), and which other claim makes the same
point as this one, or none (a choice per claim, with no transcript). The first
ranks the claims for the viewer, which the coverage question above does not: it
measures whether the list would lose something, not whether anyone would stop
scrolling for it. The second finds the points the debate makes twice, which the
coverage question punishes on both copies.
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

# ── Relevance: how directly the claim bears on the main claim ────────────────

DEBATE_RELEVANCE_INSTRUCTIONS = (
    "The title is the main claim this debate argues about. How directly does the claim under test "
    "bear on the main claim? Judge the claim's content against the main claim — not its stance, "
    "and not whether the speaker meant it as an argument: a reason for, a reason against and a "
    "relevant fact count equally. Rate the link as it stands in the claim's own words plus what "
    "the transcript makes plain."
)
DEBATE_RELEVANCE_LEVELS = {
    "unrelated": (
        "The claim is not about the main claim's subject: a remark about the debate itself or the "
        "opponent, an aside, a different topic."
    ),
    "tangential": (
        "The claim is about the same topic, but its bearing on the main claim is unclear or needs "
        "premises nobody states: background, history, a definition, a related but different question."
    ),
    "connected": (
        "The claim bears on the main claim through one clear step that the debate makes explicit: "
        "a premise of an argument for or against it, a comparison, a cause, or a cost or benefit "
        "of what the main claim proposes."
    ),
    "direct": (
        "The claim takes a position on the main claim itself, or gives a reason, consequence or "
        "piece of evidence that bears on it with no further step: a reader sees at once what it "
        "means for the main claim."
    ),
}

# ── Quality: does the claim stand as a faithful, single, self-contained statement ──
#
# Strict in the same way as the highlight question: a first wording that asked
# "how good is this claim?" rated 427 of the 444 evaluation claims at the top
# level (2026-10-07). This one asks for the flaw and reserves the top level for
# a claim where none is found.

DEBATE_QUALITY_INSTRUCTIONS = (
    "Rate the claim under test as a standalone statement extracted from the transcript. Compare "
    "it with the passage it was drawn from and look for a flaw: it says something the speaker did "
    "not, it cannot be understood without the transcript, it bundles several assertions, or it is "
    "too vague to be argued with. Reserve the top level for a claim in which you find nothing an "
    "editor would change; most extracted claims have at least a small flaw, so a lower level is "
    "the usual answer. Do not judge whether the claim is true, important or well argued."
)
DEBATE_QUALITY_LEVELS = {
    "misleading": (
        "Misrepresents the passage: asserts something the speaker did not, drops a hedge or "
        "condition that changes the meaning, attributes a view the speaker was describing rather "
        "than holding, or is not a claim at all (a question, a fragment, a remark about the debate)."
    ),
    "unusable": (
        "Cannot be judged on its own: it needs the transcript to know what it refers to (a bare "
        "'this', 'the study', 'they'), or it bundles two or more separate assertions, or it is so "
        "general that nothing would count against it."
    ),
    "flawed": (
        "A single, faithful, self-contained assertion with one flaw a careful editor would still "
        "fix: a vague quantity or subject ('some people', 'many'), a missing scope (where, when, "
        "who), or wording longer or looser than the passage."
    ),
    "clean": (
        "Nothing to fix: one assertion, specific in subject and predicate, with the speaker's scope "
        "and hedges kept, that a reader who has not seen the transcript understands and could "
        "argue with as it stands."
    ),
}

# ── Controversy: how likely a general audience is to split on the claim ──────

DEBATE_CONTROVERSY_INSTRUCTIONS = (
    "How likely is the claim under test to generate disagreement among a general audience "
    "watching this debate? Judge the claim's content as viewers would receive it: whether "
    "reasonable, informed people would split on it. Not whether the two debaters disagreed, not "
    "whether the claim is true, and not how strongly it is worded: a plain fact, a definition or "
    "a truism generates no disagreement however boldly it is put; a value judgement, a policy "
    "position or a contested empirical question does."
)
DEBATE_CONTROVERSY_LEVELS = {
    "none": (
        "A plain fact, definition, truism or personal anecdote nobody would dispute, or not "
        "something one can agree or disagree with."
    ),
    "mild": "Broadly accepted, or an opinion few would bother to dispute; disagreement would be rare.",
    "debatable": "Most viewers would lean one way, but a real minority would push back.",
    "divisive": (
        "Reasonable people split on it: a value judgement, a policy stance or a contested "
        "empirical question where a large share of viewers would disagree."
    ),
}

# ── Moment: which claim would a viewer most want to watch ────────────────────
#
# One choice over every claim of the debate. The claims are the options, so the
# model weighs them against each other; the probabilities of one answer sum to
# one, which makes them comparable within a debate and not across debates.

DEBATE_MOMENT_INSTRUCTIONS = (
    "The options are the claims extracted from this debate. Which one would a viewer most want to "
    "watch as a clip of the debate? Think of someone choosing what to play from a list of moments: "
    "they stop for a claim that makes a point land, takes a clear side, or sets the two debaters "
    "against each other, and pass over background facts, hedges, set-up and small details. Where "
    "the same point is made twice, either copy is as good as the other. Judge the claim, not where "
    "in the debate it was said."
)

# ── Repeat: which other claim makes the same point ───────────────────────────
#
# One choice per claim, answered without the transcript: the options are the
# other claims of the debate (number, speaker, passage and text) and `none`.
# The wording is the same-point rubric the three labellers of 2026-10-23 worked
# from, so the question is measured against labels that mean the same thing by
# "same point". A variant that spelled out the cross-speaker case and asked
# whether the viewer would learn anything new moved the answers by 0.004 on
# average, so the plain wording is kept. Asked as a choice over the whole list
# rather than a yes/no per pair: the pair form ranked the labelled pairs at AUC
# 0.971 against 0.989 for this form, which weighs each candidate against the
# others and against `none`.

DEBATE_REPEAT_INSTRUCTIONS = (
    "The target claim was extracted from a recorded debate (the title is the motion). The options are the other "
    "claims extracted from the same debate. Pick the option that makes the SAME POINT as the target claim: a viewer "
    "who had just heard the target would feel they were hearing the same thing again. Same point: the same assertion "
    "in other words, the same side's same argument at another level of detail or generality, a looser or stricter "
    "quantifier on the same assertion, a general rule and its application, or the same conclusion reached by the "
    "same reasoning. The speaker does not matter. Different points: a different reason for the same conclusion, a "
    "cause and its consequence, a claim and a reply to it, the same topic with a different assertion, or an added "
    "condition, mechanism or example that changes what is asserted. If no option makes the same point, pick none."
)
DEBATE_REPEAT_NONE = "No other claim in this debate makes the same point as the target claim."

# The scales by axis name: instructions and the level texts, lowest first. The
# prompt registry turns these into keys `claims_score_highlights.debate.<axis>`
# and `.<axis>.<level>`; the task reads them back by the same names.
DEBATE_AXES = {
    "relevance": (DEBATE_RELEVANCE_INSTRUCTIONS, DEBATE_RELEVANCE_LEVELS),
    "quality": (DEBATE_QUALITY_INSTRUCTIONS, DEBATE_QUALITY_LEVELS),
    "controversy": (DEBATE_CONTROVERSY_INSTRUCTIONS, DEBATE_CONTROVERSY_LEVELS),
}
