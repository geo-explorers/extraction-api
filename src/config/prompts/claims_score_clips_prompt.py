"""Rubric for claims.score_clips, per media type.

A decision model answers questions about one candidate clip at a time, with the
whole debate transcript and every candidate clip in view; one more question
compares the clips with each other. Plain text, no format slots; every text is a
prompt key, so a run can override any of them through `prompt_overrides`.

Each question names a defect the 10 Nov 2026 review of 65 debates' moment reels
found again and again (GEO-3614), and asks about the clip *as heard*: the words
between the cut points, not the claim the clip was cut for.

- own_side: the review's "claim-reading" and "wrong-side" clips, where a debater
  reads out the motion they argue against, concedes and stops, or ends on the
  view they reject; and its "garbled" clips, where a misheard word reverses the
  meaning ("advocate" for "abdicate").
- standalone: "mid-start" clips that open on "that", "they" or "the first is",
  and rebuttals whose prompt the reel never shows.
- complete: "cut punchline" clips that stop one sentence before the payoff.
- responds_to: which other clip a rebuttal answers, so a picker can keep a
  prompt and its answer together or leave the answer out.
- opener: "weak opener" reels whose first clip is a flat statement while a
  sharper line sits in the same debate. A choice over all the clips, asked in
  shuffled orders and averaged, like claims.score_highlights' moment question.
"""

DEBATE_OWN_SIDE_INSTRUCTIONS = (
    "The title is the motion of a recorded debate, and the context says which side each debater "
    "argues. The clip under test is the exact words a viewer will hear, cut from the transcript. "
    "Taken as heard, on its own, does the clip argue its speaker's own side? Read the words as "
    "transcribed, not what the speaker probably meant: a misheard word that reverses the meaning "
    "reverses it for the viewer too. Reading the motion aloud before arguing against it, "
    "describing the other side's view, or granting a point does not count as arguing one's side "
    "unless the clip goes on to the speaker's own answer."
)
DEBATE_OWN_SIDE_YES = (
    "A viewer hearing only this clip comes away with the speaker's own position or a reason for "
    "it: the clip puts forward the side the context gives them."
)
DEBATE_OWN_SIDE_NO = (
    "A viewer could come away with the wrong idea of the speaker's side: most of the clip is the "
    "motion read aloud or the other side's view, it grants a point and stops before the reply, it "
    "ends on the view the speaker rejects, or a transcription error makes it say the opposite of "
    "what the speaker argues."
)

DEBATE_STANDALONE_INSTRUCTIONS = (
    "Imagine a viewer who has seen nothing else of this debate: the clip plays alone, with only "
    "the motion on screen. Would they understand what the speaker is saying?"
)
DEBATE_STANDALONE_YES = (
    "Yes: the clip opens on its own subject, every 'this', 'that', 'they' or 'it' has its "
    "referent inside the clip or in the motion, and its point is clear from its own words."
)
DEBATE_STANDALONE_NO = (
    "No: it opens on a reference to something outside the clip ('that', 'they', 'the first one', "
    "'as I said'), starts partway through a thought, or only makes sense as a reply to a point "
    "the viewer has not heard."
)

DEBATE_COMPLETE_INSTRUCTIONS = (
    "Find the clip under test in the transcript and read what the same speaker says right after "
    "its last word. Does the clip end once its point has landed?"
)
DEBATE_COMPLETE_YES = (
    "Yes: the clip ends on its conclusion or punchline. What the speaker says next is a new "
    "point, a repetition, filler, or the end of their turn."
)
DEBATE_COMPLETE_NO = (
    "No: the speaker's punchline, conclusion, the example that makes the point, or the second "
    "half of a contrast ('…but actually…', '…and that's why…') comes right after the clip ends."
)

DEBATE_RESPONDS_TO_INSTRUCTIONS = (
    "Does the clip under test reply to a point the other debater made, so that a viewer needs "
    "that point to follow it? If so, pick the other debater's clip that contains that point. If "
    "the clip makes its own point without needing anything the other debater said, pick none. If "
    "it replies to something the other debater said that none of these clips contains, pick "
    "elsewhere."
)
DEBATE_RESPONDS_TO_NONE = "The clip makes its point on its own; it does not depend on anything the other debater said."
DEBATE_RESPONDS_TO_ELSEWHERE = (
    "The clip replies to a point the other debater made, but none of the listed clips contains "
    "that point."
)

DEBATE_OPENER_INSTRUCTIONS = (
    "The options are candidate clips from one debate, each the exact words a viewer will hear. "
    "Which one should open a one-minute highlight reel for a stranger scrolling past? Pick the "
    "clip most likely to make them stop and keep watching: a sharp line, a surprising claim, a "
    "vivid example or a clear clash of the two sides, that makes sense on its own and fairly "
    "states its speaker's side. Pass over flat statements of position, set-up, hedges, a clip "
    "that reads out the motion, and one that needs earlier context. Judge the words, not where "
    "in the debate they were said."
)
