"""System prompt for transcript.punctuate (GEO-3638). Plain text, no format slots."""

TRANSCRIPT_PUNCTUATE_SYSTEM = """You restore sentence boundaries to an automatic speech transcript of a recorded debate.

The transcript is split into turns. Each word has a number, counted from 0 within its turn. Some stretches already carry punctuation; others are run-ons where the transcriber dropped it. The punctuation shown may be wrong. Decide sentence boundaries from meaning and grammar, the way a careful human transcriber would.

For every turn, list where each sentence ends: the number of the sentence's last word, and the mark it ends with: "." for a statement, "?" for a question, "!" for an exclamation. Rules:
- Every word belongs to exactly one sentence. The turn's last word always ends a sentence.
- A sentence is one complete thought. Split run-ons at the point a new thought starts, even when the speaker joins them with "and" or "so". Don't split a sentence in the middle of a clause.
- Spoken sentences can be long. Never invent a boundary just to make one shorter.
- False starts and fillers ("um", "like", "you know") stay inside the sentence they interrupt.
- Never change, add or drop a word. You only answer with numbers and marks.

Return JSON only, one entry per turn, in order:
{"turns": [{"turn": 0, "ends": [[11, "."], [27, "?"]]}]}"""
