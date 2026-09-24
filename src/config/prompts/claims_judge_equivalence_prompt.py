"""Rubric for claims.judge_equivalence. Moved here verbatim from
src/extraction/claim_equivalence_judge.py so the prompt registry can catalog
it; the judge prepends it to the claim + candidates listing. Plain text, no
format slots."""

CLAIMS_JUDGE_EQUIVALENCE_RUBRIC = """You compare a CLAIM against CANDIDATE claims. Two claims are EQUIVALENT only when they are the
same claim: any situation that makes one true makes the other true, and any situation that makes
one false makes the other false. Wording, word order, synonyms and sentence structure never matter;
what is asserted does.

Do this for EVERY candidate, in order:
1. Say what the CLAIM asserts and what the CANDIDATE asserts, naming the kind of assertion each one
   is: an effect ("X suppresses turnout"), an intent or purpose ("X is meant to…"), a necessity
   ("X is needed"), a signal or evidence ("X shows that…"), a mechanism ("X works by…"), a frequency
   ("Y is rare"), a scope ("nationwide"), a value judgement ("the burden is too high").
   Two sentences of different kinds are NOT equivalent, however consistent they are. A purpose is
   not an effect; a mechanism is not a purpose; a signal is not a necessity.
2. Ask: if the CLAIM is true, MUST the CANDIDATE be true? Then: if the CANDIDATE is true, MUST the
   CLAIM be true? Answer each strictly. "Would follow", "is consistent with", "supports", "is the
   reason for" are all NO.
3. Name the relation and the decisive difference.

The traps to avoid — these are NOT equivalence:
- same topic, same policy, same side of the debate, or mutually supportive statements
- one sentence entails the other but not the reverse (more specific vs more general: an added cause,
  condition, consequence, group, number, place or date on one side only)
- different hedging or modality: "may be" vs "is", "can" vs "does", "always" vs "often"
- intent vs effect; how often something happens vs whether it ever mattered; a requirement vs its
  justification; a group vs the intersection of two groups
- different quantities, dates, places, actors; opposite polarity

Judge each candidate on its own against the CLAIM; the other candidates are not context for it.
Set `unsure` only when a careful reader could not decide either direction."""
