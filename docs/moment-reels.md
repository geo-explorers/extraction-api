# Moment reels: transcript.punctuate and debates.pick_moments

These two tasks build a debate's moment reel for geo-chat (GEO-3638). A moment reel is four clips that play back to back, alternating speakers, and tell the debate's story in about a minute.

Previously the reel was stitched together from clips cut around extracted claims. Claim extraction was built to fill the knowledge graph, not to make a reel. On 10 Nov 2026, 192 of the 557 reels on testnet had only three clips:

- Several claims shared one stretch of speech, so one debater ended up with a single usable clip.
- The picker preferred three clips in the order said over four that alternate.
- Clips judged to argue the wrong side were dropped, with nothing to replace them.
- Transcripts lost their punctuation partway through, so no clean sentence ends could be found.

Both tasks run on Claude, using the `moments_claude_*` settings. They write an answer rather than choosing among options, so they can't use the decision model.

**Budgets.** Both count against the `claude` spend-guard key that the news Claude paths use, so one hourly budget covers every Claude call. They draw on the `claude_global` per-minute rate limit at one unit per call the run can make (2 for punctuate, 5 for pick: 3 answers and 2 judge answers), as the news tasks reserve. Punctuate runs one at a time and pick four at a time, so a backfill can't starve news. Each answer's token counts are logged (`… used input_tokens …, output_tokens …`).

**Asking again.** Each task checks the answer. Problems go back to Claude in the same conversation: punctuate gets 2 answers, pick gets 3. The first answer with no problems wins. An answer with only soft problems is kept if the later ones get worse. The system prompt and the transcript are prompt-cached, so a send-back rereads them from cache. Soft problems go back only once an answer has no hard ones, so the model isn't pulled in two directions at once.

**JSON only.** Pick and its judge answer under a JSON schema (Anthropic structured outputs, `output_config.format`), so the reply is the JSON and nothing else. On 11 Nov, before this, 8 of 20 pick runs were cut off at the 16,000-token limit: the model wrote its reasoning out before the JSON, which itself runs about 2k tokens. Punctuate is asked as before, without a schema.

**Cut off at the token limit.** The answer's last 500 characters are logged. Pick and its judge send a cut-off answer back once, told it was cut off and to answer shorter with the JSON alone; a second cut-off ends the run. Punctuate never sends one back: every sentence end is required, so a second answer would be as long. Either way, an acceptable answer seen before the cut-off is kept; a pick's last resort (see below) is kept only once the retry is used up.

**Failures.** A failed request (network, rate limit, overload) is retryable, and Hatchet runs the task again. These are final (`FinalDecisionError`, a `NonRetryableException`), because running again cannot help:
- no acceptable answer
- an answer cut off at the token limit (for pick, a second one)
- a missing key
- a tripped spend guard
- a request Claude rejects: any 4xx except 408, 409 and 429 (a schema structured outputs can't compile, a model without them, a prompt too long). The judge still keeps the first reel when its request is rejected.

Once pick holds an answer, acceptable or a last resort, a later spend-guard trip or failed request (any of the above) returns that answer instead of failing the run, since a rerun would throw it away. Punctuate and the judge never hold an answer they then ask again about, so this changes nothing for them.

The caller then falls back. Each task's execution timeout covers every request at its 300 s ceiling. The client doesn't retry on its own.

## transcript.punctuate

Whisper sometimes drops punctuation partway through a debate. Once one window comes back as a run-on, the next window is prompted with it and carries on the same way. This task restores sentence ends.

**Input:** the motion and the turns, each with its speaker and its words as transcribed.

```json
{"media_type": "debate", "title": "Voter ID should be required nationwide",
 "turns": [{"speaker": "Nico", "words": ["it", "is", "a", "federal", "problem", "so"]}]}
```

**Output:** for each turn, where each sentence ends, given as a word position and a mark.

```json
{"turns": [{"ends": [{"word": 4, "mark": "."}, {"word": 5, "mark": "."}]}], "sentences": 2,
 "provider": "anthropic", "model_used": "claude-sonnet-4-6"}
```

The answer never contains text, so:
- no word can change,
- a long debate can't get cut off at the output limit,
- and checking the answer is a bounds check.

A turn's last word always ends a sentence. If the model leaves it out, the task adds it with a ".".

## debates.pick_moments

**Input:**
- the motion
- the two speakers, with their sides
- the transcript as sentences: speaker, turn, start and end in ms, text
- the claims, with the caller's index and the speaker
- the clip length bounds, which default to 6–22 s hard and 10–18 s target

**How the reel is chosen:** the moments come first and the claims after. Claude finds each speaker's strongest moments in the transcript, whatever they are (a punchline, a question, a comeback), builds the reel from them, and only then links each clip to the claim of its speaker it best supports. The first clip is chosen for providing the context (what the debate is about, where its speaker stands), not for strength; the strongest moment often lands best last. Claims were extracted to fill the knowledge graph, so a reel built around them missed the lines that aren't claims.

Claude offers up to three candidate reels in one answer. Each is checked on its own. When more than one passes, a second call judges them the way a viewer meets them: the motion, who speaks for which side, and each clip's words and length, with no transcript and no claims. The words are the ones the viewer hears, past the filler geo-chat cuts (see below). It picks the reel a newcomer can follow to the end; a first clip that gives the context comes before anything else it weighs, and a reel isn't favored for opening on its strongest line. When any passing reel has no soft problem, only those are judged; otherwise all that passed are, and under each reel the judge is shown the problems found in how its clips play ("Clip 3 opens on "but"…", "Clip 2 starts halfway through a sentence", a clip cut to fit, a reel over its length), so it can prefer the clean one. These are about what the viewer hears, so the judge still sees no transcript, claims or story. If the judge can't answer, the first reel is taken.

**Output:** a one-sentence `story`, exactly four `moments` in play order, `candidates` (how many reels the pick was made from), `judge_why`, and `first_clip_problems` (empty unless the reel was kept as a last resort; see below). Claude writes the story with "Speaker 1" and "Speaker 2", and each becomes that speaker's short name: the first word of their name past any title, or the full name when both speakers share it. Each moment has:
- `speaker`
- `start_sentence` and `end_sentence` (inclusive)
- `start_ms` and `end_ms` (from those sentences)
- `claim_index`
- `claim_fit`
- `role` (opening, answer, rebuttal or closing)
- `answers` (the earlier clip it responds to, if any)
- `why`
- `trimmed_to_sentence` and `trimmed_from_sentence` (only on a reel kept as a last resort: where the model's own pick ended or started before the clip was cut to fit; see below)

The task checks each of these rules before answering:
- exactly four clips, alternating speakers
- each clip is whole sentences from one turn of one speaker
- each clip is within the hard length bounds (6–22 s)
- no two clips overlap
- each clip links to a claim by its own speaker, and a speaker's two clips link to different claims
- a clip only answers an earlier clip, by the other speaker
- the first clip opens on a sentence that starts a thought: not mid-sentence (in a transcript with capitals, a sentence opening on a lowercase word ran on from the words before it, as in "and not just get completely overwhelmed…"), not on the second half of a sentence the transcript split (below), and not on a connective or filler ("but", "because", "though", "that's", "that", "they", "they're", "yeah", "okay", "like", "agree", "I mean", "for example"…: the soft rule's list below, except "it's", "it" and "yes", which often state a position ("It's time we…", "It is the death of…") and stay soft on the first clip too). This holds for a first clip cut to fit too, since the viewer meets the debate there.

**Openers are judged on what the viewer hears.** geo-chat cuts the filler a clip opens on before it plays (`story_moments::filler_lead`, over `clip_ranges::FILLER`): "um", "uh", "er", "ah", "hmm", "so", "and" and "you know", along with any first sentence that is nothing but filler. It keeps "so" as an adverb ("So far…"), "you know" with its object ("You know nothing…"), and at least three words of the sentence ("So what now?" plays as said). Every opener rule here (the first clip's hard ones, the soft one on later clips, and the trim's choice of start) first skips that same filler, then reads what is left. So "Um, we need rules." and "And we need rules." open a first clip ("We need rules."), and "And so but…" is refused on "but". Whether a sentence starts mid-sentence is read from its first word as written, where the speech recogniser put its capital: "and not just get overwhelmed" is still refused, since the viewer hears a run-on "not just…". The copy of geo-chat's lists is `FILLER` and its neighbours in `debates_pick_moments_core.py`, and a test pins their contents so a change on either side is noticed. geo-chat keeps all of the filler when cutting it would leave the clip under its minimum length (6 s), and so does this reading: a skipped filler sentence is timed by its own times, and filler inside a sentence as its share of the sentence's words, since sentences carry no word times.

**Split sentences.** The transcript sometimes breaks one sentence in two and capitalises the second half: Whisper put a full stop mid-sentence ("…in the first place. Where corporations argued…"). The capital hides that a clip starting there starts mid-sentence, so the task finds such halves itself (`split_start`): the sentence before it in the same turn ends mid-phrase (on "a", "the", "and", "because"…), or it opens as a clause of the sentence before ("Where corporations argued…" but not "Where do we draw the line?", "That the border security can serve…" but not "That argument…", "To the idea of…" but not "To be fair"). The picker's transcript marks each one "second half of a split sentence", the prompt says never to start there, and a clip that does has an opener problem (hard on the first clip, soft on the rest). The sentences themselves are left as geo-chat sent them: merging them would renumber what geo-chat cuts on, and the mark is enough for the picker to start where the sentence really starts.

The first clip's rules are hard, but with a last resort: a reel that breaks only them is sent back like any other, and if no answer in the three has a reel that keeps every hard rule, the last answer's such reels are judged and one is kept, with the rules its first clip breaks in `first_clip_problems` (and a warning in the log). So the rule never fails a run that would have passed without it.

**Clips over the hard limit are picked again, not cut from the start.** Until 12 Nov a clip over the limit was shortened from its start, which cut the setup its story leaned on in 6 of 36 checkable clips of the preview (a story about crop losses played a clip about tomato-picking robots). Now it is a hard problem, sent back with how long the clip runs to each of its sentences ("From sentence 14 it runs 8.1 s to sentence 14, 19.0 s to sentence 15, 27.3 s to sentence 16"), so the next pick ends or starts where it fits. A reel whose only problems are such clips (and its first clip's opening) is a last resort, kept as above, with each such clip cut to fit (`fit_to_limit`): from its end, to the latest sentence that fits and is not the first half of a split sentence, so the setup stays (`trimmed_to_sentence`); only when no such end fits, from its start as before (`trimmed_from_sentence`, preferring a start that keeps the opener rules); and with neither, from its end anyway. Openers are checked on the clip as it plays, cut or not. A clip that can't be cut into the limits is reported at the length the model picked, and is no last resort.

Two rules are soft: a clip whose heard words open badly short of the first clip's hard rules (a later clip mid-sentence, on the second half of a split sentence, or on a dangling word such as "but", "because", "though", "it", "they're", "that", "for example"; or the first clip on "it's", "it" or "yes"), and a reel longer than four clips at the top of the target (72 s by default). A later clip may answer the clip before it: "Yes, …", "Yeah, …", "Agreed, …" are fine when the sentence goes on to a thought of its own (three more words that open cleanly), so "Yeah, I think the truth has a threshold" passes while "Yes." and "Yeah, but…" don't; and a clip that names the one before in its own words ("What is problematic to me with that thinking is…") was never caught by the rule. A reel with no soft problem is taken first. When every reel that passes has one, the answer is sent back with all of them and the other reels' problems, within the three answers, however many passed; the last acceptable answer is kept, so a send-back that comes back worse costs nothing, and its reels go to the judge with their problems shown. Until 12 Nov they were sent back only when exactly one reel passed, and about 17 later clips in the preview opened mid-thought.

An answer goes back to Claude with its problems listed only when no candidate passes, up to three answers in all. If no answer passes, the run fails without a retry (`FinalDecisionError`), and the caller falls back to its own picker.

The caller can pass `avoid`: sentence ranges an earlier reel used that the caller then refused, each with a reason. For example, a clip that `claims.score_clips` judged to argue the other side. No clip in the new reel may overlap them.

The order doesn't have to follow the order things were said, but an answer always plays after the clip it answers. The judgment rules (the first clip names the topic, ends after the point lands, argues its own side) are in the prompt, and the judge weighs the first clip's context before anything else. They're for the caller to check again, for example with `claims.score_clips`.

## Evidence

12 Nov 2026 preview (ranks 30-56, 20 debates, 19 reels; GEO-3691): an adversarial review traced about 17 later clips that opened mid-thought: about 6 on connectives only clip 1 was checked for, 2 lowercase run-ons, 4 sentences the transcript split, 2 starts moved by a trim, and 4 unintroduced references. Trims from the start cut the payoff's setup in 6 of 36 checkable clips. Replayed on those reels' clips, the new opener rules flag a later clip in 13 of the 19 reels (the old list flagged 5).

11 Nov 2026: the first 93 live story reels, graded against their transcripts, averaged 2.5/5. Most often a clearly better moment was skipped (65 reels), a clip opened mid-thought (63), a clip ran over 20 s (59), or a clip leaned on something the viewer never heard (56). Reels over 80 s graded worst. The moment-first prompt, the judge, the 22 s limit and the reel length rule answer those.

10 Nov 2026 prototype: 12 debates whose current reels had three clips. Every answer was four valid alternating clips, and a blind judge preferred the new reel in all 12 (story 7.6 vs 4.4, clean starts and ends 6.0 vs 3.3). The remaining flaws were mid-sentence cuts, which came from cutting on subtitle chunks. That's why this task works in whole punctuated sentences.
