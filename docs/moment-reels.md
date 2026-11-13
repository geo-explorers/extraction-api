# Moment reels: transcript.punctuate and debates.pick_moments

These two tasks build a debate's moment reel for geo-chat (GEO-3638). A moment reel is four clips that play back to back, alternating speakers, and tell the debate's story in about a minute.

Previously the reel was stitched together from clips cut around extracted claims. Claim extraction was built to fill the knowledge graph, not to make a reel. On 10 Nov 2026, 192 of the 557 reels on testnet had only three clips:

- Several claims shared one stretch of speech, so one debater ended up with a single usable clip.
- The picker preferred three clips in the order said over four that alternate.
- Clips judged to argue the wrong side were dropped, with nothing to replace them.
- Transcripts lost their punctuation partway through, so no clean sentence ends could be found.

Both tasks run on Claude. They write an answer rather than choosing among options, so they can't use the decision model.

**Models (GEO-3702).** Punctuate runs on `moments_claude_*`: `claude-sonnet-4-6` at temperature 0.2, 16,000 tokens. Pick, its judge and its claim-link call run on `moments_pick_claude_*`: `claude-sonnet-5-5` at effort `low`, 32,000 tokens. Thinking counts toward the token limit, so the limit leaves room for it beyond the ~2k-token answer. On the 20-debate offline set (12 Nov), Sonnet 5.5 at `low` was preferred 10 vs 6 over Sonnet 4.6 in blind paired grading, at $0.80-0.89 vs $1.46 for the 20 reels and about 22 s a reel. Its hooks were stronger, its clip endings and responses slightly weaker. `medium` graded slightly worse than 4.6, and `high` cost 42% more without grading better.

- A temperature is sent only to a model that takes one (`claude_json.takes_temperature`). Sonnet 5.5 rejects a temperature, so `moments_pick_claude_temperature` applies only when the pick runs on an older model.
- The effort goes in `output_config.effort`, beside the schema's `format`. An empty effort sends none.
- A thinking model's answer can start with thinking blocks. The answer is read from its text blocks, and only that text goes back with a send-back.
- An answer Claude declines (stop reason `refusal`) is logged with its category. It ends the conversation: the answer already held is kept, and otherwise the run fails as final.
- To restore the earlier picker without a code change, set `MOMENTS_PICK_CLAUDE_MODEL=claude-sonnet-4-6`, `MOMENTS_PICK_CLAUDE_EFFORT=` (empty) and `MOMENTS_PICK_CLAUDE_MAX_TOKENS=16000`. Its requests are then exactly as before, temperature 0.2 included. A single run can do the same through `llm_overrides`.

**Budgets.** Both count against the `claude` spend-guard key that the news Claude paths use, so one hourly budget covers every Claude call. They draw on the `claude_global` per-minute rate limit at one unit per call the run can make (2 for punctuate, 7 for pick: 3 answers, 2 judge answers and 2 link answers), as the news tasks reserve. Punctuate runs one at a time and pick four at a time, so a backfill can't starve news. Each answer's model and token counts are logged (`… answer 1 (claude-sonnet-5-5) used input_tokens …, output_tokens …`); output tokens include thinking. The spend guard and the rate limit count calls, not tokens, so the switch to Sonnet 5.5 changes neither.

**Asking again.** Each task checks the answer. Problems go back to Claude in the same conversation: punctuate gets 2 answers, pick gets 3. The first answer with no problems wins. An answer with only soft problems is kept if the later ones get worse. The system prompt and the transcript are prompt-cached, so a send-back rereads them from cache. Soft problems go back only once an answer has no hard ones, so the model isn't pulled in two directions at once.

**JSON only.** Pick and its judge answer under a JSON schema (Anthropic structured outputs, `output_config.format`), so the reply is the JSON and nothing else. On 11 Nov, before this, 8 of 20 pick runs were cut off at the 16,000-token limit: the model wrote its reasoning out before the JSON, which itself runs about 2k tokens. Punctuate is asked as before, without a schema.

**Cut off at the token limit.** The answer's last 500 characters are logged. Pick and its judge send a cut-off answer back once, told it was cut off and to answer shorter with the JSON alone; a second cut-off ends the run. Punctuate never sends one back: every sentence end is required, so a second answer would be as long. Either way, an acceptable answer seen before the cut-off is kept; a pick's last resort (see below) is kept only once the retry is used up.

**Failures.** A failed request (network, rate limit, overload) is retryable, and Hatchet runs the task again. These are final (`FinalDecisionError`, a `NonRetryableException`), because running again cannot help:
- no acceptable answer
- an answer cut off at the token limit (for pick, a second one)
- a missing key
- a tripped spend guard
- a request Claude rejects: any 4xx except 408, 409 and 429 (a schema structured outputs can't compile, a model without them, a prompt too long). The judge still keeps a reel when its request is rejected (the one with the fewest problems, below).

Once pick holds an answer, acceptable or a last resort, a later spend-guard trip or failed request (any of the above) returns that answer instead of failing the run, since a rerun would throw it away. Punctuate and the judge never hold an answer they then ask again about, so this changes nothing for them.

The caller then falls back. Punctuate's execution timeout covers every request at its 300 s ceiling; pick's is set by its deadline (below). The client doesn't retry on its own.

**Pick's deadline (GEO-3697).** geo-chat waits 17 minutes for a pick (`PICK_MOMENTS_POLL_TIMEOUT` in `crates/media/src/extraction.rs`), counted from when its enqueue returns, so queue time and Hatchet retries count too. After that it falls back to its own picker and drops the result. Seven calls at 300 s each (3 answers, 2 judge answers, 2 link answers) would allow a run 36 minutes, so a slow but healthy pick was thrown away. Now a pick returns within 15 minutes of being enqueued (`PICK_DEADLINE_S`). That leaves 2 minutes for the result to reach geo-chat (it polls every 2 s, each poll up to 30 s) and for clock skew between hosts. The facade stamps each run with when it enqueued it (`enqueued_at` in the run's Hatchet metadata), and the clock starts there. A run without the stamp starts its clock when its attempt starts. Near the deadline, the run takes what it has:
- No answer is asked for with less than 60 s left (`MIN_REQUEST_S`), whether a send-back or a cut-off retry. The best answer held is kept, ranked as above.
- The judge is skipped. The reel with the fewest problems is kept, the first among equals. A judge that starts in time but is cut off at the deadline keeps the same reel.
- Each request's timeout is capped at the time left, so no request runs past the deadline.
- The claim-link call (below) is skipped, and every clip is linked by words in common instead (`closest_claims`). A link call that starts in time is capped the same way, and one cut off at the deadline falls back the same way.
- With no usable reel by then, or a request that fails with too little time left, the run fails at once (`FinalDecisionError`, not retried).

Every such stop logs a warning with "shortened by the deadline" and what was skipped, so shortened picks can be counted. The engine's execution timeout is the deadline plus a minute, as a backstop. A Hatchet retry reads the same stamp, so a retry that starts too late fails at once. Neither can carry a run past geo-chat's wait. Normal-speed picks are unchanged: they finish well before the deadline. Later calls in a pick check the same deadline (`deadline.Deadline`, passed to `ask_json`).

geo-chat's story stage has its own 20-minute cap (`STORY_STAGE_TIMEOUT` in `worker.rs`), which also covers punctuation and up to three picks. The first pick can hit that cap too, when punctuation took over 5 minutes (it may poll for 10) and the pick uses its 15; so can a second pick after a slow first one. Raising it is a change on the geo-chat side.

## transcript.punctuate

Whisper sometimes drops punctuation partway through a debate. Once one window comes back as a run-on, the next window is prompted with it and carries on the same way. This task restores sentence ends.

**Input:** the motion and the turns, each with its speaker and its words as transcribed. Each turn may also carry when each word starts and ends (`starts_ms`, `ends_ms`, one per word); they're used only to judge pauses.

```json
{"media_type": "debate", "title": "Voter ID should be required nationwide",
 "turns": [{"speaker": "Nico", "words": ["it", "is", "a", "federal", "problem", "so"],
            "starts_ms": [0, 180, 300, 360, 820, 1500], "ends_ms": [150, 280, 340, 780, 1300, 1700]}]}
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

**Ends inside a clause (GEO-3694).** The model sometimes ends a sentence mid-clause ("…is a. National security risk…"), and moment clips and subtitles then start on the second half. Two guards run inside each turn, never across turns:
- An end on a word no sentence ends on (an article, a conjunction such as "and" or "because", a possessive) is dropped when the next word follows within 500 ms, so the halves join. After a longer pause the speaker trailed off, so the end stays. Without word times the pause is unknown, so the end goes back to the model as a soft problem instead.
- A sentence that opens like the rest of the one before ("That the…", "Where…" or "Which…" when it isn't a question, "To the…"), within 1 s of it, goes back as a soft problem: join them if it continues, keep it if not. Whichever the model answers is kept, so a wrong hint costs one more answer and nothing else.

The output shape is unchanged.

## debates.pick_moments

**Input:**
- the motion
- the two speakers, with their sides
- the transcript as sentences: speaker, turn, start and end in ms, text
- the claims, with the caller's index and the speaker
- the clip length bounds, which default to 6–22 s hard and 14–20 s target (10–18 s until GEO-3741; geo-chat sends the hard bounds, 6 and 22 s, and leaves the target to the default)
- `min_reel_ms`: the shortest a reel may play, 50 s by default (45 s until GEO-3741; see "Short reels" below)
- `link_claims`: `"after"` (the default) or `"in_pick"`; see below

**How the reel is chosen:** the moments come first and the claims after. Claude finds each speaker's strongest moments in the transcript, whatever they are (a punchline, a question, a comeback), and builds the reel from them. The first clip is chosen for providing the context (what the debate is about, where its speaker stands), not for strength; the strongest moment often lands best last. Claims were extracted to fill the knowledge graph, so a reel built around them missed the lines that aren't claims.

By default (`link_claims: "after"`, GEO-3692) the picker never sees the claims: its prompt, the transcript it reads and its answer's schema have none, and no claim rule is checked on its reels. Once the reel is picked (and judged), a separate call links each clip to a claim (see "Linking claims after the pick" below). With `link_claims: "in_pick"` the picker works as it did before GEO-3692: it is shown the claims and links each clip in its own answer, after choosing the moments, and its reels are checked against the claim rules. That mode is an escape hatch, to restore the old behaviour for a request without a deploy; geo-chat doesn't send the field, so it gets the default.

Claude offers up to three candidate reels in one answer. Each is checked on its own. When more than one passes, a second call judges them the way a viewer meets them: the motion, who speaks for which side, and each clip's words and length, with no transcript and no claims. The words are the ones the viewer hears, past the filler geo-chat cuts (see below). It picks the reel a newcomer can follow to the end; a first clip that gives the context comes before anything else it weighs, and a reel isn't favored for opening on its strongest line. When any passing reel has no soft problem, only those are judged; otherwise all that passed are, and under each reel the judge is shown the problems found in how its clips play ("Clip 3 opens on "but"…", "Clip 2 may start halfway through a sentence", a clip cut to fit, a reel over its length), so it can prefer the clean one. The split one is a guess, and the judge is told to check it against the clip's words. These are about what the viewer hears, so the judge still sees no transcript, claims or story. If the judge can't answer, the reel with the fewest problems is taken, the first among equals (so the first reel when all are clean).

**Output:** a one-sentence `story`, exactly four `moments` in play order, `candidates` (how many reels the pick was made from), `judge_why`, `reel_ms` and `short_reel` (see "Short reels" below), and `first_clip_problems` (empty unless the reel was kept as a last resort; see below). Claude writes the story with "Speaker 1" and "Speaker 2", and each becomes that speaker's short name: the first word of their name past any title, or the full name when both speakers share it. Each moment has:
- `speaker`
- `start_sentence` and `end_sentence` (inclusive)
- `start_ms` and `end_ms` (from those sentences)
- `claim_index`
- `claim_fit`
- `role` (opening, answer, rebuttal or closing)
- `answers` (the earlier clip it responds to, if any)
- `why`
- `trimmed_to_sentence` and `trimmed_from_sentence` (only on a reel kept as a last resort: where the model's own pick ended or started before the clip was cut to fit; see below)
- `ended_from_sentence` (under `"after"` only: where the model's own pick ended, when the clip ran on to finish a sentence the transcript broke; see below)

The task checks each of these rules before answering:
- exactly four clips, alternating speakers
- each clip is whole sentences from one turn of one speaker
- each clip is within the hard length bounds (6–22 s)
- no two clips overlap
- each clip links to a claim by its own speaker, and a speaker's two clips link to different claims (under `"in_pick"`, on the picker's reels; under `"after"`, on the link call's answer, see below)
- a clip only answers an earlier clip, by the other speaker
- the first clip opens on a sentence that starts a thought: not mid-sentence (in a transcript with capitals, a sentence opening on a lowercase word ran on from the words before it, as in "and not just get completely overwhelmed…"), and not on a connective or filler ("but", "because", "though", "that's", "that", "they", "they're", "yeah", "okay", "like", "agree", "I mean", "for example"…: the soft rule's list below, except "it's", "it" and "yes", which often state a position ("It's time we…", "It is the death of…") and stay soft on the first clip too). This holds for a first clip cut to fit too, since the viewer meets the debate there.

**Openers are judged on what the viewer hears.** geo-chat cuts the filler a clip opens on before it plays (`story_moments::filler_lead`, over `clip_ranges::FILLER`): "um", "uh", "er", "ah", "hmm", "so", "and" and "you know", along with any first sentence that is nothing but filler. It keeps "so" as an adverb ("So far…"), "you know" with its object ("You know nothing…"), and at least three words of the sentence ("So what now?" plays as said). Every opener rule here (the first clip's hard ones, the soft one on later clips, and the trim's choice of start) first skips that same filler, then reads what is left. So "Um, we need rules." and "And we need rules." open a first clip ("We need rules."), and "And so but…" is refused on "but". Whether a sentence starts mid-sentence is read from its first word as written, where the speech recogniser put its capital: "and not just get overwhelmed" is still refused, since the viewer hears a run-on "not just…". The copy of geo-chat's lists is `FILLER` and its neighbours in `debates_pick_moments_core.py`, and a test pins their contents so a change on either side is noticed. geo-chat keeps all of the filler when cutting it would leave the clip under its minimum length (6 s), and so does this reading: a skipped filler sentence is timed by its own times, and filler inside a sentence as its share of the sentence's words, since sentences carry no word times.

**Split sentences.** The transcript sometimes breaks one sentence in two and capitalises the second half: Whisper put a full stop mid-sentence ("…in the first place. Where corporations argued…"). The capital hides that a clip starting there starts mid-sentence, so the task guesses at such halves (`split_start`), only within a turn (a turn's first sentence is no half of the other speaker's): the sentence before it ends mid-phrase (on "a", "the", "and", "because"…, but not "What if?" or "Oh my!", which end on a question or exclamation mark, nor "Plan A."; a sentence of nothing but filler, such as "Um.", between the two is looked past), or it opens as a clause of the sentence before and is no question ("Where corporations argued…" but not "Where do we draw the line?" or "Where in the Constitution does it say that?", "That the border security can serve…" but not "That argument…"). The picker's transcript marks each one "second half of a split sentence" and the prompt says to avoid starting there, as a hint it may overrule when the sentence plainly starts a thought. It is a guess: "Where I come from, …" looks the same as a split half. So a clip that starts on a marked sentence is a **soft** problem on every clip, the first included: the answer is sent back, a reel without it is taken first, but it never refuses a reel. When the clip also opens on a dangling word ("That the border security…" opens on "that"), that rule decides, and the first clip is refused for the word, not the mark. Each clip's opening counts as one flaw, the strongest: mid-sentence, then a dangling word, then the mark. So "but we need…" is one problem (mid-sentence), as "But we need…" is (its "but"), and the answer ranking below is not skewed by a double count. In earlier drafts of GEO-3691 the mark was hard on the first clip, and three review rounds each found new ordinary openers it refused ("To the extent…", "To the first point…", "Where in the Constitution…?"); the "To the …" rule and its idiom list were dropped then. The sentences themselves are left as geo-chat sent them: merging them would renumber what geo-chat cuts on. The real fix, punctuation that does not split a sentence, is [GEO-3694](https://linear.app/geobrowser/issue/GEO-3694/debate-transcripts-sometimes-split-one-sentence-into-two-so-clips-and).

The first clip's rules are hard, but with a last resort: a reel that breaks only them is sent back like any other, and if no answer in the three has a reel that keeps every hard rule, the such reels of the answer whose best one has the fewest problems (the later answer among equals) are judged and one is kept, with the rules its first clip breaks in `first_clip_problems` (and a warning in the log). So the rule never fails a run that would have passed without it.

**Clips over the hard limit are picked again, not cut from the start.** Until 12 Nov a clip over the limit was shortened from its start, which cut the setup its story leaned on in 6 of 36 checkable clips of the preview (a story about crop losses played a clip about tomato-picking robots). Now it is a hard problem, sent back with how long the clip runs to each of its sentences ("From sentence 14 it runs 8.1 s to sentence 14, 19.0 s to sentence 15, 27.3 s to sentence 16"), so the next pick ends or starts where it fits. A reel whose only problems are such clips (and its first clip's opening) is a last resort, kept as above, with each such clip cut to fit (`fit_to_limit`): from its end, to the latest sentence that fits and is not the first half of a split sentence, so the setup stays (`trimmed_to_sentence`); only when no such end fits, from its start as before (`trimmed_from_sentence`, preferring a start that keeps the opener rules); and with neither, from its end anyway. Openers are checked on the clip as it plays, cut or not. A clip that can't be cut into the limits is reported at the length the model picked, and is no last resort.

Two rules are soft: a clip whose heard words open badly short of the first clip's hard rules (any clip on a sentence marked as the second half of a split sentence; a later clip mid-sentence or on a dangling word such as "but", "because", "though", "it", "they're", "that", "for example"; or the first clip on "it's", "it" or "yes"), and a reel longer than four clips at the top of the target (80 s by default). A later clip may answer the clip before it: "Yes, …", "Yeah, …", "Agreed, …" are fine when the sentence goes on to a thought of its own (three more words that open cleanly), so "Yeah, I think the truth has a threshold" passes while "Yes." and "Yeah, but…" don't; and a clip that names the one before in its own words ("What is problematic to me with that thinking is…") was never caught by the rule. A reel with no soft problem is taken first. When every reel that passes has one, the answer is sent back with all of them and the other reels' problems, within the three answers, however many passed; of the acceptable answers the one whose best reel has the fewest problems is kept (the later one among equals), so a send-back that comes back worse does not replace a better answer, and its reels go to the judge with their problems shown. Last resorts are chosen among the same way, counting their soft problems as well as the hard ones they were refused for. Until 12 Nov they were sent back only when exactly one reel passed, and about 17 later clips in the preview opened mid-thought.

**Clips run on to finish a broken sentence (GEO-3716).** Under `"after"`, a clip whose next sentence in its turn is the rest of its last one ends after it instead: that sentence opens on a lowercase word in a transcript with capitals, or is marked a split half ("…not looking to have multiple." / "partners at the same time."). It runs on only when that reaches a sentence that really ends, within the hard limit, without touching another clip's sentences or a stretch to avoid; otherwise the clip stays as picked. A clip cut to fit from its end is never run on (its end was chosen to fit); one cut from its start still is, since its end is the model's. This happens before any rule reads the clip, so the openers, the overlap check and the judge all see the finished clip; the reel's soft length rule reads the clips as picked, so a run-on never sends an answer back. The model's end is kept in `ended_from_sentence`.

**What else the judge sees (GEO-3716).** Under `"after"`, the judge reads each clip's length as heard (past the cut filler) and three more problems: a clip that plays under 9 s ("plays only 7.2 s, so it may stop before its point lands"), a clip that still stops mid-sentence, and a reel that plays under `min_reel_ms`. The two clip flaws send no answer back; the short reel does since GEO-3741 (below). In offline tests, telling the picker about them (a length rule in its prompt) cut short clips and short reels, but cost 9–35% more and did not make clip ends better; grading noise between two runs of the same picker was larger than every difference measured.

**Short reels (GEO-3716, GEO-3741).** `reel_ms` is how long the reel plays, its clips back to back from each one's first heard word (past the filler geo-chat cuts). A reel under `min_reel_ms` (50 s) is refused like a clip over the limit, under either `link_claims`: the answer is sent back, within the three answers, with how long the reel and each clip play as heard and which clips are under the target ("The reel plays 39.0 s as heard (clips 10.0, 10.0, 9.0, 10.0 s), under the 50.0 s a reel must play. Clips 0, 1, 2, 3 play under 14.0 s: lengthen them toward 14.0-20.0 s, …"), so the next pick lengthens those clips rather than adding filler that is never heard. A clip cut to fit is never told to lengthen, since the same message sends it back to be picked shorter: with some clips cut, the line gives how long the others must play together ("Clips 0, 2, 3, not over the limit, must together play at least 40.0 s as heard"), and with all cut, or when the others could not reach it within the target's top, only the reel's and the target's lengths. The line goes with every send-back of a short reel, alongside its other problems, so a reel refused for playing short was always told. The picker's numbers line says it up front ("The whole reel: under 80.0 s, and at least 50.0 s."). A short reel that keeps every other rule is a last resort, as a reel whose first clip breaks its rules or one cut to fit is: any reel that passes is kept over it, and last resorts are ranked with a reel long enough first, then by fewest problems (`answer_rank`), so whatever order the answers come in a long last resort is kept over a short one. (In the first draft of GEO-3741 shortness was a soft problem, which held a short reel over a later long last resort and then refused it, while the same last resort alone would have published.) A short candidate is also dropped before the judge when any candidate plays long enough. If the reel kept still plays short (every answer was short, or the send-back was cut short: the deadline, the spend guard, a declined or cut-off answer, or a failed request, which is refused rather than rerun), the run fails as final (`FinalDecisionError`) before the judge and the link call: `MOMENTS_PICK_SHORT_REEL=refuse`, the default since GEO-3741. geo-chat reads that as a failed pick (`task_failed`, with this message in its detail): a moment backfill over a live story reel leaves it live and reports `story_pick_failed` (GEO-3730), and a debate with no story reel gets its older, claim-built reel. `MOMENTS_PICK_SHORT_REEL=flag` returns the reel with `short_reel: true` and a warning in the log instead; geo-chat does not read the flag, so it publishes the short reel (and replaces a live one).

Why 14–20 s and 50 s (GEO-3741): on Sonnet 5.5 at low effort (GEO-3702) the picker aimed at the bottom of the old 10–18 s target. Re-cutting 10 automatic reels on 13 Nov gave clips of 12.3 s on average (17.8 s on Sonnet 4.6), reels of about 49 s (about 71 s), 8 of 40 clips under 9 s and 4 of 10 reels under 45 s. Asking for longer clips in the prompt's wording (GEO-3716) did not help, so the numbers changed instead. The hard limit stays 22 s: it is geo-chat's, sent with every request and checked again there, and raising it is a change on both sides.

An answer goes back to Claude with its problems listed when no candidate passes, or when every one that passes has a soft problem, up to three answers in all. If no answer passes and there is no last resort, the run fails without a retry (`FinalDecisionError`), and the caller falls back to its own picker.

The caller can pass `avoid`: sentence ranges an earlier reel used that the caller then refused, each with a reason. For example, a clip that `claims.score_clips` judged to argue the other side. No clip in the new reel may overlap them.

The order doesn't have to follow the order things were said, but an answer always plays after the clip it answers. The judgment rules (the first clip names the topic, ends after the point lands, argues its own side) are in the prompt, and the judge weighs the first clip's context before anything else. They're for the caller to check again, for example with `claims.score_clips`.

### Linking claims after the pick (GEO-3692)

Under `link_claims: "after"`, one more call links the picked reel's clips to claims. It reads the motion, the speakers, each speaker's claims under their name, and the four clips in play order, each with its speaker and the words the viewer hears (past the filler geo-chat cuts). It answers under a JSON schema (`LINK_SCHEMA`): for each clip, a claim number and a fit from 0 to 1.

The links keep the rules geo-chat checks (`story_moments::story_reel`), so geo-chat, the apps, web and analytics need no change:
- each clip links to a claim of its own speaker
- a speaker's two clips link to different claims
- every clip has a link (`claim_index` is never null)

An answer that breaks one goes back with its problems, up to 2 answers. When two of a speaker's clips name the same claim, the one with the higher fit keeps it. A link still broken after the second answer, and every link when the call gives no answer at all (a failed request, a cut-off answer, the spend guard), is filled without a call (`closest_claims`): the clip gets its speaker's claim, not linked to their other clip, whose words it says most (the share of the claim's words, past common words like "the" and "would", found in the clip). A speaker's two clips take the pair of claims they say most between them. Ties go to the claim the caller listed first. Such a link has `claim_fit` 0, so it can be told apart from a judged one, and the log says which clips were linked this way ("linked clip(s) … by words in common"). Claims carry no times here, so the words are the only thing to match on. Matching on words alone is what mislinked clips before (8 of 19 reels in the 12 Nov review had a clip linked to a claim it doesn't match), so it is a fallback, not the way links are made.

A link never fails the reel. The one case with no reel is a speaker with fewer than two claims: their two clips can't link different claims, so geo-chat would refuse any reel. Under `"after"` the run fails at once (`FinalDecisionError`), before any call. Under `"in_pick"` it fails as it always has, after every answer is refused for linking one claim twice. Either way geo-chat falls back to its own picker, as it did before.

The output's shape is the same in both modes: `claim_index` and `claim_fit` on every moment.

## Evidence

12 Nov 2026 (GEO-3692): the picker was run offline on the same 20 debates (ranks 30-56) with and without the claims in its prompt and rules, the claims linked afterwards for the reels picked without them, and graded blind in pairs by two graders per debate. Twice: on the picker before GEO-3691, and on GEO-3691's. Where the graders agreed, the reels picked without claims were preferred in 10 debates to 5 (2 ties), then 11 to 7: 21 to 12 in all. In the second run they scored better on story (3.02 to 3.30), hook (3.52 to 3.73), clear positions (3.33 to 3.58) and watching to the end (2.83 to 3.00), with substance flat and claim fit a little lower (4.62 to 4.50). Every reel was built both ways in the second run (20/20). In the first run's linking, with no rules at all, no clip was left unlinked, linked to the other speaker's claim, or linked to a claim another clip had (0 of 72 clips), so the link call keeps the old rules.

12 Nov 2026 preview (ranks 30-56, 20 debates, 19 reels; GEO-3691): an adversarial review traced about 17 later clips that opened mid-thought: about 6 on connectives only clip 1 was checked for, 2 lowercase run-ons, 4 sentences the transcript split, 2 starts moved by a trim, and 4 unintroduced references. Trims from the start cut the payoff's setup in 6 of 36 checkable clips. Replayed on those reels' clips, the new opener rules flag a later clip in 13 of the 19 reels (the old list flagged 5).

11 Nov 2026: the first 93 live story reels, graded against their transcripts, averaged 2.5/5. Most often a clearly better moment was skipped (65 reels), a clip opened mid-thought (63), a clip ran over 20 s (59), or a clip leaned on something the viewer never heard (56). Reels over 80 s graded worst. The moment-first prompt, the judge, the 22 s limit and the reel length rule answer those.

10 Nov 2026 prototype: 12 debates whose current reels had three clips. Every answer was four valid alternating clips, and a blind judge preferred the new reel in all 12 (story 7.6 vs 4.4, clean starts and ends 6.0 vs 3.3). The remaining flaws were mid-sentence cuts, which came from cutting on subtitle chunks. That's why this task works in whole punctuated sentences.
