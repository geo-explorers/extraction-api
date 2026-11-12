# Moment reels: transcript.punctuate and debates.pick_moments

These two tasks build a debate's moment reel for geo-chat (GEO-3638). A moment reel is four clips that play back to back, alternating speakers, and tell the debate's story in about a minute.

Previously the reel was stitched together from clips cut around extracted claims. Claim extraction was built to fill the knowledge graph, not to make a reel. On 10 Nov 2026, 192 of the 557 reels on testnet had only three clips:

- Several claims shared one stretch of speech, so one debater ended up with a single usable clip.
- The picker preferred three clips in the order said over four that alternate.
- Clips judged to argue the wrong side were dropped, with nothing to replace them.
- Transcripts lost their punctuation partway through, so no clean sentence ends could be found.

Both tasks run on Claude, using the `moments_claude_*` settings. They write an answer rather than choosing among options, so they can't use the decision model.

**Budgets.** Both count against the `claude` spend-guard key that the news Claude paths use, so one hourly budget covers every Claude call. They draw on the `claude_global` per-minute rate limit at one unit per call the run can make (2 for punctuate, 7 for pick: 3 answers, 2 judge answers and 2 link answers), as the news tasks reserve. Punctuate runs one at a time and pick four at a time, so a backfill can't starve news. Each answer's token counts are logged (`… used input_tokens …, output_tokens …`).

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
- the clip length bounds, which default to 6–22 s hard and 10–18 s target
- `link_claims`: `"after"` (the default) or `"in_pick"`; see below

**How the reel is chosen:** the moments come first and the claims after. Claude finds each speaker's strongest moments in the transcript, whatever they are (a punchline, a question, a comeback), and builds the reel from them. The first clip is chosen for providing the context (what the debate is about, where its speaker stands), not for strength; the strongest moment often lands best last. Claims were extracted to fill the knowledge graph, so a reel built around them missed the lines that aren't claims.

By default (`link_claims: "after"`, GEO-3692) the picker never sees the claims: its prompt, the transcript it reads and its answer's schema have none, and no claim rule is checked on its reels. Once the reel is picked (and judged), a separate call links each clip to a claim (see "Linking claims after the pick" below). With `link_claims: "in_pick"` the picker works as it did before GEO-3692: it is shown the claims and links each clip in its own answer, after choosing the moments, and its reels are checked against the claim rules. That mode is an escape hatch, to restore the old behaviour for a request without a deploy; geo-chat doesn't send the field, so it gets the default.

Claude offers up to three candidate reels in one answer. Each is checked on its own. When more than one passes, a second call judges them the way a viewer meets them: the motion, who speaks for which side, and each clip's words and length, with no transcript and no claims. The words are the ones the viewer hears, past the filler geo-chat cuts (see below). It picks the reel a newcomer can follow to the end; a first clip that gives the context comes before anything else it weighs, then whether the close lands (a last clip ending on a hedge, a concession, "find a balance" or a fragment loses to one that ends on its speaker's firmest line), and a reel isn't favored for opening on its strongest line. Every reel that passed is judged, soft problems and all (GEO-3696; before, reels with a soft problem were dropped when any reel had none, so the judge ran in only 10 of 20 debates and never weighed a stronger reel with one "But…" clip). The reels go to the judge with the fewest problems first, in the picker's order among equals, and under each reel the judge is shown the problems found in how its clips play ("Clip 3 opens on "but"…", "Clip 2 may start halfway through a sentence", "Clip 4 ends on a fragment", a clip cut to fit as a last resort, a reel over its length), so it can prefer the clean one unless the flawed one is clearly stronger. The split one is a guess, and the judge is told to check it against the clip's words. These are about what the viewer hears, so the judge still sees no transcript, claims or story. If the judge can't answer, the first reel (the cleanest) is taken.

**Output:** a one-sentence `story`, exactly four `moments` in play order, `candidates` (how many reels the pick was made from), `judge_why`, `first_clip_problems` (empty unless the reel was kept as a last resort; see below), `candidate_reels` (every reel the pick was made from, in the order the judge saw them, each with its `story`, `moments` and `problems`) and `judge_pick` (the position in `candidate_reels` of the reel returned, or null when there was one and no judge ran). The last two are there so the judge's choice can be graded later; geo-chat's `PickedReel` reads only `story`, `moments` and `model_used` and ignores the rest. A candidate's moments are as checked, before any claim is linked: under `"after"` each has `claim_index` -1 and `claim_fit` 0, and only `moments` is linked. Claude writes the story with "Speaker 1" and "Speaker 2", and each becomes that speaker's short name: the first word of their name past any title, or the full name when both speakers share it. Each moment has:
- `speaker`
- `start_sentence` and `end_sentence` (inclusive)
- `start_ms` and `end_ms` (from those sentences)
- `claim_index`
- `claim_fit`
- `role` (opening, answer, rebuttal or closing)
- `answers` (the earlier clip it responds to, if any)
- `why`
- `payoff_sentence` (the sentence the picker named as the clip's payoff, always inside the clip; null when its answer had none, as under `"in_pick"`)
- `fitted_around_payoff` (true when the clip ran over the hard limit and was fitted around its payoff; see below)
- `trimmed_to_sentence` and `trimmed_from_sentence` (where the model's own pick ended or started before the clip was shortened: fitted around its payoff, or cut to fit on a reel kept as a last resort; see below)

The task checks each of these rules before answering:
- exactly four clips, alternating speakers
- each clip is whole sentences from one turn of one speaker
- each clip is within the hard length bounds (6–22 s)
- no two clips overlap
- each clip links to a claim by its own speaker, and a speaker's two clips link to different claims (under `"in_pick"`, on the picker's reels; under `"after"`, on the link call's answer, see below)
- a clip only answers an earlier clip, by the other speaker
- a clip's `payoff_sentence`, when given, is one of its own sentences
- the first clip opens on a sentence that starts a thought: not mid-sentence (in a transcript with capitals, a sentence opening on a lowercase word ran on from the words before it, as in "and not just get completely overwhelmed…"), and not on a connective or filler ("but", "because", "though", "that's", "that", "they", "they're", "yeah", "okay", "like", "agree", "I mean", "for example"…: the soft rule's list below, except "it's", "it" and "yes", which often state a position ("It's time we…", "It is the death of…") and stay soft on the first clip too). This holds for a first clip cut to fit too, since the viewer meets the debate there.

**Openers are judged on what the viewer hears.** geo-chat cuts the filler a clip opens on before it plays (`story_moments::filler_lead`, over `clip_ranges::FILLER`): "um", "uh", "er", "ah", "hmm", "so", "and" and "you know", along with any first sentence that is nothing but filler. It keeps "so" as an adverb ("So far…"), "you know" with its object ("You know nothing…"), and at least three words of the sentence ("So what now?" plays as said). Every opener rule here (the first clip's hard ones, the soft one on later clips, and the trim's choice of start) first skips that same filler, then reads what is left. So "Um, we need rules." and "And we need rules." open a first clip ("We need rules."), and "And so but…" is refused on "but". Whether a sentence starts mid-sentence is read from its first word as written, where the speech recogniser put its capital: "and not just get overwhelmed" is still refused, since the viewer hears a run-on "not just…". The copy of geo-chat's lists is `FILLER` and its neighbours in `debates_pick_moments_core.py`, and a test pins their contents so a change on either side is noticed. geo-chat keeps all of the filler when cutting it would leave the clip under its minimum length (6 s), and so does this reading: a skipped filler sentence is timed by its own times, and filler inside a sentence as its share of the sentence's words, since sentences carry no word times.

**Split sentences.** The transcript sometimes breaks one sentence in two and capitalises the second half: Whisper put a full stop mid-sentence ("…in the first place. Where corporations argued…"). The capital hides that a clip starting there starts mid-sentence, so the task guesses at such halves (`split_start`), only within a turn (a turn's first sentence is no half of the other speaker's): the sentence before it ends mid-phrase (on "a", "the", "and", "because"…, but not "What if?" or "Oh my!", which end on a question or exclamation mark, nor "Plan A."; a sentence of nothing but filler, such as "Um.", between the two is looked past), or it opens as a clause of the sentence before and is no question ("Where corporations argued…" but not "Where do we draw the line?" or "Where in the Constitution does it say that?", "That the border security can serve…" but not "That argument…"). The picker's transcript marks each one "second half of a split sentence" and the prompt says to avoid starting there, as a hint it may overrule when the sentence plainly starts a thought. It is a guess: "Where I come from, …" looks the same as a split half. So a clip that starts on a marked sentence is a **soft** problem on every clip, the first included: the answer is sent back, a reel without it is taken first, but it never refuses a reel. When the clip also opens on a dangling word ("That the border security…" opens on "that"), that rule decides, and the first clip is refused for the word, not the mark. Each clip's opening counts as one flaw, the strongest: mid-sentence, then a dangling word, then the mark. So "but we need…" is one problem (mid-sentence), as "But we need…" is (its "but"), and the answer ranking below is not skewed by a double count. In earlier drafts of GEO-3691 the mark was hard on the first clip, and three review rounds each found new ordinary openers it refused ("To the extent…", "To the first point…", "Where in the Constitution…?"); the "To the …" rule and its idiom list were dropped then. The sentences themselves are left as geo-chat sent them: merging them would renumber what geo-chat cuts on. The real fix, punctuation that does not split a sentence, is [GEO-3694](https://linear.app/geobrowser/issue/GEO-3694/debate-transcripts-sometimes-split-one-sentence-into-two-so-clips-and).

The first clip's rules are hard, but with a last resort: a reel that breaks only them is sent back like any other, and if no answer in the three has a reel that keeps every hard rule, the such reels of the answer whose best one has the fewest problems (the later answer among equals) are judged and one is kept, with the rules its first clip breaks in `first_clip_problems` (and a warning in the log). So the rule never fails a run that would have passed without it.

**Clips over the hard limit are fitted around their payoff (GEO-3696).** The story picker (`"after"`) names each clip's `payoff_sentence`: the line that makes it worth showing. A clip over the limit is shortened in code around it (`fit_around_payoff`): it ends at the payoff, or on the first sentence after it that ends cleanly (not mid-phrase and not before its second half, the ending rules below), and starts on the earliest sentence that opens a thought (no opener problem at all, hard or soft, as the clip plays) and still fits within 6–22 s. So it keeps the payoff and as much of its setup as fits. The model's own start and end always count as clean: a fit never adds a flaw the pick did not have. A fitted clip is no problem at all, so the answer isn't sent back for it and its reel isn't a last resort; the judge isn't told it was shortened, and reads its words as they play. It stays inside the clip the model picked, so every rule geo-chat checks again (one turn of one speaker, 6–22 s, alternation, no overlap) still holds. Before, the whole answer went back and the next answer usually swapped the moment for a weaker one rather than shortening the clip: in the #76 offline run 13 of the 30 moments graders named as better were in the picker's first answer, in a clip of 25–42 s that was sent back, 4 of them its planned close.

Only a clip that can't be fitted around its payoff (the payoff sentence alone runs over the limit, or no clean start and end keep it within the limits), or one from an answer with no payoff (the `"in_pick"` picker, whose prompt and schema are unchanged), goes on as below. **Such clips are picked again, not cut from the start.** Until 12 Nov a clip over the limit was shortened from its start, which cut the setup its story leaned on in 6 of 36 checkable clips of the preview (a story about crop losses played a clip about tomato-picking robots). Now it is a hard problem, sent back with how long the clip runs to each of its sentences ("From sentence 14 it runs 8.1 s to sentence 14, 19.0 s to sentence 15, 27.3 s to sentence 16"), and that no clean fit keeps its payoff, so the next pick ends or starts where it fits. A reel whose only problems are such clips (and its first clip's opening) is a last resort, kept as above, with each such clip cut to fit (`fit_to_limit`): from its end, to the latest sentence that fits and is not the first half of a split sentence, so the setup stays (`trimmed_to_sentence`); only when no such end fits, from its start as before (`trimmed_from_sentence`, preferring a start that keeps the opener rules); and with neither, from its end anyway. Openers are checked on the clip as it plays, cut or not. A clip that can't be cut into the limits is reported at the length the model picked, and is no last resort.

**How a clip ends (GEO-3696).** The story picker's prompt asks for clip 4 to end on its speaker's firmest statement of their position or their most memorable line, with its payoff, and never on a hedge, a concession, a call to "find a balance" or a fragment; "Shorter is better" no longer outranks the payoff. The judge weighs whether the close lands right after the first clip. Two structural signals are checked on every clip, as soft problems (sent back, shown to the judge, never refusing a reel):
- **mid-thought**: its last sentence stops mid-phrase (on "a", "the", "and", "because"… as the split-sentence rule reads it, not "What if?" or "Plan A."), or the next sentence of its turn reads as its second half;
- **fragment**: its last sentence, after others, is under five words and opens on a dangling word or filler ("But like beyond that.", "So yeah.", "And stuff."). A sentence inside the transcript's last one counts: the punctuate step marks the ends it adds, but the words keep Whisper's own ("…they could work on. But like beyond that."). Short alone is no fragment: "Conviction means conviction." and "It's that simple." pass.

A hedge or a concession has no reliable shape in the words ("my honest stance… is how people actually use it", "So we need to find a good balance there", "It's just hard to get to that point"), and a word list for them would refuse firm closes too ("That said, we must act."), so those are left to the prompt and the judge. Replayed on the 60 reels graded on 12 Nov (master, #76, and #76 picking without claims; 240 clips), the two checks flag 4 closing clips and no other clip: the "But like beyond that." close of one debate in all three runs, which every grader tagged weak, and one clip ending before a split sentence's second half (tagged weak by 1 of 2 graders). They catch few of the 16 weak closes graders tagged; the prompt and the judge are meant to catch the rest.

Two rules are soft besides: a clip whose heard words open badly short of the first clip's hard rules (any clip on a sentence marked as the second half of a split sentence; a later clip mid-sentence or on a dangling word such as "but", "because", "though", "it", "they're", "that", "for example"; or the first clip on "it's", "it" or "yes"), and a reel longer than four clips at the top of the target (72 s by default). A later clip may answer the clip before it: "Yes, …", "Yeah, …", "Agreed, …" are fine when the sentence goes on to a thought of its own (three more words that open cleanly), so "Yeah, I think the truth has a threshold" passes while "Yes." and "Yeah, but…" don't; and a clip that names the one before in its own words ("What is problematic to me with that thinking is…") was never caught by the rule. A reel with no soft problem is taken first. When every reel that passes has one, the answer is sent back with all of them and the other reels' problems, within the three answers, however many passed; of the acceptable answers the one whose best reel has the fewest problems is kept (the later one among equals), so a send-back that comes back worse does not replace a better answer, and its reels go to the judge with their problems shown. Last resorts are chosen among the same way, counting their soft problems as well as the hard ones they were refused for. Until 12 Nov they were sent back only when exactly one reel passed, and about 17 later clips in the preview opened mid-thought.

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
