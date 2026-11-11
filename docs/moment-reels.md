# Moment reels: transcript.punctuate and debates.pick_moments

These two tasks build a debate's moment reel for geo-chat (GEO-3638). A moment reel is four clips that play back to back, alternating speakers, and tell the debate's story in about a minute.

Previously the reel was stitched together from clips cut around extracted claims. Claim extraction was built to fill the knowledge graph, not to make a reel. On 10 Nov 2026, 192 of the 557 reels on testnet had only three clips:

- Several claims shared one stretch of speech, so one debater ended up with a single usable clip.
- The picker preferred three clips in the order said over four that alternate.
- Clips judged to argue the wrong side were dropped, with nothing to replace them.
- Transcripts lost their punctuation partway through, so no clean sentence ends could be found.

Both tasks run on Claude, using the `moments_claude_*` settings. They write an answer rather than choosing among options, so they can't use the decision model.

**Budgets.** Both count against the `claude` spend-guard key that the news Claude paths use, so one hourly budget covers every Claude call. They draw on the `claude_global` per-minute rate limit at one unit per call the run can make (2 for punctuate, 3 for pick), as the news tasks reserve. They run one at a time (`concurrency=1`), so a backfill can't starve news.

**Asking again.** Each task checks the answer. Problems go back to Claude in the same conversation: punctuate gets 2 answers, pick gets 3. The first answer with no problems wins. An answer with only soft problems is kept if the later ones get worse. The system prompt and the transcript are prompt-cached, so a send-back rereads them from cache. Soft problems go back only once an answer has no hard ones, so the model isn't pulled in two directions at once.

**Failures.** A failed request (network, rate limit, overload) is retryable, and Hatchet runs the task again. These are final (`FinalDecisionError`, a `NonRetryableException`), because running again cannot help:
- no acceptable answer
- an answer cut off at the token limit
- a missing key
- a tripped spend guard

The caller then falls back. Each task's execution timeout covers every request at its 180 s ceiling. The client doesn't retry on its own.

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

Claude offers up to three candidate reels in one answer. Each is checked on its own. When more than one passes, a second call judges them the way a viewer meets them: the motion, who speaks for which side, and each clip's words and length, with no transcript and no claims. It picks the reel a newcomer can follow to the end; a first clip that gives the context comes before anything else it weighs, and a reel isn't favored for opening on its strongest line. When any passing reel has no soft problem, only those are judged; otherwise all that passed are. If the judge can't answer, the first reel is taken.

**Output:** a one-sentence `story`, exactly four `moments` in play order, `candidates` (how many reels the pick was made from) and `judge_why`. Claude writes the story with "Speaker 1" and "Speaker 2", and each becomes that speaker's short name: the first word of their name past any title, or the full name when both speakers share it. Each moment has:
- `speaker`
- `start_sentence` and `end_sentence` (inclusive)
- `start_ms` and `end_ms` (from those sentences)
- `claim_index`
- `claim_fit`
- `role` (opening, answer, rebuttal or closing)
- `answers` (the earlier clip it responds to, if any)
- `why`

The task checks each of these rules before answering:
- exactly four clips, alternating speakers
- each clip is whole sentences from one turn of one speaker
- each clip is within the hard length bounds (6–22 s)
- no two clips overlap
- each clip links to a claim by its own speaker, and a speaker's two clips link to different claims
- a clip only answers an earlier clip, by the other speaker

A clip over the hard limit that spans several sentences is shortened from its start until it fits, and not onto a sentence that opens on a dangling word when a cleaner start fits. The payoff at its end stays. The reported `start_sentence` and `start_ms` are the trimmed ones, and `trimmed_from_sentence` says where the model's own pick started. A trim is final: it isn't sent back (a model content with it would only send the same pick again), and a trimmed clip's opener isn't held against it, since the model didn't choose that start. A clip that can't be trimmed into the limits is reported at the length the model picked.

Two rules are soft: a clip that opens on a dangling word ("and", "so", "but"...), and a reel longer than four clips at the top of the target (72 s by default). When only one reel passes and it breaks a soft rule, the answer is sent back with that and the other reels' problems, within the three answers, and the last acceptable reel is kept. When two or more pass, nothing is sent back: the judge chooses.

An answer goes back to Claude with its problems listed only when no candidate passes, up to three answers in all. If no answer passes, the run fails without a retry (`FinalDecisionError`), and the caller falls back to its own picker.

The caller can pass `avoid`: sentence ranges an earlier reel used that the caller then refused, each with a reason. For example, a clip that `claims.score_clips` judged to argue the other side. No clip in the new reel may overlap them.

The order doesn't have to follow the order things were said, but an answer always plays after the clip it answers. The judgment rules (starts where a thought starts, ends after the point lands, argues its own side) are in the prompt. They're for the caller to check again, for example with `claims.score_clips`.

## Evidence

11 Nov 2026: the first 93 live story reels, graded against their transcripts, averaged 2.5/5. Most often a clearly better moment was skipped (65 reels), a clip opened mid-thought (63), a clip ran over 20 s (59), or a clip leaned on something the viewer never heard (56). Reels over 80 s graded worst. The moment-first prompt, the judge, the 22 s limit and the reel length rule answer those.

10 Nov 2026 prototype: 12 debates whose current reels had three clips. Every answer was four valid alternating clips, and a blind judge preferred the new reel in all 12 (story 7.6 vs 4.4, clean starts and ends 6.0 vs 3.3). The remaining flaws were mid-sentence cuts, which came from cutting on subtitle chunks. That's why this task works in whole punctuated sentences.
