# Testing a prompt without a deploy

Every prompt the extraction service sends to a model can be replaced **for one run** from the
Hatchet dashboard. You paste your version of the prompt into the run's input, trigger the run,
and read the result. Nothing changes for anyone else: the next run without your text uses the
prompt from the code again. The same input can also switch the model, the temperature or the
thinking level for that one run.

This guide has three parts:

1. **Part 1** — a walkthrough for the dashboard. No coding.
2. **Part 2** — reference: every prompt key, and copy-paste inputs for every task.
3. **Part 3** — instructions written for a coding agent, so you can hand the testing loop to one.

> Every run costs real model calls. Keep test inputs short, and reuse the same input when you
> compare two versions of a prompt.

## Part 1 — From the Hatchet dashboard

### Before you start

- Dashboard access to the Hatchet environment you want to test in.
- The prompt you want to change. Each prompt has a **key** (like `claims_extract.factuality`).
  The table in Part 2 lists every key, what runs use it, and the file on GitHub where its current
  text lives. Open that file on GitHub and copy the text as your starting point. A coding agent can
  also fetch it for you (Part 3).

### The walkthrough

1. **Open the task.** In the dashboard, go to *Workflows* and open the task you want to run, for
   example `claims.extract`. Click **Trigger**.
2. **Paste an input.** Copy the matching template from *Payload templates* in Part 2 into the
   *Input* box, replacing whatever is there. Fill in the real data fields (the document, headline,
   sources, and so on).
   *Shortcut:* open any earlier run of that task, copy its *Input*, and add the
   `prompt_overrides` block to it. That way you test your prompt on exactly the material a real run
   saw.
3. **Put in your prompt.** The template lists every prompt key that task reads, each with an empty
   `""`. Paste your text into the one you are changing and **delete the keys you are not using**
   (an empty prompt is rejected).
   - **Line breaks.** The input box is JSON, and a JSON string cannot contain a real line break.
     Each line break must be written as the two characters `\n`, each `"` inside your text as
     `\"`. Do not do this by hand: ask your coding agent to "convert this text to a JSON string",
     or use the one-line command in Part 3. Do not paste company prompts into random websites to
     convert them.
   - **Curly braces.** In most prompts, `{like_this}` is a placeholder the code fills in (the table
     in Part 2 lists which ones each prompt may use). A brace you want to appear literally, for
     example in a JSON example inside the prompt, must be doubled: `{{` and `}}`.
4. **Optionally change the model.** The template also has an `llm_overrides` block filled with the
   current defaults. Change a value to test another model or temperature; delete the block to leave
   everything as is. For a fair before/after comparison set the temperature to `0`.
5. **Trigger** and open the run.
6. **Read the outcome.**
   - *Output* shows the result exactly as a normal run would return it.
   - *Logs* contain one line per step that starts with `overrides[...]`. `applied=[...]` lists the
     keys that took effect; `unused=[...]` lists keys the step did not read. Your key must be under
     `applied` somewhere, otherwise the run never used your text (see below).
7. **Compare.** Trigger the same input once more without `prompt_overrides`. Two runs, same input,
   one difference: that is the comparison that tells you what your change did.

### When something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| The run fails right away with `unknown prompt key` | A typo in the key | Copy the key from the table in Part 2 |
| `uses slots [...] that the renderer does not fill` | Your text contains a `{word}` that is not a placeholder for that prompt | Double the braces (`{{word}}`) or use only the slots listed for that key |
| `is not a valid format template` | An unpaired `{` or `}` somewhere in your text | Find and double it |
| `must be a number in [0, 1.0]` | Temperature out of range | Use a value between 0 and 1 |
| `unknown llm setting` | A typo in an `llm_overrides` name | Copy the name from the template |
| The run fails later with a message from Google or Anthropic mentioning a model | The model name in `llm_overrides` is not available | Remove that line or pick a model you know works |
| Your key shows under `unused` in the log | That step did not read the prompt | Some sections only apply with an option on, for example `claims_extract.factuality` needs `"classify_factuality": true` in the input |
| The output looks identical to the baseline | Either the change made no difference or it was not applied | Check `applied` in the log first |

### Making a change permanent

An override only lives in that run. Once you are happy with a prompt, hand the final text to an
engineer or a coding agent: it goes into the file listed in the table in Part 2, and a pull request
ships it. The test suite checks that every prompt in those files is registered, so nothing gets
lost on the way.

## Part 2 — Reference

### Prompt keys

"Slots" are the placeholders the code fills in. A prompt may use any subset of its slots and no
others. Prompts marked plain text have no slots and may contain any braces.

| Prompt key | Used by | Slots you may use | Where the current text lives |
|---|---|---|---|
| `podcast_extract.topics` | `podcast.extract_claims` | `{title}`, `{description}`, `{transcript}` | [`src/config/prompts/topics_of_discussion_extraction_prompt.py`](../src/config/prompts/topics_of_discussion_extraction_prompt.py) |
| `podcast_extract.claims` | `podcast.extract_claims` | `{topics_of_discussion}`, `{transcript}` | [`src/config/prompts/claim_extraction_prompt.py`](../src/config/prompts/claim_extraction_prompt.py) |
| `podcast_extract.takeaways` | `podcast.extract_claims` | `{topics_with_claims}` | [`src/config/prompts/key_takeaways_prompt.py`](../src/config/prompts/key_takeaways_prompt.py) |
| `keyword_extraction` | `keyword.extract` | `{episode}`, `{topics_list}`, `{min_keywords}`, `{max_keywords}`, `{min_topics}`, `{max_topics}` | [`src/config/prompts/keyword_extraction_prompt.py`](../src/config/prompts/keyword_extraction_prompt.py) |
| `media_keyword_extraction` | `POST /extract/media/keywords` | `{media_type}`, `{media}`, `{topics_list}`, `{min_keywords}`, `{max_keywords}`, `{min_topics}`, `{max_topics}` | [`src/config/prompts/media_keyword_extraction_prompt.py`](../src/config/prompts/media_keyword_extraction_prompt.py) |
| `claim_keyword_extraction` | `POST /extract/claim-keywords` | `{title}`, `{description}`, `{claims}`, `{topics_list}`, `{min_keywords}`, `{max_keywords}`, `{min_topics}`, `{max_topics}` | [`src/config/prompts/claim_keyword_extraction_prompt.py`](../src/config/prompts/claim_keyword_extraction_prompt.py) |
| `guest_extraction` | `guest.extract` | `{title}`, `{description}`, `{truncated_transcript}` | [`src/config/prompts/guest_extraction_prompt.py`](../src/config/prompts/guest_extraction_prompt.py) |
| `host_extraction` | `host.extract` | `{possible_hosts}`, `{title}`, `{description}`, `{truncated_transcript}` | [`src/config/prompts/host_extraction_prompt.py`](../src/config/prompts/host_extraction_prompt.py) |
| `news_claim_extract` | `news.extract_claims`, `news.extract_claims_claude`, `news.extract_topics_and_claims` | `{calendar}`, `{headline}`, `{sources}`, `{topics}` | [`src/config/prompts/news_claim_extract_prompt.py`](../src/config/prompts/news_claim_extract_prompt.py) |
| `news_claim_extract.claude_system` | `news.extract_claims_claude` | none (plain text; braces allowed) | [`src/config/prompts/news_claim_extract_prompt.py`](../src/config/prompts/news_claim_extract_prompt.py) |
| `news_debate_claim` | `news.extract_claims`, `news.extract_claims_claude`, `news.extract_debate_claims` | `{headline}`, `{central_claims}`, `{sources}` | [`src/config/prompts/news_debate_claim_prompt.py`](../src/config/prompts/news_debate_claim_prompt.py) |
| `news_debate_claim.claude_system` | `news.extract_claims_claude` | none (plain text; braces allowed) | [`src/config/prompts/news_debate_claim_prompt.py`](../src/config/prompts/news_debate_claim_prompt.py) |
| `news_debate_completion` | `news.extract_claims`, `news.extract_claims_claude`, `news.extract_debate_claims` | `{survivor_count}`, `{minimum_needed}`, `{maximum_new}`, `{headline}`, `{central_claims}`, `{surviving_candidates}`, `{attempted_axes}`, `{sources}` | [`src/config/prompts/news_debate_completion_prompt.py`](../src/config/prompts/news_debate_completion_prompt.py) |
| `news_debate_semantic_review` | `news.extract_claims`, `news.extract_claims_claude`, `news.extract_debate_claims` | `{headline}`, `{claims}`, `{candidates}`, `{prior_axes}`, `{sources}` | [`src/config/prompts/news_debate_semantic_review_prompt.py`](../src/config/prompts/news_debate_semantic_review_prompt.py) |
| `news_debate_semantic_review.claude_system` | `news.extract_claims_claude` | none (plain text; braces allowed) | [`src/config/prompts/news_debate_semantic_review_prompt.py`](../src/config/prompts/news_debate_semantic_review_prompt.py) |
| `news_topics_extract` | `news.extract_topics_and_claims` | `{headline}`, `{content}` | [`src/config/prompts/news_topics_extract_prompt.py`](../src/config/prompts/news_topics_extract_prompt.py) |
| `news_topics_extract.regenerate` | `news.extract_topics_and_claims` | `{feedback}`, `{headline}`, `{content}`, `{previous}` | [`src/config/prompts/news_topics_extract_prompt.py`](../src/config/prompts/news_topics_extract_prompt.py) |
| `news_topics_extract.system` | `news.extract_topics_and_claims` | none (plain text; braces allowed) | [`src/config/prompts/news_topics_extract_prompt.py`](../src/config/prompts/news_topics_extract_prompt.py) |
| `news_topics_entities.system` | `news.extract_topics_and_entities` | none (plain text; braces allowed) | [`src/config/prompts/news_topics_entities_prompt.py`](../src/config/prompts/news_topics_entities_prompt.py) |
| `news_topics_entities.curated_prefix` | `news.extract_topics_and_entities` | none (plain text; braces allowed) | [`src/config/prompts/news_topics_entities_prompt.py`](../src/config/prompts/news_topics_entities_prompt.py) |
| `news_topics_entities.topic_rule_curated` | `news.extract_topics_and_entities` | none (plain text; braces allowed) | [`src/config/prompts/news_topics_entities_prompt.py`](../src/config/prompts/news_topics_entities_prompt.py) |
| `news_topics_entities.topic_rule_free` | `news.extract_topics_and_entities` | none (plain text; braces allowed) | [`src/config/prompts/news_topics_entities_prompt.py`](../src/config/prompts/news_topics_entities_prompt.py) |
| `news_topics_entities.user_header` | `news.extract_topics_and_entities` | `{headline}`, `{summary}` | [`src/config/prompts/news_topics_entities_prompt.py`](../src/config/prompts/news_topics_entities_prompt.py) |
| `news_topics_entities.user_json_example` | `news.extract_topics_and_entities` | none (plain text; braces allowed) | [`src/config/prompts/news_topics_entities_prompt.py`](../src/config/prompts/news_topics_entities_prompt.py) |
| `news_topics_entities.user_rules` | `news.extract_topics_and_entities` | `{topic_rule}`, `{entity_types}` | [`src/config/prompts/news_topics_entities_prompt.py`](../src/config/prompts/news_topics_entities_prompt.py) |
| `news_collection_review.group` | `news.review_collections` | `{headline}`, `{claims}`, `{feedback}`, `{min_block}`, `{max_block}` | [`src/config/prompts/news_collection_review_prompt.py`](../src/config/prompts/news_collection_review_prompt.py) |
| `news_collection_review.rescue` | `news.review_collections` | `{headline}`, `{claims}`, `{lone}`, `{sources}`, `{calendar}`, `{min_words}` | [`src/config/prompts/news_collection_review_prompt.py`](../src/config/prompts/news_collection_review_prompt.py) |
| `news_collection_review.check` | `news.review_collections` | `{headline}`, `{blocks}`, `{lone}` | [`src/config/prompts/news_collection_review_prompt.py`](../src/config/prompts/news_collection_review_prompt.py) |
| `news_collection_review.order` | `news.review_collections` | `{headline}`, `{blocks}`, `{order_rule}` | [`src/config/prompts/news_collection_review_prompt.py`](../src/config/prompts/news_collection_review_prompt.py) |
| `news_collection_review.order_rule` | `news.review_collections` | none (plain text; braces allowed) | [`src/config/prompts/news_collection_review_prompt.py`](../src/config/prompts/news_collection_review_prompt.py) |
| `news_collection_review.review` | `news.review_collections` | `{headline}`, `{claims}`, `{collections}`, `{max_words}`, `{max_growth}` | [`src/config/prompts/news_collection_review_prompt.py`](../src/config/prompts/news_collection_review_prompt.py) |
| `news_collection_review.source_check` | `news.review_collections` | `{sentences}`, `{sources}` | [`src/config/prompts/news_collection_review_prompt.py`](../src/config/prompts/news_collection_review_prompt.py) |
| `claims_link_entities` | `claims.link_entities` | `{context_block}`, `{claims_block}`, `{facets_block}`, `{selections_example}` | [`src/config/prompts/claims_link_entities_prompt.py`](../src/config/prompts/claims_link_entities_prompt.py) |
| `claims_judge_equivalence.rubric` | `claims.judge_equivalence` | none (plain text; braces allowed) | [`src/config/prompts/claims_judge_equivalence_prompt.py`](../src/config/prompts/claims_judge_equivalence_prompt.py) |
| `claims_extract.role` | `claims.extract` | `{media_noun}`, `{grouped_clause}` | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.core` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/core.py`](../src/config/prompts/claims_extract/core.py) |
| `claims_extract.media.debate` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/media_layers.py`](../src/config/prompts/claims_extract/media_layers.py) |
| `claims_extract.media.news` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/media_layers.py`](../src/config/prompts/claims_extract/media_layers.py) |
| `claims_extract.media.podcast` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/media_layers.py`](../src/config/prompts/claims_extract/media_layers.py) |
| `claims_extract.media.research_paper` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/media_layers.py`](../src/config/prompts/claims_extract/media_layers.py) |
| `claims_extract.media.talk` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/media_layers.py`](../src/config/prompts/claims_extract/media_layers.py) |
| `claims_extract.media.generic` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/media_layers.py`](../src/config/prompts/claims_extract/media_layers.py) |
| `claims_extract.topics` | `claims.extract` | `{media_noun}`, `{focus_topics_block}`, `{inputs}` | [`src/config/prompts/claims_extract/topics.py`](../src/config/prompts/claims_extract/topics.py) |
| `claims_extract.takeaways` | `claims.extract` | `{media_noun}`, `{claims}` | [`src/config/prompts/claims_extract/takeaways.py`](../src/config/prompts/claims_extract/takeaways.py) |
| `claims_extract.grouping` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.flat` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.quotes` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.summary` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.factuality` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.contestability` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.topic_vocabulary` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.consolidation` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.focus_topics` | `claims.extract` | `{focus_topics}` | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.language` | `claims.extract` | `{language}` | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.max_claims` | `claims.extract` | `{max_claims}` | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.custom_instructions` | `claims.extract` | `{custom_instructions}` | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.output_contract` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.keep_groups_empty` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.keep_quotes_empty` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.keep_summary_empty` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.keep_factuality_null` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.keep_contestability_null` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `claims_extract.keep_assigned_topics_empty` | `claims.extract` | none (plain text; braces allowed) | [`src/config/prompts/claims_extract/sections.py`](../src/config/prompts/claims_extract/sections.py) |
| `space_assignment.system` | `geo.assign_spaces_to_sheet` | none (plain text; braces allowed) | [`src/config/prompts/space_assignment_prompt.py`](../src/config/prompts/space_assignment_prompt.py) |
| `space_assignment.output` | `geo.assign_spaces_to_sheet` | none (plain text; braces allowed) | [`src/config/prompts/space_assignment_prompt.py`](../src/config/prompts/space_assignment_prompt.py) |

### Payload templates

Each block is a complete input for the Hatchet *Trigger* box: realistic example data you can
replace, every prompt key that task reads as an empty `""`, and `llm_overrides` filled with the
current defaults. Delete the prompt keys you are not overriding. For `POST /tasks` wrap it as
`{"type": "<task name>", "payload": {...}}`.

#### `claims.extract`

```json
{
  "media_type": "debate",
  "title": "Cities should ban cars from their centers",
  "documents": [
    {
      "content": "Moderator: Tonight's motion: cities should ban cars from their centers.\nAna: Copenhagen cut car traffic in its center by 30 percent between 2010 and 2020.\nBen: Deliveries still need road access; a total ban would raise costs for restaurants."
    }
  ],
  "grouping": false,
  "include_summary": true,
  "prompt_overrides": {
    "claims_extract.role": "",
    "claims_extract.core": "",
    "claims_extract.media.debate": "",
    "claims_extract.media.news": "",
    "claims_extract.media.podcast": "",
    "claims_extract.media.research_paper": "",
    "claims_extract.media.talk": "",
    "claims_extract.media.generic": "",
    "claims_extract.grouping": "",
    "claims_extract.flat": "",
    "claims_extract.consolidation": "",
    "claims_extract.quotes": "",
    "claims_extract.summary": "",
    "claims_extract.factuality": "",
    "claims_extract.contestability": "",
    "claims_extract.topic_vocabulary": "",
    "claims_extract.focus_topics": "",
    "claims_extract.language": "",
    "claims_extract.max_claims": "",
    "claims_extract.custom_instructions": "",
    "claims_extract.output_contract": "",
    "claims_extract.topics": "",
    "claims_extract.takeaways": ""
  },
  "llm_overrides": {
    "claims_extract_model": "gemini-2.5-pro",
    "claims_extract_temperature": 0.2,
    "claims_extract_thinking_level": ""
  }
}
```

#### `news.extract_claims`

```json
{
  "headline": "City council votes to close Main Street to cars",
  "sources": [
    {
      "index": 0,
      "url": "https://example.com/a",
      "title": "Council votes 12-3",
      "content": "The council voted 12-3 on Tuesday to close Main Street to cars from June.",
      "published_at": "2026-03-05"
    }
  ],
  "topics": [
    "Traffic policy"
  ],
  "prompt_overrides": {
    "news_claim_extract": "",
    "news_debate_claim": "",
    "news_debate_completion": "",
    "news_debate_semantic_review": ""
  },
  "llm_overrides": {
    "gemini_news_claim_model": "gemini-3.5-flash",
    "gemini_news_claim_temperature": 0.2,
    "gemini_news_claim_thinking_level": "low",
    "gemini_news_debate_model": "gemini-3.5-flash",
    "gemini_news_debate_temperature": 0.0,
    "gemini_news_debate_thinking_level": "medium",
    "gemini_news_debate_review_model": "gemini-3.5-flash",
    "gemini_news_debate_review_temperature": 0.0,
    "gemini_news_debate_review_thinking_level": "high"
  }
}
```

#### `news.extract_claims_claude`

```json
{
  "headline": "City council votes to close Main Street to cars",
  "sources": [
    {
      "index": 0,
      "url": "https://example.com/a",
      "title": "Council votes 12-3",
      "content": "The council voted 12-3 on Tuesday to close Main Street to cars from June.",
      "published_at": "2026-03-05"
    }
  ],
  "topics": [
    "Traffic policy"
  ],
  "prompt_overrides": {
    "news_claim_extract": "",
    "news_claim_extract.claude_system": "",
    "news_debate_claim": "",
    "news_debate_claim.claude_system": "",
    "news_debate_completion": "",
    "news_debate_semantic_review": "",
    "news_debate_semantic_review.claude_system": ""
  },
  "llm_overrides": {
    "news_claim_claude_model": "claude-sonnet-4-5",
    "news_claim_claude_max_tokens": 32000,
    "gemini_news_claim_temperature": 0.2,
    "gemini_news_debate_temperature": 0.0,
    "gemini_news_debate_review_temperature": 0.0
  }
}
```

#### `news.extract_topics_and_claims`

```json
{
  "headline": "City council votes to close Main Street to cars",
  "sources": [
    {
      "index": 0,
      "url": "https://example.com/a",
      "title": "Council votes 12-3",
      "content": "The council voted 12-3 on Tuesday to close Main Street to cars from June.",
      "published_at": "2026-03-05"
    }
  ],
  "prompt_overrides": {
    "news_topics_extract": "",
    "news_topics_extract.regenerate": "",
    "news_topics_extract.system": "",
    "news_claim_extract": ""
  },
  "llm_overrides": {
    "news_claim_claude_model": "claude-sonnet-4-5",
    "gemini_news_claim_model": "gemini-3.5-flash",
    "gemini_news_claim_temperature": 0.2,
    "gemini_news_claim_thinking_level": "low"
  }
}
```

#### `news.extract_debate_claims`

```json
{
  "headline": "City council votes to close Main Street to cars",
  "sources": [
    {
      "index": 0,
      "url": "https://example.com/a",
      "title": "Council votes 12-3",
      "content": "The council voted 12-3 on Tuesday to close Main Street to cars from June.",
      "published_at": "2026-03-05"
    }
  ],
  "claims": [
    {
      "text": "The city council voted 12-3 on March 3, 2026 to close Main Street to cars from June 2026.",
      "topic": "Traffic policy"
    }
  ],
  "prompt_overrides": {
    "news_debate_claim": "",
    "news_debate_completion": "",
    "news_debate_semantic_review": ""
  },
  "llm_overrides": {
    "gemini_news_debate_model": "gemini-3.5-flash",
    "gemini_news_debate_temperature": 0.0,
    "gemini_news_debate_thinking_level": "medium",
    "gemini_news_debate_review_model": "gemini-3.5-flash",
    "gemini_news_debate_review_temperature": 0.0,
    "gemini_news_debate_review_thinking_level": "high"
  }
}
```

#### `news.review_collections`

```json
{
  "headline": "City council votes to close Main Street to cars",
  "sources": [
    {
      "index": 0,
      "url": "https://example.com/a",
      "title": "Council votes 12-3",
      "content": "The council voted 12-3 on Tuesday to close Main Street to cars from June.",
      "published_at": "2026-03-05"
    }
  ],
  "claims": [
    {
      "text": "The city council voted 12-3 on March 3, 2026 to close Main Street to cars from June 2026.",
      "topic": "Traffic policy"
    },
    {
      "text": "The closure of Main Street to cars takes effect in June 2026.",
      "topic": "Traffic policy"
    }
  ],
  "collections": [
    {
      "name": "Traffic policy",
      "type": "topic",
      "claim_indices": [
        0,
        1
      ]
    }
  ],
  "prompt_overrides": {
    "news_collection_review.group": "",
    "news_collection_review.rescue": "",
    "news_collection_review.review": "",
    "news_collection_review.check": "",
    "news_collection_review.order": "",
    "news_collection_review.order_rule": "",
    "news_collection_review.source_check": ""
  },
  "llm_overrides": {
    "news_collection_review_model": "gemini-3.5-flash",
    "news_collection_review_temperature": 0.1
  }
}
```

#### `news.extract_topics_and_entities`

```json
{
  "headline": "City council votes to close Main Street to cars",
  "summary": "The council voted 12-3 to close Main Street to cars from June.",
  "prompt_overrides": {
    "news_topics_entities.system": "",
    "news_topics_entities.curated_prefix": "",
    "news_topics_entities.topic_rule_curated": "",
    "news_topics_entities.topic_rule_free": "",
    "news_topics_entities.user_header": "",
    "news_topics_entities.user_json_example": "",
    "news_topics_entities.user_rules": ""
  },
  "llm_overrides": {
    "news_claim_claude_model": "claude-sonnet-4-5",
    "gemini_news_claim_temperature": 0.2
  }
}
```

#### `claims.link_entities`

```json
{
  "claims": [
    {
      "index": 0,
      "text": "Ana Ruiz said Copenhagen cut car traffic 30 percent by 2020."
    }
  ],
  "facets": [
    {
      "key": "people",
      "vocabulary": [
        {
          "index": 0,
          "name": "Ana Ruiz"
        },
        {
          "index": 1,
          "name": "Ben Ortiz"
        }
      ],
      "criteria": "people who are the subject of or actor in the claim"
    }
  ],
  "prompt_overrides": {
    "claims_link_entities": ""
  },
  "llm_overrides": {
    "claims_link_model": "gemini-3.5-flash",
    "claims_link_temperature": 0.1,
    "claims_link_thinking_level": "low"
  }
}
```

#### `claims.judge_equivalence`

```json
{
  "claim": {
    "text": "Copenhagen cut car traffic in its center by 30 percent between 2010 and 2020."
  },
  "candidates": [
    {
      "text": "Car traffic in central Copenhagen fell 30% from 2010 to 2020."
    },
    {
      "text": "Copenhagen expanded its pedestrian zones in 2015."
    }
  ],
  "prompt_overrides": {
    "claims_judge_equivalence.rubric": ""
  },
  "llm_overrides": {
    "claims_equivalence_model": "gemini-3.5-flash",
    "claims_equivalence_temperature": 0.0,
    "claims_equivalence_thinking_level": ""
  }
}
```

#### `podcast.extract_claims`

```json
{
  "episode_id": 1,
  "title": "Episode title",
  "description": "Episode description",
  "transcript": "<paste a transcript in the chosen format>",
  "transcript_format": "podscribe",
  "prompt_overrides": {
    "podcast_extract.topics": "",
    "podcast_extract.claims": "",
    "podcast_extract.takeaways": ""
  },
  "llm_overrides": {
    "gemini_premium_model": "gemini-2.5-pro",
    "gemini_premium_temperature": 0.2
  }
}
```

#### `guest.extract`

```json
{
  "title": "Episode 12: Cars and cities with Ana Ruiz",
  "description": "Host Ben Ortiz talks to urbanist Ana Ruiz.",
  "truncated_transcript": "Ben: Welcome Ana.\nAna: Thanks for having me.",
  "prompt_overrides": {
    "guest_extraction": ""
  },
  "llm_overrides": {
    "gemini_extraction_model": "gemini-2.5-flash",
    "gemini_extraction_temperature": 0.0
  }
}
```

#### `host.extract`

```json
{
  "title": "Episode 12: Cars and cities with Ana Ruiz",
  "description": "Host Ben Ortiz talks to urbanist Ana Ruiz.",
  "truncated_transcript": "Ben: Welcome Ana.\nAna: Thanks for having me.",
  "possible_hosts": [
    "Ben Ortiz"
  ],
  "prompt_overrides": {
    "host_extraction": ""
  },
  "llm_overrides": {
    "gemini_extraction_model": "gemini-2.5-flash",
    "gemini_extraction_temperature": 0.0
  }
}
```

#### `keyword.extract`

```json
{
  "episode": {
    "title": "Episode 12: Cars and cities",
    "claims": [
      "Copenhagen cut car traffic 30 percent by 2020."
    ]
  },
  "topics_list": [
    "Urban planning",
    "Transport"
  ],
  "prompt_overrides": {
    "keyword_extraction": ""
  },
  "llm_overrides": {
    "gemini_extraction_model": "gemini-2.5-flash",
    "gemini_extraction_temperature": 0.0
  }
}
```

#### `geo.assign_spaces_to_sheet`

```json
{
  "type_id": "<Geo type entity id>",
  "prompt_overrides": {
    "space_assignment.system": "",
    "space_assignment.output": ""
  },
  "llm_overrides": {
    "gemini_space_assignment_model": "gemini-2.5-flash",
    "gemini_space_assignment_temperature": 0.0
  }
}
```


## Part 3 — For coding agents

You are testing prompt changes against the extraction-api service. You need the API base URL and
an `X-API-Key`; ask the user for both. Use `curl` and `jq`. Never paste prompts into third-party
tools.

**Endpoints**

- `GET /prompts` — every prompt key with its `formatted` flag and `slots`, plus every overridable
  LLM setting (`llm_settings`) with its current value.
- `GET /prompts/{key}?format=text` — the current prompt text. Start every edit from this.
- `POST /tasks` with `{"type": "<task>", "payload": {...}}` → `201 {"id": ...}`;
  `GET /tasks/{id}` → `{"status", "result", "error"}`; poll until `COMPLETED`, `FAILED` or
  `CANCELLED`. Invalid overrides return `422` immediately with the reason.
- Sync endpoints (`POST /extract/guests`, `/extract/hosts`, `/extract/keywords`,
  `/extract/media/keywords`, `/extract/claim-keywords`, `/extract/news/claims`,
  `/extract/news/claims/claude`) accept the same two maps in the request body and answer in one call.

**Payload contract**

- `prompt_overrides`: `{"<key>": "<text>"}`. The key must exist in `GET /prompts`. If the prompt is
  `formatted`, the text may use only that key's `slots`, written `{name}`, and any literal brace
  must be doubled. Empty strings are rejected.
- `llm_overrides`: `{"<setting>": value}` from `llm_settings`. `*_model` is a non-empty string,
  `*_temperature` is a number in [0, 1], `*_thinking_level` is one of `""`, `minimal`, `low`,
  `medium`, `high`, `*_max_tokens` is a positive integer.
- Both maps apply to that run only. Any other field is the task's normal input; the templates in
  Part 2 show the shape for every task.

**The loop**

```bash
API=<base url>; KEY=<api key>
H=(-H "X-API-Key: $KEY" -H "Content-Type: application/json")

curl -s "${H[@]}" "$API/prompts" | jq '.prompts[] | {key, slots}'          # discover
curl -s "${H[@]}" "$API/prompts/claims_extract.factuality?format=text" > prompt.txt
# edit prompt.txt

cat > input.json <<'EOF'
{"media_type": "debate", "documents": [{"content": "...fixed test document..."}],
 "grouping": false, "classify_factuality": true,
 "llm_overrides": {"claims_extract_temperature": 0}}
EOF
jq --rawfile p prompt.txt '.prompt_overrides["claims_extract.factuality"] = $p' input.json > variant.json

run() { curl -s "${H[@]}" -d "$(jq -n --slurpfile b "$1" '{type: "claims.extract", payload: $b[0]}')" "$API/tasks"; }
wait_for() { while :; do s=$(curl -s "${H[@]}" "$API/tasks/$1"); case $(jq -r .status <<<"$s") in
  COMPLETED|FAILED|CANCELLED) echo "$s"; return;; esac; sleep 5; done; }

BASE=$(run input.json | jq -r .id);  VAR=$(run variant.json | jq -r .id)
wait_for "$BASE" > base.json;        wait_for "$VAR" > var.json
diff <(jq '.result.claims[] | {text, is_factual}' base.json) <(jq '.result.claims[] | {text, is_factual}' var.json)
jq -r '.error // "ok"' var.json
```

To turn a multi-line prompt into a JSON string on its own: `jq -Rs . prompt.txt` (or
`python3 -c 'import json,sys; print(json.dumps(open(sys.argv[1]).read()))' prompt.txt`).

**Discipline**

- Change one prompt key per run. Keep the input and every `llm_overrides` value identical between
  the baseline and the variant, with temperature `0`.
- Read the `422` message before retrying; it names the allowed slots or setting names.
- A retired or misspelled model name fails the run with the provider's error in `.error`; there is
  no fallback.
- Every run is billed. Keep test documents short; `claims.extract` is up to three model calls.
- What you cannot see over `curl`: the `overrides[...] applied=[...] unused=[...]` line is in the
  run's log in the Hatchet dashboard and the worker log, not in `GET /tasks/{id}`. If a result
  looks unchanged, ask the user to check that line, or confirm the key is read by the step (some
  `claims.extract` sections only render when their option is on).

**Making it permanent**

Put the final text into the file listed in the *Prompt keys* table, keeping the constant name and
its slots, run `uv run pytest` (the suite checks registration and template validity), and open a
pull request.
