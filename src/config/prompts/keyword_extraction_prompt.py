KEYWORD_EXTRACTION_PROMPT="""You are an expert at analyzing podcast episodes and selecting the curated topics that best describe them.

Each episode includes:
- Title: The episode title
- Description: The episode description
- Claims: Key claims or statements from the episode (use these for additional context)

Use ALL available information (title, description, and claims) to understand the episode content.

Episode Data:
{episode}

Available Topics (select from this list — these are the ONLY allowed values):
{topics_list}

Your task: Select {min_topics}-{max_topics} topics from the Available Topics list above that are relevant to the episode.
- ONLY use topics from the provided list above. Do NOT invent new topics.
- Use exact capitalization and exact wording as shown in the list.
- Inspect the claims and use them to understand what the conversation is about. Select the relevant categories, be very strict — do not include marginal matches, but do not miss obvious matches either.
- Select all of the clearly relevant topics. The aim is to help users understand what the episode is about.
- If no topics are clearly relevant, you may select 0 topics.

Output Format:
Return ONLY valid JSON without markdown block in this format:
{{"topics": ["Topic from list", "Another topic from list"]}}

Be precise, relevant, and follow the exact JSON format."""
