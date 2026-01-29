CLAIM_KEYWORD_EXTRACTION_PROMPT = """You are an expert at analyzing podcast claims and extracting relevant keywords and topics.

Episode Context:
- Title: {title}
- Description: {description}

Claims to Analyze:
{claims}

Available Topics (select from this list):
{topics_list}

Your task has THREE parts:

PART 1 - Generate Keywords and Assign to Claims:
For each claim, generate {min_keywords}-{max_keywords} keywords that capture the main themes and concepts of that specific claim.
- Create NEW keywords based on claim content (do NOT use the topics list above)
- Use sentence case capitalization (first letter capitalized, rest lowercase unless proper noun)
- Examples of good keywords: "Neural networks", "Quantum computing", "Climate policy", "Economic growth"
- Keywords should be specific to what each claim discusses
- Consider the episode context (title/description) for relevance
- Make each keyword about a single, atomic idea
- Use the standard, canonical name (Wikipedia article title style)
- Use full names: European Central Bank, Google DeepMind, etc.
- Disambiguate explicitly if needed: Inflation (Economy), AI Alignment, etc.
- Do not include the guest or host of the episode as the keyword
- Make sure the keywords are not ambiguous and have distinct meaning
- Select keywords that are meaningful even without the context provided, do not generate keywords that are not understandable or ambigious when seen without the context of the claim or the episode.

PART 2 - Select Topics for Each Claim:
For each claim, select {min_topics}-{max_topics} topics from the Available Topics list that are relevant.
- ONLY use topics from the provided list above
- Use exact capitalization as shown in the list
- Be strict: do not include marginal matches but do not miss obvious matches either
- If no topics are clearly relevant for a claim, you may select 0 topics

PART 3 - Associate Keywords with Topics:
For each unique topic used across all claims, identify which keywords are subcategories or closely related.
- A keyword is a subcategory if it represents a more specific concept within the topic's domain
- Example: If topic is "Artificial intelligence" and keywords include "Neural networks", that would be a subcategory
- Not all keywords need to belong to a topic - only assign those that clearly fit
- Only choose the most relevant connections, be strict
- If no keywords fit under a topic, omit that topic from topic_keywords

Output Format:
Return ONLY valid JSON without markdown block in this format:
{{"claim_keywords": {{"claim_id_1": ["Keyword A", "Keyword B"], "claim_id_2": ["Keyword C"]}}, "claim_topics": {{"claim_id_1": ["Topic X"], "claim_id_2": ["Topic Y"]}}, "topic_keywords": {{"Topic X": ["Keyword A"]}}}}

IMPORTANT:
- Use the EXACT claim IDs as provided in the claims list above
- All keywords in claim_keywords must be consistent (same keyword spelled the same way across claims)
- All topics in claim_topics must exist in the Available Topics list

Be precise, relevant, and follow the exact JSON format."""
