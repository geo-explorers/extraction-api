KEYWORD_EXTRACTION_PROMPT="""You are an expert at analyzing podcast episodes and extracting relevant information.

Each episode includes:
- Title: The episode title
- Description: The episode description
- Claims: Key claims or statements from the episode (use these for additional context)

Use ALL available information (title, description, and claims) to understand the episode content.

Episode Data:
{episode}

Available Topics (select from this list for topics field):
{topics_list}

Your task has THREE parts:

PART 1 - Generate Keywords (Free-form):
Generate {min_keywords}-{max_keywords} NEW keywords that capture the main themes and concepts discussed in the episode.
- Create NEW keywords based on episode content (do NOT use the topics list above)
- Use sentence case capitalization (first letter capitalized, rest lowercase unless proper noun)
- Examples of good keywords: "Neural networks", "Quantum computing", "Climate policy", "Economic growth", "Military strategy", "Game design", "Cryptocurrency trading"
- Focus on specific concepts, technologies, people, events, or ideas discussed
- Keywords should be searchable and descriptive
- Focus on providing relevant keywords that help people understand what this episode is about
- Do not use guests of the episode as keywords
- Make each topic about a single, atomic idea. Avoid using 'A and B' as a keyword if that is not the canonical form of the idea. Use 'A' and 'B' as seperate topics preferrably.
- Use the standard, canonical name, prefer the wording you’d expect as a Wikipedia article title.
- Use full names: European Central Bank, Google DeepMind, European Union, etc.
- Disambiguate explicitly. If a term has multiple meanings or interpretations., add a clear qualifier that makes the meaning clear in context. Pattern: Inflation (Economy), Inflation (cosmology), AI Alignment, Marketing alignment etc.
- Do not include the guest or host of the episode as the keyword
- Do not include people's names as keywords, person names are not keywords
- Make sure the keywords are not ambiguous and have distinct meaning

PART 2 - Select Topics (From provided list):
Select {min_topics}-{max_topics} topics from the Available Topics list above that are relevant to the episode.
- ONLY use topics from the provided list above
- Use exact capitalization as shown in the list
- Inspect the claims and use them to understand what the conversation is about, Select the relevant categories, be very strict, do not include marginal matches but do not miss obvious mathces either.
- Make sure to select all of the clearly relevant topics, the aim is to help users understand what the episode is about.
- If no topics are clearly relevant, you may select 0 topics

PART 3 - Associate Keywords with Topics:
For each topic you selected in PART 2, identify which keywords from PART 1 are subcategories or closely related to that topic.
- A keyword is a subcategory if it represents a more specific concept within the topic's domain
- Example: If topic is "Artificial intelligence" and keywords include "Neural networks" and "GPT-4", those would be subcategories
- Not all keywords need to belong to a topic - only assign those that clearly fit
- Only choose the most relevant connections. Add a keyword under a topic if it is obviously relevant.
- If no keywords fit under a topic, omit that topic from topic_keywords

Output Format:
Return ONLY valid JSON without markdown block in this format:
{{"keywords": ["Keyword One", "Keyword Two", "Keyword Three"], "topics": ["Topic from list", "Another topic from list"], "topic_keywords": {{"Topic from list": ["Keyword One"]}}}}

Be precise, relevant, and follow the exact JSON format."""
