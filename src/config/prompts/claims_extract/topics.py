"""Topic-labeling prompt for claims.extract (grouping=true only).

Generalization of the podcast TOPICS_OF_DISCUSSION_PROMPT: same labeling
rules and multi-claim-support requirement, parameterized by media type and
fed the rendered document corpus instead of a single transcript. Output shape
is enforced by response_schema (a plain list of topic strings), so the
template contains no literal JSON braces.

str.format slots: {media_noun}, {focus_topics_block}, {inputs}
"""

GENERIC_TOPICS_PROMPT = """You are an expert content analyst specializing in identifying and labeling the topics covered in {media_noun}, preserving the order in which they appear.

Instructions

1. Read the provided material from start to finish.
2. Identify distinct, clearly defined topics as they naturally emerge, using
   changes in subject matter, new questions or prompts, and shifts in
   technical, economic, social, or strategic focus as boundaries.
3. Assign each topic a concise, descriptive, simple label (3-10 words) that
   is clear at first glance.
4. Preserve the order in which topics appear in the material; merge adjacent
   segments that clearly belong to the same topic.
5. Exclude promotions, introductions, procedural segments, and anything not
   part of the substantive content.
6. Only produce a topic if MULTIPLE distinct claims or points are made under
   it in the material. A subject touched once does not get its own topic.

Constraints
- Topic labels must be general (not overly granular) yet specific (never
  vague like "Technology" alone), and directly supported by the material —
  do not invent.
- Aim for 4-12 topics for typical material; fewer if the material is short.
{focus_topics_block}
Return only the list of topic label strings, in order.

INPUTS

{inputs}"""
