"""Claim validation prompt for context independence checking."""

CLAIM_VALIDATION_PROMPT = """You are evaluating whether a claim is understandable in isolation, without requiring any additional explanation.

A claim is VALID if:
- It is self-contained and can be understood without additional context
- All entities, references, and pronouns are refered to by their names or can be understood from the remaining parts of the claim
- If the proper and complete name is used for an entity, we should count that as context independent, even if it is not well known.
- No missing information is needed to understand what the claim is saying
- The reader can fully grasp the meaning without knowing what came before or after
- The entities refered to are refered by their complete proper names.
- A well informed reader can understand the claim without further explanations
- acronyms or abbreviations are allowed which is well known are easy to understand like US, FBI, NASA etc. 
- If the acronyms or abbreviations can be understood from the context should be also be allowed.

A claim is INVALID if:
- It contains unresolved references (e.g., "he", "she", "they", "this", "that", "the company", "the project")
- It requires prior context to understand who or what is being discussed
- Key terms or entities are linguistically ambigius
- It references "the above", "as mentioned", "the previous", or similar context-dependent phrases

Claim to evaluate:
{claim_text}

First, provide a brief explanation (1-3 sentences) of why the claim is or is not context-independent.
Then, provide your verdict (is_valid: true or false).
"""
