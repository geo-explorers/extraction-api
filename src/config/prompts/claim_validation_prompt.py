"""Claim validation prompt for context independence checking."""

CLAIM_VALIDATION_PROMPT = """You are evaluating whether a claim is understandable in isolation, without requiring any additional context from the original conversation or text.

A claim is VALID if:
- It is self-contained and can be understood without additional context
- All entities, references, and pronouns are clear and explicit
- No missing information is needed to understand what the claim is saying
- The reader can fully grasp the meaning without knowing what came before or after

A claim is INVALID if:
- It contains unresolved references (e.g., "he", "she", "they", "this", "that", "the company", "the project")
- It requires prior context to understand who or what is being discussed
- Key terms or entities are ambiguous without context
- It references "the above", "as mentioned", "the previous", or similar context-dependent phrases
- It uses acronyms or abbreviations that are not universally known without first defining them

Claim to evaluate:
{claim_text}

First, provide a brief explanation (1-2 sentences) of why the claim is or is not context-independent.
Then, provide your verdict (is_valid: true or false).
"""
