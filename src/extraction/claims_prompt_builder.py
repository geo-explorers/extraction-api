"""Prompt assembly for the generalized claims.extract pipeline.

Pure string composition — no LLM calls, no I/O beyond TranscriptParser — so
every section-inclusion rule is unit-testable. Section order for the main
extraction prompt: role -> inputs description -> media layer -> mode ->
core rules -> consolidation -> knob sections -> final validation -> fenced
caller instructions -> output contract -> INPUTS (corpus last, matching the
news/podcast prompt layout).
"""

from functools import lru_cache
from typing import List

from src.api.schemas.claims_extract_schema import ClaimsExtractInput, InputDocument
from src.preprocessing.transcript_parser import TranscriptParser
from src.config.overrides import prompts
from src.config.prompts.claims_extract import MEDIA_NOUNS

_SECTION_SEP = "\n\n"

# Every section is fetched through the registry at build time (never bound at
# import) so a run's prompt_overrides can replace any one of them by key.
_P = "claims_extract."


@lru_cache(maxsize=1)
def _parser() -> TranscriptParser:
    return TranscriptParser()


def _render_document_content(doc: InputDocument) -> str:
    """Plain text passes through; vendor transcript formats are parsed and
    rendered as "speaker: text" lines (NOT full_text, which drops speakers —
    quote attribution needs them)."""
    if doc.format == "plain":
        return doc.content.strip()
    parsed = _parser().parse(doc.content, format=doc.format)
    if not parsed.segments:
        return parsed.full_text
    return "\n".join(f"{seg.speaker}: {seg.clean_text}" for seg in parsed.segments)


def render_documents(input: ClaimsExtractInput) -> str:
    """Render the corpus with stable 0-based indices and per-document metadata."""
    blocks: List[str] = []
    for i, doc in enumerate(input.documents):
        lines = [f"--- DOCUMENT {i} ---"]
        if doc.title:
            lines.append(f"Title: {doc.title}")
        if doc.publisher:
            lines.append(f"Publisher: {doc.publisher}")
        if doc.published_at:
            lines.append(f"Published: {doc.published_at}")
        if doc.url:
            lines.append(f"URL: {doc.url}")
        for key, value in doc.metadata.items():
            lines.append(f"{key}: {value}")
        lines.append("Content:")
        blocks.append("\n".join(lines) + "\n" + _render_document_content(doc))
    return "\n\n".join(blocks)


def _overall_context_lines(input: ClaimsExtractInput) -> List[str]:
    lines: List[str] = []
    if input.title:
        lines.append(f"Overall title: {input.title}")
    if input.context:
        lines.append(f"Caller-provided context: {input.context}")
    return lines


def build_topics_prompt(input: ClaimsExtractInput) -> str:
    focus_block = ""
    if input.focus_topics:
        focus_block = (
            "- The caller is especially interested in these areas; give each "
            "its own topic when the material substantively supports it: "
            f"{', '.join(input.focus_topics)}.\n"
        )

    inputs_parts = _overall_context_lines(input)
    inputs_parts.append("DOCUMENTS\n\n" + render_documents(input))

    return prompts.get(_P + "topics").format(
        media_noun=MEDIA_NOUNS[input.media_type],
        focus_topics_block=focus_block,
        inputs="\n\n".join(inputs_parts),
    )


def _inputs_description(input: ClaimsExtractInput, grouping: bool) -> str:
    lines = [
        "Inputs",
        "",
        "You will be provided with (under INPUTS at the end of this prompt):",
        "- documents: an ordered list of documents, each with a numeric index. "
        "Each claim's document_indices must record which documents support it.",
    ]
    if input.title:
        lines.append(
            "- an overall title for the material (the debate motion, episode "
            "title, or story headline)."
        )
    if input.context:
        lines.append("- caller-provided context describing the material.")
    if input.topic_vocabulary:
        lines.append(
            "- TOPIC VOCABULARY: a numbered, closed list of topics; each "
            "claim's vocabulary_topic_indices selects from it."
        )
    if grouping:
        lines.append(
            "- topics: an ordered list of topic labels for this material; "
            "claims are grouped under them per the extraction mode below."
        )
    return "\n".join(lines)


def _final_validation(input: ClaimsExtractInput, grouping: bool) -> str:
    checks = [
        "─────────────────────────────────────────────",
        "FINAL VALIDATION",
        "─────────────────────────────────────────────",
        "",
        "Before outputting, verify every one of the following:",
        "- No claim contains unresolved pronouns or generic noun phrases "
        "(\"the company\", \"the study\") — replace with proper names.",
        "- No claim uses a resolvable relative date, and no date was invented.",
        "- No claim exceeds 35 words (apply the split test).",
        "- Every document_indices entry points to a document that actually "
        "supports the claim.",
    ]
    if grouping:
        checks.append(
            "- Every group has at least 2 claims, and reading back the claim "
            "at each index in claim_indices confirms it belongs to the group."
        )
    if input.include_quotes:
        checks.append(
            "- Every quote is verbatim and its claim_index points to the claim "
            "it supports."
        )
    if input.include_summary:
        checks.append(
            "- The summary is 350-500 characters and asserts no specific fact "
            "that is missing from the claims."
        )
    if input.media_type == "debate":
        checks.append(
            "- No claim reports a speech act: no participant's name is the "
            "subject of said/argued/questioned/suggested/claimed/conceded/"
            "pointed out — every such claim is rewritten as the proposition "
            "itself."
        )
        if input.title:
            checks.append(
                "- No claim restates the overall title (the motion) or its "
                "plain negation; the motion is already recorded."
            )
    if input.classify_factuality:
        checks.append(
            "- Every claim has is_factual set to an explicit true or false."
        )
        checks.append(
            "- is_factual is true only where the claim asserts one specific "
            "checkable thing (a named actor and act, a dated event, a "
            "quantity, a named study's finding, an on-record statement), and "
            "false for hedges, appraisals, unreferenced generalizations, "
            "causal theses, forecasts and normative positions."
        )
    if input.classify_contestability:
        checks.append(
            "- Every claim has is_contestable set to an explicit true or false, "
            "judged on the scope of its main assertion rather than its truth."
        )
    if input.topic_vocabulary:
        checks.append(
            "- Every vocabulary_topic_indices entry is a valid 0-based index "
            "into the TOPIC VOCABULARY, and a claim unrelated to every "
            "vocabulary topic has an empty list."
        )
    if len(input.documents) > 1:
        checks.append(
            "- The same fact does not appear as multiple near-duplicate claims "
            "from different documents."
        )
    return "\n".join(checks)


def _output_contract(input: ClaimsExtractInput, grouping: bool) -> str:
    lines = [prompts.get(_P + "output_contract")]
    if not grouping:
        lines.append(prompts.get(_P + "keep_groups_empty"))
    if not input.include_quotes:
        lines.append(prompts.get(_P + "keep_quotes_empty"))
    if not input.include_summary:
        lines.append(prompts.get(_P + "keep_summary_empty"))
    if not input.classify_factuality:
        lines.append(prompts.get(_P + "keep_factuality_null"))
    if not input.classify_contestability:
        lines.append(prompts.get(_P + "keep_contestability_null"))
    if not input.topic_vocabulary:
        lines.append(prompts.get(_P + "keep_assigned_topics_empty"))
    return "\n".join(lines)


def build_extract_prompt(input: ClaimsExtractInput, topics: List[str]) -> str:
    grouping = input.grouping and bool(topics)
    media_noun = MEDIA_NOUNS[input.media_type]

    role = prompts.get(_P + "role").format(
        media_noun=media_noun,
        grouped_clause=", grouped under the provided topics" if grouping else "",
    )

    sections = [
        role,
        _inputs_description(input, grouping),
        prompts.get(_P + "media." + input.media_type),
        prompts.get(_P + ("grouping" if grouping else "flat")),
        prompts.get(_P + "core"),
    ]
    if len(input.documents) > 1:
        sections.append(prompts.get(_P + "consolidation"))
    if input.include_quotes:
        sections.append(prompts.get(_P + "quotes"))
    if input.include_summary:
        sections.append(prompts.get(_P + "summary"))
    if input.classify_factuality:
        sections.append(prompts.get(_P + "factuality"))
    if input.classify_contestability:
        sections.append(prompts.get(_P + "contestability"))
    if input.topic_vocabulary:
        sections.append(prompts.get(_P + "topic_vocabulary"))
    if input.focus_topics:
        sections.append(
            prompts.get(_P + "focus_topics").format(
                focus_topics=", ".join(input.focus_topics)
            )
        )
    if input.language and input.language.lower() != "en":
        sections.append(prompts.get(_P + "language").format(language=input.language))
    if input.max_claims is not None:
        sections.append(
            prompts.get(_P + "max_claims").format(max_claims=input.max_claims)
        )

    sections.append(_final_validation(input, grouping))

    if input.custom_instructions:
        sections.append(
            prompts.get(_P + "custom_instructions").format(
                custom_instructions=input.custom_instructions
            )
        )

    sections.append(_output_contract(input, grouping))

    inputs_parts = ["INPUTS"] + _overall_context_lines(input)
    if input.topic_vocabulary:
        inputs_parts.append(
            "TOPIC VOCABULARY\n"
            + "\n".join(f"{i}. {t.label}" for i, t in enumerate(input.topic_vocabulary))
        )
    if grouping:
        inputs_parts.append("topics\n" + "\n".join(f"- {t}" for t in topics))
    inputs_parts.append("DOCUMENTS\n\n" + render_documents(input))
    sections.append("\n\n".join(inputs_parts))

    return _SECTION_SEP.join(sections)


def format_claims_for_takeaways(claims: List[dict]) -> str:
    """Render extracted claims for the takeaway selection pass. Claims with
    topic labels are grouped under them (first-seen order); flat-mode claims
    render as a plain list."""
    labeled = [c for c in claims if c.get("topic")]
    if not labeled:
        return "\n".join(f"- {c['text']}" for c in claims)

    by_topic: dict[str, List[str]] = {}
    for c in claims:
        by_topic.setdefault(c.get("topic") or "Other", []).append(c["text"])
    sections = []
    for topic, texts in by_topic.items():
        sections.append(f"Topic: {topic}\n" + "\n".join(f"- {t}" for t in texts))
    return "\n\n".join(sections)


def build_takeaways_prompt(input: ClaimsExtractInput, claims: List[dict]) -> str:
    return prompts.get(_P + "takeaways").format(
        media_noun=MEDIA_NOUNS[input.media_type],
        claims=format_claims_for_takeaways(claims),
    )
