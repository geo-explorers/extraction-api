"""Prompt registry + per-run overrides (prompt_overrides / llm_overrides).

Engine-independent: exercises the registry catalog, the validators that run
at enqueue, the override scope and its propagation, and the builders that
now fetch their prompt through the registry.
"""

import asyncio
import importlib
import inspect
import pkgutil
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from src.config import prompts as prompts_pkg
from src.config.overrides import (
    activate,
    bind_context,
    current_scope,
    llm,
    llm_setting_names,
    prompts,
    validate_llm_overrides,
    validate_prompt_overrides,
)
from src.config.prompt_registry import PROMPTS, template_slots
from src.config.settings import settings


# ── Registry catalog ─────────────────────────────────────────────────────────

# Constants under src/config/prompts that are not prompt texts.
_NOT_PROMPTS = {"MEDIA_NOUNS", "ENTITY_TYPE_NAMES", "ENTITY_TYPE_PROMPT"}


def _prompt_modules():
    for info in pkgutil.walk_packages(prompts_pkg.__path__, prompts_pkg.__name__ + "."):
        yield importlib.import_module(info.name)


def test_every_public_prompt_constant_is_registered():
    registered = {e.text for e in PROMPTS.values()}
    missing = []
    for mod in _prompt_modules():
        for name in dir(mod):
            if not name.isupper() or name.startswith("_") or name in _NOT_PROMPTS:
                continue
            value = getattr(mod, name)
            if isinstance(value, str):
                if value not in registered:
                    missing.append(f"{mod.__name__}.{name}")
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, str) and v not in registered:
                        missing.append(f"{mod.__name__}.{name}[{k!r}]")
    assert not missing, f"prompt constants missing from prompt_registry: {missing}"


def test_registry_keys_are_dotted_slugs_with_parseable_defaults():
    for key, entry in PROMPTS.items():
        assert key == entry.key
        assert key.replace(".", "_").replace("_", "a").isalnum(), key
        assert entry.text
        if entry.formatted:
            template_slots(entry.text)  # raises on a malformed default


def test_registry_covers_every_task_family():
    keys = set(PROMPTS)
    for expected in [
        "news_claim_extract",
        "news_debate_claim",
        "news_debate_completion",
        "news_debate_semantic_review",
        "news_topics_extract",
        "news_topics_entities.user_rules",
        "news_collection_review.group",
        "claims_extract.core",
        "claims_extract.media.debate",
        "claims_extract.factuality",
        "claims_link_entities",
        "claims_judge_equivalence.rubric",
        "podcast_extract.claims",
        "guest_extraction",
        "host_extraction",
        "keyword_extraction",
        "space_assignment.system",
    ]:
        assert expected in keys, expected


def test_template_slots_accepts_only_plain_named_slots():
    assert template_slots("a {x} b {{lit}} {y} {x}") == ["x", "y"]
    for bad in ["unbalanced {", "{} positional", "{0} positional", "{x.attr}", "{x[0]}", "{x!r}", "{x:>10}"]:
        with pytest.raises(ValueError):
            template_slots(bad)


# ── Validation (what OverridesMixin runs at enqueue) ─────────────────────────


def test_validate_prompt_overrides_accepts_subset_of_slots():
    out = validate_prompt_overrides({"news_topics_extract": "Only {headline} please."})
    assert out == {"news_topics_extract": "Only {headline} please."}


def test_validate_prompt_overrides_rejects_unknown_key_and_extra_slot():
    with pytest.raises(ValueError, match="unknown prompt key"):
        validate_prompt_overrides({"nope": "x"})
    with pytest.raises(ValueError, match="slots \\['bogus'\\]"):
        validate_prompt_overrides({"news_topics_extract": "{headline} {bogus}"})
    with pytest.raises(ValueError, match="not a valid format template"):
        validate_prompt_overrides({"news_topics_extract": "{headline"})
    # traversal into the format arguments is rejected, not truncated to 'headline'
    with pytest.raises(ValueError, match="not a plain"):
        validate_prompt_overrides({"news_topics_extract": "{headline.__class__} {content}"})
    with pytest.raises(ValueError, match="non-empty"):
        validate_prompt_overrides({"news_topics_extract": "   "})


def test_validate_prompt_overrides_allows_braces_in_plain_prompts():
    # A plain (non-formatted) prompt is concatenated verbatim: braces are fine.
    assert not PROMPTS["claims_extract.core"].formatted
    out = validate_prompt_overrides({"claims_extract.core": 'Return {"x": 1}'})
    assert out["claims_extract.core"] == 'Return {"x": 1}'


def test_llm_setting_names_are_the_tunable_model_params():
    names = llm_setting_names()
    assert "claims_extract_model" in names
    assert "gemini_news_claim_thinking_level" in names
    assert "news_claim_claude_max_tokens" in names
    assert "news_collection_review_model" in names
    assert not any("embedding" in n or "rate_limit" in n for n in names)


def test_every_llm_looking_setting_has_an_explicit_override_decision():
    # A new *_model / *_temperature / *_thinking_level / *_max_tokens field must be
    # listed as overridable or as deliberately not — never exposed by name alone.
    from src.config.overrides import LLM_SETTINGS, LLM_SETTINGS_NOT_OVERRIDABLE, LLM_SETTING_SUFFIXES
    from src.config.settings import Settings

    looks_like_llm = {n for n in Settings.model_fields if n.endswith(LLM_SETTING_SUFFIXES)}
    decided = set(LLM_SETTINGS) | set(LLM_SETTINGS_NOT_OVERRIDABLE)
    assert looks_like_llm - decided == set(), "undecided LLM-looking settings"
    assert set(LLM_SETTINGS) & set(LLM_SETTINGS_NOT_OVERRIDABLE) == set()
    assert set(LLM_SETTINGS) <= set(Settings.model_fields)


def test_validate_llm_overrides_types_and_ranges():
    out = validate_llm_overrides(
        {
            "claims_extract_model": " gemini-3.5-pro ",
            "claims_extract_temperature": 0,
            "claims_extract_thinking_level": "Low",
            "news_claim_claude_max_tokens": 4000,
        }
    )
    assert out == {
        "claims_extract_model": "gemini-3.5-pro",
        "claims_extract_temperature": 0.0,
        "claims_extract_thinking_level": "low",
        "news_claim_claude_max_tokens": 4000,
    }
    for bad in [
        {"not_a_setting": "x"},
        {"claims_extract_model": ""},
        {"claims_extract_temperature": 3},
        {"gemini_news_claim_temperature": 1.5},  # also feeds Claude, whose ceiling is 1.0
        {"claims_extract_temperature": True},
        {"claims_extract_thinking_level": "max"},
        {"news_claim_claude_max_tokens": 0},
        {"news_claim_claude_max_tokens": 1.5},
    ]:
        with pytest.raises(ValueError):
            validate_llm_overrides(bad)


def test_mixin_validates_at_model_construction():
    from src.api.schemas.news_claim_extract_schema import NewsClaimExtractRequest

    base = {"headline": "H", "sources": [], "topics": []}
    ok = NewsClaimExtractRequest(**base)
    assert ok.prompt_overrides == {} and ok.llm_overrides == {}
    with pytest.raises(ValidationError, match="unknown prompt key"):
        NewsClaimExtractRequest(**base, prompt_overrides={"typo": "x"})
    with pytest.raises(ValidationError, match="unknown llm setting"):
        NewsClaimExtractRequest(**base, llm_overrides={"typo": "x"})


def test_every_registered_task_input_carries_the_override_fields():
    from src.tasks.registry import get_task, task_names

    for name in task_names():
        fields = get_task(name).input_model.model_fields
        assert "prompt_overrides" in fields and "llm_overrides" in fields, name


# ── Scope: resolution, bookkeeping, propagation ──────────────────────────────


def test_prompts_get_prefers_active_override_and_records_it():
    default = PROMPTS["news_topics_extract"].text
    assert current_scope() is None
    assert prompts.get("news_topics_extract") == default
    with activate({"news_topics_extract": "custom {headline}"}) as scope:
        assert prompts.get("news_topics_extract") == "custom {headline}"
        assert prompts.get("news_topics_extract.system") == PROMPTS["news_topics_extract.system"].text
        assert scope.applied == {"news_topics_extract"}
        assert scope.prompts_read == {"news_topics_extract", "news_topics_extract.system"}
        assert scope.unused == set()
    assert prompts.get("news_topics_extract") == default


def test_llm_get_prefers_active_override_and_flags_unused():
    with activate(llm_overrides={"claims_extract_model": "m-x", "claims_link_model": "m-y"}) as scope:
        assert llm.get("claims_extract_model") == "m-x"
        assert llm.get("claims_extract_temperature") == settings.claims_extract_temperature
        assert scope.applied == {"llm:claims_extract_model"}
        assert scope.unused == {"llm:claims_link_model"}
        line = scope.summary("t")
        assert "applied=['llm:claims_extract_model']" in line
        assert "unused=['llm:claims_link_model']" in line
    assert llm.get("claims_extract_model") == settings.claims_extract_model


def test_scope_propagates_through_to_thread_and_bind_context():
    async def run():
        with activate(llm_overrides={"claims_extract_model": "in-thread"}):
            via_to_thread = await asyncio.to_thread(llm.get, "claims_extract_model")
            with ThreadPoolExecutor(max_workers=2) as pool:
                bare = list(pool.map(llm.get, ["claims_extract_model"]))
                bound = list(pool.map(bind_context(llm.get), ["claims_extract_model"] * 3))
        return via_to_thread, bare, bound

    via_to_thread, bare, bound = asyncio.run(run())
    assert via_to_thread == "in-thread"
    assert bare == [settings.claims_extract_model]  # a bare pool thread has no scope
    assert bound == ["in-thread"] * 3


def test_with_overrides_activates_from_the_input_and_logs_summary(caplog):
    import logging

    from src.tasks.base import with_overrides
    from src.tasks.ping import PingInput

    seen = {}

    @with_overrides(label="probe")
    async def step(input, ctx):
        seen["model"] = llm.get("claims_extract_model")
        seen["prompt"] = prompts.get("claims_extract.core")
        return "done"

    inp = PingInput(
        prompt_overrides={"claims_extract.core": "CORE!"},
        llm_overrides={"claims_extract_model": "probe-model", "claims_link_model": "unused"},
    )
    with caplog.at_level(logging.INFO, logger="src.config.overrides"):
        assert asyncio.run(step(inp, ctx=None)) == "done"
    assert seen == {"model": "probe-model", "prompt": "CORE!"}
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("overrides[probe]")]
    assert len(lines) == 1
    assert "applied=['claims_extract.core', 'llm:claims_extract_model']" in lines[0]
    assert "unused=['llm:claims_link_model']" in lines[0]
    assert current_scope() is None

    # No overrides given: defaults resolve and the line says so.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="src.config.overrides"):
        asyncio.run(step(PingInput(), ctx=None))
    assert seen["model"] == settings.claims_extract_model
    assert any("overrides[probe] none given" in r.getMessage() for r in caplog.records)


def test_dag_steps_carry_qualified_labels():
    import re
    from pathlib import Path

    for path in Path("src/tasks").glob("*.py"):
        text = path.read_text()
        for m in re.finditer(r"@(\w+_workflow)\.task\(", text):
            # every DAG step is decorated with a "<workflow>:<step>" label
            after = text[m.end():]
            assert re.search(r'@with_overrides\(label="[a-z_.]+:\w+"\)\s*\nasync def', after), (path, m.start())


def test_every_http_handler_with_an_override_payload_is_decorated():
    import inspect

    from src.api.main import app

    checked = 0
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None or "POST" not in (getattr(route, "methods", None) or set()):
            continue
        params = inspect.signature(endpoint).parameters.values()
        takes_payload = any(
            hasattr(p.annotation, "model_fields") and "prompt_overrides" in p.annotation.model_fields
            for p in params
        )
        if route.path == "/tasks":
            continue  # the facade enqueues; the worker activates
        if takes_payload:
            assert getattr(endpoint, "overrides_label", None), f"{route.path} lacks @with_payload_overrides"
            checked += 1
    assert checked >= 7


def test_with_payload_overrides_wraps_sync_and_async_handlers():
    from src.config.overrides import with_payload_overrides
    from src.tasks.ping import PingInput

    @with_payload_overrides("http:probe")
    def sync_handler(request: PingInput) -> str:
        return llm.get("claims_extract_model")

    @with_payload_overrides("http:probe-async")
    async def async_handler(request: PingInput, extra: int = 0) -> str:
        return llm.get("claims_extract_model")

    req = PingInput(llm_overrides={"claims_extract_model": "http-model"})
    assert sync_handler(req) == "http-model"
    assert sync_handler(request=req) == "http-model"
    assert asyncio.run(async_handler(req, extra=1)) == "http-model"
    assert sync_handler(PingInput()) == settings.claims_extract_model
    assert inspect.signature(sync_handler).parameters["request"].annotation is PingInput
    assert current_scope() is None




# ── Builders honor the registry ──────────────────────────────────────────────


def test_claims_extract_builder_uses_overridden_sections():
    from src.api.schemas.claims_extract_schema import ClaimsExtractInput, InputDocument
    from src.extraction.claims_prompt_builder import build_extract_prompt, build_topics_prompt

    inp = ClaimsExtractInput(
        media_type="debate",
        documents=[InputDocument(content="A: hello")],
        classify_factuality=True,
    )
    baseline = build_extract_prompt(inp, ["T1"])
    assert PROMPTS["claims_extract.factuality"].text in baseline
    assert PROMPTS["claims_extract.media.debate"].text in baseline
    with activate(
        {
            "claims_extract.factuality": "FACT-RULES-X",
            "claims_extract.media.debate": "DEBATE-LAYER-X",
            "claims_extract.role": "ROLE for {media_noun}{grouped_clause}",
            "claims_extract.topics": "TOPICS {media_noun}\n{focus_topics_block}{inputs}",
        }
    ):
        out = build_extract_prompt(inp, ["T1"])
        assert "FACT-RULES-X" in out and "DEBATE-LAYER-X" in out
        assert out.startswith("ROLE for debate transcripts, grouped under the provided topics")
        assert PROMPTS["claims_extract.factuality"].text not in out
        assert build_topics_prompt(inp).startswith("TOPICS debate transcripts")
    assert build_extract_prompt(inp, ["T1"]) == baseline


def test_news_topics_entities_user_prompt_is_byte_identical_to_legacy_assembly():
    from src.config.prompts.news_topics_entities_prompt import (
        ENTITY_TYPE_PROMPT,
        TOPIC_RULE_CURATED,
        TOPIC_RULE_FREE,
        build_user_prompt,
    )

    def legacy(headline, summary, has_curated):
        topic_rule = TOPIC_RULE_CURATED if has_curated else TOPIC_RULE_FREE
        header = (
            "Identify the key topics and entities for this news story.\n"
            "\n"
            f"## Story: \"{headline}\"\n"
            f"{summary}\n"
            "\n"
        )
        json_block = (
            "---\n"
            "\n"
            "Return JSON:\n"
            "```json\n"
            "{\n"
            "  \"topics\": [\n"
            "    {\"name\": \"Regulation\", \"relevance\": 0.95},\n"
            "    {\"name\": \"DeFi\", \"relevance\": 0.7},\n"
            "    {\"name\": \"Stablecoin Depegging\", \"relevance\": 0.8}\n"
            "  ],\n"
            "  \"entities\": [\n"
            "    {\"name\": \"Ripple\", \"type\": \"Project\", \"role\": \"Subject — launched the buyback\"},\n"
            "    {\"name\": \"Brad Garlinghouse\", \"type\": \"Person\", \"role\": \"CEO, announced the initiative\"},\n"
            "    {\"name\": \"San Francisco\", \"type\": \"City\", \"role\": \"Location of company headquarters\"}\n"
            "  ]\n"
            "}\n"
            "```\n"
            "\n"
        )
        rules = (
            "Rules:\n"
            f"{topic_rule}\n"
            "- Entities: People, companies, projects, organizations, cities, countries mentioned\n"
            f"  - {ENTITY_TYPE_PROMPT}\n"
            "  - Choose the most specific type that fits (e.g., \"City\" not \"Place\").\n"
            "  - Any company, business, startup, exchange, or for-profit organization → use \"Project\" (never \"Company\" or \"Organization\").\n"
            "  - Any individual person — including public figures, politicians, celebrities, founders, executives — → use \"Person\" (never \"Public figure\").\n"
            "  - role: brief description of their involvement (one phrase)\n"
            "- Use official/full names for entities\n"
            "- Relevance: 0.0-1.0 how central this topic/entity is to the story"
        )
        return header + json_block + rules

    for has_curated in (True, False):
        assert build_user_prompt("Head {x}", "Sum", has_curated) == legacy("Head {x}", "Sum", has_curated)
    with activate({"news_topics_entities.user_rules": "RULES {topic_rule} | {entity_types}"}):
        out = build_user_prompt("H", "S", False)
        assert out.endswith("RULES " + TOPIC_RULE_FREE + " | " + ENTITY_TYPE_PROMPT)


def test_service_builders_render_overrides():
    from src.api.schemas.news_claim_extract_schema import NewsArticleSource
    from src.api.services.news_claim_extract_service import _build_prompt
    from src.extraction.claim_equivalence_judge import build_prompt as judge_prompt

    src = [NewsArticleSource(index=0, url="u", title="t", content="body", published_at="2026-03-05")]
    with activate({"news_claim_extract": "NEWS {headline} / {topics}", "claims_judge_equivalence.rubric": "RUBRIC-X"}):
        assert _build_prompt("Head", src, ["T"]) == "NEWS Head / ['T']"
        assert judge_prompt("c", ["a", "b"]).startswith("RUBRIC-X\n\nCLAIM:\nc")


def test_prompts_endpoint_lists_catalog_and_serves_text():
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)
    headers = {"X-API-Key": settings.api_key}
    listing = client.get("/prompts", headers=headers).json()
    keys = {p["key"] for p in listing["prompts"]}
    assert keys == set(PROMPTS)
    entry = next(p for p in listing["prompts"] if p["key"] == "news_topics_extract")
    assert entry["formatted"] and entry["slots"] == ["headline", "content"]
    assert {s["name"] for s in listing["llm_settings"]} == set(llm_setting_names())

    one = client.get("/prompts/claims_extract.media.debate", headers=headers).json()
    assert one["text"] == PROMPTS["claims_extract.media.debate"].text
    text = client.get("/prompts/claims_extract.media.debate", params={"format": "text"}, headers=headers)
    assert text.headers["content-type"].startswith("text/plain") and text.text == one["text"]
    assert client.get("/prompts/nope", headers=headers).status_code == 404


def test_tasks_facade_rejects_bad_override_with_422():
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)
    resp = client.post(
        "/tasks",
        headers={"X-API-Key": settings.api_key},
        json={"type": "ping", "payload": {"message": "hi", "prompt_overrides": {"typo": "x"}}},
    )
    assert resp.status_code == 422
    assert "unknown prompt key" in resp.text


def test_decorated_route_activates_scope_through_fastapi(monkeypatch):
    """The decorator must survive FastAPI's signature resolution (body parsing
    via functools.wraps) and its threadpool for sync handlers."""
    from fastapi.testclient import TestClient

    from src.api.main import app
    from src.api.routers import guest_extraction as router_mod

    def fake_extract(title, description, truncated_transcript=""):
        return [{"name": llm.get("gemini_extraction_model"), "urls": [prompts.get("guest_extraction")[:12]]}]

    monkeypatch.setattr(router_mod, "extract_podcast_guests", fake_extract)
    client = TestClient(app)
    resp = client.post(
        "/extract/guests",
        headers={"X-API-Key": settings.api_key},
        json={
            "title": "t", "description": "d", "truncated_transcript": "x",
            "llm_overrides": {"gemini_extraction_model": "route-model"},
            "prompt_overrides": {"guest_extraction": "PROMPT-VIA-ROUTE {title}"},
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["guests"] == [{"name": "route-model", "urls": ["PROMPT-VIA-R"]}]
    assert current_scope() is None
