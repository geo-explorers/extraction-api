"""Offline experiment for the Debate/Simple scope classification — no Hatchet, no API.

The product change is a scope rule inside claims.extract; this is the fast loop that
lets the rule be iterated against a labelled set before it goes anywhere near the
extraction prompt. Classifying fixed claims costs one call per run instead of a full
extraction, so a rubric edit is scored in ~20 seconds.

    uv run python -m src.cli.eval_debatable gold --rubric v1 --runs 3
    uv run python -m src.cli.eval_debatable gold --rubric v1 --rubric v2   # A/B

Rubric source: the Debate tag definition (Category "Form", counterpart "Simple",
verified 2026-09-10). Debate classifies SCOPE, not truth and not whether anyone is
currently arguing about it.
"""

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.api.schemas.claims_extract_schema import ClaimsExtractInput
from src.config.prompts.claims_extract import CONTESTABILITY_SECTION, FACTUALITY_SECTION
from src.config.settings import settings
from src.extraction.claims_extractor import ClaimsExtractor
from src.extraction.claims_prompt_builder import build_extract_prompt
from src.pipeline.claims_extract_core import assemble_result

# Profile of the 1,378 claims carrying Geo's `Debate` tag, measured 2026-09-10. The
# target an extraction run is drifting toward or away from.
CORPUS_PROFILE = {
    "median_words": 10.0,
    "normative_pct": 52.3,
    "hedged_pct": 1.9,
    "reports_finding_pct": 0.1,
    "cites_study_pct": 0.3,
}

_SHAPE = {
    "normative_pct": r"\b(should|must|ought to)\b",
    # `may` only as the modal: case-sensitive, and not the month before a day number,
    # which would otherwise count the rubric's own dated examples as hedging.
    "hedged_pct": r"\b(can|could|might)\b|\bmay\b(?!\s+\d)",
    "reports_finding_pct": r"\b(found|according to|study (found|shows)|reported|studies have shown)\b",
    "cites_study_pct": r"\b(study|studies|research|peer-reviewed|survey)\b",
}


def shape_metrics(texts: List[str]) -> Dict[str, float]:
    """How far this run sits from the curated Debate corpus. Cheap, deterministic,
    and the first thing to look at after a prompt edit: hedging and length move
    long before the classifier's verdict does."""
    if not texts:
        return {}
    out = {"median_words": statistics.median(len(t.split()) for t in texts)}
    for key, rx in _SHAPE.items():
        flags = 0 if key == "hedged_pct" else re.I
        out[key] = round(100 * sum(1 for t in texts if re.search(rx, t, flags)) / len(texts), 1)
    return out

# ── Rubric variants ─────────────────────────────────────────────────────────
# Keep every version that has been scored; the winner is what moves into
# src/config/prompts/claims_extract/sections.py.


# `shipped` is the section this repo actually renders into the extraction prompt —
# the only variant whose score says anything about production. To A/B a candidate,
# add it here written the way the shipped sections are (instruct the model to set
# the boolean field) and pass `--rubric shipped --rubric <candidate>`.
#
# Recorded from the iteration that produced these: an asymmetric bar needs worked
# examples of BOTH classes. A draft that said "if not certain, answer not_factual"
# with only positive examples labelled all 30 gold claims not_factual, including
# "Tyler Robinson killed Charlie Kirk".
RUBRICS: Dict[str, str] = {"shipped": CONTESTABILITY_SECTION}


FACT_RUBRICS: Dict[str, str] = {"shipped": FACTUALITY_SECTION}

# strict_v1 collapsed: told "if not certain, answer not_factual", the model answered
# not_factual for all 30, including "Tyler Robinson killed Charlie Kirk". Two causes —
# the asymmetry had no counterweight, and "checkable" was read as "I can verify it
# myself". v2 keeps the asymmetry, anchors both classes with worked examples, and
# separates checkability from the model's own knowledge.



FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

# The shipped sections instruct the model to set a boolean field, so the classifier
# answers `true`/`false` and the scorer maps that onto each task's gold vocabulary.
# Grading the shipped text verbatim is the whole point: a paraphrase would score a
# document that never runs.
TASKS: Dict[str, Dict[str, Any]] = {
    "scope": {
        "rubrics": RUBRICS,
        "default_rubric": "shipped",
        "flag": "is_contestable",
        "positive": "debate",
        "negative": "simple",
        "gold": FIXTURES / "claim_scope_gold.json",
        "header": "You are classifying the form of extracted claims for a knowledge graph.",
    },
    "factual": {
        "rubrics": FACT_RUBRICS,
        "default_rubric": "shipped",
        "flag": "is_factual",
        "positive": "factual",
        "negative": "not_factual",
        "gold": FIXTURES / "claim_factuality_gold.json",
        "header": "You are classifying whether extracted claims assert specific checkable facts.",
    },
}


class ScopeLabel(BaseModel):
    index: int = Field(description="0-based index of the claim being classified.")
    # Task-agnostic on purpose. This description is part of the decode-constrained
    # schema, so naming one task's labels here silently forces every other task's
    # answers into the fallback — which reads as a confident, uniform verdict.
    label: str = Field(description="Exactly one of the labels defined in the instructions.")


class ScopeLabels(BaseModel):
    labels: List[ScopeLabel] = Field(default_factory=list)


def build_prompt(rubric: str, claims: List[str], task: str = "scope") -> str:
    cfg = TASKS[task]
    header = cfg["header"]
    listing = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    return (
        f"{header}\n\n"
        f"{rubric}\n\n"
        f"Answer for `{cfg['flag']}` on each claim. Return one label per claim, by "
        "index, as exactly `true` or `false`. Every claim gets exactly one label.\n\n"
        f"CLAIMS\n{listing}"
    )


async def classify(
    extractor: ClaimsExtractor, rubric: str, claims: List[str], task: str = "scope"
) -> List[str]:
    """One batched call. Anything the model omits or mislabels falls to the task's
    conservative label — a claim we cannot confidently call broad should not become a
    debate motion, and one we cannot confidently call checkable should not be flagged
    factual."""
    cfg = TASKS[task]
    # The shipped rubric sets a boolean, so `false` is the conservative answer for
    # both tasks: an unflagged claim is neither published as a motion nor sent to
    # be verified.
    fallback = cfg["negative"]
    by_label = {"true": cfg["positive"], "false": cfg["negative"]}
    text = await extractor._call_gemini(
        prompt=build_prompt(rubric, claims, task),
        config=extractor._config(ScopeLabels),
        step_name=f"{task} classification",
    )
    parsed = ScopeLabels.model_validate_json(text)
    out = [fallback] * len(claims)
    filled: set = set()
    for row in parsed.labels:
        mapped = by_label.get(str(row.label).strip().lower())
        if 0 <= row.index < len(claims) and mapped is not None:
            out[row.index] = mapped
            filled.add(row.index)
    recognised = len(filled)
    # A run where most answers fell through to the fallback is instrumentation
    # failure, not a finding. Say so loudly rather than reporting a clean collapse.
    if recognised < 0.8 * len(claims):
        seen = sorted({str(row.label) for row in parsed.labels})
        raise RuntimeError(
            f"{task}: only {recognised}/{len(claims)} claims got a label "
            f"(expected true/false, model returned {seen}) — the fallback would "
            f"have produced a uniform result"
        )
    return out


async def score_rubric(
    extractor: ClaimsExtractor, name: str, rubric: str, gold: List[Dict[str, Any]], runs: int,
    task: str = "scope",
) -> Dict[str, Any]:
    """Shuffle per run so a batch cannot be answered by balancing the classes, then
    take the majority label per claim across runs."""
    votes: List[List[str]] = [[] for _ in gold]
    accuracies: List[float] = []
    for run in range(runs):
        order = list(range(len(gold)))
        random.Random(1000 + run).shuffle(order)
        labels = await classify(extractor, rubric, [gold[i]["text"] for i in order], task)
        correct = 0
        for slot, gold_index in enumerate(order):
            votes[gold_index].append(labels[slot])
            if labels[slot] == gold[gold_index]["label"]:
                correct += 1
        accuracies.append(correct / len(gold))
        print(f"  {name} run {run + 1}: accuracy {100 * accuracies[-1]:.1f}%")

    positive = TASKS[task]["positive"]
    # Ties break toward the conservative label, and deterministically: `max` over a
    # set is at the mercy of per-process string hash randomisation.
    negative = TASKS[task]["negative"]
    majority = [positive if v.count(positive) > v.count(negative) else negative for v in votes]
    tp = sum(1 for g, m in zip(gold, majority) if g["label"] == positive and m == positive)
    fp = sum(1 for g, m in zip(gold, majority) if g["label"] != positive and m == positive)
    fn = sum(1 for g, m in zip(gold, majority) if g["label"] == positive and m != positive)
    tn = sum(1 for g, m in zip(gold, majority) if g["label"] != positive and m != positive)
    unstable = sum(1 for v in votes if len(set(v)) > 1)
    return {
        "rubric": name,
        "runs": runs,
        "accuracy_mean": round(statistics.mean(accuracies), 3),
        "accuracy_runs": [round(a, 3) for a in accuracies],
        # Precision on `debate` is the number that matters: a false positive publishes a
        # narrow fact as a debate motion, which is the failure this feature exists to stop.
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "unstable_claims": unstable,
        "majority": majority,
    }


async def cmd_gold(args: argparse.Namespace) -> None:
    gold = json.loads(Path(args.gold).read_text())
    extractor = ClaimsExtractor()
    results = []
    for name in args.rubric:
        print(f"\n=== [{args.task}] rubric {name} ({len(gold)} gold claims, {args.runs} runs) ===")
        results.append(
            await score_rubric(extractor, name, TASKS[args.task]["rubrics"][name], gold, args.runs, args.task)
        )

    for r in results:
        print(f"\n--- {r['rubric']}: accuracy {100 * r['accuracy_mean']:.1f}%  "
              f"precision {r['precision']}  recall {r['recall']}  "
              f"unstable {r['unstable_claims']}")
        print(f"    confusion {r['confusion']}")
        print("    disagreements (gold -> predicted):")
        for g, m in zip(gold, r["majority"]):
            if g["label"] != m:
                print(f"      [{g['label']:6} -> {m:6}] ({g['source']}) {g['text'][:96]}")

    if args.hard:
        hard = json.loads(Path(args.hard).read_text())
        labels = await classify(
            extractor, TASKS[args.task]["rubrics"][args.rubric[-1]], [h["text"] for h in hard], args.task
        )
        print("\n--- held-out borderline cases (no gold label; for judgement, not scoring)")
        for h, label in zip(hard, labels):
            print(f"      [{label:6}] {h['text'][:88]}  <- {h['note']}")

    if args.out:
        Path(args.out).write_text(json.dumps({"gold_size": len(gold), "results": results}, indent=1))
        print(f"\nwrote {args.out}")


async def cmd_pipeline(args: argparse.Namespace) -> None:
    """Extract from real debate transcripts, then classify every claim. Answers the
    question a rubric alone cannot: how much of what we generate is publishable as a
    debate motion, and what exactly gets dropped."""
    extractor = ClaimsExtractor()
    rubric = RUBRICS[args.rubric]
    report: List[Dict[str, Any]] = []

    for path in args.inputs:
        spec = json.loads(Path(path).read_text())
        inp = ClaimsExtractInput(**spec["payload"])
        # Both headline numbers are read off flags the payload has to ask for. A spec
        # that omits them yields all-null claims, which would print as a flawless
        # "0 over-flagged" instead of "this run measured nothing".
        missing = [
            name
            for name, on in (
                ("classify_factuality", inp.classify_factuality),
                ("classify_contestability", inp.classify_contestability),
            )
            if not on
        ]
        if missing:
            raise RuntimeError(f"{path}: payload does not request {', '.join(missing)} — nothing to measure")
        # The prompt is always built flat (no topics pass here), so assemble must be
        # told the same thing or it would run the grouping path over a flat answer.
        inp = inp.model_copy(update={"grouping": False})
        extraction = await extractor.extract_claims(build_extract_prompt(inp, []))
        result = assemble_result(inp, extraction.model_dump(), [], model_used=settings.claims_extract_model)
        claims = [{"text": c.text, "is_factual": c.is_factual,
                   "prompt_contestable": c.is_contestable} for c in result.claims]
        labels = await classify(extractor, rubric, [c["text"] for c in claims]) if claims else []
        # Second opinion on factuality: what the strict rubric would flag, next to what
        # the extraction prompt itself flagged. The gap is the size of the prompt change.
        facts = await classify(extractor, FACT_RUBRICS[args.fact_rubric],
                               [c["text"] for c in claims], "factual") if claims else []
        for c, label, fact in zip(claims, labels, facts):
            c["scope"] = label
            c["strict_factual"] = fact == "factual"
        kept = [c for c in claims if c["scope"] == "debate"]
        report.append({"name": spec["name"], "motion": spec.get("motion"), "claims": claims,
                       "kept": len(kept), "total": len(claims),
                       "shape_all": shape_metrics([c["text"] for c in claims]),
                       "shape_kept": shape_metrics([c["text"] for c in kept])})
        print(f"  {spec['name']}: {len(kept)}/{len(claims)} survive the debate filter")

    print("\n=== corpus profile (target = Geo's 1,378 Debate-tagged claims) ===")
    everything = [c["text"] for r in report for c in r["claims"]]
    if not everything:
        print("  no claims extracted — nothing to profile")
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1))
        return
    survivors = [c["text"] for r in report for c in r["claims"] if c["scope"] == "debate"]
    allm, keptm = shape_metrics(everything), shape_metrics(survivors)
    print(f"  {'metric':22} {'target':>8} {'extracted':>10} {'survivors':>10}")
    for k, target in CORPUS_PROFILE.items():
        print(f"  {k:22} {target:>8} {allm.get(k, 0):>10} {keptm.get(k, 0):>10}")

    total = sum(r["total"] for r in report)
    kept = sum(r["kept"] for r in report)
    from collections import Counter
    fact_all = Counter(str(c["is_factual"]) for r in report for c in r["claims"])
    fact_kept = Counter(str(c["is_factual"]) for r in report for c in r["claims"] if c["scope"] == "debate")
    print(f"\n  survival: {kept}/{total} ({100 * kept / total:.0f}%)" if total else "\n  survival: 0/0")
    print(f"  is_factual all       : {dict(fact_all)}")
    print(f"  is_factual survivors : {dict(fact_kept)}   (Geo corpus: 4% of flagged are true)")
    surv = [c for r in report for c in r["claims"] if c["scope"] == "debate"]
    strict_true = sum(1 for c in surv if c["strict_factual"])
    prompt_true = sum(1 for c in surv if c["is_factual"])
    flipped = [c for c in surv if c["is_factual"] and not c["strict_factual"]]
    # Does the rubric still hold once embedded in the full extraction prompt, rather
    # than run standalone over fixed text? Agreement is a SIGNAL, not a score: the
    # standalone classifier was validated on a hand-judged gold set, which is a
    # different distribution from raw extraction output, and on inspection it is the
    # one that errs on some disputes (it misses rule 3 comparatives). Read the
    # disagreements and judge them; do not treat either side as ground truth.
    scored = [c for r in report for c in r["claims"] if c["prompt_contestable"] is not None]
    if not scored:
        print("\n  no claim carried an in-prompt contestability flag — skipping agreement")
    if scored:
        agree = sum(1 for c in scored if c["prompt_contestable"] == (c["scope"] == "debate"))
        pc = sum(1 for c in scored if c["prompt_contestable"])
        sc = sum(1 for c in scored if c["scope"] == "debate")
        print(f"\n  contestable — in-prompt: {pc}/{len(scored)}  standalone rubric: {sc}/{len(scored)}"
              f"  agreement {100 * agree / len(scored):.0f}%")
        for c in scored:
            if c["prompt_contestable"] != (c["scope"] == "debate"):
                print(f"      in-prompt={c['prompt_contestable']!s:5} standalone={c['scope']:6} {c['text'][:82]}")
    if not surv:
        print("\n  nothing survived the contestability filter — no factuality comparison")
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1))
        return
    print(f"\n  survivors flagged factual — extraction prompt: {prompt_true}/{len(surv)} "
          f"({100 * prompt_true / len(surv):.0f}%)")
    print(f"  survivors flagged factual — strict rubric:     {strict_true}/{len(surv)} "
          f"({100 * strict_true / len(surv):.0f}%)")
    print(f"  the prompt over-flags {len(flipped)} of them:")
    for c in flipped[:14]:
        print(f"      {c['text'][:104]}")

    print("\n=== DROPPED as `simple` ===")
    for r in report:
        for c in r["claims"]:
            if c["scope"] != "debate":
                print(f"  [{str(c['is_factual'])[:5]:5}] {c['text'][:110]}")
    print("\n=== KEPT as `debate` ===")
    for r in report:
        for c in r["claims"]:
            if c["scope"] == "debate":
                print(f"  [{str(c['is_factual'])[:5]:5}] {c['text'][:110]}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gold")
    # Defaults resolve per task after parsing: the two gold sets use disjoint label
    # vocabularies, so one shared default would silently grade the wrong file.
    g.add_argument("--gold", default=None)
    g.add_argument("--hard", default=None)
    g.add_argument("--task", default="scope", choices=sorted(TASKS))
    g.add_argument("--rubric", action="append", default=None)
    g.add_argument("--runs", type=int, default=3)
    g.add_argument("--out", default=None)
    r = sub.add_parser("pipeline")
    r.add_argument("inputs", nargs="+")
    r.add_argument("--rubric", default="v1", choices=sorted(RUBRICS))
    r.add_argument("--fact-rubric", dest="fact_rubric", default="strict_v2", choices=sorted(FACT_RUBRICS))
    r.add_argument("--out", default=None)

    args = p.parse_args(argv)
    if args.cmd == "pipeline":
        print(f"model {settings.claims_extract_model} @ temp {settings.claims_extract_temperature}")
        asyncio.run(cmd_pipeline(args))
        return
    cfg = TASKS[args.task]
    args.gold = args.gold or str(cfg["gold"])
    args.rubric = args.rubric or [cfg["default_rubric"]]
    unknown = [name for name in args.rubric if name not in cfg["rubrics"]]
    if unknown:
        p.error(f"unknown {args.task} rubric(s) {unknown}; choose from {sorted(cfg['rubrics'])}")
    print(f"model {settings.claims_extract_model} @ temp {settings.claims_extract_temperature}")
    asyncio.run(cmd_gold(args))


if __name__ == "__main__":
    main(sys.argv[1:])
