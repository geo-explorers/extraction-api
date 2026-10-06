"""Score claims.score_highlights against tests/fixtures/claim_highlights_eval.json.

Live:    PERPLEXITY_API_KEY=… uv run python scripts/eval_claim_highlights.py [--debates N] [--out results.json]
No key:  uv run python scripts/eval_claim_highlights.py --dry-run   (prints the first request, sends nothing)
Table:   uv run python scripts/eval_claim_highlights.py --table 12 --from results.json
         (one debate as a Markdown table: every claim verbatim, the score, the reader's label)

Runs the task's own code path (same questions, same content, same score) over each labelled
debate, then reports how well the score separates the claims a reader selected as highlights
from the ones they left out. The labels are one reader's and preliminary: read the
disagreements before treating them as errors.
"""

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.schemas.claims_score_highlights_schema import ClaimsScoreHighlightsInput  # noqa: E402
from src.config.settings import settings  # noqa: E402
from src.decisions import get_decision_model, provider_names  # noqa: E402
from src.pipeline.claims_score_highlights_core import (  # noqa: E402
    DECISIONS,
    assemble_result,
    attribution,
    build_content,
    build_questions,
    decide_claims,
)

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "claim_highlights_eval.json"


def task_input(debate: dict) -> ClaimsScoreHighlightsInput:
    return ClaimsScoreHighlightsInput(
        media_type="debate",
        title=debate["title"],
        context=debate["context"],
        documents=debate["documents"],
        claims=[{"text": c["text"], "document_indices": c["document_indices"]} for c in debate["claims"]],
    )


def auc(positives: list[float], negatives: list[float]) -> float:
    """Probability that a selected claim outscores an unselected one (ties count half)."""
    if not positives or not negatives:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def dry_run(debates: list[dict]) -> int:
    first = task_input(debates[0])
    content, questions = build_content(first, 0), build_questions("debate")
    print(json.dumps({"content": content, "questions": {k: dataclasses.asdict(q) for k, q in questions.items()}}, indent=1))
    requests = sum(len(d["claims"]) for d in debates)
    chars = sum(
        len(json.dumps(build_content(task_input(d), i))) + len(json.dumps({k: dataclasses.asdict(q) for k, q in questions.items()}))
        for d in debates
        for i in range(len(d["claims"]))
    )
    print(f"\n{len(debates)} debates, {requests} requests, about {chars // 4:,} input tokens in all (chars/4)", file=sys.stderr)
    return 0


def reference(row: dict) -> str:
    """The reader's verdict in one word: the job they named, or why they left the claim out."""
    return row["gold_job"] if row["label"] == "selected" else row["label"].replace("_", " ")


def table(debate: dict, rows: list[dict], number: int) -> str:
    """One debate as Markdown: claims verbatim in API order, each decision asked, the score, and
    the reader's reference. `rows` are this debate's scored rows in the same order. When the score
    is the only decision, one column stands for both."""
    inp = task_input(debate)
    decisions = [d for d in DECISIONS if any(d in r["decisions"] for r in rows)]
    show_decisions = len(decisions) > 1
    header = ["#", "Speaker", "Claim", *(d.title() for d in decisions if show_decisions), "Score", "Reference"]
    lines = [
        f"**{number}. {debate['title']}** — {debate['context']}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row, claim in zip(rows, inp.claims):
        who = attribution(claim, inp.documents)
        text = row["text"].replace("|", "\\|")
        if row["score"] is None:
            cells = ["—"] * (len(decisions) * show_decisions + 1)
        else:
            cells = [f"{row['decisions'][d]:.2f}" for d in decisions if show_decisions]
            cells.append(f"**{row['score']:.2f}**")
        lines.append(f"| {row['index'] + 1} | {who} | {text} | " + " | ".join(cells) + f" | {reference(row)} |")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--debates", type=int, default=None, help="score only the first N debates")
    parser.add_argument("--provider", default=settings.claims_highlights_provider, choices=provider_names())
    parser.add_argument("--model", default=settings.claims_highlights_model)
    parser.add_argument("--out", type=Path, default=None, help="write every claim's label and scores as JSON")
    parser.add_argument("--dry-run", action="store_true", help="print the first request and exit")
    parser.add_argument("--table", type=int, default=None, metavar="N", help="print debate N as a Markdown table")
    parser.add_argument("--from", dest="source", type=Path, default=None, help="with --table: read scores from this --out file instead of calling the provider")
    args = parser.parse_args()

    all_debates = json.loads(FIXTURE.read_text())["debates"]
    debates = all_debates[: args.debates]
    if args.dry_run:
        return dry_run(debates)

    if args.table is not None and args.source is not None:
        rows = [r for r in json.loads(args.source.read_text()) if r["debate"] == args.table]
        if not rows:
            print(f"no rows for debate {args.table} in {args.source}", file=sys.stderr)
            return 1
        print(table(all_debates[args.table - 1], sorted(rows, key=lambda r: r["index"]), args.table))
        return 0

    if args.table is not None:
        debates = [all_debates[args.table - 1]]

    decider = get_decision_model(args.provider)
    print(f"provider={args.provider} model={args.model} debates={len(debates)}")
    rows: list[dict] = []
    precision_at_k: list[float] = []
    loose_precision_at_k: list[float] = []
    started = time.time()
    for debate in debates:
        number = all_debates.index(debate) + 1
        inp = task_input(debate)
        result = assemble_result(inp, await decide_claims(inp, decider, args.model), args.provider, args.model)
        scored = []
        for gold, claim in zip(debate["claims"], result.claims):
            row = {"debate": number, "label": gold["label"], "gold_job": gold.get("job"), **claim.model_dump()}
            rows.append(row)
            if claim.score is not None:
                scored.append(row)
        k = sum(1 for r in scored if r["label"] == "selected")
        top = sorted(scored, key=lambda r: r["score"], reverse=True)[:k]
        if k:
            precision_at_k.append(sum(r["label"] == "selected" for r in top) / k)
            loose_precision_at_k.append(sum(r["label"] in ("selected", "runner_up") for r in top) / k)
        print(
            f"{number:>2} {debate['title'][:58]:<58} claims={len(debate['claims']):>2} scored={len(scored):>2} "
            f"selected={k} top-{k} hit={sum(r['label'] == 'selected' for r in top)}"
        )
        for r in top:
            mark = "ok  " if r["label"] == "selected" else "near" if r["label"] == "runner_up" else "MISS"
            print(f"     {mark} {r['score']:.2f} [{r['label']}] {r['text'][:90]}")

    if args.table is not None:
        print()
        print(table(debates[0], [r for r in rows if r["debate"] == args.table], args.table))

    scored = [r for r in rows if r["score"] is not None]
    selected = [r for r in scored if r["label"] == "selected"]
    rest = [r for r in scored if r["label"] != "selected"]
    by_label = defaultdict(list)
    for r in scored:
        by_label[r["label"]].append(r)

    print(f"\n{len(scored)}/{len(rows)} claims scored in {time.time() - started:.0f}s")
    print(f"AUC, selected vs the rest: {auc([r['score'] for r in selected], [r['score'] for r in rest]):.3f}")
    if len(DECISIONS) > 1:
        for decision in DECISIONS:
            print(
                f"  {decision} alone: "
                f"{auc([r['decisions'][decision] for r in selected], [r['decisions'][decision] for r in rest]):.3f}"
            )
    if precision_at_k:
        print(
            f"precision at k (k = the reader's picks per debate): {sum(precision_at_k) / len(precision_at_k):.3f}; "
            f"counting runner-ups as hits: {sum(loose_precision_at_k) / len(loose_precision_at_k):.3f}"
        )
    print("\nlabel                  n   mean score   median    min    max")
    for label, group in sorted(by_label.items(), key=lambda kv: -sum(r["score"] for r in kv[1]) / len(kv[1])):
        scores = sorted(r["score"] for r in group)
        print(
            f"{label:<20} {len(group):>3}   {sum(scores) / len(scores):>10.3f}   {scores[len(scores) // 2]:>6.2f}  "
            f"{scores[0]:>5.2f}  {scores[-1]:>5.2f}"
        )
    for gold in ("position", "clash", "turn"):
        group = [r for r in selected if r["gold_job"] == gold]
        if group:
            print(f"  reader's {gold:<9} (n={len(group):>2}): mean score {sum(r['score'] for r in group) / len(group):.2f}")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
