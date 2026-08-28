"""Offline A/B eval for the claims.extract prompt — no Hatchet, no API.

Runs the DAG's pure path (build_extract_prompt -> ClaimsExtractor.extract_claims
-> assemble_result) over saved payloads N times per input and records the
verbatim claims, so a prompt change can be scored against the deployed prompt
on the same material before it ships.

    uv run python -m src.cli.eval_claims_extract run \
        --label baseline --runs 3 --out results/baseline.json inputs/*.json
    uv run python -m src.cli.eval_claims_extract compare \
        results/baseline.json results/candidate.json [--all-runs] [--md report.md]

Input file shape: {"name": str, "speakers": [str], "payload": ClaimsExtractInput}.
Each run costs one Gemini call on the configured claims model.
"""

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from src.api.schemas.claims_extract_schema import ClaimsExtractInput
from src.config.settings import settings
from src.extraction.claims_extractor import ClaimsExtractor
from src.extraction.claims_prompt_builder import build_extract_prompt
from src.pipeline.claims_extract_core import assemble_result

REPORTING_VERBS = (
    r"(?:said|says|argued?|argues|questioned|questions|suggested|suggests|"
    r"claimed|claims|stated|states|noted|notes|asked|asks|wondered|wonders|"
    r"pointed out|believes|thinks|contended|contends|conceded|concedes|"
    r"proposed|proposes|maintained|asserted|asserts|countered|counters|"
    r"expressed|acknowledged|disputed|raised|highlighted|emphasized|explained)"
)


def _git_rev() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], capture_output=True).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _name_patterns(speakers: List[str]) -> List[re.Pattern]:
    names = set()
    for s in speakers:
        s = (s or "").strip()
        if not s:
            continue
        names.add(s)
        names.add(s.split()[0])  # first name alone ("Preston questioned...")
    return [re.compile(rf"\b{re.escape(n)}\b", re.IGNORECASE) for n in sorted(names, key=len, reverse=True)]


def score_claim(text: str, patterns: List[re.Pattern]) -> Dict[str, bool]:
    mentions = any(p.search(text) for p in patterns)
    speech_act = mentions and re.search(REPORTING_VERBS, text, re.IGNORECASE) is not None
    return {"name_mention": mentions, "speech_act": speech_act}


def run_metrics(claims: List[Dict[str, Any]], speakers: List[str]) -> Dict[str, Any]:
    patterns = _name_patterns(speakers)
    flags = [score_claim(c["text"], patterns) for c in claims]
    n = len(claims)
    n_false = sum(1 for c in claims if c["is_factual"] is False)
    n_null = sum(1 for c in claims if c["is_factual"] is None)
    return {
        "claims": n,
        "true": n - n_false - n_null,
        "false": n_false,
        "null": n_null,
        "false_share": round(n_false / n, 2) if n else None,
        "speech_acts": sum(1 for f in flags if f["speech_act"]),
        "name_mentions": sum(1 for f in flags if f["name_mention"]),
        "avg_words": round(statistics.mean(len(c["text"].split()) for c in claims), 1) if n else None,
    }


async def _one_run(extractor: ClaimsExtractor, inp: ClaimsExtractInput, prompt: str, sem: asyncio.Semaphore) -> Dict[str, Any]:
    async with sem:
        t0 = time.time()
        extraction = await extractor.extract_claims(prompt)
        result = assemble_result(inp, extraction.model_dump(), [], model_used=settings.claims_extract_model)
        return {
            "elapsed_s": round(time.time() - t0, 1),
            "claims": [
                {
                    "text": c.text,
                    "is_factual": c.is_factual,
                    "document_indices": c.document_indices,
                    "confidence": c.confidence,
                }
                for c in result.claims
            ],
        }


async def cmd_run(args: argparse.Namespace) -> None:
    extractor = ClaimsExtractor()
    sem = asyncio.Semaphore(args.concurrency)
    inputs = []
    for path in args.inputs:
        spec = json.loads(Path(path).read_text())
        inp = ClaimsExtractInput(**spec["payload"])
        prompt = build_extract_prompt(inp, [])
        inputs.append((spec, inp, prompt))
    tasks = {
        (spec["name"], i): asyncio.create_task(_one_run(extractor, inp, prompt, sem))
        for spec, inp, prompt in inputs
        for i in range(args.runs)
    }
    await asyncio.gather(*tasks.values())
    out = {
        "label": args.label,
        "git": _git_rev(),
        "model": settings.claims_extract_model,
        "temperature": settings.claims_extract_temperature,
        "runs_per_input": args.runs,
        "inputs": [],
    }
    for spec, inp, prompt in inputs:
        runs = [tasks[(spec["name"], i)].result() for i in range(args.runs)]
        for r in runs:
            r["metrics"] = run_metrics(r["claims"], spec.get("speakers", []))
        out["inputs"].append(
            {
                "name": spec["name"],
                "speakers": spec.get("speakers", []),
                "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:12],
                "prompt_chars": len(prompt),
                "runs": runs,
            }
        )
        print(f"{spec['name']}: " + " | ".join(
            f"run{i+1} {r['metrics']['claims']}c {r['metrics']['false']}F {r['metrics']['speech_acts']}SA {r['elapsed_s']}s"
            for i, r in enumerate(runs)
        ))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out} ({out['git']}, {out['model']})")


def _fmt_claim(c: Dict[str, Any], patterns: List[re.Pattern]) -> str:
    flag = score_claim(c["text"], patterns)
    tag = {True: "T", False: "F", None: "-"}[c["is_factual"]]
    mark = " ⚠SA" if flag["speech_act"] else (" ⚠name" if flag["name_mention"] else "")
    return f"- [{tag}] d{c['document_indices']} {c['text']}{mark}"


def cmd_compare(args: argparse.Namespace) -> None:
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    lines: List[str] = []
    w = lines.append
    w(f"# claims.extract A/B — `{a['label']}` ({a['git']}) vs `{b['label']}` ({b['git']})")
    w(f"model {a['model']} @ temp {a['temperature']}; {a['runs_per_input']} runs per input\n")
    by_name_b = {i["name"]: i for i in b["inputs"]}
    for ia in a["inputs"]:
        ib = by_name_b.get(ia["name"])
        if not ib:
            continue
        patterns = _name_patterns(ia["speakers"])
        w(f"## {ia['name']}  (speakers: {', '.join(ia['speakers'])})\n")
        w("| variant | run | claims | true | false | false share | speech acts | name mentions | avg words |")
        w("|---|---|---|---|---|---|---|---|---|")
        for label, item in ((a["label"], ia), (b["label"], ib)):
            for i, r in enumerate(item["runs"]):
                m = r["metrics"]
                w(f"| {label} | {i+1} | {m['claims']} | {m['true']} | {m['false']} | {m['false_share']} | {m['speech_acts']} | {m['name_mentions']} | {m['avg_words']} |")
        w("")
        run_idx = range(a["runs_per_input"]) if args.all_runs else [0]
        for i in run_idx:
            w(f"### run {i+1} — {a['label']}")
            for c in ia["runs"][i]["claims"]:
                w(_fmt_claim(c, patterns))
            w(f"\n### run {i+1} — {b['label']}")
            for c in ib["runs"][i]["claims"]:
                w(_fmt_claim(c, patterns))
            w("")
    report = "\n".join(lines)
    print(report)
    if args.md:
        Path(args.md).write_text(report)


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("inputs", nargs="+")
    r.add_argument("--label", required=True)
    r.add_argument("--runs", type=int, default=3)
    r.add_argument("--concurrency", type=int, default=3)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--all-runs", action="store_true")
    c.add_argument("--md")
    args = p.parse_args(argv)
    if args.cmd == "run":
        asyncio.run(cmd_run(args))
    else:
        cmd_compare(args)


if __name__ == "__main__":
    main(sys.argv[1:])
