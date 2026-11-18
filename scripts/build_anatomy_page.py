#!/usr/bin/env python3
"""Turn results/anatomy/*.json into the explorable pages.

Two pages from one template:

  designed    results/anatomy/anatomy.json
              -> chopping_anatomy.html
  benchmark   results/anatomy/benchmark_<model>.json (one per model)
              -> chopping_anatomy_benchmark.html

Each page is written twice: a standalone document for opening locally, and
(with --body DIR) the bare body the Artifact publisher wraps itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "anatomy"
SHORT = {"claude-haiku-4-5": "Haiku", "claude-sonnet-5": "Sonnet"}


def _observed(log: List[Dict[str, Any]]) -> str:
    revs = [e["decision"] for e in log if e["ev"] == "revision"]
    if revs:
        return revs[0]
    declared = next(e for e in log if e["ev"] == "declared")["plan"]
    return "certified as declared" if declared["certifies"] else "merged at launch"


def compact_run(log: List[Dict[str, Any]], workflows: Dict[str, Any],
                oracle: List[Dict[str, Any]], **meta: Any) -> Dict[str, Any]:
    """One run, reduced to what the page draws."""
    return {
        **meta,
        "names": {w: "AB"[i] for i, w in enumerate(workflows)},
        "observed": _observed(log),
        "declared": next(e for e in log if e["ev"] == "declared")["plan"],
        "certified": next(e for e in log if e["ev"] == "certified")["plan"],
        "merges": next(e for e in log if e["ev"] == "certified")["merges"],
        "revisions": [e for e in log if e["ev"] == "revision"],
        "events": [{k: e.get(k) for k in
                    ("t", "ev", "wf", "op", "obj", "refused")}
                   for e in log
                   if e["ev"] in ("call", "blocked", "think", "done")],
        "workflows": workflows,
        "oracle": oracle,
    }


def compact(files: List[Path], suite: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"suite": suite, "models": [], "window": [],
                           "scenarios": []}
    by_key: Dict[str, Dict[str, Any]] = {}
    for f in files:
        d = json.loads(f.read_text())
        out["models"].append(d["model"])
        out["window"] = out["window"] or d.get("window", [])
        for s in d["scenarios"]:
            sc = by_key.get(s["key"])
            if sc is None:
                sc = by_key[s["key"]] = {
                    **{k: s.get(k) for k in ("key", "title", "story",
                                             "expect", "suite", "tasks")},
                    "runs": [compact_run(
                        s["scripted"]["log"], s["scripted"]["workflows"],
                        s["oracle"], kind="scripted", label="scripted")],
                }
                out["scenarios"].append(sc)
            short = SHORT.get(d["model"], d["model"])
            for r in s["llm"]:
                sc["runs"].append(compact_run(
                    r["log"], r["workflows"], r["oracle"], kind="llm",
                    model=short, seed=r["seed"],
                    label=f"{short} {r['seed'] + 1}"))
    return out


def write_page(data: Dict[str, Any], name: str, title: str,
               body_dir: Path | None) -> None:
    body = (ROOT / "scripts" / "anatomy_template.html").read_text().replace(
        "__TITLE__", title).replace(
        "__DATA__", json.dumps(data, separators=(",", ":"), default=str))
    standalone = OUT / f"{name}.html"
    standalone.write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "</head><body>" + body + "</body></html>")
    print(f"wrote {standalone} ({len(body) // 1024} KB)")
    if body_dir is not None:
        body_dir.mkdir(parents=True, exist_ok=True)
        (body_dir / f"{name}.html").write_text(body)
        print("wrote", body_dir / f"{name}.html")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", type=Path, default=None,
                    help="also write bare bodies here, for publishing")
    args = ap.parse_args()

    designed = OUT / "anatomy.json"
    if designed.exists():
        write_page(compact([designed], "designed"), "chopping_anatomy",
                   "Chopping Anatomy", args.body)
    bench = sorted(OUT.glob("benchmark_*.json"))
    if bench:
        write_page(compact(bench, "benchmark"), "chopping_anatomy_benchmark",
                   "Chopping On Benchmarks", args.body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
