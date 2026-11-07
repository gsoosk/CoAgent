#!/usr/bin/env bash
# Reproduce every experiment and figure.
# Makes NO network calls: every agent here is a scripted plan.
# The real-LLM path is scripts/run_llm_matrix.py, run separately.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== tests ===";            python3 -m unittest discover -s tests -q
echo; echo "=== matrix (Fig 5) ===";      python3 scripts/run_matrix.py --trials 40 --out results/matrix_n40.csv
echo; echo "=== case study (Fig 6) ===";  python3 scripts/case_study.py
echo; echo "=== livelock (Fig 2) ===";    python3 scripts/livelock.py
echo; echo "=== cold start (Fig 7a) ==="; python3 scripts/toolsmith_coldstart.py
echo; echo "=== figures ===";             python3 scripts/make_figures.py --csv results/matrix_n40.csv
# plots an earlier real-LLM run if one exists; never calls an API itself
python3 scripts/make_llm_figures.py 2>/dev/null || echo "(no LLM runs recorded yet)"
