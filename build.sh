#!/bin/sh
# Rebuild everything from the raw results, then the PDF.
#
# analyze.py regenerates paper/tables/macros.tex and verify_numbers.py appends
# the \NVerifyChecks macro the paper cites, so the paper cannot be built until
# the verifier has run and passed. set -e stops the build at the first failure.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

if ! command -v pdflatex >/dev/null 2>&1; then
  PATH="$PATH:$HOME/Library/TinyTeX/bin/universal-darwin"
  export PATH
fi
command -v pdflatex >/dev/null 2>&1 || {
  echo "pdflatex not found. Install TinyTeX, MacTeX, or TeX Live." >&2
  exit 1
}

$PY src/analyze.py
$PY src/make_figures.py
if [ -f src/make_diagrams.py ]; then $PY src/make_diagrams.py; fi
sh paper/figures/render_html_figs.sh
$PY src/check_figures.py
$PY src/test_checks.py
$PY src/analyze.py
$PY src/verify_numbers.py

cd paper
pdflatex -interaction=nonstopmode Paper.tex >/dev/null
bibtex Paper >/dev/null
pdflatex -interaction=nonstopmode Paper.tex >/dev/null
pdflatex -interaction=nonstopmode Paper.tex >/dev/null

if grep -qiE 'Warning.*(undefined|Citation|Reference)' Paper.log; then
  echo "build produced reference warnings:" >&2
  grep -iE 'Warning.*(undefined|Citation|Reference)' Paper.log >&2
  exit 1
fi
echo "built $(pwd)/Paper.pdf"
