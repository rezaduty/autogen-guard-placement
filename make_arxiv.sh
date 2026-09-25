#!/bin/sh
# Build the arXiv source tarball.
#
# arXiv rejects a PDF-only submission and runs its own TeX pass over the
# source, so the tarball must contain the source and must NOT contain the
# built PDF. Four things that make that pass fail are handled here:
#
#   1. the built PDF and the aux/log files are excluded, so arXiv cannot
#      mistake the PDF for the submission
#   2. Paper.bbl is included, so arXiv does not need to run bibtex
#   3. IEEEtran.cls and orcidlink.sty are bundled, in case the installed
#      TeX Live on arXiv differs from the one used locally
#   4. only .pdf figures are copied, since mixing .eps and .pdf in one
#      document breaks the automated pass
#
# The result is compiled in a clean directory before the tarball is written,
# using only the files that will be uploaded.
set -e
cd "$(dirname "$0")"

if ! command -v pdflatex >/dev/null 2>&1; then
  PATH="$PATH:$HOME/Library/TinyTeX/bin/universal-darwin"
  export PATH
fi

NAME=title36-autogen-guard-placement_arxiv
OUT=../arxiv_uploads/$NAME
mkdir -p "$OUT/tables" "$OUT/figures"
rm -rf "${OUT:?}"/*
mkdir -p "$OUT/tables" "$OUT/figures"

# Source, generated tables, and the bibliography as already-resolved .bbl.
cp paper/Paper.tex "$OUT/"
cp paper/Paper.bbl "$OUT/"
# Generated tables and the verbatim listings the appendices read.
cp paper/tables/*.tex paper/tables/*.json paper/tables/*.txt paper/tables/*.yaml "$OUT/tables/"

# Figures: PDF only. The .html sources and the render script are build inputs,
# not figures, and arXiv has no use for them.
cp paper/figures/*.pdf "$OUT/figures/"

# Bundle the class and package files that may not match arXiv's TeX Live.
for f in IEEEtran.cls orcidlink.sty; do
  src=$(kpsewhich "$f")
  [ -n "$src" ] && cp "$src" "$OUT/"
done

# Compile in a pristine copy containing only what will be uploaded, so a file
# resolved from the local TeX tree rather than the tarball is caught here.
TMP=$(mktemp -d)
cp -R "$OUT"/. "$TMP"/
# Three passes: with many floats the labels need a third pass to settle.
( cd "$TMP" && pdflatex -interaction=nonstopmode Paper.tex >/dev/null 2>&1 \
            && pdflatex -interaction=nonstopmode Paper.tex >/dev/null 2>&1 \
            && pdflatex -interaction=nonstopmode Paper.tex >/dev/null 2>&1 )
if [ ! -f "$TMP/Paper.pdf" ]; then
  echo "clean-room build FAILED, tarball not written" >&2
  grep -E '^!' "$TMP/Paper.log" | head -20 >&2
  rm -rf "$TMP"
  exit 1
fi
PAGES=$(grep -oE 'Output written on Paper\.pdf \([0-9]+ pages' "$TMP/Paper.log" \
        | grep -oE '[0-9]+' | head -1)
# Only warnings that mean something is undefined. "Label(s) may have changed"
# mentions cross-references too but is not a missing reference.
UNDEF=$(grep -ciE 'Warning: (Reference|Citation) .* undefined|There were undefined references' "$TMP/Paper.log" || true)
rm -rf "$TMP"
if [ "$UNDEF" != "0" ]; then
  echo "clean-room build has $UNDEF undefined references, tarball not written" >&2
  exit 1
fi

# The tarball must not carry the built PDF or any aux file.
( cd ../arxiv_uploads && tar --disable-copyfile \
    --exclude='.DS_Store' --exclude='*.aux' --exclude='*.log' \
    --exclude='*.out' --exclude='*.blg' --exclude='Paper.pdf' \
    -czf "$NAME.tar.gz" "$NAME" )

echo "clean-room build: $PAGES pages, $UNDEF undefined references"
echo "wrote ../arxiv_uploads/$NAME.tar.gz"
tar -tzf "../arxiv_uploads/$NAME.tar.gz" | sed 's/^/  /'
