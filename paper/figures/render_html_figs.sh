#!/bin/sh
# Render the HTML+SVG concept diagrams to one-page PDFs.
set -e
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DIR=$(cd "$(dirname "$0")" && pwd)
for f in fig_placements fig_frameworks fig_protocol fig_studio; do
  "$CHROME" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$DIR/$f.pdf" "file://$DIR/$f.html" 2>/dev/null
  echo "  wrote $DIR/$f.pdf"
done
