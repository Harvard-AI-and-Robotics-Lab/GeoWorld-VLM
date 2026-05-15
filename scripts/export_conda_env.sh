#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-environment.yml}"

conda env export --no-builds > "$OUT"
echo "Wrote $OUT"
echo "For a leaner dependency list, also run:"
echo "  pip freeze > requirements-freeze.txt"

