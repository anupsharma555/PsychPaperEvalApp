#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PDF_PATH="${1:-$ROOT/test/sharma-et-al-2017-common-dimensional-reward-deficits-across-mood-and-psychotic-disorders-a-connectome-wide-association.pdf}"
REFERENCE_MD="${2:-$ROOT/test/text/sharma_2017_chatgpt_extraction.md}"
OUT_DIR="${3:-$ROOT/test/text}"
DB_PATH="${4:-/tmp/paper_eval_compare_local.db}"

MODE="${MODE:-auto}"
PARSER_ENGINE="${PARSER_ENGINE:-validated}"
BACKEND_PROFILE="${BACKEND_PROFILE:-fast}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$PDF_PATH" ]]; then
  echo "PDF not found: $PDF_PATH" >&2
  exit 1
fi

if [[ ! -f "$REFERENCE_MD" ]]; then
  echo "Reference markdown not found: $REFERENCE_MD" >&2
  exit 1
fi

PARSER_ENGINE="$PARSER_ENGINE" \
  .venv/bin/python scripts/compare_pdf_against_reference.py \
    --mode "$MODE" \
    --parser-engine "$PARSER_ENGINE" \
    --backend-profile "$BACKEND_PROFILE" \
    --pdf "$PDF_PATH" \
    --reference-md "$REFERENCE_MD" \
    --out-dir "$OUT_DIR" \
    --db-path "$DB_PATH"
