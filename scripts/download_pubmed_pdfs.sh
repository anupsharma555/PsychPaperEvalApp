#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: download_pubmed_pdfs.sh [--count N] [--output-dir DIR] [--from-date YYYY-MM-DD]

Downloads additional PubMed-linked, PMC Open Access PDFs into the test folder.
Safe to rerun: existing PMCIDs in metadata are skipped.
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

COUNT=10
OUT_DIR="$APP_DIR/test"
FROM_DATE="2025-01-01"

while [ $# -gt 0 ]; do
  case "$1" in
    --count)
      COUNT="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --from-date)
      FROM_DATE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [ "$COUNT" -lt 1 ]; then
  echo "--count must be a positive integer" >&2
  exit 1
fi

need_cmd curl
need_cmd jq
need_cmd rg
need_cmd tar
need_cmd file
need_cmd perl
need_cmd awk

META_FILE="$OUT_DIR/metadata.tsv"
SEEN_PMCIDS_FILE="$OUT_DIR/.seen_pmcids.txt"
SEEN_JOURNALS_FILE="$OUT_DIR/.seen_journals.txt"

mkdir -p "$OUT_DIR"

if [ ! -f "$META_FILE" ]; then
  printf 'index\ttopic\tpmid\tpmcid\tyear\tjournal\ttitle\tpubmed_url\tpmc_url\tpdf_file\n' > "$META_FILE"
fi

awk -F'\t' 'NR>1 && $4 != "" {print $4}' "$META_FILE" | sort -u > "$SEEN_PMCIDS_FILE"
awk -F'\t' 'NR>1 && $6 != "" {print $6}' "$META_FILE" | sort -u > "$SEEN_JOURNALS_FILE"

current_index=$(awk -F'\t' 'NR>1 && $1 ~ /^[0-9]+$/ {if ($1>max) max=$1} END {print max+1}' "$META_FILE")
[ -n "$current_index" ] || current_index=1

added=0

parse_records() {
  perl -0777 -ne 'while(/<record id="(PMC\d+)" citation="([^"]+)" license="[^"]*" retracted="([^"]+)"><link format="(pdf|tgz)"[^>]*href="([^"]+)"/sg){print "$1\t$2\t$3\t$4\t$5\n";}'
}

next_feed_url() {
  perl -0777 -ne 'if(/<resumption><link token="[^"]+" href="([^"]+)"/s){print $1;}'
}

download_record() {
  local pmcid="$1"
  local citation="$2"
  local format="$3"
  local href="$4"
  local journal pmid summary title year pmc_url base out_pdf url pkg pdf_in_tar

  if grep -Fxq "$pmcid" "$SEEN_PMCIDS_FILE"; then
    return 1
  fi

  journal="$(printf '%s' "$citation" | sed -E 's/ [0-9]{4}.*$//' | sed -E 's/[[:space:]]+$//')"
  [ -n "$journal" ] || return 1

  # Source diversity: try unique journals first.
  if grep -Fxq "$journal" "$SEEN_JOURNALS_FILE"; then
    return 1
  fi

  pmid=$(curl -sS -L --max-time 20 -G 'https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/' \
    --data-urlencode "ids=$pmcid" \
    --data-urlencode format=json | jq -r '.records[0].pmid // empty')
  [ -n "$pmid" ] || return 1

  summary=$(curl -sS --max-time 20 -G 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi' \
    --data-urlencode db=pubmed \
    --data-urlencode id="$pmid" \
    --data-urlencode retmode=json)

  title=$(printf '%s' "$summary" | jq -r --arg pmid "$pmid" '.result[$pmid].title // "Untitled"' | tr '\t\n\r' '   ')
  year=$(printf '%s' "$summary" | jq -r --arg pmid "$pmid" '.result[$pmid].pubdate // ""' | grep -Eo '^[0-9]{4}' || true)
  [ -n "$year" ] || year="NA"

  pmc_url="https://pmc.ncbi.nlm.nih.gov/articles/${pmcid}/"
  base="$(printf '%02d_%s_%s' "$current_index" "$pmid" "$pmcid" | tr -cd '[:alnum:]_.-')"
  out_pdf="$OUT_DIR/${base}.pdf"
  rm -f "$out_pdf"

  url="$(printf '%s' "$href" | sed 's#^ftp://ftp.ncbi.nlm.nih.gov/#https://ftp.ncbi.nlm.nih.gov/#')"
  if [ "$format" = "pdf" ]; then
    curl -sS -L --max-time 90 --fail "$url" -o "$out_pdf" || {
      rm -f "$out_pdf"
      return 1
    }
  else
    pkg=$(mktemp)
    curl -sS -L --max-time 90 --fail "$url" -o "$pkg" || {
      rm -f "$pkg"
      return 1
    }
    pdf_in_tar=$(tar -tzf "$pkg" 2>/dev/null | rg '\.pdf$' -m 1 || true)
    if [ -z "$pdf_in_tar" ]; then
      rm -f "$pkg"
      return 1
    fi
    tar -xzf "$pkg" -O "$pdf_in_tar" > "$out_pdf" 2>/dev/null || {
      rm -f "$pkg" "$out_pdf"
      return 1
    }
    rm -f "$pkg"
  fi

  if ! file "$out_pdf" | rg -q 'PDF'; then
    rm -f "$out_pdf"
    return 1
  fi

  printf '%s\toa_feed\t%s\t%s\t%s\t%s\t%s\thttps://pubmed.ncbi.nlm.nih.gov/%s/\thttps://pmc.ncbi.nlm.nih.gov/articles/%s/\t%s\n' \
    "$current_index" "$pmid" "$pmcid" "$year" "$journal" "$title" "$pmid" "$pmcid" "$(basename "$out_pdf")" >> "$META_FILE"

  echo "$pmcid" >> "$SEEN_PMCIDS_FILE"
  echo "$journal" >> "$SEEN_JOURNALS_FILE"
  added=$((added + 1))
  current_index=$((current_index + 1))
  return 0
}

feed_url="https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?from=${FROM_DATE}"
while [ "$added" -lt "$COUNT" ] && [ -n "$feed_url" ]; do
  xml=$(curl -sS --max-time 90 --fail "$feed_url" || true)
  [ -n "${xml:-}" ] || break

  while IFS=$'\t' read -r pmcid citation retracted format href; do
    [ "$added" -ge "$COUNT" ] && break
    [ -n "${pmcid:-}" ] || continue
    [ "$retracted" = "no" ] || continue
    download_record "$pmcid" "$citation" "$format" "$href" || true
  done <<< "$(printf '%s' "$xml" | parse_records)"

  feed_url="$(printf '%s' "$xml" | next_feed_url)"
done

rm -f "$SEEN_PMCIDS_FILE" "$SEEN_JOURNALS_FILE"

if [ "$added" -lt "$COUNT" ]; then
  echo "Added $added PDFs (requested $COUNT)." >&2
  exit 1
fi

echo "Added $added PDFs into $OUT_DIR"
