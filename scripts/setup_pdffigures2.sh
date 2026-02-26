#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT/tools"
PDF_DIR="$TOOLS_DIR/pdffigures2"

mkdir -p "$TOOLS_DIR"

if [ ! -d "$PDF_DIR" ]; then
  echo "Cloning PDFFigures2 into $PDF_DIR"
  git clone https://github.com/allenai/pdffigures2.git "$PDF_DIR"
else
  echo "PDFFigures2 already exists at $PDF_DIR"
fi

cd "$PDF_DIR"

echo "Building PDFFigures2 (sbt assembly)..."
sbt assembly

JAR_PATH=$(ls -1 target/scala-2.12/*assembly*.jar 2>/dev/null | head -n 1 || true)
if [ -z "$JAR_PATH" ]; then
  echo "Could not find the assembly jar under target/scala-2.12/"
  exit 1
fi

echo "\nPDFFigures2 build complete."
echo "Set this in PsychPaperEvalApp/backend/.env:"
echo "PDFFIGURES2_JAR=$JAR_PATH"
