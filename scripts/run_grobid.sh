#!/usr/bin/env bash
set -euo pipefail

echo "Starting GROBID Docker container on http://localhost:8070"
echo "Using grobid/grobid:0.8.2-crf (smaller image, CPU-only)"
exec docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2-crf
