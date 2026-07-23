#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/paper"

cd "$PAPER_DIR"
rm -f paper.aux paper.fdb_latexmk paper.fls paper.log paper.out

echo "Cleaned LaTeX aux files in: $PAPER_DIR"
