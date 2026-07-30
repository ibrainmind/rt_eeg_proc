#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/paper"

if [[ ! -f "$PAPER_DIR/paper.tex" ]]; then
  echo "error: missing $PAPER_DIR/paper.tex" >&2
  exit 1
fi

# Prefer user-local TinyTeX if available.
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$HOME/.local/bin:$PATH"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "error: pdflatex not found. Install TinyTeX or LaTeX first." >&2
  exit 1
fi

cd "$PAPER_DIR"

pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex

pdflatex -interaction=nonstopmode -halt-on-error convergence_note.tex
pdflatex -interaction=nonstopmode -halt-on-error convergence_note.tex

echo "Built: $PAPER_DIR/paper.pdf"
