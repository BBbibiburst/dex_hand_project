#!/usr/bin/env bash
set -euo pipefail
REVIEW_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$REVIEW_ROOT/../../../.." && pwd)"
THESIS_ROOT="$PROJECT_ROOT/thesis"
export PATH="$THESIS_ROOT/.tools/.TinyTeX/bin/x86_64-linux:$PATH"
export TEXMFVAR="$THESIS_ROOT/.tools/texmf-var"
export TEXMFCONFIG="$THESIS_ROOT/.tools/texmf-config"
export TEXMFHOME="$THESIS_ROOT/.tools/texmf-home"
cd "$THESIS_ROOT"
mkdir -p "$REVIEW_ROOT/build"
latexmk -norc -r "$THESIS_ROOT/latexmkrc" -r "$THESIS_ROOT/latexmk.projectrc" -pdf -halt-on-error -outdir="$REVIEW_ROOT/build" "$REVIEW_ROOT/ch03_preview.tex" > "$REVIEW_ROOT/build.log" 2>&1
