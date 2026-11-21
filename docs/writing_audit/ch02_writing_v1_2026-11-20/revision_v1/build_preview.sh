#!/usr/bin/env bash
set -euo pipefail
REV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$REV_ROOT/../../../.." && pwd)"
THESIS_ROOT="$PROJECT_ROOT/thesis"
export PATH="$THESIS_ROOT/.tools/.TinyTeX/bin/x86_64-linux:$PATH"
export TEXMFVAR="$THESIS_ROOT/.tools/texmf-var"
export TEXMFCONFIG="$THESIS_ROOT/.tools/texmf-config"
export TEXMFHOME="$THESIS_ROOT/.tools/texmf-home"
cd "$THESIS_ROOT"
mkdir -p "$REV_ROOT/build"
latexmk -norc -r "$THESIS_ROOT/latexmkrc" -r "$THESIS_ROOT/latexmk.projectrc" -pdf -halt-on-error -outdir="$REV_ROOT/build" "$REV_ROOT/ch02_preview.tex" 2>&1 | tee "$REV_ROOT/build_console.log"
