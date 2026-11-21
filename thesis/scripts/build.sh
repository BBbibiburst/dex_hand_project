#!/usr/bin/env bash
set -euo pipefail
THESIS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
THESIS_TEXBIN="$THESIS_ROOT/.tools/.TinyTeX/bin/x86_64-linux"
if [[ -x "$THESIS_TEXBIN/latexmk" ]]; then
  export PATH="$THESIS_TEXBIN:$PATH"
fi
if ! command -v latexmk >/dev/null || ! command -v xelatex >/dev/null || ! command -v bibtex >/dev/null; then
  echo '缺少 latexmk、XeLaTeX 或 BibTeX。请阅读 README.md 的依赖说明。' >&2
  exit 127
fi
export TEXMFVAR="$THESIS_ROOT/.tools/texmf-var"
export TEXMFCONFIG="$THESIS_ROOT/.tools/texmf-config"
export TEXMFHOME="$THESIS_ROOT/.tools/texmf-home"
cd "$THESIS_ROOT"
mkdir -p build/{body,front,back,tests} logs "$TEXMFVAR" "$TEXMFCONFIG" "$TEXMFHOME"
THESIS_MAIN="${1:-thesis.tex}"
THESIS_LOG="logs/$(basename "$THESIS_MAIN" .tex)_$(date +%Y%m%d_%H%M%S).console.log"
# 原样加载上游latexmkrc；其中pdf模式的命令即XeLaTeX，引用走BibTeX。
THESIS_EXTRA=()
if [[ "${THESIS_FORCE_REBUILD:-0}" == "1" ]]; then THESIS_EXTRA=(-g); fi
latexmk "${THESIS_EXTRA[@]}" -norc -r "$THESIS_ROOT/latexmkrc" -r "$THESIS_ROOT/latexmk.projectrc" -pdf -halt-on-error -outdir=build "$THESIS_MAIN" 2>&1 | tee "$THESIS_LOG"
