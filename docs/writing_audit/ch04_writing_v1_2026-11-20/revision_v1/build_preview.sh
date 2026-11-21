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
for name in chapter04_preview chapter04_preview_with_references; do
 mkdir -p "$REVIEW_ROOT/build/$name"
 latexmk -norc -r "$THESIS_ROOT/latexmkrc" -r "$THESIS_ROOT/latexmk.projectrc" -pdf -halt-on-error -outdir="$REVIEW_ROOT/build/$name" "$REVIEW_ROOT/$name.tex"
 cp "$REVIEW_ROOT/build/$name/$name.pdf" "$REVIEW_ROOT/$name.pdf"
done
# Resolve citations with BibTeX first, then export the same body pages without references.
PYTHONPATH="$THESIS_ROOT/.tools/python" python3 - "$REVIEW_ROOT" <<'PY'
import sys
from pathlib import Path
import pymupdf as fitz
root=Path(sys.argv[1]); doc=fitz.open(root/'chapter04_preview_with_references.pdf')
ref=next(i for i,p in enumerate(doc) if p.get_text().startswith('哈尔滨工业大学硕士学位论文\n参考文献'))
body=fitz.open(); body.insert_pdf(doc,to_page=ref-1)
body.save(root/'chapter04_body.tmp.pdf');body.close()
(root/'chapter04_body.tmp.pdf').replace(root/'chapter04_preview.pdf')
PY
