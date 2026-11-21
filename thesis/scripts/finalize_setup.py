from pathlib import Path
import hashlib,json,subprocess,ast
R=Path(__file__).resolve().parents[1];D=R/'docs'
for p in (R/'scripts').glob('*.py'):ast.parse(p.read_text())
for p in list((R/'.vscode').glob('*.json'))+[R/'thesis.code-workspace']:json.loads(p.read_text())
expected=json.loads((D/'template_provenance.json').read_text())
for name,h in expected['unmodified_runtime_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
assert not subprocess.check_output(['git','-C',str(R/'upstream/hithesis'),'status','--porcelain'],text=True).strip()
assert subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()=='ff7b1a1'
assert not subprocess.check_output(['git','diff','--name-only'],text=True).strip()
assert not subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip()
for name in ['project_files.txt','full_file_inventory.txt','project_integrity.json']:(D/name).touch(exist_ok=True)
def sourcefile(p):return p.is_file() and not any(x in {'.tools','upstream','build','logs','__pycache__'} for x in p.relative_to(R).parts)
sources=sorted(p for p in R.rglob('*') if sourcefile(p))
(D/'project_files.txt').write_text('\n'.join(str(p.relative_to(R)) for p in sources)+'\n')
allfiles=sorted(p for p in R.rglob('*') if p.is_file())
(D/'full_file_inventory.txt').write_text('\n'.join(str(p.relative_to(R)) for p in allfiles)+'\n')
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.name!='project_integrity.json'}
(D/'project_integrity.json').write_text(json.dumps(dict(date='2026-11-19',project_sources=len(sources),full_file_count=len(allfiles),files=manifest,template_core_unchanged=True,upstream_unchanged=True,no_tracked_project_files_modified=True,no_git_staging=True,no_commit_push=True,no_research_training_or_collection=True,HEAD='ff7b1a17e3122d7456dfa3f48c1bb7a0c2c7f421',GUI_test='not performed; configured and CLI verified'),ensure_ascii=False,indent=2)+'\n')
print('SETUP VERIFIED',len(sources),'project files;',len(allfiles),'including tools, upstream and generated artifacts; no Git changes staged')
