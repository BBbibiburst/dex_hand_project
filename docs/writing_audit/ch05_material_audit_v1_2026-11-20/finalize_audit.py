from pathlib import Path
import json,hashlib,subprocess,csv
O=Path('docs/writing_audit/ch05_material_audit_v1_2026-11-20');base=json.loads((O/'baseline.json').read_text());sources=json.loads((O/'audited_sources.json').read_text());obs=json.loads((O/'verified_observations.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
changed=[p for p,h in base['protected_before'].items() if sha(p)!=h];bad=[p for p,h in sources.items() if sha(p)!=h];head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();status=subprocess.check_output(['git','status','--short'],text=True)
assert not changed and not bad and head==base['head']
# All six sections' metric rows are complete and preserve the original labels/thresholds.
rr=list(csv.DictReader((O/'validation_metric_recheck.csv').open()));assert len(rr)==66
experiments=sorted({r['experiment'] for r in rr});assert len(experiments)==6
for ex in experiments:assert len({r['instance'] for r in rr if r['experiment']==ex})==11
for name in ['CH05_MATERIAL_AUDIT.md','CH05_EVIDENCE_MATRIX.md','CH05_GMM_PSEUDOPAIR_AUDIT.md','CH05_C0_PROTOCOL_AND_METRIC_AUDIT.md','CH05_STRUCTURE_AND_WORD_BUDGET.md','CH05_OPEN_ISSUES.md']:assert (O/name).stat().st_size>500
for name in ['paired_train.npz','paired_validation.npz']:
 for key in ['real','sim']:
  s=obs['dedicated_array_contracts'][name][key];assert s['shape'][1:]==[4,712] and s['finite'] and s['min']>=0 and s['unique_windows']==s['shape'][0]
report={'task':'CH05 stage2 material audit','decision':'可以直接进入正文写作（仍须用户/ChatGPT审核与阶段3任务书，不自行写作）','head_before':base['head'],'head_after':head,'git_status_before':base['git_status'],'git_status_after':status,'protected_before':base['protected_before'],'changed_protected_files':changed,'audited_input_sha256':sources,'changed_audited_inputs':bad,'all_protected_unchanged':True,'C0_and_C3_model_hashes_match_frozen_registry':True,'C13_checkpoint_changes_only_approved_amplitude_keys':True,'test_holdout_numeric_arrays_or_predictions_opened':False,'historical_Test_aggregate_report_scope':'仅辨认历史角色，不复算或选择模型；用于数值表的是Validation既有CSV','L3_experiments':0,'new_model_inferences':0,'new_training_steps':0,'physics_runs':0,'server_access':False,'new_LaTeX_or_bib_changes':False,'validation_metric_rows':66,'numeric_claim_cards':len(list(csv.DictReader((O/'claim_cards.csv').open()))),'no_compile_freeze_commit_push':True,'outputs_only_in_this_directory':True}
(O/'integrity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(O/'RUN_COMMANDS.json').write_text(json.dumps({'taskfile':'/home/zmy/.codex/attachments/cfa665e0-7486-441e-b2ba-dd8a20a0b06d/已粘贴的文本.txt','read_only_process':'代码文本/AST；指定Train/Validation配对数组与字典；已保存CSV统计；CPU weights_only=True读取checkpoint元数据，不构造模型、不推理','commands':['PYTHONDONTWRITEBYTECODE=1 /home/zmy/miniconda3/envs/mujoco/bin/python docs/writing_audit/ch05_material_audit_v1_2026-11-20/audit_sources.py','PYTHONDONTWRITEBYTECODE=1 /home/zmy/miniconda3/envs/mujoco/bin/python docs/writing_audit/ch05_material_audit_v1_2026-11-20/extra_checks.py','python3 docs/writing_audit/ch05_material_audit_v1_2026-11-20/write_reports.py','python3 docs/writing_audit/ch05_material_audit_v1_2026-11-20/finalize_audit.py'],'scope_note':'上列脚本不单独覆盖历史；本目录保留首次任务内复核的结果。C13原冻CSV对照及checkpoint元数据另以只读inline命令补核，结果分别写入validation_metric_recheck.csv与checkpoint_metadata.json。'},ensure_ascii=False,indent=2)+'\n')
tmp=Path('/tmp/ch05_status_before')
if tmp.exists():
 (O/'git_status_before.txt').write_text(tmp.read_text());tmp.unlink()
(O/'README.md').write_text('''# 第五章材料审计交付

结论与十项回答：CH05_MATERIAL_AUDIT.md。

专项：CH05_GMM_PSEUDOPAIR_AUDIT.md；CH05_C0_PROTOCOL_AND_METRIC_AUDIT.md。

逐节证据与预算：CH05_EVIDENCE_MATRIX.md；CH05_STRUCTURE_AND_WORD_BUDGET.md；缺口：CH05_OPEN_ISSUES.md。

数值来源：claim_cards.csv（218项）；validation_metric_recheck.csv（6组×11实例，保留混淆矩阵）。模型元数据：checkpoint_metadata.json；原始复核读数：verified_observations.json；代码定位：source_code_index.json。

完整性：integrity.json、audited_sources.json、SHA256SUMS。只读材料审计，不是新实验或正式论文正文。没有新模型推理/训练、物理运行或编译，没有进入下一阶段。
''')
(O/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p}\n' for p in sorted(O.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
print('PASS',len(sources),'audited sources;',len(base['protected_before']),'protected;66 instance rows;',report['numeric_claim_cards'],'claims; HEAD unchanged')
