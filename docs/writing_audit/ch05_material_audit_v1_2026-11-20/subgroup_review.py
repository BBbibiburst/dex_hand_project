from pathlib import Path
import json,csv,hashlib
O=Path('docs/writing_audit/ch05_material_audit_v1_2026-11-20');sources=json.loads((O/'audited_sources.json').read_text());cards=list(csv.DictReader((O/'claim_cards.csv').open()));extra=[]
def read(p):sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return list(csv.DictReader(p.open()))
a=Path('outputs/tactile_c23_real_reconstruction_c0_2026-11-18/Session_reconstruction.csv');sr=read(a)
b=Path('outputs/tactile_p43_c4_root_cause_audit_2026-11-16/reconstruction_by_task_stage_group.csv');rr=read(b);tr=[r for r in rr if r['dataset']=='Natural_validation' and r['group_type']=='tasks']
assert sum(int(r['windows']) for r in tr)==905
text='''\n## Session、Task与时间合同补核\n\nC23/Session_reconstruction.csv的四个Session并非完全相同行为：\n\n|Session|窗口|同源MSE|Mask IoU|\n|---|---:|---:|---:|\n'''
for r in sr:text+=f"|{r['Session']}|{r['windows']}|{float(r['paired_MSE']):.6f}|{float(r['paired_Mask_IoU']):.6f}|\n"
text+='\nC4/reconstruction_by_task_stage_group.csv的Natural_validation、group_type=tasks记录的是**Sim VAE自身重建**，不是Sim→Real的物理对应：\n\n|任务|验证窗|Sim同域Mask IoU|\n|---|---:|---:|\n'
for r in tr:text+=f"|{r['group']}|{r['windows']}|{float(r['mask_IoU']):.6f}|\n"
text+='''\n这说明任务构成与同域重建困难不同，不能把全局指标变化全部归为域迁移。Root/Session逐组原始行在同CSV group_type=g，阶段为s；未获确认的不同任务阶段不硬合并为共同物理状态。P39/stage_metrics.csv按CLOSE/HOLD/LIFT提供旧协议分阶段BA/AUC/D，不能用来补齐新Natural任务的共同条件认证。\n\nP37/time_contract_diagnostics.json历史统计中，Real帧间隔中位约0.06525秒、四帧跨度约0.18798秒；Sim约0.05/0.15秒。其分位数组和历史生成定义可追溯，但不是本轮重读原始时间戳得到的新统计；C24又明确Real是顺序保留帧而非每一硬件采样。因此所有变化量只按保留帧索引解释，不能作同物理时间动态强弱的直接比较。\n'''
p=O/'CH05_EVIDENCE_MATRIX.md';p.write_text(p.read_text()+text)
for group,rows,source,groupkey,fields in [('C23_Session',sr,a,'Session',['paired_MSE','paired_Mask_IoU']),('C4_Natural_Task',tr,b,'group',['mask_IoU'])]:
 for r in rows:
  for k in fields:
   row=dict(cards[0]);row.update(claim=group+'_'+r[groupkey]+'_'+k,source=str(source),selector=f"{groupkey}={r[groupkey]};{k}"+(';dataset=Natural_validation;group_type=tasks' if group=='C4_Natural_Task' else ''),value=r[k],model_seed='4313' if group=='C23_Session' else '4302',data_role='Validation',evaluator='Real自重建' if group=='C23_Session' else 'Sim VAE自重建',preprocessing='原模型合同，原CSV统计',threshold='未调整',denominator=r['windows']+'窗',evidence='L2已有CSV逐组核对',allowed_claim='组间异质性，不是跨域因果分解')
   cards.append(row)
with (O/'claim_cards.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(cards[0]));w.writeheader();w.writerows(cards)
# Keep raw input Hash coverage complete for this additional focused inspection.
c=Path('outputs/tactile_p43_c14_repaired_transfer_probe_screen_2026-11-18/probe_metrics.csv');sources[str(c)]=hashlib.sha256(c.read_bytes()).hexdigest()
(O/'audited_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
p=O/'CH05_MATERIAL_AUDIT.md';p.write_text(p.read_text()+'\n补核：已直接以CPU、weights_only=True读取三个C3与C13 checkpoint元数据及张量形状，C3所存Epoch均30；C13仅6个current/amplitude张量改变。未构造或运行模型，见checkpoint_metadata.json。\n')
p=O/'finalize_audit.py';s=p.read_text().replace("if tmp.exists():tmp.replace(O/'git_status_before.txt')","if tmp.exists():\n (O/'git_status_before.txt').write_text(tmp.read_text());tmp.unlink()")
s=s.replace('claim_cards.csv（205项）','claim_cards.csv（'+str(len(cards))+'项）');p.write_text(s)
print('subgroup numeric cards',len(cards))
