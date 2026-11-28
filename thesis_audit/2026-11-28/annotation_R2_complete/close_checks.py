from pathlib import Path
import json,csv,re,hashlib
R=Path('/home/zmy/dex_hand_project');D=R/'thesis_audit/2026-11-28/annotation_R2_complete'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
P=D/'ANNOTATION_LEDGER.csv';rows=list(csv.DictReader(P.open(encoding='utf-8-sig')))
anchors={1:'firstpagecdate',2:'ctitlecover',3:'cdate=',4:'csubject=',5:'esubject=',6:'edate=',7:'实机',8:'RealMAN',10:'数据',14:'eabstract',15:'灵巧手操作依靠',16:'真实机器人试验涉及',17:'仿真允许保存',18:'RealMAN',20:'TACTO',32:'手部采用刚体关节',36:'本文实现组合位置控制器',37:'fig:ch02_control',38:'固定腱执行器的位置',39:'tab:ch02_control_parameters',40:'0.05',45:'sec:tactile_processing',46:'fig:ch02_processing',47:'fig:ch02_historical_heatmaps',51:'本文',54:'DexEvolve'}
for r in rows:
 i=int(r['id'])
 if '沿用前批' not in r['modified_location']:continue
 p=R/r['modified_file'];lines=p.read_text().splitlines();a=anchors.get(i,'')
 n=next((i for i,s in enumerate(lines,1) if a and a in s),None)
 if n:r['modified_location']=f'当前第{n}行：{a}（删除项定位到现存对应段，不表示原选中文字仍保留）'
with P.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
b=json.loads((D/'baseline.json').read_text());eq={}
pat=r'\\begin\{(equation\*?|align\*?|gather\*?)\}(.*?)\\end\{\1\}'
for p in json.loads((D/'integrity.json').read_text())['changed_existing']:
 if not p.endswith('.tex'):continue
 before=(D/'before'/p).read_text();after=(R/p).read_text()
 eq[p]=re.findall(pat,before,re.S)==re.findall(pat,after,re.S)
(D/'equation_preservation.json').write_text(json.dumps(eq,ensure_ascii=False,indent=2))
check=json.loads((D/'source_bundle_check.json').read_text());integ=json.loads((D/'integrity.json').read_text());integ['source_bundle_independent_compile']=check;integ['numbered_equation_blocks_unchanged']=eq
integ['wordcount_source_hashes_current']=all(sha(R/p)==v for p,v in json.loads((D/'wordcount/THESIS_WORD_COUNT.json').read_text())['source_hashes'].items())
integ['reference_annotated_pdf_current_sha256']=sha(R/'thesis_draft_commented.pdf')
integ['artifacts']={str(p.relative_to(R)):sha(p) for p in sorted(D.iterdir()) if p.is_file() and p.name not in ['integrity.json','SHA256SUMS']}
(D/'integrity.json').write_text(json.dumps(integ,ensure_ascii=False,indent=2))
with (D/'REVIEW.md').open('a') as f:f.write('\n## 源码包独立编译复核\n\n解压69个源/模板/图稿文件后从独立目录执行XeLaTeX、BibTeX和后续XeLaTeX；成功生成106页。未借用原目录的图稿或实验数据；仅复用已安装TeX工具链。日志见source_bundle_build.log。PDF字节可能因构建时间和路径不同而不同，因此核对源Hash、成功状态和页数，不谎称字节完全一致。编号数学环境与本轮基线逐块一致，见equation_preservation.json。\n')
# Hash the final form of reports, not the earlier pre-append report.
integ['artifacts'][str((D/'REVIEW.md').relative_to(R))]=sha(D/'REVIEW.md')
(D/'integrity.json').write_text(json.dumps(integ,ensure_ascii=False,indent=2))
files=[p for p in D.iterdir() if p.is_file() and p.name!='SHA256SUMS']+list((R/'thesis/figures/annotation_R2').glob('*.pdf'))
(D/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(R))+'\n' for p in sorted(files)))
print(json.dumps({'numbered_math_preserved':eq,'wordcount_hashes_current':integ['wordcount_source_hashes_current'],'all_source_hashes_match':integ['source_inputs_hash_match'],'bundle':check,'status':integ['status_counts']},ensure_ascii=False,indent=2))
