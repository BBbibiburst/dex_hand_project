from pathlib import Path
import re,json,csv,hashlib,difflib,collections
root=Path('thesis');out=root/'docs/writing_audit/ch01_1_2_writing_v2_2026-11-20'
chapter=root/'body/ch01_introduction.tex';bib=root/'reference.bib';oldchapter=(out/'before/body/ch01_introduction.tex').read_bytes();oldbib=(out/'before/reference.bib').read_bytes()
assert chapter.read_bytes().startswith(oldchapter),'Section 1.1 changed'
assert bib.read_bytes().startswith(oldbib),'Old bibliography changed'
section=chapter.read_bytes()[len(oldchapter):].decode();assert section.startswith('\n\\section{国内外研究现状}')
keys=re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib.read_text());dups=[k for k,n in collections.Counter(keys).items() if n>1];assert not dups
allcites=set(k.strip() for t in re.findall(r'\\cite\{([^}]+)\}',chapter.read_text()) for k in t.split(','));assert not allcites-set(keys)
used=set(k.strip() for t in re.findall(r'\\cite\{([^}]+)\}',section) for k in t.split(','))
refs=json.loads((out/'references_used.json').read_text());bykey={x['bibtex_key']:x for x in refs};assert used==set(bykey)
# Check LaTeX/BibTeX grouping and environment nesting, without invoking TeX or BibTeX.
def balanced(t):
 n=0
 for i,c in enumerate(t):
  if c not in '{}':continue
  backslashes=0;j=i-1
  while j>=0 and t[j]=='\\':backslashes+=1;j-=1
  if backslashes%2:continue
  n+=1 if c=='{' else -1
  assert n>=0,(i,t[max(0,i-30):i+30])
 assert n==0,n
balanced(section);balanced(bib.read_text())
stack=[]
for kind,name in re.findall(r'\\(begin|end)\{([^}]+)\}',section):
 if kind=='begin':stack.append(name)
 else:assert stack.pop()==name
assert not stack
labels=re.findall(r'\\label\{([^}]+)\}',chapter.read_text());assert len(labels)==len(set(labels))
assert set(re.findall(r'\\ref\{([^}]+)\}',section))<=set(labels)
urls=re.findall(r'^\s*url\s*=\s*\{([^\n]*)\}',bib.read_text(),re.M);assert all(u.startswith('https://') and not any(x in u for x in ['[',']',' ','\\']) for u in urls)
protected=json.loads((out/'before_hashes.json').read_text());allowed={'thesis/body/ch01_introduction.tex','thesis/reference.bib'}
changed=[s for s,h in protected.items() if hashlib.sha256(Path(s).read_bytes()).hexdigest()!=h];assert set(changed)==allowed
for rel in ['body/ch01_introduction.tex','reference.bib']:
 a=(out/'before'/rel).read_text().splitlines(keepends=True);b=(root/rel).read_text().splitlines(keepends=True)
 (out/(Path(rel).name+'.diff')).write_text(''.join(difflib.unified_diff(a,b,fromfile='before/'+rel,tofile='thesis/'+rel)))
# Counting policy: narrative excludes headings, table, labels and citations; count Han and all nonspace characters separately.
narrative=section.split('\\begin{table}')[0];narrative=re.sub(r'\\(?:section|subsection|cite|label|ref)\{[^}]*\}','',narrative)
counts=[]
for idx,part in enumerate(section.split('\\subsection')[1:],1):
 heading=part.split('}')[0][1:];part=part.split('\\begin{table}')[0];part=re.sub(r'^\{[^}]*\}','',part);part=re.sub(r'\\(?:cite|label|ref)\{[^}]*\}','',part)
 counts.append({'section':f'1.2.{idx}','heading':heading,'han_characters':len(re.findall(r'[\u4e00-\u9fff]',part)),'nonspace_characters':len(re.sub(r'\s','',part)),'paragraphs':len([v for v in part.split('\n\n') if v.strip()])})
# Review copy keeps citations mapped to the V1 IDs. This is derived mechanically from the actual TeX.
plain=re.sub(r'\\cite\{([^}]+)\}',lambda m:'['+'、'.join(bykey[q]['source_id'] for q in m[1].split(','))+']',section)
plain=re.sub(r'\\label\{[^}]*\}','',plain).split('\\begin{table}')[0]
plain=re.sub(r'\\(?:subsection|section)\{([^}]+)\}',lambda m:'\n'+m[1]+'\n',plain).replace(r'表\ref{tab:related_work_routes}','下表')
table='''技术路线对比表（与LaTeX表内容对应）
技术方向｜代表方法｜主要输入与监督｜解决的问题｜适用条件及局限
抓取优化｜DexGraspNet、GraspQP｜物体和手模型、接触质量能量｜多指抓取候选生成｜依赖接触假设，不保证整机执行
物理细化｜DexEvolve｜解析候选、仿真反馈｜稳定性与多样性优化｜受物理模型及执行条件影响
目标执行｜UniDexGrasp｜点云、抓取目标和策略学习｜连接候选与操作过程｜依赖机器人与控制设置
触觉模拟｜TACTO、Taxim｜接触几何、光学或形变标定｜视觉触觉读出生成｜不能直接等同标量阵列响应
示教构建｜MimicGen及多模态数据集｜源示教、场景和传感器记录｜轨迹扩展与数据利用｜需核查任务假设及数据关联
物理迁移｜BioTac潜投影[S25]｜FEM形变、真实电信号和配对监督｜跨模态潜表示转换｜依赖监督对应和传感器模型
非配对迁移｜CycleGAN、UNIT｜两域样本及循环或共享约束｜约束缺少配对的映射｜不认证唯一物理对应
特征适应｜DANN｜源域任务标签及两域观测｜任务相关的域不变表示｜不等于完整触觉重建
'''
(out/'section_1_2_review.txt').write_text('TH-CH01-1.2-V2 审核阅读稿\nS编号对应V1文献目录，不是最终排版文献序号；正式文本以body/ch01_introduction.tex为准。\n'+plain+'\n'+table)
qids=[['Q01','Q02'],['Q02','Q03'],['Q04','Q05'],['Q06'],['Q07','Q08','Q09'],['Q10','Q11'],['Q12','Q14'],['Q12','Q13','Q14'],['Q15','Q16'],['Q16','Q17','Q18'],['Q18','Q19','Q20'],['Q21','Q22','Q23']]
rows=[];i=0
for sid,part in enumerate(section.split('\\subsection')[1:],1):
 part=re.sub(r'^\{[^}]*\}\n\\label\{[^}]*\}\n','',part).split('\\begin{table}')[0]
 for j,para in enumerate([t for t in part.split('\n\n') if t.strip()],1):
  ks=[k.strip() for t in re.findall(r'\\cite\{([^}]+)\}',para) for k in t.split(',')]
  rows.append({'paragraph':f'1.2.{sid}-P{j}','claim_ids':';'.join(qids[i]),'source_ids':';'.join(bykey[k]['source_id'] for k in ks),'bibtex_keys':';'.join(ks),'narrative_start':para[:40]});i+=1
assert i==12
with (out/'paragraph_evidence.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result={'task':'TH-CH01-1.2-V2','task_date':'2026-11-19','execution_date':'2026-11-20','timezone':'Asia/Shanghai','stage':'3 completed; awaiting user review','section_1_1_byte_identical':True,'old_bib_entries_byte_identical':True,'changed_existing_files':changed,'all_other_protected_files_unchanged':True,'citations_in_section':len(used),'new_bib_entries':sum(x['status']=='added' for x in refs),'reused_bib_entries':sum(x['status']=='reused_unchanged' for x in refs),'duplicate_keys':dups,'missing_keys':sorted(allcites-set(keys)),'narrative_han_characters':len(re.findall(r'[\u4e00-\u9fff]',narrative)),'narrative_nonspace_characters':len(re.sub(r'\s','',narrative)),'section_counts':counts,'table_count':section.count('\\begin{table}'),'brace_and_environment_static_check':'passed','compilation_performed':False,'bibtex_backend_execution_performed':False,'training_or_simulation_performed':False,'commit_push_performed':False,'output_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [chapter,bib]}}
(out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
