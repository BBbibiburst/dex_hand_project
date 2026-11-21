#!/usr/bin/env python3
"""TeXcount-backed statistics. Python resolves literal includes and formats reports; it never counts prose words."""
from pathlib import Path
import argparse, collections, csv, datetime, hashlib, html, json, os, re, shutil, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]; TH=ROOT/'thesis'
ENGINE=TH/'.tools/texcount/texcount.pl'; RULES=ROOT/'scripts/texcount_rules.tex'
BUDGET=ROOT/'scripts/thesis_length_budget.json'; TEST=ROOT/'scripts/tests/thesis_wordcount'
COUNTER_NAMES=['text','headings','captions_footnotes','heading_count','float_count','inline_math','display_math']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def rel(p):
 try:return str(p.resolve().relative_to(ROOT))
 except ValueError:return str(p.resolve())
def mask_comments(s):
    chars=list(s)
    for m in re.finditer(r'%[^\n]*',s):
        i=m.start();j=i-1
        while j>=0 and s[j]==chr(92):j-=1
        if (i-1-j)%2==0:chars[i:m.end()]=' '*(m.end()-i)
    return ''.join(chars)

def masked(s):
    # Preserve offsets. Ignore literal listing bodies and comments when discovering include commands.
    def blank(m):return ''.join('\n' if c=='\n' else ' ' for c in m[0])
    s=re.sub(r'\\begin\{(lstlisting|verbatim|minted)\}.*?\\end\{\1\}',blank,s,flags=re.S)
    s=re.sub(r'%TC:ignore.*?%TC:endignore',blank,s,flags=re.S)
    s=mask_comments(s)
    return s
class Resolver:
    def __init__(self,base):self.base=base;self.seen=set();self.files=[];self.edges=[];self.missing=[];self.duplicates=[];self.warnings=[]
    def expand(self,p):
        p=p.resolve()
        if p in self.seen:self.duplicates.append(rel(p));return '% Duplicate source omitted: '+rel(p)+'\n'
        if not p.is_file():self.missing.append(rel(p));return '% MISSING '+rel(p)+'\n'
        self.seen.add(p);self.files.append(p);s=p.read_text(encoding='utf-8');m=masked(s)
        if re.search(r'\\(?:includeonly|import|subimport|inputiffileexists|includegraphics.*\\)',m,re.I):self.warnings.append(rel(p)+': unsupported/dynamic include or includeonly: inspect audit')
        pat=r'\\(input|include)\s*(?:\{([^{}]+)\}|([^\s{}%]+))';parts=[];end=0
        for x in re.finditer(pat,m):
            token=(x[2] or x[3]).strip();parts.append(s[end:x.start()]);end=x.end()
            if '\\' in token or '#' in token:
                self.warnings.append(rel(p)+': dynamic input '+token);parts.append('% dynamic input unresolved\n');continue
            name=token if Path(token).suffix else token+'.tex';candidates=[self.base/name,p.parent/name];child=next((q for q in candidates if q.is_file()),candidates[0]).resolve()
            self.edges.append({'parent':rel(p),'child':rel(child),'command':x[1]});parts.append('\n% Included from '+rel(child)+'\n'+self.expand(child)+'\n')
        parts.append(s[end:]);return ''.join(parts)
    def record(self):return dict(files=[rel(p) for p in self.files],edges=self.edges,missing=self.missing,duplicate_occurrences=self.duplicates,warnings=self.warnings)

class Counter:
    def __init__(self,out):self.out=out;self.commands=[];self.cache={};(out/'generated').mkdir(parents=True,exist_ok=True);(out/'raw').mkdir(exist_ok=True)
    def run(self,s,name,verbose=False):
        p=self.out/'generated'/(name+'.tex');p.write_text(RULES.read_text()+'\n'+s)
        cmd=['perl',str(ENGINE),'-utf8','-ch','-sum=1,1,1,0,0,0,0','-sub=subsection']
        if verbose:cmd+=['-html','-v']
        cmd+=[str(p)];self.commands.append(cmd);r=subprocess.run(cmd,cwd=TH,text=True,encoding='utf-8',capture_output=True)
        if r.returncode:raise RuntimeError('TeXcount failed: '+r.stderr+r.stdout)
        if verbose:
            (self.out/'wordcount_audit.html').write_text(r.stdout);return
        (self.out/'raw'/(name+'.txt')).write_text(r.stdout);(self.out/'raw'/(name+'.stderr.txt')).write_text(r.stderr)
        labels=['Words in text','Words in headers',r'Words outside text \(captions, etc\.\)','Number of headers','Number of floats/tables/figures','Number of math inlines','Number of math displayed']
        values=[]
        for lab in labels:
            m=re.search('^'+lab+r': (\d+)$',r.stdout,re.M)
            if not m:raise RuntimeError('Cannot parse actual TeXcount output: '+r.stdout)
            values.append(int(m[1]))
        counts=dict(zip(COUNTER_NAMES,values));counts['total']=sum(values[:3]);counts['raw_output']=rel(self.out/'raw'/(name+'.txt'))
        counts['subcounts']=[]
        for m in re.finditer(r'^\s+(\d+)\+(\d+)\+(\d+) \((\d+)/(\d+)/(\d+)/(\d+)\) (.+)$',r.stdout,re.M):
            c=dict(zip(COUNTER_NAMES,map(int,m.groups()[:7])));c.update(title=m[8],total=sum(int(m[j]) for j in [1,2,3]));counts['subcounts'].append(c)
        counts['warnings']=[line for line in (r.stdout+'\n'+r.stderr).splitlines() if re.search(r'warning|error',line,re.I)]
        # Code is deliberately not counted as prose. Report physical lines/blocks separately, not a claimed code-word count.
        blocks=re.findall(r'\\begin\{(lstlisting|verbatim|minted)\}(.*?)\\end\{\1\}',masked_comments(s),re.S)
        counts['code_blocks']=len(blocks);counts['code_physical_lines']=sum(len(b.strip('\n').splitlines()) for _,b in blocks)
        return counts

def masked_comments(s):return mask_comments(s)
def chapter_title(s):
    m=re.search(r'\\chapter(?:\[[^\]]*\])?\{([^}]+)\}',masked(s));return m[1] if m else None

def test(counter):
    results=[]
    b=counter.run((TEST/'basic.tex').read_text(),'test_basic');expected={'text':7,'headings':4,'captions_footnotes':7,'total':18,'inline_math':1,'display_math':1,'float_count':2,'code_blocks':1}
    results.append(dict(name='UTF8/text/headings/table/caption/footnote/math/comments/cite/ref/code/bibliography',expected=expected,actual={k:b[k] for k in expected},passed=all(b[k]==v for k,v in expected.items())))
    res=Resolver(TEST);x=counter.run(res.expand(TEST/'includes.tex'),'test_includes_unique')
    results.append(dict(name='literal input+include; same canonical file counted once',expected_total=7,actual_total=x['total'],duplicates=res.duplicates,passed=x['total']==7 and len(res.duplicates)==1))
    c=counter.run((TEST/'custom.tex').read_text(),'test_custom');results.append(dict(name='custom text macro rule',expected_total=3,actual_total=c['total'],passed=c['total']==3))
    e=counter.run((TEST/'extended.tex').read_text(),'test_extended');results.append(dict(name='longtable/multicolumn/multirow/minted/verbatim/hitsetup/ignored-region',expected_total=7,actual_total=e['total'],passed=e['total']==7 and e['code_blocks']==2 and e['float_count']==1))
    cmd=['perl',str(ENGINE),'-utf8','-ch','-sum=1,1,1,0,0,0,0','-merge','-dir='+str(TEST),str(TEST/'includes.tex')]
    raw=subprocess.run(cmd,text=True,capture_output=True,encoding='utf-8');(counter.out/'raw/test_native_merge.txt').write_text(raw.stdout+raw.stderr);m=re.search(r'Sum count: (\d+)',raw.stdout)
    results.append(dict(name='native TeXcount include integration (comparison, not primary dedup policy)',native_total=int(m[1]) if m else None,returncode=raw.returncode,passed=raw.returncode==0 and m is not None,note='Primary count uses canonical-file deduplication resolver; native trace retained.'))
    save(counter.out/'tests.json',results)
    if not all(r['passed'] for r in results):raise RuntimeError('Wordcount self-test failed; see tests.json')
    return results

def budget_document(cfg):
    lines=['# 硕士论文篇幅预算与统计口径','', '日期：2026-11-20。依据THESIS-LENGTH-01及THESIS_PLAN_V2七章42节。','',
    '完整PDF目标90–100页、最低目标超过80页；正文65–75页。七章TeXcount目标49,500–56,500，中心53,000。**均为写作规划，不是学校官方最低要求，也不是由字数换算出的页数保证。**', '',
    '正文统计量 = TeXcount中文UTF-8模式的正文词数＋标题词数＋图表题/脚注词数。中文按字符、英文按词，表格单元格文字纳入；数学公式数量、代码块/行数另列，不折算为正文。总量包含标题及表格文字，区别于前期仅数汉字的审核稿统计。', '',
    '## 七章预算','', '|章|按V2原题名|最低|最高|中心|','|---|---|---:|---:|---:|']
    for c in cfg['chapters']:lines.append(f"|{c['number']}|{c['title']}|{c['minimum']}|{c['maximum']}|{c['center']:g}|")
    lines+=['|合计||49500|56500|53000|','','## 42节分配','','按论证任务、方法细节、已有证据与图表需求分配，不平均分配。标题严格来自V2；1.2已授权的四个下级小节共享1.2预算，不另增规划节数。','', '|节|题名|最低|最高|中心|分配理由|','|---|---|---:|---:|---:|---|']
    for c in cfg['chapters']:
        for s in c['sections']:lines.append(f"|{s['id']}|{s['title']}|{s['minimum']}|{s['maximum']}|{s['center']:g}|{s['rationale']}|")
    lines+=['','## 结构与证据边界','',
    '主文件仍为六章旧骨架。第一章可以直接对应V2；其余V2章节使用配置中的拟议路径，尚未建立的标MISSING，不将旧第3/4/5章强行算为新章节。拟议路径仅为统计映射，不创建文件、不修改include。旧六章逐文件统计另列。',
    '第四章有条件保留：来源、Episode lineage、动作/观测时间关系、质量审计及统计证据不足时先报告缺口，不以工程扩写凑字。其预算属于七章总目标，若经审核调整章节，需同步重新分配，不能暗中降低目标。',
    '完成百分比以章/节中心预算为分母，允许超过100%；另报距最低目标的缺口。它是篇幅进度，不是科研完成度。缺文件记MISSING而非伪造实测0字。',
    '摘要、致谢、附录、参考文献分别列出。未启用的back文件只报库存，不并入当前编译源。reference.bib的条目数不是参考文献列表的正文词数。封面hitsetup元数据不展开；TeXcount不能推算最终封面与参考文献的排版长度。', '',
    '## 一键操作','', '在项目根目录执行：','', '```sh','./scripts/count_thesis_words.sh --all','./scripts/count_thesis_words.sh --chapter 1','./scripts/count_thesis_words.sh --sections','./scripts/count_thesis_words.sh --files','./scripts/count_thesis_words.sh --audit','./scripts/count_thesis_words.sh --test','```', '',
    '默认输出outputs/thesis_wordcount/。--all/--chapter/--sections/--files/--audit都会重新读取源文件并更新完整报告；参数只改变终端摘要。--test只运行统计规则样例。可用--output-dir指定独立输出目录。',
    '依赖Python3标准库、Perl及本地TeXcount。脚本不会自动联网安装、编译论文或运行实验；缺依赖时明确失败。引擎位于thesis/.tools/texcount/texcount.pl，来自CTAN原件，未修改；规则位于scripts/texcount_rules.tex。', '',
    '## 解析边界与后续交付','',
    'TeXcount是基于规则的LaTeX解析，不是完整TeX展开器。脚本只解析字面量input/include，先按规范化物理路径去重。动态文件名、includeonly、import或条件编译需检查警告，不能宣称精确复现TeX运行。未知自定义宏、子图、复杂表格、外部代码和宏生成文字需先补规则及样例再纳入。',
    '宏定义不会像TeX一样完全展开；规则忽略label/cite/ref参数、图片路径、表格列格式和hitsetup，不计参考文献环境；图表正文/标题和脚注均计入。英文连字符、数字和标点的边界沿用TeXcount，不使用正则替代词数。',
    'wordcount_audit.html为TeXcount实际着色解析。raw/保存原始输出，generated/是仅供统计的展开副本，正式源文件不改。公式按环境/行内片段数量报告，不按符号数量折算。代码单独报块数和物理行数，不伪称为TeXcount程序词数。',
    '现有PDF若早于源文件则视为STALE：可列该旧产物页数，但当前正文页数仍为UNKNOWN。字数与最终页数不是固定比例，图表、公式和排版需在批准编译后检查。',
    '--chapter参数按V2七章编号，旧六章文件请看--files和实际include表。每次指定章节写作交付附该章真实TeXcount结果；每完成一章重新运行全文统计。不能为预算重复背景、堆砌定义、虚构实验或改动已核验结果。通过动机、定义、方法、实现、设置、结果解释、消融与局限增加信息量；证据不足先报告。','']
    (TH/'THESIS_LENGTH_BUDGET.md').write_text('\n'.join(lines))

def protected_snapshot():
    files=[p for p in TH.iterdir() if p.is_file() and p.suffix in ['.tex','.bib','.cls','.cfg','.sty','.bst']]
    for folder in ['body','front','back']:
        files+=list((TH/folder).rglob('*.tex'))
    files += [TH/n for n in ['THESIS_PLAN_V2.md','AGENTS.md','THESIS_WRITING_STANDARD.md']]
    return {rel(p):sha(p) for p in files}

def main():
    ap=argparse.ArgumentParser(description=__doc__);g=ap.add_mutually_exclusive_group();g.add_argument('--all',action='store_true');g.add_argument('--chapter',type=int,choices=range(1,8));g.add_argument('--sections',action='store_true');g.add_argument('--files',action='store_true');g.add_argument('--audit',action='store_true');g.add_argument('--test',action='store_true');ap.add_argument('--output-dir',default=str(ROOT/'outputs/thesis_wordcount'));args=ap.parse_args()
    if not ENGINE.is_file() or not shutil.which('perl'):raise SystemExit('Missing Perl or local TeXcount; see thesis/THESIS_LENGTH_BUDGET.md')
    formal_before=protected_snapshot()
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True);counter=Counter(out);tests=test(counter)
    if args.test:print(f'PASS: {len(tests)} TeXcount integration tests; '+str(out/'tests.json'));return
    cfg=json.loads(BUDGET.read_text());plan=(TH/'THESIS_PLAN_V2.md').read_text();ps=re.findall(r'^### ([1-7]\.\d+) (.+)$',plan,re.M);assert ps==[(s['id'],s['title']) for c in cfg['chapters'] for s in c['sections']]
    budget_document(cfg)
    graph=Resolver(TH);expanded=graph.expand(TH/'thesis.tex');actual_full=counter.run(expanded,'actual_full')
    file_counts=[];body_roots=[]
    for p in graph.files:
        if p==TH/'thesis.tex':continue
        counts=counter.run(p.read_text(),'file_'+p.relative_to(TH).as_posix().replace('/','__').removesuffix('.tex'))  # own-file count, no included children counted twice
        top=p.relative_to(TH).parts[0];category=top if top in ['body','front','back'] else 'support'
        file_counts.append(dict(path=rel(p),category=category,title=chapter_title(p.read_text()),content_status='SKELETON_ONLY' if counts['text']==0 and counts['captions_footnotes']==0 else 'DRAFT_OR_PLACEHOLDER',**counts))
        if category=='body' and not any(e['child']==rel(p) and e['parent'].startswith('thesis/body/') for e in graph.edges):body_roots.append(p)
    body_res=Resolver(TH);body_text='\n'.join(body_res.expand(p) for p in body_roots);actual_body=counter.run(body_text,'actual_body');counter.run(body_text,'actual_body_audit',verbose=True)
    planned=[];all_sections=[];planned_res=Resolver(TH)
    for c in cfg['chapters']:
        p=TH/c['source_file'];exists=p.is_file();record={k:v for k,v in c.items() if k!='sections'};record['source_path']=rel(p);record['status']='PRESENT' if exists else 'MISSING'
        count=None
        if exists:
            count=counter.run(planned_res.expand(p),'planned_'+str(c['number']));record['source_in_actual_graph']=p.resolve() in graph.seen
        record['metrics']=count;record['actual_count']=count['total'] if count else None;record['completion_percent']=100*count['total']/c['center'] if count else None;record['gap_to_minimum']=max(0,c['minimum']-count['total']) if count else None
        section_entries=[];current=None
        for sub in count['subcounts'] if count else []:
            if sub['title'].startswith('Section: '):
                title=sub['title'][9:];current=next((s for s in c['sections'] if s['title']==title),None)
                if current:section_entries.append(dict(title=title,metrics=dict.fromkeys(COUNTER_NAMES,0),id=current['id']))
            elif sub['title'].startswith('Chapter: '):current=None
            if current:
                target=next(x for x in section_entries if x['id']==current['id'])
                for k in COUNTER_NAMES:target['metrics'][k]+=sub[k]
        for s in c['sections']:
            found=next((v for v in section_entries if v['id']==s['id']),None);n=sum(found['metrics'][k] for k in COUNTER_NAMES[:3]) if found else None
            row={**s,'chapter':c['number'],'source_path':rel(p),'status':'PRESENT' if found else 'MISSING_SECTION' if exists else 'MISSING_FILE','actual_count':n,'completion_percent':100*n/s['center'] if n is not None else None,'gap_to_minimum':max(0,s['minimum']-n) if n is not None else None,'metrics':found['metrics'] if found else None};all_sections.append(row)
        record['sections']=[s for s in all_sections if s['chapter']==c['number']];planned.append(record)
    # Inactive inventories are measured separately and are explicitly not added to compiled/body totals.
    ancillary=[]
    for p in [TH/'front/abstract.tex',TH/'front/cover.tex',TH/'front/denotation.tex',TH/'back/acknowledgements.tex',TH/'back/appendix.tex',TH/'back/publications.tex']:
        if p.is_file():
            r=Resolver(TH);m=counter.run(r.expand(p),'inventory_'+p.stem);ancillary.append(dict(path=rel(p),enabled=p.resolve() in graph.seen,status='ACTIVE' if p.resolve() in graph.seen else 'INACTIVE',metrics=m))
    pdf=TH/'build/thesis.pdf';pdfinfo={'current_pdf_pages':'UNKNOWN','artifact':rel(pdf),'artifact_pages':None,'status':'NOT_FOUND'}
    if pdf.is_file():
        pdfinfo['status']='STALE' if any(p.stat().st_mtime>pdf.stat().st_mtime for p in graph.files+[TH/'reference.bib']) else 'EXISTING_NOT_REBUILT_OR_VERIFIED';pdfinfo['mtime']=datetime.datetime.fromtimestamp(pdf.stat().st_mtime,datetime.timezone(datetime.timedelta(hours=8))).isoformat()
        try:
            # Use existing optional PyMuPDF only; no dependency installation.
            sys.path.insert(0,str(TH/'.tools/python'));import pymupdf
            with pymupdf.open(pdf) as doc:pdfinfo['artifact_pages']=len(doc)
        except ImportError:pdfinfo['page_reader']='PyMuPDF unavailable; artifact page count UNKNOWN'
    bib=(TH/'reference.bib').read_text();bibliography={'enabled':bool(re.search(r'\\bibliography\s*\{',masked((TH/'thesis.tex').read_text()))),'word_count':0,'status':'NOT_ENABLED','verified_bib_entry_count':len(re.findall(r'@\w+\s*\{',bib)),'note':'Bib entries are metadata, not bibliography prose. Existing smoke-test bbl is never treated as thesis bibliography.'}
    # A future enabled bibliography requires a current .bbl: do not silently label it measured zero.
    if bibliography['enabled']:bibliography.update(word_count=None,status='UNKNOWN_REQUIRES_CURRENT_BBL')
    warnings=graph.warnings+graph.missing
    result={'generated_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'engine':subprocess.check_output(['perl',str(ENGINE),'-version'],text=True).strip(),'engine_path':rel(ENGINE),'engine_sha256':sha(ENGINE),'rules_sha256':sha(RULES),'metric':'TeXcount -utf8 -ch; sum=1,1,1,0,0,0,0; include text+headings+captions/footnotes/table cells; no math weight','target':[49500,56500],'center':53000,'actual_include_graph':graph.record(),'actual_compilation_source_body':actual_body,'actual_full_source':actual_full,'planned_body_known_subtotal':sum(c['actual_count'] or 0 for c in planned),'planned_known_progress_percent':100*sum(c['actual_count'] or 0 for c in planned)/cfg['center'],'planned_gap_to_global_minimum_from_known':cfg['count_range'][0]-sum(c['actual_count'] or 0 for c in planned),'planned_missing_chapters':[c['number'] for c in planned if c['status']=='MISSING'],'planned_unique_source_graph':planned_res.record(),'chapters':planned,'sections':all_sections,'front_active_count':sum(f['total'] for f in file_counts if f['category']=='front'),'actual_files':file_counts,'ancillary':ancillary,'bibliography':bibliography,'pdf':pdfinfo,'warnings':warnings,'source_hashes':{rel(p):sha(p) for p in graph.files},'tests_passed':all(t['passed'] for t in tests),'limitations':['Static literal include/include only, not arbitrary TeX macro expansion.','Missing sources/sections are not measured zeros.','Full TeX rendering, class-generated cover and stale bibliography not counted as prose.','Code blocks/lines use a separate descriptive inventory; not TeXcount word counts.','Old six-chapter scaffold is not automatically mapped onto seven planned chapters.']}
    save(out/'THESIS_WORD_COUNT.json',result);save(out/'run_commands.json',counter.commands)
    rows=[]
    for c in planned:rows.append(dict(**{k:(c['metrics'][k] if c['metrics'] else None) for k in COUNTER_NAMES},kind='chapter',id=str(c['number']),title=c['title'],source=c['source_path'],status=c['status'],minimum=c['minimum'],maximum=c['maximum'],center=c['center'],actual=c['actual_count'],completion_percent=c['completion_percent'],gap_to_minimum=c['gap_to_minimum']))
    for s in all_sections:rows.append(dict(**{k:(s['metrics'][k] if s['metrics'] else None) for k in COUNTER_NAMES},kind='section',id=s['id'],title=s['title'],source=s['source_path'],status=s['status'],minimum=s['minimum'],maximum=s['maximum'],center=s['center'],actual=s['actual_count'],completion_percent=s['completion_percent'],gap_to_minimum=s['gap_to_minimum']))
    for f in file_counts:rows.append(dict(kind='actual_file',id=f['path'],title=f['title'] or '',source=f['path'],status='ACTIVE',actual=f['total'],**{k:f[k] for k in COUNTER_NAMES}))
    for a in ancillary:rows.append(dict(kind='ancillary',id=a['path'],title='',source=a['path'],status=a['status'],actual=a['metrics']['total'],**{k:a['metrics'][k] for k in COUNTER_NAMES}))
    rows.append(dict(kind='summary',id='actual_body',title='六章当前编译源正文',source='thesis/thesis.tex',status='MEASURED',actual=actual_body['total'],**{k:actual_body[k] for k in COUNTER_NAMES}))
    rows.append(dict(kind='bibliography',id='bibliography',title='参考文献列表',source='thesis/thesis.tex',status=bibliography['status'],actual=bibliography['word_count']))
    columns=['kind','id','title','source','status','minimum','maximum','center','actual','completion_percent','gap_to_minimum']+COUNTER_NAMES
    with (out/'THESIS_WORD_COUNT.csv').open('w',newline='') as fh:w=csv.DictWriter(fh,fieldnames=columns);w.writeheader();w.writerows(rows)
    n=lambda v:'MISSING' if v is None else str(v)
    report=['# 论文自动字数统计','',result['generated_at'],'',result['engine'],'',result['metric'],'','**规划目标，不是学校官方要求。MISSING不等于实测0字；完成率是篇幅进度，不是研究完成度。**','','## 当前真实结果','',f"- 当前六章编译源正文合计：{actual_body['total']}（text {actual_body['text']}，标题 {actual_body['headings']}，图表题/脚注 {actual_body['captions_footnotes']}）。",f"- 七章规划可匹配已测小计：{result['planned_body_known_subtotal']}；缺章：{result['planned_missing_chapters']}。其余缺失不作伪测量；按已匹配文字计，中心目标进度 {result['planned_known_progress_percent']:.2f}%，距49500尚差 {result['planned_gap_to_global_minimum_from_known']}。",f"- 正文行内公式 {actual_body['inline_math']}，独立公式 {actual_body['display_math']}；代码 {actual_body['code_blocks']}块/{actual_body['code_physical_lines']}物理行，均不计目标词数。",f"- 当前活动前置源计数：{sum(f['total'] for f in file_counts if f['category']=='front')}（摘要占位；宏生成封面未展开）；完整主文件静态计数 {actual_full['total']}，不与正文目标混用。",f"- 当前PDF页数：UNKNOWN；已有PDF：{pdfinfo['status']}，旧产物页数 {pdfinfo['artifact_pages']}。未重新编译，不能将旧PDF页数配给本轮正文。",f"- 参考文献列表：{bibliography['status']}；已收录BibTeX条目 {bibliography['verified_bib_entry_count']}，不作为正文词数。",'','## 七章进度','','|章|题名|最低–最高|中心|当前|完成率（中心）|距最低缺口|状态|','|---|---|---:|---:|---:|---:|---:|---|']
    for c in planned:report.append(f"|{c['number']}|{c['title']}|{c['minimum']}–{c['maximum']}|{c['center']:g}|{n(c['actual_count'])}|{c['completion_percent']:.2f}%|{c['gap_to_minimum']}|{c['status']}|" if c['actual_count'] is not None else f"|{c['number']}|{c['title']}|{c['minimum']}–{c['maximum']}|{c['center']:g}|MISSING|UNKNOWN|UNKNOWN|MISSING|")
    report+=['','## 42节预算与实际','','|节|标题|最低–最高|当前|中心完成率|距最低缺口|状态|','|---|---|---:|---:|---:|---:|---|']
    for s in all_sections:report.append(f"|{s['id']}|{s['title']}|{s['minimum']}–{s['maximum']}|{n(s['actual_count'])}|"+(f"{s['completion_percent']:.2f}%|{s['gap_to_minimum']}" if s['actual_count'] is not None else 'UNKNOWN|UNKNOWN')+f"|{s['status']}|")
    report+=['','## 六章实际include文件（不套用七章预算）','','|文件|实际章标题|统计量|正文/标题/其他|内容状态|','|---|---|---:|---|---|']
    for f in file_counts:
        if f['category']=='body':report.append(f"|{f['path']}|{f['title']}|{f['total']}|{f['text']}/{f['headings']}/{f['captions_footnotes']}|{f['content_status']}|")
    report+=['','## 第一章细分（TeXcount实际subcounts）','','|层级与标题|正文|标题|图表题/脚注|合计|','|---|---:|---:|---:|---:|']
    for sub in planned[0]['metrics']['subcounts'] if planned[0]['metrics'] else []:report.append(f"|{sub['title']}|{sub['text']}|{sub['headings']}|{sub['captions_footnotes']}|{sub['total']}|")
    report+=['','## 前后置库存（与七章正文分开）','','|文件|是否启用|统计量|','|---|---|---:|']
    for a in ancillary:report.append(f"|{a['path']}|{a['status']}|{a['metrics']['total']}|")
    report+=['','封面hitsetup是元数据而非直接排版文字，按规则忽略，0不表示最终封面无文字。摘要/致谢等当前为占位符，记录其真实计数但不当成已完成内容。未启用的附录、符号表和成果目录只作库存。', '', '## 结构差异与待写内容','',f'V2逐章路径和状态由JSON自动读取。当前缺失的规划章节为：{result["planned_missing_chapters"]}。旧骨架各章实际标题与计数见上表，不自动改名映射为新章节。脚本没有创建章节或修改include。', '', '第四章有条件保留：先补来源、lineage、采样/动作关联、审计与统计证据；其他缺口按THESIS_LENGTH_BUDGET.md和原规划论证卡处理，不自动填正文。','','## 核验与限制','','- tests.json保存已知答案的中文、英文、标题、表格、脚注、公式、注释、引用、代码及input/include测试。','- wordcount_audit.html为真实TeXcount着色解析；raw/为原输出，generated/为统计副本。','- 官方规则文档：https://tug.ctan.org/support/texcount/doc/TeXcount.pdf','- 静态文件名去重不等价任意TeX程序执行；动态输入/自定义宏需规则与样例。','- 只计数，不编译、不训练、不修改正式正文、文献或模板。',f"- include解析警告：{warnings}",'','操作：项目根目录执行 `./scripts/count_thesis_words.sh --all`。按V2规划章：`--chapter 1`；按节：`--sections`；文件：`--files`；解析：`--audit`。','']
    (out/'THESIS_WORD_COUNT.md').write_text('\n'.join(report))
    bad=[p for p,h in formal_before.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h];assert not bad,bad;save(out/'integrity.json',dict(protected_files=len(formal_before),changed=bad,formal_files_unchanged=True,before_hashes=formal_before,note='Per-run snapshot: authorized future writing is allowed; this script itself must not modify formal sources.'))
    # Detail modes affect console only; all reports are always reproducible and complete.
    if args.chapter:
        c=planned[args.chapter-1];print(f"Chapter {c['number']} {c['title']}: {n(c['actual_count'])}; target {c['minimum']}–{c['maximum']}")
        for s in c['sections']:print(s['id'],s['title'],n(s['actual_count']),s['status'])
    elif args.sections:
        for s in all_sections:print(s['id'],s['title'],n(s['actual_count']),s['status'])
    elif args.files:
        for f in file_counts:print(f['path'],f['total'])
    else:print(f"Current six-chapter source body: {actual_body['total']}; seven-chapter measured subtotal: {result['planned_body_known_subtotal']}; missing: {result['planned_missing_chapters']}")
    print('Reports:',out/'THESIS_WORD_COUNT.md');print('Audit:',out/'wordcount_audit.html')
if __name__=='__main__':main()
