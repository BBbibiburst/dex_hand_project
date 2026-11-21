from pathlib import Path
import hashlib,json,re,difflib,shutil
import pymupdf as f
R=Path.cwd();O=R/'docs/writing_audit/ch04_writing_v1_2026-11-20/revision_v2';src=R/'thesis/body/ch04_multimodal_data.tex';old=(O/'ch04_before.tex').read_text();new=src.read_text();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pattern=r'\\begin\{figure\}.*?\\label\{fig:ch04_sink_precision\}.*?\\end\{figure\}'
# Match only the individual Sink float, not all preceding figure blocks.
blocks=lambda s:list(re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',s,re.S))
a=next(x for x in blocks(old) if 'fig:ch04_sink_precision' in x.group());b=next(x for x in blocks(new) if 'fig:ch04_sink_precision' in x.group())
assert old[:a.start()]+old[a.end():]==new[:b.start()]+new[b.end():]
assert re.search(r'\\caption\{([^\n]+)',a.group()).group()==re.search(r'\\caption\{([^\n]+)',b.group()).group()
protected=json.loads((O/'protected_before.json').read_text());changed=[p for p,h in protected.items() if sha(R/p)!=h];assert not changed
oldcheck=json.loads((O.parent/'revision_v1/static_checks.json').read_text());expectedlabels={k:v['number'] for k,v in oldcheck['compiled_labels'].items()}
B=O/'build/chapter04_preview_with_references';log=(B/'chapter04_preview_with_references.log').read_text();aux=(B/'chapter04_preview_with_references.aux').read_text();labels={k:{'number':n,'page':p} for k,n,p in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{([^}]+)\}',aux)}
assert expectedlabels=={k:v['number'] for k,v in labels.items()}
assert not re.search(r'Overfull|undefined|Undefined control sequence',log)
d=f.open(O/'chapter04_preview.pdf');dr=f.open(O/'chapter04_preview_with_references.pdf');assert len(d)==10 and len(dr)==11
oldpdf=f.open(O.parent/'revision_v1/chapter04_preview.pdf')
clean=lambda t:re.sub(r'\s+','',re.sub(r'-\s*\d+\s*-','',t.replace('哈尔滨工业大学硕士学位论文','')))
# Text is unchanged outside plot labels/ticks; the old source is the authoritative exact content comparison.
pagechecks=[]
for i,p in enumerate(d):
 txt=p.get_text();assert '\ufffd' not in txt
 pagechecks.append({'page':i+1,'chars':len(txt),'nonempty':len(txt)>100})
assert '在上述数据条件下' in d[-1].get_text() and '同步物理对应' in d[-1].get_text()
wc=json.loads((O/'wordcount/THESIS_WORD_COUNT.json').read_text());ch=next(c for c in wc['chapters'] if c['number']==4);n=ch['metrics']['total']
inputfile=R/'outputs/five_task_p38_exact_replay_2026-11-03/comparison.json';data=json.loads(inputfile.read_text());vals=[data['legacy_replay']['max_qpos_component_error'],data['exact_replay']['max_qpos_component_error']]
previous=json.loads((O.parent/'figure_provenance.json').read_text());assert vals==previous['values'] and sha(inputfile)==previous['input_sha256']
plot=f.open(R/'thesis/figures/ch04/sink_historical_precision_r2.pdf');oldplot=f.open(R/'thesis/figures/ch04/sink_historical_precision.pdf')
fontsizes=lambda doc:sorted({s['size'] for bl in doc[0].get_text('dict')['blocks'] if 'lines' in bl for ln in bl['lines'] for s in ln['spans']})
assert fontsizes(plot)==fontsizes(oldplot)
check={'status':'R2分页优化完成，等待人工审核；未冻结','body_pages_before':11,'body_pages_after':10,'with_references_pages':11,'texcount_before':7632,'texcount_after':n,'exact_source_unchanged_outside_sink_figure':True,'caption_unchanged':True,'figures':5,'tables':6,'equations':4,'compiled_labels':labels,'changed_protected_files':changed,'protected_file_hashes':protected,'plot_values_unchanged':vals,'source_hash':sha(inputfile),'plot_fonts_before':fontsizes(oldplot),'plot_fonts_after':fontsizes(plot),'page_checks':pagechecks,'overfull':[],'undefined':[],'underfull':re.findall(r'Underfull[^\n]*',log),'frozen':False}
(O/'integrity.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
(O/'changes.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='chapter04_R1.tex',tofile='thesis/body/ch04_multimodal_data.tex')))
shutil.copyfile(src,O/src.name)
(O/'LAYOUT_REVIEW.md').write_text(f'''# 第四章R2章末分页优化

状态：**临时预览完成，等待人工审核；未冻结。**

## 结果

正文由R1的11页缩为**10页**，4.6及末尾文字完整留在第10页；含参考文献版本为11页（第11页仅参考文献）。没有修改正式主文件，页码仅为独立预览页码。

## 唯一调整区域

仅图4-5浮动块及其新建R2排版资产调整。等比例缩小的初步尝试仍留第11页，且会缩小标签，因此未采用。最终保持原0.78 textwidth和原图10pt字号，将原图6.6×3.6英寸画布改为6.6×1.3英寸紧凑画布；纵轴标题只改换行，减少刻度标签密度，柱高、纵轴数值范围、两条测量数值和颜色不变。原图资产原样保留，新文件为sink_historical_precision_r2.pdf/png。

浮动选项改为[!htbp]，只在该浮动内部将abovecaptionskip设为0pt，并局部使用上方-10pt、下方-14pt的垂直补偿，消减图内与浮动边界留白；未更改全局字体、字号、页边距、行距或其他浮动体。最终图4-5位于第9页表4-6之后，图注与后续正文不重叠。

## PDF检查

已逐页查看10页缩略图，放大检查第9页Sink图/表与第10页小结。5张图、6张表和4个公式的编号均与R1相同；除图4-5及章末分页相关位置，前8页布局保持原有结构。无空白页、新孤立标题、越界或重叠；最后一段完整保留。

Sink图使用矢量PDF，字体及绘制宽度与R1相同，柱顶1.628897和0.000000标注、横轴标签和纵轴标题均可读。图形较矮是本轮局部紧凑排版选择，供人工确认。未变更图表数据、公式、表格数据、引用或任何段落。

最终日志无Overfull、未解析引用或编译错误；保留原有2项Underfull提示和模板字体尺寸/glossary提示。BibTeX仍为既有元数据缺项提示，未改文献。build.log保留全部尝试，包括一次不适用的captionsetup命令错误；最终已改为模板支持的局部setlength并编译成功，最终引擎日志在build/。

## 内容和保护

TeX源在图4-5浮动块之外逐字一致；图注也保持一致。原图数据来源Hash和数值与V1登记一致；图内字号与原图同为10pt。第一至三章、主文件、参考文献、模板/构建配置、生产代码、冻结PDF及原图的受保护Hash全部一致，见integrity.json。TeXcount为{n}，未修改统计规则。

不冻结，不commit/push，不进入第五章。
''')
(O/'WORDCOUNT_REVIEW.md').write_text(f'''# R2实际TeXcount

命令：`./scripts/count_thesis_words.sh --chapter 4 --output-dir docs/writing_audit/ch04_writing_v1_2026-11-20/revision_v2/wordcount`

R1：7632；R2：**{n}**。差值{n-7632}来自本次局部排版命令的工具解析，正式正文逐字未变；没有修改TeXcount规则。原始输出、JSON、CSV和审计HTML在wordcount/。

|节|实际统计量|
|---|---:|
'''+''.join(f"|{x['id']} {x['title']}|{x['actual_count']}|\n" for x in ch['sections'])+'\n正文预览10页，含参考文献11页。\n')
(O/'RUN_COMMANDS.json').write_text(json.dumps({'commands':['/home/zmy/miniconda3/envs/mujoco/bin/python docs/writing_audit/ch04_writing_v1_2026-11-20/revision_v2/plot_sink_compact.py','bash docs/writing_audit/ch04_writing_v1_2026-11-20/revision_v2/build_preview.sh','./scripts/count_thesis_words.sh --chapter 4 --output-dir docs/writing_audit/ch04_writing_v1_2026-11-20/revision_v2/wordcount','PYTHONPATH=thesis/.tools/python python3 docs/writing_audit/ch04_writing_v1_2026-11-20/revision_v2/package_r2.py'],'no_experiments':True},ensure_ascii=False,indent=2))
paths=[src,R/'thesis/figures/ch04/sink_historical_precision_r2.pdf',R/'thesis/figures/ch04/sink_historical_precision_r2.png']+[p for p in O.rglob('*') if p.is_file() and p.name!='SHA256SUMS']
(O/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(R)}\n' for p in sorted(paths)))
print('PASS: body10 / withrefs11 / count',n,'protected',len(protected))
