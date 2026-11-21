from pathlib import Path
import json,re,hashlib,difflib,subprocess
O=Path('docs/writing_audit/ch05_writing_v1_2026-11-20');T=O/'section_5_3.tex';tex=T.read_text();wc=json.loads((O/'wordcount/result.json').read_text());base=json.loads((O/'baseline.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Produce the reading draft from the exact section source, not an independent rewrite.
reading=re.sub(r'^%.*\n','',tex,flags=re.M)
reading=re.sub(r'\\section\{([^}]+)\}',r'## 5.3 \1',reading)
names={'fig:ch05_gmm_physical_pipeline':'方法流程图','fig:ch05_gmm_candidate_selection':'候选关系图','tab:ch05_probe_dictionary':'字典结构表'}
def fig(m):
 x=m[0];caption=re.search(r'\\caption\{([^}]+)\}',x)[1];inner=re.search(r'\\begin\{minipage\}\{[^}]+\}\\centering(.*?)\\end\{minipage\}',x,re.S)[1]
 inner=re.sub(r'\\\\(?:\[[^]]+\])?','\n',inner).replace('\\rightarrow','→').replace('\\downarrow','↓').strip()
 return '\n**方法示意：'+caption+'**\n\n'+ '\n'.join('> '+line.strip() for line in inner.splitlines() if line.strip())+'\n'
reading=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',fig,reading,flags=re.S)
def table(m):
 x=m[0];caption=re.search(r'\\caption\{([^}]+)\}',x)[1];inner=x[x.index('\\hline')+len('\\hline'):x.index('\\end{tabular}')].replace('\\hline','')
 rows=[r.strip() for r in re.split(r'\\\\',inner) if r.strip()];out=['\n**表：'+caption+'**\n','|'+rows[0].replace('&','|')+'|','|---|---|']+['|'+r.replace('&','|')+'|' for r in rows[1:]]
 return '\n'.join(out)+'\n'
reading=re.sub(r'\\begin\{table\}.*?\\end\{table\}',table,reading,flags=re.S)
reading=reading.replace('\\begin{equation}','\n$$').replace('\\end{equation}','$$\n')
reading=re.sub(r'\\label\{[^}]+\}','',reading)
for k,v in names.items():
 reading=reading.replace(('表' if k.startswith('tab:') else '图')+'\\ref{'+k+'}',v).replace('\\ref{'+k+'}',v)
reading=reading.replace('~',' ').strip()+'\n'
(O/'SECTION_5_3_READING_DRAFT.md').write_text('# 第五章5.3阅读稿\n\n状态：阶段3，待用户与ChatGPT审核；未编译、未冻结。以下由section_5_3.tex转换，示意图为实现流程，不是实验截图。\n\n'+reading)
# Semantic static checks do not substitute for the unapproved PDF stage.
labels=re.findall(r'\\label\{([^}]+)\}',tex);refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',tex);stack=[];errs=[]
for m in re.finditer(r'\\(begin|end)\{([^}]+)\}',tex):
 if m[1]=='begin':stack.append(m[2])
 elif not stack or stack.pop()!=m[2]:errs.append(m[0])
assert not stack and not errs
assert len(labels)==len(set(labels)) and not set(refs)-set(labels)
existing_labels=[]
for p in Path('thesis/body').rglob('*.tex'):existing_labels+=re.findall(r'\\label\{([^}]+)\}',p.read_text())
assert not set(existing_labels)&set(labels)
brace=0
for i,c in enumerate(tex):
 if c in '{}' and (i==0 or tex[i-1]!='\\'):
  brace+=1 if c=='{' else -1;assert brace>=0
assert brace==0
assert len(re.findall(r'\\begin\{figure\}',tex))==2 and len(re.findall(r'\\begin\{table\}',tex))==1 and len(re.findall(r'\\begin\{equation\}',tex))==3
assert 1800<=wc['total']<=2000 and not re.search(r'\\cite(?:\[[^]]*\])?\{',tex)
changed=[p for p,h in base['protected_before'].items() if sha(p)!=h];assert not changed
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert head==base['HEAD']
engine=Path('thesis/.tools/texcount/texcount.pl');rules=Path('scripts/texcount_rules.tex');prior=json.loads(Path('docs/writing_audit/ch04_writing_v1_2026-11-20/revision_v2/wordcount/THESIS_WORD_COUNT.json').read_text());assert sha(engine)==prior['engine_sha256'] and sha(rules)==prior['rules_sha256']
checks={'status':'5.3 stage3 draft ready, waiting review','count':wc,'figures':2,'tables':1,'equations':3,'labels':labels,'missing_reference_targets':[],'duplicate_labels':[],'environment_errors':[],'braces_balanced':True,'new_citations':[],'compiled':False,'frozen':False}
(O/'static_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
(O/'WORDCOUNT_REVIEW.md').write_text(f'''# 5.3实际TeXcount

**实测{wc['total']}，符合1800–2000目标。** 统计工具为项目原有TeXcount 3.1.1中文模式；复用`scripts/count_thesis_words.py::Counter`及原规则，没有用正则估字数，没有改统计规则。

- 正文与表格、流程框文字：{wc['text']}。
- 节标题：{wc['headings']}。
- 图表标题：{wc['captions_footnotes']}。
- 合计：{wc['total']}。
- 行内数学：{wc['inline_math']}；独立公式：{wc['display_math']}，均不折为正文统计量。

两图一表的文字已由原引擎独立检查，分项见wordcount/float_counts.json；完整解析着色见wordcount/wordcount_audit.html，原始输出见wordcount/raw/section_5_3.txt。

命令：`PYTHONDONTWRITEBYTECODE=1 python3 docs/writing_audit/ch05_writing_v1_2026-11-20/count_section.py`。

现有全章CLI仅接受七章规划路径，不支持指定节片段；本次只调用其同一个Counter计算独立section_5_3.tex，没有创建虚假5.1/5.2或修改主文件。文件未接入正式编译树，不将该值称为第五章全章字数；PDF页数UNKNOWN，未编译。
''')
(O/'EVIDENCE_CITATION_MAP.md').write_text('''# 5.3逐段论点—实现—数据对应

根目录简称：G=`outputs/tactile_gmm_dual_vae_2026-11-16/`；C3=`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/`；审计A=`docs/writing_audit/ch05_material_audit_v1_2026-11-20/`。源代码均只读，不导入执行。既有审计完整索引与本次输入Hash保留在baseline.json。

|正文位置|论点与公式/图表|代码、配置与数据|证据及边界|
|---|---|---|---|
|第1段、流程图|无同步物理配对，以Real统计引导物理Sim；不直接画高斯响应|A/CH05_MATERIAL_AUDIT.md；C3/code/expand.py 41–80、inverse_physics.py::run；既有传感器调用链|L2代码；未新物理执行，不声明迁移成功|
|第2段、式GMM|网格、质心偏移、分Session抽样、48/16/2048|G/code/fit_gmm.py::patch_frames、samples，第6–18行；gmm.py::grid|L2；正值P5仅内部拟合筛选，无Real Validation拟合|
|第3段|完整协方差、EM、K1/2/3、60次、0.0025/0.0004正则、BIC、稳定性回退、ALL阶段|G/code/gmm.py::fit_em，第10–17行；fit_gmm.py第20–41行；C3/expand.py g[components][0,pi]|L2；不是VAE latent的GMM，不将48帧说成48独立主体|
|第4–5段、式责任度|w为四帧均值，责任度、响应量m、加权中心c；拟合/逆构造坐标差异|C3/code/expand.py第39–51行；inverse.py同构段；G/fit_gmm.py::samples|L2；公式含正常分量后验关系，代码用logsumexp稳定计算；质量m称响应量，不赋予物理质量单位|
|字典表|16×5×3×2=480，4mm，32帧取12–15，160指纹|C3/code/dictionary.py 11–25行；dictionary_complete.json；A/verified_observations.json dictionary及dedicated_array_contracts|L2已审计Manifest/数组和本轮代码复核；不称独立Root|
|字典检索段|Top-5、posture0/0.25mm非零条目、最近响应中心、深度插值、四帧profile|C3/code/expand.py第52–60行；dictionary_plan与registration；inverse_physics.py profile命令|L2；深度非真实材料变形，两域processed没有共同力学标定|
|反馈段、候选关系图|倍率范围0.5–2、基础深度0.1–1mm；最多3次候选，首个通过即停止|C3/code/expand.py第64–80行及registration.json|L2；真实代码反馈含指数1/0.7；正文简述未引入新计算方法|
|选择公式及解释|先在已执行候选中取结构代价最小，再检查准入；相似度/锚点/协方差与0.8/0.2/0.1门槛|C3/code/matching.py::inherited_metrics；expand.py第81–87行|L2；不是对所有未知物理条件求全局最优，也不是先过滤合格候选再取最优；距离在索引空间|
|登记段|固定元数据顺序、无C0评分、Real/Sim数值重复检查和来源绑定|C3/code/expand.py第20–38、81–91行；pair_expansion_manifest及既有审计|L2；不同Hash不证明独立Root；失败结果保留，不在本轮重建|
|末段|1200/240，Top-5与单一对应丢失的多解信息；Natural覆盖不获保证|C3/expansion_complete.json；A/verified_observations.json dedicated_array_contracts、来源重计；C3/pipeline_cpu.py读取sim/real生成潜变量监督|L2；不引用任何最终迁移通过率，不提前写5.4/5.5模型细节；未来软监督收益未验证|

## 公式和实现等价范围

式GMM描述二维统计密度；式责任度用固定网格q代入冻结参数，明确没有假设它与拟合偏移u具有已认证物理对应。BIC中的6K−1来自二维均值、完整对称协方差与混合权重自由度。选择代价与代码一致；距离无双方活跃帧时不满足登记条件，实际实现为选优提供有限替代项，不在公式中伪装存在有效质心距离。

图均为代码流程示意，不含仿造的GMM椭圆、接触截图或成绩。正文不新增外部引文、不修改BibTeX。项目方法事实由原代码和登记产物支撑；没有用新的未经核验文献补齐。
''')
(O/'WRITING_REVIEW.md').write_text('''# 5.3写作自审

状态：**阶段3初稿，等待用户与ChatGPT审核；未编译、未冻结。**

## 十项逐项回答

1. **GMM建模对象**：每Patch在固定索引网格上、按Real响应抽样并减去帧质心得到的二维扩散偏移点。
2. **与VAE潜空间区分**：正文明确统计空间不是原始712D数组，也不是后续VAE latent；本节没有计算两套独立latent间距离。
3. **软责任度作用**：将Taxel响应分配到多个分量，形成m及加权中心c；不是完全没有利用软信息。
4. **如何到物理探针**：分量筛选→字典响应中心选位置→响应/深度插值与帧profile→球形碰撞体实际执行→原有touch及processed读出。
5. **选择机制**：至多3个实际候选，遇到通过者停止，在已执行集合取结构代价最小者，再作门槛和重复检查；没有改为全局最优或合格子集最优。
6. **不能称物理真配对的原因**：无同步接触过程，物理Taxel对应尚未认证，匹配仅约束摘要，索引距离非物理接触误差。
7. **两次压缩的信息损失**：Top-5可能删弱分量；单一最佳对应舍弃替代物理条件及多解概率。没有把这些概率当已实现的Translator软监督。
8. **实现与未来分开**：实际流程、字典与指定配对数量有证据；未来保留概率、多候选及软监督的收益未验证。本轮不实施改进。
9. **章节边界**：不复述传感器建模、数据字段与存储合同；仅一句说明配对向后续Translator提供监督，不写其模型、训练损失和最终结果。
10. **没有扩大为物理语义一致性**：明确单位、坐标、同步与独立性边界，不宣称完整迁移成功，不引用C3/C13/C23/C24/C27最终通过率。

## 组织和表达

正文按监督缺口→Train-only先验→分量分解→字典与加载→实际候选/准入→信息损失组织，未列额外小标题，未写成文件和函数罗列。用“分量响应量”而非物理质量称m；公式后的变量含义与实际计算对应。两张流程图任务不同：一张解释统计到物理输出，一张解释多解信息如何压缩；表格集中登记字典结构。

## 静态检查和剩余事项

3个核心公式、2图、1表；标签无重复，ref目标均存在，环境和花括号配对正常，无新增cite。使用已有模板支持的figure/table/fbox/minipage、tabular与equation/aligned，不增加宏包。未编译，因此不宣称图宽、公式断行和PDF页码已经验收；这些留待用户批准后的编译阶段。

section_5_3.tex作为独立节片段保存于本工作目录，未创建或填充5.1/5.2等内容，未接入正式主文件，也未覆盖旧六章骨架。独立片段在未来第五章上下文才显示5.3编号；本轮没有人为修改章节计数。

没有需要猜测补全的数值。Real物理映射/同步关系为未认证，不补写为确定关系；Natural覆盖仍不能由统计准入保证。正式创新性与后续实验结论留待相应章节比较，不因本节完成而扩大主张。
''')
# Diff of authored deliverables (new files only); old audit and formal source remain untouched.
files=['section_5_3.tex','SECTION_5_3_READING_DRAFT.md','EVIDENCE_CITATION_MAP.md','WORDCOUNT_REVIEW.md','WRITING_REVIEW.md','count_section.py','package_section.py']
patch=''
for n in files:
 p=O/n;patch+=''.join(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile=str(p)))
(O/'changes.diff').write_text(patch)
integrity={'task':'CH05-5.3 stage3 only','HEAD_before':base['HEAD'],'HEAD_after':head,'git_status_before':base['git_status'],'git_status_after':subprocess.check_output(['git','status','--short'],text=True),'protected_hashes':base['protected_before'],'changed_protected_files':changed,'read_inputs':base['read_inputs'],'section_sha256':sha(T),'texcount_engine_sha256':sha(engine),'texcount_rules_sha256':sha(rules),'same_engine_and_rules_as_prior_verified_count':True,'only_section_5_3_written':True,'no_formal_main_include_change':True,'other_chapters_unchanged':True,'reference_bib_unchanged':True,'Test_Holdout_payloads_read':False,'new_models_fitted':0,'physics_runs':0,'dataset_generation':0,'PDF_compiled':False,'frozen':False,'commit_push':False,'no_next_section':True}
(O/'integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+'\n')
(O/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p}\n' for p in sorted(O.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
print('PASS:',wc['total'],'2 figures /1 table /3 equations;',len(base['protected_before']),'protected unchanged')
