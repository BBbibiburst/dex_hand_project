from pathlib import Path
import json,csv,collections
r=Path(__file__).resolve().parents[3];d=Path(__file__).parent
ann=json.loads((d/'annotations_original.json').read_text())
# Status means the narrow original comment, not approval of the entire section.
done={1:'中文封面日期统一为2026年12月；CSV同步。',3:'内封去除14日；CSV同步。',4:'中文专业改为计算机技术；CSV同步。',5:'英文专业改为Computer Technology；CSV同步。',6:'英文日期改为December 2026。',14:'英文摘要根据本轮中文摘要同步重写。',15:'绪论起句改为多指接触与稳定持物的具体问题。',17:'补充初态保存/恢复、完整状态读取、碰撞筛查与失败复现的作用。',18:'绪论首次说明RealMAN（睿尔曼智能）RM75B。',20:'改为解释TACTO和Taxim的读出机制，不重复否定其作为本文物理依据。',30:'首次解释MJCF及模型文件组织内容。',32:'删除闭链/Flex/硅胶的连续否定，改为说明刚体与固定腱模型。',34:'删除段落中重复的非任务姿态声明，保留图题对静态模型的说明。',38:'直接解释固定腱位置由关联关节加权组合确定。',39:'保留默认参数可覆盖的必要说明，删去逐历史Episode否定。',40:'删去传感器带宽旁论，说明积分与采样间隔。',41:'删去电极等价性的重复旁论，说明安装与通道顺序。',43:'改为说明各区域分别从STL计算采样点。',47:'补入五条历史仿真记录的完整序列热图，来源及Hash见figure_sources.json；不是实机测量图。',51:'当前第一、三章的本项目改为本文。',54:'删除未完整复现Isaac的独立重复声明，保留DexEvolve设计参考与适配定位。'}
partial={7:'摘要已按实机采集代价→仿真用途→研究问题重写；新增实验对照尚未开展。',8:'摘要加入RealMAN（睿尔曼智能）；官方品牌来源见本轮报告。',9:'摘要区分指标适配与参考设计的搜索，加入既有10条局部扰动Strict记录；未虚构原创性或方法优势。',10:'摘要按平台/示教→数据质量→跨域方法与评价衔接，尚待人工确认。',11:'压缩数据处理细项，保留指定340行字段一致性结果。',13:'加入抓取与指定数据修复结果；迁移未通过事实保留于结果结尾，不能改写为成功。',16:'摘要改以采集操作、复位和维护解释实机代价，其他背景段落仍需逐段复核。',22:'将已有平台相关工作移到1.2.1，再接抓取和数据；SAPIEN/robosuite/ManiSkill原文调研未完成。',33:'删除堆叠的否定句，正面说明资产与模型表示；实物几何/物理误差测量未补。',42:'说明阵列按指节/手掌感知区组织；尚不能认证与实物电极逐点配准。',45:'删去突然出现的touch_grid插件并说明box与touch绑定；原生读出与压力语义仍在本节必要位置说明。',49:'补五任务操作对象说明及历史视频起终帧；更详细的各任务目标/成功判据图仍待补。',55:'改为说明档案保留不同构型、后续执行验证；尚无公平多样性基准。'}
conflict={12:'按字面改为默认物理一样将改变事实，未执行。摘要改以统计先验构造伪配对正面叙述；同步物理真值与统计配对仍须区分。',23:'原批注要求SAPIEN/欠驱动优势解释，不能无对照断言其他引擎不能模拟欠驱动；待原文与功能核验。'}
pending={2:'字体保持当前模板Fandol；须先核对学院正式字体要求及字体授权，再决定替换。',19:'执行内部时序关联与跨设备物理配对需在该段具体改写，本轮未将该条标完成。',21:'绪论已说明硬件选择对应实验室；逐模型几何/动力学一致性仍待证据，不能扩成全参数对齐。',24:'应补作者实际遥操作困难及经核验相关文献；本轮未添加未经查证文献。',25:'Narang方法与本文工作的对照需按监督来源/读出差异重新组织。',26:'需具体解释两域接触几何、加载和时间同步条件，避免再堆缺少配对声明。',27:'表1-1留白需在完整排版审核中定位；本轮未靠补字填空。',28:'该句需在上下文中换为执行结果与记录质量分别判定。',29:'需以真实模块输入/输出、控制周期和模型关联重画架构，未将换样式视为完成。',31:'需实际关节轴与固定腱驱动标注，不能画替代机构示意。',35:'需逐一核对接触维数、摩擦及求解参数含义后重写。',36:'需将本文控制器设计与MuJoCo位置执行器职责分开，补实际控制结构。',37:'控制图需Jacobian/求解/限幅/执行时序等实际分支。',44:'需对三类实际采样算法按曲率适用条件逐步解释；已有代表形状图不等于该条全部完成。',46:'需重画读出处理图，明确步骤、数组与实际非线性。',48:'2.6仍有多段重复边界；本轮仅加入实际观测热图与场景，不宣称该节已重构。',50:'图3-1需真正贯穿同一物体候选/可达性/轨迹与任务。',52:'需在官方指标调用与本文适配段落集中解释归属，尚未整体改写该段。',53:'图3-2需候选形状与物理反馈的真实例子。',56:'需结合实际腕姿网格与预算说明，删除重复独立性否定但保留统计单位。',57:'缺指定案例腕部扰动与可达性原始可视化。',58:'需读取指定轨迹坐标画连续IK/执行对照，不能用框图替代。',59:'第二章已加五任务场景；第三章专家阶段图还需关联真实轨迹分段。',60:'Push定义仍需定点正面改写。',61:'失败案例图题与正文需集中说明记录范围，未将已有图当作闭合。',62:'需将n14/n127批次条件移入统一比较说明，再分析能比较的结果。',63:'训练角色限制应集中到第四章准入，本轮未修改该句。',64:'数据载体图需真实文件/索引/父来源实例，不能再画通用方框。'}
def source(i):
 if i<=6:return 'thesis/front/cover.tex'
 if i<=14:return 'thesis/front/abstract.tex'
 if i<=27:return 'thesis/body/ch01_introduction.tex'
 if i<=49:return 'thesis/body/ch02_simulation_platform.tex'
 if i<=63:return 'thesis/body/ch03_grasp_demonstrations.tex'
 return 'thesis/body/ch04_multimodal_data.tex'
rows=[]
for a in ann:
 i=a['id'];p=source(i);lines=(r/p).read_text().splitlines();mark=a['marked_text'].replace(' ','');scores=[len(set(mark)&set(x)) for x in lines];line=max(range(len(lines)),key=lambda j:scores[j])+1
 # Approximate matching is explicitly marked, not presented as precise text alignment.
 status='已处理（待人工验收）' if i in done else '部分处理' if i in partial else '证据限制/不按字面执行' if i in conflict else '待处理'
 note=done.get(i,partial.get(i,conflict.get(i,pending.get(i,'待逐段核对'))))
 rows.append(dict(id=i,pdf_page=a['page'],comment=a['comment'].strip(),original_marked_text=a['marked_text'],status=status,source=p,source_line_hint=line,line_hint_method='字符重合定位，仅导航，不是精确批注锚点',action_or_remaining=note))
with (d/'ANNOTATION_LEDGER.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
md=['# 根目录PDF原始64条批注落实表','来源：`thesis_draft_commented.pdf`；旧`references/thesis_draft_commented.pdf`不是本次批注源。保留原始批注、选中文本与页码于annotations_original.json。行号仅作导航，详细修改以changes.diff为准。','本表的“已处理”仅指该条窄义文字/图示要求已实施，均待人工验收，不表示整章或科研补强完成。']
for row in rows:
 md += [f"\n## {row['id']:02d}｜原PDF第{row['pdf_page']}页｜{row['status']}",'> '+row['comment'].replace('\n',' '),f"来源：`{row['source']}`；{row['action_or_remaining']}"]
(d/'ANNOTATION_LEDGER.md').write_text('\n\n'.join(md)+'\n')
print(dict(collections.Counter(x['status'] for x in rows)))
