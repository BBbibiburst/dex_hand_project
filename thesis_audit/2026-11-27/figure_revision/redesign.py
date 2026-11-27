from pathlib import Path
import re,json,hashlib,difflib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.font_manager import FontProperties
root=Path('/home/zmy/dex_hand_project');out=Path(__file__).resolve().parent;assets=root/'thesis/figures/revision_20261127';assets.mkdir(exist_ok=True);before=out/'before';before.mkdir(exist_ok=True)
f=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');blue='#245878';light='#ecf3f8';orange='#b36c35'
# Labels abbreviate existing reviewed text; graph edges denote existing data/control dependencies.
spec={
'ch02_architecture':(['任务与初态\n配置入口','模型层\n机械臂 · 手 · 物体','控制层\n动作 → 执行器目标','感知层\n状态 · 触觉观测','任务层\n评价 · 终止','记录层\n动作—观测关联'],[(0,1),(1,2),(2,3),(3,4),(3,5),(4,5)]),
'ch02_drive':(['机械臂\n7个位置通道','机械臂\n7个转动关节','四指\n每指1个屈曲通道','固定腱\n每指2个转动关节','拇指对掌 / 屈曲\n2个独立控制通道','直接 / 固定腱\n1 + 3个转动关节'],[(0,1),(2,3),(4,5)]),
'ch02_control':(['position模式\n关节目标','ik模式\n位姿 → IK → 限步/平滑','位置执行器\n目标保持','物理推进\n50 × 0.001 s','观测读取\n状态 · 触觉','任务评价\n下一控制周期'],[(0,2),(1,2),(2,3),(3,4),(4,5)]),
'ch02_processing':(['曲面感知区域\nsite + native touch','原生读数\nsensordata','响应变换\n死区 → 截断 → 幂','时序与邻域\nIIR → 邻域扩散','可选处理\n高斯 / 噪声','规范读出\n归一化 → float32[712]'],[(0,1),(1,2),(2,3),(3,4),(4,5)]),
'ch02_validation':(['当前实现\nXML · Builder · 接口','历史产物\nNPZ · Manifest · 结果','模型与地址\n静态检查','数组合同\n形状 · 类型 · 有效值','历史观测\n非零 · 时间间隔','有限功能证据\n不替代动态复验/实机标定'],[(0,2),(1,3),(3,4),(2,5),(4,5)]),
'ch03_overall_pipeline':(['物体与任务\n几何 · 初态 · 条件','候选生成\n几何优化 / 物理反馈','机器人约束\n目标转换 · 可达筛选','轨迹执行\n腕姿扩展 · 分段控制','独立质量检查\nStrict Replay','专家示教\n任务结果 · 轨迹记录'],[(0,1),(1,2),(2,3),(3,4),(4,5)]),
'ch03_candidate_sources':(['物体表面\n六驱动手部代理','初始候选\nGraspQP指标 + 本地约束','反馈细化\nDexEvolve-style搜索','候选档案\n保留来源与质量记录','局部腕姿扩展\nWrist Lattice','条件后续分支\nMJWarp PPO模板/驱动编辑'],[(0,1),(1,2),(2,3),(3,4),(4,5)]),
'ch03_reachability':(['局部腕姿编辑\n平移 · 旋转 · 耦合格点','目标坐标转换\n物体 → 世界 → 安装补偿','预抓取目标\nIK求解','抓取目标\nIK求解','有限候选排序\n残差 · 编辑幅度 · 跟踪误差','动态执行\n不等同全轨迹碰撞认证'],[(0,1),(1,2),(2,3),(3,4),(4,5)]),
'ch03_strict_replay':(['当前状态\n分段插值目标','在线控制\nIK → 位置命令','物理执行\n轨迹与阶段记录','独立回放\n低层ctrl序列','联合质量检查\n抬升 · 速度 · 接触 · 穿透','状态登记\nVERIFIED / REJECTED'],[(0,1),(1,2),(2,3),(3,4),(4,5)]),
'ch03_task_transfer':(['Lift抓持与承载\n共用轨迹来源','PickPlace\n搬移 · 释放后缀','Stack\n支撑 · 稳定放置后缀','Push专属控制\n接近 · 接触 · 反馈推动','NutAssembly专属控制\n抓持 · 姿态 · 对准下降','统一Episode合同\n不等同统一控制逻辑'],[(0,1),(0,2),(1,5),(2,5),(3,5),(4,5)]),
'ch04_carriers':(['源执行记录\nNPZ + Manifest','字段 / 时序 / 来源\n合同核对','规范modalities\n元数据 + 验证记录','导出映射\n按指定版本合同','LeRobot载体\n数值 · 视频 · Episode索引','Real独立来源\n不自动构成物理配对'],[(0,1),(1,2),(2,3),(3,4)]),
'ch04_admission':(['执行结果\n任务 / Strict状态','结构与数值\n维度 · 有效性 · 阶段','时序与语义\n动作—观测对应','来源与版本\nHash · 内容重复','VALIDATING后登记\nEXPORTED / DUPLICATE','异常隔离\nQUARANTINED'],[(0,1),(1,2),(2,3),(3,4),(1,5),(2,5),(3,5)]),
'ch05_gmm_physical_pipeline':(['Real Train\nPatch二维索引偏移','统计先验\n拟合并冻结GMM','目标分解\n分量响应 · 质心','字典与加载\n位置检索 · 深度估计','物理探针执行\n原生触觉 + 处理链','统计准入\n4×712伪配对登记'],[(0,1),(1,2),(2,3),(3,4),(4,5)]),
'ch05_gmm_candidate_selection':(['指定Real目标\n非同步物理真值','软分量分解\n启发式责任度与响应量','Top-5压缩\n按分量响应量排序','至多3次物理候选\n字典组合 · 执行 · 反馈','已执行集合取最小代价\n再检查最终准入条件','单一伪配对\n未保留完整多解概率'],[(0,1),(1,2),(2,3),(3,4),(4,5)]),
'ch05_c0_validation_protocol':(['原始训练数据\nReal + Natural Sim','原C0训练并冻结\n11个分类器实例','验证输入\nReal + Translated Sim','固定预处理\nlog1p · Scaler · 通道顺序','固定阈值0.5\nBA / AUC / D逐实例检查','联合验收\n11实例全部满足条件'],[(0,1),(2,3),(1,4),(3,4),(4,5)]),
}
def node(ax,xy,text):
 x,y=xy;ax.add_patch(FancyBboxPatch((x-.142,y-.11),.284,.22,boxstyle='round,pad=0.012,rounding_size=.018',fc=light,ec=blue,lw=1.2));ax.text(x,y,text,ha='center',va='center',fontproperties=f,fontsize=13,color='#1e3446',linespacing=1.5)
def draw(key,labels,edges):
 fig,ax=plt.subplots(figsize=(10,3.9));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');pos=[(.17,.74),(.5,.74),(.83,.74),(.17,.28),(.5,.28),(.83,.28)]
 # use snake order for sequential paths, with explicit directed edges
 pos=[(.17,.74),(.5,.74),(.83,.74),(.83,.28),(.5,.28),(.17,.28)]
 if key=='ch02_drive':pos=[(.2,.82),(.75,.82),(.2,.5),(.75,.5),(.2,.18),(.75,.18)]
 for a,b in edges:
  dx=pos[b][0]-pos[a][0];dy=pos[b][1]-pos[a][1];t=min(.159/abs(dx) if dx else 99,.129/abs(dy) if dy else 99)
  start=(pos[a][0]+t*dx,pos[a][1]+t*dy);end=(pos[b][0]-t*dx,pos[b][1]-t*dy)
  ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,color=blue,lw=1.2,shrinkA=0,shrinkB=0))
 for xy,t in zip(pos,labels):node(ax,xy,t)
 fig.savefig(assets/(key+'.pdf'),bbox_inches='tight',pad_inches=.03);plt.close(fig)
for k,(n,e) in spec.items():draw(k,n,e)
# Explicit train-vs-deployment and independent-domain network diagrams.
for key,labels,edges in [
('ch05_dual_vae',['Real 4×712\n独立输入Scaler','Real Encoder → Zr\n训练采样 / 推理均值','Real Decoder\n自回归Mask + Gamma幅值','Sim 4×712\nNatural与Paired混合','Sim Encoder → Zs\n与Real参数独立','Sim Decoder\n真实Mask训练 / 预测Mask推理'],[(0,1),(1,2),(3,4),(4,5)]),
('ch05_translator_paths',['Paired Sim → Es\n冻结Encoder取后验均值','Translator\n唯一更新模块','潜变量Huber\n配对Real经Er提供目标','冻结Real Decoder\n真实Mask条件NLL','Natural Sim → Es → T\n自由推理路径','Real Decoder自由读出\n预测Mask × Gamma混合均值'],[(0,1),(1,2),(1,3),(4,5)])]:draw(key,labels,edges)
# Timing and sampling figures use explicit time axes, not prose boxes.
for key in ['ch04_timing','ch04_multirate']:
 fig,ax=plt.subplots(figsize=(10,2.7));ax.set_xlim(-.8,9);ax.set_ylim(-.6,2.6);ax.axis('off')
 if key.endswith('multirate'):
  for y,title,idx in [(2,'控制行 / 状态 / 触觉（20 Hz）',range(9)),(1,'默认RGB（20 Hz）',range(9)),(0,'降频RGB示例（5 Hz）',range(0,9,4))]:
   ax.plot([0,8],[y,y],color='#aebfc9',lw=1);ax.scatter(list(idx),[y]*len(list(idx)),s=42,color=blue);ax.text(-.5,y+.25,title,fontproperties=f,fontsize=10)
  for i in range(9):ax.text(i,-.4,str(i),ha='center',fontsize=10)
 else:
  for y,title,seq in [(1.7,'执行前记录路径',[r'观测 $o_i^-$',r'动作 $u_i$',r'观测 $o_i^+$']),(0,'历史执行后记录路径',[r'上一行 $o_{i-1}^+$',r'动作 $u_i$',r'当前行 $o_i^+$'])]:
   ax.annotate('',xy=(8,y),xytext=(0,y),arrowprops={'arrowstyle':'->','color':blue});ax.text(0,y+.6,title,fontproperties=f,fontsize=11)
   for x,t in zip([1,4,7],seq):ax.scatter(x,y,s=50,color=blue);ax.text(x,y+.18,t,fontproperties=f,ha='center',fontsize=11)
 fig.savefig(assets/(key+'.pdf'),bbox_inches='tight',pad_inches=.05);plt.close(fig)
# Back up every changed source and replace only figure bodies; keep labels/captions.
changes=[]
for p in (root/'thesis/body').rglob('*.tex'):
 s=p.read_text();original=s
 def repl(m):
  b=m.group(0);a=re.search(r'\\label\{fig:([^}]+)\}',b)
  if not a or a.group(1) not in set(spec)|{'ch04_timing','ch04_multirate','ch05_dual_vae','ch05_translator_paths'}:return b
  key=a.group(1);cap=b[b.index('\\caption'):];return '\\begin{figure}[htbp]\n  \\centering\n  \\includegraphics[width=0.97\\textwidth]{figures/revision_20261127/'+key+'.pdf}\n  '+cap
 s=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',repl,s,flags=re.S)
 if s!=original:
  rel=p.relative_to(root);b=before/rel;b.parent.mkdir(parents=True,exist_ok=True);b.write_text(original);p.write_text(s);changes.append(str(rel))
(out/'changed_files.json').write_text(json.dumps(changes,indent=2));print('converted diagrams',len(spec)+4,'files',len(changes))
