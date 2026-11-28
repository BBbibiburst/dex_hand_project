from style import *
from matplotlib.patches import Rectangle,FancyArrowPatch
import numpy as np,pandas as pd
v4=R/'outputs/five_task_p44_integration_2026-11-05/dataset/episodes/pick_tomato_10e460303770c865bf7e';e=json.loads((v4/'episode.json').read_text());mod=v4/'modalities.npz';src=v4/'source_episode/episode.npz'
with np.load(mod,allow_pickle=False) as z:shapes={k:list(z[k].shape) for k in ['state','action','tactile','timestamp']};tact=z['tactile'][300].copy()
report=R/'outputs/dp_p2_2026-10-11/diagnostics/action_chain/action_chain_report.json';q=json.loads(report.read_text());sm=Path(q['manifest']);npz=sm.parent/'episode.npz';lp=R/'outputs/dp_p2_2026-10-11/lerobot_single_sugar_fixed/data/chunk-000/file-000.parquet';df=pd.read_parquet(lp,columns=['episode_index','frame_index','index','timestamp']);assert len(df)==340
fig=plt.figure(figsize=(11.2,9.9));gs=fig.add_gridspec(3,1,height_ratios=[1.3,1.15,.55],hspace=.3)
a=fig.add_subplot(gs[0]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');title(a,'(a) v4抓取放置实例：来源执行与规范数据包',13)
text(a,.015,.86,'源执行',12,ha='left',color=BLUE);text(a,.015,.67,'source_episode/episode.npz\nstate[638,39] / position_action[638,13]\nsensor_tactile[638,712]',11,ha='left');text(a,.015,.28,'manifest.json：原候选 / 阶段 / 在线合同\nagentview.mp4：独立关联视频',11,ha='left')
a.add_patch(FancyArrowPatch((.42,.66),(.59,.66),arrowstyle='-|>',mutation_scale=15,color=BLUE,lw=1.5));text(a,.50,.84,'按字段映射',11);text(a,.50,.45,'执行索引保留\n父来源保留',11)
text(a,.62,.86,'规范包：pick_tomato_10e460…',12,ha='left',color=TEAL);text(a,.62,.66,'modalities.npz\naction ← position_action\ntactile ← sensor_tactile',11,ha='left');text(a,.62,.30,'episode.json\nlineage '+e['lineage_hash'][:12]+'…\nasset '+e['asset_hash'][:12]+'…',11,ha='left')
text(a,.015,.04,'来源关系：origin_case → source_episode副本 → modalities；Root独立性另行核对',11,ha='left')
b=fig.add_subplot(gs[1]);title(b,'(b) Sugar Box：原始477行与LeRobot 340帧\n导出帧j：动作=j+137；观测命令区/触觉=j+136',13);b.set(xlim=(-15,495),ylim=(-.8,3.7));b.set_yticks([0,1,2,3],['LeRobot frame_index','源触觉 / 状态命令','源ctrl动作','源记录全部行'],fontproperties=FONT,fontsize=11);b.set_xticks([0,137,339,476]);b.set_xlabel('各行内的索引（零基）',fontproperties=FONT)
b.broken_barh([(0,477)],(2.75,.5),facecolors='#dce4e9');b.broken_barh([(137,340)],(1.75,.5),facecolors=BLUE);b.broken_barh([(136,340)],(.75,.5),facecolors=TEAL);b.broken_barh([(0,340)],(-.25,.5),facecolors=ORANGE)
for yy,left,right in [(3,0,476),(2,137,476),(1,136,475),(0,0,339)]:
 b.text(left,yy+.30,str(left),fontsize=10);b.text(right,yy+.30,str(right),fontsize=10,ha='right')

b.spines[['top','right','left']].set_visible(False)
c=fig.add_subplot(gs[2]);c.axis('off');title(c,'(c) 数据来源与划分单位',13);text(c,.02,.63,'仿真：Episode → Frame → 派生窗口；分组依据Root来源关系',12,ha='left');text(c,.02,.18,'真实触觉：Real Session → 4×712窗口（不补造仿真动作或RGB）',12,ha='left');save(fig,'data_lineage_instances')
provenance('data_lineage_instances',[v4/'episode.json',v4/'source_episode/manifest.json',src,mod,report,sm,npz,lp,R/'outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/data_contract.csv',R/'source/data/production_artifacts.py'],{'v4_shapes':shapes,'v4_admission':e['admission'],'lineage_hash':e['lineage_hash'],'origin_case':e['origin_case'],'fixed_export':{'rows':len(df),'episode_indices':df.episode_index.unique().tolist(),'frame_index_range':[int(df.frame_index.min()),int(df.frame_index.max())],'source_rows':477,'source_action_range':[137,476],'source_observation_range':[136,475]},'two_routes':'v4 PickPlace and LeRobot Sugar are different traced examples, not a fictional single export chain','Root':'UNKNOWN independent physical roots; lineage_hash is not a root count','Real':'existing frozen contract only; no Real/Test/Holdout raw arrays read'})
print('lineage complete')
