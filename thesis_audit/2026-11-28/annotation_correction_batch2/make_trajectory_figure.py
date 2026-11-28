from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
r=Path(__file__).resolve().parents[3];d=Path(__file__).parent;o=r/'thesis/figures/annotation_batch2';p=r/'outputs/five_task_p32_b_2026-11-03/lift_sugar_rgb_00/source_episode';fnt=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
with np.load(p/'episode.npz',allow_pickle=False) as z:
 t=z['timestamp'].copy();stage=z['stage'].copy();ctrl=z['ctrl'].copy();pos=z['object_position'].copy();contacts=z['robot_object_contact_count'].copy()
m=json.loads((p/'manifest.json').read_text());dt=t-t[0];bounds=np.r_[0,np.flatnonzero(np.diff(stage))+1,len(t)]
fig,axs=plt.subplots(3,1,figsize=(10.2,7.5),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.5,1,1]});colors=plt.cm.Set2(np.linspace(0,1,8));names={0:'稳定',1:'过渡',2:'预抓取',3:'接近',4:'闭合',5:'保持',6:'抬升',7:'验证'}
for a in axs:
 for start,end in zip(bounds[:-1],bounds[1:]):a.axvspan(dt[start],dt[end-1],color=colors[int(stage[start])%8],alpha=.18)
 a.grid(alpha=.2);a.spines[['top','right']].set_visible(False)
for j in range(7):axs[0].plot(dt,ctrl[:,j],lw=1.3,label=f'q{j+1}')
axs[0].legend(ncol=7,loc='lower left',frameon=False,fontsize=10)
axs[1].plot(dt,pos[:,2],color='#285a83',lw=1.8)
axs[2].step(dt,contacts,where='post',color='#ac6726',lw=1.6)
for a,title,ylabel in zip(axs,['(a) 保存的机械臂位置命令','(b) 物体中心的世界坐标高度','(c) 记录的机器人—物体接触数'],['目标 / rad','高度 / m','接触数']):a.set_title(title,fontproperties=fnt,fontsize=13,loc='left');a.set_ylabel(ylabel,fontproperties=fnt,fontsize=12)
axs[2].set_xlabel('相对记录时间 / s',fontproperties=fnt,fontsize=12)
# Vertical boundary labels are located above first panel, rather than written over curves.
for start,end in zip(bounds[:-1],bounds[1:]):
 axs[0].text((dt[start]+dt[end-1])/2,1.02,names.get(int(stage[start]),str(stage[start])),transform=axs[0].get_xaxis_transform(),ha='center',va='bottom',rotation=35,fontproperties=fnt,fontsize=10)
axs[0].set_title('(a) 保存的机械臂位置命令',fontproperties=fnt,fontsize=13,loc='left',pad=35)
fig.savefig(o/'historical_lift_trajectory.pdf',bbox_inches='tight');fig.savefig(o/'historical_lift_trajectory.png',dpi=150,bbox_inches='tight')
(d/'trajectory_source.json').write_text(json.dumps({'source_files':{str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'episode.npz',p/'manifest.json']},'fields':{'timestamp':list(t.shape),'stage':list(stage.shape),'ctrl':list(ctrl.shape),'object_position':list(pos.shape),'robot_object_contact_count':list(contacts.shape)},'stages':m['metadata']['stage_codes'],'calculation':'time axis timestamp-timestamp[0]; all rows retained; no smoothing or new success computation','role':'historical P32 Lift example; not current Strict threshold revalidation or end-effector tracking measurement'},ensure_ascii=False,indent=2)+'\n')
