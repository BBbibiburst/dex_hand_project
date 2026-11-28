from style import *
import numpy as np
cases=[('原抓持命令','nut_original_handle_00',ORANGE),('降低抓点','nut_handle_lower_00','#9b7b52'),('降低抓点并保留抓持','nut_retain_lower_rgb_00',TEAL)]
f,axs=plt.subplots(3,1,figsize=(10.8,7.6),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.3,1,.6]});paths=[];records=[]
for i,(label,name,color) in enumerate(cases):
 p=R/'outputs/five_task_p32_b_2026-11-03'/name;z=np.load(p/'source_episode/episode.npz',allow_pickle=False);result=json.loads((p/'result.json').read_text());t=z['timestamp']-z['timestamp'][0];pos=z['object_position'];contact=z['robot_object_contact_count'];stage=z['stage']
 axs[0].plot(t,pos[:,2],color=color,lw=1.5,label=label+('（成功）' if result['success'] else '（失败）'));axs[1].plot(t,contact,color=color,lw=1.3)
 bounds=np.r_[0,np.flatnonzero(np.diff(stage))+1,len(stage)]
 for start,end in zip(bounds[:-1],bounds[1:]):
  axs[2].broken_barh([(t[start],max(float(t[end-1]-t[start]),.05))],(i-.3,.6),facecolors=plt.cm.tab20(int(stage[start])%20));axs[2].text((t[start]+t[end-1])/2,i,{0:'稳定',1:'过渡',2:'预抓取',3:'接近',4:'闭合',5:'保持',6:'抬升',8:'运输',9:'下降',12:'验证'}.get(int(stage[start]),str(stage[start])),ha='center',va='center',fontsize=8,fontproperties=FONT)
 paths += [p/'source_episode/episode.npz',p/'source_episode/manifest.json',p/'result.json'];records.append({'case':name,'label':label,'rows':len(t),'recorded_success':result['success'],'recorded_failure_reason':result.get('failure_reason')})
axs[0].set_ylabel('物体高度 / m',fontproperties=FONT);axs[1].set_ylabel('接触数',fontproperties=FONT);axs[2].set_yticks(range(3),[x[0] for x in cases],fontproperties=FONT,fontsize=10);axs[2].set_xlabel('各记录的相对保存时间 / s',fontproperties=FONT)
for ax in axs[:2]:ax.grid(alpha=.2)
axs[0].legend(prop=FONT,frameon=False,fontsize=10)
for ax,name in zip(axs,['(a) 装配物体的保存高度','(b) 机器人—物体接触反馈','(c) 各记录的保存阶段']):title(ax,name,12)
save(f,'assembly_failure_records');provenance('assembly_failure_records',paths,{'cases':records,'method':'all recorded rows plotted with source timestamps; no re-execution, new success detection or smoothing','comparison_scope':'historical local conditions; not multi-seed ablation or uniform durations','stage_map':'original shared stage codes; row duration differs between cases','video_status':'failed cases had no RGB; arrays used directly, not borrowed screenshots'})
