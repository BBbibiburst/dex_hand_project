from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import av
root=Path(__file__).resolve().parents[3];base=root/'outputs/five_task_p32_b_2026-11-03';out=root/'thesis/figures/annotation_r2';audit=Path(__file__).parent
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size=10)
items=[('抬升','lift_sugar_rgb_00'),('抓取放置','pick_tomato_rgb_00'),('堆叠','stack_ring_rgb_01'),('推动','push_tomato_rgb_00'),('螺母装配','nut_retain_lower_rgb_00')]
records=[];fig,axes=plt.subplots(5,1,figsize=(9.2,8),layout='constrained');scene,axs=plt.subplots(5,2,figsize=(9.2,13.6),layout='constrained')
for i,(label,n) in enumerate(items):
 p=base/n;npz=p/'source_episode/episode.npz';video=p/'agentview.mp4';manifest=p/'source_episode/manifest.json'
 with np.load(npz,allow_pickle=False) as z:x=z['sensor_tactile'].copy();times=z['timestamp'].copy()
 assert x.shape[1]==712 and np.isfinite(x).all() and x.min()>=0
 im=axes[i].imshow(x,aspect='auto',interpolation='nearest',vmin=0,vmax=1,cmap='magma',rasterized=True)
 for k in np.cumsum([32,32,56]*5+[112])[:-1]:axes[i].axvline(k-.5,lw=.35,color='white',alpha=.4)
 axes[i].set_title(label+'：'+str(len(x))+'帧',fontproperties=font,loc='left');axes[i].set_ylabel('记录帧索引',fontproperties=font)
 first=last=None;count=0;pts=[]
 with av.open(str(video)) as container:
  stream=container.streams.video[0];rate=str(stream.average_rate)
  for frame in container.decode(stream):
   if count==0:first=frame.to_ndarray(format='rgb24')
   last=frame.to_ndarray(format='rgb24');pts.append(float(frame.time) if frame.time is not None else None);count+=1
 assert count==len(x),(n,count,len(x))
 for col,img,idx in [(0,first,0),(1,last,count-1)]:
  axs[i,col].imshow(img);axs[i,col].axis('off');axs[i,col].set_title(f'{label}｜'+('记录起点' if col==0 else '记录终点')+f'（帧{idx}）',fontproperties=font)
 records.append(dict(task=label,source=n,field='sensor_tactile',shape=list(x.shape),dtype=str(x.dtype),range=[float(x.min()),float(x.max())],video_frames=count,video_rate=rate,selected_frame_indices=[0,count-1],selected_video_times=[pts[0],pts[-1]],record_timestamp_first_last=[float(times[0]),float(times[-1])],files={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [npz,video,manifest]}))
axes[-1].set_xlabel('Taxel通道索引（白线为Patch边界）',fontproperties=font)
cbar=fig.colorbar(im,ax=list(axes),fraction=.018,pad=.015);cbar.set_label('处理后数值响应',fontproperties=font)
fig.savefig(out/'historical_tactile_heatmaps.pdf',dpi=220);fig.savefig(out/'historical_tactile_heatmaps.png',dpi=140)
scene.savefig(out/'historical_five_task_scenes.pdf',dpi=180);scene.savefig(out/'historical_five_task_scenes.png',dpi=120)
(audit/'figure_sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print('created 2 figures; all five videos sequentially decoded; frame counts agree with source records')
