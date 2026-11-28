from style import *
import numpy as np,av
cases=[('抬升','lift_sugar_rgb_00',[('预抓取',2),('闭合',4),('抬升',6),('验证',7)]),('抓取放置','pick_tomato_rgb_00',[('闭合',4),('运输',8),('释放',10),('任务验证',12)]),('堆叠','stack_ring_rgb_01',[('运输',8),('下降',9),('释放',10),('任务验证',12)]),('推动','push_tomato_rgb_00',[('预接近',2),('接触接近',3),('推动',8),('任务验证',12)]),('装配','nut_retain_lower_rgb_00',[('闭合',4),('抬升',6),('调整/运输',8),('装配/下降',9)])]
f,axs=plt.subplots(5,4,figsize=(12.2,10.3),layout='constrained');records=[];paths=[]
for row,(task,case,phases) in enumerate(cases):
 p=R/'outputs/five_task_p32_b_2026-11-03'/case;npz=p/'source_episode/episode.npz';manifest=p/'source_episode/manifest.json';video=p/'agentview.mp4';paths += [npz,manifest,video]
 with np.load(npz,allow_pickle=False) as z:stage=z['stage'].copy();time=z['timestamp'].copy()
 chosen=[int(np.flatnonzero(stage==code)[len(np.flatnonzero(stage==code))//2]) for _,code in phases];imgs={}
 with av.open(str(video)) as c:
  count=0
  for frame in c.decode(video=0):
   if count in chosen:imgs[count]=frame.to_ndarray(format='rgb24')
   count+=1
 assert count==len(stage)
 for col,((label,code),idx) in enumerate(zip(phases,chosen)):
  ax=axs[row,col];im=imgs[idx];h,w=im.shape[:2];ax.imshow(im[int(.25*h):int(.98*h),int(.32*w):int(.98*w)]);ax.axis('off');title(ax,f'{task}：{label}\n帧{idx} / {float(time[idx]-time[0]):.2f}s',10)
 records.append({'task':task,'source':case,'stage_code_and_frame':[{'label':label,'code':code,'frame':idx} for (label,code),idx in zip(phases,chosen)],'selection':'midpoint saved row of each registered stage, no new execution','display_crop_normalized':[.32,.25,.98,.98],'decoded_video_frames':count,'nut_note':'transport/descend are source stage labels; detailed object-pose servo semantics checked in nut_transfer.py'})
save(f,'five_task_phase_records');provenance('five_task_phase_records',paths+[R/'source/grasping/executor.py',R/'source/grasping/task_transfer.py',R/'source/grasping/stack_transfer.py',R/'source/grasping/push_transfer.py',R/'source/grasping/nut_transfer.py'],{'cases':records,'role':'historical video frames, not strict replay re-render or new validation'})
print('task phases complete')
