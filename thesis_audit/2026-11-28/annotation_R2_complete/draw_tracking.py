from style import *
import os,gzip,numpy as np
os.environ['MUJOCO_GL']='egl'
import mujoco
base=R/'outputs/five_task_p44_integration_2026-11-05/dataset/episodes/lift_sugar_ac9b12bc48cbf5775e50';p=base/'source_episode'
with np.load(p/'episode.npz',allow_pickle=False) as z:qs=z['qpos'].copy();targets=z['ik_action'].copy();times=z['next_timestamp'].copy();stages=z['stage'].copy()
m=mujoco.MjModel.from_binary_path('saved.mjb',assets={'saved.mjb':gzip.decompress((p/'model.mjb.gz').read_bytes())});assert qs.shape[1]==m.nq;data=mujoco.MjData(m);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'right_hand_site');assert site>=0
measured=[];rot_errors=[]
for q,target in zip(qs,targets):
 data.qpos[:]=q;mujoco.mj_kinematics(m,data);measured.append(data.site_xpos[site].copy());quat=np.empty(4);mujoco.mju_mat2Quat(quat,data.site_xmat[site]);target_q=target[3:7].astype(float);target_q/=np.linalg.norm(target_q);rot_errors.append(2*np.arccos(np.clip(abs(np.dot(quat,target_q)),0,1)))
measured=np.array(measured);ep=np.linalg.norm(measured-targets[:,:3],axis=1);er=np.array(rot_errors);t=times-times[0]
# No IK solve, integration or simulation: show saved qpos with saved compiled model.
m.vis.global_.offwidth=1000;m.vis.global_.offheight=750;renderer=mujoco.Renderer(m,height=750,width=1000);images=[];selected=[]
for name,code in [('预抓取',2),('闭合',4),('抬升',6),('验证',7)]:
 ids=np.flatnonzero(stages==code);idx=int(ids[len(ids)//2]);data.qpos[:]=qs[idx];mujoco.mj_kinematics(m,data);mujoco.mj_camlight(m,data);renderer.update_scene(data,camera='agentview');im=renderer.render().copy();hh,ww=im.shape[:2];images.append(im[int(.25*hh):int(.98*hh),int(.32*ww):int(.98*ww)]);selected.append({'stage':name,'code':code,'frame':idx})
renderer.close();strict=json.loads((base/'strict_final.json').read_text());assert strict['manifest_sha256']==__import__('hashlib').sha256((p/'manifest.json').read_bytes()).hexdigest();ss=strict['result']
f=plt.figure(figsize=(11.7,8.4));gs=f.add_gridspec(3,4,height_ratios=[.9,1,1.05],hspace=.5)
for i,(img,row) in enumerate(zip(images,selected)):
 ax=f.add_subplot(gs[0,i]);ax.imshow(img);ax.axis('off');title(ax,row['stage']+f" / 帧{row['frame']}",11)
a=f.add_subplot(gs[1,:2]);a.plot(t,ep*1000,color=BLUE,lw=1.4);a.set(xlabel='recorded step-end time / s',ylabel='position difference / mm');title(a,'(a) 保存目标与步末末端位置的差',12);a.grid(alpha=.2)
a=f.add_subplot(gs[1,2:]);a.plot(t,er,color=ORANGE,lw=1.4);a.set(xlabel='recorded step-end time / s',ylabel='orientation difference / rad');title(a,'(b) 保存目标与步末末端朝向的差',12);a.grid(alpha=.2)
a=f.add_subplot(gs[2,:2],projection='3d');a.plot(*targets[:,:3].T,color='#92aabd',label='saved targets');a.plot(*measured.T,color=BLUE,label='step-end pose');a.legend(fontsize=9,frameon=False);a.set(xlabel='x / m',ylabel='y / m',zlabel='z / m');title(a,'(c) 同一模型下的保存轨迹几何重建',12)
a=f.add_subplot(gs[2,2:]);a.axis('off');title(a,'(d) 同源历史Strict回放记录',12);text(a,.01,.80,ss['verification_status']+f"；回放{ss['frames']}帧",12,ha='left',color=TEAL);text(a,.01,.57,f"最终抬升 {ss['final_lift']:.5f} m",12,ha='left');text(a,.01,.34,f"尾段最大速度 {ss['tail_max_speed']:.5f} m/s",12,ha='left');text(a,.01,.10,'447帧源记录与322帧回放分别计量',11,ha='left');save(f,'recorded_tracking_strict')
np.savez_compressed(A/'kinematic_tracking_reconstruction.npz',timestamp=times,recorded_target=targets[:,:7],measured_pose_position=measured,position_difference=ep,orientation_difference=er)
provenance('recorded_tracking_strict',[p/'model.mjb.gz',p/'episode.npz',p/'manifest.json',base/'strict_final.json',base/'independent_replay.json',R/'source/robots/arms/rm75b.py',R/'source/data/production_recorder.py',R/'source/grasping/executor.py'],{'method':'stored qpos + stored MJB + mj_kinematics only; no mj_step, new IK solve or new Strict replay','target_semantics':'saved raw ik_action, not internal filtered target; pose at recorded qpos is post-step; compare to target for same row','error_definition':'norm(site_xpos(qpos_i)-ik_action_i[:3]); 2 arccos(abs(dot(normalized target quaternion, site quaternion)))','not_ik_residual':'tracking difference includes target filtering and actuator/dynamics; not solver iteration residual','frames':len(qs),'selected_frames':selected,'display_crop_normalized':[.32,.25,.98,.98],'historical_strict':ss,'historical_strict_frames_separate':True})
print('tracking complete')
