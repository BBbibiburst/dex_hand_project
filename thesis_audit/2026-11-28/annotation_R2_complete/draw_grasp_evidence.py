from style import *
import numpy as np,trimesh,json
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from source.grasping.catalog import resolve_object_mesh
from source.envs.manipulation.object_catalog import resolve_record,limited_mesh_scale
base=R/'outputs/dex_hand_top100_v2/grasp/ycb_004_sugar_box/seed_0000';p=resolve_object_mesh('ycb:004_sugar_box');mesh=trimesh.load(p,force='mesh',process=False);v=np.array(mesh.vertices);b=np.array(mesh.bounds);scale=limited_mesh_scale(resolve_record('ycb:004_sugar_box'),b[1]-b[0]);v=(v-(b[0]+b[1])/2)*scale;tri=v[mesh.faces]
initial=json.loads((base/'graspqp_seed/manifest.json').read_text());final=json.loads((base/'manifest.json').read_text());run=json.loads((base/'run.json').read_text());archive=json.loads((base/'dexevolve_archive.json').read_text());latticepath=R/'outputs/dex_hand_top100_v2/lattice/ycb_004_sugar_box/index.json';lat=json.loads(latticepath.read_text())['templates']
def obj(ax):
 ax.add_collection3d(Poly3DCollection(tri,facecolor='#7394a8',alpha=.48,edgecolor='none',rasterized=True));ax.view_init(25,-60);ax.set_box_aspect((1,1,1));ax.set(xlabel='x / m',ylabel='y / m',zlabel='z / m');ax.tick_params(labelsize=8);ax.xaxis.set_major_locator(plt.MaxNLocator(3));ax.yaxis.set_major_locator(plt.MaxNLocator(3));ax.zaxis.set_major_locator(plt.MaxNLocator(3));ax.grid(False)
def contact(ax,c):
 pts=np.array(c['contact_points']);norm=np.array(c['contact_normals']);obj(ax);ax.scatter(*pts.T,s=28,color=ORANGE,depthshade=False);ax.quiver(*pts.T,*(norm*.012).T,color=BLUE,linewidth=1.0);lo=np.minimum(v.min(0),pts.min(0));hi=np.maximum(v.max(0),pts.max(0));cen=(lo+hi)/2;h=max(hi-lo)*.6;ax.set(xlim=(cen[0]-h,cen[0]+h),ylim=(cen[1]-h,cen[1]+h),zlim=(cen[2]-h,cen[2]+h))
f=plt.figure(figsize=(10.7,10.0));gs=f.add_gridspec(3,2,height_ratios=[1.3,.8,.8],hspace=.42)
for i,(m,label) in enumerate([(initial,'(a) 保存的GraspQP代表候选'),(final,'(b) 保存的进化阶段代表候选')]):
 ax=f.add_subplot(gs[0,i],projection='3d');contact(ax,m['candidate']);title(ax,label,12)
ax=f.add_subplot(gs[1,0]);history=run['history'];ax.plot([z['generation'] for z in history],[z['best_fitness'] for z in history],'-o',color=BLUE,ms=3);ax.set(xlabel='generation',ylabel='stored best fitness');title(ax,'(c) 同次运行的档案最优适应度记录',12);ax.grid(alpha=.2)
ax=f.add_subplot(gs[1,1]);fr=np.stack([initial['candidate']['actuator_fractions'],final['candidate']['actuator_fractions']]);x=np.arange(6);ax.bar(x-.18,fr[0],.36,color='#90a9ba',label='GraspQP seed');ax.bar(x+.18,fr[1],.36,color=ORANGE,label='selected evolved');ax.set_xticks(x,['a0','a1','a2','a3','a4','a5'],fontsize=9);ax.set_ylim(0,1.05);title(ax,'(d) 两个代表候选的六维驱动比例',12);ax.legend(frameon=False,fontsize=9)
ax=f.add_subplot(gs[2,0]);off=np.array([q['translation_offset'] for q in lat])*1000;rot=np.array([q['rotation_offset_degrees'] for q in lat]);ax.scatter(rot[:,2],off[:,0],color=TEAL,s=25);ax.set(xlabel='yaw / deg',ylabel='x edit / mm');title(ax,'(e) 后续腕姿模板：保存的局部编辑',11)
rlp=R/'outputs/dex_hand_top100_v2/rl/gso_3D_Dollhouse_Refrigerator/best_trajectory/manifest.json';rl=json.loads(rlp.read_text());ax=f.add_subplot(gs[2,1]);ax.bar(range(6),rl['metadata']['hand_edit_normalized'],color=ORANGE);ax.set(ylim=(-1,1),xlabel='six hand-edit channels',ylabel='normalized edit');title(ax,'(f) 条件PPO分支：另物体的保存动作',11);ax.text(.01,.02,'Refrigerator / template 3',transform=ax.transAxes,fontsize=9);save(f,'candidate_geometry_feedback')
# Precheck residuals recorded for compiled templates; failure is execution failure, never relabel unreachable.
f=plt.figure(figsize=(10.8,7.3));gs=f.add_gridspec(2,2,height_ratios=[1.4,1]);ax=f.add_subplot(gs[0,0],projection='3d');obj(ax)
chosen=[lat[0],lat[1],next(q for q in lat if not q['success'])];col=[BLUE,TEAL,ORANGE]
for i,(q,c) in enumerate(zip(chosen,col)):
 m=json.loads(Path(q['manifest']).read_text());md=m['metadata'];pr=np.array(md['pregrasp_ee_position']);gr=np.array(md['grasp_ee_position']);rotq=np.array(md['object_initial_quaternion_wxyz']);from scipy.spatial.transform import Rotation;rm=Rotation.from_quat(rotq[[1,2,3,0]]).as_matrix();origin=np.array(md['object_initial_position']);pr=rm.T@(pr-origin);gr=rm.T@(gr-origin);ax.plot(*np.stack([pr,gr]).T,color=c,lw=2,label=['source','roll -15deg','execution failed'][i]);ax.scatter(*gr,s=35,color=c);ax.scatter(*pr,s=30,facecolors='none',edgecolors=c)
ax.legend(frameon=False,fontsize=8);title(ax,'(a) 物体坐标下保存的两类末端目标',11)
ax=f.add_subplot(gs[0,1]);xx=np.arange(len(lat));err=np.array([q['precheck_position_error'] for q in lat])*1000;ax.bar(xx,err,color=[TEAL if q['success'] else ORANGE for q in lat]);ax.set(xlabel='stored template index',ylabel='max two-target position error / mm');title(ax,'(b) 已保存模板的两目标最大IK位置残差',11);ax.grid(axis='y',alpha=.2)
ax=f.add_subplot(gs[1,0]);rot=np.array([q['rotation_offset_degrees'] for q in lat]);offset=np.array([q['translation_offset'] for q in lat])*1000;ax.scatter(rot[:,2],offset[:,0],s=45,c=[TEAL if q['success'] else ORANGE for q in lat]);ax.set(xlabel='yaw edit / deg',ylabel='x translation edit / mm');title(ax,'(c) 保存模板的腕姿编辑与历史执行状态',11)
ax=f.add_subplot(gs[1,1]);ax.axis('off');text(ax,.02,.82,'同次日志：39个编辑，18个通过IK预检',11,ha='left');text(ax,.02,.57,'索引保存12个模板：源模板＋11个执行编辑',11,ha='left');text(ax,.02,.32,'蓝绿：执行成功；橙色：执行失败',11,ha='left');text(ax,.02,.08,'低IK残差与执行失败可同时出现',11,ha='left');save(f,'wrist_recorded_targets')
from PIL import Image
import av
with av.open(str(R/'outputs/five_task_p32_b_2026-11-03/lift_sugar_rgb_00/agentview.mp4')) as container:
 for idx,frame in enumerate(container.decode(video=0)):
  if idx==312:Image.fromarray(frame.to_ndarray(format='rgb24')).save(A/'lift_frame312.png');break
# End-to-end illustrated evidence, not a claim that initial seed evolved monotonically into final seed.
from PIL import Image
f=plt.figure(figsize=(11.5,9.0));gs=f.add_gridspec(2,3,height_ratios=[1,1.1])
ax=f.add_subplot(gs[0,0],projection='3d');obj(ax);ax.set_axis_off();title(ax,'(a) Sugar Box物体几何',12)
ax=f.add_subplot(gs[0,1],projection='3d');contact(ax,initial['candidate']);ax.set_axis_off();title(ax,'(b) 几何候选与接触法向',12)
ax=f.add_subplot(gs[0,2],projection='3d');contact(ax,final['candidate']);ax.set_axis_off();title(ax,'(c) 物理反馈后选择的候选',12)
ax=f.add_subplot(gs[1,0]);ax.imshow(Image.open(A/'lift_frame312.png'));ax.axis('off');title(ax,'(d) 历史抓持与抬升记录',12)
ax=f.add_subplot(gs[1,1]);j=np.load(R/'outputs/five_task_p32_b_2026-11-03/lift_sugar_rgb_00/source_episode/episode.npz',allow_pickle=False);tt=j['timestamp']-j['timestamp'][0];ax.plot(tt,j['object_position'][:,2],color=BLUE);ax.set(xlabel='record time / s',ylabel='object z / m');title(ax,'(e) 动力学执行中的物体运动',12)
ax=f.add_subplot(gs[1,2]);ax.axis('off');text(ax,0,.85,'候选集合 → 进化档案 → 腕姿编辑',11,ha='left');text(ax,0,.62,'IK预检 → 连续命令 → 物理执行',11,ha='left');text(ax,0,.39,'Strict按抬升、稳定性、接触检查',11,ha='left');text(ax,0,.16,'同物体不同记录，各自保留来源',11,ha='left');save(f,'grasp_evidence_pipeline')
provenance('grasp_evidence',[p,base/'graspqp_seed/manifest.json',base/'manifest.json',base/'run.json',base/'dexevolve_archive.json',latticepath,rlp,R/'outputs/dex_hand_top100_v2/logs/ycb_004_sugar_box.log',R/'source/rl/grasp_edit/templates.py',R/'source/grasping/executor.py',R/'outputs/five_task_p32_b_2026-11-03/lift_sugar_rgb_00/source_episode/episode.npz'],{'method':'stored candidate contact points/normals and run history; current source mesh read-only centering/scaling; no optimization/IK/model inference or mj_step','selected_candidates':{'graspqp_seed':initial['candidate']['seed_index'],'final':final['candidate']['seed_index']},'no_causal_seed_claim':'Different representative candidates from same run, not a one-to-one tracked candidate optimization trajectory','lattice_precheck_role':'maximum residual across pregrasp and grasp, not separately stored residuals','missing':'Unreachable candidates were filtered before template index; no per-candidate rejected IK state is available here. Orange means physical execution failed after precheck, NOT IK unreachable.','PPO':{'source':str(rlp.relative_to(R)),'role':'different object Refrigerator conditional route, saved 6D normalized edit; no rerun and no claim of same Sugar Box execution','success':rl['success']},'archive_history':history})
print('grasp figures complete')
