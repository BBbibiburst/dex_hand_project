from style import *
import os,xml.etree.ElementTree as ET,numpy as np
os.environ['MUJOCO_GL']='egl'
import mujoco
from source.robots.builder import build_robot_model_from_config
from PIL import Image
m,d=build_robot_model_from_config('configs/robot_profiles/rm75b_dex_hand.json');mujoco.mj_forward(m,d)
m.vis.global_.offwidth=1800;m.vis.global_.offheight=1500
rend=mujoco.Renderer(m,height=1500,width=1800);op=mujoco.MjvOption();op.sitegroup[:]=0
jointnames=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
armids=[i for i,n in enumerate(jointnames) if n in [f'joint{k}' for k in range(1,8)]];handids=[i for i,n in enumerate(jointnames) if any(n.endswith(v) for v in ['thumb_rotate_j','thumb_first_j','thumb_second_j','thumb_third_j']) or 'finger_first_' in n or 'finger_second_' in n]
cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
for name,ids,dist,az,elev in [('arm_joint_geometry',armids,2.4,125,-12),('hand_joint_geometry',handids,.42,125,-18)]:
 pts=d.xanchor[ids];cam.lookat[:]=(pts.min(0)+pts.max(0))/2;cam.distance=dist;cam.azimuth=az;cam.elevation=elev
 rend.update_scene(d,camera=cam,scene_option=op)
 for index,jid in enumerate(ids):
  g=rend.scene.geoms[rend.scene.ngeom];pos=d.xanchor[jid].copy();scale=.012 if name.startswith('arm') else .0025
  mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([scale]*3),pos,np.eye(3).ravel(),np.array([.95,.28,.1,1]));g.label=('A' if name.startswith('arm') else 'H')+str(index+1);rend.scene.ngeom+=1
 Image.fromarray(rend.render()).save(O/(name+'.png'))
rend.close()
root=ET.parse(R/'assets/grippers/dex_hand/dex_hand.xml').getroot();joints=[x.attrib['name'] for x in root.findall('.//worldbody//joint') if 'name' in x.attrib];tendons=root.findall('./tendon/fixed');act=root.findall('./actuator/position');C=np.zeros((len(tendons),len(joints)))
for i,t in enumerate(tendons):
 for q in t.findall('joint'):C[i,joints.index(q.attrib['joint'])]=float(q.attrib['coef'])
fig=plt.figure(figsize=(8.4,11.2));gs=fig.add_gridspec(4,2,height_ratios=[1.05,.12,.70,.85],hspace=.16)
for col,name,label in [(0,'arm_joint_geometry','(a) RM75B：7个转动关节'),(1,'hand_joint_geometry','(b) 灵巧手：12个转动关节')]:
 ax=fig.add_subplot(gs[0,col]);ax.imshow(Image.open(O/(name+'.png')));ax.axis('off');title(ax,label,12)
ax=fig.add_subplot(gs[1,:]);ax.axis('off');text(ax,.01,.55,'A1–A7：joint1–joint7 对应 pos_joint1–pos_joint7',12,ha='left')
ax=fig.add_subplot(gs[2,:]);ax.imshow(C,cmap='Blues',vmin=0,vmax=.009,aspect='auto')
for i in range(6):
 for j in range(12):
  if C[i,j]:ax.text(j,i,f'{C[i,j]:g}',ha='center',va='center',fontsize=11,color='white' if C[i,j]>.006 else '#18354a')
ax.set_xticks(range(12),[f'H{i+1}' for i in range(12)],fontsize=11);ax.set_yticks(range(6),[f'T{i+1}' for i in range(6)],fontsize=11)
title(ax,'(c) 固定腱系数矩阵：l = Cq（空白为0）',12)
ax=fig.add_subplot(gs[3,:]);ax.axis('off');text(ax,.01,.98,'关节编号与XML名称',12,ha='left');text(ax,.54,.98,'固定腱与位置执行器',12,ha='left')
for i,n in enumerate(joints):text(ax,.01,.89-i*.075,f'H{i+1}  '+n,11,ha='left')
for i,t in enumerate(tendons):
 actuator=next(z.attrib['name'] for z in act if z.attrib['tendon']==t.attrib['name']);text(ax,.54,.88-i*.15,f'T{i+1}  '+t.attrib['name']+'\n'+actuator,10.5,ha='left')
save(fig,'joint_tendon_geometry')
provenance('joint_tendon_geometry',[R/'assets/grippers/dex_hand/dex_hand.xml',R/'assets/robots/rm75b/rm75b.xml',R/'configs/robot_profiles/rm75b_dex_hand.json'],{'method':'current model compilation + mj_forward + MuJoCo renderer only, zero mj_step/IK/physical execution','joints':joints,'tendons':[t.attrib['name'] for t in tendons],'coefficient_matrix':C.tolist(),'joint_anchor_ids':{'arm':[jointnames[i] for i in armids],'hand':[jointnames[i] for i in handids]},'unit_note':'fixed tendon coefficients have length per angular coordinate; diagram visualizes linear transmission, not literal routed tendons'})
print('joint geometry complete')
