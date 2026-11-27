from pathlib import Path
import os
os.environ['MUJOCO_GL']='egl'
import mujoco,numpy as np,json,hashlib,colorsys
from PIL import Image
from source.robots.builder import build_robot_model_from_config
out=Path(__file__).resolve().parent;root=out.parents[2]
m,d=build_robot_model_from_config('configs/robot_profiles/rm75b_dex_hand.json')
# Geometry-only forward kinematics: no mj_step, control execution or simulated experiment.
mujoco.mj_forward(m,d)
m.vis.global_.offwidth=1600;m.vis.global_.offheight=1200
r=mujoco.Renderer(m,height=1200,width=1600);opt=mujoco.MjvOption();opt.sitegroup[:]=0
cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[0,0,1.0];cam.distance=3.2;cam.azimuth=135;cam.elevation=-18
r.update_scene(d,camera=cam,scene_option=opt);Image.fromarray(r.render()).save(out/'robot_mujoco.png')
ids=[i for i in range(m.nsite) if 'taxel' in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SITE,i) or '').lower()]
if not ids:
 ids=list(set(int(m.sensor_objid[i]) for i in range(m.nsensor) if m.sensor_type[i]==mujoco.mjtSensor.mjSENS_TOUCH))
assert len(ids)==712,len(ids)
p=d.site_xpos[ids];cam.lookat[:]=(p.min(0)+p.max(0))/2;cam.distance=.42;cam.azimuth=125;cam.elevation=-15
r.update_scene(d,camera=cam,scene_option=opt);Image.fromarray(r.render()).save(out/'hand_mujoco.png')
for i in ids:
 name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SITE,i);h=sum(ord(c) for c in name.rsplit('_',2)[0])*.61803398875%1;m.site_rgba[i]=[*colorsys.hsv_to_rgb(h,.75,.95),.65];m.site_group[i]=4
opt.sitegroup[4]=1
for name,az in [('tactile_mujoco',125),('tactile_mujoco_reverse',-55)]:
 cam.azimuth=az;r.update_scene(d,camera=cam,scene_option=opt);Image.fromarray(r.render()).save(out/(name+'.png'))
r.close()
(out/'render_provenance.json').write_text(json.dumps({'method':'MuJoCo EGL Renderer; mj_forward only; no mj_step','site_count':len(ids),'qpos':d.qpos.tolist(),'config':'configs/robot_profiles/rm75b_dex_hand.json','mujoco_version':mujoco.__version__,'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('assets/robots/rm75b/rm75b.xml'),Path('assets/grippers/dex_hand/dex_hand.xml'),Path('source/sensors/tactile/dex_hand.py')]}},indent=2))
print('rendered',len(ids))
