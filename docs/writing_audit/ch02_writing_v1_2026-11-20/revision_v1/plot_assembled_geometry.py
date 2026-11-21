"""Offline XML/STL transforms at q=0; no MuJoCo compile/forward/step."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from source.sensors.tactile.fitting.stl import read_stl_triangles
R=Path.cwd();O=R/'docs/writing_audit/ch02_writing_v1_2026-11-20/revision_v1'
inputs={};meshes=[];body_frames={};seen=set()
def pose(el,override_pos=None):
 t=np.fromstring(el.get('pos','0 0 0'),sep=' ') if override_pos is None else np.array(override_pos)
 if el.get('quat'):
  w,x,y,z=np.fromstring(el.get('quat'),sep=' ');rot=Rotation.from_quat([x,y,z,w]).as_matrix()
 elif el.get('euler'):raise RuntimeError('Euler XML pose requires explicit compiler convention')
 else:rot=np.eye(3)
 T=np.eye(4);T[:3,:3]=rot;T[:3,3]=t;return T

def walk(body,parent,assets,label,override_root_pos=None,root=False):
 T=parent@pose(body,override_root_pos if root else None);body_frames[(label,body.get('name'))]=T
 for g in body.findall('geom'):
  if g.get('type','mesh')!='mesh' or not g.get('mesh'):continue
  name=g.get('mesh');m=assets[name];path=m['path'];GT=T@pose(g)
  key=(label,name,tuple(np.round(GT.ravel(),12)))
  if key in seen:continue
  seen.add(key);tri=read_stl_triangles(path)*m['scale']
  tri=tri@GT[:3,:3].T+GT[:3,3]
  color=('#a2b4c6' if label=='arm' else '#438788') if not name.startswith('skin') else '#d59a53'
  meshes.append((tri,color,label,name));inputs[str(path.relative_to(R))]=hashlib.sha256(path.read_bytes()).hexdigest()
 for child in body.findall('body'):walk(child,T,assets,label)
def readmodel(path):
 p=R/path;inputs[path]=hashlib.sha256(p.read_bytes()).hexdigest();root=ET.parse(p).getroot()
 md=root.find('compiler').get('meshdir','.')
 assets={}
 for m in root.findall('asset/mesh'):
  assert not any(m.get(k) for k in ['refpos','refquat']), 'Unsupported mesh reference transform'
  assets[m.get('name')]={'path':(p.parent/md/m.get('file')).resolve(),'scale':np.fromstring(m.get('scale','1 1 1'),sep=' ')}
 return root,assets
arm,am=readmodel('assets/robots/rm75b/rm75b.xml');hand,hm=readmodel('assets/grippers/dex_hand/dex_hand.xml')

for extra in ['assets/bases/rethink_minimal_mount.xml','source/robots/builder.py','source/robots/arms/rm75b.py']:
 inputs[extra]=hashlib.sha256((R/extra).read_bytes()).hexdigest()
base=ET.parse(R/'assets/bases/rethink_minimal_mount.xml').getroot();mount=base.find('.//site[@name="arm_mount"]');assert mount is not None
# Builder overwrites arm root pos/quat with arm_mount local pose.
armroot=arm.find('worldbody/body');armroot.set('pos',mount.get('pos','0 0 0'));armroot.set('quat',mount.get('quat','1 0 0 0'))
walk(armroot,np.eye(4),am,'arm')
attach=body_frames[('arm','right_hand')];extra=np.eye(4);extra[:3,:3]=Rotation.from_euler('xyz',[-90,-90,0],degrees=True).as_matrix()
walk(hand.find('worldbody/body'),attach@extra,hm,'hand',override_root_pos=[0,0,0],root=True)
fig=plt.figure(figsize=(9,6));axes=[fig.add_subplot(121,projection='3d'),fig.add_subplot(122,projection='3d')]
for ax,selection,title in [(axes[0],meshes,'RM75B + Dex Hand, q = 0'),(axes[1],[m for m in meshes if m[2]=='hand'],'Hand detail (same rigid transform)')]:
 for tri,color,label,name in selection:
  pc=Poly3DCollection(tri,facecolors=color,edgecolors='none',linewidths=0,alpha=1,rasterized=True);ax.add_collection3d(pc)
 pts=np.concatenate([x[0].reshape(-1,3) for x in selection]);lo=pts.min(0);hi=pts.max(0);center=(lo+hi)/2;span=max(hi-lo)*.55
 ax.set_xlim(center[0]-span,center[0]+span);ax.set_ylim(center[1]-span,center[1]+span);ax.set_zlim(center[2]-span,center[2]+span);ax.set_box_aspect([1,1,1]);ax.view_init(18,-52);ax.set_title(title,fontsize=10)
 for k in ['x','y','z']:getattr(ax,'set_'+k+'label')(k+' (m)',fontsize=8)
 ax.tick_params(labelsize=6);ax.locator_params(nbins=4)
fig.subplots_adjust(left=.01,right=.98,bottom=.07,top=.90,wspace=.02)
fig.savefig(O/'assembled_robot.pdf',dpi=190,bbox_inches='tight',pad_inches=.15);fig.savefig(O/'assembled_robot.png',dpi=150,bbox_inches='tight',pad_inches=.15);plt.close(fig)
(O/'geometry_provenance.json').write_text(json.dumps({'method':'XML rigid transforms and actual STL surfaces only','no_mujoco_simulation':True,'configuration':'all revolute joint coordinates zero; static geometry, not task pose','arm_mount':mount.attrib,'hand_attach_euler_xyz_deg':[-90,-90,0],'hand_root_translation_reset':True,'base_geometry_drawn':False,'mesh_count':len(meshes),'input_hashes':inputs,'limitations':'No collision hull, sensor physics or dynamic validation; mesh duplicate references deduplicated at same transform'},indent=2)+'\n')
print('Generated accurate static XML/STL assembly:',len(meshes),'mesh instances')
