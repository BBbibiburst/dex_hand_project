from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from source.sensors.tactile.dex_hand import create_dex_hand_tactile_sensor
root=Path('/home/zmy/dex_hand_project');out=Path(__file__).resolve().parent;dest=root/'thesis/figures/revision_20261127';font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');sensor=create_dex_hand_tactile_sensor();records=[]
def panel(ax,name,title,small=False):
 data=sensor.surface_plot_data(name);tri=data.triangles*1000;pts=data.points*1000 if hasattr(data,'points') else data.samples*1000
 ax.add_collection3d(Poly3DCollection(tri,facecolor='#aab6c1',edgecolor='none',alpha=.17,rasterized=True))
 grid=pts.reshape(data.rows,data.cols,3)
 for row in grid:ax.plot(*row.T,color='#2a6381',lw=.6,alpha=.7)
 for col in grid.transpose(1,0,2):ax.plot(*col.T,color='#3d9f94',lw=.6,alpha=.7)
 ax.scatter(*pts.T,color='#c35b39',s=9 if small else 13,depthshade=False)
 allv=tri.reshape(-1,3);lo=allv.min(0);hi=allv.max(0);center=(lo+hi)/2;half=max(hi-lo)*.54
 ax.set_xlim(center[0]-half,center[0]+half);ax.set_ylim(center[1]-half,center[1]+half);ax.set_zlim(center[2]-half,center[2]+half);ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-60)
 if name=='skin_palm_p':ax.view_init(elev=12,azim=12)
 ax.set_xlabel('x / mm',fontsize=8,labelpad=0);ax.set_ylabel('y / mm',fontsize=8,labelpad=0);ax.set_zlabel('z / mm',fontsize=8,labelpad=0);ax.tick_params(labelsize=6,pad=0)
 ax.set_title(title,fontproperties=font,fontsize=11 if not small else 9,pad=1)
 return data,pts
names=[('skin_0_0_p','(a) 指节区域0：segment，4×8'),('skin_0_1_p','(b) 指节区域1：segment，4×8'),('skin_0_2_p','(c) 指尖：swept-shell + Bezier，7×8'),('skin_palm_p','(d) 手掌：mesh-uv，7×16')]
fig=plt.figure(figsize=(9,7.2))
for i,(name,title) in enumerate(names):
 ax=fig.add_subplot(2,2,i+1,projection='3d');data,pts=panel(ax,name,title);mesh=next((root/'assets/grippers/dex_hand').rglob(name+'.STL'));records.append({'patch':name,'shape':[data.rows,data.cols],'strategy':data.kind,'mesh':str(mesh.relative_to(root)),'mesh_sha256':hashlib.sha256(mesh.read_bytes()).hexdigest(),'points_sha256':hashlib.sha256(np.ascontiguousarray(pts/1000).tobytes()).hexdigest()});np.savez_compressed(out/(name+'_geometry.npz'),triangles=data.triangles,samples=pts/1000)
fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.95,wspace=.05,hspace=.13);fig.savefig(dest/'surface_layout_strategies.pdf',bbox_inches='tight',pad_inches=.03);fig.savefig(out/'surface_layout_strategies.png',dpi=180,bbox_inches='tight',pad_inches=.03);plt.close(fig)
(out/'provenance.json').write_text(json.dumps({'method':'offline current production surface_plot_data; no MuJoCo model compilation, integration or simulation','coordinates':'local STL, m converted to mm for axis display; each panel equal 3D aspect, panels separately scaled','representatives':records,'legend':'gray: original STL; blue/green: connections between adjacent samples; orange: sample centers; connections are not electrode or collision edges'},ensure_ascii=False,indent=2))
print('DONE')
