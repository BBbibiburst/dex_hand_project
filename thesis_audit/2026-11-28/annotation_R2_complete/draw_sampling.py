from style import *
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from source.sensors.tactile.fitting.stl import read_stl_triangles
from source.sensors.tactile.fitting.segment import _finger_segment_raw_shell_grid_points,finger_segment_grid_points,_fit_finger_segment_surfaces
from source.sensors.tactile.fitting.fingertip import _fingertip_swept_shell_raw_grid_points,fingertip_ellipsoid_grid_points
from source.sensors.tactile.fitting.projection import mesh_uv_grid_points
from scipy.spatial import cKDTree
records=[];paths=[];fig=plt.figure(figsize=(11.8,10.8));configs=[('skin_0_0_p',4,8,'指节 / Segment'),('skin_0_2_p',7,8,'指尖 / Swept-shell'),('skin_palm_p',7,16,'手掌 / Mesh-UV')]
for row,(name,nr,nc,label) in enumerate(configs):
 p=R/'assets/grippers/dex_hand/meshes'/(name+'.STL');paths.append(p);tri=read_stl_triangles(p);v=tri.reshape(-1,3);center=v.mean(0);_,_,vh=np.linalg.svd(v-center,full_matrices=False)
 final=(finger_segment_grid_points if row==0 else fingertip_ellipsoid_grid_points if row==1 else mesh_uv_grid_points)(p,nr,nc)
 raw=_finger_segment_raw_shell_grid_points(tri,max(18,nr*4),max(32,nc*4)) if row==0 else _fingertip_swept_shell_raw_grid_points(tri,max(12,nr*3),max(24,nc*3)) if row==1 else None
 for col in range(3):
  if row==2 and col==1:
   ax=fig.add_subplot(3,3,row*3+col+1);uv=(v-center)@vh[:2].T*1000;ax.triplot(uv[:,0],uv[:,1],np.arange(len(v)).reshape(-1,3),lw=.2,color='#a7b5bf');sf=(final-center)@vh[:2].T*1000;ax.scatter(sf[:,0],sf[:,1],s=8,color=ORANGE);ax.set_aspect('equal');ax.set_xlabel('PCA u / mm');ax.set_ylabel('PCA v / mm');title(ax,'投影三角形与参数格点',11);continue
  ax=fig.add_subplot(3,3,row*3+col+1,projection='3d');scaled=tri*1000;ax.add_collection3d(Poly3DCollection(scaled,facecolor='#bac5cb',alpha=.23,edgecolor='none',rasterized=True))
  if col==0:
   if row==0:
    fit=_fit_finger_segment_surfaces(v);center=fit.center;vh=np.stack([fit.axis,fit.section_x,fit.section_y])
   for i,color in enumerate([BLUE,ORANGE,TEAL]):ax.quiver(*(center*1000),*(vh[i]*15),color=color,lw=1.5) if row!=1 else None
   st='STL与实际拟合局部轴' if row==0 else '原始指尖STL' if row==1 else 'STL与PCA参数轴'
  elif col==1:
   ax.scatter(*(raw*1000).T,color=ORANGE,s=2,alpha=.7);st='外壳识别后的密集样本'
  else:
   g=final.reshape(nr,nc,3)*1000
   for x in g:ax.plot(*x.T,color=BLUE,lw=.8)
   for x in g.transpose(1,0,2):ax.plot(*x.T,color=TEAL,lw=.8)
   ax.scatter(*(final*1000).T,s=11,color=ORANGE);st=f'{nr}×{nc}最终采样中心'
  lo=v.min(0)*1000;hi=v.max(0)*1000;cen=(lo+hi)/2;h=max(hi-lo)*.56
  ax.set(xlim=(cen[0]-h,cen[0]+h),ylim=(cen[1]-h,cen[1]+h),zlim=(cen[2]-h,cen[2]+h));ax.set_box_aspect([1,1,1]);ax.view_init(22,-60 if row<2 else 12);ax.set_axis_off();title(ax,('('+chr(97+row*3+col)+') '+label+'\n' if col==0 else '')+st,11)
 nn=cKDTree(final).query(final,k=2)[0][:,1]*1000;record={'patch':name,'shape':[nr,nc],'mesh_triangles':len(tri),'raw_shell_sample_count':len(raw) if raw is not None else None,'nearest_neighbor_mm':{'min':float(nn.min()),'median':float(np.median(nn))},'method':'actual production geometry functions; no MuJoCo step or model training'};records.append(record);np.savez_compressed(A/(name+'_sampling.npz'),triangles=tri,raw_shell=raw if raw is not None else np.empty((0,3)),samples=final)
fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.96,wspace=.08,hspace=.15);save(fig,'sampling_processes')
provenance('sampling_processes',paths+[R/'source/sensors/tactile/fitting/segment.py',R/'source/sensors/tactile/fitting/fingertip.py',R/'source/sensors/tactile/fitting/projection.py'],{'records':records,'metric_role':'descriptive geometry only; no new contact performance, coverage advantage or physical calibration claim'})
print('sampling complete')
