"""Offline STL/sampling plots only; no MuJoCo model or physical stepping."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from source.sensors.tactile.fitting.fingertip import fingertip_ellipsoid_grid_points
from source.sensors.tactile.fitting.stl import read_stl_triangles
from source.sensors.tactile.dex_hand import _taxel_frame
root=Path.cwd();folder=root/'thesis/figures/ch02'
mesh=root/'assets/grippers/dex_hand/meshes/skin_0_2_p.STL'
if not mesh.exists():
 matches=list((root/'assets/grippers/dex_hand').rglob('skin_0_2_p.*'));mesh=next(p for p in matches if p.suffix.lower()=='.stl')
tri=read_stl_triangles(mesh);points=fingertip_ellipsoid_grid_points(mesh,7,8);grid=points.reshape(7,8,3)
fig=plt.figure(figsize=(7,5));ax=fig.add_subplot(111,projection='3d')
ax.add_collection3d(Poly3DCollection(tri,facecolor='#adb8c8',edgecolor='none',alpha=.16))
ax.scatter(*points.T,s=16,color='#006699',depthshade=False)
for row in grid:ax.plot(*row.T,color='#006699',lw=.6)
for col in grid.transpose(1,0,2):ax.plot(*col.T,color='#4aa694',lw=.6)
for i,k in enumerate(['x','y','z']):getattr(ax,'set_'+k+'label')(k+' (m, mesh frame)')
ax.auto_scale_xyz(tri[:,:,0].ravel(),tri[:,:,1].ravel(),tri[:,:,2].ravel());ax.set_title('skin_0_2_p: STL and 7 x 8 fitted samples')
fig.tight_layout();fig.savefig(folder/'fingertip_sampling.pdf');fig.savefig(folder/'fingertip_sampling.png',dpi=130);plt.close(fig)
# Use the same finite-difference tangent construction as production helper.
import inspect
print('frame helper signature',inspect.signature(_taxel_frame))
np.savez(folder/'fingertip_sampling_points.npz',points=points,triangles=tri)
fig=plt.figure(figsize=(6,5));ax=fig.add_subplot(111,projection='3d');ax.scatter(*points.T,s=10,color='#8997a4')
r,c=3,4;p=grid[r,c];dx=grid[r,c+1]-grid[r,c-1];dy=grid[r+1,c]-grid[r-1,c]
frame=_taxel_frame(grid,r,c);x,y,z=frame[:,0],frame[:,1],frame[:,2]
for v,color,label in [(x,'#bd3232','column tangent'),(y,'#16846b','row tangent'),(z,'#285f9e','cross-product normal')]:ax.quiver(*p,*v,length=.005,color=color,label=label)
ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)');ax.legend(fontsize=7);ax.set_title('Local frame before outward-orientation correction')
fig.tight_layout();fig.savefig(folder/'taxel_local_frame.pdf');fig.savefig(folder/'taxel_local_frame.png',dpi=130);plt.close(fig)
print('Offline geometry plots written from',mesh)
