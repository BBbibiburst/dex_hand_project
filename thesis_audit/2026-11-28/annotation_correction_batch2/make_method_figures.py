from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.font_manager import FontProperties
R=Path(__file__).resolve().parents[3];O=R/'thesis/figures/annotation_batch2';O.mkdir(exist_ok=True)
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.size':12,'mathtext.fontset':'stix','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':3})
blue='#285a83';orange='#ac6726';ink='#233444';grey='#667986'
def txt(ax,x,y,s,size=12,color=ink,ha='center'):
 return ax.text(x,y,s,fontproperties=font,fontsize=size,color=color,ha=ha,va='center',linespacing=1.6)
def box(ax,x,y,w,h,title,body,fill='#edf4f8',color=blue):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.015',linewidth=1,edgecolor=color,facecolor=fill));txt(ax,x+w/2,y+h*.73,title,13,color);txt(ax,x+w/2,y+h*.32,body,11)
def arrow(ax,a,b,color=blue,style='-'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,linewidth=1.2,color=color,linestyle=style))
# Code-specific control diagram with branches, temporary-state IK and feedback timing.
f=plt.figure(figsize=(10.6,7.6));ax=f.add_axes([.02,.27,.96,.70]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
txt(ax,.01,.98,'(a) 控制目标到位置执行器：本文实现的组合控制器',14,ha='left')
box(ax,.01,.68,.20,.19,'关节位置模式','7维关节目标\n范围约束')
box(ax,.01,.36,.20,.23,'末端位姿模式','位置＋四元数\n目标滤波 / SLERP')
box(ax,.28,.34,.32,.27,'临时状态内迭代 IK','位姿误差 → 加权 Jacobian\nSVD 阻尼伪逆＋零空间姿态项\n步长 / 关节限幅')
box(ax,.67,.36,.30,.23,'关节目标平滑','差分速度滤波 / 速度限幅\n保存上一目标供下一次求解')
arrow(ax,(.21,.47),(.28,.47));arrow(ax,(.60,.47),(.67,.47))
box(ax,.67,.69,.30,.18,'机械臂位置命令','7个关节执行器')
arrow(ax,(.82,.59),(.82,.69));arrow(ax,(.21,.78),(.67,.78))
box(ax,.02,.03,.26,.19,'手部位置命令','6维目标 → 控制范围约束',fill='#fff4e8',color=orange)
box(ax,.36,.03,.28,.19,'固定腱位置执行器','传动坐标 = 关节加权组合',fill='#fff4e8',color=orange)
arrow(ax,(.28,.125),(.36,.125),orange)
box(ax,.71,.03,.26,.19,'MuJoCo 执行与反馈','写入 ctrl → 物理推进\n读取 q、末端位姿及观测',fill='#f1f3f5',color=grey)
arrow(ax,(.64,.125),(.71,.125),orange);arrow(ax,(.97,.76),(.99,.76));arrow(ax,(.99,.76),(.99,.125));arrow(ax,(.99,.125),(.97,.125))
# Feedback to IK as measured state at each control cycle.
arrow(ax,(.71,.08),(.66,.08),grey,'--');arrow(ax,(.66,.08),(.66,.28),grey,'--');arrow(ax,(.66,.28),(.44,.28),grey,'--');arrow(ax,(.44,.28),(.44,.34),grey,'--');txt(ax,.50,.245,'当前关节与末端状态',10,grey)
ax2=f.add_axes([.09,.07,.84,.14]);ax2.set(xlim=(-.001,.052),ylim=(-.4,1.2));ax2.axis('off');txt(ax2,-.001,1.1,'(b) 指定配置的控制与积分时序',13,ha='left')
ax2.hlines(.4,0,.05,color=grey,lw=1)
for t in np.arange(51)*.001:ax2.vlines(t,.30,.5,color=blue,lw=.7)
for t in [0,.05]:ax2.vlines(t,.05,.8,color=orange,lw=2)
txt(ax2,0,-.17,'控制更新',11,orange);txt(ax2,.05,-.17,'下一控制更新',11,orange);txt(ax2,.025,.77,'保持执行器目标；50个物理子步 × 0.001 s',12)
f.savefig(O/'control_implementation.pdf',bbox_inches='tight');f.savefig(O/'control_implementation.png',dpi=150,bbox_inches='tight');plt.close(f)
# Actual algebra and 8-neighbor operator, illustrative curves only, no new simulation.
f,axs=plt.subplots(2,2,figsize=(10.8,7.7),layout='constrained',gridspec_kw={'height_ratios':[1,1]})
a=axs[0,0];x=np.linspace(0,1.12,700);v=np.clip(x-.002,0,1);a.plot(x,v,color=grey,lw=1.5,label='clip(r-d,0,s)');a.plot(x,v**.7,color=blue,lw=2,label='s(a/s)^p');a.set(xlabel='raw response r',ylabel='transformed response',xlim=(0,1.12),ylim=(0,1.05));a.set_title('(a) 死区、饱和与幂响应',fontproperties=font,fontsize=13,loc='left');a.legend(frameon=False,fontsize=11);a.text(.36,.14,'d=0.002, s=1, p=0.7',fontsize=11);a.grid(alpha=.16)
a=axs[0,1];a.axis('off');a.set(xlim=(0,1),ylim=(0,1));txt(a,.03,.94,'(b) 时间滤波的状态读写',13,ha='left');box(a,.02,.50,.24,.23,'当前幅值',r'$b_t$');box(a,.38,.50,.58,.23,'IIR 更新',r'$v_t=0.45b_t+0.55v_{t-1}$');arrow(a,(.26,.61),(.38,.61));box(a,.38,.10,.58,.24,'保存时间状态',r'$v_t\ \rightarrow\ v_{t-1}$');arrow(a,(.67,.50),(.67,.34));arrow(a,(.38,.22),(.30,.22));arrow(a,(.30,.22),(.30,.80));arrow(a,(.30,.80),(.67,.80));arrow(a,(.67,.80),(.67,.73));txt(a,.04,.02,'reset 与首次调用分支决定初值',10,ha='left')
a=axs[1,0];k=np.full((3,3),.12/8);k[1,1]=.88;a.imshow(k,cmap='Blues',vmin=0,vmax=1)
for i in range(3):
 for j in range(3):a.text(j,i,'0.88' if (i,j)==(1,1) else '0.015',ha='center',va='center',color='white' if (i,j)==(1,1) else ink,fontsize=13)
a.set_xticks([]);a.set_yticks([]);a.set_title('(c) 内部源单元的响应分配',fontproperties=font,fontsize=13,loc='left');a.set_xlabel('保留88%；其余12%均分给8个邻居\n边界按实际邻居数分配；不跨Patch',fontproperties=font,fontsize=11)
a=axs[1,1];a.axis('off');a.set(xlim=(0,1),ylim=(0,1));txt(a,.01,.94,'(d) 实际调用顺序与配置开关',13,ha='left');rows=[('幅值变换','减死区 → 截断 → 幂响应'),('时间状态','IIR更新 → 保存滤波状态'),('逐Patch处理','邻域扩散 → 高斯滤波（本配置关闭）'),('观测输出','噪声（关闭）→ 除以s → float32')]
for j,(title,body) in enumerate(rows):
 y=.66-j*.20;box(a,.03,y,.92,.15,title,body)
 if j<3:arrow(a,(.49,y),(.49,y-.05))
f.savefig(O/'tactile_processing_operators.pdf',bbox_inches='tight');f.savefig(O/'tactile_processing_operators.png',dpi=150,bbox_inches='tight');plt.close(f)
print('created two vector method figures')
