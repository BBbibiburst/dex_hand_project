from pathlib import Path
import shutil,re
from PIL import Image
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
r=Path('/home/zmy/dex_hand_project');d=Path(__file__).resolve().parent;a=r/'thesis/figures/revision_20261127';f=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
for name in ['robot_mujoco','hand_mujoco','tactile_mujoco','tactile_mujoco_reverse']:shutil.copy2(d/(name+'.png'),a/(name+'.png'))
for dest,names,titles in [('robot_model',['robot_mujoco','hand_mujoco'],['(a) 当前组合模型','(b) 灵巧手局部']),('tactile_layout',['tactile_mujoco','tactile_mujoco_reverse'],['(a) 掌侧感知区域','(b) 反向视角'])]:
 fig,axes=plt.subplots(1,2,figsize=(10,4));
 for ax,name,title in zip(axes,names,titles):
  image=Image.open(a/(name+'.png'))
  if name=='robot_mujoco':image=image.crop((560,280,1050,1100))
  ax.imshow(image);ax.axis('off');ax.set_title(title,fontproperties=f,fontsize=11)
 fig.subplots_adjust(wspace=.02,left=0,right=1,bottom=0,top=.91);fig.savefig(a/(dest+'.pdf'),bbox_inches='tight',pad_inches=.02);plt.close(fig)
p=r/'thesis/body/ch02_simulation_platform.tex';s=p.read_text();s=s.replace('figures/ch02/assembled_robot_tachin.pdf','figures/revision_20261127/robot_model.pdf').replace('由当前XML与STL离线组合的机械臂和手部模型（静态关节零位，右侧为手部放大图；未显示底座，非任务执行截图）','当前XML/STL模型的MuJoCo静态渲染：机械臂、底座与灵巧手组合及手部局部（关节零位，非任务执行或实机照片）')
def repl(m):
 block=m.group(0)
 if 'fig:ch02_patch_order' not in block:return block
 return '\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=0.97\\textwidth]{figures/revision_20261127/tactile_layout.pdf}\n\\caption{当前MuJoCo模型中的16个Patch与712个box感知区域（颜色区分局部阵列，不表示真实压力或实机标定电极）}\n\\label{fig:ch02_patch_order}\n\\end{figure}'
s=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',repl,s,flags=re.S);p.write_text(s)
# Real historical video frames, never regenerated simulation.
base=r/'thesis_audit/2026-11-27/stage_b2/candidate_frames';fig,axes=plt.subplots(1,2,figsize=(10,3.7))
for ax,filename,title,note in zip(axes,['lift_sugar_rgb_00_frame_0401_verify.png','nut_retain_lower_rgb_00_frame_0548_task_verify.png'],['(a) Lift成功记录：verify阶段','(b) Nut保留抓持命令后的成功记录'],['原视频帧401 / 20.05 s','原视频帧548 / 27.40 s']):
 shutil.copy2(base/filename,a/filename);ax.imshow(Image.open(a/filename));ax.axis('off');ax.set_title(title,fontproperties=f,fontsize=10);ax.text(.5,-.06,note,transform=ax.transAxes,fontproperties=f,ha='center',fontsize=9)
fig.text(.05,.025,'失败对照：原抓取与降低抓点两条历史记录均未通过任务检查；未找到直接视频，故不展示替代画面。',fontproperties=f,fontsize=9)
fig.subplots_adjust(wspace=.04,left=0,right=1,bottom=.16,top=.9);fig.savefig(a/'historical_examples.pdf',bbox_inches='tight',pad_inches=.02);plt.close(fig)
p=r/'thesis/body/ch03_grasp_demonstrations.tex';s=p.read_text();s=s.replace('保留真实案例图位，后续只能从对应视频和manifest取得画面，不使用合成图片代替物理执行结果。','展示由对应原始视频提取的成功案例画面；两条失败记录仅呈现其结果，不使用其他运行的画面替代。')
def repl2(m):
 b=m.group(0)
 if 'fig:ch03_examples' not in b:return b
 return '\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=0.97\\textwidth]{figures/revision_20261127/historical_examples.pdf}\n\\caption{历史成功Episode原视频画面与失败结果对照（非Strict回放截图，失败案例缺直接录像）}\n\\label{fig:ch03_examples}\n\\end{figure}'
s=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',repl2,s,flags=re.S);p.write_text(s)
