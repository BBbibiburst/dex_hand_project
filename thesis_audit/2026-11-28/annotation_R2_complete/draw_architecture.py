from style import *
from matplotlib.patches import FancyArrowPatch
from PIL import Image
import numpy as np
src=R/'outputs/five_task_p32_b_2026-11-03/lift_sugar_rgb_00/source_episode/episode.npz'
with np.load(src,allow_pickle=False) as z:x=z['sensor_tactile'].copy();ctrl=z['ctrl'][200].copy()
f=plt.figure(figsize=(11.5,8.3));gs=f.add_gridspec(3,3,width_ratios=[.95,1.15,1.1],height_ratios=[.55,1.4,.65],hspace=.32,wspace=.22)
a=f.add_subplot(gs[0,:]);a.axis('off');title(a,'任务层：抬升 / 抓取放置 / 堆叠 / 推动 / 装配',14);text(a,.01,.49,'初态与目标 → 阶段状态机 → 控制目标；任务评价读取执行后的物体与接触状态',12,ha='left');text(a,.01,.10,'模型层：机械臂XML ＋ 手部XML/STL ＋ 底座与物体资产',12,ha='left')
a=f.add_subplot(gs[1,0]);a.axis('off');title(a,'控制层：20 Hz',13);text(a,.02,.85,'机械臂：7维关节目标\n或位置＋四元数 → IK',12,ha='left');text(a,.02,.58,'手部：6维固定腱位置目标',12,ha='left');text(a,.02,.32,'统一写入13路位置执行器\n命令在物理子步间保持',12,ha='left');text(a,.02,.05,'关节 / 速度约束与连续求解',12,ha='left')
a=f.add_subplot(gs[1,1]);a.imshow(Image.open(O/'arm_joint_geometry.png'));a.axis('off');title(a,'MuJoCo：1000 Hz物理推进',13)
a=f.add_subplot(gs[1,2]);a.imshow(x,aspect='auto',cmap='magma',vmin=0,vmax=1,rasterized=True);a.set(xlabel='taxel channel',ylabel='saved frame');title(a,'感知层：原生touch → 712D处理后响应',12);a.tick_params(labelsize=9)
a=f.add_subplot(gs[2,:]);a.axis('off');title(a,'记录层：保存状态、命令、触觉与关联视频',13);text(a,.01,.65,'NPZ：state[T,39] / position_action[T,13] / sensor_tactile[T,712]',12,ha='left');text(a,.01,.34,'Manifest：源执行、阶段、配置与Hash；RGB以控制行索引关联',12,ha='left');text(a,.01,.02,'规范modalities / LeRobot转换：保留父来源与字段、时间索引映射',12,ha='left')
# Overlay distinct streams in figure coordinates around actual model and response.
for start,end,color,style in [((.33,.53),(.43,.53),BLUE,'-'),((.63,.53),(.71,.53),TEAL,'-'),((.58,.40),(.58,.27),ORANGE,'--')]:f.add_artist(FancyArrowPatch(start,end,transform=f.transFigure,arrowstyle='-|>',mutation_scale=16,lw=2,color=color,linestyle=style))
f.text(.36,.61,'控制',fontproperties=FONT,color=BLUE,fontsize=12);f.text(.67,.61,'观测',fontproperties=FONT,color=TEAL,fontsize=12);f.text(.59,.32,'数据',fontproperties=FONT,color=ORANGE,fontsize=12);save(f,'platform_actual_architecture')
provenance('platform_actual_architecture',[src,R/'source/robots/builder.py',R/'source/control/composite.py',R/'source/envs/rl_env.py',R/'configs/robot_profiles/rm75b_dex_hand.json',R/'source/data/production_artifacts.py',R/'source/data/lerobot_recorder.py'],{'model_thumbnail':'current compiled zero-joint model; see joint_tendon_geometry_sources.json','response':'all 447x712 rows of historical Lift sensor_tactile; no simulated new response','diagram':'control/observation/data are distinct arrows; LeRobot is conversion capability, not a claim that every recorded episode is exported','timing':'profile control_dt .05 s and builder physics timestep .001 s'})
print('architecture complete')
