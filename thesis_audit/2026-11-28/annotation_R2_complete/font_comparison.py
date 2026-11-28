from style import *
from matplotlib.font_manager import FontProperties
f,axs=plt.subplots(2,1,figsize=(8,4.8),layout='constrained')
fonts=[('当前Fandol宋体（保持现行模板）',FONT),('Noto Serif CJK（开放字体外观参照，非学校指定替代）',FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'))]
for ax,(label,fp) in zip(axs,fonts):
 ax.axis('off');text(ax,.01,.9,label,12,ha='left');ax.text(.01,.55,'机器人多任务仿真平台与触觉感知建模',fontproperties=fp,fontsize=16,transform=ax.transAxes);ax.text(.01,.25,'完整观测、动作与接触反馈的记录及验证。',fontproperties=fp,fontsize=12,transform=ax.transAxes)
f.savefig(A/'FONT_COMPARISON.pdf',bbox_inches='tight');f.savefig(A/'FONT_COMPARISON.png',dpi=150,bbox_inches='tight')
