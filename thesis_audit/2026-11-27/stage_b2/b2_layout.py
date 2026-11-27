from pathlib import Path
import json,sys,difflib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
r=Path('/home/zmy/dex_hand_project');d=r/'thesis_audit/2026-11-27/stage_b2';o=d/'candidate_layouts';o.mkdir(exist_ok=True);f=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');rows=json.loads((d/'candidate_frames.json').read_text())
for scheme in ['a','b']:
 fig=plt.figure(figsize=(12,7 if scheme=='a' else 8));gs=fig.add_gridspec(2,2 if scheme=='a' else 3,hspace=.35,wspace=.08)
 if scheme=='a':
  for j,x in enumerate([rows[2],rows[5]]):
   ax=fig.add_subplot(gs[0,j]);ax.imshow(Image.open(d/x['file']));ax.axis('off');ax.set_title(('Lift原记录成功' if j==0 else 'Nut retain-lower原记录成功')+f"\n帧{x['frame_index_zero_based']}，{x['stage']}，{x['timestamp_seconds']:.2f}s",fontproperties=f,fontsize=11)
  for j,case in enumerate(['nut_original_handle_00','nut_handle_lower_00']):
   ax=fig.add_subplot(gs[1,j]);ax.axis('off');ax.text(.05,.75,case+'\n历史任务检查：未通过\n原因字段：抓持螺栓未稳定插入固定螺母\n原记录941行；直接视频缺失\n此处仅呈现结果记录，不替代失败画面',fontproperties=f,fontsize=12,va='top',bbox=dict(boxstyle='round',facecolor='#eef2f6',edgecolor='#778899'))
 else:
  for j,x in enumerate(rows):
   ax=fig.add_subplot(gs[j//3,j%3]);ax.imshow(Image.open(d/x['file']));ax.axis('off');ax.set_title(('Lift' if j<3 else 'Nut retain-lower')+f"：{x['stage']}\n帧{x['frame_index_zero_based']} / {x['timestamp_seconds']:.2f}s",fontproperties=f,fontsize=10)
  fig.text(.05,.015,'失败记录分类：原抓取失败 / 降低抓点后仍失败。两者均缺直接录像；相同终止原因字段不证明唯一物理成因。',fontproperties=f,fontsize=10)
 fig.suptitle('图3-6候选方案'+scheme.upper()+'（待人工批准，未接入论文）',fontproperties=f,fontsize=15)
 fig.savefig(o/f'fig3_6_scheme_{scheme}.pdf',bbox_inches='tight');fig.savefig(o/f'fig3_6_scheme_{scheme}.png',dpi=140,bbox_inches='tight');plt.close(fig)
 src=r/'thesis/body/ch03_grasp_demonstrations.tex';s=src.read_text();start=s.rfind('\\begin{figure}',0,s.index('\\label{fig:ch03_examples}'));end=s.index('\\end{figure}',start)+len('\\end{figure}')
 new='\\begin{figure}[htbp]\n  \\centering\n  \\includegraphics[width=0.95\\textwidth]{'+str((o/f'fig3_6_scheme_{scheme}.pdf').relative_to(r))+'}\n  \\caption{历史成功案例原始画面与失败结果记录（失败案例无直接视频）}\n  \\label{fig:ch03_examples}\n\\end{figure}'
 (o/f'scheme_{scheme}.diff').write_text(''.join(difflib.unified_diff(s[start:end].splitlines(True),(new+'\n').splitlines(True),fromfile='CURRENT figure3-6 block',tofile='CANDIDATE ONLY; path relative to project root (adjust before integration)')))
print(o)
