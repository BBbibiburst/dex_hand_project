"""Plot only two previously saved historical replay measurements."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();p=root/'outputs/five_task_p38_exact_replay_2026-11-03/comparison.json';a=json.loads(p.read_text());out=root/'thesis/figures/ch04'
plt.rcParams['pdf.fonttype']=42
vals=[a['legacy_replay']['max_qpos_component_error'],a['exact_replay']['max_qpos_component_error']]
fig,ax=plt.subplots(figsize=(6.6,1.3));bars=ax.bar([0,1],vals,color=['#b8684b','#397b83'],width=.48)
ax.set_xticks([0,1],['Legacy float32 replay','Full state + float64 control'])
ax.set_ylabel('Maximum qpos\ncomponent\ndeviation');ax.set_ylim(0,max(vals)*1.22)
ax.set_yticks([0,1.0]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
for x,v in enumerate(vals):ax.annotate(f'{v:.6f}',(x,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=10)
ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(out/'sink_historical_precision_r2.pdf');fig.savefig(out/'sink_historical_precision_r2.png',dpi=150);plt.close(fig)
