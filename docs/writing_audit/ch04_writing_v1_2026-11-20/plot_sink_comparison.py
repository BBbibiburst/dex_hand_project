"""Plot only two previously saved historical replay measurements."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();p=root/'outputs/five_task_p38_exact_replay_2026-11-03/comparison.json';a=json.loads(p.read_text());out=root/'thesis/figures/ch04'
plt.rcParams['pdf.fonttype']=42
vals=[a['legacy_replay']['max_qpos_component_error'],a['exact_replay']['max_qpos_component_error']]
fig,ax=plt.subplots(figsize=(6.6,3.6));bars=ax.bar([0,1],vals,color=['#b8684b','#397b83'],width=.48)
ax.set_xticks([0,1],['Legacy float32 replay','Full state + float64 control'])
ax.set_ylabel('Maximum qpos component deviation');ax.set_ylim(0,max(vals)*1.22)
ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
for x,v in enumerate(vals):ax.annotate(f'{v:.6f}',(x,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=10)
ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(out/'sink_historical_precision.pdf');fig.savefig(out/'sink_historical_precision.png',dpi=150);plt.close(fig)
(root/'docs/writing_audit/ch04_writing_v1_2026-11-20/figure_provenance.json').write_text(json.dumps({'input':str(p.relative_to(root)),'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'fields':['legacy_replay.max_qpos_component_error','exact_replay.max_qpos_component_error'],'values':vals,'source_evidence':'L1 historical replay results; not a new experiment','no_training_or_physics':True},indent=2)+'\n')
