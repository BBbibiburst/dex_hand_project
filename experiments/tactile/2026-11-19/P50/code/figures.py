from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
O.joinpath('figures').mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'savefig.dpi':160})
def box(ax,x,y,w,h,text,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015',fc=color,ec='#465366'));ax.text(x+w/2,y+h/2,text,ha='center',va='center')
def arr(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#344054',lw=1.5))
fig,ax=plt.subplots(figsize=(12,4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
for x,w,t,c in [(.01,.13,'Natural Sim\n4 x 712','#e6edf5'),(.19,.13,'Sim Encoder\nfrozen','#dddfe6'),(.37,.13,'Shared field\ntrainable','#c6dfef'),(.55,.13,'Real Decoder\nfrozen','#dddfe6'),(.76,.15,'Residual refiner\ntrainable','#d2eadc')]:box(ax,x,.48,w,.22,t,c)
for a,b in [((.14,.59),(.19,.59)),((.32,.59),(.37,.59)),((.50,.59),(.55,.59)),((.68,.59),(.76,.59)),((.91,.59),(.99,.59))]:arr(ax,a,b)
ax.text(.97,.78,'Output\n4 x 712',ha='center');ax.text(.23,.2,'z: frozen Sim posterior mean',ha='center');arr(ax,(.26,.48),(.26,.3));arr(ax,(.26,.3),(.82,.3));arr(ax,(.82,.3),(.82,.48));ax.set_title('P50: source-conditioned output residual refinement (K=1)');fig.tight_layout();fig.savefig(O/'figures/generator_architecture.png');fig.savefig(O/'figures/generator_architecture.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(11,5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');box(ax,.02,.38,.16,.23,'Full tactile\n4 x 712','#d2eadc')
for y,t,c in [(.75,'Global: full log1p response\n2848 -> 128 -> 64 -> 1','#c6dfef'),(.43,'Spatial: 16 index Patches\nmean / second moment / max\n192 -> 128 -> 64 -> 1','#eadfc6'),(.10,'Temporal: adjacent log differences\n2136 -> 128 -> 64 -> 1','#d9d0eb')]:
 box(ax,.33,y,.39,.19,t,c);arr(ax,(.18,.49),(.33,y+.095));arr(ax,(.72,y+.095),(.86,.49))
box(ax,.86,.38,.12,.23,'Mean BCE\nAND at 0.5\nfor Joint FR','#e6edf5');ax.set_title('P50: three continuous, differentiable discriminator views');fig.tight_layout();fig.savefig(O/'figures/discriminator_architecture.png');fig.savefig(O/'figures/discriminator_architecture.svg');plt.close(fig)
rows=list(csv.DictReader((O/'training_log.csv').open()));fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,phase,cols in [(axes[0],'D',['global_CE','spatial_CE','temporal_CE']),(axes[1],'G',['adv_global','adv_spatial','adv_temporal'])]:
 rr=[v for v in rows if v['phase']==phase];ep=sorted(set(int(v['epoch']) for v in rr))
 for col in cols:ax.plot(ep,[np.mean([float(v[col]) for v in rr if int(v['epoch'])==e]) for e in ep],label=col)
 ax.set(title=phase+' training (fixed endpoint)',xlabel='Epoch',ylabel='BCE');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(O/'figures/branch_training_curves.png');plt.close(fig)
rows=list(csv.DictReader((O/'acceptance_metrics.csv').open()));fig,ax=plt.subplots(figsize=(9,4))
names=['F0','P49_F3','F50_without_refiner','F50'];x=np.arange(4)
for j,b in enumerate(['Global','Spatial','Temporal','Joint']):ax.bar(x+(j-1.5)*.19,[100*float(next(v['FR_macro'] for v in rows if v['model']==n and v['branch']==b)) for n in names],.19,label=b)
ax.axhline(90,c='k',ls='--',lw=1);ax.set(xticks=x,xticklabels=names,ylabel='Root macro FR (%)',ylim=(0,105));ax.legend();fig.tight_layout();fig.savefig(O/'figures/acceptance_comparison.png');plt.close(fig)
rows=list(csv.DictReader((O/'distribution_metrics.csv').open()));fig,axes=plt.subplots(1,2,figsize=(11,4));names=['Original','F0','P49_F3','F50_without_refiner','F50']
for ax,col in zip(axes,['SWD','MMD']):ax.bar(names,[float(next(v[col] for v in rows if v['model']==n)) for n in names]);ax.set(title=col+' (lower is closer)',ylabel=col);ax.tick_params(axis='x',rotation=25)
fig.tight_layout();fig.savefig(O/'figures/distribution_comparison.png');plt.close(fig)
rows=list(csv.DictReader((O/'branch_gradients.csv').open()));fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,col in zip(axes,['field_grad','refiner_grad']):
 for b in ['Global','Spatial','Temporal']:
  rr=[v for v in rows if v['branch']==b];ax.plot([int(v['epoch']) for v in rr],[float(v[col]) for v in rr],label=b)
 ax.set(yscale='log',xlabel='Epoch (first batch)',ylabel='Gradient norm',title=col);ax.legend()
fig.tight_layout();fig.savefig(O/'figures/branch_gradients.png');plt.close(fig)
print('FIGURES_COMPLETE')
