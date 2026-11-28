from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
R=Path(__file__).resolve().parents[3];A=Path(__file__).parent;O=R/'thesis/figures/annotation_R2';O.mkdir(exist_ok=True)
FONT=FontProperties(fname=str(R/'thesis/.tools/.TinyTeX/texmf-dist/fonts/opentype/public/fandol/FandolSong-Regular.otf'))
plt.rcParams.update({'font.size':11,'mathtext.fontset':'stix','pdf.fonttype':3,'axes.spines.top':False,'axes.spines.right':False})
BLUE='#255f85';ORANGE='#bd642b';TEAL='#258c85';GREY='#697986'
def title(ax,s,size=13):ax.set_title(s,fontproperties=FONT,fontsize=size,loc='left',pad=8)
def text(ax,x,y,s,size=11,ha='center',**kwargs):return ax.text(x,y,s,fontproperties=FONT,fontsize=size,ha=ha,va='center',**kwargs)
def save(fig,name):
 fig.savefig(O/(name+'.pdf'),bbox_inches='tight');fig.savefig(O/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
def provenance(name,paths,extra):
 (A/(name+'_sources.json')).write_text(json.dumps({'files':{str(Path(p).relative_to(R)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},**extra},ensure_ascii=False,indent=2)+'\n')
