"""Render editable SVG and WebP diagrams from the supplied detections."""
import io
from pathlib import Path
import sys

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from nms import load, suppress

plt.rcParams.update({"svg.hashsalt":"wifa324","svg.fonttype":"none","text.color":"#163548","font.size":12})

def draw(name, example, title, labels, agnostic=False):
    ds=load(OUT.parent/'examples'/f'{example}.json')
    result=suppress(ds,.5,.5,agnostic=agnostic)
    trace={d['id']:d['decision'] for d in result['trace']}
    fig,ax=plt.subplots(figsize=(7.5,4.3),facecolor='#eff7f8')
    fig.subplots_adjust(left=.10,right=.94,bottom=.17,top=.75)
    fig.text(.055,.90,title,fontsize=21,weight='bold')
    fig.text(.055,.81,labels,fontsize=12)
    ax.set_facecolor('#eff7f8')
    for i,d in enumerate(ds):
        x,y,u,v=d.box
        keep=trace[d.id]=='kept'
        color='#076b62' if keep else '#a04423'
        ax.add_patch(Rectangle((x,y),u-x,v-y,fill=False,lw=2,
                              linestyle=['-','--',':','-.'][i],color=color))
        ax.text(x+.25, v+1.1+i*.85, d.id+': '+trace[d.id],fontsize=11,color=color,weight='bold')
    ax.set(xlim=(-1,20),ylim=(-1,16),xlabel='x coordinate',ylabel='y coordinate')
    ax.set_aspect('equal', adjustable='box')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/(name+'.svg'),metadata={'Date':None})
    b=io.BytesIO()
    fig.savefig(b,format='png',dpi=144)
    with Image.open(b) as im: im.convert('RGB').save(OUT/(name+'.webp'),quality=87,method=6)
    plt.close(fig)

if __name__=='__main__':
    draw('nms-overlap-chain','chain','Only kept boxes suppress','IoU limit 0.50 · A removes B · C remains')
    draw('nms-threshold-boundary','boundary','Equality keeps a box','IoU(A, B) = 0.50 · strict suppression above 0.50')
