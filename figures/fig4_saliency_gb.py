"""Fig 4 v4 — TPs all GB-paired (D27+GB 0.978, D25+GB 0.953, D26+GB 0.949);
TNs unchanged (D1+GB, D2+GB, D23+GB)."""
import sys
sys.path.insert(0, '../../Package_B/code')
sys.path.insert(0, '../../Package_C/code')
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from corrected_smiles import DYES, AGENTS

ELEMS = ['C','N','O','S','P','F','Cl','Br','I','As']
def af(a):
    f=[0]*10; s=a.GetSymbol()
    if s in ELEMS: f[ELEMS.index(s)]=1
    df=[0]*7; df[min(a.GetDegree(),6)]=1
    ff=[0]*5; ff[min(max(a.GetFormalCharge()+2,0),4)]=1
    return f+df+ff+[int(a.GetIsAromatic())]
BTS=[Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]
def bf(b):
    f=[0]*4
    if b.GetBondType() in BTS: f[BTS.index(b.GetBondType())]=1
    return f
def mol2g(smi):
    m=Chem.MolFromSmiles(smi)
    X=torch.tensor([af(a) for a in m.GetAtoms()], dtype=torch.float32)
    e=[]; ef=[]
    for b in m.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); fb=bf(b)
        e+=[(i,j),(j,i)]; ef+=[fb,fb]
    if not e: e=[(0,0)]; ef=[[0,0,0,0]]
    return m,X,torch.tensor(e,dtype=torch.long).t().contiguous(),torch.tensor(ef,dtype=torch.float32)

class L(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.lm=nn.Linear(h+4,h); self.ls=nn.Linear(h,h)
    def forward(self,X,E,EF):
        s,d=E[0],E[1]
        m=F.relu(self.lm(torch.cat([X[s],EF],dim=-1)))
        a=torch.zeros_like(X); a.index_add_(0,d,m)
        return F.relu(self.ls(X)+a)
class MPNN(nn.Module):
    def __init__(self,h=96,n=3):
        super().__init__()
        self.e=nn.Linear(23,h); self.layers=nn.ModuleList([L(h) for _ in range(n)])
        self.cls=nn.Sequential(nn.Linear(h*4,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,1))
    def enc(self,X,E,EF):
        h=F.relu(self.e(X))
        for L in self.layers: h=L(h,E,EF)
        return torch.cat([h.mean(0),h.max(0).values],dim=-1)
    def forward(self,d,a):
        return self.cls(torch.cat([self.enc(*d),self.enc(*a)],dim=-1)).squeeze(-1)

model=MPNN(h=96,n=3); model.load_state_dict(torch.load('../../Package_G/verification/mpnn_big_final.pt')); model.eval()

DYE_DAT={n:(name,smi,mol2g(smi)) for n,name,_,smi,_ in DYES}
AG_DAT={c:(name,smi,mol2g(smi)) for c,name,_,smi in AGENTS}

# Force selections: TPs all GB-paired, TNs all GB-paired
tp_choices = [(27,'GB'),(25,'GB'),(26,'GB')]   # D27+GB, D25+GB, D26+GB
tn_choices = [(1,'GB'),(2,'GB'),(23,'GB')]      # D1+GB, D2+GB, D23+GB

# Compute saliency
def saliency_for_pair(dye_id, agent_code):
    m_d, X_d, E_d, EF_d = DYE_DAT[int(dye_id)][2]
    m_a, X_a, E_a, EF_a = AG_DAT[agent_code][2]
    base = torch.sigmoid(model((X_d,E_d,EF_d), (X_a,E_a,EF_a))).item()
    sal_d=[]
    for ai in range(X_d.shape[0]):
        Xm=X_d.clone(); Xm[ai]=0
        with torch.no_grad():
            p=torch.sigmoid(model((Xm,E_d,EF_d),(X_a,E_a,EF_a))).item()
        sal_d.append(base-p)
    sal_a=[]
    for ai in range(X_a.shape[0]):
        Xm=X_a.clone(); Xm[ai]=0
        with torch.no_grad():
            p=torch.sigmoid(model((X_d,E_d,EF_d),(Xm,E_a,EF_a))).item()
        sal_a.append(base-p)
    return m_d, np.array(sal_d), m_a, np.array(sal_a), base

def render_panel(ax, dye_id, ag, label_letter, is_tp):
    dye_name = DYE_DAT[int(dye_id)][0]
    m_d,sal_d,m_a,sal_a,base = saliency_for_pair(dye_id, ag)
    combo_smi = DYE_DAT[dye_id][1] + '.' + AG_DAT[ag][1]
    m_combo = Chem.MolFromSmiles(combo_smi)
    AllChem.Compute2DCoords(m_combo)
    sal_combo = np.concatenate([sal_d, sal_a])
    smax = np.abs(sal_combo).max() if np.abs(sal_combo).max()>0 else 1
    sn = sal_combo/smax
    colors={}; radii={}
    for ai,s in enumerate(sn):
        if abs(s)<0.10: continue
        intensity = min(1.0, abs(s)*1.2)
        if (is_tp and s>0) or ((not is_tp) and s<0):
            colors[ai]=(0.55,0.10,0.18,intensity); radii[ai]=0.5+0.4*intensity
    drawer=rdMolDraw2D.MolDraw2DCairo(600,480)
    opts=drawer.drawOptions(); opts.bondLineWidth=1.6; opts.minFontSize=14; opts.maxFontSize=18
    rdMolDraw2D.PrepareAndDrawMolecule(drawer,m_combo,
        highlightAtoms=list(colors.keys()), highlightAtomColors=colors, highlightAtomRadii=radii)
    drawer.FinishDrawing()
    img=Image.open(BytesIO(drawer.GetDrawingText()))
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    bc='#C03B41' if is_tp else '#3461A0'
    for sp_pos in ['top','bottom','left','right']:
        sp=ax.spines[sp_pos]; sp.set_visible(True); sp.set_linewidth(3); sp.set_edgecolor(bc)
    title_str = f"{dye_name} + {ag}"
    ax.set_title(title_str, pad=10, fontsize=13, fontweight='bold', loc='center')
    ax.text(-0.05, 1.07, label_letter.upper(), transform=ax.transAxes,
            fontsize=20, fontweight='bold', ha='left', va='bottom', clip_on=False)
    box_color='#C03B41' if is_tp else '#3461A0'
    ax.text(0.97, 0.04, f'P = {base:.3f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=12, color='white', fontweight='bold',
            bbox=dict(facecolor=box_color, edgecolor='none', boxstyle='round,pad=0.35'))

fig=plt.figure(figsize=(20,12))
g=gs.GridSpec(2,4,figure=fig,hspace=0.30,wspace=0.10,
              width_ratios=[0.10,1,1,1],
              left=0.04,right=0.97,top=0.93,bottom=0.05)
ax_tp=fig.add_subplot(g[0,0]); ax_tp.axis('off')
ax_tp.text(0.5,0.5,'True Positive\n(High Confidence)',transform=ax_tp.transAxes,
            rotation=90,ha='center',va='center',fontsize=15,fontweight='bold',color='#C03B41')
ax_tn=fig.add_subplot(g[1,0]); ax_tn.axis('off')
ax_tn.text(0.5,0.5,'True Negative\n(High Confidence)',transform=ax_tn.transAxes,
            rotation=90,ha='center',va='center',fontsize=15,fontweight='bold',color='#3461A0')

for j,(letter,(d,ag)) in enumerate(zip(['a','b','c'],tp_choices)):
    ax=fig.add_subplot(g[0,j+1]); render_panel(ax,d,ag,letter,is_tp=True)
for j,(letter,(d,ag)) in enumerate(zip(['d','e','f'],tn_choices)):
    ax=fig.add_subplot(g[1,j+1]); render_panel(ax,d,ag,letter,is_tp=False)

plt.savefig('rendered/Fig4.png',dpi=200,bbox_inches='tight')
print('Fig 4 v4 (all-GB) saved')
print('TPs:'); [print(f"  D{d}+{ag}: P=", end='') or print(f"{torch.sigmoid(model(DYE_DAT[d][2][1:],AG_DAT[ag][2][1:])).item():.3f}") for d,ag in tp_choices]
print('TNs:'); [print(f"  D{d}+{ag}: P=", end='') or print(f"{torch.sigmoid(model(DYE_DAT[d][2][1:],AG_DAT[ag][2][1:])).item():.3f}") for d,ag in tn_choices]
