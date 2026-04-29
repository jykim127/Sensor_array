"""Larger MPNN: h=96, 3 rounds, 10-fold CV, 12 epochs/fold, batch=32.
Saves checkpoint after EACH fold so we can resume on timeout."""
import sys, json, time, pickle, os
sys.path.insert(0, '../../Package_B/code')
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score
from corrected_smiles import DYES, AGENTS

SEED=42; HOLDOUT={4,16,19,20,21}
torch.manual_seed(SEED); np.random.seed(SEED)

ELEMS=['C','N','O','S','P','F','Cl','Br','I','As']
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
    X=torch.tensor([af(a) for a in m.GetAtoms()],dtype=torch.float32)
    e=[]; ef=[]
    for b in m.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); fb=bf(b)
        e+=[(i,j),(j,i)]; ef+=[fb,fb]
    if not e: e=[(0,0)]; ef=[[0,0,0,0]]
    return X,torch.tensor(e,dtype=torch.long).t().contiguous(),torch.tensor(ef,dtype=torch.float32)

DG={n:mol2g(smi) for n,_,_,smi,_ in DYES}
AG={c:mol2g(smi) for c,_,_,smi in AGENTS}

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

df=pd.read_excel('../../Package_B/results/new_analysis_matrix.xlsx')
train=df[~df.dye_num.isin(HOLDOUT)].reset_index(drop=True)
hold=df[df.dye_num.isin(HOLDOUT)].reset_index(drop=True)
y_tr=train.response_binary.astype(int).values
y_h=hold.response_binary.astype(int).values

# Determine which fold to start from based on existing checkpoint
ckpt_dir='../verification/checkpoints'
os.makedirs(ckpt_dir, exist_ok=True)
folds=[]
for fn in sorted(os.listdir(ckpt_dir)):
    if fn.startswith('fold_') and fn.endswith('.json'):
        folds.append(json.load(open(f'{ckpt_dir}/{fn}')))
done_folds=set(f['fold'] for f in folds)
print(f"Already done: {sorted(done_folds)}", flush=True)

# Fold split (deterministic with seed)
skf=StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
fold_splits=list(skf.split(np.arange(len(train)), y_tr))

# Allow command line arg to specify max number of folds to do this run
import sys
max_folds=int(sys.argv[1]) if len(sys.argv)>1 else 5

t0=time.time()
done_this_run=0
for fi,(tr_i,te_i) in enumerate(fold_splits, 1):
    if fi in done_folds: continue
    if done_this_run >= max_folds: break
    if time.time()-t0 > 32:  # leave time for save
        print(f"  time budget hit at fold {fi}, saving and exiting", flush=True)
        break
    torch.manual_seed(SEED+fi); np.random.seed(SEED+fi)
    model=MPNN(h=96,n=3)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    pos_w=torch.tensor(float((y_tr[tr_i]==0).sum()/max((y_tr[tr_i]==1).sum(),1)))
    for epoch in range(12):
        model.train(); np.random.shuffle(tr_i)
        for bi in range(0,len(tr_i),32):
            opt.zero_grad()
            ls=[]
            for idx in tr_i[bi:bi+32]:
                r=train.iloc[idx]
                logit=model(DG[int(r.dye_num)], AG[r.agent_name])
                ls.append(F.binary_cross_entropy_with_logits(logit,torch.tensor(float(y_tr[idx])),pos_weight=pos_w))
            torch.stack(ls).mean().backward(); opt.step()
    model.eval()
    with torch.no_grad():
        probs=np.array([torch.sigmoid(model(DG[int(train.iloc[i].dye_num)],AG[train.iloc[i].agent_name])).item() for i in te_i])
    yt=y_tr[te_i]; pred=(probs>=0.5).astype(int)
    f={'fold':fi,'AUC':roc_auc_score(yt,probs),'BA':balanced_accuracy_score(yt,pred),
       'Prec':precision_score(yt,pred,zero_division=0),'Rec':recall_score(yt,pred,zero_division=0),
       'F1':f1_score(yt,pred,zero_division=0)}
    folds.append(f)
    elapsed=time.time()-t0
    print(f"  fold {fi}: AUC={f['AUC']:.3f} BA={f['BA']:.3f} ({elapsed:.0f}s)", flush=True)
    json.dump(f, open(f'{ckpt_dir}/fold_{fi:02d}.json','w'))
    done_this_run+=1

print(f"\nProgress: {len(folds)}/10 folds done", flush=True)
if len(folds)==10:
    fold_df=pd.DataFrame(folds).sort_values('fold')
    fold_df.to_excel('../verification/mpnn_big_cv_metrics.xlsx',index=False)
    mean=fold_df.mean(numeric_only=True); sd=fold_df.std(numeric_only=True)
    print(f"\n=== 10-fold CV (h=96, n=3, 12ep) ===")
    print(f"  AUC = {mean.AUC:.3f} ± {sd.AUC:.3f}")
    print(f"  BA  = {mean.BA:.3f} ± {sd.BA:.3f}")
    print(f"  Prec= {mean.Prec:.3f} ± {sd.Prec:.3f}")
    print(f"  Rec = {mean.Rec:.3f} ± {sd.Rec:.3f}")
