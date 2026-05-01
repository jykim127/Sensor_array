"""Fig 2 v3 — panel labels OUTSIDE top-left + Y patch (mean fold AUC)."""
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import _common as C

df = pd.read_excel('../results/new_analysis_matrix.xlsx')
holdout={4,16,19,20,21}
train = df[~df.dye_num.isin(holdout)].reset_index(drop=True)
exclude={'dye_num','dye_name','agent','agent_name','response_binary','agent_group'}
feats=[c for c in df.columns if c not in exclude]
X=train[feats].values; y=train.response_binary.astype(int).values
SEED=42
skf=StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
proba_cv=cross_val_predict(
    RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=SEED, n_jobs=-1),
    X, y, cv=skf, method='predict_proba', n_jobs=-1)[:,1]

# Per-fold AUC mean+/-std (Y patch)
fold_aucs=[]
for tr,te in skf.split(X,y):
    rf_f=RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=SEED, n_jobs=-1).fit(X[tr],y[tr])
    p=rf_f.predict_proba(X[te])[:,1]
    fold_aucs.append(roc_auc_score(y[te], p))
auc_mean=float(np.mean(fold_aucs)); auc_std=float(np.std(fold_aucs, ddof=1))  # sample std (N-1) — matches analysis_main.py pandas .std()
auc=roc_auc_score(y, proba_cv); fpr,tpr,thr=roc_curve(y, proba_cv); j=np.argmax(tpr-fpr); ystar=thr[j]
print(f'mean fold AUC={auc_mean:.3f}+/-{auc_std:.3f}  aggregated AUC={auc:.3f}', flush=True)

pred47=(proba_cv>=0.47).astype(int); cm=confusion_matrix(y, pred47)
prec=cm[1,1]/(cm[1,1]+cm[0,1]); rec=cm[1,1]/(cm[1,1]+cm[1,0])
rf_final=RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=SEED, n_jobs=-1).fit(X,y)
fi=pd.Series(rf_final.feature_importances_, index=feats).sort_values(ascending=True).tail(12)

def plabel(ax, label):
    ax.text(-0.13, 1.04, label, transform=ax.transAxes,
            fontsize=20, fontweight='bold', ha='left', va='bottom', clip_on=False)

fig, ax = plt.subplots(2,3, figsize=(20,10.5))
plt.subplots_adjust(hspace=0.55, wspace=0.42, left=0.08, right=0.97, top=0.93, bottom=0.08)

# (A) ROC
a=ax[0,0]
a.plot(fpr, tpr, color=C.KU[0], lw=2.5,
       label=f'10-fold CV\n(mean fold AUC = {auc_mean:.3f}$\\pm${auc_std:.3f};\naggregated = {auc:.3f})')
a.plot([0,1],[0,1], color='#999999', ls='--', lw=1)
fpr_47=cm[0,1]/(cm[0,0]+cm[0,1]); tpr_47=cm[1,1]/(cm[1,0]+cm[1,1])
a.scatter([fpr_47],[tpr_47], color=C.KU[1], s=90, zorder=5, label='theta = 0.47 (operating)')
a.scatter([fpr[j]],[tpr[j]], color=C.KU[3], s=90, zorder=5, marker='D', label=f'Youden theta = {ystar:.2f}')
a.set_xlim(0,1); a.set_ylim(0,1.02)
a.set_xlabel('False Positive Rate'); a.set_ylabel('True Positive Rate')
a.legend(loc='lower right', fontsize=10, framealpha=0.95, edgecolor='#333333', frameon=True)
plabel(a,'A')

# (B) Confusion matrix
b=ax[0,1]
b.imshow(cm, cmap='Reds', aspect='auto', vmin=0)
for i in range(2):
    for k in range(2):
        b.text(k, i, str(cm[i,k]), ha='center', va='center',
               color='white' if cm[i,k]>cm.max()*0.5 else '#333333',
               fontsize=16, fontweight='bold')
b.set_xticks([0,1]); b.set_yticks([0,1])
b.set_xticklabels(['Non-resp.','Responsive'])
b.set_yticklabels(['Non-resp.','Responsive'])
b.set_xlabel('Predicted'); b.set_ylabel('True')
b.text(0.5, 1.04,
       f'$\\theta = 0.47$:  Precision = {prec*100:.1f}%   Recall = {rec*100:.1f}%\n'
       f'aggregated AUC = {auc:.3f};  mean fold AUC = {auc_mean:.3f}$\\pm${auc_std:.3f}',
       transform=b.transAxes, ha='center', va='bottom', fontsize=10,
       bbox=dict(facecolor='white', edgecolor='#333333', boxstyle='round,pad=0.30'))
b.grid(False)
plabel(b,'B')

# (C) Feature importance
c=ax[0,2]
c.barh(fi.index, fi.values, color=C.KU[2], edgecolor='white')
c.set_xlabel('RF Feature Importance')
c.tick_params(axis='y', labelsize=11)
c.grid(axis='x', color='#DDDDDD', ls='--', lw=0.6)
plabel(c,'C')

# (D-F) Binned response rate
cn=0
panels=[('dye_LogP',(1,0),0,'Dye LogP'),('delta_LogP',(1,1),1,'|dLogP|'),('dye_MW',(1,2),2,'Dye MW (Da)')]
for feat, posn, color_idx, x_label in panels:
    d=ax[posn]
    bins=np.histogram_bin_edges(df[feat], bins=8)
    df['_b']=pd.cut(df[feat], bins=bins, include_lowest=True)
    g=df.groupby('_b').agg(rate=('response_binary','mean'), n=('response_binary','size'))
    g['rate']=g['rate']*100
    centers=[(iv.left+iv.right)/2 for iv in g.index]
    width=(bins[1]-bins[0])*0.92
    d.bar(centers, g['rate'], width=width, color=C.KU[color_idx], edgecolor='#333333', linewidth=0.5)
    for x,h,nv in zip(centers, g['rate'], g['n']):
        d.text(x, min(h+3,108), f'n={nv}', ha='center', va='bottom', fontsize=9, color='#444444')
    d.set_xlabel(x_label); d.set_ylabel('Response rate (%)')
    d.set_ylim(0, 115)
    d.set_xlim(bins[0]-(bins[1]-bins[0])*0.5, bins[-1]+(bins[1]-bins[0])*0.5)
    d.grid(axis='y', color='#DDDDDD', ls='--', lw=0.6)
    plabel(d, ['D','E','F'][cn])
    cn=cn+1

print('about to savefig', flush=True)
plt.savefig('rendered/Fig2.png', dpi=200, bbox_inches='tight')
print('Fig 2 v3 saved (Y patch)', flush=True)
