"""Fig S57 (FigS29.png) — Holdout validation with new RFC data.
3 panels:
(a) Bar chart: observed vs predicted response rate per holdout dye + Acc%
(b) Pair-by-pair 5x16 prediction matrix heatmap (TP/TN/FP/FN color)
(c) ROC curve comparison: training CV (AUC=0.880) vs Holdout (AUC=0.619)
"""
import sys
sys.path.insert(0, '../../Package_C/code')
import pandas as pd, numpy as np, pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import _common as C
import warnings; warnings.filterwarnings('ignore')

df = pd.read_excel('../results/new_analysis_matrix.xlsx')
HOLDOUT_DYES = [4, 16, 19, 20, 21]
SEED = 42
exclude = {'dye_num','dye_name','agent','agent_name','response_binary','agent_group'}
feats = [c for c in df.columns if c not in exclude]

train = df[~df.dye_num.isin(HOLDOUT_DYES)].reset_index(drop=True)
hold = df[df.dye_num.isin(HOLDOUT_DYES)].reset_index(drop=True)
X_tr, y_tr = train[feats].values, train.response_binary.astype(int).values
X_h, y_h = hold[feats].values, hold.response_binary.astype(int).values

# Train RF on training subset
rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                              random_state=SEED, n_jobs=-1)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
proba_cv = cross_val_predict(rf, X_tr, y_tr, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
auc_cv = roc_auc_score(y_tr, proba_cv)
fpr_cv, tpr_cv, _ = roc_curve(y_tr, proba_cv)

rf_final = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                    random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
proba_h = rf_final.predict_proba(X_h)[:, 1]
pred_h = (proba_h >= 0.47).astype(int)
auc_h = roc_auc_score(y_h, proba_h)
fpr_h, tpr_h, _ = roc_curve(y_h, proba_h)
acc_h = accuracy_score(y_h, pred_h)

print(f'CV AUC = {auc_cv:.3f}')
print(f'Holdout AUC = {auc_h:.3f}, accuracy = {acc_h*100:.1f}%')

# Per-dye stats
hold = hold.assign(pred=pred_h, proba=proba_h)
agent_order = ['A230','A232','A234','A242','AC','CG','CK','GA','GB','GD','GF','HD','HN3','L','PS','VX']
dye_names = hold.groupby('dye_num').dye_name.first().to_dict()
hold_dye_names = {d: dye_names[d] for d in HOLDOUT_DYES}
print('\nPer-dye holdout accuracy:')
for d in HOLDOUT_DYES:
    sub = hold[hold.dye_num == d]
    obs_rate = sub.response_binary.mean()
    pred_rate = sub.pred.mean()
    acc = (sub.response_binary == sub.pred).mean()
    print(f'  D{d} {hold_dye_names[d][:25]:<25}  obs={obs_rate*100:5.1f}%  pred={pred_rate*100:5.1f}%  acc={acc*100:5.1f}%')

# Layout: 3 panels horizontal
fig = plt.figure(figsize=(20, 6.5))
g = gs.GridSpec(1, 3, figure=fig, wspace=0.32, left=0.05, right=0.97, top=0.92, bottom=0.12,
                width_ratios=[1.0, 1.5, 1.0])

def plabel(ax, label):
    ax.text(-0.13, 1.06, label, transform=ax.transAxes,
            fontsize=20, fontweight='bold', ha='left', va='bottom', clip_on=False)

# (a) Per-dye bar: observed vs predicted rate
ax_a = fig.add_subplot(g[0, 0])
y_pos = np.arange(5)
obs_rates = [hold[hold.dye_num==d].response_binary.mean()*100 for d in HOLDOUT_DYES]
pred_rates = [hold[hold.dye_num==d].pred.mean()*100 for d in HOLDOUT_DYES]
accs = [(hold[hold.dye_num==d].response_binary == hold[hold.dye_num==d].pred).mean()*100 for d in HOLDOUT_DYES]
w = 0.35
ax_a.barh(y_pos - w/2, obs_rates, w, color='#3F8E8C', edgecolor='white', label='Observed')
ax_a.barh(y_pos + w/2, pred_rates, w, color='#999999', edgecolor='white', hatch='///',
          label='Predicted')
ax_a.set_yticks(y_pos)
labels = [f"{hold_dye_names[d][:18]}\n(D{d}, Acc={accs[i]:.0f}%)" for i, d in enumerate(HOLDOUT_DYES)]
ax_a.set_yticklabels(labels, fontsize=10)
ax_a.invert_yaxis()
ax_a.set_xlim(0, 110)
ax_a.set_xlabel('Response rate (%)', fontsize=12)
ax_a.legend(loc='lower right', fontsize=10, framealpha=0.95, edgecolor='#cccccc')
ax_a.grid(axis='x', color='#DDDDDD', ls='--', lw=0.5)
plabel(ax_a, 'A')

# (b) Pair-by-pair 5x16 matrix (TP/TN/FP/FN)
mat = np.zeros((5, 16), dtype=int)  # 0=TN, 1=FP, 2=FN, 3=TP
for i, d in enumerate(HOLDOUT_DYES):
    for j, ag in enumerate(agent_order):
        sub = hold[(hold.dye_num == d) & (hold.agent_name == ag)]
        if not len(sub): continue
        y_t = int(sub.response_binary.iloc[0])
        y_p = int(sub.pred.iloc[0])
        if y_t == 0 and y_p == 0: mat[i, j] = 0  # TN
        elif y_t == 0 and y_p == 1: mat[i, j] = 1  # FP
        elif y_t == 1 and y_p == 0: mat[i, j] = 2  # FN
        elif y_t == 1 and y_p == 1: mat[i, j] = 3  # TP

n_tn = (mat==0).sum(); n_fp = (mat==1).sum(); n_fn = (mat==2).sum(); n_tp = (mat==3).sum()
print(f'\nHoldout CM: TN={n_tn} FP={n_fp} FN={n_fn} TP={n_tp}  (total {mat.size})')

ax_b = fig.add_subplot(g[0, 1])
# Custom colormap: TN=light grey, FP=orange, FN=red, TP=teal
cmap = ListedColormap(['#EEEEEE', '#E8943A', '#C94F4A', '#3F8E8C'])
ax_b.imshow(mat, aspect='auto', cmap=cmap, vmin=-0.5, vmax=3.5, interpolation='nearest')
# White grid
for x in np.arange(-0.5, 16, 1): ax_b.axvline(x, color='white', linewidth=1)
for y in np.arange(-0.5, 5, 1): ax_b.axhline(y, color='white', linewidth=1)
ax_b.set_xticks(np.arange(16))
ax_b.set_xticklabels(agent_order, rotation=45, ha='right', fontsize=10)
ax_b.set_yticks(np.arange(5))
ax_b.set_yticklabels([f'D{d}' for d in HOLDOUT_DYES], fontsize=11)
ax_b.set_xlabel('Chemical Warfare Agents', fontsize=12)
ax_b.set_ylabel('Holdout Dyes', fontsize=12)
# Legend below the heatmap
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3F8E8C', edgecolor='#333333', label=f'TP ({n_tp})'),
    Patch(facecolor='#EEEEEE', edgecolor='#333333', label=f'TN ({n_tn})'),
    Patch(facecolor='#E8943A', edgecolor='#333333', label=f'FP ({n_fp})'),
    Patch(facecolor='#C94F4A', edgecolor='#333333', label=f'FN ({n_fn})'),
]
ax_b.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.10),
            ncol=4, fontsize=10, framealpha=0.95, edgecolor='#cccccc')
ax_b.grid(False)
plabel(ax_b, 'B')

# (c) ROC curves: CV vs Holdout
ax_c = fig.add_subplot(g[0, 2])
ax_c.plot(fpr_cv, tpr_cv, color='#3F8E8C', lw=2.5, label=f'Training CV (AUC = {auc_cv:.3f})')
ax_c.plot(fpr_h, tpr_h, color='#C94F4A', lw=2.5, linestyle='--', label=f'Holdout (AUC = {auc_h:.3f})')
ax_c.plot([0,1], [0,1], color='#999999', ls=':', lw=1)
ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1.02)
ax_c.set_xlabel('False Positive Rate', fontsize=12)
ax_c.set_ylabel('True Positive Rate', fontsize=12)
ax_c.legend(loc='lower right', fontsize=11, framealpha=0.95, edgecolor='#cccccc')
ax_c.grid(True, color='#DDDDDD', ls='--', lw=0.5)
ax_c.set_title(f'Holdout accuracy: {acc_h*100:.1f}%  ({n_tp+n_tn}/80)', fontsize=11, pad=8)
plabel(ax_c, 'C')

plt.savefig('rendered/FigS29.png', dpi=200, bbox_inches='tight')
print('\nFigS29.png (Fig S57) saved')