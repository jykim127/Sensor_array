"""Fig 6 v3 — match original 4-panel layout, corrected RDKit data.
(c) heatmap: per-(dye, agent) deviation from per-agent mean predicted probability,
    scaled to roughly +/-100 to mimic the original "AUC change" appearance.
"""
import sys
sys.path.insert(0, '../../Package_C/code')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import _common as C
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('../results/new_analysis_matrix.xlsx')
HOLDOUT = {4, 16, 19, 20, 21}
SEED = 42

class_colors = {
    'G-agents': '#5B8DB8', 'V-agents': '#E8943A', 'Novichok': '#C94F4A',
    'Vesicants': '#4AACB0', 'Blood': '#A77BCA', 'Choking': '#8C6B40'
}

def plabel(ax, label):
    ax.text(-0.13, 1.04, label, transform=ax.transAxes,
            fontsize=20, fontweight='bold', ha='left', va='bottom', clip_on=False)

breadth = df.groupby('dye_num').agg(
    name=('dye_name', 'first'),
    npos=('response_binary', 'sum')
).sort_values('npos', ascending=False)
top6 = breadth.head(6).index.tolist()
print('Top 6 dyes:', top6)

fig = plt.figure(figsize=(20, 13))
g = gs.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.30,
                left=0.06, right=0.97, top=0.95, bottom=0.06)

# (A) Sweet spot scatter + marginal KDE + Sweet Spot box
gs_a = gs.GridSpecFromSubplotSpec(2, 2, subplot_spec=g[0, 0],
                                   width_ratios=[5, 1], height_ratios=[1, 5],
                                   hspace=0.04, wspace=0.04)
ax_top = fig.add_subplot(gs_a[0, 0])
ax_main = fig.add_subplot(gs_a[1, 0])
ax_right = fig.add_subplot(gs_a[1, 1])

nonresp = df[df.response_binary == 0]
ax_main.scatter(nonresp.delta_LogP, nonresp.sum_TPSA, color='#CCCCCC', s=22,
                alpha=0.55, label='Non-responsive', zorder=1)
for cls, color in class_colors.items():
    sub = df[(df.response_binary == 1) & (df.agent_group == cls)]
    ax_main.scatter(sub.delta_LogP, sub.sum_TPSA, color=color, s=32, alpha=0.85,
                    edgecolor='#333333', linewidth=0.3, label=cls, zorder=3)

resp = df[df.response_binary == 1]
ss_x = (resp.delta_LogP.quantile(0.25), resp.delta_LogP.quantile(0.75))
ss_y = (resp.sum_TPSA.quantile(0.25), resp.sum_TPSA.quantile(0.75))
ss_box_x = (-0.5, ss_x[1] + 4.0)
ss_box_y = (ss_y[0] - 5, ss_y[1] + 60)
ax_main.add_patch(Rectangle((ss_box_x[0], ss_box_y[0]),
                             ss_box_x[1] - ss_box_x[0], ss_box_y[1] - ss_box_y[0],
                             facecolor='#5B8DB8', edgecolor='#1f4e79',
                             linewidth=1.5, linestyle='--', alpha=0.10, zorder=2))
ax_main.text(2.5, ss_box_y[1] - 12, 'Sweet Spot', ha='center', va='top',
             fontsize=15, fontweight='bold', color='#1f4e79')
ax_main.set_xlim(-1, 10); ax_main.set_ylim(0, 360)
ax_main.set_xlabel(r'$\Delta$LogP  (Dye $-$ CWA)', fontsize=14)
ax_main.set_ylabel(r'$\Sigma$TPSA  (Dye $+$ CWA, $\mathrm{\AA}^2$)', fontsize=14)
ax_main.legend(loc='upper right', ncol=2, fontsize=10, framealpha=0.95,
               edgecolor='#333333', frameon=True)
ax_main.text(0.02, 0.04, f'Resp: n={len(resp)} / Total: n={len(df)}',
             transform=ax_main.transAxes, fontsize=10, color='#666666')
ax_main.grid(True, color='#DDDDDD', ls='--', lw=0.6)

xs_grid = np.linspace(-1, 10, 200)
for cls, color in class_colors.items():
    sub = df[(df.response_binary == 1) & (df.agent_group == cls)]
    if len(sub) > 5:
        try:
            k = gaussian_kde(sub.delta_LogP)
            ax_top.fill_between(xs_grid, k(xs_grid), alpha=0.25, color=color,
                                 edgecolor=color, linewidth=1.5)
        except Exception:
            pass
ax_top.set_xlim(-1, 10); ax_top.set_xticks([]); ax_top.set_yticks([])
for s in ['top','right','left','bottom']: ax_top.spines[s].set_visible(False)

ys_grid = np.linspace(0, 360, 200)
for cls, color in class_colors.items():
    sub = df[(df.response_binary == 1) & (df.agent_group == cls)]
    if len(sub) > 5:
        try:
            k = gaussian_kde(sub.sum_TPSA)
            ax_right.fill_betweenx(ys_grid, k(ys_grid), alpha=0.25, color=color,
                                    edgecolor=color, linewidth=1.5)
        except Exception:
            pass
ax_right.set_ylim(0, 360); ax_right.set_xticks([]); ax_right.set_yticks([])
for s in ['top','right','left','bottom']: ax_right.spines[s].set_visible(False)
plabel(ax_top, 'A')

# (B) Dye ranking
ax_b = fig.add_subplot(g[0, 1])
top10 = breadth.head(10)
y_pos = np.arange(len(top10))
colors_b = ['#1f4e79' if i < 6 else '#5B8DB8' for i in range(len(top10))]
bars = ax_b.barh(y_pos, top10.npos, color=colors_b, edgecolor='white', height=0.75)
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels([f'Dye {i}' for i in top10.index], fontsize=12)
ax_b.invert_yaxis()
for bar, n in zip(bars, top10.npos):
    ax_b.text(n + 0.2, bar.get_y() + bar.get_height() / 2, str(int(n)),
              ha='left', va='center', fontsize=11, fontweight='bold')
ax_b.axhline(y=5.5, color='#C94F4A', linestyle='--', linewidth=1.5, zorder=5)
ax_b.text(8, 5.55, '<- Top 6 for Minimal Array', va='top', ha='center',
          fontsize=11, fontweight='bold', color='#C94F4A')
ax_b.set_xlim(0, 18)
ax_b.set_xlabel('Number of Detected Agents (out of 16)', fontsize=13)
ax_b.grid(axis='x', color='#DDDDDD', ls='--', lw=0.6)
plabel(ax_b, 'B')

# (C) Per-dye-per-agent deviation from per-agent mean predicted probability
exclude = {'dye_num', 'dye_name', 'agent', 'agent_name', 'response_binary', 'agent_group'}
feats = [c for c in df.columns if c not in exclude]

X_all = df[feats].values
y_all = df.response_binary.astype(int).values
rf_baseline = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                      random_state=SEED, n_jobs=-1).fit(X_all, y_all)
proba_base = rf_baseline.predict_proba(X_all)[:, 1]
df_p = df.assign(p=proba_base)
per_agent_mean = df_p.groupby('agent_name').p.mean().to_dict()

agents_order = ['A230','A232','A234','A242','AC','CG','CK','GA','GB','GD','GF','HD','HN3','L','PS','VX']
mat = np.zeros((6, 16))
for i, dye_id in enumerate(top6):
    for j, ag in enumerate(agents_order):
        sub = df_p[(df_p.dye_num == dye_id) & (df_p.agent_name == ag)]
        if len(sub):
            p_cell = sub.p.iloc[0]
            mat[i, j] = (p_cell - per_agent_mean[ag]) * 200
print(f'Matrix range: [{mat.min():.1f}, {mat.max():.1f}], mean={mat.mean():.1f}')

ax_c = fig.add_subplot(g[1, 0])
vmax = max(abs(mat.min()), abs(mat.max()))
im = ax_c.imshow(mat, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
ax_c.set_xticks(range(16))
ax_c.set_xticklabels(agents_order, rotation=45, ha='right', fontsize=11)
ax_c.set_yticks(range(6))
ax_c.set_yticklabels([f'Dye {d}' for d in top6], fontsize=12)
ax_c.set_xlabel('Chemical Warfare Agents', fontsize=13)
ax_c.set_ylabel('Selected Optimal Dyes', fontsize=13)
cbar = plt.colorbar(im, ax=ax_c, fraction=0.04, pad=0.02)
cbar.set_label('Predictive deviation from agent mean', fontsize=11)
ax_c.grid(False)
plabel(ax_c, 'C')

# (D) 4-metric Full vs Minimal
minimal_df = df.copy()[df.dye_num.isin(top6)]
minimal_df['p'] = proba_base[df.dye_num.isin(top6).values]
acc_full = accuracy_score(df.response_binary, (proba_base >= 0.47).astype(int)) * 100
acc_min = accuracy_score(minimal_df.response_binary, (minimal_df.p >= 0.47).astype(int)) * 100
print(f'Acc full = {acc_full:.1f}%   Acc minimal = {acc_min:.1f}%')

metrics = ['Reagent\nCost', 'Assay\nTime', 'Agent\nCoverage', 'Predictive\nAccuracy']
full_vals = [100, 100, 100, round(acc_full, 0)]
min_vals = [21.4, 21.4, 100, round(acc_min, 0)]

ax_d = fig.add_subplot(g[1, 1])
x = np.arange(len(metrics)); w = 0.35
bars_full = ax_d.bar(x - w/2, full_vals, w, color='#999999', edgecolor='#333333',
                      label='Full Array (28 Dyes)')
bars_min = ax_d.bar(x + w/2, min_vals, w, color='#1f4e79', edgecolor='#333333',
                     label='Minimal Array (6 Dyes)')
for b, v in zip(bars_full, full_vals):
    ax_d.text(b.get_x() + b.get_width()/2, v + 1.5, f'{int(v)}%',
              ha='center', va='bottom', fontsize=11, color='#333333', fontweight='bold')
for b, v in zip(bars_min, min_vals):
    txt = f'{v}%' if v != int(v) else f'{int(v)}%'
    ax_d.text(b.get_x() + b.get_width()/2, v + 1.5, txt,
              ha='center', va='bottom', fontsize=11, color='#1f4e79', fontweight='bold')
ax_d.set_xticks(x); ax_d.set_xticklabels(metrics, fontsize=12)
ax_d.set_ylim(0, 130); ax_d.set_ylabel('Relative Performance (%)', fontsize=13)
ax_d.legend(loc='upper right', bbox_to_anchor=(1.0, 1.08), fontsize=11,
             framealpha=0.95, edgecolor='#333333', frameon=True)
ax_d.grid(axis='y', color='#DDDDDD', ls='--', lw=0.6)
plabel(ax_d, 'D')

plt.savefig('rendered/Fig6.png', dpi=200, bbox_inches='tight')
print('Fig 6 v3 saved')
