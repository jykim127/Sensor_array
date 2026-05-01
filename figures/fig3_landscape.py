"""Fig 3 v3 — match original design.
(a) Horizontal bar with error bars + 'Overall mean' dashed vertical line + legend
(b-e) Scatter with centered title, filled blue KDE contours, blue/grey dots, centroid star, legend
Panel labels lowercase 'a', 'b', etc. at outside top-left.
"""
import sys
sys.path.insert(0, '../../Package_C/code')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.colors as mc
from scipy.stats import gaussian_kde
import _common as C

df = pd.read_excel('../results/new_analysis_matrix.xlsx')

# Panel (a) class order matches original (G-agents on top)
classes = ['G-agents','V-agents','Novichok','Vesicants','Blood','Choking']
class_colors = ['#2C7BB6','#80B4D9','#D7404F','#F4B084','#5BAA56','#A8D58F']  # blue, light blue, red, peach, green, light green

def plabel(ax, label):
    ax.text(-0.13, 1.04, label, transform=ax.transAxes,
            fontsize=18, fontweight='bold', ha='left', va='bottom', clip_on=False)

# Compute per-class agg + std (across agents within class)
# Per-class mean response rate is across all 28 dyes × n_agents pairs (binary mean)
# For error bars, use per-agent variation within each class
class_data = []
for cls in classes:
    sub = df[df.agent_group == cls]
    rates_per_agent = sub.groupby('agent_name').response_binary.mean().values
    class_data.append({
        'class': cls,
        'mean': sub.response_binary.mean(),
        'std': rates_per_agent.std() if len(rates_per_agent) > 1 else 0,
        'n': len(sub)
    })
overall_mean = df.response_binary.mean()
print(f'Overall mean: {overall_mean:.2f}')
for d in class_data: print(f"  {d['class']:<10} mean={d['mean']:.2f}±{d['std']:.2f} n={d['n']}")

# Layout: A on left (full height), B/C/D/E on right (2x2)
fig = plt.figure(figsize=(20, 11))
g = gs.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.25,
                width_ratios=[1.4, 0.02, 1.0, 1.0],
                left=0.05, right=0.97, top=0.93, bottom=0.08)

# ── (a) Horizontal bar with error bars + overall mean line ────────────
ax_a = fig.add_subplot(g[:, 0])
y_pos = np.arange(len(classes))
means = [d['mean'] for d in class_data]
stds  = [d['std'] for d in class_data]
bars = ax_a.barh(y_pos, means, xerr=stds, color=class_colors, edgecolor='white',
                  height=0.65, capsize=5,
                  error_kw=dict(elinewidth=1.5, ecolor='#444444'))
# Add value labels — placed to the right of the error bar end (m + std + offset)
for bar, m, s, color in zip(bars, means, stds, class_colors):
    ax_a.text(m + s + 0.025, bar.get_y() + bar.get_height()/2, f'{m:.2f}',
              ha='left', va='center', fontsize=12, color=color, fontweight='bold')
ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(classes, fontsize=13)
ax_a.invert_yaxis()
ax_a.set_xlim(0, 0.9)
ax_a.set_xlabel('Aggregate Response Rate', fontsize=14)
ax_a.axvline(x=overall_mean, color='#444444', linestyle='--', linewidth=1.5,
              label=f'Overall mean\n({overall_mean:.2f})')
ax_a.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='#cccccc',
             frameon=True)
ax_a.grid(axis='x', color='#DDDDDD', ls='--', lw=0.6)
plabel(ax_a, 'a')

# ── (b-e) Scatter with filled blue KDE + centroid + legend ─────────────
panel_letters = ['b','c','d','e']
target_classes = ['G-agents','V-agents','Novichok','Vesicants']
panel_titles = ['G-series','V-series','Novichok','Vesicants']
positions = [(0,2),(0,3),(1,2),(1,3)]

x_max = df.delta_LogP.max() + 1
y_max = df.sum_TPSA.max() + 20

for i, (g_name, letter, title, (r,c)) in enumerate(zip(target_classes, panel_letters, panel_titles, positions)):
    ax = fig.add_subplot(g[r, c])
    sub = df[df.agent_group == g_name]
    resp = sub[sub.response_binary == 1]
    nonresp = sub[sub.response_binary == 0]

    # KDE filled contours (blue gradient)
    if len(resp) > 5:
        try:
            kde = gaussian_kde(np.vstack([resp.delta_LogP, resp.sum_TPSA]))
            xx, yy = np.meshgrid(np.linspace(-1, x_max, 100), np.linspace(-10, y_max, 100))
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            # Use multiple filled contour levels for smooth gradient
            ax.contourf(xx, yy, zz, levels=8, cmap='Blues', alpha=0.45)
        except Exception:
            pass
    # Non-responsive grey, very faint
    ax.scatter(nonresp.delta_LogP, nonresp.sum_TPSA, color='#DDDDDD', s=18,
                alpha=0.5, edgecolor='none', label='Non-responsive', zorder=2)
    # Responsive blue
    ax.scatter(resp.delta_LogP, resp.sum_TPSA, color='#1f77b4', s=30, alpha=0.85,
                edgecolor='white', linewidth=0.5, label=f'Responsive (n={len(resp)})', zorder=3)
    # Centroid star
    cx, cy = resp.delta_LogP.mean(), resp.sum_TPSA.mean()
    ax.scatter([cx], [cy], color='#C94F4A', s=240, marker='*', zorder=10,
                edgecolor='#333333', linewidth=1.2, label=f'Centroid ({cx:.1f}, {cy:.0f})')

    ax.set_xlim(-1, 12); ax.set_ylim(-10, 360)
    ax.set_xlabel(r'$\Delta$LogP  (Dye $-$ Agent)', fontsize=12)
    ax.set_ylabel(r'$\Sigma$TPSA  (Dye $+$ Agent)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='#cccccc',
               frameon=True)
    ax.grid(True, color='#DDDDDD', ls='--', lw=0.6)
    plabel(ax, letter)

plt.savefig('rendered/Fig3.png', dpi=200, bbox_inches='tight')
print('Fig 3 v3 saved')
