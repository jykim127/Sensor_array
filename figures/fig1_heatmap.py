"""Fig 1 v2 — match original design.
- 28 dyes × 16 agents binary response heatmap (teal = responsive, light grey = non)
- Y-axis: dye 1 -> dye 28 (sorted by dye_num), labeled with name + (LogP) using corrected RDKit LogP
- X-axis: 16 agents grouped by class with colored class headers at top
  G (4) | V (1) | Novichok (4) | Vesicant (3) | Blood (2) | Choking (2)
- Right marginal: per-dye response rate (teal bars) + dashed overall mean
- Bottom marginal: per-agent response rate (colored by class) + dashed overall mean
"""
import sys
sys.path.insert(0, '../../Package_C/code')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.colors import ListedColormap
import _common as C

df = pd.read_excel('../results/new_analysis_matrix.xlsx')

# Class header colors (match Fig 3 panel a bar colors)
class_info = [
    ('G',          ['GA','GB','GD','GF'],         '#2C7BB6'),
    ('V',          ['VX'],                         '#80B4D9'),
    ('Novichok',   ['A230','A232','A234','A242'], '#D7404F'),
    ('Vesicant',   ['HD','HN3','L'],               '#F4B084'),
    ('Blood',      ['AC','CK'],                    '#5BAA56'),
    ('Choking',    ['CG','PS'],                    '#A8D58F'),
]
agent_order = []
agent_colors = []
class_boundaries = []  # x-positions for class headers
pos = 0
for label, agents, color in class_info:
    class_boundaries.append((label, pos, pos + len(agents) - 1, color))
    agent_order.extend(agents)
    agent_colors.extend([color]*len(agents))
    pos += len(agents)

# Build 28-row heatmap, dye_num 1..28 (top to bottom)
dye_order = list(range(1, 29))
mat = df.pivot_table(index='dye_num', columns='agent_name', values='response_binary',
                      aggfunc='first').reindex(index=dye_order, columns=agent_order)

# Per-dye name + corrected LogP for y-axis labels
dye_info = df.groupby('dye_num').agg(
    name=('dye_name','first'),
    logP=('dye_LogP','first')
).reindex(dye_order)
ylabels = [f"{r['name']:<26}  ({r['logP']:+.1f})" for _, r in dye_info.iterrows()]

# Per-dye / per-agent response rates
dye_rate = mat.mean(axis=1).values
agent_rate = mat.mean(axis=0).values
overall = df.response_binary.mean()
print(f'Overall mean: {overall:.2f}')

# Layout: 4-zone composition
# top class headers | main heatmap (with row labels on left) | right marginal (Dye RR)
# bottom-left empty | bottom marginal (Agent RR)            | bottom-right empty
fig = plt.figure(figsize=(20, 14))
g = gs.GridSpec(3, 3, figure=fig,
                width_ratios=[7, 0.10, 1.5],   # main / spacer (narrower) / right bar
                height_ratios=[0.25, 11, 1.5], # class headers (slimmer) / main / bottom bar
                hspace=0.02, wspace=0.02,
                left=0.16, right=0.95, top=0.96, bottom=0.06)

ax_top = fig.add_subplot(g[0, 0])
ax_main = fig.add_subplot(g[1, 0])
ax_right = fig.add_subplot(g[1, 2])
ax_bot = fig.add_subplot(g[2, 0])

# (1) class headers
ax_top.set_xlim(-0.5, 15.5); ax_top.set_ylim(0, 1)
for label, start, end, color in class_boundaries:
    midx = (start + end) / 2
    # bottom-anchored text → labels sit just above heatmap edge
    ax_top.text(midx, 0.05, label, ha='center', va='bottom',
                fontsize=15, fontweight='bold', color=color)
ax_top.axis('off')

# (2) main heatmap — teal/light-grey binary
cmap = ListedColormap(['#EFEFEF', '#3F8E8C'])  # teal #3F8E8C
ax_main.imshow(mat.values, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
ax_main.set_xticks(np.arange(16))
ax_main.set_xticklabels([])  # x labels covered by bottom bar
ax_main.set_yticks(np.arange(28))
ax_main.set_yticklabels(ylabels, fontsize=10, family='monospace')
ax_main.tick_params(axis='y', length=0)
ax_main.tick_params(axis='x', length=0)
ax_main.grid(False)
# white grid lines for cell separation
for x in np.arange(-0.5, 16, 1):
    ax_main.axvline(x, color='white', linewidth=1)
for y in np.arange(-0.5, 28, 1):
    ax_main.axhline(y, color='white', linewidth=1)
# class boundary thicker lines
for label, start, end, color in class_boundaries:
    if start > 0:
        ax_main.axvline(start - 0.5, color='white', linewidth=2.5)

# (3) right marginal: per-dye response rate (Dye RR)
ax_right.barh(np.arange(28), dye_rate, color='#3F8E8C', edgecolor='white', height=0.85)
ax_right.set_xlim(0, 1.05); ax_right.set_ylim(-0.5, 27.5)
ax_right.invert_yaxis()
ax_right.set_yticks([]); ax_right.set_xticks([0, 0.5, 1])
ax_right.set_xticklabels(['0','0.5','1'], fontsize=11)
ax_right.set_xlabel('Dye RR', fontsize=12, labelpad=2)
ax_right.axvline(overall, color='#C94F4A', linestyle='--', linewidth=1.3, zorder=3)
ax_right.spines['top'].set_visible(False); ax_right.spines['right'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.grid(axis='x', color='#DDDDDD', ls='--', lw=0.5)

# (4) bottom marginal: per-agent response rate (Agent RR)
ax_bot.bar(np.arange(16), agent_rate, color=agent_colors, edgecolor='white', width=0.85)
ax_bot.set_xlim(-0.5, 15.5); ax_bot.set_ylim(0, 1.05)
ax_bot.set_xticks(np.arange(16))
ax_bot.set_xticklabels(agent_order, fontsize=10, rotation=0)
ax_bot.set_yticks([0, 0.5, 1])
ax_bot.set_yticklabels(['0','0.5','1'], fontsize=11)
ax_bot.set_ylabel('Agent RR', fontsize=12, labelpad=2)
ax_bot.axhline(overall, color='#C94F4A', linestyle='--', linewidth=1.3, zorder=3)
ax_bot.spines['top'].set_visible(False); ax_bot.spines['right'].set_visible(False)
ax_bot.grid(axis='y', color='#DDDDDD', ls='--', lw=0.5)

plt.savefig('rendered/Fig1.png', dpi=200, bbox_inches='tight')
print('Fig 1 v2 saved')
