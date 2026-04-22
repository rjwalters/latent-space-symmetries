"""Figure 3: KL divergence matrix for the 'room is on fire' case study.

Shows that reordering the same 3 facts produces substantially different
next-token distributions, with orderings sharing the same final fact
clustering together (recency effect).
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
})

# KL divergence matrix from the experiment (computed directly)
# Orderings: 012, 021, 102, 120, 201, 210
# Facts: 0=fire, 1=research, 2=wednesday
kl_matrix = np.array([
    [0.000, 0.211, 0.220, 0.352, 0.228, 0.272],
    [0.205, 0.000, 0.353, 0.363, 0.026, 0.225],
    [0.177, 0.277, 0.000, 0.189, 0.262, 0.199],
    [0.346, 0.359, 0.197, 0.000, 0.291, 0.060],
    [0.186, 0.025, 0.301, 0.274, 0.000, 0.150],
    [0.231, 0.218, 0.225, 0.063, 0.157, 0.000],
])

perm_labels = ['012', '021', '102', '120', '201', '210']

# Color-code by final fact
final_fact = [2, 1, 2, 0, 1, 0]  # 0=fire, 1=research, 2=wed
fact_names = ['Fire', 'Research', 'Wednesday']
fact_colors = ['#d73027', '#4575b4', '#fee090']

# Readable labels showing fact order
readable = [
    'Fire\nResearch\nWed',
    'Fire\nWed\nResearch',
    'Research\nFire\nWed',
    'Research\nWed\nFire',
    'Wed\nFire\nResearch',
    'Wed\nResearch\nFire',
]

short_labels = [
    'F-R-W', 'F-W-R', 'R-F-W', 'R-W-F', 'W-F-R', 'W-R-F'
]

fig, ax = plt.subplots(figsize=(4.5, 3.8))

im = ax.imshow(kl_matrix, cmap='YlOrRd', vmin=0, vmax=0.4, aspect='equal')

# Add text annotations
for i in range(6):
    for j in range(6):
        val = kl_matrix[i, j]
        color = 'white' if val > 0.25 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xticklabels(short_labels, fontsize=7, rotation=45, ha='right')
ax.set_yticklabels(short_labels, fontsize=7)

# Color-code tick labels by final fact
for i, tick in enumerate(ax.get_yticklabels()):
    tick.set_color(fact_colors[final_fact[i]])
    tick.set_fontweight('bold')
for i, tick in enumerate(ax.get_xticklabels()):
    tick.set_color(fact_colors[final_fact[i]])
    tick.set_fontweight('bold')

ax.set_xlabel('Ordering', fontsize=9)
ax.set_ylabel('Ordering', fontsize=9)

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('KL divergence', fontsize=8)

# Legend for fact colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=fact_colors[0], label='Ends with Fire'),
    Patch(facecolor=fact_colors[1], label='Ends with Research'),
    Patch(facecolor=fact_colors[2], label='Ends with Wednesday'),
]
ax.legend(handles=legend_elements, fontsize=6.5, loc='upper left',
          bbox_to_anchor=(0.0, -0.22), ncol=3, frameon=False)

plt.tight_layout()
fig.savefig('fig3_kl_heatmap.pdf')
fig.savefig('fig3_kl_heatmap.png')
print("Saved fig3_kl_heatmap.{pdf,png}")
