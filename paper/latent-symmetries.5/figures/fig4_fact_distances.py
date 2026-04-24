"""Figure 4: Fact reordering activation distances.

Within-set (same facts, different order) vs between-set (different facts)
activation distances across layers. The ratio shows ordering changes
representations 35-60% as much as entirely different content.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
})

data_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "results"

with open(data_dir / "fact_order_permutation.json") as f:
    fo = json.load(f)

layers = [0, 5, 10, 16, 21, 23]

# Extract S_3 data (more interesting than S_2)
within_3 = []
between_3 = []
ratio_3 = []
within_2 = []
between_2 = []
ratio_2 = []

for layer in layers:
    d3 = fo["s3_distances"][str(layer)]
    within_3.append(d3["within_set_mean_rel_dist"])
    between_3.append(d3["between_set_mean_rel_dist"])
    ratio_3.append(d3["within_over_between_ratio"])

    d2 = fo["s2_distances"][str(layer)]
    within_2.append(d2["within_set_mean_rel_dist"])
    between_2.append(d2["between_set_mean_rel_dist"])
    ratio_2.append(d2["within_over_between_ratio"])

x = np.arange(len(layers))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

# Panel A: Absolute distances (3-fact)
ax1.bar(x - width/2, within_3, width, label='Within-set\n(same facts, reordered)',
        color='#4575b4', alpha=0.85)
ax1.bar(x + width/2, between_3, width, label='Between-set\n(different facts)',
        color='#d73027', alpha=0.85)
ax1.set_xlabel('Layer')
ax1.set_ylabel('Relative activation distance')
ax1.set_title('(a) 3-fact sequences', fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels(layers)
ax1.legend(fontsize=6.5, loc='upper left')
ax1.set_ylim(0, 0.85)

# Panel B: Ratio across layers (both S_2 and S_3)
ax2.plot(x, ratio_2, 'o-', color='#2166ac', markersize=5, linewidth=1.5, label='2-fact ($S_2$)')
ax2.plot(x, ratio_3, 's-', color='#b2182b', markersize=5, linewidth=1.5, label='3-fact ($S_3$)')
ax2.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax2.axhline(y=1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax2.set_xlabel('Layer')
ax2.set_ylabel('Within / Between ratio')
ax2.set_title('(b) Ordering sensitivity ratio', fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(layers)
ax2.legend(fontsize=7)
ax2.set_ylim(0, 0.75)

# Add annotations
ax2.text(0.5, 0.68, 'Ratio = 0 $\\rightarrow$ order invariant', fontsize=6.5,
         color='green', fontstyle='italic', transform=ax2.transAxes, ha='center')
ax2.text(0.5, 0.05, 'Ratio = 1 $\\rightarrow$ order as different as content', fontsize=6.5,
         color='red', fontstyle='italic', transform=ax2.transAxes, ha='center')

plt.tight_layout()
fig.savefig('fig4_fact_distances.pdf')
fig.savefig('fig4_fact_distances.png')
print("Saved fig4_fact_distances.{pdf,png}")
