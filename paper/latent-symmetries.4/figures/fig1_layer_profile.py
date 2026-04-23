"""Figure 1: Layer-by-layer operator fit quality and group relation errors.

Shows the U-shaped test error profile and the flat ~1.0 relation errors
for both Qwen2.5-0.5B and 1.5B. The key visual: good fit (low test error)
does NOT imply good group structure (relation errors stay ~1.0).
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

# Load 0.5B alignment data
with open(data_dir / "activation_alignment.json") as f:
    align_05 = json.load(f)["S_3"]

# Load 1.5B alignment data
with open(data_dir / "activation_alignment_1.5B.json") as f:
    align_15 = json.load(f)["S_3"]

# Load 0.5B group relations
with open(data_dir / "group_relations.json") as f:
    gr_05 = json.load(f)["group_relations"]

# Load 1.5B group relations
with open(data_dir / "group_relations_1.5B.json") as f:
    gr_15 = json.load(f)["group_relations"]

# Extract 0.5B data
layers_05 = sorted([int(k) for k in align_05["layers"].keys()])
test_err_05 = [align_05["layers"][str(l)]["mean_test_error"] for l in layers_05]

gr_layers_05 = sorted([int(k) for k in gr_05.keys()])
s2e_05 = [gr_05[str(l)]["s3_relations"]["s^2 = e"]["relative"] for l in gr_layers_05]
r3e_05 = [gr_05[str(l)]["s3_relations"]["r^3 = e"]["relative"] for l in gr_layers_05]

# Extract 1.5B data
layers_15 = sorted([int(k) for k in align_15["layers"].keys()])
test_err_15 = [align_15["layers"][str(l)]["mean_test_error"] for l in layers_15]

gr_layers_15 = sorted([int(k) for k in gr_15.keys()])
s2e_15 = [gr_15[str(l)]["s3_relations"]["s^2 = e"]["relative"] for l in gr_layers_15]
r3e_15 = [gr_15[str(l)]["s3_relations"]["r^3 = e"]["relative"] for l in gr_layers_15]

# Normalize layers to [0, 1] for comparison
norm_layers_05 = [l / max(layers_05) for l in layers_05]
norm_layers_15 = [l / max(layers_15) for l in layers_15]
norm_gr_05 = [l / max(gr_layers_05) for l in gr_layers_05]
norm_gr_15 = [l / max(gr_layers_15) for l in gr_layers_15]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8), sharey=False)

# Panel A: Test error (operator fit quality)
c05 = '#2166ac'
c15 = '#b2182b'
ax1.plot(norm_layers_05, test_err_05, 'o-', color=c05, markersize=3, linewidth=1.2, label='0.5B (24 layers)')
ax1.plot(norm_layers_15, test_err_15, 's-', color=c15, markersize=3, linewidth=1.2, label='1.5B (28 layers)')
ax1.set_xlabel('Relative layer depth')
ax1.set_ylabel('Test-set relative error')
ax1.set_title('(a) Operator fit quality', fontsize=10)
ax1.legend(fontsize=7, loc='upper left')
ax1.set_ylim(0, 0.6)
ax1.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

# Panel B: Group relation errors
ax2.plot(norm_gr_05, s2e_05, 'o-', color=c05, markersize=3, linewidth=1.2, label='0.5B $s^2\\!=\\!e$')
ax2.plot(norm_gr_05, r3e_05, 'o--', color=c05, markersize=3, linewidth=1.2, alpha=0.6, label='0.5B $r^3\\!=\\!e$')
ax2.plot(norm_gr_15, s2e_15, 's-', color=c15, markersize=3, linewidth=1.2, label='1.5B $s^2\\!=\\!e$')
ax2.plot(norm_gr_15, r3e_15, 's--', color=c15, markersize=3, linewidth=1.2, alpha=0.6, label='1.5B $r^3\\!=\\!e$')
ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, label='No structure')
ax2.axhspan(0, 0.1, alpha=0.08, color='green')
ax2.text(0.5, 0.05, 'Group structure\ndetected', fontsize=6, color='green',
         fontstyle='italic', alpha=0.7, ha='center', va='center')
ax2.set_xlabel('Relative layer depth')
ax2.set_ylabel('Relative error')
ax2.set_title('(b) Group relation errors ($S_3$)', fontsize=10)
ax2.legend(fontsize=6.5, loc='upper right', ncol=1)
ax2.set_ylim(0, 1.2)

plt.tight_layout()
fig.savefig('fig1_layer_profile.pdf')
fig.savefig('fig1_layer_profile.png')
print("Saved fig1_layer_profile.{pdf,png}")
