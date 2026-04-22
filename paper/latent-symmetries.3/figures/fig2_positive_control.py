"""Figure 2: Positive control calibration curve.

Shows group relation errors vs noise level for synthetic data with exact
S_3 structure, with the real model's position marked. The key visual:
the real model has low test error (good fit) but high relation error
(no structure) — a combination impossible under true group structure.
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

with open(data_dir / "positive_control.json") as f:
    pc = json.load(f)

# Extract clean_d10 results
clean = pc["clean_d10"]
noise_levels = []
test_errs = []
s2e_errs = []
r3e_errs = []
comp_errs = []

for noise_str, r in sorted(clean.items(), key=lambda x: float(x[0])):
    noise = float(noise_str)
    noise_levels.append(noise)
    test_vals = [v["test_error"] for v in r["fit_quality"].values() if v["test_error"] is not None]
    test_errs.append(np.mean(test_vals))
    s2e_errs.append(r["s3_relations"]["s^2 = e"]["relative"])
    r3e_errs.append(r["s3_relations"]["r^3 = e"]["relative"])
    comp_errs.append(r["mean_composition_error"])

fig, ax = plt.subplots(figsize=(4.5, 3.2))

# Plot calibration curves
ax.plot(test_errs, s2e_errs, 'o-', color='#2166ac', markersize=5, linewidth=1.5, label='$s^2 = e$')
ax.plot(test_errs, r3e_errs, 's-', color='#b2182b', markersize=5, linewidth=1.5, label='$r^3 = e$')
ax.plot(test_errs, comp_errs, '^-', color='#4daf4a', markersize=5, linewidth=1.5, label='Mean composition')

# Add noise labels
for i, noise in enumerate(noise_levels):
    if noise in [0.0, 0.1, 0.5, 1.0]:
        ax.annotate(f'$\\sigma$={noise}', (test_errs[i], comp_errs[i]),
                    textcoords="offset points", xytext=(8, -3), fontsize=7, color='gray')

# Mark real model positions
real_05_test = 0.0775
real_05_s2 = 1.023
real_05_r3 = 1.094
real_05_comp = 1.079

ax.plot(real_05_test, real_05_s2, '*', color='#2166ac', markersize=14, markeredgecolor='black',
        markeredgewidth=0.8, zorder=10)
ax.plot(real_05_test, real_05_r3, '*', color='#b2182b', markersize=14, markeredgecolor='black',
        markeredgewidth=0.8, zorder=10)
ax.plot(real_05_test, real_05_comp, '*', color='#4daf4a', markersize=14, markeredgecolor='black',
        markeredgewidth=0.8, zorder=10)

ax.annotate('Real model\n(layer 0)', (real_05_test, real_05_comp),
            textcoords="offset points", xytext=(12, -15), fontsize=8,
            fontweight='bold', color='black',
            arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

# Styling
ax.set_xlabel('Operator test error')
ax.set_ylabel('Group relation error')
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(-0.02, 0.85)
ax.set_ylim(-0.05, 1.2)

# Add region labels
ax.axhspan(-0.05, 0.15, alpha=0.08, color='green')
ax.axhspan(0.85, 1.2, alpha=0.08, color='red')
ax.text(0.60, 0.05, 'Group structure', fontsize=7, color='green', fontstyle='italic', alpha=0.8)
ax.text(0.55, 1.10, 'No structure', fontsize=7, color='red', fontstyle='italic', alpha=0.8)

plt.tight_layout()
fig.savefig('fig2_positive_control.pdf')
fig.savefig('fig2_positive_control.png')
print("Saved fig2_positive_control.{pdf,png}")
