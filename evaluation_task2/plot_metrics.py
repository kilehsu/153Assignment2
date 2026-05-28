import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load data
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, 'harmony_metrics.json'), 'r') as f:
    data = json.load(f)

# Extract values
models = ['Our Model', 'Random', 'Real Bach']
colors = ['#4C72B0', '#DD8452', '#55A868']

scale_consistency = [
    data['our_model']['scale_consistency'],
    data['random_baseline']['scale_consistency'],
    data['real_bach']['scale_consistency']
]

voice_crossing = [
    data['our_model']['voice_crossing_rate'],
    data['random_baseline']['voice_crossing_rate'],
    data['real_bach']['voice_crossing_rate']
]

pitch_kl = [
    data['our_model']['pitch_kl_divergence'],
    data['random_baseline']['pitch_kl_divergence'],
    data['real_bach']['pitch_kl_divergence']
]

# Create figure
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.dpi = 120

# Left panel: Grouped bar chart
x = np.arange(2)  # Two metrics
width = 0.25  # Width of bars

bars1 = ax1.bar(x - width, [scale_consistency[0], voice_crossing[0]], width, label='Our Model', color=colors[0])
bars2 = ax1.bar(x, [scale_consistency[1], voice_crossing[1]], width, label='Random', color=colors[1])
bars3 = ax1.bar(x + width, [scale_consistency[2], voice_crossing[2]], width, label='Real Bach', color=colors[2])

ax1.set_ylabel('Percent (%)', fontsize=11)
ax1.set_ylim(0, 100)
ax1.set_xticks(x)
ax1.set_xticklabels(['Scale\nConsistency ↑ (%)', 'Voice Crossing\nRate ↓ (%)'])
ax1.legend(loc='upper right', fontsize=10)
ax1.set_title('Scale Consistency & Voice Crossing', fontsize=12, fontweight='bold')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

# Right panel: Simple bar chart for pitch KL divergence
x_pos = np.arange(len(models))
bars = ax2.bar(x_pos, pitch_kl, color=colors)

ax2.set_ylabel('KL Divergence', fontsize=11)
ax2.set_ylim(0, 3.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models)
ax2.set_title('Pitch Class KL Divergence ↓\n(vs. Real Bach distribution)', fontsize=12, fontweight='bold')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=9)

# Add note at bottom
note_text = '* Random has low KL by chance\n(uniform sampling ≈ uniform reference)'
ax2.text(0.5, -0.35, note_text, transform=ax2.transAxes,
        ha='center', va='top', fontsize=8, style='italic', color='gray')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(script_dir), 'images', 'task2_evaluation.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=120, bbox_inches='tight')
print(f"Figure saved successfully to {output_path}")
plt.close()
