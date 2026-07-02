"""
Compare training losses across all PPCM configurations.
Run after training is complete.

Usage:
    python scripts/ppcm_new/compare_losses.py
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK_DIR = 'work_dirs/ppcm_new'
VIZ_DIR  = os.path.join(WORK_DIR, 'viz')
os.makedirs(VIZ_DIR, exist_ok=True)

CONFIGS = [
    ('s1=False_s2=False', 'Baseline (no PPCM)', 'blue',     'o'),
    ('s1=True_s2=False',  'Stage 1 only',       'green',    's'),
    ('s1=False_s2=True',  'Stage 2 only',       'orange',   '^'),
    ('s1=True_s2=True',   'Full PPCM (S1+S2)',  'red',      'D'),
]

print("=" * 60)
print("Training Loss Comparison")
print("=" * 60)

found = {}
for mode, label, color, marker in CONFIGS:
    path = os.path.join(WORK_DIR, f'{mode}_losses.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        found[label] = data['losses']
        print(f"\n{label}:")
        print(f"  Epoch 1:  {data['losses'][0]:.4f}")
        print(f"  Epoch 8:  {data['losses'][7]:.4f}  ← LR step")
        print(f"  Epoch 12: {data['losses'][-1]:.4f}")
    else:
        print(f"\n{label}: NOT FOUND ({path})")

if len(found) < 2:
    print("\nNeed at least 2 configs to compare. Exiting.")
    exit()

# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: full curve
ax = axes[0]
for (mode, label, color, marker), losses in zip(
        [c for c in CONFIGS if c[1] in found],
        [found[c[1]] for c in CONFIGS if c[1] in found]):
    ax.plot(losses, label=label, color=color,
            marker=marker, linewidth=2, markersize=6)

ax.axvline(x=7, color='gray', linestyle='--', alpha=0.5, label='LR step')
ax.set_xlabel('Epoch')
ax.set_ylabel('Avg Loss')
ax.set_title('Training Loss — Full Curve')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: zoom on last 4 epochs
ax2 = axes[1]
for (mode, label, color, marker), losses in zip(
        [c for c in CONFIGS if c[1] in found],
        [found[c[1]] for c in CONFIGS if c[1] in found]):
    ax2.plot(range(8, 12), losses[8:], label=label, color=color,
             marker=marker, linewidth=2, markersize=8)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Avg Loss')
ax2.set_title('Training Loss — Epoch 8-12 (zoomed)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle('PPCM Pipeline: Training Loss Comparison', fontsize=13)
plt.tight_layout()

save_path = os.path.join(VIZ_DIR, 'loss_comparison.png')
plt.savefig(save_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"\nSaved: {save_path}")

# ── Delta summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Final Loss Delta vs Baseline")
print("=" * 60)
baseline_loss = found.get('Baseline (no PPCM)', [None])[-1]
if baseline_loss:
    for label, losses in found.items():
        delta = losses[-1] - baseline_loss
        marker = ' ← best' if losses[-1] == min(
            l[-1] for l in found.values()) else ''
        print(f"  {label:<30} {losses[-1]:.4f}  ({delta:+.4f}){marker}")