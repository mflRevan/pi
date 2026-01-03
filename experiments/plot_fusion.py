#!/usr/bin/env python3
"""
Visualize fusion comparison results.
"""

import matplotlib.pyplot as plt
import numpy as np

# Results from the experiment
distances = [5, 10, 20, 30, 50]
additive_acc = [93.8, 96.2, 95.0, 96.2, 95.3]
multiplicative_acc = [78.8, 79.1, 74.7, 75.9, 71.2]

# Gradient magnitudes
components = ['Embed', 'Attention', 'Resonant', 'Output']
additive_grads = [0.101301, 0.141490, 0.525088, 0.801866]
mult_grads = [0.031672, 0.116896, 0.066742, 0.464092]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Accuracy by distance
ax = axes[0]
x_pos = np.arange(len(distances))
width = 0.35

bars1 = ax.bar(x_pos - width/2, additive_acc, width, label='Additive', color='#2ecc71', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, multiplicative_acc, width, label='Multiplicative', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Distance (tokens)', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_title('Needle Task Performance by Distance', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(distances)
ax.set_ylim(60, 100)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

# 2. Gradient magnitudes
ax = axes[1]
x_pos = np.arange(len(components))

bars1 = ax.bar(x_pos - width/2, additive_grads, width, label='Additive', color='#2ecc71', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, mult_grads, width, label='Multiplicative', color='#e74c3c', alpha=0.8)

ax.set_ylabel('Gradient Magnitude (mean)', fontsize=11, fontweight='bold')
ax.set_title('Gradient Flow by Component', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(components)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_yscale('log')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8, rotation=0)

# 3. Performance degradation vs distance
ax = axes[2]
dist_range = np.array(distances)
add_line = ax.plot(dist_range, additive_acc, 'o-', linewidth=2.5, markersize=8, 
                   label='Additive', color='#2ecc71')
mult_line = ax.plot(dist_range, multiplicative_acc, 's-', linewidth=2.5, markersize=8,
                    label='Multiplicative', color='#e74c3c')

ax.set_xlabel('Distance (tokens)', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_title('Degradation with Distance', fontsize=12, fontweight='bold')
ax.set_ylim(65, 100)
ax.legend(fontsize=10, loc='lower left')
ax.grid(True, alpha=0.3)

# Shade the long-range region (dist >= 30)
ax.axvspan(30, 55, alpha=0.1, color='gray')
ax.text(40, 68, 'Long-range', fontsize=9, style='italic', color='gray')

plt.tight_layout()
plt.savefig('/home/aiman/pi/results/fusion_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: /home/aiman/pi/results/fusion_comparison.png")

# Statistics
print("\n" + "="*70)
print("FUSION COMPARISON STATISTICS")
print("="*70)

add_avg = np.mean(additive_acc)
mult_avg = np.mean(multiplicative_acc)
add_std = np.std(additive_acc)
mult_std = np.std(multiplicative_acc)

print(f"\nAccuracy:")
print(f"  Additive:       {add_avg:.1f}% ± {add_std:.1f}%")
print(f"  Multiplicative: {mult_avg:.1f}% ± {mult_std:.1f}%")
print(f"  Difference:     {(add_avg - mult_avg):.1f}% (absolute)")

add_short = np.mean(additive_acc[:2])
add_long = np.mean(additive_acc[-2:])
mult_short = np.mean(multiplicative_acc[:2])
mult_long = np.mean(multiplicative_acc[-2:])

print(f"\nShort-range (dist 5-10):")
print(f"  Additive:       {add_short:.1f}%")
print(f"  Multiplicative: {mult_short:.1f}%")

print(f"\nLong-range (dist 30-50):")
print(f"  Additive:       {add_long:.1f}%")
print(f"  Multiplicative: {mult_long:.1f}%")

add_degrad = add_short - add_long
mult_degrad = mult_short - mult_long

print(f"\nDegradation (short to long):")
print(f"  Additive:       {add_degrad:.1f}% drop")
print(f"  Multiplicative: {mult_degrad:.1f}% drop")

print(f"\nGradient Analysis (Resonant layer):")
res_add = 0.525088
res_mult = 0.066742
ratio = res_add / res_mult
print(f"  Additive resonant gradient:       {res_add:.6f}")
print(f"  Multiplicative resonant gradient: {res_mult:.6f}")
print(f"  Ratio (Additive / Multiplicative): {ratio:.1f}x")
print(f"\n  → Additive preserves {ratio:.1f}× stronger resonant learning signal")

print("\n" + "="*70)
print("CONCLUSION: Additive fusion enables superior long-range performance")
print("="*70)
