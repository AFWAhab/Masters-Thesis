import matplotlib.pyplot as plt
import numpy as np

# Example data: 8 coefficients in range [-0.2, 0.3]
coefs_embryo = np.array([0.00719559082627874,
                         0.16873053840976135,
                         -0.16905599392966844,
                         0.13469012759563734,
                         0.08707040697560196,
                         0.001525409920900947,
                         -0.1977495655347899,
                         0.23628167320376628])
coefs_median = np.array([
-0.0060021832748551485,
0.23468055446822753,
-0.19091254249871817,
0.06559731633470728,
0.03972576202696909,
-0.06513697605916206,
-0.21777050978240398,
0.28907327450600084
])
coefs_human = np.array([
    0.05,
    0.09,
    0.11,
    0.23,
    -0.01,
    -0.03,
    -0.06,
    0.25
])

labels = [
    "5’ UTR length",
    "3’ UTR length",
    "ORF length",
    "5’ UTR GC content",
    "3’ UTR GC content",
    "ORF GC content",
    "Intron length",
    "ORF exon density"
]

# Y positions (top-down)
y_pos = np.arange(len(coefs_embryo))[::-1]
y_median = np.arange(len(coefs_median))[::-1]
y_human = np.arange(len(coefs_human))[::-1]

# Plot
fig, ax = plt.subplots(figsize=(6, 4))

# Horizontal error bars (points with CI)
ax.errorbar(coefs_embryo, y_pos, fmt='o', color='purple', capsize=3)
ax.errorbar(coefs_median, y_pos, fmt='o', color='red', capsize=3)
ax.errorbar(coefs_human, y_pos, fmt='o', color='blue', capsize=3)

# Reference line at x=0
ax.axvline(0, linestyle='dotted', color='black')

# Hide y-axis
ax.set_yticks([])
ax.spines['left'].set_visible(False)

# Set manual label positions and axis limits
label_x = -0.3  # farther to the left to avoid overlap
for i, label in enumerate(labels):
    ax.text(label_x, y_pos[i], label, ha='right', va='center', fontsize=10)

# Custom legend on the right
legend_x = 0.33  # adjust as needed
legend_y = np.mean(y_pos)  # middle of the y-range

ax.plot(legend_x, legend_y, 'o', color='purple', markersize=6)
ax.text(legend_x + 0.03, legend_y, "zebrafish embryo", color='black', ha='left', va='center', fontsize=10, fontweight='bold')
ax.plot(legend_x, legend_y+0.3, 'o', color='red', markersize=6)
ax.text(legend_x + 0.03, legend_y+0.3, "zebrafish median", color='black', ha='left', va='center', fontsize=10, fontweight='bold')
ax.plot(legend_x, legend_y+0.6, 'o', color='blue', markersize=6)
ax.text(legend_x + 0.03, legend_y+0.6, "human median", color='black', ha='left', va='center', fontsize=10, fontweight='bold')

# Make sure axis includes the left-side labels and negative values
ax.set_xlim(label_x - 0.05, max(coefs_median) + 0.05)

# X-axis label and limits
ax.set_xlabel("Spearman correlation between gene sequence features and\nmedian zebrafish mRNA expression values")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()