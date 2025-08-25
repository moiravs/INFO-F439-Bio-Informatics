import matplotlib.pyplot as plt
import numpy as np

# Number of genes corresponding to each accuracy point
gene_counts = [300, 500, 700, 1000, 2000, 5000]

# Data
methods = ['Proposed w/ GAT', 'Proposed w/ GAT (ours)', 'Proposed w/ GCN', 'Proposed w/ GCN (ours)', 'FC-NN', 'FC-NN (ours)',
'GCN (Original)', 'GCN (Original) (ours)',  'GCN (Modified)',
           'GCN (Modified) (ours)', 'Multi-omics GCN (Original)','Multi-omics GCN (Original) (ours)', 'Multi-omics GCN (Modified)', 'Multi-omics GCN (Modified) (ours)',
           'Multi-omics GAT (Original)', 'Multi-omics GAT (Original) (ours)',  'Multi-omics GAT (Modified)', 'Multi-omics GAT (Modified) (ours)']

# Accuracy values (first value of each pair)
accuracy_data = [
    [83.8, 84.8, 86.4, 88.9, 82.8, 81.8],  # Proposed w/ GAT (odd)
    [70.7, 63.2, 59.8, 73.9, 54.4, 54.44],   # Proposed w/ GAT (even)
    [81.8, 81.8, 83.8, 84.8, 86.9, 90.1],   # Proposed w/ GCN (odd)
    [79.2, 77.2, 79.8, 83.0, 83.0, 83.1],   # Proposed w/ GCN (even)
    [72.8, 75.8, 78.8, 81.8, 81.8, 82.8],   # FC-NN (odd)
    [66.7, 67.1, 54.5, 54.5, 54.5, 54.5],   # FC-NN (even)
    [77.8, 79.8, 82.8, 83.8, 84.8, 87.9],   # GCN (Original) (odd)
    [75.4, 82.4, 77.4, 75.6, 77.0, 86.1],   # GCN (Original) (even)
    [74.2, 77.4, 81.8, 81.8, 78.3, 75.6],   # GCN (Modified) (odd)
    [78.2, 78.8, 83.6, 80.4, 79.0, 84.8],   # GCN (Modified) (even)
    [77.8, 79.8, 81.8, 82.8, 84.8, 86.9],   # Multi-omics GCN (Original) (odd)
    [79.8, 79.0, 80.8, 81.6, 79.0, 78.0],   # Multi-omics GCN (Original) (even)
    [81.8, 82.8, 82.8, 83.8, 84.8, 85.9],   # Multi-omics GCN (Modified) (odd)
    [78.4, 81.2, 82.0, 82.4, 84.0, 61.2],   # Multi-omics GCN (Modified) (even)
    [76.8, 81.8, 81.8, 83.8, 80.8, 78.8],   # Multi-omics GAT (Original) (odd)
    [67.5, 68.3, 66.1, 67.9, 60.4, 54.4],   # Multi-omics GAT (Original) (even)
    [77.8, 81.8, 82.8, 86.9, 79.8, 77.8],   # Multi-omics GAT (Modified) (odd)
    [76.0, 78.6, 69.5, 63.2, 70.7, 54.4]    # Multi-omics GAT (Modified) (even)
]

# Create figure and subplots
plt.figure(figsize=(14, 10))

# Generate unique colors for each line
colors = plt.cm.tab20(np.linspace(0, 1, len(accuracy_data)))

# Plot accuracy data with unique colors
for i, data in enumerate(accuracy_data):
    if i % 2 == 0:  # Odd rows (dotted line)
        plt.plot(gene_counts, data, 'o--', color=colors[i], alpha=0.7, linewidth=2, markersize=6, label=methods[i])
    else:  # Even rows (solid line)
        plt.plot(gene_counts, data, 'o-', color=colors[i], alpha=0.7, linewidth=2, markersize=6, label=methods[i])

# Customize accuracy plot
plt.title('Accuracy Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=14)
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.xticks(gene_counts)
plt.xlabel('Number of genes', fontsize=14, color='red')

# Create a unified legend
plt.legend(
    methods, loc='upper right', bbox_to_anchor=(1.15, 0.5), ncol=1, fontsize=10,
    handlelength=5,      # Length of the legend handles
    handletextpad=0.5,   # Space between handle and text
    borderpad=1,         # Padding inside legend border
    labelspacing=0.5     # Vertical space between entries
)

plt.tight_layout()
plt.show()
plt.savefig('try.jpg')




