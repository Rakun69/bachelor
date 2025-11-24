#!/usr/bin/env python3
"""
Quick plotting script - creates a simple comprehensive plot of all results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up the data directory
base_dir = Path("data/visualizations")

# Find all md_warm_nr* directories
subdirs = sorted([d.name for d in base_dir.iterdir() 
                  if d.is_dir() and d.name.startswith('md_warm_nr')])

print(f"Found {len(subdirs)} directories: {subdirs}")

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Colors for different experiments
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

for i, subdir in enumerate(subdirs):
    csv_file = base_dir / subdir / "crossover_summary.csv"
    
    if not csv_file.exists():
        print(f"Warning: {csv_file} does not exist")
        continue
    
    try:
        df = pd.read_csv(csv_file)
        color = colors[i % len(colors)]
        
        print(f"Processing {subdir}: {len(df)} data points")
        print(f"  Readings: {df['IoT Readings'].min()} - {df['IoT Readings'].max()}")
        print(f"  Standard: {df['Standard Zeit (s)'].min():.2f} - {df['Standard Zeit (s)'].max():.2f}s")
        print(f"  Nova: {df['Nova Zeit (s)'].min():.2f} - {df['Nova Zeit (s)'].max():.2f}s")
        
        # Plot time comparison
        ax1.plot(df['IoT Readings'], df['Standard Zeit (s)'], 
                'o-', color=color, linewidth=2, markersize=6, 
                label=f'Standard - {subdir}')
        
        ax1.plot(df['IoT Readings'], df['Nova Zeit (s)'], 
                's--', color=color, linewidth=2, markersize=6, 
                label=f'Nova - {subdir}')
        
        # Plot size comparison  
        ax2.plot(df['IoT Readings'], df['Standard Größe (KB)'], 
                'o-', color=color, linewidth=2, markersize=6, 
                label=f'Standard - {subdir}')
        
        ax2.plot(df['IoT Readings'], df['Nova Größe (KB)'], 
                's--', color=color, linewidth=2, markersize=6, 
                label=f'Nova - {subdir}')
                
    except Exception as e:
        print(f"Error processing {subdir}: {e}")
        continue

# Configure time plot
ax1.set_xlabel('Number of IoT Readings', fontsize=12)
ax1.set_ylabel('Proving Time (seconds)', fontsize=12)
ax1.set_title('Proving Time Comparison: Standard vs Nova', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

# Configure size plot
ax2.set_xlabel('Number of IoT Readings', fontsize=12)
ax2.set_ylabel('Proof Size (KB)', fontsize=12)
ax2.set_title('Proof Size Comparison: Standard vs Nova', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

plt.tight_layout()

# Save the plot
output_file = base_dir / "all_results_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

# Show the plot
plt.show()

print("\nPlot created successfully!")
