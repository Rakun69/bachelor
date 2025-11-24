#!/usr/bin/env python3
"""
Simple script to plot all crossover results from md_warm_nr* directories.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    base_dir = "data/visualizations"
    
    # Find all md_warm_nr* directories
    base_path = Path(base_dir)
    subdirs = sorted([d.name for d in base_path.iterdir() 
                      if d.is_dir() and d.name.startswith('md_warm_nr')])
    
    print(f"Found {len(subdirs)} directories: {subdirs}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors for different configurations
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, subdir in enumerate(subdirs):
        csv_file = base_path / subdir / "crossover_summary.csv"
        
        if not csv_file.exists():
            print(f"Warning: {csv_file} does not exist")
            continue
            
        try:
            df = pd.read_csv(csv_file)
            color = colors[i % len(colors)]
            
            print(f"Processing {subdir}: {len(df)} data points")
            print(f"  Readings: {df['IoT Readings'].min()} - {df['IoT Readings'].max()}")
            
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
    
    # Configure plots
    ax1.set_xlabel('Number of IoT Readings')
    ax1.set_ylabel('Proving Time (seconds)')
    ax1.set_title('Proving Time: Standard vs Nova')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.set_xlabel('Number of IoT Readings')
    ax2.set_ylabel('Proof Size (KB)')
    ax2.set_title('Proof Size: Standard vs Nova')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    output_file = base_path / "all_results_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
