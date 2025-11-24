#!/usr/bin/env python3
"""
Clean plotting script - shows only the most important configurations.
Much cleaner and more readable.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def extract_config_from_settings(settings_file):
    """Extract CPU and memory configuration from settings.txt file."""
    try:
        with open(settings_file, 'r') as f:
            content = f.read()
        
        cpu_match = re.search(r'--cpus=([\d.]+)', content)
        memory_match = re.search(r'--memory=([\d.]+)g', content)
        
        cpu = float(cpu_match.group(1)) if cpu_match else None
        memory = float(memory_match.group(1)) if memory_match else None
        
        return cpu, memory
    except Exception as e:
        return None, None

def main():
    base_dir = Path("data/visualizations")
    
    # Find all experiments
    subdirs = sorted([d.name for d in base_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('md_warm_nr')])
    
    experiments = []
    
    for subdir in subdirs:
        csv_file = base_dir / subdir / "crossover_summary.csv"
        settings_file = base_dir / subdir / "settings.txt"
        
        if not csv_file.exists():
            continue
            
        try:
            df = pd.read_csv(csv_file)
            cpu, memory = extract_config_from_settings(settings_file)
            
            experiments.append({
                'name': subdir,
                'data': df,
                'cpu': cpu,
                'memory': memory,
                'config_label': f"CPU: {cpu} cores, RAM: {memory}GB" if cpu and memory else subdir
            })
            
        except Exception as e:
            continue
    
    print(f"Loaded {len(experiments)} experiments")
    
    # SELECTION 1: Show only extreme configurations (lowest vs highest resources)
    low_resource = [exp for exp in experiments if exp['cpu'] == 0.5 and exp['memory'] == 1.0]
    high_resource = [exp for exp in experiments if exp['cpu'] == 4.0 and exp['memory'] == 2.0]
    
    if low_resource and high_resource:
        print("\n1. EXTREME CONFIGURATIONS (Lowest vs Highest Resources):")
        selected_experiments = [low_resource[0], high_resource[0]]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        colors = ['red', 'blue']
        
        for i, exp in enumerate(selected_experiments):
            df = exp['data']
            color = colors[i]
            
            # Plot Standard times
            ax1.plot(df['IoT Readings'], df['Standard Zeit (s)'], 
                    'o-', color=color, linewidth=3, markersize=8, 
                    label=f'Standard - {exp["config_label"]}')
            
            # Plot Nova times
            ax1.plot(df['IoT Readings'], df['Nova Zeit (s)'], 
                    's--', color=color, linewidth=3, markersize=8, 
                    label=f'Nova - {exp["config_label"]}')
            
            # Plot size comparison
            ax2.plot(df['IoT Readings'], df['Standard Größe (KB)'], 
                    'o-', color=color, linewidth=3, markersize=8, 
                    label=f'Standard - {exp["config_label"]}')
            
            ax2.plot(df['IoT Readings'], df['Nova Größe (KB)'], 
                    's--', color=color, linewidth=3, markersize=8, 
                    label=f'Nova - {exp["config_label"]}')
        
        # Configure plots
        ax1.set_xlabel('Number of IoT Readings', fontsize=14)
        ax1.set_ylabel('Proving Time (seconds)', fontsize=14)
        ax1.set_title('Proving Time: Extreme Configurations', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.legend(fontsize=12)
        
        ax2.set_xlabel('Number of IoT Readings', fontsize=14)
        ax2.set_ylabel('Proof Size (KB)', fontsize=14)
        ax2.set_title('Proof Size: Extreme Configurations', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        ax2.legend(fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        output_file = base_dir / "extreme_configurations.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
        
        plt.show()
    
    # SELECTION 2: Show only configurations with crossover points
    crossover_configs = []
    for exp in experiments:
        df = exp['data']
        if 'Nova' in df['Gewinner'].values:
            crossover_configs.append(exp)
    
    if crossover_configs:
        print(f"\n2. CONFIGURATIONS WITH CROSSOVER POINTS:")
        print(f"   Found {len(crossover_configs)} configs with Nova advantages")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(crossover_configs)))
        
        for i, exp in enumerate(crossover_configs):
            df = exp['data']
            color = colors[i]
            
            # Plot Standard times
            ax1.plot(df['IoT Readings'], df['Standard Zeit (s)'], 
                    'o-', color=color, linewidth=3, markersize=8, 
                    label=f'Standard - {exp["config_label"]}')
            
            # Plot Nova times
            ax1.plot(df['IoT Readings'], df['Nova Zeit (s)'], 
                    's--', color=color, linewidth=3, markersize=8, 
                    label=f'Nova - {exp["config_label"]}')
            
            # Plot size comparison
            ax2.plot(df['IoT Readings'], df['Standard Größe (KB)'], 
                    'o-', color=color, linewidth=3, markersize=8, 
                    label=f'Standard - {exp["config_label"]}')
            
            ax2.plot(df['IoT Readings'], df['Nova Größe (KB)'], 
                    's--', color=color, linewidth=3, markersize=8, 
                    label=f'Nova - {exp["config_label"]}')
        
        # Configure plots
        ax1.set_xlabel('Number of IoT Readings', fontsize=14)
        ax1.set_ylabel('Proving Time (seconds)', fontsize=14)
        ax1.set_title('Proving Time: Configurations with Crossover Points', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        ax2.set_xlabel('Number of IoT Readings', fontsize=14)
        ax2.set_ylabel('Proof Size (KB)', fontsize=14)
        ax2.set_title('Proof Size: Configurations with Crossover Points', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        output_file = base_dir / "crossover_configurations.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
        
        plt.show()
    
    print("\nClean plots created successfully!")

if __name__ == "__main__":
    main()
