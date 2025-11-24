#!/usr/bin/env python3
"""
Comprehensive plotting script for all crossover results.
Extracts configuration from settings.txt and creates unified plots.
"""

import os
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
        
        # Extract CPU and memory from docker run command
        cpu_match = re.search(r'--cpus=([\d.]+)', content)
        memory_match = re.search(r'--memory=([\d.]+)g', content)
        
        cpu = float(cpu_match.group(1)) if cpu_match else None
        memory = float(memory_match.group(1)) if memory_match else None
        
        return cpu, memory
    except Exception as e:
        print(f"Warning: Could not parse settings from {settings_file}: {e}")
        return None, None

def load_experiment_data(base_dir):
    """Load all experiment data from md_warm_nr* directories."""
    
    base_path = Path(base_dir)
    subdirs = sorted([d.name for d in base_path.iterdir() 
                      if d.is_dir() and d.name.startswith('md_warm_nr')])
    
    if not subdirs:
        print(f"No md_warm_nr* directories found in {base_dir}")
        return []
    
    print(f"Found {len(subdirs)} experiment directories: {subdirs}")
    
    experiments = []
    
    for subdir in subdirs:
        dir_path = base_path / subdir
        csv_file = dir_path / "crossover_summary.csv"
        settings_file = dir_path / "settings.txt"
        
        if not csv_file.exists():
            print(f"Warning: {csv_file} does not exist")
            continue
            
        try:
            # Load CSV data
            df = pd.read_csv(csv_file)
            
            # Extract configuration
            cpu, memory = extract_config_from_settings(settings_file)
            
            experiments.append({
                'name': subdir,
                'data': df,
                'cpu': cpu,
                'memory': memory,
                'config_label': f"CPU: {cpu} cores, RAM: {memory}GB" if cpu and memory else subdir
            })
            
            print(f"Loaded {subdir}: {len(df)} data points, {cpu} cores, {memory}GB RAM")
            print(f"  Readings: {df['IoT Readings'].min()} - {df['IoT Readings'].max()}")
            print(f"  Standard time: {df['Standard Zeit (s)'].min():.2f} - {df['Standard Zeit (s)'].max():.2f}s")
            print(f"  Nova time: {df['Nova Zeit (s)'].min():.2f} - {df['Nova Zeit (s)'].max():.2f}s")
            
        except Exception as e:
            print(f"Error loading {subdir}: {e}")
            continue
    
    return experiments

def create_time_plot(experiments, output_dir):
    """Create comprehensive time comparison plot."""
    
    plt.figure(figsize=(14, 10))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for i, exp in enumerate(experiments):
        df = exp['data']
        color = colors[i]
        
        # Plot Standard times
        plt.plot(df['IoT Readings'], df['Standard Zeit (s)'], 
                'o-', color=color, linewidth=3, markersize=8, 
                label=f'Standard - {exp["config_label"]}')
        
        # Plot Nova times
        plt.plot(df['IoT Readings'], df['Nova Zeit (s)'], 
                's--', color=color, linewidth=3, markersize=8, 
                label=f'Nova - {exp["config_label"]}')
    
    plt.xlabel('Number of IoT Readings', fontsize=14)
    plt.ylabel('Proving Time (seconds)', fontsize=14)
    plt.title('Comprehensive Proving Time Comparison: Standard vs Nova\nAcross Different Resource Configurations', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_dir) / "comprehensive_time_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Time plot saved to: {output_file}")
    
    plt.show()

def create_size_plot(experiments, output_dir):
    """Create comprehensive size comparison plot."""
    
    plt.figure(figsize=(14, 10))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for i, exp in enumerate(experiments):
        df = exp['data']
        color = colors[i]
        
        # Plot Standard sizes
        plt.plot(df['IoT Readings'], df['Standard Größe (KB)'], 
                'o-', color=color, linewidth=3, markersize=8, 
                label=f'Standard - {exp["config_label"]}')
        
        # Plot Nova sizes
        plt.plot(df['IoT Readings'], df['Nova Größe (KB)'], 
                's--', color=color, linewidth=3, markersize=8, 
                label=f'Nova - {exp["config_label"]}')
    
    plt.xlabel('Number of IoT Readings', fontsize=14)
    plt.ylabel('Proof Size (KB)', fontsize=14)
    plt.title('Comprehensive Proof Size Comparison: Standard vs Nova\nAcross Different Resource Configurations', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_dir) / "comprehensive_size_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Size plot saved to: {output_file}")
    
    plt.show()

def create_summary_table(experiments, output_dir):
    """Create summary table of all experiments."""
    
    summary_data = []
    
    for exp in experiments:
        df = exp['data']
        
        # Find crossover point (where Nova becomes faster)
        crossover_point = None
        for idx, row in df.iterrows():
            if row['Gewinner'] == 'Nova':
                crossover_point = row['IoT Readings']
                break
        
        # Get max readings tested
        max_readings = df['IoT Readings'].max()
        
        # Get performance at max readings
        max_row = df[df['IoT Readings'] == max_readings].iloc[0]
        
        summary_data.append({
            'Configuration': exp['config_label'],
            'Max Readings': max_readings,
            'Crossover Point': crossover_point if crossover_point else 'Not reached',
            'Standard Time (max)': f"{max_row['Standard Zeit (s)']:.2f}s",
            'Nova Time (max)': f"{max_row['Nova Zeit (s)']:.2f}s",
            'Standard Size (max)': f"{max_row['Standard Größe (KB)']:.1f}KB",
            'Nova Size (max)': f"{max_row['Nova Größe (KB)']:.1f}KB",
            'Time Advantage': f"{max_row['Zeit Vorteil']:.2f}x",
            'Size Advantage': f"{max_row['Größe Vorteil']:.2f}x"
        })
    
    # Create and save summary table
    summary_df = pd.DataFrame(summary_data)
    summary_file = Path(output_dir) / "experiment_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary table saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

def main():
    base_dir = "data/visualizations"
    
    print("Loading experiment data...")
    experiments = load_experiment_data(base_dir)
    
    if not experiments:
        print("No experiment data found!")
        return
    
    print(f"\nLoaded {len(experiments)} experiments successfully!")
    
    # Create plots
    print("\nCreating time comparison plot...")
    create_time_plot(experiments, base_dir)
    
    print("\nCreating size comparison plot...")
    create_size_plot(experiments, base_dir)
    
    print("\nCreating summary table...")
    create_summary_table(experiments, base_dir)
    
    print("\nAll plots and summaries created successfully!")

if __name__ == "__main__":
    main()
