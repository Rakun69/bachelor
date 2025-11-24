#!/usr/bin/env python3
"""
Plot all crossover results from md_warm_nr* directories in a single comprehensive plot.
Shows proving times for both Standard and Nova approaches across different configurations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
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

def load_results_from_directory(base_dir, subdir):
    """Load results from a specific md_warm_nr* directory."""
    dir_path = Path(base_dir) / subdir
    
    if not dir_path.exists():
        print(f"Warning: Directory {dir_path} does not exist")
        return None, None, None
    
    # Load CSV data
    csv_file = dir_path / "crossover_summary.csv"
    if not csv_file.exists():
        print(f"Warning: CSV file {csv_file} does not exist")
        return None, None, None
    
    try:
        df = pd.read_csv(csv_file)
        
        # Load settings
        settings_file = dir_path / "settings.txt"
        cpu, memory = extract_config_from_settings(settings_file)
        
        return df, cpu, memory
    except Exception as e:
        print(f"Error loading data from {dir_path}: {e}")
        return None, None, None

def create_comprehensive_plot(base_dir="data/visualizations"):
    """Create a comprehensive plot showing all results."""
    
    # Find all md_warm_nr* directories
    base_path = Path(base_dir)
    subdirs = sorted([d.name for d in base_path.iterdir() 
                      if d.is_dir() and d.name.startswith('md_warm_nr')])
    
    if not subdirs:
        print(f"No md_warm_nr* directories found in {base_dir}")
        return
    
    print(f"Found {len(subdirs)} result directories: {subdirs}")
    
    # Set up the plot
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Color palette for different configurations
    colors = plt.cm.Set3(np.linspace(0, 1, len(subdirs)))
    
    # Store all data for legend
    legend_elements = []
    
    for i, subdir in enumerate(subdirs):
        df, cpu, memory = load_results_from_directory(base_dir, subdir)
        
        if df is None:
            continue
        
        color = colors[i]
        
        # Configuration label
        config_label = f"CPU: {cpu} cores, RAM: {memory}GB" if cpu and memory else subdir
        
        print(f"Processing {subdir}: {len(df)} data points")
        print(f"  Readings range: {df['IoT Readings'].min()} - {df['IoT Readings'].max()}")
        print(f"  Standard time range: {df['Standard Zeit (s)'].min():.2f} - {df['Standard Zeit (s)'].max():.2f}s")
        print(f"  Nova time range: {df['Nova Zeit (s)'].min():.2f} - {df['Nova Zeit (s)'].max():.2f}s")
        
        # Plot Standard times
        ax1.plot(df['IoT Readings'], df['Standard Zeit (s)'], 
                'o-', color=color, linewidth=2, markersize=6, 
                label=f'Standard - {config_label}')
        
        # Plot Nova times
        ax1.plot(df['IoT Readings'], df['Nova Zeit (s)'], 
                's--', color=color, linewidth=2, markersize=6, 
                label=f'Nova - {config_label}')
        
        # Plot size comparison
        ax2.plot(df['IoT Readings'], df['Standard Größe (KB)'], 
                'o-', color=color, linewidth=2, markersize=6, 
                label=f'Standard - {config_label}')
        
        ax2.plot(df['IoT Readings'], df['Nova Größe (KB)'], 
                's--', color=color, linewidth=2, markersize=6, 
                label=f'Nova - {config_label}')
        
        # Add to legend elements
        legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=3, label=config_label))
    
    # Configure time plot
    ax1.set_xlabel('Number of IoT Readings', fontsize=12)
    ax1.set_ylabel('Proving Time (seconds)', fontsize=12)
    ax1.set_title('Proving Time Comparison: Standard vs Nova', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Configure size plot
    ax2.set_xlabel('Number of IoT Readings', fontsize=12)
    ax2.set_ylabel('Proof Size (KB)', fontsize=12)
    ax2.set_title('Proof Size Comparison: Standard vs Nova', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Create separate legends for configurations and methods
    config_legend = ax1.legend(handles=legend_elements, title='Configuration', 
                             loc='upper left', bbox_to_anchor=(0, 1))
    ax1.add_artist(config_legend)
    
    # Add method legend
    method_legend_elements = [
        plt.Line2D([0], [0], color='black', linestyle='-', marker='o', linewidth=2, label='Standard'),
        plt.Line2D([0], [0], color='black', linestyle='--', marker='s', linewidth=2, label='Nova')
    ]
    ax1.legend(handles=method_legend_elements, title='Method', loc='upper right')
    
    # Same for size plot
    ax2.legend(handles=legend_elements, title='Configuration', 
               loc='upper left', bbox_to_anchor=(0, 1))
    ax2.add_artist(ax2.legend(handles=method_legend_elements, title='Method', loc='upper right'))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = Path(base_dir) / "comprehensive_results_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Also create a summary table
    create_summary_table(base_dir, subdirs)
    
    plt.show()

def create_summary_table(base_dir, subdirs):
    """Create a summary table of all configurations and their crossover points."""
    
    summary_data = []
    
    for subdir in subdirs:
        df, cpu, memory = load_results_from_directory(base_dir, subdir)
        
        if df is None:
            continue
        
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
            'Configuration': f"CPU: {cpu} cores, RAM: {memory}GB" if cpu and memory else subdir,
            'Max Readings': max_readings,
            'Crossover Point': crossover_point if crossover_point else 'Not reached',
            'Standard Time (max)': f"{max_row['Standard Zeit (s)']:.2f}s",
            'Nova Time (max)': f"{max_row['Nova Zeit (s)']:.2f}s",
            'Standard Size (max)': f"{max_row['Standard Größe (KB)']:.1f}KB",
            'Nova Size (max)': f"{max_row['Nova Größe (KB)']:.1f}KB"
        })
    
    # Create and save summary table
    summary_df = pd.DataFrame(summary_data)
    summary_file = Path(base_dir) / "experiment_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary table saved to: {summary_file}")
    print("\nExperiment Summary:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    # Set the base directory
    base_dir = "data/visualizations"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        exit(1)
    
    create_comprehensive_plot(base_dir)
