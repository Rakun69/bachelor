#!/usr/bin/env python3
"""
Interactive plotting script - allows user to select which configurations to display.
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
        print(f"Warning: Could not parse settings from {settings_file}: {e}")
        return None, None

def load_experiment_data(base_dir):
    """Load all experiment data and extract configurations."""
    
    base_path = Path(base_dir)
    subdirs = sorted([d.name for d in base_path.iterdir() 
                      if d.is_dir() and d.name.startswith('md_warm_nr')])
    
    experiments = []
    
    for subdir in subdirs:
        dir_path = base_path / subdir
        csv_file = dir_path / "crossover_summary.csv"
        settings_file = dir_path / "settings.txt"
        
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
            print(f"Error loading {subdir}: {e}")
            continue
    
    return experiments

def create_plot(experiments, selected_configs, title_suffix=""):
    """Create a plot with selected configurations."""
    
    # Filter experiments
    filtered_experiments = [exp for exp in experiments if exp['name'] in selected_configs]
    
    if not filtered_experiments:
        print("No experiments selected!")
        return
    
    print(f"\nCreating plot with {len(filtered_experiments)} configurations:")
    for exp in filtered_experiments:
        print(f"  - {exp['config_label']}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_experiments)))
    
    for i, exp in enumerate(filtered_experiments):
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
    
    # Configure time plot
    ax1.set_xlabel('Number of IoT Readings', fontsize=14)
    ax1.set_ylabel('Proving Time (seconds)', fontsize=14)
    ax1.set_title(f'Proving Time Comparison{title_suffix}', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Configure size plot
    ax2.set_xlabel('Number of IoT Readings', fontsize=14)
    ax2.set_ylabel('Proof Size (KB)', fontsize=14)
    ax2.set_title(f'Proof Size Comparison{title_suffix}', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path("data/visualizations") / f"selected_comparison{title_suffix.replace(' ', '_').lower()}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()

def main():
    base_dir = "data/visualizations"
    
    print("Loading experiment data...")
    experiments = load_experiment_data(base_dir)
    
    if not experiments:
        print("No experiment data found!")
        return
    
    print(f"\nLoaded {len(experiments)} experiments:")
    for i, exp in enumerate(experiments):
        print(f"  {i+1}. {exp['name']}: {exp['config_label']}")
    
    # Predefined selections for common use cases
    print("\n" + "="*60)
    print("PREDEFINED SELECTIONS:")
    print("="*60)
    
    # 1. Show only extreme configurations
    low_resource = [exp for exp in experiments if exp['cpu'] == 0.5 and exp['memory'] == 1.0]
    high_resource = [exp for exp in experiments if exp['cpu'] == 4.0 and exp['memory'] == 2.0]
    
    if low_resource and high_resource:
        print("\n1. EXTREME CONFIGURATIONS (Lowest vs Highest Resources):")
        selected_extreme = [low_resource[0]['name'], high_resource[0]['name']]
        create_plot(experiments, selected_extreme, " - Extreme Configurations")
    
    # 2. Show configurations with crossover points
    crossover_configs = []
    for exp in experiments:
        df = exp['data']
        if 'Nova' in df['Gewinner'].values:
            crossover_configs.append(exp['name'])
    
    if crossover_configs:
        print(f"\n2. CONFIGURATIONS WITH CROSSOVER POINTS:")
        print(f"   Found {len(crossover_configs)} configs with Nova advantages")
        create_plot(experiments, crossover_configs, " - With Crossover Points")
    
    # 3. Show only first 3 configurations (less cluttered)
    first_three = [exp['name'] for exp in experiments[:3]]
    print(f"\n3. FIRST THREE CONFIGURATIONS (Less Cluttered):")
    create_plot(experiments, first_three, " - First Three")
    
    # 4. Show only last 3 configurations
    last_three = [exp['name'] for exp in experiments[-3:]]
    print(f"\n4. LAST THREE CONFIGURATIONS:")
    create_plot(experiments, last_three, " - Last Three")
    
    print("\n" + "="*60)
    print("CUSTOM SELECTION:")
    print("="*60)
    print("To create custom plots, modify the script and add your selection.")
    print("Example: create_plot(experiments, ['md_warm_nr1', 'md_warm_nr6'], ' - Custom')")
    
    # Example custom selection
    custom_selection = ['md_warm_nr1', 'md_warm_nr6']  # Change this to your preferred configs
    print(f"\n5. CUSTOM SELECTION (Example): {custom_selection}")
    create_plot(experiments, custom_selection, " - Custom Selection")

if __name__ == "__main__":
    main()
