#!/usr/bin/env python3
"""
Visualize crossover measurement results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def main():
    # Load data
    csv_path = Path("data/real_measurements/crossover_summary.csv")
    json_path = Path("data/real_measurements/crossover_results.json")
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print("📊 Data loaded:")
    print(df)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('IoT ZK-SNARK Performance Comparison', fontsize=16)
    
    # 1. Proof Time Comparison
    ax1 = axes[0, 0]
    ax1.plot(df['reading_count'], df['standard_time_median'], 'b-o', label='Standard SNARK', linewidth=2)
    ax1.plot(df['reading_count'], df['nova_time_median'], 'r-s', label='Nova Recursive', linewidth=2)
    ax1.set_xlabel('Number of IoT Readings')
    ax1.set_ylabel('Proof Time (seconds)')
    ax1.set_title('Proof Generation Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Proof Size Comparison
    ax2 = axes[0, 1]
    ax2.plot(df['reading_count'], df['standard_size_median'], 'b-o', label='Standard SNARK', linewidth=2)
    ax2.plot(df['reading_count'], df['nova_size_median'], 'r-s', label='Nova Recursive', linewidth=2)
    ax2.set_xlabel('Number of IoT Readings')
    ax2.set_ylabel('Proof Size (bytes)')
    ax2.set_title('Proof Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Time Ratio (Nova/Standard)
    ax3 = axes[1, 0]
    time_ratio = df['nova_time_median'] / df['standard_time_median']
    ax3.plot(df['reading_count'], time_ratio, 'g-^', linewidth=2, markersize=8)
    ax3.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Break-even')
    ax3.set_xlabel('Number of IoT Readings')
    ax3.set_ylabel('Time Ratio (Nova/Standard)')
    ax3.set_title('Performance Ratio (Lower = Better)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Size Ratio (Nova/Standard)
    ax4 = axes[1, 1]
    size_ratio = df['nova_size_median'] / df['standard_size_median']
    ax4.plot(df['reading_count'], size_ratio, 'm-v', linewidth=2, markersize=8)
    ax4.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Break-even')
    ax4.set_xlabel('Number of IoT Readings')
    ax4.set_ylabel('Size Ratio (Nova/Standard)')
    ax4.set_title('Size Efficiency (Lower = Better)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("data/visualizations/crossover_analysis.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📈 Visualization saved: {output_path}")
    
    # Show plot
    plt.show()
    
    # Find crossover points
    print("\n🎯 CROSSOVER ANALYSIS:")
    print("=" * 50)
    
    # Time crossover
    time_crossover = None
    for i in range(len(time_ratio)):
        if time_ratio.iloc[i] < 1.0:
            time_crossover = df['reading_count'].iloc[i]
            break
    
    # Size crossover  
    size_crossover = None
    for i in range(len(size_ratio)):
        if size_ratio.iloc[i] < 1.0:
            size_crossover = df['reading_count'].iloc[i]
            break
    
    if time_crossover:
        print(f"⏱️  Time Crossover Point: {time_crossover} readings")
    else:
        print("⏱️  Time Crossover Point: Not reached")
        
    if size_crossover:
        print(f"📦 Size Crossover Point: {size_crossover} readings")
    else:
        print("📦 Size Crossover Point: Not reached")

if __name__ == "__main__":
    main()
