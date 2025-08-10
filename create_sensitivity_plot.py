#!/usr/bin/env python3
"""
Create Crossover Sensitivity Analysis Plot
Generates the missing crossover_sensitivity_analysis.png for the thesis
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Load crossover analysis data
data_file = Path("/home/ramon/bachelor/data/visualizations/crossover_analysis_report.json")
with open(data_file, 'r') as f:
    data = json.load(f)

sensitivity = data["sensitivity_analysis"]

# Create the sensitivity analysis plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Crossover Point Sensitivity Analysis\nImpact of Key Parameters on Recursive SNARK Threshold', 
             fontsize=16, fontweight='bold')

# 1. Batch Size Sensitivity (Top Left)
ax1 = axes[0, 0]
batch_sizes = [10, 20, 30, 40, 50, 80, 120]
batch_crossovers = sensitivity["batch_size"]
ax1.plot(batch_sizes, batch_crossovers, 'bo-', linewidth=2, markersize=8)
ax1.axhline(y=171, color='red', linestyle='--', alpha=0.7, label='Baseline (171)')
ax1.set_xlabel('Batch Size (items)', fontweight='bold')
ax1.set_ylabel('Crossover Point (items)', fontweight='bold')
ax1.set_title('📦 Batch Size Impact', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Folding Speedup Sensitivity (Top Right)
ax2 = axes[0, 1]
speedups = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
folding_crossovers = sensitivity["folding_speedup"]
ax2.plot(speedups, folding_crossovers, 'go-', linewidth=2, markersize=8)
ax2.axhline(y=171, color='red', linestyle='--', alpha=0.7, label='Baseline (171)')
ax2.set_xlabel('Folding Speedup Factor', fontweight='bold')
ax2.set_ylabel('Crossover Point (items)', fontweight='bold')
ax2.set_title('⚡ Folding Efficiency Impact', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Memory Constraint Sensitivity (Bottom Left)
ax3 = axes[1, 0]
memory_factors = [0.5, 0.75, 1.0, 1.5, 2.0]
memory_crossovers = sensitivity["memory_constraint"]
ax3.plot(memory_factors, memory_crossovers, 'mo-', linewidth=2, markersize=8)
ax3.axhline(y=171, color='red', linestyle='--', alpha=0.7, label='Baseline (171)')
ax3.set_xlabel('Memory Constraint Factor', fontweight='bold')
ax3.set_ylabel('Crossover Point (items)', fontweight='bold')
ax3.set_title('🧠 Memory Constraint Impact', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Setup Cost Ratio Sensitivity (Bottom Right)
ax4 = axes[1, 1]
setup_ratios = [2, 4, 8, 12, 20]
setup_crossovers = sensitivity["setup_cost_ratio"]
ax4.plot(setup_ratios, setup_crossovers, 'co-', linewidth=2, markersize=8)
ax4.axhline(y=171, color='red', linestyle='--', alpha=0.7, label='Baseline (171)')
ax4.set_xlabel('Setup Cost Ratio (Recursive/Standard)', fontweight='bold')
ax4.set_ylabel('Crossover Point (items)', fontweight='bold')
ax4.set_title('⚙️ Setup Overhead Impact', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

# Add sensitivity ranking text box
sensitivity_text = """
📊 SENSITIVITY RANKING:
1. Folding Speedup: HIGH (±70 items)
2. Memory Constraint: MEDIUM (±10 items)  
3. Setup Cost: MEDIUM (±75 items)
4. Batch Size: LOW (±8 items)
"""

fig.text(0.02, 0.02, sensitivity_text, fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
         verticalalignment='bottom')

plt.tight_layout()

# Save the plot
output_file = Path("/home/ramon/bachelor/data/visualizations/crossover_sensitivity_analysis.png")
output_file.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ Sensitivity analysis plot created: {output_file}")

# Also create a summary statistics table
summary_stats = {
    "parameter": ["Batch Size", "Folding Speedup", "Memory Constraint", "Setup Cost Ratio"],
    "baseline_value": [50, 2.5, 1.0, 8],
    "range_tested": ["10-120", "2.0-5.0", "0.5-2.0", "2-20"],
    "crossover_range": ["156-171", "116-241", "156-176", "56-206"],
    "sensitivity_score": ["Low", "Very High", "Medium", "High"]
}

print("\n📊 SENSITIVITY ANALYSIS SUMMARY:")
print("="*60)
for i, param in enumerate(summary_stats["parameter"]):
    print(f"{param:20s} | Range: {summary_stats['crossover_range'][i]:10s} | {summary_stats['sensitivity_score'][i]}")
print("="*60)
