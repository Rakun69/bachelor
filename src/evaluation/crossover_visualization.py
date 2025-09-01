"""
Crossover Analysis Visualization
Creates tables and plots for Standard vs Nova Recursive SNARKs comparison
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CrossoverAnalyzer:
    """Analyze and visualize crossover points for SNARK comparison"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.data = self._load_comparison_data()
        
    def _load_comparison_data(self) -> Dict[str, Any]:
        """Load fair comparison results"""
        results_file = self.project_root / "data" / "comparison" / "fair_systematic_comparison.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Comparison results not found: {results_file}")
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create detailed comparison table"""
        
        # Extract data for table
        table_data = []
        
        for result in self.data['results']:
            batch_size = result['batch_size']
            
            # Standard SNARK metrics
            std_prove_time = result['standard_total_time']
            std_verify_time = result.get('total_with_verification', 0) - std_prove_time
            std_total_time = std_prove_time + std_verify_time
            std_proof_size = result['standard_total_size']
            std_verify_count = result['standard_verify_count']
            
            # Nova metrics
            nova_prove_time = result['nova_prove_time']
            nova_compress_time = result['nova_compress_time']
            nova_verify_time = result['nova_verify_time']
            nova_total_time = result['nova_total_time']
            nova_proof_size = result['nova_proof_size']
            
            # Advantages
            time_advantage = result['time_advantage_factor']
            verification_advantage = result['verification_advantage']
            storage_advantage = std_proof_size / nova_proof_size
            
            table_data.append({
                'Batch Size': batch_size,
                'Items': result['data_items'],
                
                # Standard SNARKs
                'Standard Prove (s)': f"{std_prove_time:.2f}",
                'Standard Verify (s)': f"{std_verify_time:.2f}",
                'Standard Total (s)': f"{std_total_time:.2f}",
                'Standard Proofs': std_verify_count,
                'Standard Size (KB)': f"{std_proof_size/1024:.1f}",
                
                # Nova Recursive
                'Nova Prove (s)': f"{nova_prove_time:.2f}",
                'Nova Compress (s)': f"{nova_compress_time:.2f}",
                'Nova Verify (s)': f"{nova_verify_time:.2f}",
                'Nova Total (s)': f"{nova_total_time:.2f}",
                'Nova Proofs': 1,
                'Nova Size (KB)': f"{nova_proof_size/1024:.1f}",
                
                # Advantages
                'Time Speedup': f"{time_advantage:.1f}x",
                'Verify Reduction': f"{verification_advantage}:1",
                'Storage Ratio': f"{storage_advantage:.1f}:1"
            })
        
        df = pd.DataFrame(table_data)
        
        # Save as CSV for thesis
        output_file = self.project_root / "data" / "comparison" / "crossover_analysis_table.csv"
        df.to_csv(output_file, index=False)
        print(f"📊 Table saved to: {output_file}")
        
        return df
    
    def create_crossover_plots(self):
        """Create publication-quality crossover plots"""
        
        # Extract data for plotting
        batch_sizes = []
        std_times = []
        nova_times = []
        time_advantages = []
        verification_counts = []
        
        for result in self.data['results']:
            batch_sizes.append(result['batch_size'])
            std_times.append(result['standard_total_time'])
            nova_times.append(result['nova_total_time'])
            time_advantages.append(result['time_advantage_factor'])
            verification_counts.append(result['verification_advantage'])
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Standard vs Nova Recursive SNARKs: Crossover Analysis\nSmart Home IoT Data Processing', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Proving Times Comparison
        ax1.plot(batch_sizes, std_times, 'o-', label='Standard SNARKs', linewidth=2, markersize=8, color='#e74c3c')
        ax1.plot(batch_sizes, nova_times, 's-', label='Nova Recursive', linewidth=2, markersize=8, color='#3498db')
        
        # Mark crossover point
        crossover_point = self.data['crossover_analysis']['time_crossover']
        ax1.axvline(x=crossover_point, color='green', linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(crossover_point + 5, max(std_times) * 0.8, f'Crossover\n{crossover_point} items', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        ax1.set_xlabel('Batch Size (IoT Items)')
        ax1.set_ylabel('Total Time (seconds)')
        ax1.set_title('A) Proving Time Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Time Advantage Factor
        ax2.plot(batch_sizes, time_advantages, 'o-', linewidth=2, markersize=8, color='#9b59b6')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Break-even (1x)')
        ax2.axvline(x=crossover_point, color='green', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Batch Size (IoT Items)')
        ax2.set_ylabel('Nova Speedup Factor')
        ax2.set_title('B) Nova Performance Advantage')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add text annotations for key points
        for i, (x, y) in enumerate(zip(batch_sizes, time_advantages)):
            if x in [25, 100, 200]:  # Key points
                ax2.annotate(f'{y:.1f}x', (x, y), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontweight='bold')
        
        # Plot 3: Verification Scaling
        std_verifications = batch_sizes  # N verifications for standard
        nova_verifications = [1] * len(batch_sizes)  # Always 1 for Nova
        
        ax3.plot(batch_sizes, std_verifications, 'o-', label='Standard SNARKs (N proofs)', 
                linewidth=2, markersize=8, color='#e74c3c')
        ax3.plot(batch_sizes, nova_verifications, 's-', label='Nova Recursive (1 proof)', 
                linewidth=2, markersize=8, color='#3498db')
        
        ax3.set_xlabel('Batch Size (IoT Items)')
        ax3.set_ylabel('Number of Verifications Required')
        ax3.set_title('C) Verification Complexity Scaling')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cost per Item Analysis
        std_cost_per_item = [t/n for t, n in zip(std_times, batch_sizes)]
        nova_cost_per_item = [t/n for t, n in zip(nova_times, batch_sizes)]
        
        ax4.plot(batch_sizes, std_cost_per_item, 'o-', label='Standard SNARKs', 
                linewidth=2, markersize=8, color='#e74c3c')
        ax4.plot(batch_sizes, nova_cost_per_item, 's-', label='Nova Recursive', 
                linewidth=2, markersize=8, color='#3498db')
        
        ax4.axvline(x=crossover_point, color='green', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Batch Size (IoT Items)')
        ax4.set_ylabel('Time per Item (seconds)')
        ax4.set_title('D) Cost Efficiency per IoT Reading')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.project_root / "data" / "visualizations" / "crossover_analysis_plots.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📈 Plots saved to: {output_file}")
        
        plt.show()
    
    def create_summary_table(self):
        """Create executive summary table"""
        
        crossover = self.data['crossover_analysis']
        best = self.data['summary']['best_nova_advantage']
        
        summary_data = {
            'Metric': [
                'Time Crossover Point',
                'Verification Crossover',
                'Storage Crossover',
                'Best Speedup (200 items)',
                'Best Verification Reduction',
                'Recommendation Threshold'
            ],
            'Value': [
                f"{crossover['time_crossover']} items",
                f"{crossover['verification_crossover']} items",
                f"{crossover['storage_crossover']} items",
                f"{best['time_factor']:.1f}x faster",
                f"{best['verification_reduction']}:1 reduction",
                f"≥{crossover['time_crossover']} items → Nova"
            ],
            'Significance': [
                'Nova becomes faster than Standard',
                'Nova always better (1 vs N verifications)',
                'Nova needs less storage space',
                'Maximum observed performance gain',
                'Dramatic verification complexity reduction',
                'Decision boundary for IoT applications'
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        output_file = self.project_root / "data" / "comparison" / "executive_summary.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"📋 Executive summary saved to: {output_file}")
        
        return summary_df
    
    def print_thesis_ready_table(self, df: pd.DataFrame):
        """Print LaTeX-ready table for thesis"""
        
        print("\n" + "="*100)
        print("📚 THESIS-READY TABLE")
        print("="*100)
        
        # Pretty print the table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print(df.to_string(index=False))
        
        print("\n" + "="*100)
        print("📊 CROSSOVER ANALYSIS SUMMARY")
        print("="*100)
        
        crossover = self.data['crossover_analysis']
        print(f"🎯 Time Crossover Point: {crossover['time_crossover']} items")
        print(f"✅ Verification Advantage: {crossover['verification_crossover']} items (always better)")
        print(f"💾 Storage Crossover: {crossover['storage_crossover']} items")
        
        best = self.data['summary']['best_nova_advantage']
        print(f"\n🏆 Best Performance (200 items):")
        print(f"   📈 Speedup: {best['time_factor']:.1f}x faster")
        print(f"   🔍 Verification: {best['verification_reduction']}:1 reduction")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"   📱 Standard SNARKs: < {crossover['time_crossover']} items (real-time)")
        print(f"   🏠 Nova Recursive: ≥ {crossover['time_crossover']} items (batch processing)")

def main():
    """Generate all crossover analysis outputs"""
    
    print("🚀 GENERATING CROSSOVER ANALYSIS...")
    
    analyzer = CrossoverAnalyzer("/home/ramon/bachelor")
    
    # Create detailed comparison table
    print("\n📊 Creating detailed comparison table...")
    df = analyzer.create_comparison_table()
    
    # Create executive summary
    print("\n📋 Creating executive summary...")
    summary_df = analyzer.create_summary_table()
    
    # Create plots
    print("\n📈 Creating crossover analysis plots...")
    analyzer.create_crossover_plots()
    
    # Print thesis-ready output
    analyzer.print_thesis_ready_table(df)
    
    print("\n🎉 CROSSOVER ANALYSIS COMPLETED!")
    print("📁 Files generated:")
    print("   📊 data/comparison/crossover_analysis_table.csv")
    print("   📋 data/comparison/executive_summary.csv") 
    print("   📈 data/visualizations/crossover_analysis_plots.png")

if __name__ == "__main__":
    main()
