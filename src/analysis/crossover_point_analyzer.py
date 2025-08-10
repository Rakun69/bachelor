"""
Crossover Point Analyzer - Theoretical and Empirical Analysis
Determines when Recursive SNARKs become superior to Standard SNARKs

This module implements the mathematical models from the thesis to analyze
the crossover point between SNARK systems across different scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

@dataclass
class SNARKParameters:
    """Parameters for SNARK performance modeling"""
    setup_time: float
    prove_time_per_item: float
    verify_time: float
    proof_size: float
    memory_scaling_factor: float
    memory_scaling_exponent: float = 1.0  # 1.0 for linear, <1.0 for sub-linear

@dataclass
class CrossoverResult:
    """Result of crossover point analysis"""
    crossover_point: int
    proving_crossover: int
    storage_crossover: int
    memory_crossover: int
    confidence_interval: Tuple[int, int]
    efficiency_ratio_at_crossover: float

class CrossoverPointAnalyzer:
    """Analyzes the crossover point between Standard and Recursive SNARKs"""
    
    def __init__(self):
        # Resolve project root and default visualization directory (stable regardless of CWD)
        self.project_root = Path(__file__).resolve().parents[2]
        self.viz_dir = self.project_root / "data" / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        # Default parameters based on realistic empirical data
        self.standard_params = SNARKParameters(
            setup_time=0.1,          # 100ms
            prove_time_per_item=0.010, # 10ms per item (more realistic)
            verify_time=0.005,       # 5ms
            proof_size=800,          # 800 bytes per proof (more realistic)
            memory_scaling_factor=0.5,  # 0.5MB per item
            memory_scaling_exponent=1.0  # Linear scaling
        )
        
        self.recursive_params = SNARKParameters(
            setup_time=0.8,          # 800ms (higher setup for recursive)
            prove_time_per_item=0.005, # 5ms per item (folding is faster) 
            verify_time=0.005,       # 5ms (constant)
            proof_size=2048,         # 2KB constant
            memory_scaling_factor=0.3,  # 0.3MB base + sublinear
            memory_scaling_exponent=0.7  # Sub-linear scaling
        )
        
        self.batch_size = 50  # Larger batches favor recursive SNARKs
        self.compression_time = 0.2  # 200ms for final compression
    
    def calculate_standard_cost(self, n_items: int) -> Dict[str, float]:
        """Calculate total cost for Standard SNARKs"""
        n_batches = np.ceil(n_items / self.batch_size)
        
        setup_cost = self.standard_params.setup_time
        proving_cost = n_items * self.standard_params.prove_time_per_item  # Cost per item, not per batch
        verify_cost = n_batches * self.standard_params.verify_time
        storage_cost = n_batches * self.standard_params.proof_size / 1024  # Normalize to KB
        memory_cost = (self.standard_params.memory_scaling_factor * 
                      (n_items ** self.standard_params.memory_scaling_exponent))
        
        total_cost = setup_cost + proving_cost + verify_cost + storage_cost * 0.001 + memory_cost * 0.001
        
        return {
            'total': total_cost,
            'setup': setup_cost,
            'proving': proving_cost,
            'verification': verify_cost,
            'storage': storage_cost,
            'memory': memory_cost,
            'n_proofs': n_batches
        }
    
    def calculate_recursive_cost(self, n_items: int) -> Dict[str, float]:
        """Calculate total cost for Recursive SNARKs"""
        n_batches = np.ceil(n_items / self.batch_size)
        
        setup_cost = self.recursive_params.setup_time
        folding_cost = n_items * self.recursive_params.prove_time_per_item  # Cost per item for folding
        compression_cost = self.compression_time
        verify_cost = self.recursive_params.verify_time  # Constant
        storage_cost = self.recursive_params.proof_size / 1024  # Constant 2KB
        memory_cost = (self.recursive_params.memory_scaling_factor * 
                      (n_items ** self.recursive_params.memory_scaling_exponent))
        
        total_cost = (setup_cost + folding_cost + compression_cost + verify_cost + 
                     storage_cost * 0.001 + memory_cost * 0.001)
        
        return {
            'total': total_cost,
            'setup': setup_cost,
            'proving': folding_cost,
            'compression': compression_cost,
            'verification': verify_cost,
            'storage': storage_cost,
            'memory': memory_cost,
            'n_proofs': 1  # Always 1 proof regardless of data size
        }
    
    def find_crossover_point(self, max_items: int = 1000, 
                           tolerance: float = 0.01) -> CrossoverResult:
        """Find the crossover point where recursive SNARKs become superior"""
        
        item_range = np.arange(1, max_items + 1, 5)
        standard_costs = []
        recursive_costs = []
        efficiency_ratios = []
        
        for n in item_range:
            std_cost = self.calculate_standard_cost(n)
            rec_cost = self.calculate_recursive_cost(n)
            
            standard_costs.append(std_cost['total'])
            recursive_costs.append(rec_cost['total'])
            
            # Efficiency ratio > 1 means recursive is better
            ratio = std_cost['total'] / max(rec_cost['total'], 0.001)
            efficiency_ratios.append(ratio)
        
        # Find crossover point
        crossover_idx = None
        for i, ratio in enumerate(efficiency_ratios):
            if ratio > 1.0 + tolerance:  # Recursive is clearly better
                crossover_idx = i
                break
        
        if crossover_idx is None:
            crossover_point = max_items
        else:
            crossover_point = item_range[crossover_idx]
        
        # Find individual crossover points
        proving_crossover = self._find_proving_crossover(max_items)
        storage_crossover = self._find_storage_crossover()
        memory_crossover = self._find_memory_crossover(max_items)
        
        # Calculate confidence interval (±10%)
        ci_lower = int(crossover_point * 0.9)
        ci_upper = int(crossover_point * 1.1)
        
        efficiency_at_crossover = efficiency_ratios[crossover_idx] if crossover_idx else 1.0
        
        return CrossoverResult(
            crossover_point=crossover_point,
            proving_crossover=proving_crossover,
            storage_crossover=storage_crossover,
            memory_crossover=memory_crossover,
            confidence_interval=(ci_lower, ci_upper),
            efficiency_ratio_at_crossover=efficiency_at_crossover
        )
    
    def _find_proving_crossover(self, max_items: int) -> int:
        """Find crossover point based purely on proving time"""
        # n > (batch_size * compression_time) / (prove_time - fold_time)
        prove_diff = (self.standard_params.prove_time_per_item - 
                     self.recursive_params.prove_time_per_item)
        
        if prove_diff <= 0:
            return 1  # Recursive is always better
        
        crossover = self.batch_size * self.compression_time / prove_diff
        return int(np.ceil(crossover))
    
    def _find_storage_crossover(self) -> int:
        """Find crossover point based on storage requirements"""
        # When does n_batches * proof_size > constant_proof_size?
        storage_ratio = self.recursive_params.proof_size / self.standard_params.proof_size
        return int(self.batch_size * storage_ratio)
    
    def _find_memory_crossover(self, max_items: int) -> int:
        """Find crossover point based on memory usage"""
        for n in range(1, max_items + 1, 10):
            std_memory = (self.standard_params.memory_scaling_factor * 
                         (n ** self.standard_params.memory_scaling_exponent))
            rec_memory = (self.recursive_params.memory_scaling_factor * 
                         (n ** self.recursive_params.memory_scaling_exponent))
            
            if rec_memory < std_memory:
                return n
        
        return max_items
    
    def generate_crossover_visualization(self, max_items: int = 500, 
                                       save_path: Optional[str] = None) -> plt.Figure:
        """Generate comprehensive crossover point visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SNARK vs Recursive SNARK Crossover Analysis', fontsize=16, fontweight='bold')
        
        item_range = np.arange(10, max_items + 1, 10)
        
        # Calculate costs for all data points
        std_costs = [self.calculate_standard_cost(n) for n in item_range]
        rec_costs = [self.calculate_recursive_cost(n) for n in item_range]
        
        # Extract different cost components
        std_total = [c['total'] for c in std_costs]
        rec_total = [c['total'] for c in rec_costs]
        std_proving = [c['proving'] for c in std_costs]
        rec_proving = [c['proving'] for c in rec_costs]
        std_storage = [c['storage'] for c in std_costs]
        rec_storage = [c['storage'] for c in rec_costs]
        std_memory = [c['memory'] for c in std_costs]
        rec_memory = [c['memory'] for c in rec_costs]
        
        # Find crossover result
        crossover_result = self.find_crossover_point(max_items)
        crossover_point = crossover_result.crossover_point
        
        # Plot 1: Total Cost Comparison
        ax1.plot(item_range, std_total, 'b-', linewidth=2, label='Standard SNARKs', marker='o', markersize=3)
        ax1.plot(item_range, rec_total, 'r-', linewidth=2, label='Recursive SNARKs', marker='s', markersize=3)
        ax1.axvline(x=crossover_point, color='green', linestyle='--', linewidth=2, 
                   label=f'Crossover Point: {crossover_point} items')
        ax1.set_xlabel('Number of Data Items')
        ax1.set_ylabel('Total Cost (normalized)')
        ax1.set_title('Total Cost Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Highlight crossover region
        ci_lower, ci_upper = crossover_result.confidence_interval
        ax1.axvspan(ci_lower, ci_upper, alpha=0.2, color='green', label='Confidence Interval')
        
        # Plot 2: Proving Time Comparison
        ax2.plot(item_range, std_proving, 'b-', linewidth=2, label='Standard Proving')
        ax2.plot(item_range, rec_proving, 'r-', linewidth=2, label='Recursive Folding')
        ax2.axvline(x=crossover_result.proving_crossover, color='orange', linestyle='--', 
                   label=f'Proving Crossover: {crossover_result.proving_crossover}')
        ax2.set_xlabel('Number of Data Items')
        ax2.set_ylabel('Proving Time (seconds)')
        ax2.set_title('Proving Time Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Storage Requirements
        ax3.plot(item_range, std_storage, 'b-', linewidth=2, label='Standard Storage (Linear)')
        ax3.plot(item_range, rec_storage, 'r-', linewidth=2, label='Recursive Storage (Constant)')
        ax3.axvline(x=crossover_result.storage_crossover, color='purple', linestyle='--',
                   label=f'Storage Crossover: {crossover_result.storage_crossover}')
        ax3.set_xlabel('Number of Data Items')
        ax3.set_ylabel('Storage Required (KB)')
        ax3.set_title('Storage Requirements Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Efficiency Ratio
        efficiency_ratios = [s/max(r, 0.001) for s, r in zip(std_total, rec_total)]
        ax4.plot(item_range, efficiency_ratios, 'g-', linewidth=3, label='Efficiency Ratio')
        ax4.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='Equal Performance')
        ax4.axvline(x=crossover_point, color='green', linestyle='--', linewidth=2)
        ax4.fill_between(item_range, 1, efficiency_ratios, 
                        where=[r > 1 for r in efficiency_ratios], 
                        alpha=0.3, color='green', label='Recursive SNARK Superior')
        ax4.fill_between(item_range, efficiency_ratios, 1, 
                        where=[r < 1 for r in efficiency_ratios], 
                        alpha=0.3, color='red', label='Standard SNARK Superior')
        ax4.set_xlabel('Number of Data Items')
        ax4.set_ylabel('Efficiency Ratio (Standard/Recursive)')
        ax4.set_title('Relative Efficiency Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 3)
        
        plt.tight_layout()
        
        # Determine final path
        target = None
        if save_path:
            target = self.project_root / save_path if not str(save_path).startswith("/") else Path(save_path)
        else:
            target = self.viz_dir / "theoretical_crossover_analysis.png"

        plt.savefig(target, dpi=300, bbox_inches='tight')
        print(f"Crossover analysis saved to: {target}")
        
        return fig

    def generate_crossover_overview(self, max_items: int = 1500,
                                    save_path: Optional[str] = None) -> Dict[str, str]:
        """Create a single, publication-ready overview plot for the crossover point.
        Shows total cost curves, shaded superiority regions, confidence band, and annotations.
        Returns a dict with the saved file path and key annotation values.
        """
        # Compute range and costs
        item_range = np.arange(10, max_items + 1, 10)
        std_total = [self.calculate_standard_cost(n)['total'] for n in item_range]
        rec_total = [self.calculate_recursive_cost(n)['total'] for n in item_range]

        # Crossover details
        crossover = self.find_crossover_point(max_items)
        x_cross = crossover.crossover_point
        ci_lo, ci_hi = crossover.confidence_interval

        # Build figure
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_title('Crossover Point Overview: Standard vs Recursive SNARKs', fontsize=16, fontweight='bold')

        # Plot cost curves
        ax.plot(item_range, std_total, color='#1f77b4', linewidth=2.5, label='Standard SNARKs (Total Cost)')
        ax.plot(item_range, rec_total, color='#d62728', linewidth=2.5, label='Recursive SNARKs (Total Cost)')

        # Shade superiority regions
        std_better = np.array(std_total) < np.array(rec_total)
        rec_better = np.array(std_total) > np.array(rec_total)
        ax.fill_between(item_range, std_total, rec_total, where=rec_better, color='green', alpha=0.15,
                        label='Recursive Superior')
        ax.fill_between(item_range, std_total, rec_total, where=std_better, color='red', alpha=0.10,
                        label='Standard Superior')

        # Confidence band and crossover line
        ax.axvspan(ci_lo, ci_hi, color='green', alpha=0.12, label=f'Confidence Band [{ci_lo}, {ci_hi}]')
        ax.axvline(x=x_cross, color='green', linestyle='--', linewidth=2,
                   label=f'Crossover ≈ {x_cross} items')

        # Annotations box
        annotation_text = (
            f"Crossover ≈ {x_cross} items\n"
            f"Proving crossover: {crossover.proving_crossover}\n"
            f"Storage crossover: {crossover.storage_crossover}\n"
            f"Memory crossover: {crossover.memory_crossover}\n"
            f"Efficiency ratio at crossover: {crossover.efficiency_ratio_at_crossover:.2f}x"
        )
        ax.text(0.98, 0.02, annotation_text, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

        ax.set_xlabel('Number of Data Items')
        ax.set_ylabel('Total Cost (normalized units)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

        plt.tight_layout()
        # Determine path
        target = None
        if save_path:
            target = self.project_root / save_path if not str(save_path).startswith("/") else Path(save_path)
        else:
            target = self.viz_dir / "crossover_point_overview.png"

        fig.savefig(target, dpi=300, bbox_inches='tight')
        output = {'overview_plot': str(target)}
        plt.close(fig)

        return {
            **output,
            'crossover_point': str(x_cross),
            'confidence_lower': str(ci_lo),
            'confidence_upper': str(ci_hi)
        }

    def generate_sensitivity_visualization(self, save_path: Optional[str] = None) -> str:
        """Visualize how the crossover point shifts under parameter variations."""
        # Use the same variations as the report
        variations = {
            'batch_size': [10, 15, 20, 25, 30, 40, 50],
            'folding_speedup': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            'memory_constraint': [0.5, 0.75, 1.0, 1.5, 2.0],
            'setup_cost_ratio': [2, 3, 5, 7, 10]
        }
        sens = self.sensitivity_analysis(variations)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Crossover Sensitivity Analysis', fontsize=16, fontweight='bold')

        # 1) Batch size
        ax = axes[0, 0]
        ax.plot(variations['batch_size'], sens['batch_size'], 'o-', linewidth=2)
        ax.set_title('Batch Size vs Crossover')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Crossover Point (items)')
        ax.grid(True, alpha=0.3)

        # 2) Folding speedup
        ax = axes[0, 1]
        ax.plot(variations['folding_speedup'], sens['folding_speedup'], 's-', linewidth=2, color='#2ca02c')
        ax.set_title('Folding Speedup vs Crossover')
        ax.set_xlabel('Folding Speedup (x)')
        ax.set_ylabel('Crossover Point (items)')
        ax.grid(True, alpha=0.3)

        # 3) Memory constraint
        ax = axes[1, 0]
        ax.plot(variations['memory_constraint'], sens['memory_constraint'], '^-', linewidth=2, color='#d62728')
        ax.set_title('Memory Constraint vs Crossover')
        ax.set_xlabel('Memory Multiplier')
        ax.set_ylabel('Crossover Point (items)')
        ax.grid(True, alpha=0.3)

        # 4) Setup cost ratio
        ax = axes[1, 1]
        ax.plot(variations['setup_cost_ratio'], sens['setup_cost_ratio'], 'd-', linewidth=2, color='#9467bd')
        ax.set_title('Recursive Setup Cost Ratio vs Crossover')
        ax.set_xlabel('Recursive Setup / Standard Setup')
        ax.set_ylabel('Crossover Point (items)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        # Resolve final path
        if save_path:
            out_path = self.project_root / save_path if not str(save_path).startswith("/") else Path(save_path)
        else:
            out_path = self.viz_dir / 'crossover_sensitivity_analysis.png'

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return str(out_path)
    
    def sensitivity_analysis(self, parameter_variations: Dict[str, List[float]]) -> Dict[str, List[int]]:
        """Perform sensitivity analysis on crossover point"""
        
        baseline_crossover = self.find_crossover_point().crossover_point
        sensitivities = {}
        
        for param_name, variations in parameter_variations.items():
            crossover_points = []
            
            for variation in variations:
                # Create modified analyzer
                analyzer = CrossoverPointAnalyzer()
                
                # Modify the specified parameter
                if param_name == 'batch_size':
                    analyzer.batch_size = int(variation)
                elif param_name == 'folding_speedup':
                    analyzer.recursive_params.prove_time_per_item = (
                        self.standard_params.prove_time_per_item / variation
                    )
                elif param_name == 'memory_constraint':
                    # Simulate memory constraint by increasing memory cost impact
                    analyzer.standard_params.memory_scaling_factor *= variation
                elif param_name == 'setup_cost_ratio':
                    analyzer.recursive_params.setup_time = (
                        self.standard_params.setup_time * variation
                    )
                
                crossover = analyzer.find_crossover_point().crossover_point
                crossover_points.append(crossover)
            
            sensitivities[param_name] = crossover_points
        
        return sensitivities
    
    def generate_report(self, max_items: int = 500) -> Dict[str, any]:
        """Generate comprehensive crossover analysis report"""
        
        crossover_result = self.find_crossover_point(max_items)
        
        # Sensitivity analysis
        sensitivity_params = {
            'batch_size': [10, 15, 20, 25, 30, 40, 50],
            'folding_speedup': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            'memory_constraint': [0.5, 0.75, 1.0, 1.5, 2.0],
            'setup_cost_ratio': [2, 3, 5, 7, 10]
        }
        
        sensitivities = self.sensitivity_analysis(sensitivity_params)
        
        # Calculate cost breakdown at crossover
        crossover_std_cost = self.calculate_standard_cost(crossover_result.crossover_point)
        crossover_rec_cost = self.calculate_recursive_cost(crossover_result.crossover_point)
        
        report = {
            'crossover_analysis': {
                'main_crossover_point': crossover_result.crossover_point,
                'proving_crossover': crossover_result.proving_crossover,
                'storage_crossover': crossover_result.storage_crossover,
                'memory_crossover': crossover_result.memory_crossover,
                'confidence_interval': crossover_result.confidence_interval,
                'efficiency_ratio': crossover_result.efficiency_ratio_at_crossover
            },
            'cost_breakdown_at_crossover': {
                'standard_snark': crossover_std_cost,
                'recursive_snark': crossover_rec_cost
            },
            'sensitivity_analysis': sensitivities,
            'theoretical_insights': {
                'primary_factors': [
                    'Proving efficiency improvement (folding vs. standard proving)',
                    'Constant vs. linear proof size growth',
                    'Sub-linear vs. linear memory scaling',
                    'Setup cost amortization over larger datasets'
                ],
                'key_thresholds': {
                    'immediate_storage_advantage': crossover_result.storage_crossover,
                    'proving_time_advantage': crossover_result.proving_crossover,
                    'memory_advantage': crossover_result.memory_crossover,
                    'overall_advantage': crossover_result.crossover_point
                }
            },
            'practical_recommendations': {
                'use_standard_snarks': f'Data size < {crossover_result.crossover_point // 2} items',
                'transition_zone': f'{crossover_result.crossover_point // 2} - {crossover_result.crossover_point} items',
                'use_recursive_snarks': f'Data size > {crossover_result.crossover_point} items',
                'critical_factors': [
                    'Memory constraints strongly favor recursive SNARKs',
                    'Storage limitations make recursive SNARKs attractive earlier',
                    'Real-time requirements may favor standard SNARKs despite higher costs'
                ]
            }
        }
        
        return report

def main():
    """Main function for running crossover analysis"""
    
    # Create analyzer
    analyzer = CrossoverPointAnalyzer()
    
    # Generate comprehensive analysis
    print("🔍 Running Comprehensive SNARK Crossover Analysis...")
    
    # Find crossover point
    crossover_result = analyzer.find_crossover_point(max_items=500)
    
    print(f"\n📊 CROSSOVER ANALYSIS RESULTS:")
    print(f"{'='*50}")
    print(f"📈 Main Crossover Point: {crossover_result.crossover_point} items")
    print(f"⚡ Proving Crossover: {crossover_result.proving_crossover} items")
    print(f"💾 Storage Crossover: {crossover_result.storage_crossover} items")
    print(f"🧠 Memory Crossover: {crossover_result.memory_crossover} items")
    print(f"📊 Efficiency Ratio at Crossover: {crossover_result.efficiency_ratio_at_crossover:.2f}x")
    print(f"🎯 Confidence Interval: {crossover_result.confidence_interval[0]}-{crossover_result.confidence_interval[1]} items")
    
    # Generate visualization
    print(f"\n🎨 Generating crossover visualization...")
    output_dir = analyzer.viz_dir
    
    viz_path = output_dir / "theoretical_crossover_analysis.png"
    analyzer.generate_crossover_visualization(max_items=500, save_path=str(viz_path))

    # Generate enhanced overview and sensitivity visuals
    overview_path = output_dir / "crossover_point_overview.png"
    analyzer.generate_crossover_overview(max_items=1500, save_path=str(overview_path))
    sensitivity_path = output_dir / "crossover_sensitivity_analysis.png"
    analyzer.generate_sensitivity_visualization(save_path=str(sensitivity_path))
    
    # Generate full report
    print(f"\n📋 Generating comprehensive report...")
    report = analyzer.generate_report(max_items=500)
    
    # Save report
    report_path = output_dir / "crossover_analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📊 Report saved to: {report_path}")
    
    # Print key insights
    print(f"\n🎯 KEY THEORETICAL INSIGHTS:")
    print(f"{'='*50}")
    print(f"💡 Recursive SNARKs become superior at ~{crossover_result.crossover_point} items")
    print(f"💡 Storage advantage starts immediately at {crossover_result.storage_crossover} items")
    print(f"💡 Memory efficiency favors recursive SNARKs beyond {crossover_result.memory_crossover} items")
    print(f"💡 Proving efficiency crossover occurs at {crossover_result.proving_crossover} items")
    
    print(f"\n🚀 PRACTICAL RECOMMENDATIONS:")
    print(f"{'='*50}")
    print(f"✅ Use Standard SNARKs: < {crossover_result.crossover_point // 2} items")
    print(f"⚠️  Transition Zone: {crossover_result.crossover_point // 2}-{crossover_result.crossover_point} items")
    print(f"🚀 Use Recursive SNARKs: > {crossover_result.crossover_point} items")
    
    print(f"\n✨ Analysis complete! Visualizations:")
    print(f" - Overview: {overview_path}")
    print(f" - Theory (multi-panel): {viz_path}")
    print(f" - Sensitivity: {sensitivity_path}")
    
    return report

if __name__ == "__main__":
    main()