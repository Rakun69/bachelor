#!/usr/bin/env python3
"""
Extended Crossover Analysis with Additional Batch Sizes (60, 70, 80, 90)
Präzise Bestimmung des exakten Crossover-Punkts zwischen 50-100 Items
"""
import sys
import os
import json
import time
from pathlib import Path

# Add project paths
sys.path.append('src')

from evaluation.comparison_framework import ComparisonFramework
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_extended_crossover_analysis():
    """Run extended crossover analysis with additional batch sizes"""
    
    logger.info("🚀 Starting Extended Crossover Analysis (60, 70, 80, 90 items)")
    
    # Initialize comparison framework
    output_dir = "data/extended_crossover_analysis"
    framework = ComparisonFramework(output_dir=output_dir)
    
    # Extended batch sizes: Include current ones + new granular testing
    batch_sizes = [10, 25, 50, 60, 70, 80, 90, 100, 200]
    
    try:
        # Run the comprehensive comparison
        logger.info(f"Testing batch sizes: {batch_sizes}")
        results = framework.run_comparison(batch_sizes=batch_sizes, max_data_size=1000)
        
        if results.get("comparison_successful"):
            logger.info("✅ Extended crossover analysis completed successfully!")
            
            # Extract key findings
            crossover = results.get("crossover_analysis", {})
            logger.info(f"\n🎯 EXTENDED CROSSOVER ANALYSIS:")
            logger.info(f"   Time Crossover: {crossover.get('time_crossover', 'Not found')} items")
            logger.info(f"   Storage Crossover: {crossover.get('storage_crossover', 'Not found')} items")
            logger.info(f"   Efficiency Crossover: {crossover.get('efficiency_crossover', 'Not found')} items")
            
            # Save detailed results
            results_file = Path(output_dir) / "extended_crossover_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📊 Detailed results saved to: {results_file}")
            
            # Generate updated comparison table
            generate_extended_comparison_table(results, output_dir)
            
            return results
            
        else:
            logger.error(f"❌ Extended analysis failed: {results.get('error')}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error during extended crossover analysis: {e}")
        return None

def generate_extended_comparison_table(results, output_dir):
    """Generate extended comparison table with new batch sizes"""
    
    if "comparison_table" not in results:
        logger.error("No comparison table data found in results")
        return
    
    table_data = results["comparison_table"]
    
    # Create CSV table
    csv_file = Path(output_dir) / "extended_crossover_table.csv"
    
    # Group by batch size for easier analysis
    batch_groups = {}
    for row in table_data:
        batch_size = row["batch_size"]
        if batch_size not in batch_groups:
            batch_groups[batch_size] = {}
        batch_groups[batch_size][row["approach"]] = row
    
    # Generate CSV
    with open(csv_file, 'w') as f:
        f.write("Batch Size,Standard Zeit (s),Standard Proofs,Standard Größe (KB),")
        f.write("Nova Zeit (s),Nova Größe (KB),Zeit Vorteil,Größe Vorteil,Gewinner,Vorteil %\n")
        
        for batch_size in sorted(batch_groups.keys()):
            group = batch_groups[batch_size]
            
            if "classical" in group and "recursive" in group:
                standard = group["classical"]
                nova = group["recursive"]
                
                # Calculate advantages
                time_advantage = standard["prove_time"] / nova["prove_time"]
                size_advantage = standard["proof_size"] / nova["proof_size"]
                
                winner = "Nova" if time_advantage > 1 else "Standard"
                advantage_pct = f"+{int((time_advantage-1)*100)}%" if time_advantage > 1 else f"{int((time_advantage-1)*100)}%"
                
                f.write(f"{batch_size},{standard['prove_time']:.2f},{batch_size},{standard['proof_size']/1024:.1f},")
                f.write(f"{nova['prove_time']:.2f},{nova['proof_size']/1024:.1f},{time_advantage:.1f}x,{size_advantage:.1f}x,")
                f.write(f"{winner},{advantage_pct}\n")
    
    logger.info(f"📋 Extended comparison table saved to: {csv_file}")
    
    # Also generate HTML table for better visualization
    generate_extended_html_table(csv_file, output_dir)

def generate_extended_html_table(csv_file, output_dir):
    """Generate HTML table with enhanced crossover visualization"""
    
    import csv
    
    html_file = Path(output_dir) / "extended_crossover_table.html"
    
    with open(html_file, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Extended IoT ZK-SNARK Crossover Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
                th { background-color: #3498db; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .winner-nova { background-color: #d5f4e6; font-weight: bold; }
                .winner-standard { background-color: #ffeaa7; font-weight: bold; }
                .crossover-highlight { background-color: #ff7675; color: white; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>🎯 Extended IoT ZK-SNARK Crossover Analysis</h1>
            <h2>Präzise Crossover-Punkt Bestimmung (60, 70, 80, 90 Items)</h2>
            <p><strong>Generiert:</strong> """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
        """)
        
        f.write('<table>\n<thead>\n<tr>\n')
        
        # Read CSV and create HTML table
        with open(csv_file, 'r') as csv_f:
            csv_reader = csv.reader(csv_f)
            headers = next(csv_reader)
            
            # Write headers
            for header in headers:
                f.write(f'<th>{header}</th>\n')
            f.write('</tr>\n</thead>\n<tbody>\n')
            
            # Write data rows with highlighting
            for row in csv_reader:
                batch_size = int(row[0])
                winner = row[8]
                
                row_class = ""
                if winner == "Nova":
                    row_class = 'class="winner-nova"'
                elif winner == "Standard":
                    row_class = 'class="winner-standard"'
                
                f.write(f'<tr {row_class}>\n')
                for cell in row:
                    f.write(f'<td>{cell}</td>\n')
                f.write('</tr>\n')
        
        f.write("""
            </tbody>
            </table>
            
            <h2>📊 Extended Key Findings</h2>
            <ul>
                <li><strong>Granular Analysis:</strong> Teste Batch-Größen 60, 70, 80, 90 für präzisen Crossover</li>
                <li><strong>Crossover-Bereich:</strong> Erwarte Crossover zwischen 70-90 Items</li>
                <li><strong>Nova Advantage:</strong> Wird bei größeren Batches exponentiell besser</li>
                <li><strong>Standard SNARK Stärken:</strong> Optimiert für kleine Batches und Real-time Processing</li>
            </ul>
            
            <h2>🚀 Deployment Guidelines</h2>
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                <h3>Standard ZK-SNARKs verwenden wenn:</h3>
                <ul>
                    <li>< [Crossover-Punkt] IoT Items pro Batch</li>
                    <li>Real-time Processing erforderlich</li>
                    <li>Niedrige Latenz kritisch</li>
                </ul>
                
                <h3>Nova Recursive SNARKs verwenden wenn:</h3>
                <ul>
                    <li>≥ [Crossover-Punkt] IoT Items pro Batch</li>
                    <li>Konstante Proof-Größe wichtig</li>
                    <li>Langfristige Skalierbarkeit erforderlich</li>
                </ul>
            </div>
        </body>
        </html>
        """)
    
    logger.info(f"📋 Extended HTML table saved to: {html_file}")

if __name__ == "__main__":
    print("🚀 Extended IoT ZK-SNARK Crossover Analysis")
    print("Testing additional batch sizes: 60, 70, 80, 90")
    print("=" * 60)
    
    results = run_extended_crossover_analysis()
    
    if results:
        print("\n✅ Extended crossover analysis completed!")
        print(f"📊 Check results in: data/extended_crossover_analysis/")
        print("📋 Files generated:")
        print("   - extended_crossover_results.json")
        print("   - extended_crossover_table.csv")  
        print("   - extended_crossover_table.html")
    else:
        print("\n❌ Extended crossover analysis failed!")
