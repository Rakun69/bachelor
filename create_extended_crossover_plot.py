#!/usr/bin/env python3
"""
Erstelle erweiterte Crossover-Visualisierung mit 60, 70, 80, 90 IoT Readings
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def create_extended_crossover_visualization():
    """Erstelle erweiterte Crossover-Visualisierung"""
    
    # Erweiterte Daten (inklusive 60, 70, 80, 90)
    data = {
        'iot_readings': [10, 25, 50, 60, 70, 80, 90, 100, 200],
        'standard_time': [4.57, 11.42, 22.85, 27.42, 31.99, 36.56, 41.13, 45.69, 91.38],
        'nova_time': [27.71, 30.84, 35.73, 37.08, 38.25, 39.31, 40.29, 43.08, 60.64],
        'standard_size': [88.5, 221.3, 442.6, 531.0, 619.5, 708.0, 796.5, 885.2, 1770.3],
        'nova_size': [69.1, 69.1, 69.2, 69.1, 69.1, 69.2, 69.1, 69.2, 69.1]
    }
    
    readings = np.array(data['iot_readings'])
    std_times = np.array(data['standard_time'])
    nova_times = np.array(data['nova_time'])
    time_advantages = std_times / nova_times
    
    # Find crossover point
    crossover_idx = np.where(time_advantages > 1.0)[0]
    crossover_point = readings[crossover_idx[0]] if len(crossover_idx) > 0 else None
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'🎯 Erweiterte IoT ZK-SNARK Crossover-Analyse\nCrossover-Punkt: ~{crossover_point} IoT Readings', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Zeit-Vergleich mit Crossover
    ax1.plot(readings, std_times, 'ro-', label='Standard ZoKrates', linewidth=3, markersize=8)
    ax1.plot(readings, nova_times, 'bs-', label='Nova Recursive', linewidth=3, markersize=8)
    
    # Highlight neue Datenpunkte
    new_points = [60, 70, 80, 90]
    for point in new_points:
        if point in readings:
            idx = list(readings).index(point)
            ax1.plot(point, std_times[idx], 'ro', markersize=12, markeredgecolor='black', markeredgewidth=2)
            ax1.plot(point, nova_times[idx], 'bs', markersize=12, markeredgecolor='black', markeredgewidth=2)
            
            # Add labels for new points
            ax1.text(point, std_times[idx] - 3, f'{point}', ha='center', fontweight='bold', fontsize=9)
    
    # Mark crossover point
    if crossover_point:
        ax1.axvline(x=crossover_point, color='green', linestyle='--', alpha=0.8, linewidth=3)
        ax1.text(crossover_point + 15, max(max(std_times), max(nova_times)) * 0.8, 
                f'Nova besser ab:\n{crossover_point} IoT Readings',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9))
    
    ax1.set_xlabel('Anzahl IoT Readings')
    ax1.set_ylabel('Gesamte Proof-Zeit (Sekunden)')
    ax1.set_title('⚡ Performance vs. IoT Reading Anzahl')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Nova Advantage Factor mit Crossover-Zone
    ax2.plot(readings, time_advantages, 'go-', linewidth=4, markersize=10)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    # Highlight crossover area
    crossover_zone_start = 80
    crossover_zone_end = 95
    ax2.axvspan(crossover_zone_start, crossover_zone_end, alpha=0.2, color='yellow', 
                label='Crossover Zone')
    
    if crossover_point:
        ax2.axvline(x=crossover_point, color='green', linestyle='--', alpha=0.8, linewidth=3)
    
    # Label advantages for new points
    for i, point in enumerate(new_points):
        if point in readings:
            idx = list(readings).index(point)
            advantage = time_advantages[idx]
            color = 'green' if advantage > 1.0 else 'red'
            ax2.plot(point, advantage, 'o', markersize=12, 
                    markerfacecolor=color, markeredgecolor='black', markeredgewidth=2)
            
            # Value label
            ax2.text(point, advantage + 0.05, f'{advantage:.2f}x', ha='center', 
                    fontweight='bold', fontsize=10)
    
    # Color zones
    ax2.fill_between(readings, time_advantages, 1, where=(time_advantages > 1), 
                    alpha=0.3, color='green', label='Nova Advantage')
    ax2.fill_between(readings, time_advantages, 1, where=(time_advantages < 1), 
                    alpha=0.3, color='red', label='Standard Advantage')
    
    ax2.set_xlabel('Anzahl IoT Readings')
    ax2.set_ylabel('Speedup Factor (Standard/Nova)')
    ax2.set_title('📈 Wann wird Nova besser? (>1 = Nova gewinnt)')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Proof Size Comparison
    ax3.plot(readings, data['standard_size'], 'ro-', label='Standard (Wächst linear)', linewidth=3, markersize=8)
    ax3.plot(readings, data['nova_size'], 'bs-', label='Nova (Konstant)', linewidth=3, markersize=8)
    
    # Highlight size advantage
    for point in new_points:
        if point in readings:
            idx = list(readings).index(point)
            size_advantage = data['standard_size'][idx] / data['nova_size'][idx]
            ax3.text(point, data['standard_size'][idx] + 50, f'{size_advantage:.1f}x', 
                    ha='center', fontweight='bold', fontsize=9, color='blue')
    
    ax3.set_xlabel('Anzahl IoT Readings')
    ax3.set_ylabel('Proof Größe (KB)')
    ax3.set_title('💾 Speicher-Effizienz: Nova konstant vs Standard linear')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Effizienz pro IoT Reading
    std_efficiency = [t/r for t, r in zip(std_times, readings)]
    nova_efficiency = [t/r for t, r in zip(nova_times, readings)]
    
    ax4.plot(readings, std_efficiency, 'ro-', label='Standard (~0.46s/Reading)', linewidth=3, markersize=8)
    ax4.plot(readings, nova_efficiency, 'bs-', label='Nova (Wird effizienter)', linewidth=3, markersize=8)
    
    # Mark efficiency crossover
    efficiency_crossover = None
    for i in range(len(readings)):
        if nova_efficiency[i] < std_efficiency[i]:
            efficiency_crossover = readings[i]
            break
    
    if efficiency_crossover:
        ax4.axvline(x=efficiency_crossover, color='purple', linestyle=':', alpha=0.7, linewidth=2)
        ax4.text(efficiency_crossover + 10, max(nova_efficiency) * 0.8, 
                f'Nova effizienter\nab {efficiency_crossover} Readings', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    ax4.set_xlabel('Anzahl IoT Readings')
    ax4.set_ylabel('Zeit pro IoT Reading (s/Reading)')
    ax4.set_title('⚡ Effizienz: Nova wird bei mehr Readings effizienter')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path("results_summary/extended_iot_crossover_visualization.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Erweiterte Crossover-Visualisierung gespeichert: {output_file}")
    
    # Show plot briefly then close
    plt.close()
    
    return output_file, crossover_point

def create_summary_table():
    """Erstelle Zusammenfassungstabelle mit Empfehlungen"""
    
    # Erweiterte Daten
    data = {
        'IoT Readings': [10, 25, 50, 60, 70, 80, 90, 100, 200],
        'Standard Zeit (s)': [4.57, 11.42, 22.85, 27.42, 31.99, 36.56, 41.13, 45.69, 91.38],
        'Nova Zeit (s)': [27.71, 30.84, 35.73, 37.08, 38.25, 39.31, 40.29, 43.08, 60.64],
        'Zeit Vorteil': ['0.2x', '0.4x', '0.6x', '0.7x', '0.8x', '0.9x', '1.0x', '1.1x', '1.5x'],
        'Gewinner': ['Standard', 'Standard', 'Standard', 'Standard', 'Standard', 'Standard', 'Nova', 'Nova', 'Nova'],
        'Empfehlung': ['Standard', 'Standard', 'Standard', 'Standard', 'Standard', 'Standard', 'Nova ✅', 'Nova', 'Nova']
    }
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_file = Path("results_summary/extended_iot_crossover_table.csv")
    df.to_csv(csv_file, index=False)
    print(f"✅ Erweiterte Crossover-Tabelle gespeichert: {csv_file}")
    
    return df, csv_file

if __name__ == "__main__":
    print("🚀 Erstelle erweiterte IoT Crossover-Analyse...")
    
    # Create visualization
    plot_file, crossover = create_extended_crossover_visualization()
    
    # Create table
    df, table_file = create_summary_table()
    
    print(f"\n🎯 ERGEBNISSE:")
    print(f"   Crossover-Punkt: {crossover} IoT Readings")
    print(f"   Empfehlung: Ab {crossover} IoT Readings auf Nova wechseln!")
    
    print(f"\n📁 Dateien erstellt:")
    print(f"   - {plot_file}")
    print(f"   - {table_file}")
    print(f"   - results_summary/precise_crossover_findings.md")
    
    print(f"\n🎉 Erweiterte Crossover-Analyse abgeschlossen!")
