#!/usr/bin/env python3
"""
Script zum Ausführen des echten IoT-Systems
Kombiniert alle Komponenten: Sensoren, Keys, Kommunikation, Integrity, Privacy
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Füge src zum Python-Pfad hinzu
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iot_simulation.integrated_iot_system import IntegratedIoTSystem

def setup_logging(level: str = "INFO"):
    """Setup Logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('iot_system.log')
        ]
    )

def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description="Run Real IoT System with ZK-Proofs")
    parser.add_argument("--duration", type=int, default=10, 
                       help="Duration in minutes (default: 10)")
    parser.add_argument("--sensors", type=int, default=15,
                       help="Number of sensors to create (default: 15)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    parser.add_argument("--output-dir", type=str, default="data/real_iot_results",
                       help="Output directory for results (default: data/real_iot_results)")
    
    args = parser.parse_args()
    
    # Setup Logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Starting Real IoT System ===")
    logger.info(f"Duration: {args.duration} minutes")
    logger.info(f"Sensors: {args.sensors}")
    logger.info(f"Output: {args.output_dir}")
    
    try:
        # Erstelle Output-Verzeichnis
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Erstelle IoT-System
        iot_system = IntegratedIoTSystem("Real_IoT_System")
        
        # Führe Test durch
        results = iot_system.run_comprehensive_test(duration_minutes=args.duration)
        
        # Speichere Ergebnisse
        results_file = output_dir / "iot_system_results.json"
        with open(results_file, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        
        # Zeige Ergebnisse
        if results["success"]:
            logger.info("✅ IoT System Test Successful!")
            logger.info(f"📊 Total readings: {results['system_status']['system_stats']['total_readings']}")
            logger.info(f"🔒 Verified readings: {results['system_status']['system_stats']['verified_readings']}")
            logger.info(f"⚡ Proof success rate: {results['performance_metrics']['proof_success_rate']:.2%}")
            logger.info(f"🔐 Verification success rate: {results['performance_metrics']['verification_success_rate']:.2%}")
            logger.info(f"🛡️ Privacy violations: {results['performance_metrics']['privacy_violations']}")
            logger.info(f"📡 Communication errors: {results['performance_metrics']['communication_errors']}")
            logger.info(f"📈 Readings per minute: {results['performance_metrics']['readings_per_minute']:.1f}")
            
            # Zeige Sensor-Details
            logger.info("\n🏠 Sensor Details:")
            for sensor_id, data in results['data_summary']['sensors_data'].items():
                logger.info(f"  {sensor_id}: {data['count']} readings, "
                          f"avg={data.get('average', 0):.1f}, "
                          f"proof_rate={data.get('proof_rate', 0):.2%}")
            
            logger.info(f"\n📁 Results saved to: {results_file}")
        else:
            logger.error(f"❌ IoT System Test Failed: {results.get('error', 'Unknown error')}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ System Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
