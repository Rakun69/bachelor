#!/usr/bin/env python3
"""
Einfacher Test des IoT-Systems
Vereinfachte Version ohne komplexe Integration
"""

import sys
import os
from pathlib import Path

# Füge src zum Python-Pfad hinzu
sys.path.insert(0, str(Path(__file__).parent / "src"))

from iot_simulation.simple_power_sensor import create_simple_power_system

def main():
    """Teste einfaches IoT-System"""
    print("=== Test Simple IoT System ===")
    
    try:
        # Erstelle einfaches Power System
        gateway, sensor = create_simple_power_system()
        
        print("✅ Simple IoT System Test Successful!")
        print(f"📊 Total readings: {len(gateway.received_readings)}")
        print(f"🔒 All readings verified: {all(r.signature for r in gateway.received_readings)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
