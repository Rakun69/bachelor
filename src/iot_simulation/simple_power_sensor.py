"""
Einfacher Power Sensor für Stromverbrauch
Vereinfachte Version ohne komplexe Privacy Levels
"""

import json
import time
import hashlib
import hmac
import secrets
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PowerReading:
    """Stromverbrauch-Reading"""
    device_id: str
    timestamp: str
    kwh: float
    signature: str
    zk_proof: Optional[str] = None

class SimplePowerSensor:
    """Einfacher IoT-Sensor für Stromverbrauch"""
    
    def __init__(self, device_id: str, gateway_url: str = "http://localhost:8888"):
        self.device_id = device_id
        self.gateway_url = gateway_url
        self.private_key = self._generate_device_key()
        self.last_reading_time = None
        
        logger.info(f"Initialized power sensor {device_id}")
    
    def _generate_device_key(self) -> str:
        """Generiere einfachen Device Key"""
        return secrets.token_hex(32)  # 256-bit HMAC Key
    
    def _sign_data(self, data: str) -> str:
        """Signiere Daten mit Device Key"""
        return hmac.new(
            self.private_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _generate_simple_zk_proof(self, kwh: float) -> str:
        """Generiere einfachen ZK-Proof für kWh-Wert"""
        # Einfacher Proof: Wert liegt in erlaubtem Bereich (0-100 kWh)
        # In der Praxis würde hier ein ZoKrates Circuit laufen
        
        # Für Demo: Simuliere Proof
        proof_data = {
            "kwh": kwh,
            "range_min": 0,
            "range_max": 100,
            "proof_hash": hashlib.sha256(f"{kwh}:{self.device_id}".encode()).hexdigest()
        }
        
        return json.dumps(proof_data)
    
    def generate_reading(self, kwh: float, timestamp: str = None) -> PowerReading:
        """Generiere Power-Reading mit Signatur und ZK-Proof"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Erstelle Signatur
        message_data = f"{self.device_id}:{timestamp}:{kwh}"
        signature = self._sign_data(message_data)
        
        # Generiere ZK-Proof
        zk_proof = self._generate_simple_zk_proof(kwh)
        
        reading = PowerReading(
            device_id=self.device_id,
            timestamp=timestamp,
            kwh=kwh,
            signature=signature,
            zk_proof=zk_proof
        )
        
        self.last_reading_time = timestamp
        logger.debug(f"Generated reading: {kwh} kWh at {timestamp}")
        
        return reading
    
    def send_reading(self, reading: PowerReading) -> bool:
        """Sende Reading an Gateway"""
        try:
            # Erstelle Nachricht
            message = {
                "device_id": reading.device_id,
                "timestamp": reading.timestamp,
                "kwh": reading.kwh,
                "signature": reading.signature,
                "zk_proof": reading.zk_proof
            }
            
            # Sende an Gateway
            response = requests.post(
                f"{self.gateway_url}/power-data",
                json=message,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully sent {reading.kwh} kWh to gateway")
                return True
            else:
                logger.warning(f"Gateway returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send reading: {e}")
            return False
    
    def send_hourly_consumption(self, kwh: float) -> bool:
        """Sende stündlichen Verbrauch (Hauptfunktion)"""
        reading = self.generate_reading(kwh)
        return self.send_reading(reading)
    
    def simulate_hourly_consumption(self, base_consumption: float = 2.5) -> float:
        """Simuliere realistischen stündlichen Verbrauch"""
        current_hour = datetime.now().hour
        
        # Realistische Verbrauchsmuster
        if 6 <= current_hour <= 8:  # Morgen
            multiplier = 1.5  # Kaffeemaschine, Toaster
        elif 12 <= current_hour <= 14:  # Mittag
            multiplier = 1.2  # Mikrowelle, Kühlschrank
        elif 18 <= current_hour <= 20:  # Abend
            multiplier = 2.0  # Herd, Ofen, TV
        elif 22 <= current_hour or current_hour <= 6:  # Nacht
            multiplier = 0.3  # Nur Grundverbrauch
        else:
            multiplier = 1.0  # Normal
        
        # Zufällige Variation
        variation = (secrets.randbelow(21) - 10) / 100  # ±10%
        
        kwh = base_consumption * multiplier * (1 + variation)
        return round(kwh, 2)

class SimplePowerGateway:
    """Einfaches Gateway für Power-Daten"""
    
    def __init__(self, port: int = 8888):
        self.port = port
        self.received_readings = []
        self.device_keys = {}  # device_id -> private_key
        
    def register_device(self, device_id: str, private_key: str):
        """Registriere Device"""
        self.device_keys[device_id] = private_key
        logger.info(f"Registered device {device_id}")
    
    def verify_signature(self, device_id: str, data: str, signature: str) -> bool:
        """Verifiziere Device-Signatur"""
        if device_id not in self.device_keys:
            return False
        
        expected_signature = hmac.new(
            self.device_keys[device_id].encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def receive_power_data(self, message: Dict[str, Any]) -> bool:
        """Empfange Power-Daten"""
        device_id = message["device_id"]
        timestamp = message["timestamp"]
        kwh = message["kwh"]
        signature = message["signature"]
        
        # Verifiziere Signatur
        message_data = f"{device_id}:{timestamp}:{kwh}"
        if not self.verify_signature(device_id, message_data, signature):
            logger.warning(f"Invalid signature for device {device_id}")
            return False
        
        # Speichere Reading
        reading = PowerReading(
            device_id=device_id,
            timestamp=timestamp,
            kwh=kwh,
            signature=signature,
            zk_proof=message.get("zk_proof")
        )
        
        self.received_readings.append(reading)
        logger.info(f"Received {kwh} kWh from {device_id}")
        return True
    
    def get_daily_consumption(self, device_id: str, date: str = None) -> float:
        """Berechne Tagesverbrauch für Device"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        daily_readings = [
            r for r in self.received_readings 
            if r.device_id == device_id and r.timestamp.startswith(date)
        ]
        
        return sum(r.kwh for r in daily_readings)
    
    def get_monthly_consumption(self, device_id: str, month: str = None) -> float:
        """Berechne Monatsverbrauch für Device"""
        if month is None:
            month = datetime.now().strftime("%Y-%m")
        
        monthly_readings = [
            r for r in self.received_readings 
            if r.device_id == device_id and r.timestamp.startswith(month)
        ]
        
        return sum(r.kwh for r in monthly_readings)

def create_simple_power_system():
    """Erstelle einfaches Power-System"""
    
    # Erstelle Gateway
    gateway = SimplePowerGateway()
    
    # Erstelle Power Sensor
    sensor = SimplePowerSensor("POWER_DEVICE_001")
    
    # Registriere Device am Gateway
    gateway.register_device(sensor.device_id, sensor.private_key)
    
    # Simuliere 24 Stunden Verbrauch
    print("=== Simple Power Sensor System ===")
    print("Simulating 24 hours of power consumption...")
    
    total_consumption = 0
    
    for hour in range(24):
        # Simuliere stündlichen Verbrauch
        kwh = sensor.simulate_hourly_consumption()
        
        # Sende an Gateway
        reading = sensor.generate_reading(kwh)
        success = gateway.receive_power_data({
            "device_id": reading.device_id,
            "timestamp": reading.timestamp,
            "kwh": reading.kwh,
            "signature": reading.signature,
            "zk_proof": reading.zk_proof
        })
        
        if success:
            total_consumption += kwh
            print(f"Hour {hour:2d}: {kwh:5.2f} kWh {'✅' if success else '❌'}")
        else:
            print(f"Hour {hour:2d}: {kwh:5.2f} kWh ❌")
    
    # Zeige Ergebnisse
    print(f"\n📊 Results:")
    print(f"Total consumption: {total_consumption:.2f} kWh")
    print(f"Average per hour: {total_consumption/24:.2f} kWh")
    print(f"Received readings: {len(gateway.received_readings)}")
    
    # Tagesverbrauch
    daily_consumption = gateway.get_daily_consumption("POWER_DEVICE_001")
    print(f"Daily consumption: {daily_consumption:.2f} kWh")
    
    return gateway, sensor

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_simple_power_system()
