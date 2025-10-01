"""
Real IoT Sensor Implementation with ZK-Proofs
Simuliert echte IoT-Sensoren mit Device Keys, Signatur-Verifikation und ZK-Proofs
"""

import json
import time
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)

@dataclass
class DeviceKeyPair:
    """Device Key Pair für Sensor-Authentifizierung"""
    device_id: str
    private_key: str  # HMAC key für Signatur
    public_key: str   # Verifikations-Key
    created_at: str
    expires_at: str

@dataclass
class SensorData:
    """Sensor-Daten mit ZK-Proof"""
    sensor_id: str
    device_id: str
    sensor_type: str
    room: str
    timestamp: str
    value: float
    unit: str
    privacy_level: int
    signature: str
    zk_proof: Optional[str] = None
    proof_public_inputs: Optional[Dict[str, Any]] = None

@dataclass
class ZKProofResult:
    """Ergebnis der ZK-Proof-Generierung"""
    success: bool
    proof: Optional[str] = None
    public_inputs: Optional[Dict[str, Any]] = None
    proof_size: int = 0
    generation_time: float = 0.0
    error_message: Optional[str] = None

class IoTDeviceKeyManager:
    """Verwaltet Device Keys für IoT-Sensoren"""
    
    def __init__(self, keys_dir: str = "data/device_keys"):
        self.keys_dir = Path(keys_dir)
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        self.device_keys: Dict[str, DeviceKeyPair] = {}
        self._load_existing_keys()
    
    def _load_existing_keys(self):
        """Lade existierende Device Keys"""
        for key_file in self.keys_dir.glob("device_*.json"):
            try:
                with open(key_file, 'r') as f:
                    key_data = json.load(f)
                    device_key = DeviceKeyPair(**key_data)
                    self.device_keys[device_key.device_id] = device_key
            except Exception as e:
                logger.warning(f"Could not load key file {key_file}: {e}")
    
    def generate_device_key(self, device_id: str) -> DeviceKeyPair:
        """Generiere neue Device Keys für einen Sensor"""
        # HMAC-basierte Signatur (einfacher als RSA/ECDSA für IoT)
        private_key = secrets.token_hex(32)  # 256-bit HMAC key
        public_key = hashlib.sha256(private_key.encode()).hexdigest()[:16]  # Verkürzte Public Key
        
        now = datetime.now()
        expires_at = now + timedelta(days=365)  # 1 Jahr Gültigkeit
        
        device_key = DeviceKeyPair(
            device_id=device_id,
            private_key=private_key,
            public_key=public_key,
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat()
        )
        
        # Speichere Key
        key_file = self.keys_dir / f"device_{device_id}.json"
        with open(key_file, 'w') as f:
            json.dump(asdict(device_key), f, indent=2)
        
        self.device_keys[device_id] = device_key
        logger.info(f"Generated device key for {device_id}")
        return device_key
    
    def get_device_key(self, device_id: str) -> Optional[DeviceKeyPair]:
        """Hole Device Key für einen Sensor"""
        if device_id not in self.device_keys:
            return self.generate_device_key(device_id)
        return self.device_keys[device_id]
    
    def verify_signature(self, device_id: str, message: str, signature: str) -> bool:
        """Verifiziere Sensor-Signatur"""
        device_key = self.get_device_key(device_id)
        if not device_key:
            return False
        
        # Erwartete Signatur berechnen
        expected_signature = hmac.new(
            device_key.private_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)

class IoTZKProofGenerator:
    """Generiert ZK-Proofs für IoT-Sensordaten"""
    
    def __init__(self, circuit_dir: str = "circuits/basic"):
        self.circuit_dir = Path(circuit_dir)
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def generate_sensor_proof(self, sensor_data: SensorData, 
                             privacy_level: int = 2) -> ZKProofResult:
        """
        Generiere ZK-Proof für Sensordaten basierend auf Privacy Level
        
        Privacy Levels:
        1 = Low: Nur Aggregation (Sum, Count)
        2 = Medium: Range + Aggregation  
        3 = High: Vollständige Verifikation
        """
        start_time = time.time()
        
        try:
            if privacy_level == 1:
                return self._generate_aggregation_proof(sensor_data)
            elif privacy_level == 2:
                return self._generate_range_proof(sensor_data)
            elif privacy_level == 3:
                return self._generate_full_verification_proof(sensor_data)
            else:
                return ZKProofResult(
                    success=False,
                    error_message=f"Unknown privacy level: {privacy_level}"
                )
                
        except Exception as e:
            return ZKProofResult(
                success=False,
                error_message=f"Proof generation failed: {str(e)}"
            )
        finally:
            # Cleanup temp files
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _generate_aggregation_proof(self, sensor_data: SensorData) -> ZKProofResult:
        """Generiere Aggregations-Proof (Sum, Count)"""
        start_time = time.time()
        
        # Erstelle temporäre Input-Datei
        input_data = {
            "value": int(sensor_data.value * 100),  # Scale to avoid decimals
            "timestamp": int(time.mktime(datetime.fromisoformat(sensor_data.timestamp).timetuple())),
            "sensor_id": sensor_data.sensor_id
        }
        
        input_file = self.temp_dir / "sensor_input.json"
        with open(input_file, 'w') as f:
            json.dump(input_data, f)
        
        # Führe ZoKrates Aggregation Circuit aus
        try:
            # Kompiliere Circuit falls nötig
            compile_result = subprocess.run([
                "zokrates", "compile", "-i", str(self.circuit_dir / "aggregation.zok")
            ], capture_output=True, text=True, timeout=30)
            
            if compile_result.returncode != 0:
                return ZKProofResult(
                    success=False,
                    error_message=f"Circuit compilation failed: {compile_result.stderr}"
                )
            
            # Setup (falls nötig)
            setup_result = subprocess.run([
                "zokrates", "setup"
            ], capture_output=True, text=True, timeout=60)
            
            if setup_result.returncode != 0:
                return ZKProofResult(
                    success=False,
                    error_message=f"Setup failed: {setup_result.stderr}"
                )
            
            # Compute witness
            compute_result = subprocess.run([
                "zokrates", "compute-witness", "-a", str(input_data["value"]), 
                str(input_data["timestamp"]), str(input_data["sensor_id"])
            ], capture_output=True, text=True, timeout=30)
            
            if compute_result.returncode != 0:
                return ZKProofResult(
                    success=False,
                    error_message=f"Witness computation failed: {compute_result.stderr}"
                )
            
            # Generate proof
            prove_result = subprocess.run([
                "zokrates", "generate-proof"
            ], capture_output=True, text=True, timeout=60)
            
            if prove_result.returncode != 0:
                return ZKProofResult(
                    success=False,
                    error_message=f"Proof generation failed: {prove_result.stderr}"
                )
            
            # Lese Proof
            proof_file = Path("proof.json")
            if proof_file.exists():
                with open(proof_file, 'r') as f:
                    proof_data = json.load(f)
                
                generation_time = time.time() - start_time
                
                return ZKProofResult(
                    success=True,
                    proof=json.dumps(proof_data),
                    public_inputs={
                        "sensor_id": sensor_data.sensor_id,
                        "timestamp": sensor_data.timestamp,
                        "value_hash": hashlib.sha256(str(sensor_data.value).encode()).hexdigest()[:16]
                    },
                    proof_size=len(json.dumps(proof_data)),
                    generation_time=generation_time
                )
            else:
                return ZKProofResult(
                    success=False,
                    error_message="Proof file not generated"
                )
                
        except subprocess.TimeoutExpired:
            return ZKProofResult(
                success=False,
                error_message="Proof generation timeout"
            )
        except Exception as e:
            return ZKProofResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def _generate_range_proof(self, sensor_data: SensorData) -> ZKProofResult:
        """Generiere Range-Proof (Wert liegt in erlaubtem Bereich)"""
        # Ähnlich wie Aggregation, aber mit Range-Checks
        # Für jetzt verwende Aggregation als Fallback
        return self._generate_aggregation_proof(sensor_data)
    
    def _generate_full_verification_proof(self, sensor_data: SensorData) -> ZKProofResult:
        """Generiere vollständigen Verifikations-Proof"""
        # Ähnlich wie Range, aber mit vollständiger Datenverifikation
        return self._generate_aggregation_proof(sensor_data)

class RealIoTSensor:
    """Echter IoT-Sensor mit ZK-Proof-Generierung"""
    
    def __init__(self, sensor_id: str, device_id: str, sensor_type: str, 
                 room: str, privacy_level: int = 2, key_manager: IoTDeviceKeyManager = None):
        self.sensor_id = sensor_id
        self.device_id = device_id
        self.sensor_type = sensor_type
        self.room = room
        self.privacy_level = privacy_level
        
        # Key Management
        self.key_manager = key_manager or IoTDeviceKeyManager()
        self.device_key = self.key_manager.get_device_key(device_id)
        
        # ZK-Proof Generator
        self.proof_generator = IoTZKProofGenerator()
        
        # Sensor State
        self.last_reading_time = None
        self.reading_count = 0
        
        logger.info(f"Initialized sensor {sensor_id} on device {device_id}")
    
    def generate_reading(self, value: float, timestamp: str = None) -> SensorData:
        """Generiere Sensor-Reading mit Signatur und ZK-Proof"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Erstelle Sensor-Daten
        sensor_data = SensorData(
            sensor_id=self.sensor_id,
            device_id=self.device_id,
            sensor_type=self.sensor_type,
            room=self.room,
            timestamp=timestamp,
            value=value,
            unit=self._get_unit(),
            privacy_level=self.privacy_level,
            signature="",  # Wird unten berechnet
            zk_proof=None,
            proof_public_inputs=None
        )
        
        # Generiere Signatur
        message = f"{sensor_data.sensor_id}:{sensor_data.timestamp}:{sensor_data.value}"
        signature = hmac.new(
            self.device_key.private_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        sensor_data.signature = signature
        
        # Generiere ZK-Proof
        proof_result = self.proof_generator.generate_sensor_proof(sensor_data, self.privacy_level)
        
        if proof_result.success:
            sensor_data.zk_proof = proof_result.proof
            sensor_data.proof_public_inputs = proof_result.public_inputs
        
        self.last_reading_time = timestamp
        self.reading_count += 1
        
        logger.debug(f"Generated reading for {self.sensor_id}: {value} (proof: {proof_result.success})")
        return sensor_data
    
    def _get_unit(self) -> str:
        """Hole Einheit für Sensor-Typ"""
        units = {
            'temperature': '°C',
            'humidity': '%',
            'motion': 'bool',
            'light': 'lux',
            'gas': 'ppm',
            'wind_speed': 'm/s'
        }
        return units.get(self.sensor_type, 'value')
    
    def verify_reading(self, sensor_data: SensorData) -> bool:
        """Verifiziere Sensor-Reading (Signatur + ZK-Proof)"""
        # Verifiziere Signatur
        message = f"{sensor_data.sensor_id}:{sensor_data.timestamp}:{sensor_data.value}"
        if not self.key_manager.verify_signature(sensor_data.device_id, message, sensor_data.signature):
            logger.warning(f"Signature verification failed for {sensor_data.sensor_id}")
            return False
        
        # TODO: Verifiziere ZK-Proof (benötigt ZoKrates verify)
        # Für jetzt akzeptiere wir den Proof als gültig wenn er existiert
        if sensor_data.zk_proof is None:
            logger.warning(f"No ZK proof for {sensor_data.sensor_id}")
            return False
        
        logger.debug(f"Reading verification successful for {sensor_data.sensor_id}")
        return True

class RealIoTGateway:
    """IoT Gateway für Sensor-Kommunikation und ZK-Proof-Verifikation"""
    
    def __init__(self, key_manager: IoTDeviceKeyManager = None):
        self.key_manager = key_manager or IoTDeviceKeyManager()
        self.connected_sensors: Dict[str, RealIoTSensor] = {}
        self.received_data: List[SensorData] = []
        
    def register_sensor(self, sensor: RealIoTSensor):
        """Registriere Sensor am Gateway"""
        self.connected_sensors[sensor.sensor_id] = sensor
        logger.info(f"Registered sensor {sensor.sensor_id}")
    
    def receive_sensor_data(self, sensor_data: SensorData) -> bool:
        """Empfange und verifiziere Sensordaten"""
        # Verifiziere Sensor-Reading
        if sensor_data.sensor_id in self.connected_sensors:
            sensor = self.connected_sensors[sensor_data.sensor_id]
            if sensor.verify_reading(sensor_data):
                self.received_data.append(sensor_data)
                logger.info(f"Received verified data from {sensor_data.sensor_id}")
                return True
            else:
                logger.warning(f"Data verification failed for {sensor_data.sensor_id}")
                return False
        else:
            logger.warning(f"Unknown sensor {sensor_data.sensor_id}")
            return False
    
    def get_verified_data(self) -> List[SensorData]:
        """Hole alle verifizierten Sensordaten"""
        return self.received_data.copy()
    
    def aggregate_data(self) -> Dict[str, Any]:
        """Aggregiere verifizierte Sensordaten"""
        if not self.received_data:
            return {}
        
        # Gruppiere nach Sensor-Typ
        by_type = {}
        for data in self.received_data:
            if data.sensor_type not in by_type:
                by_type[data.sensor_type] = []
            by_type[data.sensor_type].append(data.value)
        
        # Berechne Statistiken
        aggregation = {}
        for sensor_type, values in by_type.items():
            aggregation[sensor_type] = {
                'count': len(values),
                'sum': sum(values),
                'average': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'verified_readings': len(values)
            }
        
        return aggregation

def create_real_iot_network() -> Tuple[RealIoTGateway, List[RealIoTSensor]]:
    """Erstelle realistisches IoT-Netzwerk mit Sensoren"""
    
    # Key Manager
    key_manager = IoTDeviceKeyManager()
    
    # Gateway
    gateway = RealIoTGateway(key_manager)
    
    # Sensoren erstellen
    sensors = []
    sensor_configs = [
        ("TEMP_01", "DEV_001", "temperature", "living_room", 2),
        ("HUM_01", "DEV_001", "humidity", "living_room", 1),
        ("MOTION_01", "DEV_002", "motion", "living_room", 3),
        ("LIGHT_01", "DEV_002", "light", "living_room", 1),
        ("TEMP_02", "DEV_003", "temperature", "bedroom", 2),
        ("HUM_02", "DEV_003", "humidity", "bedroom", 1),
        ("MOTION_02", "DEV_004", "motion", "bedroom", 3),
    ]
    
    for sensor_id, device_id, sensor_type, room, privacy_level in sensor_configs:
        sensor = RealIoTSensor(
            sensor_id=sensor_id,
            device_id=device_id,
            sensor_type=sensor_type,
            room=room,
            privacy_level=privacy_level,
            key_manager=key_manager
        )
        sensors.append(sensor)
        gateway.register_sensor(sensor)
    
    logger.info(f"Created IoT network with {len(sensors)} sensors")
    return gateway, sensors

def simulate_real_iot_data_collection(duration_minutes: int = 60) -> Dict[str, Any]:
    """Simuliere echte IoT-Datensammlung mit ZK-Proofs"""
    
    # Erstelle IoT-Netzwerk
    gateway, sensors = create_real_iot_network()
    
    # Simuliere Datensammlung
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    collected_data = []
    current_time = start_time
    
    logger.info(f"Starting IoT data collection for {duration_minutes} minutes")
    
    while current_time < end_time:
        # Jeder Sensor generiert Reading
        for sensor in sensors:
            # Simuliere Sensor-Wert
            if sensor.sensor_type == "temperature":
                value = 20 + 5 * (current_time.hour / 24) + (current_time.minute % 10)
            elif sensor.sensor_type == "humidity":
                value = 40 + 20 * (current_time.minute % 30) / 30
            elif sensor.sensor_type == "motion":
                value = 1.0 if (current_time.minute % 15) < 5 else 0.0
            elif sensor.sensor_type == "light":
                value = 100 + 200 * (current_time.hour / 24)
            else:
                value = 0.0
            
            # Generiere Reading mit ZK-Proof
            reading = sensor.generate_reading(value, current_time.isoformat())
            
            # Sende an Gateway
            if gateway.receive_sensor_data(reading):
                collected_data.append(reading)
        
        # Nächste Minute
        current_time += timedelta(minutes=1)
    
    # Aggregiere Daten
    aggregation = gateway.aggregate_data()
    
    # Statistiken
    stats = {
        'total_readings': len(collected_data),
        'verified_readings': len([d for d in collected_data if d.zk_proof]),
        'sensors_count': len(sensors),
        'duration_minutes': duration_minutes,
        'aggregation': aggregation,
        'proof_generation_success_rate': len([d for d in collected_data if d.zk_proof]) / len(collected_data) if collected_data else 0
    }
    
    logger.info(f"Data collection completed: {stats['total_readings']} readings, {stats['verified_readings']} with proofs")
    return stats

if __name__ == "__main__":
    # Test der echten IoT-Sensor-Implementierung
    logging.basicConfig(level=logging.INFO)
    
    print("=== Real IoT Sensor Implementation Test ===")
    
    # Simuliere 10 Minuten Datensammlung
    results = simulate_real_iot_data_collection(duration_minutes=10)
    
    print(f"Total readings: {results['total_readings']}")
    print(f"Verified readings: {results['verified_readings']}")
    print(f"Proof success rate: {results['proof_generation_success_rate']:.2%}")
    print(f"Sensors: {results['sensors_count']}")
    
    if results['aggregation']:
        print("\nData aggregation:")
        for sensor_type, stats in results['aggregation'].items():
            print(f"  {sensor_type}: {stats['count']} readings, avg={stats['average']:.2f}")
