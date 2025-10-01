"""
Device Key Management System für IoT-Sensoren
Verwaltet Proving/Verification Keys und Device-Authentifizierung
"""

import json
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class DeviceCredentials:
    """Device Credentials für IoT-Sensor"""
    device_id: str
    device_name: str
    device_type: str  # "sensor", "gateway", "controller"
    created_at: str
    expires_at: str
    is_active: bool = True
    
@dataclass
class DeviceKeyPair:
    """Device Key Pair für ZK-Proofs"""
    device_id: str
    proving_key: str      # Private Key für Proof-Generierung
    verification_key: str # Public Key für Proof-Verifikation
    signature_key: str    # HMAC Key für Device-Signatur
    created_at: str
    expires_at: str
    
@dataclass
class ZKCircuitConfig:
    """Konfiguration für ZK-Circuits basierend auf Privacy Level"""
    privacy_level: int
    circuit_name: str
    public_inputs: List[str]
    private_inputs: List[str]
    description: str

class IoTDeviceKeyManager:
    """Zentrales Key Management für IoT-Devices"""
    
    def __init__(self, keys_dir: str = "data/device_keys"):
        self.keys_dir = Path(keys_dir)
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        
        # Device Registries
        self.device_credentials: Dict[str, DeviceCredentials] = {}
        self.device_keys: Dict[str, DeviceKeyPair] = {}
        self.circuit_configs: Dict[int, ZKCircuitConfig] = {}
        
        # Lade existierende Daten
        self._load_existing_data()
        self._setup_circuit_configs()
    
    def _load_existing_data(self):
        """Lade existierende Device-Daten"""
        # Lade Device Credentials
        credentials_file = self.keys_dir / "device_credentials.json"
        if credentials_file.exists():
            try:
                with open(credentials_file, 'r') as f:
                    data = json.load(f)
                    for device_id, cred_data in data.items():
                        self.device_credentials[device_id] = DeviceCredentials(**cred_data)
            except Exception as e:
                logger.warning(f"Could not load device credentials: {e}")
        
        # Lade Device Keys
        for key_file in self.keys_dir.glob("device_keys_*.json"):
            try:
                with open(key_file, 'r') as f:
                    key_data = json.load(f)
                    device_key = DeviceKeyPair(**key_data)
                    self.device_keys[device_key.device_id] = device_key
            except Exception as e:
                logger.warning(f"Could not load key file {key_file}: {e}")
    
    def _setup_circuit_configs(self):
        """Setup ZK-Circuit-Konfigurationen für verschiedene Privacy Levels"""
        self.circuit_configs = {
            1: ZKCircuitConfig(
                privacy_level=1,
                circuit_name="aggregation",
                public_inputs=["sensor_id", "timestamp", "value_hash"],
                private_inputs=["value", "device_signature"],
                description="Low Privacy: Nur Aggregation (Sum, Count)"
            ),
            2: ZKCircuitConfig(
                privacy_level=2,
                circuit_name="range_verification",
                public_inputs=["sensor_id", "timestamp", "value_range_min", "value_range_max"],
                private_inputs=["value", "device_signature", "sensor_calibration"],
                description="Medium Privacy: Range-Verifikation + Aggregation"
            ),
            3: ZKCircuitConfig(
                privacy_level=3,
                circuit_name="full_verification",
                public_inputs=["sensor_id", "timestamp", "device_id", "proof_hash"],
                private_inputs=["value", "device_signature", "sensor_calibration", "environmental_data"],
                description="High Privacy: Vollständige Datenverifikation"
            )
        }
    
    def register_device(self, device_id: str, device_name: str, 
                       device_type: str = "sensor") -> DeviceCredentials:
        """Registriere neues IoT-Device"""
        now = datetime.now()
        expires_at = now + timedelta(days=365)  # 1 Jahr Gültigkeit
        
        credentials = DeviceCredentials(
            device_id=device_id,
            device_name=device_name,
            device_type=device_type,
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            is_active=True
        )
        
        self.device_credentials[device_id] = credentials
        self._save_credentials()
        
        logger.info(f"Registered device {device_id} ({device_name})")
        return credentials
    
    def generate_device_keys(self, device_id: str, privacy_level: int = 2) -> DeviceKeyPair:
        """Generiere Device Keys für ZK-Proofs"""
        if device_id not in self.device_credentials:
            raise ValueError(f"Device {device_id} not registered")
        
        # Generiere Keys
        proving_key = secrets.token_hex(32)  # 256-bit für Proving
        verification_key = hashlib.sha256(proving_key.encode()).hexdigest()[:32]  # Verkürzte Verification Key
        signature_key = secrets.token_hex(32)  # 256-bit für Device-Signatur
        
        now = datetime.now()
        expires_at = now + timedelta(days=365)
        
        device_key = DeviceKeyPair(
            device_id=device_id,
            proving_key=proving_key,
            verification_key=verification_key,
            signature_key=signature_key,
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat()
        )
        
        self.device_keys[device_id] = device_key
        self._save_device_keys(device_id)
        
        logger.info(f"Generated keys for device {device_id} (privacy level {privacy_level})")
        return device_key
    
    def get_device_keys(self, device_id: str) -> Optional[DeviceKeyPair]:
        """Hole Device Keys"""
        return self.device_keys.get(device_id)
    
    def get_circuit_config(self, privacy_level: int) -> Optional[ZKCircuitConfig]:
        """Hole ZK-Circuit-Konfiguration für Privacy Level"""
        return self.circuit_configs.get(privacy_level)
    
    def sign_data(self, device_id: str, data: str) -> str:
        """Signiere Daten mit Device Key"""
        device_key = self.get_device_keys(device_id)
        if not device_key:
            raise ValueError(f"No keys found for device {device_id}")
        
        signature = hmac.new(
            device_key.signature_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_signature(self, device_id: str, data: str, signature: str) -> bool:
        """Verifiziere Device-Signatur"""
        device_key = self.get_device_keys(device_id)
        if not device_key:
            return False
        
        expected_signature = hmac.new(
            device_key.signature_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def get_public_inputs_for_privacy_level(self, privacy_level: int, 
                                           sensor_data: Dict[str, any]) -> Dict[str, any]:
        """Bestimme Public Inputs basierend auf Privacy Level"""
        config = self.get_circuit_config(privacy_level)
        if not config:
            return {}
        
        public_inputs = {}
        
        if privacy_level == 1:  # Low Privacy
            public_inputs = {
                "sensor_id": sensor_data.get("sensor_id", ""),
                "timestamp": sensor_data.get("timestamp", ""),
                "value_hash": hashlib.sha256(str(sensor_data.get("value", 0)).encode()).hexdigest()[:16]
            }
        
        elif privacy_level == 2:  # Medium Privacy
            public_inputs = {
                "sensor_id": sensor_data.get("sensor_id", ""),
                "timestamp": sensor_data.get("timestamp", ""),
                "value_range_min": sensor_data.get("expected_min", 0),
                "value_range_max": sensor_data.get("expected_max", 100)
            }
        
        elif privacy_level == 3:  # High Privacy
            public_inputs = {
                "sensor_id": sensor_data.get("sensor_id", ""),
                "timestamp": sensor_data.get("timestamp", ""),
                "device_id": sensor_data.get("device_id", ""),
                "proof_hash": hashlib.sha256(
                    f"{sensor_data.get('sensor_id')}{sensor_data.get('timestamp')}".encode()
                ).hexdigest()[:16]
            }
        
        return public_inputs
    
    def get_private_inputs_for_privacy_level(self, privacy_level: int, 
                                            sensor_data: Dict[str, any]) -> Dict[str, any]:
        """Bestimme Private Inputs basierend auf Privacy Level"""
        config = self.get_circuit_config(privacy_level)
        if not config:
            return {}
        
        private_inputs = {
            "value": sensor_data.get("value", 0),
            "device_signature": sensor_data.get("device_signature", "")
        }
        
        if privacy_level >= 2:  # Medium/High Privacy
            private_inputs["sensor_calibration"] = sensor_data.get("calibration_factor", 1.0)
        
        if privacy_level >= 3:  # High Privacy
            private_inputs["environmental_data"] = sensor_data.get("environmental_context", "")
        
        return private_inputs
    
    def _save_credentials(self):
        """Speichere Device Credentials"""
        credentials_file = self.keys_dir / "device_credentials.json"
        credentials_data = {
            device_id: asdict(credentials) 
            for device_id, credentials in self.device_credentials.items()
        }
        
        with open(credentials_file, 'w') as f:
            json.dump(credentials_data, f, indent=2)
    
    def _save_device_keys(self, device_id: str):
        """Speichere Device Keys"""
        device_key = self.device_keys[device_id]
        key_file = self.keys_dir / f"device_keys_{device_id}.json"
        
        with open(key_file, 'w') as f:
            json.dump(asdict(device_key), f, indent=2)
    
    def list_devices(self) -> List[Dict[str, any]]:
        """Liste alle registrierten Devices"""
        devices = []
        for device_id, credentials in self.device_credentials.items():
            device_info = {
                "device_id": device_id,
                "device_name": credentials.device_name,
                "device_type": credentials.device_type,
                "is_active": credentials.is_active,
                "has_keys": device_id in self.device_keys,
                "created_at": credentials.created_at,
                "expires_at": credentials.expires_at
            }
            devices.append(device_info)
        
        return devices
    
    def revoke_device(self, device_id: str):
        """Revoke Device (deaktiviere)"""
        if device_id in self.device_credentials:
            self.device_credentials[device_id].is_active = False
            self._save_credentials()
            logger.info(f"Revoked device {device_id}")
    
    def cleanup_expired_keys(self):
        """Bereinige abgelaufene Keys"""
        now = datetime.now()
        expired_devices = []
        
        for device_id, credentials in self.device_credentials.items():
            if datetime.fromisoformat(credentials.expires_at) < now:
                expired_devices.append(device_id)
        
        for device_id in expired_devices:
            if device_id in self.device_credentials:
                del self.device_credentials[device_id]
            if device_id in self.device_keys:
                del self.device_keys[device_id]
        
        if expired_devices:
            self._save_credentials()
            logger.info(f"Cleaned up {len(expired_devices)} expired devices")

def create_test_iot_network() -> IoTDeviceKeyManager:
    """Erstelle Test-IoT-Netzwerk mit verschiedenen Devices"""
    
    key_manager = IoTDeviceKeyManager()
    
    # Registriere verschiedene Device-Typen
    devices = [
        ("TEMP_SENSOR_01", "Living Room Temperature", "sensor"),
        ("HUM_SENSOR_01", "Living Room Humidity", "sensor"),
        ("MOTION_SENSOR_01", "Living Room Motion", "sensor"),
        ("GATEWAY_01", "Main Gateway", "gateway"),
        ("CONTROLLER_01", "Smart Home Controller", "controller"),
    ]
    
    for device_id, device_name, device_type in devices:
        key_manager.register_device(device_id, device_name, device_type)
        
        # Generiere Keys für Sensoren
        if device_type == "sensor":
            privacy_level = 2 if "TEMP" in device_id or "HUM" in device_id else 3
            key_manager.generate_device_keys(device_id, privacy_level)
    
    logger.info(f"Created test IoT network with {len(devices)} devices")
    return key_manager

if __name__ == "__main__":
    # Test des Device Key Management Systems
    logging.basicConfig(level=logging.INFO)
    
    print("=== IoT Device Key Management Test ===")
    
    # Erstelle Test-Netzwerk
    key_manager = create_test_iot_network()
    
    # Liste Devices
    devices = key_manager.list_devices()
    print(f"\nRegistered devices ({len(devices)}):")
    for device in devices:
        print(f"  {device['device_id']}: {device['device_name']} ({device['device_type']}) - Keys: {device['has_keys']}")
    
    # Test Signatur
    test_device = "TEMP_SENSOR_01"
    test_data = "sensor_reading:25.5:2024-01-01T12:00:00"
    
    signature = key_manager.sign_data(test_device, test_data)
    print(f"\nTest signature for {test_device}: {signature[:16]}...")
    
    # Verifiziere Signatur
    is_valid = key_manager.verify_signature(test_device, test_data, signature)
    print(f"Signature verification: {'✓' if is_valid else '✗'}")
    
    # Test Privacy Level Konfigurationen
    print(f"\nPrivacy Level Configurations:")
    for level in [1, 2, 3]:
        config = key_manager.get_circuit_config(level)
        if config:
            print(f"  Level {level}: {config.description}")
            print(f"    Public inputs: {config.public_inputs}")
            print(f"    Private inputs: {config.private_inputs}")
