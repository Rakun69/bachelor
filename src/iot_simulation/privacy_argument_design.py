"""
Privacy Argument Design für IoT-ZK-Proofs
Definiert welche Argumente public/private sind basierend auf Privacy Levels
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class PrivacyArgument:
    """Einzelnes Privacy Argument"""
    argument_name: str
    argument_type: str  # "public", "private", "witness"
    data_type: str  # "string", "number", "boolean", "hash"
    description: str
    privacy_sensitivity: int  # 1=low, 2=medium, 3=high
    required_for_levels: List[int]  # Privacy Levels die dieses Argument benötigen

@dataclass
class PrivacyLevelConfig:
    """Konfiguration für Privacy Level"""
    level: int
    name: str
    description: str
    public_arguments: List[str]
    private_arguments: List[str]
    witness_arguments: List[str]
    circuit_name: str
    use_cases: List[str]

@dataclass
class SensorPrivacyProfile:
    """Privacy-Profil für einen Sensor"""
    sensor_id: str
    sensor_type: str
    room: str
    privacy_level: int
    allowed_public_args: Set[str]
    allowed_private_args: Set[str]
    restricted_args: Set[str]
    data_retention_days: int
    anonymization_required: bool

class IoTPrivacyArgumentDesigner:
    """Designer für Privacy Arguments in IoT-ZK-Proofs"""
    
    def __init__(self):
        self.privacy_arguments: Dict[str, PrivacyArgument] = {}
        self.privacy_levels: Dict[int, PrivacyLevelConfig] = {}
        self.sensor_profiles: Dict[str, SensorPrivacyProfile] = {}
        
        # Initialisiere Standard-Argumente
        self._setup_standard_arguments()
        self._setup_privacy_levels()
    
    def _setup_standard_arguments(self):
        """Setup Standard Privacy Arguments"""
        
        # === PUBLIC ARGUMENTS (Sichtbar für alle) ===
        
        self.privacy_arguments["sensor_id"] = PrivacyArgument(
            argument_name="sensor_id",
            argument_type="public",
            data_type="string",
            description="Eindeutige Sensor-ID (nicht personenbezogen)",
            privacy_sensitivity=1,
            required_for_levels=[1, 2, 3]
        )
        
        self.privacy_arguments["timestamp"] = PrivacyArgument(
            argument_name="timestamp",
            argument_type="public",
            data_type="string",
            description="Zeitstempel der Messung (kann anonymisiert werden)",
            privacy_sensitivity=2,
            required_for_levels=[1, 2, 3]
        )
        
        self.privacy_arguments["device_id"] = PrivacyArgument(
            argument_name="device_id",
            argument_type="public",
            data_type="string",
            description="Device-ID (kann anonymisiert werden)",
            privacy_sensitivity=2,
            required_for_levels=[2, 3]
        )
        
        self.privacy_arguments["value_hash"] = PrivacyArgument(
            argument_name="value_hash",
            argument_type="public",
            data_type="hash",
            description="Hash des Messwerts (nicht rückverfolgbar)",
            privacy_sensitivity=1,
            required_for_levels=[1, 2, 3]
        )
        
        self.privacy_arguments["proof_hash"] = PrivacyArgument(
            argument_name="proof_hash",
            argument_type="public",
            data_type="hash",
            description="Hash des ZK-Proofs (Integrität)",
            privacy_sensitivity=1,
            required_for_levels=[2, 3]
        )
        
        self.privacy_arguments["room"] = PrivacyArgument(
            argument_name="room",
            argument_type="public",
            data_type="string",
            description="Raum-Information (kann generalisiert werden)",
            privacy_sensitivity=2,
            required_for_levels=[1, 2, 3]
        )
        
        # === PRIVATE ARGUMENTS (Nur in ZK-Proof) ===
        
        self.privacy_arguments["actual_value"] = PrivacyArgument(
            argument_name="actual_value",
            argument_type="private",
            data_type="number",
            description="Tatsächlicher Messwert (hoch sensibel)",
            privacy_sensitivity=3,
            required_for_levels=[1, 2, 3]
        )
        
        self.privacy_arguments["device_signature"] = PrivacyArgument(
            argument_name="device_signature",
            argument_type="private",
            data_type="string",
            description="Device-Signatur (Authentifizierung)",
            privacy_sensitivity=2,
            required_for_levels=[1, 2, 3]
        )
        
        self.privacy_arguments["sensor_calibration"] = PrivacyArgument(
            argument_name="sensor_calibration",
            argument_type="private",
            data_type="number",
            description="Sensor-Kalibrierung (technisch sensibel)",
            privacy_sensitivity=2,
            required_for_levels=[2, 3]
        )
        
        self.privacy_arguments["environmental_context"] = PrivacyArgument(
            argument_name="environmental_context",
            argument_type="private",
            data_type="string",
            description="Umgebungskontext (privat)",
            privacy_sensitivity=3,
            required_for_levels=[3]
        )
        
        self.privacy_arguments["user_presence"] = PrivacyArgument(
            argument_name="user_presence",
            argument_type="private",
            data_type="boolean",
            description="Benutzer-Anwesenheit (hoch sensibel)",
            privacy_sensitivity=3,
            required_for_levels=[3]
        )
        
        # === WITNESS ARGUMENTS (Interne Berechnungen) ===
        
        self.privacy_arguments["computation_hash"] = PrivacyArgument(
            argument_name="computation_hash",
            argument_type="witness",
            data_type="hash",
            description="Hash der Berechnung (Integrität)",
            privacy_sensitivity=1,
            required_for_levels=[1, 2, 3]
        )
        
        self.privacy_arguments["range_validation"] = PrivacyArgument(
            argument_name="range_validation",
            argument_type="witness",
            data_type="boolean",
            description="Bereichs-Validierung (interne Logik)",
            privacy_sensitivity=1,
            required_for_levels=[2, 3]
        )
        
        self.privacy_arguments["aggregation_result"] = PrivacyArgument(
            argument_name="aggregation_result",
            argument_type="witness",
            data_type="number",
            description="Aggregations-Ergebnis (interne Berechnung)",
            privacy_sensitivity=1,
            required_for_levels=[1, 2, 3]
        )
    
    def _setup_privacy_levels(self):
        """Setup Privacy Level Konfigurationen"""
        
        # Level 1: Low Privacy (Nur Aggregation)
        self.privacy_levels[1] = PrivacyLevelConfig(
            level=1,
            name="Low Privacy",
            description="Nur Aggregation - Keine individuellen Werte",
            public_arguments=[
                "sensor_id", "timestamp", "value_hash", "room"
            ],
            private_arguments=[
                "actual_value", "device_signature"
            ],
            witness_arguments=[
                "computation_hash", "aggregation_result"
            ],
            circuit_name="aggregation",
            use_cases=[
                "Energieverbrauch-Aggregation",
                "Temperatur-Mittelwerte",
                "Anonyme Statistiken"
            ]
        )
        
        # Level 2: Medium Privacy (Range + Aggregation)
        self.privacy_levels[2] = PrivacyLevelConfig(
            level=2,
            name="Medium Privacy",
            description="Range-Verifikation + Aggregation",
            public_arguments=[
                "sensor_id", "timestamp", "device_id", "value_hash", 
                "room", "proof_hash"
            ],
            private_arguments=[
                "actual_value", "device_signature", "sensor_calibration"
            ],
            witness_arguments=[
                "computation_hash", "range_validation", "aggregation_result"
            ],
            circuit_name="range_verification",
            use_cases=[
                "Smart Home Automation",
                "Energieeffizienz-Monitoring",
                "Geräte-Status-Überwachung"
            ]
        )
        
        # Level 3: High Privacy (Vollständige Verifikation)
        self.privacy_levels[3] = PrivacyLevelConfig(
            level=3,
            name="High Privacy",
            description="Vollständige Datenverifikation",
            public_arguments=[
                "sensor_id", "timestamp", "device_id", "proof_hash"
            ],
            private_arguments=[
                "actual_value", "device_signature", "sensor_calibration",
                "environmental_context", "user_presence"
            ],
            witness_arguments=[
                "computation_hash", "range_validation", "aggregation_result"
            ],
            circuit_name="full_verification",
            use_cases=[
                "Gesundheitsmonitoring",
                "Privates Wohnverhalten",
                "Sicherheitskritische Anwendungen"
        ]
        )
    
    def create_sensor_privacy_profile(self, sensor_id: str, sensor_type: str, 
                                    room: str, privacy_level: int) -> SensorPrivacyProfile:
        """Erstelle Privacy-Profil für einen Sensor"""
        
        config = self.privacy_levels.get(privacy_level)
        if not config:
            raise ValueError(f"Unknown privacy level: {privacy_level}")
        
        # Bestimme erlaubte Argumente basierend auf Privacy Level
        allowed_public = set(config.public_arguments)
        allowed_private = set(config.private_arguments)
        
        # Bestimme eingeschränkte Argumente basierend auf Sensor-Typ
        restricted_args = set()
        if sensor_type == "motion" and room in ["bedroom", "bathroom"]:
            # Bewegungssensoren in privaten Räumen haben höhere Einschränkungen
            restricted_args.add("user_presence")
            restricted_args.add("environmental_context")
        elif sensor_type == "temperature" and room == "outdoor":
            # Außentemperatur ist weniger sensibel
            restricted_args.add("user_presence")
        
        # Bestimme Datenaufbewahrung basierend auf Privacy Level
        data_retention_days = {
            1: 30,   # Low Privacy: 30 Tage
            2: 7,     # Medium Privacy: 7 Tage
            3: 1      # High Privacy: 1 Tag
        }.get(privacy_level, 7)
        
        # Bestimme Anonymisierung basierend auf Privacy Level
        anonymization_required = privacy_level >= 2
        
        profile = SensorPrivacyProfile(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            room=room,
            privacy_level=privacy_level,
            allowed_public_args=allowed_public,
            allowed_private_args=allowed_private,
            restricted_args=restricted_args,
            data_retention_days=data_retention_days,
            anonymization_required=anonymization_required
        )
        
        self.sensor_profiles[sensor_id] = profile
        logger.info(f"Created privacy profile for {sensor_id} (Level {privacy_level})")
        return profile
    
    def get_arguments_for_privacy_level(self, privacy_level: int) -> Dict[str, List[str]]:
        """Hole Argumente für Privacy Level"""
        config = self.privacy_levels.get(privacy_level)
        if not config:
            return {}
        
        return {
            "public": config.public_arguments,
            "private": config.private_arguments,
            "witness": config.witness_arguments
        }
    
    def validate_sensor_data(self, sensor_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validiere Sensordaten gegen Privacy-Profil"""
        profile = self.sensor_profiles.get(sensor_id)
        if not profile:
            return {"valid": False, "error": f"No privacy profile for {sensor_id}"}
        
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "privacy_compliance": True
        }
        
        # Prüfe ob alle erforderlichen Public Arguments vorhanden sind
        for required_arg in profile.allowed_public_args:
            if required_arg not in data:
                validation_result["errors"].append(f"Missing required public argument: {required_arg}")
                validation_result["valid"] = False
        
        # Prüfe ob alle erforderlichen Private Arguments vorhanden sind
        for required_arg in profile.allowed_private_args:
            if required_arg not in data:
                validation_result["errors"].append(f"Missing required private argument: {required_arg}")
                validation_result["valid"] = False
        
        # Prüfe ob eingeschränkte Argumente verwendet werden
        for restricted_arg in profile.restricted_args:
            if restricted_arg in data:
                validation_result["warnings"].append(f"Restricted argument used: {restricted_arg}")
                validation_result["privacy_compliance"] = False
        
        # Prüfe Datenaufbewahrung
        if "timestamp" in data:
            try:
                data_time = datetime.fromisoformat(data["timestamp"])
                age_days = (datetime.now() - data_time).days
                if age_days > profile.data_retention_days:
                    validation_result["warnings"].append(f"Data older than retention period ({profile.data_retention_days} days)")
            except:
                validation_result["errors"].append("Invalid timestamp format")
                validation_result["valid"] = False
        
        return validation_result
    
    def anonymize_sensor_data(self, sensor_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymisiere Sensordaten basierend auf Privacy-Profil"""
        profile = self.sensor_profiles.get(sensor_id)
        if not profile or not profile.anonymization_required:
            return data
        
        anonymized_data = data.copy()
        
        # Anonymisiere Device ID
        if "device_id" in anonymized_data:
            anonymized_data["device_id"] = hashlib.sha256(
                anonymized_data["device_id"].encode()
            ).hexdigest()[:8]
        
        # Anonymisiere Sensor ID
        if "sensor_id" in anonymized_data:
            anonymized_data["sensor_id"] = hashlib.sha256(
                anonymized_data["sensor_id"].encode()
            ).hexdigest()[:8]
        
        # Generalisiere Raum-Information
        if "room" in anonymized_data:
            room = anonymized_data["room"]
            if room in ["bedroom", "bathroom"]:
                anonymized_data["room"] = "private_room"
            elif room in ["living_room", "kitchen"]:
                anonymized_data["room"] = "public_room"
        
        # Generalisiere Zeitstempel (nur Stunde, nicht Minute/Sekunde)
        if "timestamp" in anonymized_data:
            try:
                dt = datetime.fromisoformat(anonymized_data["timestamp"])
                anonymized_data["timestamp"] = dt.replace(minute=0, second=0, microsecond=0).isoformat()
            except:
                pass
        
        logger.info(f"Anonymized data for {sensor_id}")
        return anonymized_data
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Hole Privacy-Summary"""
        summary = {
            "total_arguments": len(self.privacy_arguments),
            "privacy_levels": len(self.privacy_levels),
            "sensor_profiles": len(self.sensor_profiles),
            "argument_breakdown": {
                "public": len([a for a in self.privacy_arguments.values() if a.argument_type == "public"]),
                "private": len([a for a in self.privacy_arguments.values() if a.argument_type == "private"]),
                "witness": len([a for a in self.privacy_arguments.values() if a.argument_type == "witness"])
            },
            "privacy_levels_detail": {}
        }
        
        for level, config in self.privacy_levels.items():
            summary["privacy_levels_detail"][level] = {
                "name": config.name,
                "description": config.description,
                "public_args": len(config.public_arguments),
                "private_args": len(config.private_arguments),
                "witness_args": len(config.witness_arguments),
                "use_cases": config.use_cases
            }
        
        return summary

def create_privacy_argument_test():
    """Test des Privacy Argument Designs"""
    
    designer = IoTPrivacyArgumentDesigner()
    
    print("=== Privacy Argument Design Test ===")
    
    # Zeige Privacy Summary
    summary = designer.get_privacy_summary()
    print(f"Total arguments: {summary['total_arguments']}")
    print(f"Privacy levels: {summary['privacy_levels']}")
    print(f"Argument breakdown: {summary['argument_breakdown']}")
    
    # Teste verschiedene Privacy Levels
    for level in [1, 2, 3]:
        print(f"\n--- Privacy Level {level} ---")
        args = designer.get_arguments_for_privacy_level(level)
        print(f"Public args: {args.get('public', [])}")
        print(f"Private args: {args.get('private', [])}")
        print(f"Witness args: {args.get('witness', [])}")
    
    # Erstelle Sensor-Profile
    sensors = [
        ("TEMP_01", "temperature", "living_room", 1),
        ("MOTION_01", "motion", "bedroom", 3),
        ("HUM_01", "humidity", "kitchen", 2)
    ]
    
    print(f"\n--- Sensor Privacy Profiles ---")
    for sensor_id, sensor_type, room, privacy_level in sensors:
        profile = designer.create_sensor_privacy_profile(sensor_id, sensor_type, room, privacy_level)
        print(f"{sensor_id} ({sensor_type}, {room}): Level {privacy_level}")
        print(f"  Allowed public: {profile.allowed_public_args}")
        print(f"  Allowed private: {profile.allowed_private_args}")
        print(f"  Restricted: {profile.restricted_args}")
        print(f"  Retention: {profile.data_retention_days} days")
        print(f"  Anonymization: {profile.anonymization_required}")
    
    # Teste Datenvalidierung
    print(f"\n--- Data Validation Test ---")
    test_data = {
        "sensor_id": "TEMP_01",
        "timestamp": "2024-01-01T12:00:00",
        "actual_value": 25.5,
        "device_signature": "abc123",
        "room": "living_room"
    }
    
    validation = designer.validate_sensor_data("TEMP_01", test_data)
    print(f"Validation result: {validation}")
    
    # Teste Anonymisierung
    print(f"\n--- Data Anonymization Test ---")
    anonymized = designer.anonymize_sensor_data("TEMP_01", test_data)
    print(f"Original: {test_data}")
    print(f"Anonymized: {anonymized}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_privacy_argument_test()
