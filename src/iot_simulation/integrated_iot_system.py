"""
Integriertes IoT-System mit echten Sensoren
Kombiniert alle Komponenten: Sensoren, Keys, Kommunikation, Integrity, Privacy
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# Import aller Komponenten
from .real_iot_sensors import RealIoTSensor, IoTDeviceKeyManager, IoTZKProofGenerator
from .iot_communication import IoTGateway, IoTSensorCommunicator, CommunicationConfig
from .computational_integrity import IoTComputationalIntegrity
from .privacy_argument_design import IoTPrivacyArgumentDesigner

logger = logging.getLogger(__name__)

class IntegratedIoTSystem:
    """Integriertes IoT-System mit allen Komponenten"""
    
    def __init__(self, system_name: str = "SmartHome_001"):
        self.system_name = system_name
        self.is_running = False
        
        # Komponenten initialisieren
        self.key_manager = IoTDeviceKeyManager()
        self.privacy_designer = IoTPrivacyArgumentDesigner()
        self.integrity_system = IoTComputationalIntegrity()
        
        # Gateway
        self.gateway = IoTGateway(key_manager=self.key_manager)
        
        # Sensoren
        self.sensors: Dict[str, RealIoTSensor] = {}
        self.sensor_communicators: Dict[str, IoTSensorCommunicator] = {}
        
        # System Status
        self.system_stats = {
            "start_time": None,
            "total_readings": 0,
            "verified_readings": 0,
            "proof_generation_success": 0,
            "communication_errors": 0,
            "privacy_violations": 0
        }
        
        logger.info(f"Initialized integrated IoT system: {system_name}")
    
    def setup_smart_home_network(self) -> bool:
        """Setup Smart Home IoT-Netzwerk"""
        try:
            # Starte Gateway
            if not self.gateway.start():
                logger.error("Failed to start gateway")
                return False
            
            # Erstelle Sensoren
            sensor_configs = [
                # Temperatur-Sensoren
                ("TEMP_LR_01", "DEV_001", "temperature", "living_room", 2),
                ("TEMP_BR_01", "DEV_002", "temperature", "bedroom", 3),
                ("TEMP_KT_01", "DEV_003", "temperature", "kitchen", 2),
                ("TEMP_OD_01", "DEV_004", "temperature", "outdoor", 1),
                
                # Luftfeuchtigkeit
                ("HUM_LR_01", "DEV_001", "humidity", "living_room", 1),
                ("HUM_BR_01", "DEV_002", "humidity", "bedroom", 2),
                ("HUM_KT_01", "DEV_003", "humidity", "kitchen", 1),
                
                # Bewegung
                ("MOTION_LR_01", "DEV_005", "motion", "living_room", 3),
                ("MOTION_BR_01", "DEV_006", "motion", "bedroom", 3),
                ("MOTION_KT_01", "DEV_007", "motion", "kitchen", 2),
                
                # Licht
                ("LIGHT_LR_01", "DEV_005", "light", "living_room", 1),
                ("LIGHT_BR_01", "DEV_006", "light", "bedroom", 2),
                
                # Gas (Küche)
                ("GAS_KT_01", "DEV_003", "gas", "kitchen", 3),
            ]
            
            # Registriere Devices und erstelle Sensoren
            for sensor_id, device_id, sensor_type, room, privacy_level in sensor_configs:
                # Registriere Device
                if device_id not in self.key_manager.device_keys:
                    self.key_manager.generate_device_key(device_id)
                
                # Erstelle Sensor
                sensor = RealIoTSensor(
                    sensor_id=sensor_id,
                    device_id=device_id,
                    sensor_type=sensor_type,
                    room=room,
                    privacy_level=privacy_level,
                    key_manager=self.key_manager
                )
                self.sensors[sensor_id] = sensor
                
                # Erstelle Privacy-Profil
                self.privacy_designer.create_sensor_privacy_profile(
                    sensor_id, sensor_type, room, privacy_level
                )
                
                # Erstelle Communicator
                communicator = IoTSensorCommunicator(
                    device_id=device_id,
                    sensor_id=sensor_id,
                    key_manager=self.key_manager
                )
                self.sensor_communicators[sensor_id] = communicator
                
                # Verbinde zum Gateway
                if communicator.connect_to_gateway():
                    logger.info(f"Connected sensor {sensor_id} to gateway")
                else:
                    logger.warning(f"Failed to connect sensor {sensor_id}")
            
            logger.info(f"Setup complete: {len(self.sensors)} sensors, {len(set(c[1] for c in sensor_configs))} devices")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup smart home network: {e}")
            return False
    
    def start_data_collection(self, duration_minutes: int = 60):
        """Starte Datensammlung"""
        self.is_running = True
        self.system_stats["start_time"] = datetime.now()
        
        logger.info(f"Starting data collection for {duration_minutes} minutes")
        
        # Starte Datensammlung in separatem Thread
        collection_thread = threading.Thread(
            target=self._data_collection_loop,
            args=(duration_minutes,)
        )
        collection_thread.daemon = True
        collection_thread.start()
        
        return collection_thread
    
    def _data_collection_loop(self, duration_minutes: int):
        """Haupt-Datensammlung-Loop"""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        while self.is_running and datetime.now() < end_time:
            try:
                # Sammle Daten von allen Sensoren
                for sensor_id, sensor in self.sensors.items():
                    # Generiere Reading
                    reading = self._generate_sensor_reading(sensor)
                    
                    if reading:
                        # Validiere gegen Privacy-Profil
                        validation = self.privacy_designer.validate_sensor_data(sensor_id, reading.__dict__)
                        
                        if validation["valid"]:
                            # Generiere ZK-Proof
                            proof_result = sensor.proof_generator.generate_sensor_proof(reading, sensor.privacy_level)
                            
                            if proof_result.success:
                                reading.zk_proof = proof_result.proof
                                reading.proof_public_inputs = proof_result.public_inputs
                                self.system_stats["proof_generation_success"] += 1
                            
                            # Sende an Gateway
                            communicator = self.sensor_communicators.get(sensor_id)
                            if communicator:
                                if communicator.send_sensor_data(reading, reading.zk_proof):
                                    self.system_stats["verified_readings"] += 1
                                else:
                                    self.system_stats["communication_errors"] += 1
                            
                            self.system_stats["total_readings"] += 1
                        else:
                            self.system_stats["privacy_violations"] += 1
                            logger.warning(f"Privacy violation for {sensor_id}: {validation.get('errors', [])}")
                
                # Kurze Pause zwischen Readings
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Data collection error: {e}")
                self.system_stats["communication_errors"] += 1
        
        self.is_running = False
        logger.info("Data collection completed")
    
    def _generate_sensor_reading(self, sensor: RealIoTSensor) -> Optional[Any]:
        """Generiere Sensor-Reading basierend auf Typ und Zeit"""
        current_time = datetime.now()
        
        # Simuliere Sensor-Wert basierend auf Typ
        if sensor.sensor_type == "temperature":
            # Temperatur mit Tageszyklus
            hour = current_time.hour
            base_temp = 20 + 5 * (hour / 24) + (current_time.minute % 10)
            value = base_temp + (hash(sensor.sensor_id) % 10 - 5)  # Sensor-spezifische Variation
        elif sensor.sensor_type == "humidity":
            # Luftfeuchtigkeit
            value = 40 + 20 * (current_time.minute % 30) / 30 + (hash(sensor.sensor_id) % 20 - 10)
        elif sensor.sensor_type == "motion":
            # Bewegung (binär)
            activity_prob = 0.1 if 7 <= current_time.hour <= 22 else 0.01
            value = 1.0 if (hash(str(current_time)) % 100) < (activity_prob * 100) else 0.0
        elif sensor.sensor_type == "light":
            # Licht mit Tageszyklus
            hour = current_time.hour
            natural_light = max(0, 100 * (hour - 6) / 12) if 6 <= hour <= 18 else 0
            artificial_light = 50 if (hash(str(current_time)) % 100) < 30 else 0
            value = natural_light + artificial_light
        elif sensor.sensor_type == "gas":
            # Gas (meist 0, gelegentlich Spikes)
            value = 0.0 if (hash(str(current_time)) % 100) < 95 else 1.0
        else:
            value = 0.0
        
        # Generiere Reading
        return sensor.generate_reading(value, current_time.isoformat())
    
    def stop_system(self):
        """Stoppe IoT-System"""
        self.is_running = False
        
        # Trenne alle Verbindungen
        for communicator in self.sensor_communicators.values():
            communicator.disconnect()
        
        # Stoppe Gateway
        self.gateway.stop()
        
        logger.info("IoT system stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Hole System-Status"""
        status = {
            "system_name": self.system_name,
            "is_running": self.is_running,
            "sensors_count": len(self.sensors),
            "devices_count": len(set(s.device_id for s in self.sensors.values())),
            "gateway_status": "running" if self.gateway.is_running else "stopped",
            "system_stats": self.system_stats.copy()
        }
        
        # Berechne Erfolgsraten
        if self.system_stats["total_readings"] > 0:
            status["proof_success_rate"] = (
                self.system_stats["proof_generation_success"] / 
                self.system_stats["total_readings"]
            )
            status["verification_success_rate"] = (
                self.system_stats["verified_readings"] / 
                self.system_stats["total_readings"]
            )
        else:
            status["proof_success_rate"] = 0.0
            status["verification_success_rate"] = 0.0
        
        # Privacy Summary
        status["privacy_summary"] = self.privacy_designer.get_privacy_summary()
        
        return status
    
    def get_sensor_data_summary(self) -> Dict[str, Any]:
        """Hole Sensordaten-Summary"""
        gateway_messages = self.gateway.get_received_messages("sensor_data")
        
        summary = {
            "total_messages": len(gateway_messages),
            "sensors_data": {},
            "privacy_levels": {},
            "data_types": {}
        }
        
        # Gruppiere nach Sensor
        for message in gateway_messages:
            sensor_id = message.sensor_id
            payload = message.payload
            
            if sensor_id not in summary["sensors_data"]:
                summary["sensors_data"][sensor_id] = {
                    "count": 0,
                    "values": [],
                    "has_proof": 0
                }
            
            summary["sensors_data"][sensor_id]["count"] += 1
            summary["sensors_data"][sensor_id]["values"].append(payload.get("value", 0))
            if message.zk_proof:
                summary["sensors_data"][sensor_id]["has_proof"] += 1
        
        # Berechne Statistiken
        for sensor_id, data in summary["sensors_data"].items():
            if data["values"]:
                data["average"] = sum(data["values"]) / len(data["values"])
                data["min"] = min(data["values"])
                data["max"] = max(data["values"])
                data["proof_rate"] = data["has_proof"] / data["count"] if data["count"] > 0 else 0
        
        return summary
    
    def run_comprehensive_test(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """Führe umfassenden Test des IoT-Systems durch"""
        logger.info(f"Starting comprehensive test for {duration_minutes} minutes")
        
        # Setup System
        if not self.setup_smart_home_network():
            return {"success": False, "error": "Failed to setup system"}
        
        # Starte Datensammlung
        collection_thread = self.start_data_collection(duration_minutes)
        
        # Warte auf Abschluss
        collection_thread.join()
        
        # Hole Ergebnisse
        system_status = self.get_system_status()
        data_summary = self.get_sensor_data_summary()
        
        # Stoppe System
        self.stop_system()
        
        # Kombiniere Ergebnisse
        results = {
            "success": True,
            "test_duration_minutes": duration_minutes,
            "system_status": system_status,
            "data_summary": data_summary,
            "performance_metrics": {
                "readings_per_minute": system_status["system_stats"]["total_readings"] / duration_minutes,
                "proof_success_rate": system_status["proof_success_rate"],
                "verification_success_rate": system_status["verification_success_rate"],
                "privacy_violations": system_status["system_stats"]["privacy_violations"],
                "communication_errors": system_status["system_stats"]["communication_errors"]
            }
        }
        
        logger.info(f"Comprehensive test completed: {system_status['system_stats']['total_readings']} readings")
        return results

def create_integrated_iot_demo():
    """Demo des integrierten IoT-Systems"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== Integrated IoT System Demo ===")
    
    # Erstelle System
    iot_system = IntegratedIoTSystem("Demo_SmartHome")
    
    # Führe Test durch
    results = iot_system.run_comprehensive_test(duration_minutes=5)
    
    if results["success"]:
        print(f"✅ Test successful!")
        print(f"📊 Total readings: {results['system_status']['system_stats']['total_readings']}")
        print(f"🔒 Verified readings: {results['system_status']['system_stats']['verified_readings']}")
        print(f"⚡ Proof success rate: {results['performance_metrics']['proof_success_rate']:.2%}")
        print(f"🔐 Verification success rate: {results['performance_metrics']['verification_success_rate']:.2%}")
        print(f"🛡️ Privacy violations: {results['performance_metrics']['privacy_violations']}")
        print(f"📡 Communication errors: {results['performance_metrics']['communication_errors']}")
        
        print(f"\n📈 Performance:")
        print(f"  Readings per minute: {results['performance_metrics']['readings_per_minute']:.1f}")
        
        print(f"\n🏠 Sensors:")
        for sensor_id, data in results['data_summary']['sensors_data'].items():
            print(f"  {sensor_id}: {data['count']} readings, avg={data.get('average', 0):.1f}, proof_rate={data.get('proof_rate', 0):.2%}")
    else:
        print(f"❌ Test failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    create_integrated_iot_demo()
