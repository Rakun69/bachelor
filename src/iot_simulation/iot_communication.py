"""
IoT Sensor-to-Gateway Communication System
Implementiert sichere Kommunikation mit Signatur-Verifikation und ZK-Proofs
"""

import json
import time
import hashlib
import hmac
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import socket
import ssl
import struct

logger = logging.getLogger(__name__)

@dataclass
class IoTMessage:
    """IoT-Nachricht mit Metadaten"""
    message_id: str
    device_id: str
    sensor_id: str
    message_type: str  # "sensor_data", "heartbeat", "error", "command"
    timestamp: str
    payload: Dict[str, Any]
    signature: str
    zk_proof: Optional[str] = None
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical

@dataclass
class CommunicationConfig:
    """Konfiguration für IoT-Kommunikation"""
    gateway_host: str = "localhost"
    gateway_port: int = 8888
    use_ssl: bool = True
    timeout_seconds: int = 30
    retry_attempts: int = 3
    heartbeat_interval: int = 60
    max_message_size: int = 1024 * 1024  # 1MB

@dataclass
class SensorReading:
    """Sensor-Reading für Kommunikation"""
    sensor_id: str
    device_id: str
    sensor_type: str
    room: str
    timestamp: str
    value: float
    unit: str
    privacy_level: int
    calibration_data: Optional[Dict[str, Any]] = None

class IoTSensorCommunicator:
    """Kommunikations-Interface für IoT-Sensoren"""
    
    def __init__(self, device_id: str, sensor_id: str, 
                 key_manager, config: CommunicationConfig = None):
        self.device_id = device_id
        self.sensor_id = sensor_id
        self.key_manager = key_manager
        self.config = config or CommunicationConfig()
        
        # Kommunikations-Status
        self.is_connected = False
        self.last_heartbeat = None
        self.message_queue = queue.Queue()
        self.connection = None
        
        # Threading
        self.communication_thread = None
        self.stop_communication = threading.Event()
        
        logger.info(f"Initialized communicator for {sensor_id} on {device_id}")
    
    def connect_to_gateway(self) -> bool:
        """Verbinde zum IoT-Gateway"""
        try:
            # Erstelle Socket-Verbindung
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.settimeout(self.config.timeout_seconds)
            
            # SSL-Wrapper falls aktiviert
            if self.config.use_ssl:
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE  # Für Test-Umgebung
                self.connection = context.wrap_socket(
                    self.connection, 
                    server_hostname=self.config.gateway_host
                )
            
            # Verbinde zum Gateway
            self.connection.connect((self.config.gateway_host, self.config.gateway_port))
            
            # Sende Initial-Handshake
            handshake = self._create_handshake_message()
            self._send_message(handshake)
            
            self.is_connected = True
            logger.info(f"Connected to gateway: {self.config.gateway_host}:{self.config.gateway_port}")
            
            # Starte Kommunikations-Thread
            self.communication_thread = threading.Thread(target=self._communication_loop)
            self.communication_thread.daemon = True
            self.communication_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to gateway: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Trenne Verbindung zum Gateway"""
        self.stop_communication.set()
        
        if self.communication_thread and self.communication_thread.is_alive():
            self.communication_thread.join(timeout=5)
        
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
            self.connection = None
        
        self.is_connected = False
        logger.info("Disconnected from gateway")
    
    def send_sensor_data(self, reading: SensorReading, zk_proof: str = None) -> bool:
        """Sende Sensordaten an Gateway"""
        if not self.is_connected:
            logger.warning("Not connected to gateway")
            return False
        
        try:
            # Erstelle Nachricht
            message = self._create_sensor_data_message(reading, zk_proof)
            
            # Füge zur Queue hinzu
            self.message_queue.put(message)
            
            logger.debug(f"Queued sensor data from {self.sensor_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue sensor data: {e}")
            return False
    
    def send_heartbeat(self) -> bool:
        """Sende Heartbeat an Gateway"""
        if not self.is_connected:
            return False
        
        try:
            message = self._create_heartbeat_message()
            self.message_queue.put(message)
            self.last_heartbeat = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
            return False
    
    def _create_handshake_message(self) -> IoTMessage:
        """Erstelle Handshake-Nachricht"""
        payload = {
            "device_id": self.device_id,
            "sensor_id": self.sensor_id,
            "capabilities": ["sensor_data", "heartbeat"],
            "privacy_levels": [1, 2, 3],
            "version": "1.0"
        }
        
        return self._create_message("handshake", payload)
    
    def _create_sensor_data_message(self, reading: SensorReading, zk_proof: str = None) -> IoTMessage:
        """Erstelle Sensordaten-Nachricht"""
        payload = {
            "sensor_id": reading.sensor_id,
            "device_id": reading.device_id,
            "sensor_type": reading.sensor_type,
            "room": reading.room,
            "timestamp": reading.timestamp,
            "value": reading.value,
            "unit": reading.unit,
            "privacy_level": reading.privacy_level,
            "calibration_data": reading.calibration_data
        }
        
        message = self._create_message("sensor_data", payload)
        message.zk_proof = zk_proof
        message.priority = reading.privacy_level  # Priority basierend auf Privacy Level
        
        return message
    
    def _create_heartbeat_message(self) -> IoTMessage:
        """Erstelle Heartbeat-Nachricht"""
        payload = {
            "device_id": self.device_id,
            "sensor_id": self.sensor_id,
            "status": "alive",
            "uptime": time.time(),
            "last_reading": self.last_heartbeat.isoformat() if self.last_heartbeat else None
        }
        
        return self._create_message("heartbeat", payload)
    
    def _create_message(self, message_type: str, payload: Dict[str, Any]) -> IoTMessage:
        """Erstelle IoT-Nachricht mit Signatur"""
        message_id = f"{self.device_id}_{self.sensor_id}_{int(time.time())}"
        timestamp = datetime.now().isoformat()
        
        # Erstelle Nachricht
        message = IoTMessage(
            message_id=message_id,
            device_id=self.device_id,
            sensor_id=self.sensor_id,
            message_type=message_type,
            timestamp=timestamp,
            payload=payload,
            signature=""  # Wird unten berechnet
        )
        
        # Berechne Signatur
        message_data = f"{message_id}:{message_type}:{timestamp}:{json.dumps(payload, sort_keys=True)}"
        message.signature = self.key_manager.sign_data(self.device_id, message_data)
        
        return message
    
    def _send_message(self, message: IoTMessage) -> bool:
        """Sende Nachricht über Socket"""
        try:
            # Serialisiere Nachricht
            message_data = json.dumps(asdict(message))
            message_bytes = message_data.encode('utf-8')
            
            # Sende Länge + Daten
            length = struct.pack('!I', len(message_bytes))
            self.connection.send(length + message_bytes)
            
            logger.debug(f"Sent message {message.message_id} ({len(message_bytes)} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def _communication_loop(self):
        """Haupt-Kommunikations-Loop"""
        last_heartbeat = time.time()
        
        while not self.stop_communication.is_set():
            try:
                # Sende queued Nachrichten
                while not self.message_queue.empty():
                    try:
                        message = self.message_queue.get_nowait()
                        self._send_message(message)
                    except queue.Empty:
                        break
                
                # Sende Heartbeat falls nötig
                current_time = time.time()
                if current_time - last_heartbeat >= self.config.heartbeat_interval:
                    self.send_heartbeat()
                    last_heartbeat = current_time
                
                # Kurze Pause
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Communication loop error: {e}")
                break
        
        logger.info("Communication loop stopped")

class IoTGateway:
    """IoT Gateway für Sensor-Kommunikation"""
    
    def __init__(self, host: str = "localhost", port: int = 8888, 
                 key_manager=None, use_ssl: bool = True):
        self.host = host
        self.port = port
        self.key_manager = key_manager
        self.use_ssl = use_ssl
        
        # Gateway Status
        self.is_running = False
        self.connected_sensors: Dict[str, IoTSensorCommunicator] = {}
        self.received_messages: List[IoTMessage] = []
        self.message_handlers: Dict[str, Callable] = {}
        
        # Server
        self.server_socket = None
        self.server_thread = None
        
        # Registriere Standard-Handler
        self._register_default_handlers()
        
        logger.info(f"Initialized IoT Gateway on {host}:{port}")
    
    def start(self) -> bool:
        """Starte IoT Gateway"""
        try:
            # Erstelle Server Socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(10)
            
            # Starte Server Thread
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.is_running = True
            logger.info(f"Gateway started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start gateway: {e}")
            return False
    
    def stop(self):
        """Stoppe IoT Gateway"""
        self.is_running = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)
        
        logger.info("Gateway stopped")
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Registriere Message Handler"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    def _register_default_handlers(self):
        """Registriere Standard-Message-Handler"""
        self.register_message_handler("handshake", self._handle_handshake)
        self.register_message_handler("sensor_data", self._handle_sensor_data)
        self.register_message_handler("heartbeat", self._handle_heartbeat)
    
    def _server_loop(self):
        """Haupt-Server-Loop"""
        while self.is_running:
            try:
                # Akzeptiere Verbindung
                client_socket, address = self.server_socket.accept()
                logger.info(f"Accepted connection from {address}")
                
                # Behandle Verbindung in separatem Thread
                client_thread = threading.Thread(
                    target=self._handle_client_connection,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                if self.is_running:
                    logger.error(f"Server loop error: {e}")
                break
    
    def _handle_client_connection(self, client_socket, address):
        """Behandle Client-Verbindung"""
        try:
            while self.is_running:
                # Empfange Nachricht
                message = self._receive_message(client_socket)
                if not message:
                    break
                
                # Verifiziere Nachricht
                if not self._verify_message(message):
                    logger.warning(f"Invalid message from {address}")
                    continue
                
                # Speichere Nachricht
                self.received_messages.append(message)
                
                # Behandle Nachricht
                handler = self.message_handlers.get(message.message_type)
                if handler:
                    handler(message, client_socket)
                else:
                    logger.warning(f"No handler for message type: {message.message_type}")
                
        except Exception as e:
            logger.error(f"Error handling client {address}: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def _receive_message(self, client_socket) -> Optional[IoTMessage]:
        """Empfange Nachricht von Client"""
        try:
            # Empfange Länge
            length_data = client_socket.recv(4)
            if len(length_data) < 4:
                return None
            
            length = struct.unpack('!I', length_data)[0]
            
            # Empfange Daten
            message_data = b''
            while len(message_data) < length:
                chunk = client_socket.recv(min(length - len(message_data), 4096))
                if not chunk:
                    return None
                message_data += chunk
            
            # Deserialisiere Nachricht
            message_dict = json.loads(message_data.decode('utf-8'))
            return IoTMessage(**message_dict)
            
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None
    
    def _verify_message(self, message: IoTMessage) -> bool:
        """Verifiziere Nachricht (Signatur + ZK-Proof)"""
        # Verifiziere Signatur
        message_data = f"{message.message_id}:{message.message_type}:{message.timestamp}:{json.dumps(message.payload, sort_keys=True)}"
        if not self.key_manager.verify_signature(message.device_id, message_data, message.signature):
            logger.warning(f"Signature verification failed for {message.device_id}")
            return False
        
        # TODO: Verifiziere ZK-Proof falls vorhanden
        if message.zk_proof:
            # Hier würde die ZK-Proof-Verifikation stattfinden
            pass
        
        return True
    
    def _handle_handshake(self, message: IoTMessage, client_socket):
        """Behandle Handshake-Nachricht"""
        logger.info(f"Handshake from {message.device_id}:{message.sensor_id}")
        
        # Sende Bestätigung
        response = {
            "status": "accepted",
            "gateway_id": "GATEWAY_001",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response_data = json.dumps(response).encode('utf-8')
            length = struct.pack('!I', len(response_data))
            client_socket.send(length + response_data)
        except Exception as e:
            logger.error(f"Failed to send handshake response: {e}")
    
    def _handle_sensor_data(self, message: IoTMessage, client_socket):
        """Behandle Sensordaten-Nachricht"""
        logger.info(f"Received sensor data from {message.sensor_id}: {message.payload.get('value')}")
        
        # Hier würde die Datenverarbeitung stattfinden
        # - Aggregation
        # - Anomalie-Erkennung
        # - Speicherung
        # - Weiterleitung an Backend
    
    def _handle_heartbeat(self, message: IoTMessage, client_socket):
        """Behandle Heartbeat-Nachricht"""
        logger.debug(f"Heartbeat from {message.device_id}:{message.sensor_id}")
    
    def get_connected_sensors(self) -> List[Dict[str, Any]]:
        """Hole Liste verbundener Sensoren"""
        sensors = []
        for sensor_id, communicator in self.connected_sensors.items():
            sensors.append({
                "sensor_id": sensor_id,
                "device_id": communicator.device_id,
                "is_connected": communicator.is_connected,
                "last_heartbeat": communicator.last_heartbeat.isoformat() if communicator.last_heartbeat else None
            })
        return sensors
    
    def get_received_messages(self, message_type: str = None) -> List[IoTMessage]:
        """Hole empfangene Nachrichten"""
        if message_type:
            return [msg for msg in self.received_messages if msg.message_type == message_type]
        return self.received_messages.copy()

def create_iot_communication_test():
    """Test der IoT-Kommunikation"""
    
    # Erstelle Key Manager
    from .device_key_manager import IoTDeviceKeyManager
    key_manager = IoTDeviceKeyManager()
    
    # Registriere Test-Devices
    key_manager.register_device("DEV_001", "Test Device 1", "sensor")
    key_manager.generate_device_keys("DEV_001", 2)
    
    # Starte Gateway
    gateway = IoTGateway(key_manager=key_manager)
    if not gateway.start():
        logger.error("Failed to start gateway")
        return
    
    # Erstelle Sensor Communicator
    communicator = IoTSensorCommunicator(
        device_id="DEV_001",
        sensor_id="TEMP_01",
        key_manager=key_manager
    )
    
    # Verbinde zum Gateway
    if communicator.connect_to_gateway():
        logger.info("Sensor connected to gateway")
        
        # Sende Test-Daten
        reading = SensorReading(
            sensor_id="TEMP_01",
            device_id="DEV_001",
            sensor_type="temperature",
            room="living_room",
            timestamp=datetime.now().isoformat(),
            value=25.5,
            unit="°C",
            privacy_level=2
        )
        
        communicator.send_sensor_data(reading)
        time.sleep(1)
        
        # Sende Heartbeat
        communicator.send_heartbeat()
        time.sleep(1)
        
        # Trenne Verbindung
        communicator.disconnect()
    
    # Stoppe Gateway
    gateway.stop()
    
    # Zeige Ergebnisse
    messages = gateway.get_received_messages()
    logger.info(f"Gateway received {len(messages)} messages")
    
    for message in messages:
        logger.info(f"Message: {message.message_type} from {message.sensor_id}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_iot_communication_test()
