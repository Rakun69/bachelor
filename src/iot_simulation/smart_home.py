"""
Smart Home IoT Simulation
Simuliert verschiedene Sensoren in einem Smart Home Environment
"""

import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """Single sensor reading with metadata"""
    sensor_id: str
    sensor_type: str
    room: str
    timestamp: str
    value: float
    unit: str
    privacy_level: int  # 1=low, 2=medium, 3=high sensitivity

@dataclass
class SensorConfig:
    """Configuration for a sensor"""
    sensor_id: str
    sensor_type: str
    room: str
    base_value: float
    noise_level: float
    privacy_level: int
    update_frequency: int  # seconds

class SmartHomeSensors:
    """Simulates various IoT sensors in a smart home with extended time periods"""
    
    def __init__(self, config_file: str = None):
        self.sensors = {}
        self.readings_history = []
        self.start_time = datetime.now()
        
        # Time period configurations
        self.time_periods = {
            "1_day": {"hours": 24, "description": "1 Tag Simulation"},
            "1_week": {"hours": 24 * 7, "description": "1 Woche Simulation"}, 
            "1_month": {"hours": 24 * 30, "description": "1 Monat Simulation"}
        }
        
        if config_file:
            self.load_config(config_file)
        else:
            self._setup_default_sensors()
    
    def _setup_default_sensors(self):
        """Setup default smart home sensors"""
        default_sensors = [
            # Living Room
            SensorConfig("LR_TEMP_01", "temperature", "living_room", 22.0, 0.5, 2, 60),
            SensorConfig("LR_HUM_01", "humidity", "living_room", 45.0, 2.0, 1, 60),
            SensorConfig("LR_MOTION_01", "motion", "living_room", 0.0, 0.0, 3, 5),
            SensorConfig("LR_LIGHT_01", "light", "living_room", 300.0, 50.0, 1, 30),
            
            # Bedroom
            SensorConfig("BR_TEMP_01", "temperature", "bedroom", 20.0, 0.3, 2, 60),
            SensorConfig("BR_HUM_01", "humidity", "bedroom", 50.0, 3.0, 1, 60),
            SensorConfig("BR_MOTION_01", "motion", "bedroom", 0.0, 0.0, 3, 5),
            SensorConfig("BR_SLEEP_01", "sleep_sensor", "bedroom", 0.0, 0.0, 3, 10),
            
            # Kitchen
            SensorConfig("KT_TEMP_01", "temperature", "kitchen", 23.0, 1.0, 2, 60),
            SensorConfig("KT_HUM_01", "humidity", "kitchen", 60.0, 5.0, 1, 60),
            SensorConfig("KT_MOTION_01", "motion", "kitchen", 0.0, 0.0, 3, 5),
            SensorConfig("KT_GAS_01", "gas", "kitchen", 0.0, 0.1, 3, 30),
            
            # Bathroom
            SensorConfig("BT_TEMP_01", "temperature", "bathroom", 21.0, 1.0, 2, 60),
            SensorConfig("BT_HUM_01", "humidity", "bathroom", 70.0, 10.0, 2, 60),
            SensorConfig("BT_MOTION_01", "motion", "bathroom", 0.0, 0.0, 3, 5),
            
            # Outdoor
            SensorConfig("OD_TEMP_01", "temperature", "outdoor", 15.0, 2.0, 1, 60),  # Fixed: 60s statt 120s
            
            # Additional temperature sensors for more data
            SensorConfig("LR_TEMP_02", "temperature", "living_room", 22.5, 0.4, 2, 60),
            SensorConfig("BR_TEMP_02", "temperature", "bedroom", 19.5, 0.4, 2, 60),
            SensorConfig("KT_TEMP_02", "temperature", "kitchen", 23.5, 0.8, 2, 60),
            SensorConfig("BT_TEMP_02", "temperature", "bathroom", 20.5, 0.9, 2, 60),
            SensorConfig("OD_TEMP_02", "temperature", "outdoor", 14.5, 1.8, 1, 60),
            SensorConfig("OD_HUM_01", "humidity", "outdoor", 80.0, 5.0, 1, 120),
            SensorConfig("OD_WIND_01", "wind_speed", "outdoor", 5.0, 3.0, 1, 60),
        ]
        
        for sensor in default_sensors:
            self.sensors[sensor.sensor_id] = sensor
    
    def get_time_factors(self, current_time: datetime) -> Dict[str, float]:
        """Calculate time-based factors for realistic sensor variations"""
        hour = current_time.hour
        day_of_week = current_time.weekday()  # 0=Monday, 6=Sunday
        
        # Day/night cycle (0.5 to 1.5 multiplier)
        day_night_factor = 0.8 + 0.4 * (1 + np.sin(2 * np.pi * (hour - 6) / 24)) / 2
        
        # Weekday vs weekend activity
        weekend_factor = 1.2 if day_of_week >= 5 else 1.0
        
        # Activity patterns (higher during typical active hours)
        if 7 <= hour <= 9 or 18 <= hour <= 22:
            activity_factor = 1.3  # High activity
        elif 10 <= hour <= 17:
            activity_factor = 0.8  # Medium activity (work hours)
        elif 23 <= hour or hour <= 6:
            activity_factor = 0.3  # Low activity (sleep)
        else:
            activity_factor = 1.0
        
        return {
            'day_night': day_night_factor,
            'weekend': weekend_factor,
            'activity': activity_factor
        }
    
    def simulate_sensor_value(self, sensor: SensorConfig, current_time: datetime) -> float:
        """Simulate realistic sensor value based on type and time"""
        factors = self.get_time_factors(current_time)
        base_value = sensor.base_value
        
        if sensor.sensor_type == "temperature":
            # Temperature varies with time of day and room
            daily_variation = 2 * np.sin(2 * np.pi * (current_time.hour - 14) / 24)
            if sensor.room == "kitchen":
                # Kitchen gets warmer during cooking times
                if 7 <= current_time.hour <= 9 or 17 <= current_time.hour <= 20:
                    daily_variation += random.uniform(1, 4)
            elif sensor.room == "outdoor":
                daily_variation *= 3  # Outdoor has more variation
            
            noise = random.gauss(0, sensor.noise_level)
            return max(0, base_value + daily_variation + noise)
        
        elif sensor.sensor_type == "humidity":
            # Humidity varies with activities and outdoor conditions
            activity_variation = (factors['activity'] - 1) * 5
            if sensor.room == "bathroom":
                # Bathroom humidity spikes during usage
                if random.random() < factors['activity'] * 0.1:
                    activity_variation += random.uniform(10, 20)
            
            noise = random.gauss(0, sensor.noise_level)
            return max(0, min(100, base_value + activity_variation + noise))
        
        elif sensor.sensor_type == "motion":
            # Motion is binary with probability based on activity
            motion_probability = factors['activity'] * 0.1
            if sensor.room == "bedroom" and 23 <= current_time.hour or current_time.hour <= 6:
                motion_probability *= 0.1  # Very low motion in bedroom at night
            return 1.0 if random.random() < motion_probability else 0.0
        
        elif sensor.sensor_type == "light":
            # Light levels based on time of day and activity
            natural_light = max(0, 100 * np.sin(np.pi * (current_time.hour - 6) / 12))
            artificial_light = factors['activity'] * 200
            noise = random.gauss(0, sensor.noise_level)
            return max(0, natural_light + artificial_light + noise)
        
        elif sensor.sensor_type == "sleep_sensor":
            # Sleep detection (1 = sleeping, 0 = awake)
            if 23 <= current_time.hour or current_time.hour <= 6:
                return 1.0 if random.random() < 0.9 else 0.0
            else:
                return 1.0 if random.random() < 0.1 else 0.0
        
        elif sensor.sensor_type == "gas":
            # Gas levels (usually 0, occasional spikes)
            if sensor.room == "kitchen" and factors['activity'] > 1:
                return random.uniform(0, 1) if random.random() < 0.05 else 0.0
            return 0.0
        
        elif sensor.sensor_type == "wind_speed":
            # Wind speed with weather patterns
            base_wind = sensor.base_value + random.gauss(0, sensor.noise_level)
            return max(0, base_wind)
        
        else:
            # Default: base value with noise
            noise = random.gauss(0, sensor.noise_level)
            return max(0, base_value + noise)
    
    def get_sensor_unit(self, sensor_type: str) -> str:
        """Get the unit for a sensor type"""
        units = {
            'temperature': '°C',
            'humidity': '%',
            'motion': 'bool',
            'light': 'lux',
            'sleep_sensor': 'bool',
            'gas': 'ppm',
            'wind_speed': 'm/s'
        }
        return units.get(sensor_type, 'value')
    
    def generate_readings(self, duration_hours: int = 24, time_step_seconds: int = 60) -> List[SensorReading]:
        """Generate sensor readings for specified duration"""
        readings = []
        current_time = self.start_time
        end_time = current_time + timedelta(hours=duration_hours)
        
        logger.info(f"Generating sensor data from {current_time} to {end_time}")
        
        while current_time < end_time:
            for sensor in self.sensors.values():
                # Check if it's time for this sensor to update
                if (current_time - self.start_time).total_seconds() % sensor.update_frequency == 0:
                    value = self.simulate_sensor_value(sensor, current_time)
                    
                    reading = SensorReading(
                        sensor_id=sensor.sensor_id,
                        sensor_type=sensor.sensor_type,
                        room=sensor.room,
                        timestamp=current_time.isoformat(),
                        value=round(value, 2),
                        unit=self.get_sensor_unit(sensor.sensor_type),
                        privacy_level=sensor.privacy_level
                    )
                    
                    readings.append(reading)
            
            current_time += timedelta(seconds=time_step_seconds)
        
        self.readings_history.extend(readings)
        logger.info(f"Generated {len(readings)} sensor readings")
        return readings
    
    def save_readings(self, readings: List[SensorReading], filename: str):
        """Save readings to JSON file"""
        readings_dict = [asdict(reading) for reading in readings]
        with open(filename, 'w') as f:
            json.dump(readings_dict, f, indent=2)
        logger.info(f"Saved {len(readings)} readings to {filename}")
    
    def load_config(self, config_file: str):
        """Load sensor configuration from file"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        for sensor_data in config['sensors']:
            sensor = SensorConfig(**sensor_data)
            self.sensors[sensor.sensor_id] = sensor
    
    def get_statistics(self, readings: List[SensorReading] = None) -> Dict[str, Any]:
        """Get basic statistics about the sensor data"""
        if readings is None:
            readings = self.readings_history
        
        if not readings:
            return {}
        
        df = pd.DataFrame([asdict(r) for r in readings])
        
        stats = {
            'total_readings': len(readings),
            'unique_sensors': df['sensor_id'].nunique(),
            'sensor_types': df['sensor_type'].unique().tolist(),
            'rooms': df['room'].unique().tolist(),
            'time_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'privacy_levels': df['privacy_level'].value_counts().to_dict()
        }
        
        # Statistics by sensor type
        stats['by_sensor_type'] = {}
        for sensor_type in df['sensor_type'].unique():
            type_data = df[df['sensor_type'] == sensor_type]
            stats['by_sensor_type'][sensor_type] = {
                'count': len(type_data),
                'mean': float(type_data['value'].mean()),
                'std': float(type_data['value'].std()),
                'min': float(type_data['value'].min()),
                'max': float(type_data['value'].max())
            }
        
        return stats
    
    def generate_multi_period_data(self, output_dir: str = "/home/ramon/bachelor/data/raw") -> Dict[str, Any]:
        """Generate IoT data for multiple time periods (1 day, 1 week, 1 month)"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for period_name, config in self.time_periods.items():
            logger.info(f"Generating {config['description']} data...")
            
            # Adjust time step based on duration for performance
            if config['hours'] <= 24:
                time_step = 60  # 1 minute for 1 day
            elif config['hours'] <= 168:  # 1 week
                time_step = 300  # 5 minutes for 1 week  
            else:  # 1 month
                time_step = 900  # 15 minutes for 1 month
            
            # Generate readings
            readings = self.generate_readings(
                duration_hours=config['hours'],
                time_step_seconds=time_step
            )
            
            # Save data
            filename = f"iot_readings_{period_name}.json"
            filepath = output_path / filename
            self.save_readings(readings, str(filepath))
            
            # Get statistics
            stats = self.get_statistics(readings)
            stats['period'] = period_name
            stats['time_config'] = config
            stats['time_step_seconds'] = time_step
            
            # Save stats
            stats_filename = f"iot_statistics_{period_name}.json"
            stats_filepath = output_path / stats_filename
            with open(stats_filepath, 'w') as f:
                json.dump(stats, f, indent=2)
            
            results[period_name] = {
                'readings_count': len(readings),
                'data_file': str(filepath),
                'stats_file': str(stats_filepath),
                'statistics': stats
            }
            
            logger.info(f"Generated {len(readings)} readings for {config['description']}")
        
        # Save combined summary
        summary_file = output_path / "multi_period_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    """Example usage of the Smart Home IoT simulation"""
    
    # Create smart home simulation
    smart_home = SmartHomeSensors()
    
    # Generate 24 hours of data with 1-minute intervals
    readings = smart_home.generate_readings(duration_hours=24, time_step_seconds=60)
    
    # Save raw data
    smart_home.save_readings(readings, "/home/ramon/bachelor/data/raw/iot_readings_24h.json")
    
    # Get and print statistics
    stats = smart_home.get_statistics(readings)
    with open("/home/ramon/bachelor/data/raw/iot_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Generated {len(readings)} sensor readings")
    print(f"Sensor types: {stats['sensor_types']}")
    print(f"Rooms: {stats['rooms']}")
    print("\nPrivacy level distribution:")
    for level, count in stats['privacy_levels'].items():
        print(f"  Level {level}: {count} readings")

if __name__ == "__main__":
    main()