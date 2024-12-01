import csv
from datetime import datetime
from pathlib import Path
from typing import Dict

class DataLogger:
    def __init__(self, game_name: str):
        self.base_dir = Path("game_performance_analysis")
        self.logs_dir = self.base_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.logs_dir / f"{game_name}_{timestamp}.csv"
        
        self.headers = [
            'timestamp',
            'game_running',
            'frame_time_ms',
            'fps',
            'fps_1_low',
            'fps_min',
            'fps_max',
            'gpu_utilization',
            'gpu_memory_used',
            'gpu_memory_total',
            'gpu_temperature',
            'gpu_power_draw',
            'fan_speed',
            'total_cpu_usage',
            'total_memory_usage'
        ]
        
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
    
    def log_metrics(self, metrics: Dict):
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(metrics)

    def get_output_file(self) -> Path:
        return self.output_file
