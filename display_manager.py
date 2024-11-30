import os
from datetime import datetime
from pathlib import Path
from typing import Dict

class DisplayManager:
    def __init__(self, game_name: str):
        self.game_name = game_name
    
    def display_status(self, metrics: Dict, session_start: datetime, output_file: Path):
        os.system('clear')
        print(f"=== Game Performance Monitor ===")
        print(f"Monitoring: {self.game_name}")
        
        status = "🎮 RUNNING" if metrics['game_running'] else "⏹️ NOT DETECTED"
        print(f"Status: {status}")
        
        if metrics['game_running']:
            if 'fps' in metrics:
                print("\nPerformance Metrics (FrameView):")
                print(f"├── Frame Time: {metrics.get('frame_time_ms', 0):>6.2f}ms")
                print(f"├── Average FPS: {metrics.get('fps', 0):>6.1f}")
                print(f"├── 1% Low FPS: {metrics.get('fps_1_low', 0):>6.1f}")
                print(f"├── Min FPS: {metrics.get('fps_min', 0):>6.1f}")
                print(f"└── Max FPS: {metrics.get('fps_max', 0):>6.1f}")
            else:
                print("\nPerformance Metrics:")
                print("└── Waiting for FrameView data...")
        
        print("\nGPU Metrics:")
        print(f"├── Utilization: {metrics['gpu_utilization']:>6.1f}%")
        print(f"├── Memory: {metrics['gpu_memory_used']:>6.1f}MB / {metrics['gpu_memory_total']}MB")
        print(f"├── Temperature: {metrics['gpu_temperature']:>6.1f}°C")
        print(f"├── Power Draw: {metrics['gpu_power_draw']:>6.1f}W")
        print(f"└── Fan Speed: {metrics['fan_speed']:>6.1f}%")
        
        print("\nSystem Metrics:")
        print(f"├── Total CPU: {metrics['total_cpu_usage']:>6.1f}%")
        print(f"└── Total Memory: {metrics['total_memory_usage']:>6.1f}%")
        
        if session_start:
            duration = datetime.now() - session_start
            print(f"\nSession Duration: {str(duration).split('.')[0]}")
        
        print(f"\nLogging to: {output_file.name}")
        print("\nPress Ctrl+C to stop monitoring and generate report")
