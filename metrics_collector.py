import subprocess
import psutil
from datetime import datetime
from typing import Dict
import time
import csv
import os
from pathlib import Path

class MetricsCollector:
    def __init__(self, game_name: str):
        self.game_name = game_name.lower()
        self.last_log_time = None
        self.last_log_data = None
        
        # Get Windows user folder from environment or default to current user
        windows_username = os.getenv('USER', 'User')
        
        # Check common FrameView log locations using WSL mount points
        possible_log_dirs = [
            Path("/mnt/c/Users") / windows_username / "Documents" / "NVIDIA FrameView",
            Path("/mnt/c/Users") / windows_username / "Documents" / "FrameView",
            Path.cwd() / "FrameView"  # Current directory
        ]
        
        # Find first existing log directory
        self.frameview_log_dir = None
        for dir_path in possible_log_dirs:
            print(f"Checking for FrameView logs in: {dir_path}")
            if dir_path.exists():
                self.frameview_log_dir = dir_path
                print(f"Found FrameView log directory: {dir_path}")
                break
                
        if not self.frameview_log_dir:
            print("Warning: Could not find FrameView log directory in any expected location")

    def get_latest_frameview_log(self) -> tuple[Path, float]:
        """Get the most recent FrameView log file and its modification time"""
        try:
            if not self.frameview_log_dir:
                return None, None
                
            log_files = list(self.frameview_log_dir.glob("FrameView_*.csv"))
            if not log_files:
                print(f"No FrameView log files found in {self.frameview_log_dir}")
                return None, None
                
            latest = max(log_files, key=os.path.getmtime)
            mod_time = os.path.getmtime(latest)
            print(f"Found latest log file: {latest}")
            print(f"Last modified: {datetime.fromtimestamp(mod_time)}")
            return latest, mod_time
        except Exception as e:
            print(f"Error finding FrameView log: {e}")
            return None, None

    def read_frameview_metrics(self) -> Dict:
        """Read the latest metrics from FrameView's log"""
        try:
            log_file, mod_time = self.get_latest_frameview_log()
            
            if not log_file:
                return {}
                
            # If same file we already read, return cached data
            if self.last_log_time and mod_time <= self.last_log_time:
                return self.last_log_data if self.last_log_data else {}
                
            self.last_log_time = mod_time
            
            print(f"Reading new data from {log_file}")
            with open(log_file, 'r') as f:
                lines = [line for line in f.readlines() if line.strip() and not line.startswith("TimeStamp")]
                if not lines:
                    print("No data lines found in log file")
                    return {}
                    
                latest = lines[-1].strip().split(',')
                print(f"Processing latest log line: {latest[:10]}...")  # Show first 10 columns
                
                metrics = {
                    'fps': float(latest[8]),
                    'fps_1_low': float(latest[9]),
                    'fps_min': float(latest[12]),
                    'fps_max': float(latest[13]),
                    'frame_time_ms': float(latest[20])
                }
                
                print(f"Parsed metrics: {metrics}")
                self.last_log_data = metrics
                return metrics
                
        except Exception as e:
            print(f"Error reading FrameView metrics: {e}")
            print(f"Error details: {str(e.__class__.__name__)}")
            return {}

    def check_game_running(self) -> bool:
        try:
            result = subprocess.run(
                ['wmic.exe', 'process', 'get', 'name'], 
                capture_output=True, 
                text=True
            )
            return self.game_name in result.stdout.lower()
        except Exception as e:
            print(f"Error checking game process: {e}")
            return False

    def get_gpu_metrics(self) -> Dict:
        try:
            result = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed",
                "--format=csv,noheader,nounits"
            ], universal_newlines=True)
            gpu_util, mem_used, mem_total, temp, power, fan = map(float, result.strip().split(', '))
            
            return {
                'gpu_utilization': gpu_util,
                'gpu_memory_used': mem_used,
                'gpu_memory_total': mem_total,
                'gpu_temperature': temp,
                'gpu_power_draw': power,
                'fan_speed': fan
            }
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
            return None

    def collect_metrics(self) -> Dict:
        gpu_metrics = self.get_gpu_metrics()
        game_running = self.check_game_running()
        frame_metrics = self.read_frameview_metrics() if game_running else {}
        
        if gpu_metrics:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'game_running': game_running,
                **frame_metrics,
                **gpu_metrics,
                'total_cpu_usage': psutil.cpu_percent(),
                'total_memory_usage': psutil.virtual_memory().percent
            }
            return metrics
        return None

    def cleanup(self):
        try:
            # Use pkill for WSL environment
            subprocess.run(
                ['pkill', '-f', 'nvfsdksvc_x64'],
                capture_output=True,
                text=True,
                check=False
            )
        except Exception as e:
            print(f"Warning during cleanup: {e}")
            pass
