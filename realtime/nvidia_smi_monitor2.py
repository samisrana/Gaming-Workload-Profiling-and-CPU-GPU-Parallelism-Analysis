import subprocess
import csv
import time
from datetime import datetime
import psutil
from pathlib import Path
import signal
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, Dict, List  # Correctly import Tuple here

class IntegratedGameMonitor:
    def __init__(self, game_name: str = "unknown_game"):
        self.game_name = game_name.lower()
        self.is_monitoring = False
        self.session_start = None
        self.metrics_buffer = []
        self.last_frame_time = None  # Track the timestamp of the last frame
        
        # Create directory structure
        self.base_dir = Path("game_performance_analysis")
        self.logs_dir = self.base_dir / "logs"
        self.reports_dir = self.base_dir / "reports"
        self.graphs_dir = self.base_dir / "graphs"
        
        for directory in [self.logs_dir, self.reports_dir, self.graphs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.logs_dir / f"{game_name}_{timestamp}.csv"
        
        self.headers = [
            'timestamp',
            'game_running',
            'frame_time_ms',
            'fps',
            'gpu_utilization',
            'gpu_memory_used',
            'gpu_memory_total',
            'gpu_temperature',
            'gpu_power_draw',
            'fan_speed',
            'total_cpu_usage',
            'total_memory_usage'
        ]
        
        # Initialize CSV file
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        
        # Add signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle SIGINT (Ctrl+C) signal to stop monitoring gracefully."""
        print("\nReceived interrupt signal. Stopping monitoring...")
        self.stop()

    def check_game_running(self) -> bool:
        """Check if the game process is running using psutil."""
        try:
            for proc in psutil.process_iter(['name']):
                if self.game_name in proc.info['name'].lower():
                    return True
            return False
        except Exception as e:
            print(f"Error checking game process: {e}")
            return False

    def get_gpu_metrics(self) -> Dict:
        """Get GPU metrics using nvidia-smi."""
        try:
            query = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed",
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.check_output(query, universal_newlines=True)
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
            return {
                'gpu_utilization': 0,
                'gpu_memory_used': 0,
                'gpu_memory_total': 0,
                'gpu_temperature': 0,
                'gpu_power_draw': 0,
                'fan_speed': 0
            }

    def calculate_fps(self) -> Tuple[float, float]:
        """Calculate frame time and FPS based on elapsed time."""
        current_time = time.time()
        if self.last_frame_time is None:
            self.last_frame_time = current_time
            return 0, 0  # Default to 0 until we have at least two timestamps
        
        frame_time_ms = (current_time - self.last_frame_time) * 1000
        self.last_frame_time = current_time
        fps = 1000 / frame_time_ms if frame_time_ms > 0 else 0
        return frame_time_ms, fps

    def collect_metrics(self) -> Dict:
        """Collect all metrics including frame timing."""
        gpu_metrics = self.get_gpu_metrics()
        game_running = self.check_game_running()
        frame_time_ms, fps = self.calculate_fps() if game_running else (0, 0)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'game_running': game_running,
            'frame_time_ms': frame_time_ms,
            'fps': fps,
            **gpu_metrics,
            'total_cpu_usage': psutil.cpu_percent(),
            'total_memory_usage': psutil.virtual_memory().percent
        }
        return metrics

    def log_metrics(self, metrics: Dict):
        """Log metrics to CSV file."""
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(metrics)

    def live_graph(self, interval: int = 1000):
        """Display real-time FPS graph."""
        fps_data = []

        def update(frame):
            if self.metrics_buffer:
                metrics = self.metrics_buffer[-1]
                fps_data.append(metrics['fps'])
                if len(fps_data) > 60:  # Keep only the last 60 seconds of data
                    fps_data.pop(0)
                plt.clf()
                plt.plot(fps_data, label="FPS")
                plt.ylim(0, 120)  # Adjust the Y-axis limit to match expected FPS range
                plt.title(f"Real-Time FPS for {self.game_name}")
                plt.xlabel("Time (seconds)")
                plt.ylabel("FPS")
                plt.legend(loc="upper left")

        ani = FuncAnimation(plt.gcf(), update, interval=interval)
        plt.show()

    def start(self):
        """Start monitoring."""
        print(f"Starting monitoring for {self.game_name}. Press Ctrl+C to stop.")
        self.is_monitoring = True
        self.session_start = datetime.now()

        # Start live graph in a separate thread
        import threading
        graph_thread = threading.Thread(target=self.live_graph)
        graph_thread.start()

        try:
            while self.is_monitoring:
                metrics = self.collect_metrics()
                if metrics:
                    self.metrics_buffer.append(metrics)
                    self.log_metrics(metrics)
                    time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop monitoring and save the session."""
        self.is_monitoring = False
        print("\nMonitoring stopped. Log file saved at:")
        print(self.output_file)
        self.generate_report()

    def generate_report(self):
        """Generate a report based on the collected metrics."""
        print("Generating performance report...")
        df = pd.read_csv(self.output_file)

        # Simple summary stats
        print("\nPerformance Summary:")
        print(f"Average FPS: {df['fps'].mean():.2f}")
        print(f"Average GPU Utilization: {df['gpu_utilization'].mean():.2f}%")
        print(f"Average CPU Usage: {df['total_cpu_usage'].mean():.2f}%")
        print(f"Maximum GPU Temperature: {df['gpu_temperature'].max():.2f}°C")
        print(f"Maximum GPU Memory Used: {df['gpu_memory_used'].max():.2f}MB")

        # Save performance summary to a text file
        report_file = self.reports_dir / f"report_{self.game_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(f"Performance Report for {self.game_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Average FPS: {df['fps'].mean():.2f}\n")
            f.write(f"Average GPU Utilization: {df['gpu_utilization'].mean():.2f}%\n")
            f.write(f"Average CPU Usage: {df['total_cpu_usage'].mean():.2f}%\n")
            f.write(f"Maximum GPU Temperature: {df['gpu_temperature'].max():.2f}°C\n")
            f.write(f"Maximum GPU Memory Used: {df['gpu_memory_used'].max():.2f}MB\n")
        print(f"Report saved to: {report_file}")

if __name__ == "__main__":
    import sys
    game_name = sys.argv[1] if len(sys.argv) > 1 else input("Enter game name: ").strip()
    monitor = IntegratedGameMonitor(game_name)
    monitor.start()
