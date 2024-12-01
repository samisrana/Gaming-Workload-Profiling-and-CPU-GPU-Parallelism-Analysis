import signal
import time
import subprocess
from datetime import datetime
from metrics_collector import MetricsCollector
from data_logger import DataLogger
from display_manager import DisplayManager
from report_generator import ReportGenerator

class GameMonitor:
    def __init__(self, game_name: str):
        self.game_name = game_name
        self.metrics_collector = MetricsCollector(game_name)
        self.data_logger = DataLogger(game_name)
        self.display_manager = DisplayManager(game_name)
        self.report_generator = ReportGenerator(game_name)
        
        self.is_monitoring = False
        self.session_start = None
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        print("\nReceived signal to stop monitoring...")
        self.stop()
    
    def start(self):
        try:
            subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Cannot access nvidia-smi.")
            return

        print(f"Starting monitoring for {self.game_name}")
        print("Press Ctrl+C to stop monitoring")
        
        self.is_monitoring = True
        self.session_start = datetime.now()
        
        try:
            while self.is_monitoring:
                metrics = self.metrics_collector.collect_metrics()
                if metrics:
                    self.data_logger.log_metrics(metrics)
                    self.display_manager.display_status(
                        metrics,
                        self.session_start,
                        self.data_logger.get_output_file()
                    )
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        self.is_monitoring = False
        print("\nGenerating performance report...")
        self.metrics_collector.cleanup()
        report_file = self.report_generator.generate_report(
            self.data_logger.get_output_file(),
            self.session_start
        )
        print(f"\nMonitoring stopped. Files saved to:")
        print(f"- Log file: {self.data_logger.get_output_file()}")
        print(f"- Report: {report_file}")
        print(f"- Graphs: {self.report_generator.graphs_dir}")

if __name__ == "__main__":
    import sys
    
    print("\nIntegrated Game Performance Monitor")
    print("This tool will monitor your game and compare it with Phase 1 simulation results")
    print("\nEnter the name of the game to monitor.")
    print("Examples:")
    print("- For Terraria.exe, enter: terraria")
    print("- For LeagueOfLegends.exe, enter: league")
    print("Note: The search is case-insensitive\n")
    
    game_name = sys.argv[1] if len(sys.argv) > 1 else input("Enter game name: ").strip()
    if not game_name:
        game_name = "unknown_game"
    
    monitor = GameMonitor(game_name)
    monitor.start()
