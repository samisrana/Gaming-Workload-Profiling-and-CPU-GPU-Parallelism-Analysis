import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

class ReportGenerator:
    def __init__(self, game_name: str):
        self.game_name = game_name
        self.base_dir = Path("game_performance_analysis")
        self.reports_dir = self.base_dir / "reports"
        self.graphs_dir = self.base_dir / "graphs"
        
        for directory in [self.reports_dir, self.graphs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def generate_graphs(self, df: pd.DataFrame):
        plt.style.use('seaborn')
        
        # Frame Rate Graph
        game_df = df[df['game_running']]
        if not game_df.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(game_df.index, game_df['fps'], label='FPS')
            plt.title('Frame Rate Over Time')
            plt.xlabel('Sample')
            plt.ylabel('FPS')
            plt.legend()
            plt.savefig(self.graphs_dir / 'frame_rate.png')
            plt.close()

            # Frame Time Graph
            plt.figure(figsize=(12, 6))
            plt.plot(game_df.index, game_df['frame_time_ms'], label='Frame Time')
            plt.title('Frame Time Over Time')
            plt.xlabel('Sample')
            plt.ylabel('Milliseconds')
            plt.legend()
            plt.savefig(self.graphs_dir / 'frame_time.png')
            plt.close()
        
        # GPU Performance Graph
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['gpu_utilization'], label='GPU Utilization %')
        plt.plot(df.index, df['gpu_memory_used'] / df['gpu_memory_total'] * 100, 
                label='GPU Memory %')
        plt.title('GPU Performance Over Time')
        plt.xlabel('Sample')
        plt.ylabel('Percentage')
        plt.legend()
        plt.savefig(self.graphs_dir / 'gpu_performance.png')
        plt.close()
        
        # Temperature and Power Graph
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['gpu_temperature'], label='Temperature °C')
        plt.plot(df.index, df['gpu_power_draw'], label='Power Draw W')
        plt.title('GPU Temperature and Power Draw')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(self.graphs_dir / 'temperature_power.png')
        plt.close()

    def generate_report(self, log_file: Path, session_start: datetime) -> Path:
        df = pd.read_csv(log_file)
        self.generate_graphs(df)
        
        report_file = self.reports_dir / f"report_{self.game_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"Performance Report for {self.game_name}\n")
            f.write("=" * 50 + "\n\n")
            
            game_df = df[df['game_running']]
            if not game_df.empty:
                game_time = len(game_df)
                total_time = len(df)
                game_percentage = (game_time / total_time) * 100
                f.write(f"Game Active Time: {game_percentage:.1f}% of session\n\n")
                
                f.write("Performance Metrics (During Gameplay):\n")
                f.write(f"- Average FPS: {game_df['fps'].mean():.1f}\n")
                f.write(f"- Average Frame Time: {game_df['frame_time_ms'].mean():.2f}ms\n")
                f.write(f"- 1% Low FPS: {game_df['fps'].quantile(0.01):.1f}\n")
                f.write(f"- 0.1% Low FPS: {game_df['fps'].quantile(0.001):.1f}\n\n")
            
            duration = datetime.now() - session_start
            f.write(f"Session Duration: {str(duration).split('.')[0]}\n\n")
            
            f.write("GPU Performance:\n")
            f.write(f"- Average Utilization: {df['gpu_utilization'].mean():.1f}%\n")
            f.write(f"- Peak Utilization: {df['gpu_utilization'].max():.1f}%\n")
            f.write(f"- Average Memory Used: {df['gpu_memory_used'].mean():.0f}MB\n")
            f.write(f"- Peak Memory Used: {df['gpu_memory_used'].max():.0f}MB\n")
            f.write(f"- Maximum Temperature: {df['gpu_temperature'].max():.1f}°C\n")
            f.write(f"- Average Power Draw: {df['gpu_power_draw'].mean():.1f}W\n\n")
            
            f.write("System Performance:\n")
            f.write(f"- Average CPU Usage: {df['total_cpu_usage'].mean():.1f}%\n")
            f.write(f"- Peak CPU Usage: {df['total_cpu_usage'].max():.1f}%\n")
            f.write(f"- Average Memory Usage: {df['total_memory_usage'].mean():.1f}%\n\n")

            # Added Phase 1 comparison section
            f.write("Comparison with Phase 1 Simulation:\n")
            f.write("Simulation Results:\n")
            f.write("- CPU Version:\n")
            f.write("  * Light workload: ~12 FPS\n")
            f.write("  * CPU usage: 10-14%\n")
            f.write("- GPU Version:\n")
            f.write("  * Light workload: ~7 FPS\n")
            f.write("  * CPU usage: ~6%\n")
            f.write("  * GPU utilization: 20% → 8.5% → 1.5%\n")
            f.write("  * GPU memory scaling: 1900MB → 3200MB\n")
        
        return report_file
