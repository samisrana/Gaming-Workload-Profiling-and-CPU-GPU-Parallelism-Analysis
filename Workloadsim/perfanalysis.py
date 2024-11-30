import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json

class PerformanceAnalyzer:
    def __init__(self, benchmark_dir="benchmark_results"):
        self.benchmark_dir = Path(benchmark_dir)

    def load_latest_benchmarks(self):
        """Load the most recent CPU and GPU benchmark results"""
        cpu_files = sorted(self.benchmark_dir.glob("benchmark_results_2*.json"))
        gpu_files = sorted(self.benchmark_dir.glob("benchmark_results_gpu_*.json"))
        
        if not cpu_files or not gpu_files:
            raise ValueError("No benchmark files found")
        
        with open(cpu_files[-1]) as f:
            cpu_data = json.load(f)
        with open(gpu_files[-1]) as f:
            gpu_data = json.load(f)
        
        return self._process_benchmark_data(cpu_data, gpu_data)

    def _process_benchmark_data(self, cpu_data, gpu_data):
        """Convert benchmark JSON data to DataFrames"""
        def create_df(data):
            rows = []
            for entry in data:
                config = entry['configuration']
                metrics = entry['metrics']
                
                row = {
                    'resolution': f"{config['resolution'][0]}x{config['resolution'][1]}",
                    'threads': config['threads'],
                    'intensity': config['intensity'],
                    'fps': metrics['avg_fps'],
                    'frame_time': metrics['avg_frame_time'],
                    'cpu_usage': metrics['avg_cpu_usage'],
                    'memory_usage': metrics['avg_memory_usage']
                }
                
                if 'gpu_metrics' in metrics:
                    row.update({
                        'gpu_util': metrics['gpu_metrics']['gpu_utilization'],
                        'gpu_memory': metrics['gpu_metrics']['memory_used'],
                        'gpu_temp': metrics['gpu_metrics']['temperature'],
                        'gpu_power': metrics['gpu_metrics']['power_usage']
                    })
                
                rows.append(row)
            
            return pd.DataFrame(rows)

        try:
            # Create DataFrames and swap them to match file contents
            gpu_df = create_df(gpu_data)
            cpu_df = create_df(cpu_data)
            
            logging.info("GPU DataFrame columns: %s", sorted(gpu_df.columns.tolist()))
            logging.info("CPU DataFrame columns: %s", sorted(cpu_df.columns.tolist()))
            
            return cpu_df, gpu_df
            
        except Exception as e:
            logging.error(f"Error processing benchmark data: {str(e)}")
            raise

    def analyze_bottlenecks(self, cpu_metrics, gpu_metrics):
        """Analyze performance data to identify bottlenecks"""
        bottlenecks = []
        
        for resolution in ['1280x720', '1920x1080', '2560x1440']:
            cpu_data = cpu_metrics[cpu_metrics['resolution'] == resolution]
            gpu_data = gpu_metrics[gpu_metrics['resolution'] == resolution]
            
            if not cpu_data.empty and not gpu_data.empty:
                # Memory Bandwidth Bottleneck
                if 'gpu_util' in gpu_data.columns and 'gpu_memory' in gpu_data.columns:
                    if (gpu_data['gpu_util'].mean() < 30 and 
                        gpu_data['gpu_memory'].mean() > 2000):
                        bottlenecks.append({
                            'resolution': resolution,
                            'type': 'Memory Bandwidth',
                            'indicators': {
                                'gpu_util': float(gpu_data['gpu_util'].mean()),
                                'memory_used': float(gpu_data['gpu_memory'].mean()),
                                'impact': 'Limiting GPU processing efficiency'
                            }
                        })
                
                # CPU Processing Bottleneck
                if cpu_data['cpu_usage'].mean() > 80:
                    bottlenecks.append({
                        'resolution': resolution,
                        'type': 'CPU Processing',
                        'indicators': {
                            'cpu_usage': float(cpu_data['cpu_usage'].mean()),
                            'impact': 'Limiting overall processing speed'
                        }
                    })
                
                # GPU Processing Bottleneck
                if 'gpu_util' in gpu_data.columns:
                    if gpu_data['gpu_util'].mean() > 80:
                        bottlenecks.append({
                            'resolution': resolution,
                            'type': 'GPU Processing',
                            'indicators': {
                                'gpu_util': float(gpu_data['gpu_util'].mean()),
                                'impact': 'GPU compute capacity fully utilized'
                            }
                        })
            
        return bottlenecks

    def analyze_parallelism(self, cpu_metrics, gpu_metrics):
        """Analyze CPU-GPU parallelism efficiency"""
        parallelism_analysis = []
        
        for resolution in ['1280x720', '1920x1080', '2560x1440']:
            cpu_data = cpu_metrics[cpu_metrics['resolution'] == resolution]
            gpu_data = gpu_metrics[gpu_metrics['resolution'] == resolution]
            
            if not cpu_data.empty and not gpu_data.empty:
                try:
                    cpu_gpu_ratio = cpu_data['fps'].mean() / gpu_data['fps'].mean()
                    resource_utilization = 0
                    
                    if 'gpu_util' in gpu_data.columns:
                        resource_utilization = (gpu_data['gpu_util'].mean() + 
                                             cpu_data['cpu_usage'].mean()) / 200
                    
                    parallelism_analysis.append({
                        'resolution': resolution,
                        'metrics': {
                            'cpu_gpu_ratio': float(cpu_gpu_ratio),
                            'resource_utilization': float(resource_utilization),
                            'cpu_fps': float(cpu_data['fps'].mean()),
                            'gpu_fps': float(gpu_data['fps'].mean()),
                            'speedup': float(gpu_data['fps'].mean() / cpu_data['fps'].mean())
                        }
                    })
                except ZeroDivisionError:
                    logging.warning(f"Skipping parallelism analysis for {resolution} due to zero FPS")
                    continue
        
        return parallelism_analysis

    def analyze_performance(self, cpu_df, gpu_df):
        """Comprehensive performance analysis"""
        bottlenecks = self.analyze_bottlenecks(cpu_df, gpu_df)
        parallelism = self.analyze_parallelism(cpu_df, gpu_df)
        recommendations = self.generate_optimization_recommendations(bottlenecks, parallelism)
        
        return {
            'bottlenecks': bottlenecks,
            'parallelism': parallelism,
            'recommendations': recommendations
        }

    def generate_optimization_recommendations(self, bottlenecks, parallelism_analysis):
        """Generate optimization recommendations"""
        recommendations = []
        
        for analysis in parallelism_analysis:
            resolution = analysis['resolution']
            
            if analysis['metrics']['resource_utilization'] < 0.5:
                recommendations.append({
                    'resolution': resolution,
                    'type': 'Resource Utilization',
                    'recommendation': 'Improve workload distribution between CPU and GPU',
                    'expected_impact': 'Better overall resource utilization'
                })
            
            if analysis['metrics']['speedup'] < 1:
                recommendations.append({
                    'resolution': resolution,
                    'type': 'GPU Optimization',
                    'recommendation': 'Investigate GPU memory access patterns',
                    'expected_impact': 'Reduced memory bandwidth bottleneck'
                })
        
        return recommendations

    def generate_report(self, analysis):
        """Generate a formatted analysis report"""
        report = []
        report.append("Performance Analysis Report")
        report.append("=" * 30 + "\n")
        
        # Bottleneck Summary
        report.append("Performance Bottlenecks:")
        if analysis['bottlenecks']:
            for bottleneck in analysis['bottlenecks']:
                report.append(f"\nResolution: {bottleneck['resolution']}")
                report.append(f"Type: {bottleneck['type']}")
                report.append(f"Indicators: {bottleneck['indicators']}")
        else:
            report.append("\nNo significant bottlenecks detected")
        
        # Parallelism Summary
        report.append("\nParallelism Analysis:")
        for p in analysis['parallelism']:
            report.append(f"\nResolution: {p['resolution']}")
            report.append(f"CPU/GPU Ratio: {p['metrics']['cpu_gpu_ratio']:.2f}")
            report.append(f"Resource Utilization: {p['metrics']['resource_utilization']*100:.1f}%")
            report.append(f"Speedup: {p['metrics']['speedup']:.2f}x")
        
        # Recommendations
        report.append("\nOptimization Recommendations:")
        if analysis['recommendations']:
            for rec in analysis['recommendations']:
                report.append(f"\nResolution: {rec['resolution']}")
                report.append(f"Type: {rec['type']}")
                report.append(f"Recommendation: {rec['recommendation']}")
                report.append(f"Expected Impact: {rec['expected_impact']}")
        else:
            report.append("\nNo specific optimization recommendations at this time")
        
        return "\n".join(report)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        analyzer = PerformanceAnalyzer()
        
        logging.info("Loading benchmark data...")
        cpu_df, gpu_df = analyzer.load_latest_benchmarks()
        
        logging.info("Analyzing performance...")
        analysis = analyzer.analyze_performance(cpu_df, gpu_df)
        
        logging.info("Generating report...")
        report = analyzer.generate_report(analysis)
        
        with open('performance_analysis_report.txt', 'w') as f:
            f.write(report)
        
        logging.info("Analysis complete! Report saved to 'performance_analysis_report.txt'")
        print("\n" + report)
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
