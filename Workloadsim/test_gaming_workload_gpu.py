import unittest
import numpy as np
import cupy as cp
import time
import psutil
import matplotlib.pyplot as plt
from pathlib import Path
import json
from gaming_workload_simulation_gpu import GamingWorkloadSimulatorGPU

class TestGamingWorkloadGPU(unittest.TestCase):
    def setUp(self):
        """Initialize simulator with test configuration"""
        self.simulator = GamingWorkloadSimulatorGPU(
            resolution=(1280, 720),
            num_threads=2,
            workload_intensity='light'
        )
    
    def test_gpu_render_frame(self):
        """Test GPU rendering simulation"""
        result = self.simulator.gpu_render_frame()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)  # Should be converted back to CPU
        self.assertEqual(len(result.shape), 2)  # Should be 2D matrix
    
    def test_gpu_physics_simulation(self):
        """Test GPU physics simulation"""
        positions, velocities = self.simulator.gpu_physics_simulation()
        self.assertIsInstance(positions, np.ndarray)  # Should be converted back to CPU
        self.assertEqual(positions.shape[1], 3)  # 3D positions
        self.assertEqual(velocities.shape[1], 3)  # 3D velocities
        self.assertTrue(np.all(positions >= 0) and np.all(positions <= 1))  # Check boundaries
    
    def test_cpu_ai_simulation(self):
        """Test AI simulation"""
        results = self.simulator.cpu_ai_simulation()
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), self.simulator.num_threads * 2)
    
    def test_run_frame(self):
        """Test complete frame execution"""
        frame_stats = self.simulator.run_frame()
        required_keys = {'frame_time', 'fps', 'frame_size', 
                        'physics_particles', 'ai_paths', 'gpu_memory'}
        self.assertTrue(all(key in frame_stats for key in required_keys))
        self.assertGreater(frame_stats['fps'], 0)

class BenchmarkGamingWorkloadGPU:
    def __init__(self):
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def get_gpu_metrics(self):
        """Get current GPU metrics using NVML"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return {
                'gpu_utilization': utilization.gpu,
                'memory_utilization': utilization.memory,
                'memory_used': memory_info.used / 1024**2,  # Convert to MB
                'power_usage': power_usage,
                'temperature': temperature
            }
        except Exception as e:
            print(f"Warning: Could not get GPU metrics: {e}")
            return None
    
    def run_benchmark_suite(self):
        """Run a comprehensive benchmark suite"""
        configurations = [
            # (resolution, threads, intensity)
            ((1280, 720), 2, 'light'),
            ((1920, 1080), 4, 'medium'),
            ((2560, 1440), 8, 'heavy')
        ]
        
        results = []
        for resolution, threads, intensity in configurations:
            print(f"\nRunning benchmark: {resolution}, {threads} threads, {intensity} intensity")
            simulator = GamingWorkloadSimulatorGPU(
                resolution=resolution,
                num_threads=threads,
                workload_intensity=intensity
            )
            
            # Run warmup frame
            simulator.run_frame()
            
            # Collect metrics for 10 frames
            frame_times = []
            cpu_usages = []
            memory_usages = []
            gpu_metrics_list = []
            
            for _ in range(10):
                start_time = time.time()
                frame_stats = simulator.run_frame()
                frame_times.append(time.time() - start_time)
                cpu_usages.append(psutil.cpu_percent())
                memory_usages.append(psutil.virtual_memory().percent)
                
                gpu_metrics = self.get_gpu_metrics()
                if gpu_metrics:
                    gpu_metrics_list.append(gpu_metrics)
            
            # Calculate GPU metrics averages if available
            gpu_averages = {}
            if gpu_metrics_list:
                for key in gpu_metrics_list[0].keys():
                    gpu_averages[key] = np.mean([m[key] for m in gpu_metrics_list])
            
            results.append({
                'configuration': {
                    'resolution': resolution,
                    'threads': threads,
                    'intensity': intensity
                },
                'metrics': {
                    'avg_frame_time': np.mean(frame_times),
                    'avg_fps': 1.0 / np.mean(frame_times),
                    'frame_time_std': np.std(frame_times),
                    'avg_cpu_usage': np.mean(cpu_usages),
                    'avg_memory_usage': np.mean(memory_usages),
                    'gpu_metrics': gpu_averages
                }
            })
        
        self.save_and_plot_results(results)
    
    def save_and_plot_results(self, results):
        """Save benchmark results and create visualization"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        with open(self.results_dir / f"benchmark_results_gpu_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        plt.figure(figsize=(15, 10))
        
        configs = [f"{r['configuration']['resolution'][0]}x{r['configuration']['resolution'][1]}\n{r['configuration']['threads']} threads\n{r['configuration']['intensity']}"
                  for r in results]
        
        # Plot FPS
        plt.subplot(2, 2, 1)
        fps_values = [r['metrics']['avg_fps'] for r in results]
        plt.bar(configs, fps_values)
        plt.title('Average FPS by Configuration')
        plt.xticks(rotation=45)
        plt.ylabel('FPS')
        
        # Plot CPU usage
        plt.subplot(2, 2, 2)
        cpu_usage = [r['metrics']['avg_cpu_usage'] for r in results]
        plt.bar(configs, cpu_usage)
        plt.title('Average CPU Usage by Configuration')
        plt.xticks(rotation=45)
        plt.ylabel('CPU Usage (%)')
        
        # Plot GPU utilization if available
        if 'gpu_metrics' in results[0]['metrics'] and results[0]['metrics']['gpu_metrics']:
            plt.subplot(2, 2, 3)
            gpu_util = [r['metrics']['gpu_metrics']['gpu_utilization'] for r in results]
            plt.bar(configs, gpu_util)
            plt.title('GPU Utilization by Configuration')
            plt.xticks(rotation=45)
            plt.ylabel('GPU Utilization (%)')
            
            plt.subplot(2, 2, 4)
            gpu_memory = [r['metrics']['gpu_metrics']['memory_used'] for r in results]
            plt.bar(configs, gpu_memory)
            plt.title('GPU Memory Usage by Configuration')
            plt.xticks(rotation=45)
            plt.ylabel('Memory Used (MB)')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f"benchmark_results_gpu_{timestamp}.png")
        plt.close()

def main():
    # Run unit tests
    print("Running GPU tests...")
    unittest.main(argv=[''], exit=False)
    
    # Run benchmarks
    print("\nRunning GPU benchmarks...")
    benchmark = BenchmarkGamingWorkloadGPU()
    benchmark.run_benchmark_suite()

if __name__ == "__main__":
    main()
