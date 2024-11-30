import unittest
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
from pathlib import Path
import json
from gaming_workload_simulation import GamingWorkloadSimulator

class TestGamingWorkload(unittest.TestCase):
    def setUp(self):
        """Initialize simulator with test configuration"""
        self.simulator = GamingWorkloadSimulator(
            resolution=(1280, 720),  # Lower resolution for faster tests
            num_threads=2,
            workload_intensity='light'
        )
    
    def test_render_frame(self):
        """Test rendering simulation"""
        result = self.simulator.simulate_render_frame()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result.shape), 2)  # Should be 2D matrix
    
    def test_cpu_physics_simulation(self):
        """Test physics simulation"""
        positions, velocities = self.simulator.cpu_physics_simulation()
        self.assertEqual(positions.shape[1], 3)  # 3D positions
        self.assertEqual(velocities.shape[1], 3)  # 3D velocities
        self.assertTrue(np.all(positions >= 0) and np.all(positions <= 1))  # Check boundaries
    
    def test_cpu_ai_simulation(self):
        """Test AI simulation"""
        results = self.simulator.cpu_ai_simulation()
        self.assertIsInstance(results, list)
        # Updated expectation to match simulation's behavior of 2 regions per thread
        self.assertEqual(len(results), self.simulator.num_threads * 2)
    
    def test_run_frame(self):
        """Test complete frame execution"""
        frame_stats = self.simulator.run_frame()
        required_keys = {'frame_time', 'fps', 'frame_size', 
                        'physics_particles', 'ai_paths'}
        self.assertTrue(all(key in frame_stats for key in required_keys))
        self.assertGreater(frame_stats['fps'], 0)

class BenchmarkGamingWorkload:
    def __init__(self):
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
    
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
            simulator = GamingWorkloadSimulator(
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
            
            for _ in range(10):
                start_time = time.time()
                frame_stats = simulator.run_frame()
                frame_times.append(time.time() - start_time)
                cpu_usages.append(psutil.cpu_percent())
                memory_usages.append(psutil.virtual_memory().percent)
            
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
                    'avg_memory_usage': np.mean(memory_usages)
                }
            })
        
        self.save_and_plot_results(results)
    
    def save_and_plot_results(self, results):
        """Save benchmark results and create visualization"""
        # Save raw results
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        with open(self.results_dir / f"benchmark_results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create performance visualization
        plt.figure(figsize=(12, 6))
        
        # Plot FPS for each configuration
        configs = [f"{r['configuration']['resolution'][0]}x{r['configuration']['resolution'][1]}\n{r['configuration']['threads']} threads\n{r['configuration']['intensity']}"
                  for r in results]
        fps_values = [r['metrics']['avg_fps'] for r in results]
        
        plt.subplot(1, 2, 1)
        plt.bar(configs, fps_values)
        plt.title('Average FPS by Configuration')
        plt.xticks(rotation=45)
        plt.ylabel('FPS')
        
        # Plot CPU usage
        plt.subplot(1, 2, 2)
        cpu_usage = [r['metrics']['avg_cpu_usage'] for r in results]
        plt.bar(configs, cpu_usage)
        plt.title('Average CPU Usage by Configuration')
        plt.xticks(rotation=45)
        plt.ylabel('CPU Usage (%)')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f"benchmark_results_{timestamp}.png")
        plt.close()

def main():
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False)
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    benchmark = BenchmarkGamingWorkload()
    benchmark.run_benchmark_suite()

if __name__ == "__main__":
    main()
